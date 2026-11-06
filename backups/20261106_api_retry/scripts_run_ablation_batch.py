# -*- coding: utf-8 -*-
"""Ablation 배치 실행 + 에러 메시지 자동 수집.

scripts/run.py 를 task × seed 로 반복 호출하고, 각 실행의 로그/JSON 에서
실패 단계·원인 코드·에러 메시지를 뽑아 한곳에 모은다. run.py 와 M5 코드는
전혀 건드리지 않는다(순수 드라이버).

예)
  # w/o Reasoner (논문 Table III; = --grounding-mode without_reasoner), 8개 태스크, seed 0 고정, 5회 반복
  python scripts\\run_ablation_batch.py --condition wo-reasoner --tasks all --seeds 0 --repeat 5

  # 이미 끝난 seed 는 건너뛰고 이어서
  python scripts\\run_ablation_batch.py --condition wo-reasoner --tasks all --seeds 0 --repeat 5 --skip-done

  # 실행 없이 지금까지 모인 결과만 다시 집계
  python scripts\\run_ablation_batch.py --condition wo-reasoner --summarize-only

  # run.py 에 추가 인자를 그대로 넘기려면 `--` 뒤에
  python scripts\\run_ablation_batch.py --tasks c2_1 --seeds 0 -- --provider openai --model gpt-5.4-mini

출력 구조:
  output/ablation/<method>/<task>/seed_<n>/        run.py 의 --output-dir
      run.log                                       stdout+stderr 전체
      batch_record.json                             이 실행의 수집 결과(아래 필드)
  output/ablation/<method>/errors.jsonl             실행 1건 = 1줄 (append)
  output/ablation/<method>/errors_summary.md        task × seed 표 + 원인 코드 집계
  output/ablation/<method>/errors_summary.csv       같은 내용 CSV

batch_record.json 주요 필드:
  task, seed, method, exit_code, duration_s, started_at
  failure_stage   : m1 | vlm_prior | m2 | gk | independent_assignment | m4 | m5 | m5_planning | m5_execution | goal | none | unknown
  plan_status     : without_planner_result.plan_status (M4 대체 할당 결과)
  m5_status       : m5_summary.status
  success         : without_planner_result.success (현재 run.py 기준 = M5 status == SUCCESS)
  reason_code     : 로그/JSON 에서 처음 발견한 UPPER_SNAKE 원인 코드
  reason_codes    : 발견한 모든 원인 코드와 빈도
  involved_objects: 'grasp of 'spoon'' 같은 메시지에서 뽑은 물체 id
  messages        : 사람이 읽을 에러 메시지(중복 제거, 최대 40줄)
  traceback_tail  : run.log 의 마지막 Traceback
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from task_registry import TASKS  # noqa: E402

ALL_TASKS = list(TASKS)  # c1_1, c2_1, c3_1, c1_2, c2_2, c3_2, c4_1, c4_2

METHOD_DIR = {
    "full": "full",
    "without-planner": "without-planner",
    "greedy-ee-order": "greedy-ee-order",
    "without_grounding": "without-grounding",
    "without_reasoner": "without-reasoner",
}

# 실패 원인 코드: 대문자 스네이크 (예: GRASP_RETENTION_FAILED, COLLISION_MARGIN_VIOLATION)
CODE_RE = re.compile(r"\b([A-Z][A-Z0-9]+(?:_[A-Z0-9]+){1,6})\b")
# 코드처럼 생겼지만 원인이 아닌 토큰
CODE_IGNORE = {
    "SCRIPTED_GRASP_LIVE", "M5_MOTION_PLAN", "SCRIPTED_GRASP", "NOT_EVALUATED",
    "IN_PROGRESS", "UTF_8", "OPENAI_API_KEY", "TUJ_LLM_PROVIDER", "NUMBA_CACHE_DIR",
    "LIVE_EXECUTION_RUNTIME", "PYTHONIOENCODING", "PYTHONUTF8", "SG1_S1", "ID_S",
}
OBJECT_RE = re.compile(r"(?:grasp of|object|held|lost|dropped|slip(?:ped)?)[^'\"]{0,40}['\"]([A-Za-z0-9_]+)['\"]")
LOG_LINE_RE = re.compile(
    r"(\[err\]|\[중단\]|Error\b|Exception\b|FAILED|_FAILED|REJECTED|EXHAUSTED|SLIP|VIOLATION|INFEASIBLE|invalid input|실패)"
)


# ----------------------------------------------------------------- helpers
def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def _parse_tasks(spec: str) -> list[str]:
    if spec == "all":
        return list(ALL_TASKS)
    tasks = [t.strip() for t in spec.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        sys.exit(f"[batch] 등록되지 않은 task: {unknown} (등록됨: {ALL_TASKS})")
    return tasks


def _codes_in(text: str) -> list[str]:
    return [c for c in CODE_RE.findall(text or "") if c not in CODE_IGNORE]


# ------------------------------------------------------------ collection
def collect_record(out: Path, *, task: str, seed: int, method: str,
                   exit_code, duration_s, started_at, log: Path | None = None) -> dict:
    """실행 폴더 하나에서 실패 정보를 모은다. 실행 없이도 호출 가능."""
    rec: dict = {
        "task": task, "seed": seed, "method": method,
        "exit_code": exit_code, "duration_s": duration_s, "started_at": started_at,
        "output_dir": str(out),
        "failure_stage": "unknown", "plan_status": None, "m5_status": None,
        "success": None, "reason_code": None, "reason_codes": {},
        "involved_objects": [], "messages": [], "traceback_tail": None,
        "sources": [],
    }
    messages: list[str] = []
    codes: Counter = Counter()
    objects: Counter = Counter()

    def note(src: str, text) -> None:
        if not text:
            return
        text = str(text).strip()
        if not text:
            return
        rec["sources"].append(src)
        messages.append(f"[{src}] {text}")
        codes.update(_codes_in(text))
        objects.update(OBJECT_RE.findall(text))

    # 1) ablation_manifest.json — 어느 단계에서 죽었는지
    manifest = _read_json(out / "ablation_manifest.json") or {}
    stages = manifest.get("stages") or {}
    failure = manifest.get("failure") or {}
    if failure:
        rec["failure_stage"] = failure.get("stage") or "unknown"
        note("manifest.failure", f"{failure.get('type')} at stage={failure.get('stage')}")

    # 2) without_planner_result.json / without_grounding_result.json
    for name in ("without_planner_result.json", "without_grounding_result.json",
                 "greedy_ee_order_result.json"):
        result = _read_json(out / name)
        if result:
            rec["plan_status"] = result.get("plan_status")
            rec["m5_status"] = result.get("m5_status")
            rec["success"] = result.get("success")
            note("result.m5_failure_detail", result.get("m5_failure_detail"))
            break

    # 3) m4.json — 대체 할당이 실패한 경우 거절 사유
    m4 = _read_json(out / "m4.json") or {}
    if m4 and not m4.get("selected_plan"):
        rec["failure_stage"] = rec["failure_stage"] if failure else "independent_assignment"
        diag = m4.get("diagnostics") or {}
        rejections = diag.get("rejections") or m4.get("rejections") or []
        for rej in rejections[:10]:
            if isinstance(rej, dict):
                note("m4.rejection", f"{rej.get('reason_code') or rej.get('code')}: "
                                     f"{rej.get('detail') or rej.get('message') or rej}")
            else:
                note("m4.rejection", rej)
        if not rejections:
            note("m4.status", m4.get("status"))

    # 3b) run_result.json — run.py 공통 스키마 (있으면 이것이 기준)
    rr = _read_json(out / "run_result.json") or {}
    if rr:
        rec["plan_status"] = rec["plan_status"] or rr.get("plan_status")
        rec["m5_status"] = rr.get("m5_status")
        rec["success"] = rr.get("success")
        rec["exec_success"] = rr.get("exec_success")
        rec["goal_success"] = rr.get("goal_success")
        rec["task_goal_status"] = rr.get("task_goal_status")
        rec["task_goal_evaluation"] = rr.get("task_goal_evaluation")
        rec["task_goal_policy"] = rr.get("task_goal_policy")
        rec["physics_related"] = rr.get("physics_related")
        rec["m5_attempt_count"] = rr.get("m5_attempt_count")
        rec["exec_success_first_try"] = (
            (rr.get("m5_attempts") or [{}])[0].get("m5_status") == "SUCCESS"
            if rr.get("m5_attempts") else None
        )
        stage_map = {"planning": "m5_planning", "execution": "m5_execution",
                     "goal": "goal", "evaluation": "evaluation", "none": "none"}
        if rr.get("failure_stage") in stage_map and rec["failure_stage"] in ("unknown", "m5"):
            rec["failure_stage"] = stage_map[rr["failure_stage"]]
        if rr.get("failure_code"):
            codes[rr["failure_code"]] += 3  # 공통 규칙이 고른 코드를 우선
        note("run_result.detail", rr.get("detail"))

    # 3c) vlm_prior.json — w/o reasoner 의 feasible_ee / fallback
    vp = _read_json(out / "vlm_prior.json") or {}
    if vp:
        rec["feasible_ee"] = vp.get("feasible_ee")
        rec["ee_fallback"] = vp.get("ee_fallback")

    # 4) m5/m5_summary.json  (without_grounding 은 motion_attempt_NNN/m5/)
    m5_root = out
    attempts = sorted(out.glob("motion_attempt_*"))
    if attempts and not (out / "m5").exists():
        m5_root = attempts[-1]
    summary = _read_json(m5_root / "m5" / "m5_summary.json") or {}
    if summary:
        rec["m5_status"] = rec["m5_status"] or summary.get("status")
        rec["m5_mode"] = summary.get("mode")
        if not rr or not rr.get("task_goal_status"):
            rec["task_goal_status"] = summary.get("task_goal_status")
            rec["task_goal_evaluation"] = summary.get("task_goal_evaluation")
            rec["task_goal_policy"] = summary.get("task_goal_policy")
            rec["goal_success"] = {"SATISFIED": True, "FAILED": False}.get(rec["task_goal_status"])
        if not rr:
            rec["exec_success"] = summary.get("status") == "SUCCESS"
            rec["success"] = rec.get("goal_success") if rec["exec_success"] else False
            if rec["exec_success"] and rec.get("goal_success") is False:
                rec["failure_stage"] = "goal"
                codes["GOAL_NOT_ACHIEVED"] += 3
            elif rec["exec_success"] and rec.get("goal_success") is None:
                rec["failure_stage"] = "evaluation"
                codes["GOAL_EVIDENCE_MISSING"] += 3
        rec["simulation_status"] = summary.get("simulation_status")
        note("m5_summary.detail", summary.get("detail"))
        if summary.get("failure_type"):
            note("m5_summary.failure_type", summary.get("failure_type"))
        if summary.get("failure_stage") and rec["failure_stage"] in ("unknown", "m5"):
            rec["failure_stage"] = {"planning": "m5_planning", "execution": "m5_execution"}.get(
                summary["failure_stage"], rec["failure_stage"])
            if summary.get("failure_code"):
                codes[summary["failure_code"]] += 3
        if summary.get("status") not in (None, "SUCCESS") and not (rr or summary.get("failure_stage")):
            # (구버전 m5_summary) planning 단계 실패 vs 실행(시뮬) 중 실패 구분
            st = str(summary.get("status"))
            detail = str(summary.get("detail") or "")
            if st == "SIMULATION_FAILED" or "LivePlanExecutionError" in detail \
                    or "GRASP_RETENTION" in detail or "SLIP" in detail or "LOST" in detail:
                rec["failure_stage"] = "m5_execution"
            elif rec["failure_stage"] in ("unknown", "m5"):
                rec["failure_stage"] = "m5_planning"
        elif summary.get("status") == "SUCCESS" and summary.get("simulation_successful") is False:
            rec["failure_stage"] = "m5_execution"
            note("m5_summary.simulation", "planning SUCCESS but simulation_successful=False")

    # 5) m5/m5_failure.json — 전략별 거절 코드
    failures = _read_json(m5_root / "m5" / "m5_failure.json")
    if isinstance(failures, list):
        per_code: Counter = Counter()
        for item in failures:
            if not isinstance(item, dict):
                continue
            per_code[item.get("failure_code") or "?"] += 1
            for code, n in (item.get("rejected_edge_counts") or {}).items():
                per_code[code] += int(n)
        for code, n in per_code.most_common():
            codes[code] += n
        if failures:
            first = failures[0] if isinstance(failures[0], dict) else {}
            note("m5_failure[0]", f"{first.get('strategy_id')}: {first.get('failure_code')}: "
                                  f"{str(first.get('detail'))[:400]}")
        rec["m5_failure_strategy_count"] = len(failures)
        rec["m5_failure_codes"] = dict(per_code)

    # 6) live-execution-manifest.json (scripted live 와 generic simulation 둘 다)
    for mpath in (m5_root / "m5" / "live" / "live-execution-manifest.json",
                  m5_root / "m5" / "simulation" / "live-execution-manifest.json"):
        live = _read_json(mpath)
        if not live:
            continue
        rec["live_status"] = live.get("status")
        note("live.failure", live.get("failure"))
        records = live.get("records") or live.get("steps") or []
        failed = [r for r in records if isinstance(r, dict) and str(r.get("status")).upper() == "FAILED"]
        if failed:
            last = failed[-1]
            rec["live_failed_step"] = last.get("subgoal_id") or last.get("step_id") or last.get("id")
            note("live.failed_step", f"{rec['live_failed_step']}: {last.get('detail') or last.get('failure') or ''}")
            # Reclassify archived post-execution checks without changing their
            # saved verdicts or rewriting any source run files.
            from tuj.m5_motion.failure_taxonomy import classify_failure
            classification = classify_failure(live.get("failure"), steps=records)
            if (exit_code != -9 and live.get("status") == "FAILED"
                    and rec["failure_stage"] in ("unknown", "m5", "m5_planning", "m5_execution", "goal")
                    and classification["failure_stage"] == "goal"):
                rec["failure_stage"] = "goal"
                rec["failure_basis"] = classification["failure_basis"]
                codes[classification["failure_code"]] += 3

        rec["live_record_count"] = len(records)
        break

    # 7) run.log — traceback 과 에러성 줄
    # Archived runs may contain Windows paths; prefer the run-local copy.
    log_name = str(log).replace("\\", "/").rsplit("/", 1)[-1] if log else "run.log"
    candidates = [out / log_name]
    if log:
        candidates.append(Path(log))
    candidates.extend([out / "run.log", out / f"{out.name}.log"])
    log = next((p for p in candidates if p.is_file()), out / "run.log")
    if log.is_file():
        try:
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        tb_idx = [i for i, ln in enumerate(lines) if ln.startswith("Traceback (most recent call last)")]
        if tb_idx:
            tail = lines[tb_idx[-1]:]
            rec["traceback_tail"] = "\n".join(tail[-30:])
            # 마지막 예외 한 줄
            last_exc = next((ln for ln in reversed(tail) if ln and not ln.startswith(" ")), None)
            note("traceback", last_exc)
        hits = [ln.strip() for ln in lines if LOG_LINE_RE.search(ln)]
        seen = set()
        for ln in hits:
            key = ln[:160]
            if key in seen:
                continue
            seen.add(key)
            note("run.log", ln[:400])
            if len(messages) >= 60:
                break

    # A timed-out M5-only retry may leave run_result.json from its previous
    # attempt. It must not inherit that attempt's completion or goal verdict.
    if exit_code == -9:
        rec.update(success=False, exec_success=False, goal_success=None,
                   exec_success_first_try=None, task_goal_status="NOT_EVALUATED")
        if not failure:
            rec["failure_stage"] = "unknown"

    # 실패 단계 보정
    m5_ok = (rec["m5_status"] == "SUCCESS" and not failure
             and rec.get("goal_success") is True
             and rec["success"] is not False and exit_code in (0, None))
    if (rec["success"] is True or m5_ok) and exit_code in (0, None):
        # 성공한 run 에 남아 있는 m5_failure.json 은 중간에 거절된 전략일 뿐이다
        rec["failure_stage"] = "none"
        rec["planning_rejections_seen"] = dict(codes)
        codes = Counter()
        messages = [m for m in messages if not m.startswith("[m5_failure")]
    if rec["failure_stage"] == "unknown":
        if (rec["success"] is True or m5_ok) and exit_code in (0, None):
            rec["failure_stage"] = "none"
        elif failure.get("stage") == "m5":
            rec["failure_stage"] = "m5_planning"
        elif stages:
            # prepared/resumed are completed prerequisites, not failures.
            # Prefer an explicit failure, then the last running stage. M5
            # without a detailed report stays "m5"; do not guess its substage.
            for state in ("failed", "running"):
                active = [st for st, val in stages.items() if str(val).lower() == state]
                if active:
                    rec["failure_stage"] = active[-1]
                    break
    if exit_code not in (0, None) and rec["failure_stage"] == "none":
        rec["failure_stage"] = "unknown"

    if exit_code == -9:
        note("batch", "BATCH_TIMEOUT: run interrupted by the batch time limit")

    # 중복 제거 + 상한
    dedup, seen = [], set()
    for m in messages:
        if m in seen:
            continue
        seen.add(m)
        dedup.append(m)
    rec["messages"] = dedup[:40]
    rec["reason_codes"] = dict(codes.most_common())
    rec["reason_code"] = "BATCH_TIMEOUT" if exit_code == -9 else (codes.most_common(1)[0][0] if codes else None)
    rec["involved_objects"] = [o for o, _ in objects.most_common(5)]
    rec["sources"] = sorted(set(rec["sources"]))
    return rec


# ------------------------------------------------------------- summaries
def write_summaries(method_dir: Path, records: list[dict]) -> None:
    records = sorted(records, key=lambda r: (r["task"], r["seed"], r.get("rep", 0)))
    csv_path = method_dir / "errors_summary.csv"
    md_path = method_dir / "errors_summary.md"
    fields = ["task", "seed", "rep", "success", "exec_success", "goal_success", "failure_stage",
              "plan_status", "m5_status", "task_goal_status", "physics_related",
              "m5_attempt_count", "reason_code", "involved_objects", "exit_code",
              "duration_s", "first_message", "output_dir"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({
                **{k: r.get(k) for k in fields},
                "involved_objects": ",".join(r.get("involved_objects") or []),
                "first_message": (r.get("messages") or [""])[0][:300],
            })

    tasks = sorted({r["task"] for r in records}, key=lambda t: ALL_TASKS.index(t) if t in ALL_TASKS else 99)
    by_task_stage: dict[str, Counter] = {t: Counter() for t in tasks}
    by_task_code: dict[str, Counter] = {t: Counter() for t in tasks}
    all_codes: Counter = Counter()
    all_objs: Counter = Counter()
    for r in records:
        by_task_stage[r["task"]][r["failure_stage"]] += 1
        if r.get("reason_code"):
            by_task_code[r["task"]][r["reason_code"]] += 1
        all_codes.update(r.get("reason_codes") or {})
        all_objs.update(r.get("involved_objects") or [])

    report_stages = ["m1", "vlm_prior", "m2", "gk", "independent_assignment", "m4",
                     "m5", "m5_planning", "m5_execution", "goal", "evaluation", "unknown"]
    columns = ["task", "runs", "전체 성공(실행+목표)", *report_stages,
               "1차시도 exec 성공", "목표 달성/평가", "목표 미평가", "최빈 원인 코드"]
    lines = [f"# {method_dir.name} 에러 요약", "",
             f"갱신: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  총 {len(records)}회 실행", "",
             "전체 성공은 실행 완료와 목표 달성이 모두 확인된 경우만 셉니다. 기존 파일의 success=True라도 목표 미평가면 포함하지 않습니다.",
             "m5는 세부 단계가 확인되지 않은 M5 중단입니다.", "",
             "## 태스크별 성공 / 실패 단계", "",
             "| " + " | ".join(columns) + " |",
             "|" + "---|" * len(columns)]
    for t in tasks:
        rs = [r for r in records if r["task"] == t]
        succ = sum(1 for r in rs if r.get("exec_success") is True and r.get("goal_success") is True)
        evaluated = sum(r.get("goal_success") is not None for r in rs)
        goals = sum(r.get("goal_success") is True for r in rs)
        goal_result = f"{goals}/{evaluated}" if evaluated else "미평가"
        st = by_task_stage[t]
        top = by_task_code[t].most_common(1)
        first = sum(1 for r in rs if r.get("exec_success_first_try") is True)
        lines.append(
            f"| {t} | {len(rs)} | {succ}/{len(rs)} | "
            + " | ".join(str(st.get(k, 0)) for k in report_stages)
            + f" | {first}/{len(rs)} | {goal_result} | {len(rs) - evaluated} | "
            + f"{top[0][0] + ' (' + str(top[0][1]) + ')' if top else '-'} |")

    lines += ["", "## 원인 코드 빈도 (전체)", "", "| code | count |", "|---|---|"]
    for code, n in all_codes.most_common(25):
        lines.append(f"| {code} | {n} |")
    if all_objs:
        lines += ["", "## 실패 메시지에 등장한 물체 (반복 drop/slip 확인용)", "", "| object | count |", "|---|---|"]
        for obj, n in all_objs.most_common(15):
            lines.append(f"| {obj} | {n} |")

    lines += ["", "## 실행별 상세", ""]
    for r in records:
        mark = ("OK " if r.get("exec_success") is True and r.get("goal_success") is True
                else "UNKNOWN" if r.get("exec_success") is True and r.get("goal_success") is None else "FAIL")
        lines.append(f"### {mark} {r['task']} seed={r['seed']} rep={r.get('rep', 0)}  stage={r['failure_stage']}  "
                     f"code={r.get('reason_code')}  exit={r.get('exit_code')}  "
                     f"{(r.get('duration_s') or 0) / 60:.1f}min")
        lines.append("")
        for m in (r.get("messages") or [])[:8]:
            lines.append(f"- {m[:300]}")
        if r.get("traceback_tail"):
            tail = r["traceback_tail"].splitlines()
            lines.append(f"- traceback: `{tail[-1][:200]}`")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[batch] summary -> {md_path}")
    print(f"[batch] csv     -> {csv_path}")


def load_all_records(method_dir: Path, method: str) -> list[dict]:
    """errors.jsonl 대신 폴더를 다시 훑어 최신 상태로 집계한다."""
    def refresh(seed_dir, previous):
        return {**previous, **collect_record(
            seed_dir, task=previous["task"], seed=previous["seed"], method=method,
            exit_code=previous.get("exit_code"), duration_s=previous.get("duration_s"),
            started_at=previous.get("started_at"), log=previous.get("log"))}

    records = []
    for seed_dir in sorted(method_dir.glob("*/seed_*")):
        if not seed_dir.is_dir() or "_prev_" in seed_dir.name:
            continue
        jl = seed_dir / "batch_records.jsonl"
        if jl.is_file():
            history = []
            for line in jl.read_text(encoding="utf-8").splitlines():
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            if history:
                # M5-only repeats share a directory. Current artifacts belong
                # only to the last record; preserve earlier measurements.
                records.extend(history[:-1])
                records.append(refresh(seed_dir, history[-1]))
                continue
        rec = _read_json(seed_dir / "batch_record.json")
        if rec:
            records.append(refresh(seed_dir, rec))
    # batch_record 가 없는(수동 실행) 폴더도 수집
    for seed_dir in sorted(method_dir.glob("*/seed_*")):
        if (seed_dir / "batch_record.json").exists() or (seed_dir / "batch_records.jsonl").exists() \
                or not seed_dir.is_dir() or "_prev_" in seed_dir.name:
            continue
        task = seed_dir.parent.name
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        records.append(collect_record(seed_dir, task=task, seed=seed, method=method,
                                      exit_code=None, duration_s=None, started_at=None))
    return records


# --------------------------------------------------------------- running
def run_one(args, task: str, seed: int, out: Path, extra: list[str], *, rep: int = 0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "run.py"), task,
           "--seed", str(seed), "--output-dir", str(out)]
    if args.m5_only:
        # 같은 m1/m2/m4 결과로 M5(실행)만 다시: 이전 m5/ 는 m5_rep<k>/ 로 보존
        if not (out / "m4.json").is_file() and not list(out.glob("motion_attempt_*/m4.json")):
            print(f"[batch] skip {task} seed={seed}: m4.json 없음 (먼저 전체 파이프라인을 돌리십시오)")
            return {"task": task, "seed": seed, "skipped": True}
        prev = out / "m5"
        if prev.exists():
            n = len(list(out.glob("m5_rep*")))
            shutil.move(str(prev), str(out / f"m5_rep{n}"))
        for name in ("run_result.json",):
            if (out / name).exists():
                n = len(list(out.glob("m5_rep*")))
                shutil.move(str(out / name), str(out / f"run_result_rep{max(n - 1, 0)}.json"))
        cmd += ["--start-from", "m5"]
    if args.exec_retries:
        cmd += ["--exec-retries", str(args.exec_retries)]
    if args.planner_mode != "full":
        cmd += ["--planner-mode", args.planner_mode]
    if args.grounding_mode != "full":
        cmd += ["--grounding-mode", args.grounding_mode]
    if args.m5_simulate:
        cmd += ["--m5-simulate", args.m5_simulate]
    if args.provider:
        cmd += ["--provider", args.provider]
    if args.model:
        cmd += ["--model", args.model]
    cmd += extra
    if args.m5_args:
        cmd += ["--m5-args", *args.m5_args]

    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
    if sys.platform != "win32":
        env.setdefault("MUJOCO_GL", "egl")

    started = datetime.now().isoformat(timespec="seconds")
    print(f"\n[batch] ===== {task} seed={seed} -> {out}")
    print("[batch] $ " + " ".join(cmd))
    if args.dry_run:
        return {"task": task, "seed": seed, "dry_run": True}

    # run.py(ablation 모드)는 새 실행에 빈 출력 폴더를 요구하므로 로그는 폴더 밖에 쓴다.
    log_dir = out.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if not args.m5_only else f"_m5only_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log = log_dir / f"{out.name}{suffix}.log"
    t0 = time.monotonic()
    exit_code: int | None
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"# {started}\n# {' '.join(cmd)}\n\n")
        fh.flush()
        try:
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fh,
                                    stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace")
            try:
                exit_code = proc.wait(timeout=args.timeout_min * 60 if args.timeout_min else None)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                exit_code = -9
                fh.write(f"\n[batch] TIMEOUT after {args.timeout_min} min — killed\n")
        except KeyboardInterrupt:
            proc.kill()
            raise
    duration = round(time.monotonic() - t0, 1)
    try:
        shutil.copyfile(log, out / log.name)  # 실행이 끝난 뒤에는 폴더 안에도 사본을 둔다
    except OSError:
        pass
    rec = collect_record(out, task=task, seed=seed, method=args.method_name,
                         exit_code=exit_code, duration_s=duration, started_at=started, log=log)
    rec["rep"] = rep
    rec["m5_only"] = bool(args.m5_only)
    rec["log"] = str(log)
    _write_json(out / "batch_record.json", rec)
    with (out / "batch_records.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (args.method_dir / "errors.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    status = "OK" if rec.get("success") is True else "FAIL"
    print(f"[batch] {status} {task} seed={seed} exit={exit_code} {duration / 60:.1f}min "
          f"stage={rec['failure_stage']} code={rec.get('reason_code')}")
    for m in rec["messages"][:3]:
        print(f"[batch]   {m[:200]}")
    return rec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", choices=("full", "wo-reasoner", "wo-planner", "wo-grounding-legacy"),
                   default=None,
                   help="논문 조건 이름 단축키: full=Ours; wo-reasoner=--grounding-mode without_reasoner "
                        "(Reasoner 측정값만 VLM 추정으로 대체, 나머지 Full 동일); "
                        "wo-planner=--planner-mode without-planner (EE Swap-Aware Planner 제거); "
                        "wo-grounding-legacy=구 without_grounding (VLM 이 EE 를 직접 선택)")
    p.add_argument("--planner-mode", choices=("full", "without-planner", "greedy-ee-order"), default="full")
    p.add_argument("--grounding-mode", choices=("full", "without_grounding", "without_reasoner"), default="full")
    p.add_argument("--tasks", default="all", help="all 또는 c1_1,c2_1,... (기본 all)")
    p.add_argument("--seeds", default="0", help="예: 0  |  0-4  |  0,3,7-9")
    p.add_argument("--repeat", type=int, default=1,
                   help="같은 seed 를 N회 반복. 기본은 seed_<s>_r<k> 폴더로 분리(전체 파이프라인 반복), "
                        "--m5-only 와 함께면 같은 폴더에서 M5 만 N회 반복")
    p.add_argument("--m5-only", action="store_true",
                   help="기존 폴더의 m1/m2/m4 결과를 그대로 두고 M5(실행)만 다시 돌림 "
                        "(run.py --start-from m5). LLM 호출 없이 실행 노이즈만 반복 측정할 때")
    p.add_argument("--exec-retries", type=int, default=0,
                   help="run.py --exec-retries 로 전달 (실행 실패만 N회 재시도, 기본 0)")
    p.add_argument("--m5-simulate", choices=("kinematic", "controller"), default="controller",
                   help="run.py --m5-simulate (기본 controller = scripted-grasp live 실행)")
    p.add_argument("--provider", choices=("openai", "gemini"), default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--timeout-min", type=float, default=240, help="실행 1회 제한 시간(분), 기본 240 (c3_2 영상 포함 실행이 90분을 넘김). 0 = 무제한")
    p.add_argument("--skip-done", action="store_true",
                   help="batch_record.json 이 이미 있는 (task, seed) 는 건너뜀")
    p.add_argument("--retry-failed", action="store_true",
                   help="--skip-done 과 함께: 실패한 run 만 다시 돌림 (이전 폴더는 _prev_<시각> 으로 이름 변경)")
    p.add_argument("--continue-on-error", action="store_true", default=True)
    p.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    p.add_argument("--dry-run", action="store_true", help="명령만 출력")
    p.add_argument("--summarize-only", action="store_true", help="실행 없이 집계만 갱신")
    p.add_argument("--output-root", type=Path, default=None,
                   help="기본 output/ablation")
    p.add_argument("--m5-args", nargs=argparse.REMAINDER, default=[],
                   help="이 뒤 인자는 run.py --m5-args 로 M5 러너에 전달")
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    args = build_parser().parse_args(argv)
    if args.condition == "wo-reasoner":
        args.grounding_mode = "without_reasoner"
    elif args.condition == "wo-grounding-legacy":
        args.grounding_mode = "without_grounding"
    elif args.condition == "wo-planner":
        args.planner_mode = "without-planner"
    elif args.condition == "full":
        args.planner_mode = args.grounding_mode = "full"

    if args.planner_mode != "full" and args.grounding_mode != "full":
        sys.exit("[batch] planner-mode 와 grounding-mode ablation 은 따로 돌리십시오 (run.py 제약)")
    args.method_name = (METHOD_DIR[args.grounding_mode] if args.grounding_mode != "full"
                        else METHOD_DIR[args.planner_mode])
    root = (args.output_root or ROOT / "output" / "ablation").resolve()
    args.method_dir = root / args.method_name
    args.method_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        write_summaries(args.method_dir, load_all_records(args.method_dir, args.method_name))
        return 0

    tasks = _parse_tasks(args.tasks)
    seeds = _parse_seeds(args.seeds)
    plan = [(t, s, k) for s in seeds for k in range(args.repeat) for t in tasks]
    # seed 바깥 루프: seed 0 을 8개 태스크 전부 먼저 → 중간에 멈춰도 균형 잡힌 표가 남는다
    print(f"[batch] method={args.method_name} tasks={tasks} seeds={seeds} repeat={args.repeat} "
          f"total={len(plan)} runs -> {args.method_dir}")

    done = failed = skipped = 0
    try:
        for task, seed, k in plan:
            name = f"seed_{seed}" if (args.repeat == 1 or args.m5_only) else f"seed_{seed}_r{k}"
            out = args.method_dir / task / name
            prev = _read_json(out / "batch_record.json")
            if args.m5_only:
                rec = run_one(args, task, seed, out, extra, rep=k)
                if rec.get("dry_run") or rec.get("skipped"):
                    continue
                done += 1 if rec.get("success") is True else 0
                failed += 0 if rec.get("success") is True else 1
                write_summaries(args.method_dir, load_all_records(args.method_dir, args.method_name))
                continue
            if prev and args.skip_done:
                if args.retry_failed and prev.get("success") is not True:
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    shutil.move(str(out), str(out.parent / f"{name}_prev_{stamp}"))
                else:
                    skipped += 1
                    print(f"[batch] skip {task} {name} (success={prev.get('success')})")
                    continue
            elif out.exists() and any(out.iterdir()) and not args.dry_run:
                # run.py(without-planner) 는 빈 폴더를 요구한다
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                shutil.move(str(out), str(out.parent / f"{name}_prev_{stamp}"))
            rec = run_one(args, task, seed, out, extra, rep=k)
            if rec.get("dry_run"):
                continue
            if rec.get("success") is True:
                done += 1
            else:
                failed += 1
                if not args.continue_on_error:
                    break
            write_summaries(args.method_dir, load_all_records(args.method_dir, args.method_name))
    except KeyboardInterrupt:
        print("\n[batch] interrupted — 지금까지 결과로 집계합니다")
    if not args.dry_run:
        write_summaries(args.method_dir, load_all_records(args.method_dir, args.method_name))
    print(f"\n[batch] done: success={done} fail={failed} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
