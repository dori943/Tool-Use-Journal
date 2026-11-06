# -*- coding: utf-8 -*-
"""Table II 재계산 — 8개 태스크 × 잡아야 할 물체 × 3 EE: "feasible 추정" vs "실제 파지 성공".

셀 기호 (물체 × EE)
  ✓  추정 feasible 이고 시뮬 PASS        (Table II 분자·분모 모두 포함)
  ✗  추정 feasible 인데 시뮬 FAIL        (분모만 포함)  ← 추정이 틀린 곳
  ?  추정 feasible 인데 아직 실행 안 됨   (분모 포함, 분자 미정 — 표에 (?) 로 표시)
  +  추정은 불가인데 시뮬 PASS           (분모 제외; 보수적 추정 → 각주)
  ·  둘 다 아님
  ∅  실행했지만 레시피 없음(NOT_REGISTERED) 등 미실행
  ⊗  (vac 전용) 시뮬에서는 붙었지만 GT vac 규칙(표면 평면성·접촉폭·질량) 불통과 → 성공으로 세지 않음

vac 열은 시뮬 부착만으로 성공을 세지 않는다. --vac-gt (scripts/vac_gt_admissibility.py 출력) 를
주면  vac 성공 = 시뮬 PASS ∧ GT admissible  로 판정한다. 없으면 vac 셀에 "(gate 없음)" 경고를 단다.

Table II 값 = 태스크·EE 별  ✓ / (✓+✗+?) . 역사적 논문 값과 나란히 비교한다.

입력
  추정 : --m1-dir 아래 <task>/m1.json (Full M1, 노드 ee 판정). 없으면 후보 경로를 차례로 찾는다.
  실제 : --results 로 준 JSON 들. 두 스키마 모두 읽는다.
          (a) docs/ee_gear_coverage_results_20261102.json  → tasks[].cases[] {object, ee, status, reviewed_status, attachment}
          (b) scripts/verify_table2_coverage.py 출력 (docs/table2_verification.json) → [ {task, trials[] {object, ee, status}} ]
        같은 (task, object, ee) 가 여러 번이면 뒤에 준 파일이 우선 (최신을 마지막에).

  python scripts/table2_recompute.py \
      --results docs/ee_gear_coverage_results_20261102.json docs/table2_verification.json \
      --vac-gt docs/vac_gt_admissibility.json --out docs/table2_recomputed.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audit_ee_coverage import TABLE2, TASK_GRASP_OBJECTS  # noqa: E402

EES = ("2F", "3F", "vac")
TASK_ORDER = ["c1_1", "c1_2", "c2_1", "c2_2", "c3_1", "c3_2", "c4_1", "c4_2"]
# 타입 레시피는 _a/_b 인스턴스를 공유한다: 결과가 인스턴스 id 로 기록돼도 TASK_GRASP_OBJECTS 와 맞춘다.
_PASS = {"PASS"}
_FAIL = {"FAIL"}
_ERROR = {"ERROR", "TIMEOUT"}   # 구현/실행 오류: 물리 실패로 세지 않음 → "!" (재실행 필요)
_UNRUN = {"NOT_REGISTERED", "EE_MISMATCH", "NOT_RUN", "SKIPPED", "BODY_BINDING_ERROR"}


def _read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ inference
def _m1_candidates(task: str, m1_dir: Path) -> list[Path]:
    return [
        m1_dir / task / "m1.json",
        ROOT / "output" / "ablation" / "full" / task / "seed_0" / "m1.json",
        ROOT / "output" / task / "m1.json",
    ]


def inferred_feasible(task: str, m1_dir: Path) -> tuple[dict[str, list[str]], str | None]:
    """{object: [EE...]} from the Full M1 (Reasoner) verdicts. Objects without a verdict are omitted."""
    for path in _m1_candidates(task, m1_dir):
        if not path.is_file():
            continue
        try:
            m1 = _read(path)
        except json.JSONDecodeError:
            continue
        table = {}
        for n in m1.get("nodes", []):
            nid, cls = n.get("id", ""), n.get("class", "")
            short = nid[len(f"obj_{cls}_"):] if nid.startswith(f"obj_{cls}_") else nid
            ee = n.get("ee") or {}
            if ee:
                table[short] = [e for e in EES if (ee.get(e) or {}).get("feasible")]
        hits = {o: table[o] for o in TASK_GRASP_OBJECTS[task] if o in table}

        def _base(name):
            head, _, tail = name.rpartition("_")
            return head if head and tail.isdigit() else name
        for o in TASK_GRASP_OBJECTS[task]:
            if o in hits:
                continue
            same = [k for k in table if _base(k) == _base(o) and _base(k) != k]
            if len(same) == 1:                       # 인스턴스 번호만 다른 단일 물체 (turkey_1 ↔ turkey_3)
                hits[o] = table[same[0]]
        if hits:
            return hits, str(path)
    return {}, None


# -------------------------------------------------------------------- results
def load_results(paths: list[Path]) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    for path in paths:
        data = _read(path)
        records = []
        if isinstance(data, dict) and "tasks" in data:                      # (a) 팀원 체크포인트 JSON
            for t in data["tasks"]:
                for c in t.get("cases", []):
                    records.append((t["task"], c["object"], c["ee"],
                                    c.get("reviewed_status") or c.get("status"),
                                    (c.get("attachment") or {}), str(path), c.get("integration_status")))
        elif isinstance(data, list):                                        # (b) verify_table2_coverage 출력
            for t in data:
                for c in t.get("trials", []):
                    att = c.get("attachment") or ((c.get("result") or {}).get("attachment")) or {}
                    records.append((t["task"], c["object"], c["ee"], c.get("status"),
                                    att, str(path), c.get("integration_status")))
        else:
            raise ValueError(f"unknown results schema: {path}")
        for task, obj, ee, status, attachment, src, integ in records:
            out[(task, obj, ee)] = {"status": status, "attachment": attachment, "source": src,
                                    "integration": integ}
    return out


def load_vac_gt(path: Path | None) -> dict[tuple[str, str], dict]:
    """{(task, object): {admissible, reason}} from vac_gt_admissibility.json."""
    if path is None or not Path(path).is_file():
        return {}
    out = {}
    for res in _read(path):
        for name, r in (res.get("objects") or {}).items():
            if "error" in r:
                continue
            out[(res["task"], name)] = {"admissible": bool(r.get("admissible")),
                                        "reason": (r.get("vac_rule") or {}).get("reason", "")}
    return out


def _attachment_label(att: dict) -> str:
    if not att:
        return ""
    if att.get("acquisition_used"):
        return "acq-assist"
    if att.get("phase") == "POST_VALIDATED_GRASP" or att.get("used") is False:
        return "free-hold"
    return att.get("phase") or ""


# ---------------------------------------------------------------------- build
def build(task: str, m1_dir: Path, results: dict, vac_gt: dict | None = None) -> dict:
    objects = TASK_GRASP_OBJECTS[task]
    inferred, src = inferred_feasible(task, m1_dir)
    vac_gt = vac_gt or {}
    rows = []
    for obj in objects:
        cells = {}
        for ee in EES:
            inf = (obj in inferred) and (ee in inferred[obj])
            r = results.get((task, obj, ee))
            st = (r or {}).get("status")
            gate = vac_gt.get((task, obj)) if ee == "vac" else None
            if st in _PASS and ee == "vac" and gate is not None and not gate["admissible"]:
                sym = "⊗"      # 붙긴 했지만 GT vac 규칙 불통과: 성공 아님 (추정 feasible 이면 분모에는 남음)
            elif st in _PASS:
                sym = "✓" if inf else "+"
            elif st in _FAIL:
                sym = "✗" if inf else "·"
            elif st in _ERROR:
                sym = "!"
            elif st in _UNRUN or r is None:
                sym = "?" if inf else ("∅" if r is not None else "·")
            else:
                sym = "?" if inf else "·"
            cells[ee] = {"symbol": sym, "inferred": inf, "status": st,
                         "attachment": _attachment_label((r or {}).get("attachment") or {}),
                         "inference_known": obj in inferred,
                         "vac_gt": (None if gate is None else gate["admissible"]),
                         "vac_gt_reason": (None if gate is None else gate["reason"]),
                         "vac_gate_missing": (ee == "vac" and st in _PASS and gate is None),
                         "production_failed": ((r or {}).get("integration") == "VALIDATED"
                                               and st in _FAIL | _ERROR)}
        rows.append({"object": obj, "inferred": inferred.get(obj), "cells": cells})

    per_ee = {}
    for ee in EES:
        grasped = sum(1 for r in rows if r["cells"][ee]["symbol"] == "✓")
        failed = sum(1 for r in rows if r["cells"][ee]["symbol"] == "✗"
                     or (r["cells"][ee]["symbol"] == "⊗" and r["cells"][ee]["inferred"]))
        gated = [r["object"] for r in rows if r["cells"][ee]["symbol"] == "⊗"]
        gate_missing = [r["object"] for r in rows if r["cells"][ee]["vac_gate_missing"]]
        pending = sum(1 for r in rows if r["cells"][ee]["symbol"] in "?"
                      or (r["cells"][ee]["symbol"] == "!" and r["cells"][ee]["inferred"]))
        errors = [r["object"] for r in rows if r["cells"][ee]["symbol"] == "!"]
        harness = [r["object"] for r in rows if r["cells"][ee].get("production_failed")]
        untested = [r["object"] for r in rows if r["cells"][ee]["status"] is None]
        extra = [r["object"] for r in rows if r["cells"][ee]["symbol"] == "+"]
        denom = grasped + failed + pending
        hist = TABLE2.get(task, {}).get(ee)
        per_ee[ee] = {
            "grasped": grasped, "inferred": denom, "pending": pending,
            "value": f"{grasped}/{denom}" + ("(?)" if pending else ""),
            "coverage_all_objects": f"{grasped + len(extra)}/{len(objects)}",
            "extra_pass_not_inferred": extra,
            "sim_attach_only_rejected": gated,
            "errors_rerun_needed": errors,
            "production_route_failed": harness,
            "untested": untested,
            "coverage_complete": not errors and not untested,
            "vac_gate_missing": gate_missing,
            "historical": f"{hist[0]}/{hist[1]}" if hist else "-",
            "match_historical": bool(hist and not pending and grasped == hist[0] and denom == hist[1]),
        }
    unknown_inf = [o for o in objects if o not in inferred]
    return {"task": task, "inference_source": src, "objects_without_inference": unknown_inf,
            "rows": rows, "per_ee": per_ee}


def render(results_all: list[dict]) -> str:
    L = ["# Table II 재계산 — 추정 feasible vs 실제 파지 (seed 0, 파지·상승·유지)", "",
         "기호: ! 구현/실행 오류(ERROR·TIMEOUT, 물리 실패 아님 → 재실행) · ✓ 추정 feasible & PASS · ✗ 추정 feasible & FAIL · ? 추정 feasible 미실행 · + 추정 불가인데 PASS · ⊗ vac: 시뮬 부착은 됐으나 GT vac 규칙 불통과(성공 아님) · ∅ 미실행(레시피 없음) · · 해당 없음", "",
         "## 1. 논문 Table II 와 같은 정의 — 커버리지 = 잡힌 물체 / 태스크 전체 물체 (괄호 = 논문 값)", "",
         "논문: \"An object is counted as covered by an EE if it was grasped successfully with that EE\". 분모는 태스크 물체 전체.",
         "≠ = 재계산 값이 논문 값과 다름 · * = 오류·미실행 조합이 남아 미확정 · [n≠m] = 현재 장면 물체 수가 논문 분모와 다름", "",
         "| task | 2F | 3F | vac | 미완료 / 주의 |", "|---|---|---|---|---|"]
    for res in results_all:
        pe = res["per_ee"]
        cells, warn = [], []
        for e in EES:
            hist = TABLE2.get(res["task"], {}).get(e)
            cov = pe[e]["coverage_all_objects"]
            k, n = (int(x) for x in cov.split("/"))
            mark = ""
            if hist:
                if n != hist[1]:
                    mark = f" [{n}≠{hist[1]}]"
                elif k != hist[0]:
                    mark = " ≠"
            cells.append(f"{cov}{'' if pe[e]['coverage_complete'] else '*'} ({hist[0]}/{hist[1]}){mark}" if hist else cov)
            if pe[e]["errors_rerun_needed"]:
                warn.append(f"{e} 오류:{','.join(pe[e]['errors_rerun_needed'])}")
            if pe[e]["untested"]:
                warn.append(f"{e} 미실행:{','.join(pe[e]['untested'])}")
        L.append(f"| {res['task']} | " + " | ".join(cells) + f" | {'; '.join(warn) or '-'} |")
    L += ["", "## 2. Reasoner 추정 정밀도 — ✓ / 추정 feasible (논문 Table II 와 다른 지표, 비교하지 않음)", "",
          "| task | 2F | 3F | vac | 추정 없음 물체 | 추정 소스 |", "|---|---|---|---|---|---|"]
    for res in results_all:
        pe = res["per_ee"]
        L.append(f"| {res['task']} | " + " | ".join(pe[e]["value"] for e in EES)
                 + f" | {','.join(res['objects_without_inference']) or '-'} | "
                 + f"{Path(res['inference_source']).parent.name + '/m1.json' if res['inference_source'] else '-'} |")
    L += ["", "## vac: 시뮬에서는 붙었지만 GT vac 규칙 불통과 (성공으로 세지 않음)", ""]
    for res in results_all:
        g = res["per_ee"]["vac"]
        L.append(f"- {res['task']}: " + (", ".join(g["sim_attach_only_rejected"]) or "-")
                 + (f"  (GT 게이트 없음: {', '.join(g['vac_gate_missing'])})" if g["vac_gate_missing"] else ""))
    L += ["", "## 추정 밖 PASS (Reasoner 가 불가로 봤지만 실제로 잡힘)", ""]
    for res in results_all:
        pe = res["per_ee"]
        extra = "; ".join(f"{e}:{','.join(pe[e]['extra_pass_not_inferred'])}" for e in EES if pe[e]["extra_pass_not_inferred"])
        L.append(f"- {res['task']}: {extra or '-'}")
    for res in results_all:
        L += ["", f"## {res['task']}", "", "| object | 추정 feasible | 2F | 3F | vac |", "|---|---|---|---|---|"]
        for r in res["rows"]:
            inf = ",".join(r["inferred"]) if r["inferred"] is not None else "(추정 없음)"
            cells = []
            for e in EES:
                c = r["cells"][e]
                tag = c["symbol"]
                if c["status"] and c["symbol"] in "✓+✗⊗!":
                    tag += f" {c['status']}" + (f"/{c['attachment']}" if c["attachment"] else "")
                if c["symbol"] == "⊗" and c.get("vac_gt_reason"):
                    tag += f" [{c['vac_gt_reason']}]"
                cells.append(tag)
            L.append(f"| {r['object']} | {inf} | " + " | ".join(cells) + " |")
    L += ["", "* = 해당 EE 열에 오류·미실행 조합이 있어 커버리지가 확정되지 않음.", "",
          "주의: 레시피별 힘·허용값·부착 정책이 달라 PASS 건수를 균일 프로토콜 성공률로 읽지 않는다. "
          "'free-hold' 는 부착 보조 없이 유지, 'acq-assist' 는 획득 단계 부착 보조 사용.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default="all")
    p.add_argument("--m1-dir", type=Path, default=ROOT / "output")
    p.add_argument("--results", nargs="+", type=Path, default=None,
                   help="기본: docs/ee_gear_coverage_results_*.json + docs/*coverage_validation*.json (수정시각 순, 최신 우선)")
    p.add_argument("--vac-gt", type=Path, default=ROOT / "docs" / "vac_gt_admissibility.json",
                   help="scripts/vac_gt_admissibility.py 출력. vac 성공 = 시뮬 PASS ∧ GT 규칙 통과")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "table2_recomputed.md")
    args = p.parse_args(argv)
    tasks = TASK_ORDER if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    if args.results is None:
        docs = ROOT / "docs"
        found = list(docs.glob("ee_gear_coverage_results_*.json")) + list(docs.glob("*coverage_validation*.json"))
        args.results = sorted(set(found), key=lambda q: q.stat().st_mtime)
        print("[table2] 결과 입력:", ", ".join(q.name for q in args.results))
    results = load_results([r for r in args.results if r.is_file()])
    missing = [str(r) for r in args.results if not r.is_file()]
    if missing:
        print(f"[table2] 결과 파일 없음(건너뜀): {missing}")
    vac_gt = load_vac_gt(args.vac_gt)
    if not vac_gt:
        print(f"[table2] 경고: vac GT 게이트 파일 없음({args.vac_gt}) — vac 셀은 시뮬 부착만으로 표시됨")
    built = [build(t, args.m1_dir, results, vac_gt) for t in tasks]
    md = render(built)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(built, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)
    print(f"[table2] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
