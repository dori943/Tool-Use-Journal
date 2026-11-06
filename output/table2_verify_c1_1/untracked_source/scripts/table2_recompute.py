# -*- coding: utf-8 -*-
"""Table II 재계산 — 8개 태스크 × 잡아야 할 물체 × 3 EE: "feasible 추정" vs "실제 파지 성공".

셀 기호 (물체 × EE)
  ✓  추정 feasible 이고 시뮬 PASS        (Table II 분자·분모 모두 포함)
  ✗  추정 feasible 인데 시뮬 FAIL        (분모만 포함)  ← 추정이 틀린 곳
  ?  추정 feasible 인데 아직 실행 안 됨   (분모 포함, 분자 미정 — 표에 (?) 로 표시)
  +  추정은 불가인데 시뮬 PASS           (분모 제외; 보수적 추정 → 각주)
  ·  둘 다 아님
  ∅  실행했지만 레시피 없음(NOT_REGISTERED) 등 미실행

Table II 값 = 태스크·EE 별  ✓ / (✓+✗+?) . 역사적 논문 값과 나란히 비교한다.

입력
  추정 : --m1-dir 아래 <task>/m1.json (Full M1, 노드 ee 판정). 없으면 후보 경로를 차례로 찾는다.
  실제 : --results 로 준 JSON 들. 두 스키마 모두 읽는다.
          (a) docs/ee_gear_coverage_results_20261102.json  → tasks[].cases[] {object, ee, status, reviewed_status, attachment}
          (b) scripts/verify_table2_coverage.py 출력 (docs/table2_verification.json) → [ {task, trials[] {object, ee, status}} ]
        같은 (task, object, ee) 가 여러 번이면 뒤에 준 파일이 우선 (최신을 마지막에).

  python scripts/table2_recompute.py \
      --results docs/ee_gear_coverage_results_20261102.json docs/table2_verification.json \
      --out docs/table2_recomputed.md
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
_FAIL = {"FAIL", "ERROR", "TIMEOUT"}
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
                                    (c.get("attachment") or {}), str(path)))
        elif isinstance(data, list):                                        # (b) verify_table2_coverage 출력
            for t in data:
                for c in t.get("trials", []):
                    records.append((t["task"], c["object"], c["ee"], c.get("status"),
                                    c.get("attachment") or {}, str(path)))
        else:
            raise ValueError(f"unknown results schema: {path}")
        for task, obj, ee, status, attachment, src in records:
            out[(task, obj, ee)] = {"status": status, "attachment": attachment, "source": src}
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
def build(task: str, m1_dir: Path, results: dict) -> dict:
    objects = TASK_GRASP_OBJECTS[task]
    inferred, src = inferred_feasible(task, m1_dir)
    rows = []
    for obj in objects:
        cells = {}
        for ee in EES:
            inf = (obj in inferred) and (ee in inferred[obj])
            r = results.get((task, obj, ee))
            st = (r or {}).get("status")
            if st in _PASS:
                sym = "✓" if inf else "+"
            elif st in _FAIL:
                sym = "✗" if inf else "·"
            elif st in _UNRUN or r is None:
                sym = "?" if inf else ("∅" if r is not None else "·")
            else:
                sym = "?" if inf else "·"
            cells[ee] = {"symbol": sym, "inferred": inf, "status": st,
                         "attachment": _attachment_label((r or {}).get("attachment") or {}),
                         "inference_known": obj in inferred}
        rows.append({"object": obj, "inferred": inferred.get(obj), "cells": cells})

    per_ee = {}
    for ee in EES:
        grasped = sum(1 for r in rows if r["cells"][ee]["symbol"] == "✓")
        failed = sum(1 for r in rows if r["cells"][ee]["symbol"] == "✗")
        pending = sum(1 for r in rows if r["cells"][ee]["symbol"] == "?")
        extra = [r["object"] for r in rows if r["cells"][ee]["symbol"] == "+"]
        denom = grasped + failed + pending
        hist = TABLE2.get(task, {}).get(ee)
        per_ee[ee] = {
            "grasped": grasped, "inferred": denom, "pending": pending,
            "value": f"{grasped}/{denom}" + ("(?)" if pending else ""),
            "coverage_all_objects": f"{grasped + len(extra)}/{len(objects)}",
            "extra_pass_not_inferred": extra,
            "historical": f"{hist[0]}/{hist[1]}" if hist else "-",
            "match_historical": bool(hist and not pending and grasped == hist[0] and denom == hist[1]),
        }
    unknown_inf = [o for o in objects if o not in inferred]
    return {"task": task, "inference_source": src, "objects_without_inference": unknown_inf,
            "rows": rows, "per_ee": per_ee}


def render(results_all: list[dict]) -> str:
    L = ["# Table II 재계산 — 추정 feasible vs 실제 파지 (seed 0, 파지·상승·유지)", "",
         "기호: ✓ 추정 feasible & PASS · ✗ 추정 feasible & FAIL · ? 추정 feasible 미실행 · + 추정 불가인데 PASS · ∅ 미실행(레시피 없음) · · 해당 없음", "",
         "## 요약 (값 = ✓ / 추정 feasible 수, 괄호는 역사적 Table II)", "",
         "| task | 2F | 3F | vac | 추정 없음 물체 | 추정 소스 |", "|---|---|---|---|---|---|"]
    for res in results_all:
        pe = res["per_ee"]
        L.append(f"| {res['task']} | " + " | ".join(
            f"{pe[e]['value']} ({pe[e]['historical']})" + ("" if pe[e]["match_historical"] else " ≠") for e in EES)
            + f" | {','.join(res['objects_without_inference']) or '-'} | {Path(res['inference_source']).parent.name + '/m1.json' if res['inference_source'] else '-'} |")
    L += ["", "## 전체 물체 기준 커버리지 (PASS / 전체 물체, 추정과 무관)", "",
          "| task | 2F | 3F | vac | 추정 밖 PASS |", "|---|---|---|---|---|"]
    for res in results_all:
        pe = res["per_ee"]
        extra = "; ".join(f"{e}:{','.join(pe[e]['extra_pass_not_inferred'])}" for e in EES if pe[e]["extra_pass_not_inferred"])
        L.append(f"| {res['task']} | " + " | ".join(pe[e]["coverage_all_objects"] for e in EES) + f" | {extra or '-'} |")
    for res in results_all:
        L += ["", f"## {res['task']}", "", "| object | 추정 feasible | 2F | 3F | vac |", "|---|---|---|---|---|"]
        for r in res["rows"]:
            inf = ",".join(r["inferred"]) if r["inferred"] is not None else "(추정 없음)"
            cells = []
            for e in EES:
                c = r["cells"][e]
                tag = c["symbol"]
                if c["status"] and c["symbol"] in "✓+✗":
                    tag += f" {c['status']}" + (f"/{c['attachment']}" if c["attachment"] else "")
                cells.append(tag)
            L.append(f"| {r['object']} | {inf} | " + " | ".join(cells) + " |")
    L += ["", "주의: 레시피별 힘·허용값·부착 정책이 달라 PASS 건수를 균일 프로토콜 성공률로 읽지 않는다. "
          "'free-hold' 는 부착 보조 없이 유지, 'acq-assist' 는 획득 단계 부착 보조 사용.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default="all")
    p.add_argument("--m1-dir", type=Path, default=ROOT / "output")
    p.add_argument("--results", nargs="+", type=Path,
                   default=[ROOT / "docs" / "ee_gear_coverage_results_20261102.json"])
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "table2_recomputed.md")
    args = p.parse_args(argv)
    tasks = TASK_ORDER if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    results = load_results([r for r in args.results if r.is_file()])
    missing = [str(r) for r in args.results if not r.is_file()]
    if missing:
        print(f"[table2] 결과 파일 없음(건너뜀): {missing}")
    built = [build(t, args.m1_dir, results) for t in tasks]
    md = render(built)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(built, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)
    print(f"[table2] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
