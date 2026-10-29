# -*- coding: utf-8 -*-
"""Table II 검증 — "EE 로 가능하다고 추론한 물체" 를 실제 시뮬에서 잡아 본다.

Table II 정의: 각 태스크·EE 에 대해  (실제 시뮬에서 잡힌 물체 수) / (그 EE 로 가능하다고 추론한 물체 수)

  분모: Reasoner(Full M1) 가 feasible 로 판정한 (물체, EE) — output/<task>/m1.json 의 ee 판정.
        m1.json 이 없거나 접지값이 없으면(c1_1) --denominator gt-rule 로 YCB 질량+AABB 규칙을 쓴다.
  분자: scripted grasp 레시피로 PRE→GRASP→attach→LIFT→retention 을 실제 MuJoCo 에서 실행해
        RETENTION 까지 성공한 것 (scripted_grasps.validate.controller_validate_case 재사용).
        레시피가 없는 (물체, EE) 는 NOT_REGISTERED 로 따로 세며, 이 조합은 M5 의 범용 LLM 경로로만
        실행 가능하므로 여기서는 시도하지 않는다 (--llm-fallback 은 후속 작업).

실행 (robosuite/MuJoCo 필요, API 호출 없음; 한 조합당 수십 초~수 분):
  python scripts/verify_table2_coverage.py --tasks c1_1
  python scripts/verify_table2_coverage.py --tasks c1_1,c1_2,c2_1 --all-ees   # 추론과 무관하게 3 EE 전부
  python scripts/verify_table2_coverage.py --tasks c2_1 --video                # 조합별 mp4 저장

출력: output/table2_verify/<task>/<object>_<ee>/validation_report.json (+ mp4)
      docs/table2_verification.md / .json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from task_registry import TASKS  # noqa: E402
from audit_ee_coverage import TABLE2, TASK_GRASP_OBJECTS, load_ycb_masses, make_env, gt_scene_from_env  # noqa: E402


def reasoner_feasible(task: str, m1_path: Path) -> dict[str, list[str]] | None:
    if not m1_path.is_file():
        return None
    m1 = json.loads(m1_path.read_text(encoding="utf-8"))
    out = {}
    for n in m1.get("nodes", []):
        nid, cls = n.get("id", ""), n.get("class", "")
        short = nid[len(f"obj_{cls}_"):] if nid.startswith(f"obj_{cls}_") else nid
        ee = n.get("ee") or {}
        if ee:
            out[short] = sorted(k for k, v in ee.items() if v.get("feasible"))
    names = TASK_GRASP_OBJECTS[task]
    if not any(o in out for o in names):
        return None
    return {o: out.get(o, []) for o in names}


def gt_rule_feasible(task: str, seed: int) -> dict[str, list[str]]:
    from tuj.gt.labels import intrinsic_from_gt, normalize_ee_pool
    from tuj.m1_scene.ee_rules import evaluate_ee

    robot = json.loads((ROOT / "configs" / "robot_spec.json").read_text(encoding="utf-8"))
    pool = normalize_ee_pool(robot["ee_pool"])
    env = make_env(task, seed)
    try:
        gt = gt_scene_from_env(env, TASK_GRASP_OBJECTS[task], load_ycb_masses(ROOT / "configs" / "ycb_mass_gt.json"))
    finally:
        env.close()
    out = {}
    for name, rec in gt.items():
        if "error" in rec:
            out[name] = []
            continue
        intr = intrinsic_from_gt(rec)
        out[name] = [e["ee_id"] for e in pool if evaluate_ee(e, intr)["feasible"]]
    return out


def run_task(task: str, *, seed: int, denominator: str, all_ees: bool, video: bool,
             out_root: Path, ee_ids: list[str]) -> dict:
    from tuj.m5_motion.scripted_grasps.validate import controller_validate_case

    environment = TASKS[task].cls
    names = TASK_GRASP_OBJECTS[task]
    feasible = None
    source = denominator
    if denominator == "reasoner":
        feasible = reasoner_feasible(task, ROOT / "output" / task / "m1.json")
        if feasible is None:
            print(f"[table2] {task}: output/{task}/m1.json 에 Reasoner EE 판정이 없음 → gt-rule 분모로 대체")
            source = "gt-rule(fallback)"
    if feasible is None:
        feasible = gt_rule_feasible(task, seed)

    trials = []
    for name in names:
        ees = ee_ids if all_ees else feasible.get(name, [])
        for ee in ees:
            out = out_root / task / f"{name}_{ee}"
            print(f"[table2] {task} {name} × {ee} ...", flush=True)
            t0 = time.monotonic()
            report = controller_validate_case(
                name, ee, out, environment=environment, seed=seed,
                video=(out / f"{name}_{ee}.mp4") if video else None, repository=ROOT,
            )
            status = report.get("controller_status")
            trials.append({
                "object": name, "ee": ee, "inferred_feasible": ee in feasible.get(name, []),
                "status": status,  # PASS | FAIL | NOT_REGISTERED | EE_MISMATCH | ...
                "integration_status": report.get("integration_status"),
                "failures": report.get("failures"), "seconds": round(time.monotonic() - t0, 1),
                "report": str(out / "validation_report.json"),
            })
            print(f"[table2]   -> {status}  {(report.get('failures') or [''])[0][:160]}")

    per_ee = {}
    for ee in ee_ids:
        inferred = [t for t in trials if t["ee"] == ee and t["inferred_feasible"]]
        passed = sum(1 for t in inferred if t["status"] == "PASS")
        not_reg = [t["object"] for t in inferred if t["status"] in ("NOT_REGISTERED", "EE_MISMATCH")]
        failed = [t["object"] for t in inferred if t["status"] == "FAIL"]
        exp = TABLE2.get(task, {}).get(ee)
        per_ee[ee] = {
            "inferred": len(inferred), "grasped": passed,
            "ratio": f"{passed}/{len(inferred)}",
            "table2": f"{exp[0]}/{exp[1]}" if exp else "-",
            "match": bool(exp and passed == exp[0] and len(inferred) == exp[1]),
            "not_registered": not_reg, "failed": failed,
            "extra_pass_outside_inference": [t["object"] for t in trials
                                             if t["ee"] == ee and not t["inferred_feasible"] and t["status"] == "PASS"],
        }
    return {"task": task, "environment": environment, "seed": seed, "denominator_source": source,
            "inferred_feasible": feasible, "trials": trials, "per_ee": per_ee}


def render_md(results: list[dict]) -> str:
    lines = ["# Table II verification — inferred-feasible (object, EE) actually grasped in sim", ""]
    for res in results:
        lines += [f"## {res['task']} ({res['environment']}, seed={res['seed']}, 분모={res['denominator_source']})", "",
                  "| EE | grasped / inferred | Table II | match | not registered (실행 불가) | failed (레시피 있음) | 추론 밖 PASS |",
                  "|---|---|---|---|---|---|---|"]
        for ee, v in res["per_ee"].items():
            lines.append(f"| {ee} | {v['ratio']} | {v['table2']} | {'OK' if v['match'] else 'MISMATCH'} | "
                         f"{','.join(v['not_registered'])} | {','.join(v['failed'])} | {','.join(v['extra_pass_outside_inference'])} |")
        lines += ["", "| object | EE | inferred | status | integration | failure |", "|---|---|---|---|---|---|"]
        for t in res["trials"]:
            lines.append(f"| {t['object']} | {t['ee']} | {'Y' if t['inferred_feasible'] else '-'} | {t['status']} | "
                         f"{t.get('integration_status') or ''} | {((t.get('failures') or [''])[0])[:120]} |")
        lines.append("")
    lines += ["해석: Table II 분자는 PASS 만 센다. NOT_REGISTERED 는 scripted 레시피가 없어 이 하네스로는 실행하지 못한 조합이며, "
              "논문 Table II 가 이 조합을 '잡힘'으로 셌다면 그 실행은 다른 경로(LLM 키프레임 파지 등)에서 나온 것이다.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default="c1_1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--denominator", choices=("reasoner", "gt-rule"), default="reasoner",
                   help="분모: reasoner=Full M1 판정(기본), gt-rule=YCB 질량+AABB 규칙")
    p.add_argument("--all-ees", action="store_true", help="추론 결과와 무관하게 3 EE 전부 시도 (추론 밖 PASS 탐지)")
    p.add_argument("--video", action="store_true")
    p.add_argument("--out-root", type=Path, default=ROOT / "output" / "table2_verify")
    p.add_argument("--doc", type=Path, default=ROOT / "docs" / "table2_verification.md")
    args = p.parse_args(argv)
    tasks = list(TASK_GRASP_OBJECTS) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    robot = json.loads((ROOT / "configs" / "robot_spec.json").read_text(encoding="utf-8"))
    ee_ids = [e["ee_id"] for e in robot["ee_pool"]]
    results = [run_task(t, seed=args.seed, denominator=args.denominator, all_ees=args.all_ees,
                        video=args.video, out_root=args.out_root, ee_ids=ee_ids) for t in tasks]
    md = render_md(results)
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(md, encoding="utf-8")
    args.doc.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)
    print(f"[table2] -> {args.doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
