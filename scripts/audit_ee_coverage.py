# -*- coding: utf-8 -*-
"""태스크별 (파지 대상 물체 × EE) 커버리지 감사 — Table II 재검증 + SCRIPTED_GRASP_EE_INFEASIBLE 원인 추적.

각 태스크에서 실제로 잡아야 하는 물체(도구 + 타겟)마다 세 가지 집합을 나란히 놓는다.

  GT-rule      GT(질량 = configs/ycb_mass_gt.json 의 YCB 공식값, 없으면 MuJoCo subtree mass 폴백+표시;
               치수 = 컴파일된 geom AABB)를
               Full 과 같은 규칙 `m1_scene.ee_rules.evaluate_ee` 에 넣어 통과한 EE
  M1(Full)     output/<task>/m1.json 에 기록된 Reasoner 판정 (있을 때만)
  registry     scripted_grasps 레지스트리가 이 환경에서 지원하는 (물체, EE) 조합
               (production = ENABLED + ALTERNATIVE, 괄호 안은 실험/greedy 전용)

그리고 두 가지를 판정한다.
  · Table II 와 비교: EE 별 "GT-rule 로 잡을 수 있는 물체 수" vs 논문 표 숫자
  · 커버리지 공백: GT-rule 로 가능한데 registry 에 레시피가 없는 조합
    → M4 의 `--execution-environment` 제약에서 SCRIPTED_GRASP_EE_INFEASIBLE 로 죽는 원인.
      w/o reasoner 가 그 EE 를 고르면 실행도 못 해 보고 끝난다.

실행 (robosuite/MuJoCo 가 있는 환경, API 호출 없음):
  python scripts/audit_ee_coverage.py --tasks c1_1,c1_2,c2_1
  python scripts/audit_ee_coverage.py --tasks all --out docs/ee_coverage_audit.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from task_registry import TASKS  # noqa: E402

# 태스크별 "파지 대상" 물체 — Table II 분모와 맞춘다 (C1-1 5, C1-2 4, C2-1 5, C2-2 5, C3-1 5,
# C3-2 12, C4-1 current scene 5 (historical Table II: 4), C4-2 6). 환경의 obj_body_id 키 이름을 쓴다. 틀리면 여기만 고친다.
TASK_GRASP_OBJECTS: dict[str, list[str]] = {
    "c1_1": ["plate", "ladle", "fork", "scissors", "bottle_distractor"],
    "c1_2": ["bottle", "spatula", "spoon", "tongs"],
    "c2_1": ["apple", "bread", "mug", "plate", "spoon"],
    "c3_1": ["apple", "bread", "mug", "plate", "spoon"],
    "c2_2": ["bread_a", "bread_b", "turkey_1", "cheese_1", "tomato_slice"],
    "c3_2": ["plate_a", "plate_b", "mug_a", "mug_b", "fork_a", "fork_b",
             "spoon_a", "spoon_b", "bread_a", "bread_b", "fruit_a", "fruit_b"],
    # Current environment disables the knife and adds spatula_c / cutting_board.
    "c4_1": ["tool_2_spatula_a", "tool_3_spatula_b", "tool_4_spatula_c",
             "tool_5_cutting_board", "card"],
    "c4_2": ["rolling_pin", "baguette", "whisk", "cereal", "milk", "lid"],
}

# 논문 Table II (Fixed-EE object coverage). {task: {ee: (covered, total)}}
TABLE2 = {
    "c1_1": {"2F": (4, 5), "3F": (4, 5), "vac": (1, 5)},
    "c1_2": {"2F": (3, 4), "3F": (3, 4), "vac": (0, 4)},
    "c2_1": {"2F": (4, 5), "3F": (4, 5), "vac": (1, 5)},
    "c2_2": {"2F": (0, 5), "3F": (2, 5), "vac": (5, 5)},
    "c3_1": {"2F": (4, 5), "3F": (4, 5), "vac": (1, 5)},
    "c3_2": {"2F": (8, 12), "3F": (10, 12), "vac": (4, 12)},
    "c4_1": {"2F": (4, 4), "3F": (3, 4), "vac": (2, 4)},
    "c4_2": {"2F": (4, 6), "3F": (1, 6), "vac": (1, 6)},
}


# ----------------------------------------------------------------- GT from sim
def _subtree_ids(model, root_id: int) -> set[int]:
    ids = {int(root_id)}
    for bid in range(int(root_id) + 1, model.nbody):
        if int(model.body_parentid[bid]) in ids:
            ids.add(bid)
    return ids


def _world_aabb(sim, body_ids: set[int]):
    """컴파일된 geom 들의 월드 AABB. geom_aabb(MuJoCo>=2.3.2) 우선, 없으면 rbound 구."""
    import numpy as np

    model, data = sim.model, sim.data
    lows, highs = [], []
    has_aabb = hasattr(model, "geom_aabb")
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) not in body_ids:
            continue
        if int(model.geom_group[gid]) not in (0, 1):
            continue
        center = np.asarray(data.geom_xpos[gid], dtype=float)
        if has_aabb:
            local_c = np.asarray(model.geom_aabb[gid][:3], dtype=float)
            half = np.asarray(model.geom_aabb[gid][3:], dtype=float)
            rot = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)
            c = center + rot @ local_c
            ext = np.abs(rot) @ half
            lows.append(c - ext)
            highs.append(c + ext)
        else:
            r = float(model.geom_rbound[gid])
            lows.append(center - r)
            highs.append(center + r)
    if not lows:
        raise ValueError("no physical geoms")
    return np.min(lows, axis=0), np.max(highs, axis=0)


def load_ycb_masses(path: Path) -> dict[str, float]:
    """configs/ycb_mass_gt.json → {object: mass_kg} (null 제외)."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: float(v["mass_kg"]) for k, v in (data.get("objects") or {}).items()
            if isinstance(v, dict) and v.get("mass_kg") is not None}


def gt_scene_from_env(env, names: list[str], ycb: dict[str, float] | None = None) -> dict:
    import numpy as np

    ycb = ycb or {}
    sim = env.sim
    body_ids = getattr(env, "obj_body_id", None) or {}
    report = {}
    for name in names:
        bid = body_ids.get(name)
        if bid is None:
            objects = getattr(env, "objects", {}) or {}
            if name in objects:
                bid = sim.model.body_name2id(objects[name].root_body)
        if bid is None:
            report[name] = {"error": f"{name}: obj_body_id / objects 에 없음"}
            continue
        ids = _subtree_ids(sim.model, int(bid))
        lo, hi = _world_aabb(sim, ids)
        sim_mass = float(np.asarray(sim.model.body_mass)[sorted(ids)].sum())
        report[name] = {
            "mass_kg": ycb[name] if name in ycb else sim_mass,
            "mass_source": "ycb" if name in ycb else "mujoco_subtree(unverified)",
            "sim_mass_kg": sim_mass,
            "size_m": (hi - lo).tolist(),
            "position_m": np.asarray(sim.data.body_xpos[int(bid)]).tolist(),
        }
    return report


def make_env(task: str, seed: int):
    import environments  # noqa: F401  (robosuite 등록)
    from tuj.m5_motion.tool_use_journal import make_tool_use_journal_env

    env = make_tool_use_journal_env(
        ROOT, TASKS[task].cls, active_ee=None,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        ignore_done=True, seed=seed,
    )
    env.reset()
    return env


# --------------------------------------------------------------- registry view
def registry_support(object_id: str, environment: str) -> dict[str, list[str]]:
    from tuj.m5_motion.scripted_grasps import registry as R

    prod = R.ENABLED_ENTRIES + R.ALTERNATIVE_ENTRIES
    extra = tuple(R.VALIDATOR_EXPERIMENTAL_ENTRIES) + tuple(R.GREEDY_EXTRA_ENTRIES)

    def ees(entries):
        out = []
        for key in R._recipe_keys_for_target(object_id, environment):
            found = [e.ee for e in entries if e.environment == environment and e.object_id == key]
            if found:
                out = list(dict.fromkeys(found))
                break
        return out

    return {"production": ees(prod), "experimental": [e for e in ees(extra) if e not in ees(prod)]}


# ------------------------------------------------------------------- audit
def audit_task(task: str, *, seed: int, m1_path: Path | None) -> dict:
    from tuj.gt.labels import intrinsic_from_gt, normalize_ee_pool
    from tuj.m1_scene.ee_rules import evaluate_ee

    robot = json.loads((ROOT / "configs" / "robot_spec.json").read_text(encoding="utf-8"))
    pool = normalize_ee_pool(robot["ee_pool"])
    ee_ids = [e["ee_id"] for e in pool]
    environment = TASKS[task].cls
    names = TASK_GRASP_OBJECTS[task]

    ycb = load_ycb_masses(ROOT / "configs" / "ycb_mass_gt.json")
    env = make_env(task, seed)
    try:
        gt = gt_scene_from_env(env, names, ycb)
    finally:
        env.close()

    m1_nodes = {}
    if m1_path and m1_path.is_file():
        m1 = json.loads(m1_path.read_text(encoding="utf-8"))
        for n in m1.get("nodes", []):
            nid = n.get("id", "")
            cls = n.get("class", "")
            short = nid[len(f"obj_{cls}_"):] if nid.startswith(f"obj_{cls}_") else nid
            m1_nodes[short] = n
            m1_nodes[nid] = n

    rows = []
    for name in names:
        rec = gt[name]
        if "error" in rec:
            rows.append({"object": name, "error": rec["error"]})
            continue
        node = m1_nodes.get(name)
        intrinsic = intrinsic_from_gt(rec, m1_node=node)
        verdicts = {e["ee_id"]: evaluate_ee(e, intrinsic) for e in pool}
        gt_feasible = [ee for ee in ee_ids if verdicts[ee]["feasible"]]
        m1_feasible = (sorted(k for k, v in (node.get("ee") or {}).items() if v.get("feasible"))
                       if node else None)
        reg = registry_support(name, environment)
        gap = [ee for ee in gt_feasible if ee not in reg["production"]]
        unsafe = [ee for ee in reg["production"] if ee not in gt_feasible]
        rows.append({
            "object": name,
            "mass_kg": round(rec["mass_kg"], 4),
            "mass_source": rec["mass_source"],
            "sim_mass_kg": round(rec["sim_mass_kg"], 4),
            "size_mm": [round(v * 1000, 1) for v in rec["size_m"]],
            "gt_feasible": gt_feasible,
            "gt_reasons": {ee: verdicts[ee]["reason"] for ee in ee_ids if not verdicts[ee]["feasible"]},
            "m1_feasible": m1_feasible,
            "registry_production": reg["production"],
            "registry_experimental": reg["experimental"],
            "gap_gt_feasible_but_no_recipe": gap,
            "recipe_but_gt_infeasible": unsafe,
            "m1_vs_gt_mismatch": (sorted(m1_feasible) != sorted(gt_feasible)) if m1_feasible is not None else None,
        })

    per_ee = {}
    for ee in ee_ids:
        covered = sum(1 for r in rows if ee in r.get("gt_feasible", []))
        recipe = sum(1 for r in rows if ee in r.get("registry_production", []))
        both = sum(1 for r in rows if ee in r.get("gt_feasible", []) and ee in r.get("registry_production", []))
        exp = TABLE2.get(task, {}).get(ee)
        per_ee[ee] = {"gt_rule": f"{covered}/{len(names)}", "registry": f"{recipe}/{len(names)}",
                      "executable(gt∩registry)": f"{both}/{len(names)}",
                      "table2": f"{exp[0]}/{exp[1]}" if exp else "-",
                      "match_table2": (exp is not None and covered == exp[0] and len(names) == exp[1])}
    return {"task": task, "environment": environment, "seed": seed, "objects": rows, "per_ee": per_ee}


def render_md(results: list[dict]) -> str:
    lines = ["# EE coverage audit (GT rule vs Reasoner vs scripted registry vs Table II)", ""]
    for res in results:
        lines += [f"## {res['task']}  ({res['environment']}, seed={res['seed']})", "",
                  "| object | mass kg | size mm | GT-rule feasible | M1(Full) | registry prod (exp) | gap: GT ok but no recipe | recipe but GT infeasible |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in res["objects"]:
            if "error" in r:
                lines.append(f"| {r['object']} | – | – | ERROR: {r['error']} | | | | |")
                continue
            m1 = ",".join(r["m1_feasible"]) if r["m1_feasible"] is not None else "–"
            if r["m1_vs_gt_mismatch"]:
                m1 += " ≠GT"
            exp = f" ({','.join(r['registry_experimental'])})" if r["registry_experimental"] else ""
            mass = f"{r['mass_kg']}" + ("" if r.get("mass_source") == "ycb" else " (sim, 미확인)")
            lines.append(
                f"| {r['object']} | {mass} | {'×'.join(str(v) for v in r['size_mm'])} | "
                f"{','.join(r['gt_feasible']) or '∅'} | {m1} | {','.join(r['registry_production']) or '∅'}{exp} | "
                f"{','.join(r['gap_gt_feasible_but_no_recipe']) or ''} | {','.join(r['recipe_but_gt_infeasible']) or ''} |")
        lines += ["", "| EE | GT-rule coverage | registry coverage | executable (GT ∩ registry) | Table II | match |", "|---|---|---|---|---|---|"]
        for ee, v in res["per_ee"].items():
            lines.append(f"| {ee} | {v['gt_rule']} | {v['registry']} | {v['executable(gt∩registry)']} | {v['table2']} | "
                         f"{'OK' if v['match_table2'] else 'MISMATCH'} |")
        gaps = [(r["object"], r["gap_gt_feasible_but_no_recipe"]) for r in res["objects"]
                if r.get("gap_gt_feasible_but_no_recipe")]
        lines.append("")
        if gaps:
            lines.append("**SCRIPTED_GRASP_EE_INFEASIBLE 위험 조합** (규칙상 가능하지만 레시피 없음): "
                         + "; ".join(f"{o}→{','.join(e)}" for o, e in gaps))
        else:
            lines.append("레시피 공백 없음.")
        reasons = [(r["object"], r["gt_reasons"]) for r in res["objects"] if r.get("gt_reasons")]
        if reasons:
            lines += ["", "<details><summary>GT 규칙 탈락 사유</summary>", ""]
            for o, rs in reasons:
                for ee, why in rs.items():
                    lines.append(f"- {o} / {ee}: {why}")
            lines += ["", "</details>"]
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default="c1_1,c1_2,c2_1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--m1-dir", type=Path, default=ROOT / "output",
                   help="Full M1 결과(output/<task>/m1.json)를 읽을 루트; rms/seal-patch 와 M1 판정 비교용")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "ee_coverage_audit.md")
    args = p.parse_args(argv)
    tasks = list(TASK_GRASP_OBJECTS) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        sys.exit(f"[audit] 등록되지 않은 task: {unknown}")
    results = []
    for task in tasks:
        print(f"[audit] {task} ...", flush=True)
        results.append(audit_task(task, seed=args.seed, m1_path=args.m1_dir / task / "m1.json"))
    md = render_md(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)
    print(f"\n[audit] -> {args.out}  (+ {args.out.with_suffix('.json')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
