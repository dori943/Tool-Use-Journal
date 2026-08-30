"""M3 질량 추정 품질 평가 — 추정치(m3_intrinsic.json) vs sim GT(subtree mass).

배치: Tool-Use-Journal/scripts/eval_m3_mass_c1_1.py
선행: python scripts/run_m3_c1_1.py --backend siphy   (평가 대상 추정 생성)
실행: python scripts/eval_m3_mass_c1_1.py

지표:
  · rel_err  = |est − gt| / gt                (상대오차)
  · ALDE     = |ln(est / gt)|                 (SiPhy 논문 지표 — 로그 오차)
  · in_range = gt ∈ [mass_range_kg]           (불확실성 범위가 GT를 덮는가)
  · decision = est 기반 EE feasible 집합 == gt 기반 집합   ★ 논문 핵심 지표
    (질량만 GT로 치환해 evaluate_ee 재실행 — decision-sufficient accuracy)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

TASK = "c1_1"
OUT = ROOT / "output" / TASK


def _subtree_geom_ids(m, root_bid):
    ids = []
    for gid in range(m.ngeom):
        bid = m.geom_bodyid[gid]
        while bid not in (0, root_bid):
            bid = m.body_parentid[bid]
        if bid == root_bid and m.geom_contype[gid]:   # 충돌 geom만 (visual 제외)
            ids.append(gid)
    return ids


def gt_from_sim() -> dict[str, dict]:
    """sim 컴파일 결과에서 GT: 질량(subtree), μ(geom friction 1축), 치수(mesh 정점 tight bbox)."""
    import os, platform
    if platform.system() == "Windows" and os.environ.get("MUJOCO_GL", "").lower() not in ("", "wgl", "glfw"):
        os.environ["MUJOCO_GL"] = "wgl"
    import numpy as np
    import environments  # noqa: F401
    import robosuite as suite
    env = suite.make(env_name="C1_1_LegoSweep", robots="UR5e",
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, render_camera="agentview", ignore_done=True)
    env.reset()
    m, d = env.sim.model, env.sim.data
    objs = list(env.blocks) + [env.light_plate, env.heavy_plate, env.bottle_distractor]
    gt = {}
    for o in objs:
        bid = m.body_name2id(o.root_body)
        gids = _subtree_geom_ids(m, bid)
        mus = [float(m.geom_friction[g][0]) for g in gids]
        # tight bbox: mesh 정점(월드 회전 반영) ∪ primitive size
        lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
        for g in gids:
            R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]
            did = m.geom_dataid[g]
            if did >= 0:                                   # mesh geom
                v0, nv = m.mesh_vertadr[did], m.mesh_vertnum[did]
                V = m.mesh_vert[v0:v0 + nv].reshape(-1, 3) @ R.T + c
            else:                                          # primitive → size 근사
                s = m.geom_size[g]
                V = np.array([c - s, c + s])
            lo = np.minimum(lo, V.min(0)); hi = np.maximum(hi, V.max(0))
        gt[o.name] = {"mass_kg": float(m.body_subtreemass[bid]),
                      "mu": max(mus) if mus else None,
                      "size_mm": [round(float(x) * 1000, 1) for x in (hi - lo)]}
    env.close()
    return gt


def evaluate(est_cache: dict, gt: dict, ee_pool: list[dict]) -> list[dict]:
    """노드별 지표 계산 (순수 함수 — 테스트 가능). gt: {inst: mass float | {mass_kg, mu, size_mm}}"""
    from tuj.m3_grounding import evaluate_ee
    rows = []
    for nid, est in est_cache.items():
        inst = nid.split("_", 2)[-1]                   # obj_plate_light_plate → light_plate
        if inst not in gt or est.get("mass_kg") in (None, 0):
            continue
        gt_i = gt[inst] if isinstance(gt[inst], dict) else {"mass_kg": gt[inst]}
        e, g = float(est["mass_kg"]), float(gt_i["mass_kg"])
        lo, hi = est.get("mass_range_kg", [e, e])
        est_feas = {k for k, v in ((ee["ee_id"], evaluate_ee(ee, est)) for ee in ee_pool)
                    if v["feasible"]}
        gt_int = dict(est) | {"mass_kg": g}            # 질량만 GT 치환
        gt_feas = {k for k, v in ((ee["ee_id"], evaluate_ee(ee, gt_int)) for ee in ee_pool)
                   if v["feasible"]}
        row = {
            "node": nid, "gt_kg": round(g, 3), "est_kg": round(e, 3),
            "range_kg": [round(lo, 3), round(hi, 3)],
            "rel_err": round(abs(e - g) / g, 3),
            "alde": round(abs(math.log(max(e, 1e-9) / g)), 3),
            "in_range": bool(lo <= g <= hi),
            "est_feasible": sorted(est_feas), "gt_feasible": sorted(gt_feas),
            "decision_agree": est_feas == gt_feas,
        }
        # μ 비교 (※ XML friction이 물리값이 아닌 sim 튜닝값이면 GT 재정의 필요 — 콘솔 경고)
        if gt_i.get("mu") is not None and est.get("mu"):
            mu_e, mu_g = float(est["mu"]["mu"]), float(gt_i["mu"])
            row |= {"mu_est": round(mu_e, 3), "mu_gt": round(mu_g, 3),
                    "mu_rel_err": round(abs(mu_e - mu_g) / mu_g, 3)}
        # 치수 비교 (bbox 대각 기준)
        if gt_i.get("size_mm") and est.get("geometry", {}).get("extents_mm"):
            se = sorted(est["geometry"]["extents_mm"], reverse=True)
            sg = sorted(gt_i["size_mm"], reverse=True)
            row |= {"size_est_mm": [round(x, 1) for x in se], "size_gt_mm": sg,
                    "size_err_mm": round(max(abs(a - b) for a, b in zip(se, sg)), 1)}
        rows.append(row)
    return rows


def main():
    cache_p = OUT / "m3_intrinsic.json"
    if not cache_p.exists():
        sys.exit("[err] m3_intrinsic.json 없음 — 먼저 run_m3_c1_1.py --backend siphy 실행")
    est_cache = json.loads(cache_p.read_text(encoding="utf-8"))

    spec = json.loads((ROOT / "configs" / "robot_spec.json").read_text(encoding="utf-8"))
    ee_pool = []
    for e in spec["ee_pool"]:
        e = dict(e)
        if "flatness_tol_rms_mm" in e:
            e["seal_rms_tol_mm"] = e["flatness_tol_rms_mm"]
        ee_pool.append(e)

    print("[eval] sim에서 GT(질량·μ·치수) 읽는 중...")
    gt = gt_from_sim()
    rows = evaluate(est_cache, gt, ee_pool)
    if not rows:
        sys.exit("[err] 평가 가능한 노드 없음 (GT 매칭 실패)")

    print(f"\n{'node':32s} {'GT':>7s} {'est':>7s} {'range':>16s} "
          f"{'rel':>6s} {'ALDE':>6s} {'in':>3s}  est_EE -> gt_EE  agree")
    for r in rows:
        print(f"{r['node']:32s} {r['gt_kg']:7.3f} {r['est_kg']:7.3f} "
              f"{str(r['range_kg']):>16s} {r['rel_err']:6.2f} {r['alde']:6.2f} "
              f"{'O' if r['in_range'] else 'X':>3s}  "
              f"{','.join(r['est_feasible']) or '-'} -> {','.join(r['gt_feasible']) or '-'}  "
              f"{'O' if r['decision_agree'] else '★MISS'}")

    n = len(rows)
    print(f"\n[summary] n={n}  mean_rel_err={sum(r['rel_err'] for r in rows)/n:.2f}  "
          f"mean_ALDE={sum(r['alde'] for r in rows)/n:.2f}  "
          f"range_coverage={sum(r['in_range'] for r in rows)}/{n}  "
          f"decision_agreement={sum(r['decision_agree'] for r in rows)}/{n}")
    if any("mu_est" in r for r in rows):
        print(f"\n{'node':32s} {'μ_est':>6s} {'μ_gt':>6s} {'rel':>6s}   "
              f"{'size_est(정렬)':>22s} {'size_gt':>22s} {'err':>6s}")
        for r in rows:
            if "mu_est" in r:
                print(f"{r['node']:32s} {r['mu_est']:6.2f} {r['mu_gt']:6.2f} "
                      f"{r['mu_rel_err']:6.2f}   {str(r.get('size_est_mm', '-')):>22s} "
                      f"{str(r.get('size_gt_mm', '-')):>22s} {r.get('size_err_mm', float('nan')):6.1f}")
        print("[warn] μ GT는 XML geom friction — robosuite 튜닝값(예: 접시 0.95)이면 물리 재질값과")
        print("       불일치. 카탈로그 기반 GT μ 재설정 전까지 μ 오차는 참고치로만 해석할 것.")
    print("\n[note] light/heavy 접시의 decision MISS는 시각 동일성의 원리적 한계 →")
    print("       프로브 필요 케이스의 실측 증거 (둘 중 하나는 반드시 틀리는 게 정상)")

    (OUT / "eval_mass.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] -> {OUT / 'eval_mass.json'}")


if __name__ == "__main__":
    main()
