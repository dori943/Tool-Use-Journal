# -*- coding: utf-8 -*-
"""vac(흡착) GT 판정 — Table II 의 vac "실제 성공" 을 시뮬 부착만으로 세지 않기 위한 기준.

시뮬의 흡착은 접촉 게이트 + 운동학 부착이라 컵이 닿기만 하면 붙는다. 그래서 vac 의
"실제 잡힘" 은  (시뮬 PASS) ∧ (GT 기하·질량이 vac 규칙을 통과)  로 정의한다.
규칙은 Full M1 이 쓰는 `m1_scene.ee_rules.evaluate_ee` 그대로이고, 입력만 추정값 대신 GT:

  · 점군   : seed-0 장면에서 물체 메시/기본체 표면을 조밀하게 샘플한 뒤, top-down 카메라처럼
             xy 셀당 최고점만 남김 (M1 과 같은 가시성 조건, 센서 노이즈 없음)
  · 기하   : M1 과 같은 함수 surface_rms / seal_patch_rms_mm / pca_dims
  · 질량   : configs/ycb_mass_gt.json (YCB), 없으면 MuJoCo subtree mass (표시)
  · 재질   : 에셋 XML material_gt — 규칙에는 들어가지 않고 참고 열로만 기록(다공성 힌트)

출력: docs/vac_gt_admissibility.json / .md  →  scripts/table2_recompute.py --vac-gt 에 넣는다.

  python scripts/vac_gt_admissibility.py --tasks all
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from task_registry import TASKS  # noqa: E402
from audit_ee_coverage import TASK_GRASP_OBJECTS, load_ycb_masses, make_env  # noqa: E402

TASK_ORDER = ["c1_1", "c1_2", "c2_1", "c2_2", "c3_1", "c3_2", "c4_1", "c4_2"]
# 산업 자료(Piab/Schmalz) 기준 다공·투기성 재질 — 규칙에 쓰지 않고 참고 표시만.
POROUS_HINT = {"cardboard", "bread", "dough", "paper", "fabric", "foam"}


# ------------------------------------------------------------ surface sampling
def _sample_mesh(model, gid, n: int, rng):
    import numpy as np

    mid = int(model.geom_dataid[gid])
    va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    verts = np.asarray(model.mesh_vert[va:va + vn], dtype=float)
    faces = np.asarray(model.mesh_face[fa:fa + fn], dtype=int)
    tri = verts[faces]                                   # (F, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if area.sum() <= 0:
        return np.empty((0, 3))
    idx = rng.choice(len(tri), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def _sample_primitive(model, gid, n: int, rng):
    """box / cylinder / sphere / capsule / ellipsoid 표면 샘플 (geom 로컬 좌표)."""
    import mujoco
    import numpy as np

    kind = int(model.geom_type[gid])
    s = np.asarray(model.geom_size[gid], dtype=float)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        faces = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
        w = np.array([s[1] * s[2], s[1] * s[2], s[0] * s[2], s[0] * s[2], s[0] * s[1], s[0] * s[1]])
        f = rng.choice(6, size=n, p=w / w.sum())
        pts = (rng.random((n, 3)) * 2 - 1) * s
        for k in range(3):
            sel = np.abs(faces[f, k]) > 0
            pts[sel, k] = faces[f[sel], k] * s[k]
        return pts
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        return d * s[0]
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        return d * s[:3]
    if kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        r, h = s[0], s[1]
        side = 2 * np.pi * r * 2 * h
        cap = np.pi * r * r
        f = rng.choice(3, size=n, p=np.array([side, cap, cap]) / (side + 2 * cap))
        th = rng.random(n) * 2 * np.pi
        pts = np.zeros((n, 3))
        sel = f == 0
        pts[sel] = np.c_[r * np.cos(th[sel]), r * np.sin(th[sel]), (rng.random(sel.sum()) * 2 - 1) * h]
        for sign, k in ((1, 1), (-1, 2)):
            sel = f == k
            rr = r * np.sqrt(rng.random(sel.sum()))
            pts[sel] = np.c_[rr * np.cos(th[sel]), rr * np.sin(th[sel]), np.full(sel.sum(), sign * h)]
        return pts
    return np.empty((0, 3))


def object_surface_points_mm(env, body_id: int, n_total: int = 30000, seed: int = 0):
    """물체(서브트리)의 물리 geom 표면을 샘플해 월드 좌표(mm)로 반환."""
    import mujoco
    import numpy as np

    rng = np.random.default_rng(seed)
    model, data = env.sim.model, env.sim.data
    ids = {int(body_id)}
    for bid in range(int(body_id) + 1, model.nbody):
        if int(model.body_parentid[bid]) in ids:
            ids.add(bid)
    gids = [g for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in ids and int(model.geom_group[g]) in (0, 1)
            and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not gids:   # 충돌 geom 이 없으면 시각 geom 으로 대체
        gids = [g for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) in ids and int(model.geom_group[g]) in (0, 1)]
    per = max(500, n_total // max(1, len(gids)))
    out = []
    for g in gids:
        local = (_sample_mesh(model, g, per, rng) if int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH
                 else _sample_primitive(model, g, per, rng))
        if len(local) == 0:
            continue
        R = np.asarray(data.geom_xmat[g], dtype=float).reshape(3, 3)
        c = np.asarray(data.geom_xpos[g], dtype=float)
        out.append((local @ R.T + c) * 1000.0)
    pts = np.vstack(out) if out else np.empty((0, 3))
    return top_visible_points(pts)


def top_visible_points(pts_mm, cell_mm: float = 1.0):
    """위에서 내려다보는 깊이 카메라처럼, xy 셀마다 가장 높은 점만 남긴다.

    M1 의 점군은 top-down 카메라 깊이에서 오므로 측면·바닥은 보이지 않는다. 메시 전체를
    샘플하면 키 큰 물체(시리얼 상자, 우유팩, 병, 머그)의 측면이 '상면 밴드' 에 섞여
    RMS 가 부풀어 M1 과 다른 기준이 된다. 같은 가시성 조건으로 맞춘다.
    """
    import numpy as np

    if len(pts_mm) == 0:
        return pts_mm
    xy = np.floor(pts_mm[:, :2] / cell_mm).astype(np.int64)
    key = xy[:, 0] * 1_000_003 + xy[:, 1]
    order = np.lexsort((-pts_mm[:, 2], key))        # key 오름차순, 같은 key 안에서 z 내림차순
    key_sorted = key[order]
    first = np.r_[True, key_sorted[1:] != key_sorted[:-1]]
    return pts_mm[order[first]]


# ------------------------------------------------------------------- material
def material_of(object_id: str) -> str | None:
    base = object_id
    for suffix in ("_a", "_b", "_1", "_distractor"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    aliases = {"bread_a": "sandwich_bread_a", "bread_b": "sandwich_bread_b", "turkey": "turkey_slice",
               "tomato": "tomato_slice", "fruit": None, "tool_1_knife": "knife",
               "tool_2_spatula": "spatula", "tool_3_spatula": "spatula", "mug": "mug_3143a4ac"}
    for cand in (aliases.get(object_id), aliases.get(base), base, object_id):
        if not cand:
            continue
        xml = ROOT / "environments" / "assets" / "objects" / cand / "model.xml"
        if xml.is_file():
            node = ET.parse(xml).getroot().find("./custom/text[@name='material_gt']")
            if node is not None and node.get("data"):
                return node.get("data")
    return None


# ----------------------------------------------------------------------- main
def evaluate_task(task: str, seed: int, n_points: int) -> dict:
    import numpy as np
    from tuj.gt.labels import normalize_ee_pool
    from tuj.m1_scene.ee_rules import evaluate_ee
    from tuj.m1_scene.grounding import pca_dims, seal_patch_rms_mm, surface_rms

    robot = json.loads((ROOT / "configs" / "robot_spec.json").read_text(encoding="utf-8"))
    vac = next(e for e in normalize_ee_pool(robot["ee_pool"]) if e["type"] == "vacuum")
    ycb = load_ycb_masses(ROOT / "configs" / "ycb_mass_gt.json")
    names = TASK_GRASP_OBJECTS[task]
    env = make_env(task, seed)
    rows = {}
    try:
        body_ids = getattr(env, "obj_body_id", {}) or {}
        for name in names:
            bid = body_ids.get(name)
            if bid is None and name in (getattr(env, "objects", {}) or {}):
                bid = env.sim.model.body_name2id(env.objects[name].root_body)
            if bid is None:
                rows[name] = {"error": "object not found in env"}
                continue
            pts = object_surface_points_mm(env, int(bid), n_total=n_points, seed=seed)
            if len(pts) < 50:
                rows[name] = {"error": "no surface points"}
                continue
            dims = pca_dims(pts)
            bbox = np.ptp(pts, axis=0)
            geom = dims | {
                "surface_rms_mm": round(float(surface_rms(pts)), 3),
                "seal_patch_rms_mm": float(seal_patch_rms_mm(pts)),
                "footprint_mm": [round(float(bbox[0]), 1), round(float(bbox[1]), 1)],
                "height_mm": round(float(bbox[2]), 1),
            }
            ids = {int(bid)}
            for b in range(int(bid) + 1, env.sim.model.nbody):
                if int(env.sim.model.body_parentid[b]) in ids:
                    ids.add(b)
            sim_mass = float(np.asarray(env.sim.model.body_mass)[sorted(ids)].sum())
            mass = ycb.get(name, sim_mass)
            verdict = evaluate_ee(vac, {"geometry": geom, "mass_kg": mass, "mu": {"mu": 0.5}})
            material = material_of(name)
            rows[name] = {
                "geometry": geom, "mass_kg": round(mass, 4),
                "mass_source": "ycb" if name in ycb else "mujoco_subtree(unverified)",
                "material_gt": material,
                "porous_hint": bool(material and any(k in material for k in POROUS_HINT)),
                "vac_rule": {"feasible": bool(verdict["feasible"]), "reason": verdict["reason"],
                             "checks": verdict["checks"]},
                "admissible": bool(verdict["feasible"]),   # 규칙 그대로; 재질은 참고만
                "n_points": int(len(pts)),
            }
    finally:
        env.close()
    return {"task": task, "environment": TASKS[task].cls, "seed": seed,
            "rule": "m1_scene.ee_rules.evaluate_ee(vac) on GT surface samples + YCB mass",
            "vac_spec": vac, "objects": rows}


def render_md(results: list[dict]) -> str:
    L = ["# vac GT admissibility (seed 0) — Table II vac 성공 = 시뮬 PASS ∧ 아래 admissible", "",
         "| task | object | admissible | surface RMS | seal patch RMS | contact width | mass (src) | material | porous? | reason |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for res in results:
        for name, r in res["objects"].items():
            if "error" in r:
                L.append(f"| {res['task']} | {name} | ERR | | | | | | | {r['error']} |")
                continue
            g = r["geometry"]
            width = g["diameter_mm"] if g.get("diameter_mm") is not None else sorted(g["extents_mm"], reverse=True)[1]
            L.append(f"| {res['task']} | {name} | {'YES' if r['admissible'] else 'no'} | {g['surface_rms_mm']} | "
                     f"{g['seal_patch_rms_mm']} | {width} | {r['mass_kg']} ({'ycb' if r['mass_source']=='ycb' else 'sim'}) | "
                     f"{r['material_gt'] or '-'} | {'⚠' if r['porous_hint'] else ''} | {'' if r['admissible'] else r['vac_rule']['reason']} |")
    L += ["", "규칙은 Full M1 과 동일(evaluate_ee vac). 재질/다공성은 규칙에 포함되지 않으며 ⚠ 는 산업 자료 기준 참고 표시다.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--points", type=int, default=30000)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "vac_gt_admissibility.md")
    args = p.parse_args(argv)
    tasks = TASK_ORDER if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    results = []
    for t in tasks:
        print(f"[vac-gt] {t} ...", flush=True)
        results.append(evaluate_task(t, args.seed, args.points))
    md = render_md(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)
    print(f"[vac-gt] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
