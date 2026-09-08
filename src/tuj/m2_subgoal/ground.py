# -*- coding: utf-8 -*-
"""M2-6 — 접지값 접근/보완 (0908 파이프라인 재구조).

새 방법론: [M1 & M3] → M2 → M4 → M5 한 방향. M2↔M3 질의 왕복이 없다.
접지값은 m1.json 노드에 실려서 온다:
    node["ee"]            {ee_id: {feasible, margin, reason}}
    node["reachability"]  {reachable, margin_mm}
    node["predicates"]    {top_exposed: {value, blockers}, clear: {...}, flat_face: {...}}
쌍/집합 술어(fits, gap_accessible, batch, swept_space)는 M2가 관계 함수를 import해
직접 계산한다 — 여기서 도는 VLM 호출은 없다.

두 가지를 흡수한다:
  1) 관계 함수 위치. 통합 뒤 이름(relations.batch_partition, relations.swept_space)을
     먼저 찾고, 없으면 현재 레포의 m3_grounding.relational + 종전 Materializer 산술로
     폴백한다. 통합 전후 어느 쪽에서도 M2가 돈다.
  2) 접지값이 아직 m1.json에 없는 경우. m3_intrinsic.json(물성표)만 있으면 EE 판정,
     리치, top_exposed/clear/flat_face는 전부 산술이라 여기서 채운다. 물성표조차 없는
     노드는 접지값 없이 남고 해당 술어는 unknown이 된다 (몰래 VLM을 돌지 않는다).
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import numpy as np

# ── 관계 함수: 통합 후 tuj.m1_scene.relations, 통합 전 m3_grounding.relational ──
try:                                            # 통합 후 (M1&M3 merge)
    from tuj.m1_scene import relations as _rel  # type: ignore
except ImportError:                             # 통합 전 (현 main)
    from tuj.m3_grounding import relational as _rel


def _call(fn, *args, **kw):
    """시그니처에 있는 키워드만 넘긴다 (통합 전후 인자 차이 흡수)."""
    ok = set(inspect.signature(fn).parameters)
    return fn(*args, **{k: v for k, v in kw.items() if k in ok})


def fits_inside(a: dict, b: dict) -> dict:
    return _call(_rel.fits_inside, a, b)


def depth_clearance(a: dict, b: dict) -> dict:
    return _call(_rel.depth_clearance, a, b)


def gap_access(tool: dict, target: dict, gap_width_mm=None) -> dict:
    return _call(_rel.gap_access, tool, target, gap_width_mm=gap_width_mm)


def top_exposed(node_id: str, edges: list[dict]) -> dict:
    fn = getattr(_rel, "top_exposed", None)
    if fn is not None:
        return _call(fn, node_id, edges)
    blockers = [e["from"] for e in edges
                if e.get("to") == node_id and e.get("type") in ("on", "inside")]
    return {"value": not blockers, "blockers": blockers, "pass": not blockers}


def region_clear(region_id: str, edges: list[dict]) -> dict:
    fn = getattr(_rel, "region_clear", None)
    if fn is not None:
        return _call(fn, region_id, edges)
    occ = sorted({e["from"] for e in edges if e.get("to") == region_id
                  and e.get("type") in ("on", "inside", "overlaps")}
                 | {e["to"] for e in edges if e.get("from") == region_id
                    and e.get("type") == "overlaps"})
    return {"value": not occ, "occupants": occ, "pass": not occ}


def flat_face(geometry: dict) -> dict:
    fn = getattr(_rel, "flat_face", None)
    if fn is not None:
        return _call(fn, geometry)
    return _flat_face(geometry)


def batch_partition(members: list[dict], tool: dict | None = None) -> dict:
    """한 액션으로 집합을 동시에 처리할 수 있는가 + 안 되면 그룹 구성.

    tool=None(EE 직접 파지)이면 그리퍼는 한 번에 하나라 물체별 1그룹.
    도구를 쓰면 도구 폭(최소 변)이 용량, 집합의 주축 폭이 수요.
    """
    fn = getattr(_rel, "batch_partition", None)
    if fn is not None:
        return _call(fn, members, tool=tool)
    if tool is None:
        part = [[m["id"]] for m in members]
        return {"feasible": len(members) <= 1, "partition": part,
                "binding_check": "ee_pool_one_per_grasp",
                "checks": [{"rule": "ee_pool_one_per_grasp", "capacity": 1,
                            "demand": len(members), "unit": "count",
                            "margin": 1 - len(members), "pass": len(members) <= 1}]}
    safety = float(os.environ.get("TUJ_BATCH_SAFETY", "1.0"))
    cap = min(tool["bbox_mm"][0], tool["bbox_mm"][1]) * safety
    demand = _rel.group_extent(members)["value_mm"]
    span = lambda ax: (max(m["center_mm"][ax] for m in members)
                       - min(m["center_mm"][ax] for m in members))
    ax = 0 if span(0) >= span(1) else 1
    groups, cur = [], []
    for m in sorted(members, key=lambda m: m["center_mm"][ax]):
        trial = cur + [m]
        if cur and _rel.group_extent(trial)["value_mm"] > cap:
            groups.append(cur); cur = [m]
        else:
            cur = trial
    if cur:
        groups.append(cur)
    part = [[m["id"] for m in g] for g in groups]
    margin = round(cap - demand, 1)
    return {"feasible": len(part) == 1, "partition": part,
            "binding_check": "group_extent_le_tool_width",
            "checks": [{"rule": "group_extent_le_tool_width", "capacity": round(cap, 1),
                        "demand": demand, "unit": "mm", "margin": margin,
                        "pass": bool(margin > 0)}]}


def swept_space(members: list[dict], to_node: dict, others: list[dict],
                tool: dict | None = None) -> dict:
    """액션이 지나가는 통로가 비어 있는가 (정지 간격이 아니라 이동 경로)."""
    fn = getattr(_rel, "swept_space", None)
    if fn is not None:
        return _call(fn, members, to_node, others, tool=tool)
    width = (min(tool["bbox_mm"][0], tool["bbox_mm"][1]) if tool
             else max((min(m["bbox_mm"][0], m["bbox_mm"][1]) for m in members), default=0.0))
    return _rel.corridor_blockers(members, to_node, width, others)


# ── 장면 로딩 + 접지값 보완 ──────────────────────────────────────────────

def load_scene(m1_json: str | Path, points_npz: str | Path | None = None) -> dict:
    """m1.json (+ m1_points.npz) → 접지값이 실린 m1 dict. 점군은 gap() 계산에 쓰인다."""
    m1_json = Path(m1_json)
    m1 = json.loads(m1_json.read_text(encoding="utf-8"))
    npz = Path(points_npz) if points_npz else m1_json.parent / "m1_points.npz"
    if npz.exists():
        pts = np.load(npz)
        for n in m1["nodes"]:
            if n["id"] in pts:
                n["_points"] = pts[n["id"]]
    return m1


def load_ee_pool(robot_spec: str | Path) -> tuple[list[dict], float]:
    spec = json.loads(Path(robot_spec).read_text(encoding="utf-8"))
    pool = []
    for e in spec["ee_pool"]:
        e = dict(e)
        if "flatness_tol_rms_mm" in e:
            e["seal_rms_tol_mm"] = e["flatness_tol_rms_mm"]
        pool.append(e)
    return pool, float(spec["reach_mm"])


def _legacy_props(out_dir: Path) -> dict:
    """통합 전 산출물에서 물성표를 긁어온다 (m3_intrinsic.json → m3.json 순)."""
    p = out_dir / "m3_intrinsic.json"
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {k: v for k, v in raw.items() if isinstance(v, dict) and "geometry" in v}
    p = out_dir / "m3.json"
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("objects"), dict):
            return {k: v for k, v in raw["objects"].items()
                    if isinstance(v, dict) and "geometry" in v}
        merged: dict[str, dict] = {}
        for r in (raw.get("responses") if isinstance(raw, dict) else raw) or []:
            nid = r.get("node_id")
            if nid and isinstance(r.get("geometry"), dict):
                merged.setdefault(nid, {}).update(
                    {k: v for k, v in r.items()
                     if k not in ("subgoal_id", "queried_by", "node_id", "ee",
                                  "reachability", "type", "value", "pass")})
        return merged
    return {}


def _flat_face(geometry: dict, rms_tol_mm=2.0, min_face_mm=40.0, patch_tol_mm=1.5) -> dict:
    rms = geometry.get("surface_rms_mm", float("nan"))
    patch = geometry.get("seal_patch_rms_mm", 99.9)
    face = min(geometry.get("footprint_mm", [0.0, 0.0]))
    planar = (rms == rms) and rms <= rms_tol_mm and patch <= patch_tol_mm
    ok = bool(planar and face >= min_face_mm)
    return {"value": ok, "face_mm": round(float(face), 1),
            "check": f"rms_{rms}<={rms_tol_mm} & patch_{patch}<={patch_tol_mm} "
                     f"& face_{round(float(face), 1)}>={min_face_mm}"}


def ensure_measurements(m1: dict, ee_pool: list[dict] | None = None,
                        reach_mm: float | None = None,
                        out_dir: str | Path | None = None) -> list[str]:
    """접지값(ee/reachability/predicates)이 없는 노드를 채운다 — 통합 전 임시 경로.

    통합 뒤 m1.json에는 전부 실려 오므로 이 함수는 아무것도 하지 않는다.
    통합 전에는 남아 있는 m3_intrinsic.json/m3.json 물성표를 읽어 산술로 채운다
    (EE 판정·리치·단항 술어는 전부 계산 가능). 물성표에도 없는 노드는 접지값 없이
    남고 관련 술어는 unknown이 된다 — M2가 몰래 VLM을 돌지 않는다.
    통합이 머지되면 이 함수와 _legacy_props는 지워도 된다.
    """
    nodes = {n["id"]: n for n in m1["nodes"]}
    edges = m1.get("edges", [])
    props = _legacy_props(Path(out_dir)) if out_dir else {}
    logs, filled, missing = [], 0, []
    for nid, n in nodes.items():
        preds = n.setdefault("predicates", {})
        preds.setdefault("top_exposed", top_exposed(nid, edges))
        preds.setdefault("clear", region_clear(nid, edges))
        if "ee" in n and "flat_face" in preds:
            continue
        intr = {k: v for k, v in (props.get(nid) or {}).items() if not k.startswith("_")}
        if not intr.get("geometry"):
            if "ee" not in n:
                missing.append(nid)
            continue
        for k, v in intr.items():                 # 물성도 노드에 실어 둔다 (도구 측정표용)
            n.setdefault(k, v)
        if "ee" not in n and ee_pool:
            from tuj.m3_grounding.ee_conditioned import evaluate_ee, reach_check
            n["ee"] = {e["ee_id"]: evaluate_ee(e, intr) for e in ee_pool}
            n["reachability"] = reach_check(reach_mm, n["center_mm"])
        preds.setdefault("flat_face", flat_face(intr["geometry"]))
        filled += 1
    if filled:
        logs.append(f"  [접지 보완] m1에 접지값이 없어 물성표로 {filled}개 노드를 채움 "
                    f"(M1&M3 통합 머지 후에는 실행되지 않음)")
    if missing:
        logs.append(f"  [접지 없음] 물성 미측정 노드 {len(missing)}개 — 관련 술어는 unknown: "
                    f"{sorted(missing)[:8]}{' ...' if len(missing) > 8 else ''}")
    return logs
