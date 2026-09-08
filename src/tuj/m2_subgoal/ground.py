# -*- coding: utf-8 -*-
"""M2-6 — 접지값 접근/보완 (0908 파이프라인 재구조).

새 방법론: [M1 & M3] → M2 → M4 → M5 한 방향. M2↔M3 질의 왕복이 없다.
접지값은 m1.json 노드에 실려서 온다:
    node["ee"]            {ee_id: {feasible, margin, reason}}
    node["reachability"]  {reachable, margin_mm}
    node["predicates"]    {top_exposed: {value, blockers}, clear: {...}, flat_face: {...}}
쌍/집합 술어(fits, gap_accessible, batch, swept_space)는 M2가 관계 함수를 import해
직접 계산한다 — 여기서 도는 VLM 호출은 없다.

관계 함수는 통합 산출물(tuj.m1_scene.relations)에서 가져오며, 구버전 레포를 위해
m3_grounding.relational 폴백을 남겨 둔다. 함수 이름이나 인자가 달라도 _call이
시그니처에 있는 키워드만 넘겨 흡수한다.
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
