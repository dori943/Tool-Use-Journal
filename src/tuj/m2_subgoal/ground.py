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


def gap_access(tool: dict, target: dict, gap_width_mm=None,
               insertion_depth_mm=None, tip_ratio=None) -> dict:
    # insertion_depth_mm / tip_ratio 는 리치 검사를 실제 진입 깊이로 정밀화하는
    # 확장 인자. _call 이 하부 함수 시그니처에 있는 키워드만 전달한다.
    return _call(_rel.gap_access, tool, target, gap_width_mm=gap_width_mm,
                 insertion_depth_mm=insertion_depth_mm, tip_ratio=tip_ratio)


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


def container_layout(members: list[dict], container: dict) -> dict | None:
    """컨테이너 내부 배치 (0909). 구버전 관계 모듈이면 None — 배치 배정을 건너뛴다."""
    fn = getattr(_rel, "container_layout", None)
    if fn is None:
        return None
    return _call(fn, members, container)


# 컨테이너 bbox 바닥에서 내부 바닥 윗면까지 (M5 REGION_FLOOR_FALLBACK_M 와 같은 값).
_CONTAINER_FLOOR_MM = 5.0
# 줄 순서를 전부 시도하는 상한. 이보다 많으면 원래 순서/역순만 본다.
_MAX_PERMUTED_ROWS = 6


def _sweep_check(layout: dict, members: list[dict], container: dict, keepout) -> dict:
    """배치의 각 자리에서 놓인 물체 윗면이 고정 EE 궤적의 팔과 margin 이상 떨어지는가.

    keepout 은 M1 좌표계(로봇 베이스 기준 xy, 월드 z, 단위 m)의 높이장이다.
    배치 offset/uv도 관측 bbox의 월드 XY 축 기준이며 컨테이너 몸체 축이 아니다.
    M5는 이 방향을 유지한 채 실제 region-local 경계 안으로 잘라 낸다.
    물체 발자국은 긴 변 기준 정사각형으로 본다 — M5 가 놓을 때 90도 돌릴 수 있다.
    """
    from tuj.common.ee_keepout import placement_margin_m

    margin = placement_margin_m()
    cz = float(container["center_mm"][2])
    floor_mm = cz - float(container["bbox_mm"][2]) / 2.0 + _CONTAINER_FLOOR_MM
    by_id = {m["id"]: m for m in members}
    per_slot, blocked = {}, []
    for oid, slot in layout["slots"].items():
        m = by_id[oid]
        cx = float(container["center_mm"][0]) + slot["offset_mm"][0]
        cy = float(container["center_mm"][1]) + slot["offset_mm"][1]
        r = max(float(m["bbox_mm"][0]), float(m["bbox_mm"][1])) / 2.0
        top = floor_mm + float(m["bbox_mm"][2])
        gap = keepout.clearance((cx / 1000.0, cy / 1000.0), (r / 1000.0, r / 1000.0),
                                top / 1000.0, margin)
        per_slot[oid] = round(gap * 1000.0, 1) if np.isfinite(gap) else None
        if gap < 0:
            blocked.append(oid)
    finite = [v for v in per_slot.values() if v is not None]
    return {"blocked": blocked, "clearance_mm": per_slot,
            "min_clearance_mm": min(finite) if finite else None}


def container_layout_clear_of_sweep(members: list[dict], container: dict,
                                    keepout) -> dict | None:
    """container_layout 중 고정 EE 교체 궤적 아래에 키 큰 물체를 두지 않는 배치.

    EE 교체는 저장된 관절 궤적을 그대로 재생한다. 그 아래 놓인 물체는 다음 교체에서야
    걸린다 (c3_2: 트레이 A 머그가 3F->vac 궤적 3mm 아래 — 30 스텝 뒤 M5 중단).
    줄 구성은 그대로 두고 줄 순서와 x 방향만 바꿔 보며, 원래 배치가 통과하면 그대로 쓴다.
    전부 걸리면 걸리는 물체가 가장 적은 배치를 고르고 blocked 로 보고한다.
    """
    from itertools import permutations

    base = container_layout(members, container)
    if keepout is None or not base or not base.get("slots"):
        return base
    n = int(base["rows"])
    orders = (permutations(range(n)) if n <= _MAX_PERMUTED_ROWS
              else (tuple(range(n)), tuple(reversed(range(n)))))
    best = None
    for order in orders:
        for mirror in (False, True):
            layout = (base if list(order) == list(range(n)) and not mirror
                      else _call(_rel.container_layout, members, container,
                                 row_order=list(order), mirror_x=mirror))
            check = _sweep_check(layout, members, container, keepout)
            key = (len(check["blocked"]), -(check["min_clearance_mm"] or 0.0))
            if best is None or key < best[0]:
                best = (key, layout, check)
            if not check["blocked"]:
                break
        if not best[2]["blocked"]:
            break
    _, layout, check = best
    rearranged = (layout.get("row_order", list(range(n))) != list(range(n))
                  or bool(layout.get("mirror_x")))
    return dict(layout) | {"ee_sweep": check | {"rearranged": rearranged}}


_EE_POOL_CACHE: list[dict] | None = None


def _ee_pool() -> list[dict]:
    """robot_spec 의 EE 사양. 접근 가능성 판정에 capabilities 가 필요하다."""
    global _EE_POOL_CACHE
    if _EE_POOL_CACHE is None:
        path = Path(os.environ.get(
            "TUJ_ROBOT_SPEC",
            Path(__file__).resolve().parents[3] / "configs" / "robot_spec.json"))
        try:
            _EE_POOL_CACHE = json.loads(path.read_text(encoding="utf-8"))["ee_pool"]
        except (OSError, KeyError, ValueError):
            _EE_POOL_CACHE = []
    return _EE_POOL_CACHE


def graspable_on_support(node: dict, edges: list[dict]) -> dict:
    """지지면에 놓인 상태에서 접근 가능한 EE 가 하나라도 있는가.

    ee_usable 이 물체 고유 사양만 보는 데 비해 이 술어는 물체가 놓인 자리를 본다.
    EE 별 판정은 관계 함수에 맡기고 여기서는 EE 풀을 돌며 모은다. 결과의
    graspable_ees 는 gk 조립에서 노드의 EE 후보를 좁히는 데 쓰인다.
    """
    fn = getattr(_rel, "graspable_on_support", None)
    per_ee: dict[str, dict] = {}
    for ee in _ee_pool():
        ee_id = ee.get("ee_id")
        if fn is not None:
            per_ee[ee_id] = _call(fn, node, ee, edges)
            continue
        supports = [e["to"] for e in edges
                    if e.get("from") == node.get("id")
                    and e.get("type") in ("on", "inside")]
        height_mm = float((node.get("bbox_mm") or [0.0, 0.0, 0.0])[2])
        ok = (not supports
              or "suction" in set(ee.get("capabilities") or ())
              or height_mm >= 10.0)
        per_ee[ee_id] = {"type": "graspable_on_support", "ee_id": ee_id,
                         "supports": supports, "height_mm": round(height_mm, 1),
                         "value": ok, "pass": ok}
    usable = [k for k, v in per_ee.items() if v.get("value")]
    ok = bool(usable) if per_ee else None
    return {"type": "graspable_on_support", "node_id": node.get("id"),
            "value": ok, "pass": ok, "graspable_ees": usable, "per_ee": per_ee}


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
    from tuj.common.ee_keepout import load_height_field
    m1["_ee_keepout"] = load_height_field(m1_json.parent / "m1_ee_keepout.npz")
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
