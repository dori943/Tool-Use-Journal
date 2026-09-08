"""M3-(b) relational 접지: 객체 쌍 수치 (M1의 coarse 판정을 정밀 수치로 승격).

전부 bbox/점군 산술 — VLM 0회. 결과는 {value, check(계산식), pass} 형태.
"""
from __future__ import annotations

import numpy as np


def center_distance(node_a: dict, node_b: dict) -> dict:
    d = float(np.linalg.norm(np.asarray(node_a["center_mm"]) - np.asarray(node_b["center_mm"])))
    return {"type": "distance", "value_mm": round(d, 1), "check": "center_to_center", "pass": True}


def fits_inside(target: dict, container: dict, wall_mm: float = 4.0) -> dict:
    """개구(외곽 bbox − 벽두께) − 대상 footprint."""
    open_w = container["bbox_mm"][0] - 2 * wall_mm
    open_d = container["bbox_mm"][1] - 2 * wall_mm
    foot = min(target["bbox_mm"][0], target["bbox_mm"][1])
    v = min(open_w, open_d) - foot
    return {"type": "fits_inside", "value_mm": round(v, 1),
            "check": f"opening_{round(min(open_w, open_d),1)} - footprint_{round(foot,1)}",
            "pass": bool(v > 0)}


def depth_clearance(target: dict, container: dict, wall_mm: float = 4.0,
                    tip_ratio: float = 3.0) -> dict:
    """컨테이너가 대상을 담아 둘 수 있는가 — '깊이>높이'가 아니라 바닥 면적 수용 +
    전도 없음으로 판정(0902). 얕은 트레이여도 물체가 바닥에 앉고 넘어지지 않으면 담기 OK.
      · 바닥 수용: 대상 footprint(최소변) ≤ 컨테이너 개구(외곽 − 벽두께)
      · 전도 없음: 대상 높이 ≤ tip_ratio × 바닥 최소변 (가늘고 높은 물체만 탈락)
    value_mm = 두 여유 중 빡빡한 쪽. (종전: 깊이 25mm 트레이에서 사과/빵/머그 전부 unsat)"""
    open_w = container["bbox_mm"][0] - 2 * wall_mm
    open_d = container["bbox_mm"][1] - 2 * wall_mm
    base = min(target["bbox_mm"][0], target["bbox_mm"][1])
    h = target["bbox_mm"][2]
    floor_margin = min(open_w, open_d) - base
    tip_margin = tip_ratio * base - h
    v = min(floor_margin, tip_margin)
    return {"type": "clearance", "value_mm": round(float(v), 1),
            "check": (f"floor_fit(open_{round(min(open_w, open_d), 1)}-base_{round(base, 1)})"
                      f" & no_tip(h_{round(h, 1)}<={tip_ratio}x{round(base, 1)})"),
            "pass": bool(floor_margin > 0 and tip_margin > 0)}


def gap(node_a: dict, node_b: dict, max_pts: int = 1500) -> dict:
    """두 객체 사이 최소 수평 간격 (정밀판 — '소파 간격' 류 Type-B 술어).

    bbox가 xy에서 이미 겹치면 겹침량(음수, bbox 산술). 아니면 점군 최근접
    xy 거리(있으면) 또는 bbox edge-to-edge 간격. M1 coarse near(gap_mm)의 승격."""
    bbox_gap = max(
        abs(node_a["center_mm"][k] - node_b["center_mm"][k])
        - (node_a["bbox_mm"][k] + node_b["bbox_mm"][k]) / 2
        for k in range(2))
    if bbox_gap <= 0:                                  # xy 겹침 → 음수 간격
        return {"type": "gap", "value_mm": round(float(bbox_gap), 1),
                "check": "bbox_overlap", "pass": False}
    a_pts, b_pts = node_a.get("_points"), node_b.get("_points")
    if a_pts is not None and b_pts is not None:
        rng = np.random.default_rng(0)
        A = np.asarray(a_pts)[:, :2]
        B = np.asarray(b_pts)[:, :2]
        if len(A) > max_pts:
            A = A[rng.choice(len(A), max_pts, replace=False)]
        if len(B) > max_pts:
            B = B[rng.choice(len(B), max_pts, replace=False)]
        try:
            from scipy.spatial import cKDTree
            v = float(cKDTree(A).query(B)[0].min())
        except ImportError:
            v = float(np.sqrt(((A[:, None, :] - B[None, :500, :]) ** 2).sum(-1)).min())
        check = "pointcloud_min_xy"
    else:
        v, check = float(bbox_gap), "bbox_edge_gap"
    return {"type": "gap", "value_mm": round(v, 1), "check": check, "pass": bool(v > 0)}


def opening_pass(passer: dict, opening_height_mm: float, pass_height_mm: float) -> dict:
    """개구 통과: 개구 높이 − (손목+EE 통과높이) — 선반류 시나리오용."""
    v = opening_height_mm - pass_height_mm
    return {"type": "opening_pass", "value_mm": round(v, 1),
            "check": f"opening_{opening_height_mm} - pass_height_{pass_height_mm}",
            "pass": bool(v > 0)}


def gap_access(tool: dict, target: dict, gap_width_mm: float | None = None,
               tip_ratio: float = None) -> dict:
    """gap_accessible(?tool, ?target) 접지(0901 어휘 확장): 도구가 틈에 진입해
    대상에 닿을 수 있는가. 3조건을 bbox 산술로 — 전부 도구/대상 치수 비교, VLM 0회.
      · thickness_lt_gap : 도구 최소 두께 < 틈 폭
      · contact_le_target: 도구 접촉폭(중간 치수) ≤ 대상 최대 폭 (물거나 긁을 수 있음)
      · reach_ge_depth   : 도구 최대 길이 ≥ 진입 깊이(대상 높이 근사)
    gap_width_mm 미지정 시 대상 최소변으로 근사(틈에 낀 대상 자신의 폭)."""
    t = sorted(tool["bbox_mm"])
    tool_thick, tool_contact, tool_len = t[0], t[1], t[2]
    gap_w = gap_width_mm if gap_width_mm is not None else min(target["bbox_mm"])
    target_w = max(target["bbox_mm"][0], target["bbox_mm"][1])
    depth = target["bbox_mm"][2]
    checks = [
        {"rule": "thickness_lt_gap", "value_mm": round(tool_thick, 1),
         "limit_mm": round(gap_w, 1), "pass": bool(tool_thick < gap_w)},
        {"rule": "contact_le_target", "value_mm": round(tool_contact, 1),
         "limit_mm": round(target_w, 1), "pass": bool(tool_contact <= target_w)},
        {"rule": "reach_ge_depth", "value_mm": round(tool_len, 1),
         "limit_mm": round(depth, 1), "pass": bool(tool_len >= depth)},
    ]
    margin = min(gap_w - tool_thick, target_w - tool_contact, tool_len - depth)
    return {"type": "gap_accessible", "value_mm": round(float(margin), 1),
            "check": "thickness<gap & contact<=target & reach>=depth",
            "checks": checks, "pass": bool(all(c["pass"] for c in checks))}


# ── 0828 신규 (프로토타입, 수빈 작성 — push 전 협의) ────────────────────────
# batch / swept_space 질의용 집합 산술. 기존 함수들과 같은 bbox 산술, VLM 0회.
def group_extent(nodes: list[dict]) -> dict:
    """물체 집합이 xy 평면에서 차지하는 최대 폭 (bbox 경계 기준) — batch 질의의 demand."""
    spans = []
    for ax in (0, 1):
        lo = min(n["center_mm"][ax] - n["bbox_mm"][ax] / 2 for n in nodes)
        hi = max(n["center_mm"][ax] + n["bbox_mm"][ax] / 2 for n in nodes)
        spans.append(hi - lo)
    v = max(spans)
    return {"type": "group_extent", "value_mm": round(v, 1),
            "check": f"max_xy_span_of_{len(nodes)}_nodes", "pass": True}


def corridor_blockers(members: list[dict], to_node: dict, width_mm: float,
                      others: list[dict]) -> dict:
    """시작(집합 중심)→목적지 직선 통로(폭 width_mm)와 겹치는 노드 — swept_space 질의용.
    정밀 궤적이 아니라 bbox 산술 근사다. 실제로 어느 수준까지 계산할지는 협의 항목.
    """
    s = np.mean([np.asarray(n["center_mm"][:2], float) for n in members], axis=0)
    e = np.asarray(to_node["center_mm"][:2], float)
    d = e - s
    length = float(np.linalg.norm(d)) or 1.0
    u = d / length
    blockers = []
    for n in others:
        p = np.asarray(n["center_mm"][:2], float) - s
        along = float(p @ u)
        if not 0.0 <= along <= length:
            continue
        perp = abs(float(-p[0] * u[1] + p[1] * u[0]))
        r = max(n["bbox_mm"][0], n["bbox_mm"][1]) / 2
        overlap = (width_mm / 2 + r) - perp
        if overlap > 0:
            blockers.append({"node_id": n["id"], "overlap_mm": round(overlap, 1)})
    margin = round(-max((b["overlap_mm"] for b in blockers), default=-width_mm / 2), 1)
    return {"type": "swept_space", "clear": not blockers,
            "margin_mm": margin, "blockers": blockers}
