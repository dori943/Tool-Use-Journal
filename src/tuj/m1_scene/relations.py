"""관계·집합 술어 라이브러리 (구 M3-(b) relational + materialize의 집합 질의).

전부 bbox/점군 산술 — VLM 0회. 결과는 {value, check(계산식), pass} 형태.

M1·M3 통합 후 이 모듈의 역할: M2가 서브골을 세우는 시점에 직접 import해서
객체 쌍·집합 술어를 판정한다 (질의층 Materializer/run_m3 는 제거됨).
  · 쌍:   fits_inside / depth_clearance(바닥수용+전도없음) / gap / center_distance /
          gap_access(gap_accessible)
  · 단항: top_exposed / region_clear (m1.json edges 기반) / flat_face (기하 기반)
  · 집합: batch_partition(batch_feasible) / swept_space(act_space_clear)
노드 인자는 m1.json 의 노드 dict (id, center_mm, bbox_mm, geometry ...) 그대로.
"""
from __future__ import annotations

import numpy as np


def center_distance(node_a: dict, node_b: dict) -> dict:
    d = float(np.linalg.norm(np.asarray(node_a["center_mm"]) - np.asarray(node_b["center_mm"])))
    return {"type": "distance", "value_mm": round(d, 1), "check": "center_to_center", "pass": True}


def _container_opening_mm(container: dict, wall_mm: float | None = None) -> tuple[float, float, str]:
    """컨테이너 개구 (open_w, open_d, source_tag).

    우선순위:
      1) container["inner_bbox_mm"] — env 가 아는 실제 내부치수 (c4_2 packing_box 등).
         M1 이 노드에 실었으면 그대로 사용해 외곽·벽 두께 추정 오차를 없앤다.
      2) container["bbox_mm"] - 2 * wall — wall 은 명시 인자 > container["wall_mm"] > 4.0.
         점군 AABB(외곽) 에서 벽 두께 두 배를 빼 근사한다.
    """
    inner = container.get("inner_bbox_mm")
    if inner:
        return float(inner[0]), float(inner[1]), "inner_bbox"
    w = wall_mm if wall_mm is not None else float(container.get("wall_mm") or 4.0)
    return (container["bbox_mm"][0] - 2 * w,
            container["bbox_mm"][1] - 2 * w,
            f"bbox-2*wall({w:.1f})")


def fits_inside(target: dict, container: dict, wall_mm: float | None = None) -> dict:
    """개구(내부치수 우선, 없으면 외곽 bbox − 벽두께) − 대상 footprint."""
    open_w, open_d, src = _container_opening_mm(container, wall_mm)
    foot = min(target["bbox_mm"][0], target["bbox_mm"][1])
    v = min(open_w, open_d) - foot
    return {"type": "fits_inside", "value_mm": round(v, 1),
            "check": f"opening_{round(min(open_w, open_d),1)}[{src}] - footprint_{round(foot,1)}",
            "pass": bool(v > 0)}


def depth_clearance(target: dict, container: dict, wall_mm: float | None = None,
                    tip_ratio: float = 3.0) -> dict:
    """컨테이너가 대상을 담아 둘 수 있는가 — '깊이>높이'가 아니라 바닥 면적 수용 +
    전도 없음으로 판정(0902). 얕은 트레이여도 물체가 바닥에 앉고 넘어지지 않으면 담기 OK.
      · 바닥 수용: 대상 footprint(최소변) ≤ 컨테이너 개구(내부치수 우선, 없으면 외곽 − 벽두께)
      · 전도 없음: 대상 높이 ≤ tip_ratio × 바닥 최소변 (가늘고 높은 물체만 탈락)
    value_mm = 두 여유 중 빡빡한 쪽. (종전: 깊이 25mm 트레이에서 사과/빵/머그 전부 unsat)"""
    open_w, open_d, src = _container_opening_mm(container, wall_mm)
    base = min(target["bbox_mm"][0], target["bbox_mm"][1])
    h = target["bbox_mm"][2]
    floor_margin = min(open_w, open_d) - base
    tip_margin = tip_ratio * base - h
    # 0909: 벽이 무게중심보다 높으면 기울어도 벽에 기대므로 전도 검사를 면제한다.
    # 종전 규칙은 컨테이너 높이를 보지 않아, 깊은 상자에 세워 넣는 우유갑(41.5 바닥에
    # 137.6 높이)이 얕은 트레이와 똑같이 탈락했다 (c4_2).
    walled = container["bbox_mm"][2] >= h / 2.0
    ok = bool(floor_margin > 0 and (tip_margin > 0 or walled))
    v = floor_margin if walled else min(floor_margin, tip_margin)
    return {"type": "clearance", "value_mm": round(float(v), 1),
            "check": (f"floor_fit(open_{round(min(open_w, open_d), 1)}[{src}]"
                      f"-base_{round(base, 1)})"
                      + (f" & walled(container_{round(container['bbox_mm'][2], 1)}"
                         f">=h/2_{round(h / 2.0, 1)})" if walled else
                         f" & no_tip(h_{round(h, 1)}<={tip_ratio}x{round(base, 1)})")),
            "pass": ok}


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


# ── 단항 술어 (m1.json edges / geometry 기반, VLM 0회) ──────────────────────
def top_exposed(node_id: str, edges: list[dict]) -> dict:
    """상면 노출 여부: 다른 객체가 on/inside 로 위를 점유하면 False."""
    blockers = [e["from"] for e in edges
                if e.get("to") == node_id and e.get("type") in ("on", "inside")]
    return {"type": "top_exposed", "value": not blockers, "blockers": blockers,
            "pass": not blockers}


def region_clear(region_id: str, edges: list[dict]) -> dict:
    """영역 비움 여부: on/inside/overlaps 로 영역을 점유한 객체가 없으면 True."""
    occupants = sorted({e["from"] for e in edges
                        if e.get("to") == region_id
                        and e.get("type") in ("on", "inside", "overlaps")}
                       | {e["to"] for e in edges
                          if e.get("from") == region_id and e.get("type") == "overlaps"})
    return {"type": "clear", "value": not occupants, "occupants": occupants,
            "pass": not occupants}


def flat_face(geometry: dict, rms_tol_mm: float = 2.0, min_face_mm: float = 40.0,
              patch_tol_mm: float = 1.5) -> dict:
    """flat_face(?tool)(0901): 도구에 넓고 평평한 작업면이 있는가 — flatten용.
    상면 평면성(rms) + 접촉 패치 평면성(seal_patch) + 작업면 폭(footprint)."""
    rms = geometry.get("surface_rms_mm", float("nan"))
    patch = geometry.get("seal_patch_rms_mm", 99.9)
    face = min(geometry.get("footprint_mm", [0.0, 0.0]))
    planar = (rms == rms) and rms <= rms_tol_mm and patch <= patch_tol_mm
    ok = bool(planar and face >= min_face_mm)
    return {"type": "flat_face", "value": ok, "pass": ok, "face_mm": round(face, 1),
            "check": (f"rms_{rms}<={rms_tol_mm} & patch_{patch}<={patch_tol_mm}"
                      f" & face_{round(face, 1)}>={min_face_mm}")}


def graspable_on_support(node: dict, ee: dict, edges: list[dict],
                         min_grasp_height_mm: float = 10.0) -> dict:
    """graspable_on_support(?EE, ?o): 지지면에 놓인 상태로 이 EE 가 접근 가능한가.

    ee_usable 은 물체를 고립시켜 치수와 하중만 EE 사양과 대조한다. 그래서 지지면에
    붙어 있어 손가락이 들어갈 자리가 없는 물체도 통과한다 (두께 1.6mm 치즈가 접시
    위에서 2F 가능으로 판정된 사례). 이 술어는 그 누락분만 본다.

    손가락형은 물체를 옆에서 감싸야 하므로 지지면 위 높이가 필요하고, 흡착은 위에서
    닿으므로 높이와 무관하다. 판정 기준 높이는 인자로 받는다 (현재는 모션 계획의
    충돌 여유에서 유도한 값, EE 사양에 최소 파지 높이가 들어오면 그 값으로 교체).

    지지면은 on/inside 간선으로 찾는다. 노드인 지지면이 없으면(작업대 위 등) 판정
    대상이 아니므로 통과시킨다.
    """
    node_id = node.get("id")
    supports = [e["to"] for e in edges
                if e.get("from") == node_id and e.get("type") in ("on", "inside")]
    caps = set(ee.get("capabilities") or ())
    height_mm = float((node.get("bbox_mm") or [0.0, 0.0, 0.0])[2])
    base = {"type": "graspable_on_support", "ee_id": ee.get("ee_id"),
            "supports": supports, "height_mm": round(height_mm, 1)}

    if not supports:
        return base | {"value": True, "pass": True,
                       "check": "no_support_node"}
    if "suction" in caps:
        return base | {"value": True, "pass": True,
                       "check": "suction_approaches_from_above"}
    ok = height_mm >= min_grasp_height_mm
    return base | {"value": ok, "pass": ok,
                   "check": f"height_{round(height_mm, 1)}>={min_grasp_height_mm}"}


def container_layout(members: list[dict], container: dict,
                     gap_mm: float = 10.0, wall_mm: float | None = None) -> dict:
    """컨테이너 내부를 나눠 멤버마다 서로 다른 목표 자리를 준다 (0909).

    같은 컨테이너로 가는 물체들이 저마다 "그 영역에 놓아라"만 받으면, 놓는 쪽은
    매번 같은 자리(영역 중심)를 고르고 먼저 놓인 것과 겹친다 (c3_1: 트레이 중앙의
    접시 위로 머그가 내려와 관통). 어디에 무엇을 놓을지는 계획이 아는 사실이므로
    여기서 배치를 잡아 서브골마다 다른 자리를 실어 준다.

    선반(shelf) 채우기: footprint 깊이 내림차순으로 한 줄씩 x를 채우고, 폭이 모자라면
    다음 줄로 내린다. 한 층에 다 안 들어가면 남는 것은 overflow 로 보고하되(계획은
    적층을 택할 수 있다) 자리는 계속 배정한다 — 겹치더라도 서로 다른 자리가
    영역 중심 한 점보다 낫다.

    반환 slots[id] = {"uv": [u, v], "offset_mm": [dx, dy], "row": r}
      · uv 는 내부 반치수 기준 정규화 좌표(-1..1). M1 점군 mm 와 실행계 m 의
        절대 치수가 어긋나도(같은 물체를 다른 시야로 잰다) 비율은 옮겨 간다.
    """
    open_w, open_d, src_tag = _container_opening_mm(container, wall_mm)
    half = (open_w / 2.0, open_d / 2.0)
    order = sorted(members, key=lambda m: -float(m["bbox_mm"][1]))
    rows: list[dict] = []
    for m in order:
        w, d = float(m["bbox_mm"][0]), float(m["bbox_mm"][1])
        row = rows[-1] if rows else None
        if row is not None and row["width"] + gap_mm + w <= open_w:
            row["items"].append((m["id"], w, d))
            row["width"] += gap_mm + w
            row["depth"] = max(row["depth"], d)
        else:
            rows.append({"items": [(m["id"], w, d)], "width": w, "depth": d})
    used_d = sum(r["depth"] for r in rows) + gap_mm * max(0, len(rows) - 1)
    overflow = [m["id"] for m in order
                if float(m["bbox_mm"][0]) > open_w or float(m["bbox_mm"][1]) > open_d]
    if used_d > open_d:                       # 한 층에 다 못 들어감 — 줄 간격을 눌러 담는다
        squeeze = (open_d - sum(r["depth"] for r in rows)) / max(1, len(rows) - 1) \
            if len(rows) > 1 else 0.0
        step_gap = max(0.0, min(gap_mm, squeeze))
        overflow += [i for r in rows[1:] for i, _, _ in r["items"] if i not in overflow]
    else:
        step_gap = gap_mm
    total_d = sum(r["depth"] for r in rows) + step_gap * max(0, len(rows) - 1)
    slots: dict[str, dict] = {}
    y = -min(total_d, open_d) / 2.0
    for r_index, row in enumerate(rows):
        cy = y + row["depth"] / 2.0
        x = -row["width"] / 2.0
        for oid, w, _d in row["items"]:
            cx = x + w / 2.0
            slots[oid] = {
                "uv": [round(max(-1.0, min(1.0, cx / half[0])), 4),
                       round(max(-1.0, min(1.0, cy / half[1])), 4)],
                "offset_mm": [round(cx, 1), round(cy, 1)], "row": r_index}
            x += w + gap_mm
        y += row["depth"] + step_gap
    ok = not overflow
    return {"type": "container_layout", "container_id": container.get("id"),
            "slots": slots, "rows": len(rows), "overflow": overflow,
            "value_mm": round(open_d - total_d, 1), "pass": bool(ok),
            "check": (f"shelf_fit(open_{round(open_w, 1)}x{round(open_d, 1)}[{src_tag}]"
                      f", rows_{len(rows)}, used_d_{round(total_d, 1)}, gap_{gap_mm})")}


# ── 집합 술어 (구 materialize.query_batch / query_swept_space) ────────────────
def _greedy_partition(members: list[dict], cap_mm: float) -> list[list[str]]:
    """주축 정렬 그리디: 폭이 cap 안에 들어오는 만큼씩 근접한 것끼리 묶는다."""
    span = lambda ax: (max(m["center_mm"][ax] for m in members)
                       - min(m["center_mm"][ax] for m in members))
    ax = 0 if span(0) >= span(1) else 1
    order = sorted(members, key=lambda m: m["center_mm"][ax])
    groups, cur = [], []
    for m in order:
        trial = cur + [m]
        if cur and group_extent(trial)["value_mm"] > cap_mm:
            groups.append(cur)
            cur = [m]
        else:
            cur = trial
    if cur:
        groups.append(cur)
    return [[m["id"] for m in g] for g in groups]


def batch_partition(members: list[dict], tool: dict | None = None,
                    safety: float = 1.0) -> dict:
    """batch_feasible: 한 번의 액션으로 member 집합을 동시 처리할 수 있는가.
    안 되면 근접도/폭 기준 파티션(그룹 구성)까지 계산해 돌려준다.
      tool=None  → EE 직접 파지(relocate): 그리퍼는 1물체=1액션 → 물체별 1그룹.
      tool=노드  → 도구 폭(min bbox xy × safety) 안에 들어오는 만큼 묶는다.
    → {feasible, checks, binding_check, partition}"""
    if not members:
        return {"feasible": None, "checks": [], "binding_check": None, "partition": None}
    if tool is None:
        part = [[m["id"]] for m in members]
        return {"feasible": len(members) <= 1,
                "checks": [{"rule": "ee_pool_one_per_grasp", "capacity": 1,
                            "demand": len(members), "unit": "count",
                            "margin": 1 - len(members), "pass": bool(len(members) <= 1)}],
                "binding_check": "ee_pool_one_per_grasp", "partition": part}
    if len(members) < 2:
        return {"feasible": True, "checks": [], "binding_check": None,
                "partition": [[m["id"] for m in members]]}
    cap = min(tool["bbox_mm"][0], tool["bbox_mm"][1]) * safety
    demand = group_extent(members)["value_mm"]
    margin = round(cap - demand, 1)
    part = _greedy_partition(members, cap)
    return {"feasible": len(part) == 1,
            "checks": [{"rule": "group_extent_le_tool_width", "capacity": round(cap, 1),
                        "demand": demand, "unit": "mm", "margin": margin,
                        "pass": bool(margin > 0)}],
            "binding_check": "group_extent_le_tool_width", "partition": part}


def swept_space(members: list[dict], to_node: dict, others: list[dict],
                tool: dict | None = None, width_mm: float | None = None) -> dict:
    """act_space_clear: 액션이 지나가는 통로(집합 중심→목적지, 폭 width)가 비어 있는가.
      tool=노드 → 통로 폭 = 도구 min bbox xy (도구가 집합을 한 번에 밀고 감).
      tool=None(EE 직접 파지) → 물체를 하나씩 나르므로 통로 폭 = 가장 넓은 물체 하나의
      min bbox xy (집합 전체 폭으로 잡으면 지나치게 비관적). width_mm 를 주면 그 값 우선.
      others: 방해물 후보 (members·tool·to 는 호출부가 제외).
    → {clear, margin_mm, blockers}"""
    if not members or to_node is None:
        return {"clear": None, "margin_mm": None, "blockers": []}
    if width_mm is None:
        width_mm = (min(tool["bbox_mm"][0], tool["bbox_mm"][1]) if tool is not None
                    else max(min(m["bbox_mm"][0], m["bbox_mm"][1]) for m in members))
    r = corridor_blockers(members, to_node, width_mm, others)
    return {"clear": r["clear"], "margin_mm": r["margin_mm"], "blockers": r["blockers"]}
