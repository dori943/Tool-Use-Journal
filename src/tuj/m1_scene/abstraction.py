"""M1 — Scene Abstraction: bbox 노드 + coarse 관계. 태스크 무관, 에피소드당 1회.

출력 노드: {id, class, center_mm, bbox_mm} (+ 내부용 _points — M3 질의에서 사용)
출력 엣지: on / inside / overlaps / near (bbox 산술만)
common/schemas.py 의 scene 스키마와 정합 필요 시 serialize() 결과를 검증에 통과시킬 것.
"""
from __future__ import annotations

import numpy as np

from .perception import mad_filter

NEAR_MM = 150.0
ON_TOL_MM = 25.0
INSIDE_Z_TOL_MM = 30.0


def build_m1(objects) -> dict:
    """objects: [{name, cls, points}] → {"nodes": [...], "edges": [...]}"""
    nodes = []
    for o in objects:
        pts = mad_filter(o["points"])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        nodes.append({
            "id": f"obj_{o['cls']}_{o['name']}",
            "class": o["cls"],
            "center_mm": [round(float(c), 1) for c in (lo + hi) / 2],
            "bbox_mm": [round(float(s), 1) for s in hi - lo],
            "_points": pts,
        })
    return {"nodes": nodes, "edges": coarse_relations(nodes)}


def coarse_relations(nodes) -> list:
    edges, related = [], set()
    for a in nodes:
        for b in nodes:
            if a["id"] == b["id"]:
                continue
            if _inside(a, b):
                edges.append({"from": a["id"], "to": b["id"], "type": "inside"})
                related.add(frozenset((a["id"], b["id"])))
            elif _xy_overlap(a, b) > 0.4 and _on(a, b):
                edges.append({"from": a["id"], "to": b["id"], "type": "on"})
                related.add(frozenset((a["id"], b["id"])))
    for a in nodes:
        for b in nodes:
            if not (a["id"] < b["id"]) or frozenset((a["id"], b["id"])) in related:
                continue
            if _xy_overlap(a, b) > 0.15 and _z_overlap(a, b) > 0:
                edges.append({"from": a["id"], "to": b["id"], "type": "overlaps"})
            elif _gap_xy(a, b) < NEAR_MM:
                edges.append({"from": a["id"], "to": b["id"], "type": "near",
                              "gap_mm": round(_gap_xy(a, b), 1)})
    return edges


def coarse_clearance(passer_bbox_mm, opening_mm) -> dict:
    """개구 − 통과체 bbox 최소 수평치수 (M2 서브골 성립 판단용, ±수십mm 근사)."""
    v = min(opening_mm) - min(passer_bbox_mm[0], passer_bbox_mm[1])
    return {"clearance_mm": round(float(v), 1), "pass": bool(v > 0)}


def serialize(m1: dict) -> dict:
    """점군 제외 JSON-직렬화 형태 (M2 전달용 · schemas.py 검증 대상)."""
    return {"nodes": [{k: v for k, v in n.items() if k != "_points"} for n in m1["nodes"]],
            "edges": m1["edges"]}


# ── bbox 산술 ──
def _iv(n, k):
    return n["center_mm"][k] - n["bbox_mm"][k] / 2, n["center_mm"][k] + n["bbox_mm"][k] / 2

def _xy_overlap(a, b):
    ov = 1.0
    for k in range(2):
        lo = max(_iv(a, k)[0], _iv(b, k)[0]); hi = min(_iv(a, k)[1], _iv(b, k)[1])
        if hi <= lo:
            return 0.0
        ov *= (hi - lo) / max(min(a["bbox_mm"][k], b["bbox_mm"][k]), 1e-6)
    return ov

def _on(a, b):
    """a가 b 위에 놓임: a 바닥 ≈ b 상단, 그리고 a 중심이 b 중심보다 위 (얇은 객체 양방향 오발화 방지).

    위에서 내려다본 점군은 a의 아랫면·하단 측면이 가려져 관측 바닥이 실제 바닥보다
    위로 뜬다 — 볼록한 덩어리일수록 심하고, 최대 a 자체 높이만큼이다. (c1_2 반죽:
    관측 간격 26.8mm > 허용 25mm 로 탈락 → 도마 위 반죽이 near 로만 나와 M2가
    불필요한 재배치 서브골을 만들었음.) 그래서 아래쪽은 센서 노이즈(ON_TOL_MM)만,
    위쪽은 노이즈 + a 높이까지 허용한다. xy 겹침 조건은 호출부에서 별도로 건다."""
    gap = _iv(a, 2)[0] - _iv(b, 2)[1]              # a 바닥 − b 상단 (양수 = 떠 보임)
    return (-ON_TOL_MM <= gap <= ON_TOL_MM + a["bbox_mm"][2]
            and a["center_mm"][2] > b["center_mm"][2])

def _inside(a, b):
    xy_in = all(_iv(b, k)[0] <= _iv(a, k)[0] and _iv(a, k)[1] <= _iv(b, k)[1] for k in range(2))
    if not xy_in or a["bbox_mm"][0] * a["bbox_mm"][1] >= b["bbox_mm"][0] * b["bbox_mm"][1]:
        return False
    return _iv(b, 2)[0] - INSIDE_Z_TOL_MM <= _iv(a, 2)[0] <= _iv(b, 2)[1]

def _z_overlap(a, b):
    return min(_iv(a, 2)[1], _iv(b, 2)[1]) - max(_iv(a, 2)[0], _iv(b, 2)[0])

def _gap_xy(a, b):
    g = 0.0
    for k in range(2):
        d = abs(a["center_mm"][k] - b["center_mm"][k]) - (a["bbox_mm"][k] + b["bbox_mm"][k]) / 2
        g = max(g, d)
    return g
