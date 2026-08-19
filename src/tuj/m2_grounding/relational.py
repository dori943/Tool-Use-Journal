"""M2-(b) relational 접지: 객체 쌍 수치 (M0의 coarse 판정을 정밀 수치로 승격).

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


def depth_clearance(target: dict, container: dict) -> dict:
    """컨테이너 깊이 − 대상 높이 (음수 = 세워 담으면 돌출)."""
    v = container["bbox_mm"][2] - target["bbox_mm"][2]
    return {"type": "clearance", "value_mm": round(v, 1),
            "check": f"container_depth_{container['bbox_mm'][2]} - target_height_{target['bbox_mm'][2]}",
            "pass": bool(v > 0)}


def opening_pass(passer: dict, opening_height_mm: float, pass_height_mm: float) -> dict:
    """개구 통과: 개구 높이 − (손목+EE 통과높이) — 선반류 시나리오용."""
    v = opening_height_mm - pass_height_mm
    return {"type": "opening_pass", "value_mm": round(v, 1),
            "check": f"opening_{opening_height_mm} - pass_height_{pass_height_mm}",
            "pass": bool(v > 0)}
