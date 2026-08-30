"""M1 지각 프론트엔드: depth+seg 1프레임 → 객체별 점군 (베이스 프레임, mm).

robosuite GT segmentation 기준. VLM/LLM 0회.
xml_scene.py(지각 GT)와 무관 — 이쪽은 '관측', 그쪽은 '정답'.
"""
from __future__ import annotations

import numpy as np


def points_from_frame(depth_m, seg, K, T_cam2world, name_of_id,
                      base_offset_mm=(0.0, 0.0, 0.0), min_pixels=20):
    """→ [{name, cls, points(N,3 mm)}]. 마스크 1px 침식으로 실루엣 depth 오염 제거."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    seg = np.asarray(seg).squeeze()
    out = []
    for sid, (name, cls) in name_of_id.items():
        mask = (seg == sid)
        if mask.sum() < min_pixels:
            continue
        er = mask & np.roll(mask, 1, 0) & np.roll(mask, -1, 0) \
                  & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
        if er.sum() >= min_pixels:
            mask = er
        vs, us = np.nonzero(mask)
        z = np.asarray(depth_m)[vs, us].astype(np.float64)
        ok = z > 1e-4
        vs, us, z = vs[ok], us[ok], z[ok]
        pts_cam = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z, np.ones_like(z)], 1)
        pts = (T_cam2world @ pts_cam.T).T[:, :3] * 1000.0 - np.asarray(base_offset_mm)
        out.append({"name": name, "cls": cls, "points": pts})
    return out


def mad_filter(pts, k=5.0, floor_mm=25.0):
    """축별 |x−median| ≤ max(k·MAD, floor) — 잔여 outlier 제거."""
    pts = np.asarray(pts, dtype=np.float64)
    med = np.median(pts, axis=0)
    mad = np.median(np.abs(pts - med), axis=0)
    core = pts[np.all(np.abs(pts - med) <= np.maximum(k * mad, floor_mm), axis=1)]
    return core if len(core) >= 20 else pts
