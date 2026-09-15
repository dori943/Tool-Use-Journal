"""RealSense RGB-D support-plane segmentation for the existing M1/SiPhy path.

No object pose, mass, friction, or class annotation is used for segmentation.
The BOP per-frame camera calibration is used only to register metric depth.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_frame(scene: Path, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stem = f"{index:06d}.png"
    bgr = cv2.imread(str(scene / "rgb" / stem), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(scene / "depth" / stem), cv2.IMREAD_UNCHANGED)
    camera = json.loads((scene / "scene_camera.json").read_text(encoding="utf-8"))[str(index)]
    if bgr is None or depth_mm is None or bgr.shape[:2] != depth_mm.shape or depth_mm.dtype != np.uint16:
        raise ValueError(f"invalid RGB-D pair: {scene}/{stem}")
    K = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)
    R_w2c = np.asarray(camera["cam_R_w2c"], dtype=np.float64).reshape(3, 3)
    t_w2c = np.asarray(camera["cam_t_w2c"], dtype=np.float64) / 1000.0
    T_c2w = np.eye(4)
    T_c2w[:3, :3] = R_w2c.T
    T_c2w[:3, 3] = -R_w2c.T @ t_w2c
    return bgr, depth_mm, K, T_c2w


def _rays(K: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    v, u = np.indices((h, w))
    return np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)), axis=-1)


def fit_support_plane(depth_mm: np.ndarray, K: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """RANSAC dominant lower-image plane in camera coordinates (meters)."""
    h, w = depth_mm.shape
    rays = _rays(K, (h, w))
    valid = (depth_mm > 200) & (depth_mm < 3000)
    valid[: int(0.38 * h)] = False
    yx = np.argwhere(valid)
    if len(yx) < 1000:
        raise ValueError("insufficient RGB-D points for support plane")
    rng = np.random.default_rng(seed)
    yx = yx[rng.choice(len(yx), min(6000, len(yx)), replace=False)]
    pts = rays[yx[:, 0], yx[:, 1]] * (depth_mm[yx[:, 0], yx[:, 1], None] / 1000.0)
    best_count, best_plane = 0, None
    for _ in range(350):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        offset = -float(normal @ a)
        residual = np.abs(pts @ normal + offset)
        count = int(np.count_nonzero(residual < 0.008))
        if count > best_count:
            best_count, best_plane = count, (normal, offset)
    if best_plane is None or best_count < 1000:
        raise ValueError("support plane RANSAC failed")
    normal, offset = best_plane
    inliers = np.abs(pts @ normal + offset) < 0.008
    inlier_pts = pts[inliers]
    refined = np.linalg.svd(inlier_pts - inlier_pts.mean(axis=0), full_matrices=False)[2][-1]
    if refined @ normal < 0:
        refined = -refined
    normal = refined
    offset = -float(normal @ inlier_pts.mean(axis=0))
    inlier_pixels = yx[inliers][:, ::-1].astype(np.int32)
    hull_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, cv2.convexHull(inlier_pixels), 255)
    # The object rises above the rear table edge in the image.
    extent = max(3, int(0.40 * h) | 1)
    expanded = cv2.dilate(hull_mask, np.ones((extent, extent), np.uint8))
    return normal, offset, expanded, hull_mask


def segment_object(depth_mm: np.ndarray, K: np.ndarray,
                   plane: tuple[np.ndarray, float, np.ndarray, np.ndarray]) -> np.ndarray:
    normal, offset, table_footprint, table_core = plane
    rays = _rays(K, depth_mm.shape)
    denom = rays @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        table_depth_mm = -1000.0 * offset / denom
    above_mm = table_depth_mm - depth_mm.astype(np.float64)
    candidate = ((depth_mm > 0) & (table_footprint > 0)
                 & (above_mm > 12) & (above_mm < 400)
                 & np.isfinite(table_depth_mm)).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    components = [(int(np.count_nonzero((labels == i) & (table_core > 0))),
                   int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)
                  if stats[i, cv2.CC_STAT_AREA] >= 80]
    components = [item for item in components if item[0] >= 50]
    if not components:
        raise ValueError("no object component above support plane")
    _, _, largest = max(components)
    mask = (labels == largest).astype(np.uint8)
    if mask.sum() < 200:
        raise ValueError("segmentation too small")
    return mask


def refine_mask_with_rgb(bgr: np.ndarray, depth_mm: np.ndarray, K: np.ndarray,
                         plane: tuple, seed_mask: np.ndarray) -> np.ndarray:
    """Expand the depth seed across visually continuous object regions."""
    ys, xs = np.nonzero(seed_mask)
    h, w = seed_mask.shape
    x0, x1 = max(0, int(xs.min()) - 35), min(w, int(xs.max()) + 36)
    y0, y1 = max(0, int(ys.min()) - 105), min(h, int(ys.max()) + 36)
    roi = bgr[y0:y1, x0:x1]
    seed = seed_mask[y0:y1, x0:x1]
    labels = np.full(seed.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    table_bgr = np.median(bgr[int(0.75 * h):].reshape(-1, 3), axis=0).astype(np.uint8)
    table_lab = cv2.cvtColor(table_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    contrast = np.linalg.norm(lab - table_lab, axis=2) > 20.0
    normal, offset = plane[:2]
    with np.errstate(divide="ignore", invalid="ignore"):
        table_depth_mm = -1000.0 * offset / (_rays(K, depth_mm.shape) @ normal)
    above = (table_depth_mm[y0:y1, x0:x1] - depth_mm[y0:y1, x0:x1]) > 8.0
    horizontal_band = np.zeros(seed.shape, dtype=bool)
    horizontal_band[:, max(0, int(xs.min()) - 45 - x0):min(seed.shape[1], int(xs.max()) + 46 - x0)] = True
    labels[contrast & above & horizontal_band] = cv2.GC_PR_FGD
    labels[seed > 0] = cv2.GC_PR_FGD
    inner = cv2.erode(seed, np.ones((5, 5), np.uint8))
    labels[inner > 0] = cv2.GC_FGD
    labels[:3, :] = cv2.GC_BGD
    labels[-3:, :] = cv2.GC_BGD
    labels[:, :3] = cv2.GC_BGD
    labels[:, -3:] = cv2.GC_BGD
    try:
        cv2.grabCut(roi, labels, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return seed_mask
    foreground = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    choices = [(int(np.count_nonzero((components == i) & (seed > 0))), i)
               for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 100]
    if not choices:
        return seed_mask
    _, chosen = max(choices)
    result = np.zeros_like(seed_mask)
    result[y0:y1, x0:x1] = (components == chosen).astype(np.uint8)
    return result


def object_crop_and_points(scene: Path, index: int, plane=None):
    """Use M1's depth-to-points function and return black-background RGB crop."""
    from tuj.m1_scene.perception import mad_filter, points_from_frame

    bgr, depth_mm, K, T_c2w = load_frame(scene, index)
    if plane is None:
        plane = fit_support_plane(depth_mm, K)
    seed_mask = segment_object(depth_mm, K, plane)
    mask = refine_mask_with_rgb(bgr, depth_mm, K, plane, seed_mask)
    # Only registered above-plane depth contributes to the M1 point cloud.
    normal, offset = plane[:2]
    with np.errstate(divide="ignore", invalid="ignore"):
        table_depth_mm = -1000.0 * offset / (_rays(K, depth_mm.shape) @ normal)
    point_mask = ((mask > 0) & (depth_mm > 0) &
                  ((table_depth_mm - depth_mm) > 18)).astype(np.uint8)
    mask = cv2.morphologyEx(point_mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    objects = points_from_frame(depth_mm / 1000.0, point_mask, K, T_c2w,
                                {1: ("object", "object")})
    if len(objects) != 1 or len(objects[0]["points"]) < 200:
        raise ValueError("M1 point projection failed")
    ys, xs = np.nonzero(mask)
    pad = 8
    x0, x1 = max(0, int(xs.min()) - pad), min(mask.shape[1], int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(mask.shape[0], int(ys.max()) + pad + 1)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[mask == 0] = 0
    return rgb[y0:y1, x0:x1], mad_filter(objects[0]["points"]), mask, plane
