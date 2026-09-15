"""Estimate combined object-table friction from RGB-D sliding trajectories.

Uses the existing M1 depth-to-points projection and FrictionHead's free-slide
deceleration estimator. Object MoCap poses, measured GT, and scene_properties
physical values are never read here. The horizontal support plane and the
object centroid track come from RGB-D, with camera calibration from BOP.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tuj.m1_scene.grounding import FrictionHead
from tuj.m1_scene.perception import mad_filter, points_from_frame

from real_rgbd import fit_support_plane, load_frame, segment_object


def track_scene(scene: Path, reference_frame: int) -> tuple[np.ndarray, dict]:
    _, depth, K, _ = load_frame(scene, reference_frame)
    plane = fit_support_plane(depth, K)
    camera = json.loads((scene / "scene_camera.json").read_text(encoding="utf-8"))
    indices = sorted(int(k) for k in camera)
    if indices != list(range(indices[-1] + 1)):
        raise ValueError(f"{scene}: nonconsecutive annotated frames")
    centers = []
    sizes = []
    for index in indices:
        _, depth, K, T_c2w = load_frame(scene, index)
        mask = segment_object(depth, K, plane)
        obs = points_from_frame(depth / 1000.0, mask, K, T_c2w,
                                {1: ("object", "object")})
        if len(obs) != 1:
            raise ValueError(f"{scene}: object not segmented in RGB-D frame {index}")
        pts = mad_filter(obs[0]["points"])
        centers.append(np.median(pts, axis=0))
        sizes.append(len(pts))
    normal_world = T_c2w[:3, :3] @ plane[0]
    if abs(normal_world[2]) < 0.95:
        raise ValueError(f"{scene}: fitted support plane is not approximately horizontal")
    return np.asarray(centers), {"frame_count": len(indices), "min_points_per_frame": min(sizes),
                                 "table_normal_world_z_abs": round(abs(float(normal_world[2])), 4)}


def choose_slide_window(positions_mm: np.ndarray) -> tuple[int, int, dict]:
    xy = np.asarray(positions_mm, dtype=np.float64)[:, :2]
    if len(xy) < 15 or not np.isfinite(xy).all():
        raise ValueError("too few finite RGB-D centroid positions")
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    # A large segmentation jump during the initial hand push is not a slide.
    initial_outliers = np.flatnonzero(step[: min(18, len(step))] > 80.0)
    after_outliers = int(initial_outliers[-1] + 1) if len(initial_outliers) else 0
    search_end = min(len(step), after_outliers + 16)
    if search_end - after_outliers < 5:
        raise ValueError("no clean initial slide after segmentation jumps")
    smooth = np.convolve(step, np.ones(3) / 3.0, mode="same")
    start = int(after_outliers + np.argmax(smooth[after_outliers:search_end]))
    end = min(len(xy) - 1, start + 35)
    later_jumps = np.flatnonzero(step[start:end] > 80.0)
    if len(later_jumps):
        end = start + int(later_jumps[0])
    for i in range(start + 8, min(len(step) - 2, end)):
        if np.all(step[i:i + 3] < 3.0):
            end = i + 1
            break
    if end - start < 8 or np.linalg.norm(xy[end] - xy[start]) < 50.0:
        raise ValueError("free-slide window too short or under 50 mm")
    return start, end, {"start_frame": start, "end_frame": end,
                        "travel_mm": round(float(np.linalg.norm(xy[end] - xy[start])), 1),
                        "first_step_mm": round(float(step[start]), 1),
                        "last_step_mm": round(float(step[end - 1]), 1)}


def estimate_friction(scene: Path, reference_frame: int) -> tuple[float, dict]:
    positions, track_info = track_scene(scene, reference_frame)
    start, end, window_info = choose_slide_window(positions)
    mu = FrictionHead().probe_mu_from_track(positions[start:end + 1], 1.0 / 30.0)
    if mu is None or not 0.0 < mu <= 2.0:
        raise ValueError(f"{scene}: free-slide deceleration did not yield a valid coefficient")
    return mu, track_info | window_info | {"estimator": "FrictionHead.probe_mu_from_track",
                                            "friction_target": "combined_object_table",
                                            "dt_s": 1.0 / 30.0}
