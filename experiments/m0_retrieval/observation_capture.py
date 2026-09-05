"""Capture Experiment 1 observations from the single-object scene.

Uses experiments/m0_retrieval/single_object_scene.py only.
Does NOT load c1_1 / c2_1 / c2_2 production task environments.
Does NOT call SiPhy / LLM.

BBox distinction:
  - M1 BBox  : production M1 depth+seg → build_m1 → bbox_mm (SiPhy input)
  - GT/Asset BBox : MJCF reg_bbox / placement sites (validation overlay only)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from m1_bbox_extract import project_world_aabb_to_image
from objects import OBJECTS, ExperimentObject
from single_object_scene import (
    CAMERA_FOVY,
    CAMERA_NAME,
    CAMERA_POS,
    IMAGE_H,
    IMAGE_W,
    render_single_object,
)


def _log(step: str, msg: str, t0: float | None = None) -> float:
    now = time.perf_counter()
    if t0 is None:
        print(f"{step} {msg}", flush=True)
        return now
    print(f"{step} {msg}  ({now - t0:.2f}s)", flush=True)
    return now


def _draw_box(rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int, color=(0, 255, 0), t: int = 2) -> np.ndarray:
    out = np.array(rgb, copy=True)
    h, w = out.shape[:2]
    x0, x1 = max(0, min(w - 1, x0)), max(0, min(w - 1, x1))
    y0, y1 = max(0, min(h - 1, y0)), max(0, min(h - 1, y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    out[y0 : y0 + t, x0 : x1 + 1] = color
    out[y1 - t + 1 : y1 + 1, x0 : x1 + 1] = color
    out[y0 : y1 + 1, x0 : x0 + t] = color
    out[y0 : y1 + 1, x1 - t + 1 : x1 + 1] = color
    return out


def save_siphy_object_crop(
    rgb: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    min_box: int = 8,
) -> Path:
    """SiPhy-style object crop matching ``scripts/run_m1.py::save_crops``.

    Production logic (unchanged there): 2D bbox from segmentation mask, crop RGB,
    paint non-mask pixels black. Experiment-local copy — does not modify production.
    """
    from PIL import Image

    Hh, Ww = mask.shape[:2]
    mask_b = np.asarray(mask, dtype=bool)
    if not mask_b.any():
        raise RuntimeError("empty segmentation mask; cannot build SiPhy crop")
    ys, xs = np.nonzero(mask_b)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    if x1 - x0 < min_box:
        cx = (x0 + x1) // 2
        x0 = max(0, min(cx - min_box // 2, Ww - min_box))
        x1 = x0 + min_box
    if y1 - y0 < min_box:
        cy = (y0 + y1) // 2
        y0 = max(0, min(cy - min_box // 2, Hh - min_box))
        y1 = y0 + min_box
    crop = np.array(rgb[y0:y1, x0:x1], copy=True)
    crop[~mask_b[y0:y1, x0:x1]] = 0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop).save(out_path)
    return out_path


def _project_gt_asset_bbox(meta: dict[str, Any], bbox_mm: list[float]) -> tuple[int, int, int, int] | None:
    body_pos = np.asarray(meta["body_pos"], dtype=np.float64)
    body_mat = np.asarray(meta["body_mat"], dtype=np.float64).reshape(3, 3)
    cam_pos = np.asarray(meta["cam_xpos"], dtype=np.float64)
    cam_mat = np.asarray(meta["cam_xmat"], dtype=np.float64).reshape(3, 3)
    half = np.asarray(bbox_mm, dtype=np.float64) / 2000.0
    corners_local = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], half[1], half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], -half[2]],
            [half[0], half[1], half[2]],
        ]
    )
    corners_w = (body_mat @ corners_local.T).T + body_pos
    # Convert to center+extents form in world for shared projector
    lo, hi = corners_w.min(axis=0), corners_w.max(axis=0)
    center_mm = ((lo + hi) / 2.0 * 1000.0).tolist()
    extents_mm = ((hi - lo) * 1000.0).tolist()
    return project_world_aabb_to_image(center_mm, extents_mm, cam_pos, cam_mat)


def _bbox_difference(m1_bbox: list[float], gt_bbox: list[float]) -> dict[str, Any]:
    m1 = np.asarray(m1_bbox, dtype=np.float64)
    gt = np.asarray(gt_bbox, dtype=np.float64)
    diff = m1 - gt
    rel = np.where(np.abs(gt) > 1e-9, diff / gt, np.nan)
    return {
        "abs_diff_mm": [round(float(v), 2) for v in diff],
        "rel_diff": [None if not np.isfinite(v) else round(float(v), 4) for v in rel],
        "abs_diff_xyz_labels": ["dx_mm", "dy_mm", "dz_mm"],
    }


def capture_one_object(
    obj: ExperimentObject,
    out_dir: Path,
    save_images: bool = True,
    bbox_mm: list[float] | None = None,
) -> dict[str, Any]:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    total_t0 = time.perf_counter()
    print("=" * 66, flush=True)
    print(f"[OBS] Experiment 1 single-object scene  object={obj.key}", flush=True)
    print("[OBS] production tasks c1_1/c2_1/c2_2: NOT used", flush=True)
    print("[OBS] LLM/SiPhy inference: NOT used", flush=True)
    print("[OBS] SiPhy BBox input source: M1 bbox_mm (not asset GT)", flush=True)
    print(f"[OBS] MUJOCO_GL={os.environ.get('MUJOCO_GL')!r}", flush=True)

    rgb, meta = render_single_object(obj.key)
    m1_bbox = meta.get("m1_bbox_mm")
    m1_center = meta.get("m1_center_mm")

    t = _log("[6]", "Compute / overlay M1 BBox + GT/Asset BBox...", total_t0)
    m1_status, gt_status = "FAIL", "FAIL"
    m1_note, gt_note = None, None
    m1_rgb = rgb.copy()
    gt_rgb = rgb.copy()

    # M1 overlay: project M1 3D AABB (same representation as SiPhy input)
    if m1_bbox is not None and m1_center is not None:
        box = project_world_aabb_to_image(
            m1_center,
            m1_bbox,
            np.asarray(meta["cam_xpos"]),
            np.asarray(meta["cam_xmat"]),
        )
        if box is not None:
            x0, y0, x1, y1 = box
            m1_rgb = _draw_box(rgb, x0, y0, x1, y1, color=(0, 255, 255))  # cyan
            m1_status = "OK"
            m1_note = f"M1 3D bbox_mm projected xy=({x0},{y0})-({x1},{y1})"
        else:
            # fallback: 2D mask AABB
            xyxy = meta.get("m1_image_xyxy_from_mask")
            if xyxy:
                x0, y0, x1, y1 = xyxy
                m1_rgb = _draw_box(rgb, x0, y0, x1, y1, color=(0, 255, 255))
                m1_status = "OK"
                m1_note = f"M1 mask image AABB xy=({x0},{y0})-({x1},{y1}) (projection fallback)"
            else:
                m1_note = "M1 bbox projection failed"
    else:
        m1_note = "M1 bbox missing"

    if bbox_mm is None:
        gt_note = "GT/asset bbox_mm unavailable; overlay skipped"
    else:
        box = _project_gt_asset_bbox(meta, bbox_mm)
        if box is None:
            gt_note = "GT projection failed"
        else:
            x0, y0, x1, y1 = box
            gt_rgb = _draw_box(rgb, x0, y0, x1, y1, color=(0, 255, 0))  # green
            gt_status = "OK"
            gt_note = (
                f"asset/GT bbox_mm={bbox_mm} projected xy=({x0},{y0})-({x1},{y1}); "
                "validation only (NOT SiPhy input)"
            )
    _log("[6]", f"M1 overlay={m1_status}  GT overlay={gt_status}", t)

    # Comparison log
    print("\n----- BBox comparison -----", flush=True)
    print(f"Object: {obj.label}", flush=True)
    print(f"M1 BBox (SiPhy input):", flush=True)
    print(f"  bbox_mm   = {m1_bbox}", flush=True)
    print(f"  center_mm = {m1_center}", flush=True)
    print(f"  n_points  = {meta.get('m1_n_points')}", flush=True)
    print(f"  source    = {meta.get('m1_source')}", flush=True)
    print(f"GT/Asset BBox (validation only):", flush=True)
    print(f"  bbox_mm   = {bbox_mm}", flush=True)
    if m1_bbox is not None and bbox_mm is not None:
        diff = _bbox_difference(m1_bbox, bbox_mm)
        print("Difference (M1 - GT) [dx, dy, dz] mm:", flush=True)
        print(f"  abs_diff_mm = {diff['abs_diff_mm']}", flush=True)
        print(f"  rel_diff    = {diff['rel_diff']}", flush=True)
    print("Experiment 1 SiPhy input: M1 BBox 사용", flush=True)
    print("---------------------------\n", flush=True)

    raw_path = out_dir / f"{obj.key}_raw.png"
    m1_path = out_dir / f"{obj.key}_m1_bbox.png"
    gt_path = out_dir / f"{obj.key}_gt_bbox.png"
    crop_path = out_dir / f"{obj.key}_crop.png"
    # Keep legacy name as alias of gt for compatibility
    legacy_gt_path = out_dir / f"{obj.key}_bbox.png"

    t = _log("[7]", f"Save raw PNG -> {raw_path}", total_t0)
    if save_images:
        Image.fromarray(rgb).save(raw_path)
    _log("[7]", "raw PNG saved", t)

    t = _log("[7b]", f"Save SiPhy crop PNG -> {crop_path}", total_t0)
    crop_status = "FAIL"
    crop_note = None
    if save_images:
        mask = meta.get("seg_mask")
        if mask is None:
            crop_note = "seg_mask missing from scene meta"
        else:
            try:
                save_siphy_object_crop(rgb, mask, crop_path)
                crop_status = "OK"
                crop_note = (
                    "SiPhy-style crop (2D mask bbox, exterior black); "
                    "matches scripts/run_m1.py::save_crops logic"
                )
            except Exception as exc:  # noqa: BLE001
                crop_note = f"crop failed: {exc}"
    _log("[7b]", f"crop PNG status={crop_status}", t)

    t = _log("[8]", f"Save M1 bbox PNG -> {m1_path}", total_t0)
    if save_images:
        Image.fromarray(m1_rgb).save(m1_path)
    _log("[8]", "M1 bbox PNG saved", t)

    t = _log("[8b]", f"Save GT bbox PNG -> {gt_path}", total_t0)
    if save_images:
        Image.fromarray(gt_rgb).save(gt_path)
        Image.fromarray(gt_rgb).save(legacy_gt_path)
    _log("[8b]", "GT bbox PNG saved", t)

    result = {
        "object": obj.key,
        "raw_status": "OK",
        "crop_status": crop_status,
        "m1_bbox_status": m1_status,
        "gt_bbox_status": gt_status,
        # legacy fields
        "bbox_status": gt_status,
        "raw_path": str(raw_path) if save_images else None,
        "crop_path": str(crop_path) if (save_images and crop_status == "OK") else None,
        "m1_bbox_path": str(m1_path) if save_images else None,
        "gt_bbox_path": str(gt_path) if save_images else None,
        "bbox_path": str(legacy_gt_path) if save_images else None,
        "m1_bbox_note": m1_note,
        "gt_bbox_note": gt_note,
        "crop_note": crop_note,
        "bbox_note": gt_note,
        "m1_bbox_mm": m1_bbox,
        "m1_center_mm": m1_center,
        "gt_bbox_mm": bbox_mm,
        "bbox_difference": (
            _bbox_difference(m1_bbox, bbox_mm)
            if (m1_bbox is not None and bbox_mm is not None)
            else None
        ),
        "siphy_bbox_input": "m1_bbox_mm",
        "siphy_image_input": "object_crop_png",
        "task": "exp1_single_object_scene",
        "camera": CAMERA_NAME,
        "camera_pos": list(CAMERA_POS),
        "camera_fovy": CAMERA_FOVY,
        "resolution": [IMAGE_W, IMAGE_H],
        "body_name": "target_main",
        "object_class": meta.get("object_class"),
        "asset_dir": meta.get("asset_dir"),
        "object_xml": meta.get("object_xml"),
        "place_pos": meta.get("place_pos"),
        "table_top_z": meta.get("table_top_z"),
        "m1_source": meta.get("m1_source"),
        "m1_representation": meta.get("m1_representation"),
        "error": None,
        "image_used_in_inference": True,
        "elapsed_sec": round(time.perf_counter() - total_t0, 3),
        "note": (
            "raw / crop / m1_bbox / gt_bbox saved. "
            "Gemini uses object crop (+ optional M1 bbox_mm text). "
            "Overlay PNGs are validation-only."
        ),
    }
    _log("[9]", f"Done (total {result['elapsed_sec']:.2f}s)", total_t0)
    print(
        f"[GT-ASSET] object={obj.key} class={result['object_class']} "
        f"asset_dir={result['asset_dir']}",
        flush=True,
    )
    return result


def capture_all_observations(
    out_dir: Path,
    save_images: bool = True,
    object_keys: list[str] | None = None,
    gt_by_object: dict[str, dict] | None = None,
) -> list[dict[str, Any]]:
    keys = object_keys or list(OBJECTS)
    print(
        f"[OBS] capturing {len(keys)} object(s): {keys}\n"
        f"[OBS] scene: Experiment 1 single-object (table + 1 object)\n"
        f"[OBS] BBox for SiPhy: M1 production pipeline\n"
        f"[OBS] LLM calls: none",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    for key in keys:
        obj = OBJECTS[key]
        bbox_mm = None
        if gt_by_object and key in gt_by_object:
            entry = gt_by_object[key].get("bbox_mm") or {}
            if entry.get("available"):
                bbox_mm = entry.get("value")
        try:
            results.append(
                capture_one_object(obj, out_dir, save_images=save_images, bbox_mm=bbox_mm)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] object={key}: {exc}", flush=True)
            results.append(
                {
                    "object": key,
                    "raw_status": "FAIL",
                    "m1_bbox_status": "FAIL",
                    "gt_bbox_status": "FAIL",
                    "bbox_status": "FAIL",
                    "raw_path": None,
                    "m1_bbox_path": None,
                    "gt_bbox_path": None,
                    "error": str(exc),
                    "task": "exp1_single_object_scene",
                }
            )
    return results


def print_observation_table(results: list[dict[str, Any]]) -> None:
    print("\n===== OBSERVATION VALIDATION =====", flush=True)
    print(
        f"{'Object':10s} {'Raw':8s} {'Crop':8s} {'M1 BBox':10s} {'GT BBox':10s}",
        flush=True,
    )
    for r in results:
        print(
            f"{r['object']:10s} {r.get('raw_status','?'):8s} "
            f"{r.get('crop_status','?'):8s} "
            f"{r.get('m1_bbox_status', r.get('bbox_status','?')):10s} "
            f"{r.get('gt_bbox_status', r.get('bbox_status','?')):10s}",
            flush=True,
        )
        if r.get("error"):
            print(f"  error: {r['error']}", flush=True)
        if r.get("m1_bbox_mm") is not None:
            print(f"  M1 bbox_mm: {r['m1_bbox_mm']}", flush=True)
        if r.get("gt_bbox_mm") is not None:
            print(f"  GT bbox_mm: {r['gt_bbox_mm']}", flush=True)
        if r.get("bbox_difference"):
            print(f"  diff(M1-GT) mm: {r['bbox_difference']['abs_diff_mm']}", flush=True)
        if r.get("resolution"):
            print(f"  resolution: {r['resolution']}", flush=True)
        if r.get("elapsed_sec") is not None:
            print(f"  elapsed: {r['elapsed_sec']}s", flush=True)
