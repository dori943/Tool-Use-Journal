"""Read-only Real-5 input audit; creates a new, never-overwritten QA run.

This is not a model inference or score run. GT annotations are accessed only
after RGB-D tracking for an explicitly labelled diagnostic rotation check.
Prediction values and evaluation GT numerics are never read by the QA gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "c4_real5"))

from evaluate import _read_json, validate_gt, validate_manifest  # noqa: E402
from friction_from_rgbd import choose_slide_window, track_scene  # noqa: E402
from real_rgbd import (fit_support_plane, load_frame, object_crop_and_points,  # noqa: E402
                       segment_object)  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rotation_diagnostic(scene: Path, start: int, end: int) -> dict:
    """MoCap/GT pose is QA-only: never used for input selection or tracking."""
    gt = _read_json(scene / "scene_gt.json")
    rotations = [np.asarray(gt[str(i)][0]["cam_R_m2w"]).reshape(3, 3)
                 for i in range(start, end + 1)]

    def angle(left: np.ndarray, right: np.ndarray) -> float:
        cosine = np.clip((np.trace(right @ left.T) - 1.0) / 2.0, -1.0, 1.0)
        return math.degrees(math.acos(cosine))

    step = [angle(a, b) for a, b in zip(rotations[:-1], rotations[1:])]
    return {"qa_only_source": "scene_gt cam_R_m2w MoCap pose",
            "net_rotation_deg": round(angle(rotations[0], rotations[-1]), 2),
            "rotation_path_deg": round(sum(step), 2),
            "max_step_rotation_deg": round(max(step), 2)}


def track_diagnostic(centers: np.ndarray, start: int, end: int) -> dict:
    xy = centers[start:end + 1, :2].astype(float)
    displacement = xy[-1] - xy[0]
    direction = displacement / np.linalg.norm(displacement)
    longitudinal = (xy - xy[0]) @ direction
    lateral = (xy - xy[0]) @ np.array([-direction[1], direction[0]])
    t = np.arange(len(xy)) / 30.0
    design = np.column_stack([t, -0.5 * t ** 2, np.ones_like(t)])
    fit = design @ np.linalg.lstsq(design, longitudinal, rcond=None)[0]
    steps = np.diff(longitudinal)
    return {"quadratic_longitudinal_rms_mm": round(float(np.sqrt(np.mean((longitudinal-fit)**2))), 2),
            "lateral_rms_mm": round(float(np.sqrt(np.mean(lateral**2))), 2),
            "reversal_step_count": int(np.sum(steps < -2.0)),
            "slide_frames": int(end - start + 1)}


def crop_diagnostic(scene: Path, frame: int, target: Path) -> dict:
    bgr, depth, K, _ = load_frame(scene, frame)
    plane = fit_support_plane(depth, K)
    seed = segment_object(depth, K, plane)
    crop, points, mask, _ = object_crop_and_points(scene, frame, plane)
    if not cv2.imwrite(str(target), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)):
        raise OSError(f"cannot save {target}")
    ys, xs = np.nonzero(mask)
    sy, sx = np.nonzero(seed)
    crop_width, crop_height = int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)
    seed_width = int(sx.max()-sx.min()+1)
    h, w = mask.shape
    return {"frame": frame, "rgb": f"{scene.relative_to(ROOT).as_posix()}/rgb/{frame:06d}.png",
            "depth": f"{scene.relative_to(ROOT).as_posix()}/depth/{frame:06d}.png",
            "crop": target.relative_to(ROOT).as_posix(), "crop_sha256": sha256(target),
            "crop_size_px": list(crop.shape[:2][::-1]), "mask_pixels": int(mask.sum()),
            "seed_pixels": int(seed.sum()), "point_count": int(len(points)),
            "mask_bbox_width_px": crop_width, "mask_bbox_height_px": crop_height,
            "seed_bbox_width_px": seed_width,
            "mask_to_seed_width_ratio": round(crop_width / seed_width, 3),
            "mask_touches_frame_edge": bool(xs.min() == 0 or xs.max() == w-1 or
                                            ys.min() == 0 or ys.max() == h-1),
            "rgb_depth_dimensions_match": bool(bgr.shape[:2] == depth.shape),
            "rgb_depth_pixel_registration_independently_verified": False}


def sheet(records: list[dict], root: Path, frame: int) -> None:
    tile_w, tile_h, label_h = 230, 180, 25
    canvas = np.full((5 * (tile_h+label_h), 5 * tile_w, 3), 255, np.uint8)
    for n, row in enumerate(records):
        y, x = divmod(n, 5)
        area = canvas[y*(tile_h+label_h):(y+1)*(tile_h+label_h), x*tile_w:(x+1)*tile_w]
        cv2.putText(area, f"{row['sequence_id'].split('/')[-1]} {row['object_id'].split('_', 1)[1]}",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1)
        if row["mass"]["input_status"] == "DECODE_OR_SEGMENTATION_FAILED":
            continue
        crop = cv2.imread(str(ROOT / row["mass"]["crop"]))
        h, w = crop.shape[:2]
        ratio = min((tile_w-10)/w, (tile_h-10)/h, 2.0)
        resized = cv2.resize(crop, (int(w*ratio), int(h*ratio)))
        yy, xx = (tile_h-resized.shape[0])//2, (tile_w-resized.shape[1])//2
        area[label_h+yy:label_h+yy+resized.shape[0], xx:xx+resized.shape[1]] = resized
    cv2.imwrite(str(root / f"frame{frame}_crop_contact_sheet.png"), canvas)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=ROOT / "data/external/kandukuri_ev_realphys")
    p.add_argument("--source-manifest", type=Path)
    p.add_argument("--gt", type=Path, default=ROOT / "configs/c4_real5_gt.json")
    p.add_argument("--output-root", type=Path, default=ROOT / "output/table3_data_preparation")
    p.add_argument("--fixed-mass-frame", type=int, default=15,
                   help="same video-only frame index for all scenes; default is 15")
    args = p.parse_args()
    source_path = args.source_manifest or args.dataset_root / "manifest.json"
    source = _read_json(source_path)
    gt = validate_gt(_read_json(args.gt))
    mapping = validate_manifest(source, gt, args.dataset_root)
    expected_archive = "4c484ec1e5f3f66ccb6a930437404ba7c67a1bb5590791027ced9741e1136eab"
    if source["archive_sha256"] != expected_archive or len(mapping) != 25:
        raise ValueError("DATASET_MAPPING_UNVERIFIED: original SHA or 25-scene inventory differs")
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = args.output_root / token
    run.mkdir(parents=True, exist_ok=False)
    crops = run / f"frame{args.fixed_mass_frame}_crops"
    crops.mkdir()
    records = []
    for source_row in source["sequences"]:
        sid, oid = source_row["sequence_id"], source_row["object_id"]
        scene = args.dataset_root / "ev-realphys" / sid
        prop = _read_json(scene / "scene_properties.json")["object_properties"]
        camera = _read_json(scene / "scene_camera.json")
        gts = _read_json(scene / "scene_gt.json")
        bop = int(gt[oid]["bop_object_id"])
        if prop.keys() != {str(bop)} or any(len(v) != 1 or int(v[0]["obj_id"]) != bop for v in gts.values()):
            raise ValueError(f"DATASET_MAPPING_UNVERIFIED: {sid}")
        frames = sorted(int(k) for k in camera)
        frame = args.fixed_mass_frame  # Independent of GT pose and prediction errors.
        row = {"sequence_id": sid, "object_id": oid, "bop_object_id": bop,
               "annotated_frames": len(frames), "rgb_depth_pairs": source_row["rgb_depth_pair_count"],
               "fixed_mass_frame": frame,
               "historical_selected_frame": source_row["selected_frame"],
               "historical_selection_uses_projected_gt_pose_roi": True,
               "source_near_duplicate_frames_lt_1": source_row["quality"]["near_duplicate_consecutive_count_lt_1"]}
        try:
            if frame not in frames:
                raise ValueError(f"fixed frame {frame} lacks calibration/annotation")
            row["mass"] = crop_diagnostic(scene, frame, crops / f"{sid.split('/')[-1]}_{oid}.png")
            row["mass"]["input_status"] = "SEGMENTED_PENDING_VISUAL_QC"
        except (ValueError, KeyError, cv2.error) as exc:
            row["mass"] = {"input_status": "DECODE_OR_SEGMENTATION_FAILED", "reason": str(exc)}
        try:
            centers, track = track_scene(scene, frame)
            start, end, window = choose_slide_window(centers)
            tilt = math.degrees(math.acos(min(1.0, track["table_normal_world_z_abs"])))
            row["friction"] = {"input_status": "TRACKED_PENDING_FREE_SLIDE_QC",
                               **track, **window, "table_tilt_upper_bound_deg_from_rounded_normal": round(tilt, 3),
                               **track_diagnostic(centers, start, end),
                               **rotation_diagnostic(scene, start, end),
                               "contact_released_verified": False,
                               "rgbd_tracking_uses_gt_pose": False}
        except (ValueError, KeyError, IndexError, cv2.error) as exc:
            row["friction"] = {"input_status": "TRACK_OR_WINDOW_FAILED", "reason": str(exc)}
        records.append(row)
        print(f"{sid}: mass={row['mass']['input_status']} friction={row['friction']['input_status']}", flush=True)
    sheet(records, run, args.fixed_mass_frame)
    out = {"status": "INPUT_QA_ONLY", "archive_sha256": source["archive_sha256"],
           "source_manifest_sha256": sha256(source_path), "table5_gt_sha256": sha256(args.gt),
           "dataset_info_sha256": sha256(args.dataset_root / "ev-realphys/dataset_info.md"),
           "robot_spec_sha256": sha256(ROOT / "configs/robot_spec.json"),
           "fixed_mass_frame_rule": f"index {args.fixed_mass_frame} in each test_sliding sequence; RGB-D only; no GT-pose selection",
           "entries": records}
    target = run / "audit.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(target), "entries": len(records),
                      "mass_segmented": sum(r["mass"]["input_status"] == "SEGMENTED_PENDING_VISUAL_QC" for r in records),
                      "friction_tracked": sum(r["friction"]["input_status"] == "TRACKED_PENDING_FREE_SLIDE_QC" for r in records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
