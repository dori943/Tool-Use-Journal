"""Input-only Real-5 mass QC; never reads numeric GT or previous predictions.

The policy is stored in r_qc_policy_v2.yaml before candidates are inspected.
This writes diagnostics, not an automatic claim of human visual approval.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/c4_real5")]
from real_rgbd import (_rays, fit_support_plane, load_frame, object_crop_and_points,
                       refine_mask_with_rgb, segment_object)  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(mask)
    return int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1


def candidate(scene: Path, frame: int, policy: dict, dest: Path) -> dict:
    bgr, depth, k, _ = load_frame(scene, frame)
    plane = fit_support_plane(depth, k)
    seed = segment_object(depth, k, plane)
    rgb_mask = refine_mask_with_rgb(bgr, depth, k, plane, seed)
    crop, points, final_mask, _ = object_crop_and_points(scene, frame, plane=plane)
    x0, y0, x1, y1 = _bbox(final_mask)
    sx0, _, sx1, _ = _bbox(seed)
    blur = cv2.Laplacian(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY),
                         cv2.CV_64F).var()
    normal, offset = plane[:2]
    with np.errstate(divide="ignore", invalid="ignore"):
        table_depth = -1000.0 * offset / (_rays(k, depth.shape) @ normal)
    heights = table_depth - depth.astype(float)
    near = (final_mask > 0) & (heights >= 18) & (heights < 35)
    metrics = {
        "point_count": len(points),
        "valid_depth_fraction_in_rgb_mask": float(np.mean(depth[rgb_mask > 0] > 0)),
        "laplacian_variance_in_bbox": float(blur),
        "distance_to_image_edge_px": min(x0, y0, bgr.shape[1] - x1, bgr.shape[0] - y1),
        "refined_to_seed_bbox_width_ratio": (x1 - x0) / (sx1 - sx0),
        "near_plane_fraction_in_final_mask": float(near.sum() / final_mask.sum()),
        "mask_pixels": int(final_mask.sum()),
        "seed_pixels": int(seed.sum()),
        "bbox_xyxy": [x0, y0, x1, y1],
        "crop_sha256": "",
        "rgb_sha256": sha256(scene / "rgb" / f"{frame:06d}.png"),
        "depth_sha256": sha256(scene / "depth" / f"{frame:06d}.png"),
    }
    p = policy["mass"]
    checks = {
        "point_count": len(points) >= p["minimum_point_count"],
        "depth_valid": metrics["valid_depth_fraction_in_rgb_mask"] >= p["minimum_valid_depth_fraction_in_rgb_mask"],
        "blur": blur >= p["minimum_laplacian_variance_in_bbox"],
        "frame_edge": metrics["distance_to_image_edge_px"] >= p["minimum_distance_to_image_edge_px"],
        "mask_width": metrics["refined_to_seed_bbox_width_ratio"] <= p["maximum_refined_to_seed_bbox_width_ratio"],
        "table_contamination": metrics["near_plane_fraction_in_final_mask"] <= p["maximum_near_plane_fraction_in_final_mask"],
    }
    sid = scene.name
    prefix = f"{sid}_{frame:06d}"
    paths = {
        "crop": dest / "crops" / f"{prefix}.png",
        "seed_mask": dest / "original_masks" / f"{prefix}_seed.png",
        "rgb_mask": dest / "original_masks" / f"{prefix}_rgb_refined.png",
        "final_mask": dest / "original_masks" / f"{prefix}_point_filtered.png",
        "overlay": dest / "overlays" / f"{prefix}.png",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(paths["crop"]), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    for key, mask in (("seed_mask", seed), ("rgb_mask", rgb_mask), ("final_mask", final_mask)):
        cv2.imwrite(str(paths[key]), (mask > 0).astype(np.uint8) * 255)
    overlay = bgr.copy()
    overlay[final_mask > 0] = (0.55 * overlay[final_mask > 0] +
                               0.45 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.imwrite(str(paths["overlay"]), overlay)
    metrics["crop_sha256"] = sha256(paths["crop"])
    checks = {key: bool(value) for key, value in checks.items()}
    return {"frame": frame, "passes_automatic": bool(all(checks.values())),
            "failed_checks": [k for k, okay in checks.items() if not okay],
            "checks": checks, "metrics": metrics,
            "artifacts": {k: str(v.relative_to(ROOT)) for k, v in paths.items()}}


def contact_sheet(rows: list[dict], dest: Path) -> None:
    w, h = 270, 260
    canvas = np.full((5 * h, 5 * w, 3), 255, dtype=np.uint8)
    for idx, row in enumerate(rows):
        y, x = divmod(idx, 5)
        selected = row.get("selected")
        if selected is None:
            continue
        crop = cv2.imread(str(ROOT / selected["artifacts"]["crop"]))
        if crop is None:
            continue
        hh, ww = crop.shape[:2]
        scale = min(240 / max(ww, 1), 215 / max(hh, 1), 2.0)
        crop = cv2.resize(crop, (max(1, int(ww * scale)), max(1, int(hh * scale))))
        xx, yy = x * w + (w - crop.shape[1]) // 2, y * h + 30
        canvas[yy:yy + crop.shape[0], xx:xx + crop.shape[1]] = crop
        label = f"{row['sequence_id'][-6:]} f{selected['frame']} {'PASS' if selected['passes_automatic'] else 'FAIL'}"
        cv2.putText(canvas, label, (x * w + 5, y * h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(dest / "selected_mass_contact_sheet.png"), canvas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit(f"refuse to overwrite {out}")
    out.mkdir(parents=True)
    policy_file = ROOT / "experiments/c4_table3/r_qc_policy_v2.yaml"
    policy = yaml.safe_load(policy_file.read_text(encoding="utf-8"))
    manifest_file = ROOT / "data/external/kandukuri_ev_realphys/manifest.json"
    source = json.loads(manifest_file.read_text(encoding="utf-8"))
    if len(source["sequences"]) != 25 or len({s["object_id"] for s in source["sequences"]}) != 5:
        raise ValueError("not the locked 25-sequence Real-5 input")
    rows = []
    for source_row in source["sequences"]:
        sid = source_row["sequence_id"]
        scene = ROOT / "data/external/kandukuri_ev_realphys/ev-realphys" / sid
        examined = []
        for frame in policy["candidate_frames"]:
            try:
                result = candidate(scene, frame, policy, out)
            except Exception as exc:
                result = {"frame": frame, "passes_automatic": False,
                          "failed_checks": ["processing_error"],
                          "error": f"{type(exc).__name__}: {exc}"}
            examined.append(result)
            if result["passes_automatic"]:
                break
        selected = next((v for v in examined if v["passes_automatic"]), None)
        rows.append({"sequence_id": sid, "object_id": source_row["object_id"],
                     "bop_object_id": source_row["bop_object_id"],
                     "selected": selected, "candidates": examined})
        print(sid, "auto-pass" if selected else "auto-fail",
              selected["frame"] if selected else "", flush=True)
    (out / "mass_candidate_audit.json").write_text(json.dumps({
        "policy_path": str(policy_file.relative_to(ROOT)),
        "policy_sha256": sha256(policy_file),
        "source_manifest_sha256": sha256(manifest_file),
        "numeric_gt_used": False, "prior_prediction_error_used": False,
        "mask_modified": False, "pixel_registration_independently_verified": False,
        "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    contact_sheet(rows, out)
    with (out / "mass_candidate_audit.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence_id", "object_id", "selected_frame",
                                                 "automatic_pass", "review_status", "review_reason"])
        writer.writeheader()
        for row in rows:
            selected = row["selected"]
            writer.writerow({"sequence_id": row["sequence_id"], "object_id": row["object_id"],
                             "selected_frame": selected["frame"] if selected else "",
                             "automatic_pass": bool(selected),
                             "review_status": "QC_UNREVIEWED", "review_reason": "requires visual crop and overlay review"})


if __name__ == "__main__":
    main()
