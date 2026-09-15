"""Create a conservative RGB-D pose bridge for Static Real-5 inputs.

Only camera intrinsics and a depth centroid inside the locked QC bounding box
are computed.  Orientation, metric scale calibration and table-plane fitting
remain explicitly unresolved; no Real-5 GT is read.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _depth_path(row: dict[str, str], root: Path) -> Path:
    p = Path(row["depth_path"])
    return p if p.is_absolute() else root / p


def build(static_manifest: Path, qc_jsonl: Path, output: Path, repo_root: Path) -> dict[str, object]:
    rows = list(csv.DictReader(static_manifest.open(encoding="utf-8", newline="")))
    qc = {}
    for line in qc_jsonl.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        key = (item["sequence_id"], str(item["frame"]))
        qc[key] = item
    out = []
    for row in rows:
        sequence_id = row["sequence_id"]
        frame = str(int(row["frame_id"]))
        q = qc.get((sequence_id, frame))
        if q is None:
            raise ValueError(f"missing locked QC bbox for {sequence_id} frame {frame}")
        bbox = q["metrics"]["bbox_xyxy"]
        x0, y0, x1, y1 = [int(v) for v in bbox]
        depth_path = _depth_path(row, repo_root)
        depth = np.asarray(Image.open(depth_path), dtype=np.float64)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(depth.shape[1], x1), min(depth.shape[0], y1)
        patch = depth[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            raise ValueError(f"no valid depth in bbox for {row['input_id']}")
        z = float(np.median(valid))
        camera_path = depth_path.parent.parent / "scene_camera.json"
        camera = json.loads(camera_path.read_text(encoding="utf-8"))[str(int(row["frame_id"]))]
        K = camera["cam_K"]
        R = np.asarray(camera.get("cam_R_w2c", np.eye(3)), dtype=float).reshape(3, 3)
        t = np.asarray(camera.get("cam_t_w2c", [0.0, 0.0, 0.0]), dtype=float)
        fx, fy, cx, cy = float(K[0]), float(K[4]), float(K[2]), float(K[5])
        u, v = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        # Keep the native depth units; scene_camera depth_scale is recorded but
        # not assumed to mean metres until an external calibration confirms it.
        X, Y = (u - cx) * z / fx, (v - cy) * z / fy
        camera_xyz = np.asarray([X, Y, z], dtype=float)
        # The dataset provides w2c extrinsics.  Keep this transform in native
        # units and label it as such; no metre conversion is inferred here.
        world_xyz = R.T @ (camera_xyz - t)
        out.append({
            "input_id": row["input_id"], "object_id": row["object_id"],
            "sequence_id": sequence_id, "frame_id": row["frame_id"],
            "bbox_xyxy": json.dumps([x0, y0, x1, y1]),
            "camera_K": json.dumps(K), "camera_R_w2c": json.dumps(R.tolist()),
            "camera_t_w2c_native": json.dumps(t.tolist()), "depth_scale_metadata": camera.get("depth_scale"),
            "centroid_u_px": u, "centroid_v_px": v, "depth_median_native": z,
            "centroid_camera_native_xyz": json.dumps(camera_xyz.tolist()),
            "centroid_world_native_xyz": json.dumps(world_xyz.tolist()),
            "pose_status": "DEPTH_CENTROID_ONLY",
            "orientation_status": "UNRESOLVED",
            "table_plane_status": "UNRESOLVED",
            "metric_scale_status": "NEEDS_CALIBRATION",
            "mesh_pose_alignment": "UNVERIFIED",
            "extrinsics_transform_status": "NATIVE_UNITS_ONLY",
        })
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "static_sim_pose_bridge.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader(); writer.writerows(out)
    summary = {"schema_version": "c4_static_sim_pose_bridge_v1", "input_count": len(out),
               "pose_status": "DEPTH_CENTROID_ONLY", "orientation_status": "UNRESOLVED",
               "table_plane_status": "UNRESOLVED", "metric_scale_status": "NEEDS_CALIBRATION",
               "gt_access": False, "gt_fields_in_output": False}
    (output / "pose_bridge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "pose_bridge.md").write_text(
        "# Static RGB-D → simulator pose bridge\n\n"
        f"- inputs: {len(out)}\n- pose: `DEPTH_CENTROID_ONLY`\n"
        "- orientation: `UNRESOLVED`\n- table plane: `UNRESOLVED`\n"
        "- metric scale: `NEEDS_CALIBRATION`\n\n"
        "이 산출물은 잠긴 QC bbox와 depth centroid만 사용한다. 물체 자세, 테이블 평면, mesh scale을 추측하지 않으며 evaluator-only GT를 읽지 않는다. 최종 simulator trial 전에 calibration split에서 scale·orientation·table frame을 확정해야 한다.\n",
        encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-manifest", type=Path, required=True)
    ap.add_argument("--qc-jsonl", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(build(args.static_manifest, args.qc_jsonl, args.output, args.repo_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
