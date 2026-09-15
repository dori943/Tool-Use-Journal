"""Build a simulator scene manifest from mapping, pose and plane audits.

Rows are marked READY_FOR_SCENE only when the table plane was estimated.  The
manifest never hides unresolved orientation or metric scale; those fields keep
the scene at ``BLOCKED_FOR_TRIAL`` until calibration is supplied.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build(mapping_csv: Path, pose_csv: Path, plane_csv: Path, output: Path) -> dict[str, object]:
    mapping = {r["input_id"]: r for r in csv.DictReader(mapping_csv.open(encoding="utf-8", newline=""))}
    pose = {r["input_id"]: r for r in csv.DictReader(pose_csv.open(encoding="utf-8", newline=""))}
    plane = {r["input_id"]: r for r in csv.DictReader(plane_csv.open(encoding="utf-8", newline=""))}
    if set(mapping) != set(pose) or set(mapping) != set(plane):
        raise ValueError("mapping, pose and plane inputs must have identical input IDs")
    rows = []
    for input_id in sorted(mapping):
        m, p, t = mapping[input_id], pose[input_id], plane[input_id]
        plane_ok = t["table_plane_status"] == "ESTIMATED"
        status = "BLOCKED_FOR_TRIAL" if not plane_ok else "READY_FOR_CALIBRATION"
        rows.append({
            "input_id": input_id, "object_id": m["object_id"], "sequence_id": m["sequence_id"], "frame_id": m["frame_id"],
            "ycb_mesh_path": m["ycb_mesh_path"], "ycb_mesh_sha256": m["ycb_mesh_sha256"],
            "centroid_camera_native_xyz": p["centroid_camera_native_xyz"],
            "centroid_world_native_xyz": p["centroid_world_native_xyz"],
            "plane_normal_camera": t["plane_normal_camera"], "plane_offset_native": t["plane_offset_native"],
            "plane_inlier_ratio": t["inlier_ratio"], "orientation_status": p["orientation_status"],
            "metric_scale_status": p["metric_scale_status"], "mesh_pose_alignment": p["mesh_pose_alignment"],
            "table_plane_status": t["table_plane_status"], "scene_status": status,
            "mass_parameter_source": "CALIBRATION_ONLY", "friction_parameter_source": "CALIBRATION_ONLY",
            "real5_gt_in_simulator": "FORBIDDEN",
        })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "static_sim_scene_manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"schema_version": "c4_static_sim_scene_manifest_v1", "input_count": len(rows),
               "ready_for_calibration": sum(r["scene_status"] == "READY_FOR_CALIBRATION" for r in rows),
               "blocked_for_trial": sum(r["scene_status"] == "BLOCKED_FOR_TRIAL" for r in rows),
               "trial_ready": 0, "reason": "orientation and metric scale are unresolved",
               "gt_access": False}
    (output / "scene_manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "scene_manifest.md").write_text(
        "# Static simulator scene manifest\n\n"
        f"- inputs: {len(rows)}\n- ready for calibration: {summary['ready_for_calibration']}\n"
        f"- blocked for trial: {summary['blocked_for_trial']}\n- trial ready: 0\n\n"
        "테이블 평면이 추정된 행도 물체 회전과 metric scale이 미확정이므로 최종 trial 대상이 아니다. 질량·마찰은 calibration-only이며 Real-5 GT를 simulator에 복사하지 않는다.\n",
        encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--plane", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(build(args.mapping, args.pose, args.plane, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
