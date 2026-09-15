"""Freeze split-specific input eligibility from RGB-D QA, never from predictions.

Creates only new files; reuses the C4 GT and original-manifest validators. A
single visual review cannot silently certify a clean mass crop. Friction MAE
is fail-closed on unmodeled rotation and missing independent contact/level QA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/c4_real5"))
from evaluate import _read_json, validate_gt, validate_manifest  # noqa: E402


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_new(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path,
                   default=ROOT / "data/external/kandukuri_ev_realphys")
    p.add_argument("--gt", type=Path, default=ROOT / "configs/c4_real5_gt.json")
    p.add_argument("--output-dir", type=Path, default=HERE)
    args = p.parse_args()
    args.audit = args.audit.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.gt = args.gt.resolve()
    args.output_dir = args.output_dir.resolve()
    policy_path = HERE / "qc_policy.yaml"
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    audit = _read_json(args.audit)
    source_path = args.dataset_root / "manifest.json"
    gt = validate_gt(_read_json(args.gt))
    source = _read_json(source_path)
    mapping = validate_manifest(source, gt, args.dataset_root)
    if (sha256(source_path) != audit["source_manifest_sha256"] or
            sha256(args.gt) != audit["table5_gt_sha256"] or
            audit["archive_sha256"] != policy["real5_scope"]["archive_sha256"] or
            len(mapping) != 25 or len(audit["entries"]) != 25 or
            audit["fixed_mass_frame_rule"].split(" ")[1] != "15"):
        raise ValueError("DATASET_MAPPING_UNVERIFIED: original source/GT/audit/policy mismatch")
    known_mass_issues = set(policy["mass"]["known_failed_visual_gate"])
    records = []
    objects = defaultdict(lambda: {"raw_sequences": 0, "mass_segmented": 0,
                                   "friction_tracked": 0, "mass_ready": 0, "friction_ready": 0})
    for row in audit["entries"]:
        sid, oid = row["sequence_id"], row["object_id"]
        if mapping[sid] != oid or int(gt[oid]["bop_object_id"]) != row["bop_object_id"]:
            raise ValueError(f"DATASET_MAPPING_UNVERIFIED: {sid}")
        objects[oid]["raw_sequences"] += 1
        mass = dict(row["mass"])
        if mass["input_status"] == "SEGMENTED_PENDING_VISUAL_QC":
            objects[oid]["mass_segmented"] += 1
            mass["status"] = ("BLOCKED_MASK_QC" if sid in known_mass_issues
                              else "BLOCKED_PENDING_INDEPENDENT_VISUAL_QC")
            mass["evidence_note"] = ("visible non-object table fragment in frame-15 crop"
                                     if sid in known_mass_issues else
                                     "automated segmentation only; independent blind crop review missing")
        else:
            mass["status"] = "BLOCKED_SEGMENTATION"
        mass["eligible"] = False
        friction = dict(row["friction"])
        if friction["input_status"] != "TRACKED_PENDING_FREE_SLIDE_QC":
            friction["status"] = "BLOCKED_TRACK"
        else:
            objects[oid]["friction_tracked"] += 1
            rotation = friction["rotation_path_deg"]
            limit = policy["friction"]["maximum_cumulative_rotation_deg"]
            if rotation > limit:
                friction["status"] = "TARGET_SEMANTICS_MISMATCH"
                friction["reason"] = (f"{rotation} deg MoCap QA rotation in RGB-D selected window; "
                                      f"translation-only a/g omits rotational/contact dynamics (limit {limit} deg)")
            else:
                friction["status"] = "BLOCKED_EVIDENCE"
                friction["reason"] = "independent gravity-level, external-contact release and RGB-D alignment QA missing"
        friction["eligible"] = False
        friction["table_gravity_level_measured"] = False
        friction["pixel_level_rgb_depth_registration_checked"] = False
        records.append({"sequence_id": sid, "object_id": oid,
                        "bop_object_id": row["bop_object_id"],
                        "dataset_split": "test_sliding", "annotated_frames": row["annotated_frames"],
                        "rgb_depth_pairs": row["rgb_depth_pairs"],
                        "historical_selected_frame": row["historical_selected_frame"],
                        "historical_gt_pose_roi_selection": True,
                        "fixed_mass_frame": 15,
                        "mass": mass, "friction": friction,
                        "friction_failure_does_not_propagate_to_mass": True,
                        "near_duplicate_frame_count_lt_1": row["source_near_duplicate_frames_lt_1"]})
    if any(row["raw_sequences"] != 5 for row in objects.values()) or set(objects) != set(gt):
        raise ValueError("DATASET_MAPPING_UNVERIFIED: expected 5×5 verified instances")
    manifest = {
        "schema_version": "table3_real5_inputs_v1", "status": "GT_READY_INPUTS_BLOCKED",
        "scope": "official EV-RealPhys test_sliding 5 physical objects only",
        "archive_sha256": source["archive_sha256"],
        "source_url": source["source_url"],
        "dataset_version": "no official release tag in local README; pinned by exact archive SHA-256",
        "original_manifest": source_path.relative_to(ROOT).as_posix(),
        "original_manifest_sha256": sha256(source_path),
        "table5_gt": args.gt.relative_to(ROOT).as_posix(),
        "table5_gt_sha256": sha256(args.gt),
        "audit_input": args.audit.relative_to(ROOT).as_posix(),
        "audit_sha256": sha256(args.audit),
        "qc_policy": policy_path.relative_to(ROOT).as_posix(),
        "qc_policy_sha256": sha256(policy_path),
        "mask_generator": "experiments/c4_real5/real_rgbd.py",
        "mask_generator_sha256": sha256(ROOT / "experiments/c4_real5/real_rgbd.py"),
        "friction_tracker": "experiments/c4_real5/friction_from_rgbd.py",
        "friction_tracker_sha256": sha256(ROOT / "experiments/c4_real5/friction_from_rgbd.py"),
        "new_mask_pixels_modified": False,
        "input_change": "separate mass/friction gates; common frame 15 replaces GT-pose-dependent historical selection; new crops stored under distinct QA output",
        "annotation_use": "scene_gt object IDs checked for mapping; MoCap pose read only for all-scene rotation QA after RGB-D window selection; no pose/mask/physical GT passed to estimator",
        "pixel_level_rgb_depth_registration_verified": False,
        "records": records}
    status_mass = Counter(r["mass"]["status"] for r in records)
    status_friction = Counter(r["friction"]["status"] for r in records)
    counts = {
        "schema_version": "table3_metric_counts_v1",
        "manifest_path": (args.output_dir / "real5_input_manifest.json").relative_to(ROOT).as_posix(),
        "real5_raw_sequences": len(records),
        "real5_raw_objects": len(gt),
        "per_object": dict(objects),
        "mass_input_status_counts": dict(status_mass),
        "friction_input_status_counts": dict(status_friction),
        "metrics": {
            "Mass_MnRE": {"panel": "R", "planned_sequences": 25, "planned_objects": 5,
                          "segmented_candidates": sum(r["mass_segmented"] for r in objects.values()),
                          "ready_sequences": 0, "ready_objects": 0,
                          "status": "BLOCKED", "missing": "independent blind mask QC; documented re-segmentation of severe fragments"},
            "Friction_MAE": {"panel": "R", "planned_sequences": 25, "planned_objects": 5,
                             "tracked_window_candidates": sum(r["friction_tracked"] for r in objects.values()),
                             "ready_sequences": 0, "ready_objects": 0,
                             "status": "TARGET_SEMANTICS_MISMATCH",
                             "missing": "rotation-aware combined-μ inference and independent level/contact/registration QA"},
        }}
    for metric in ("Mass_Acc", "Suction_Acc", "Suction_PF", "Clearance_RelErr",
                   "Feasibility_Acc", "DA", "Crit"):
        counts["metrics"][metric] = {"panel": "D", "planned_samples": 0,
                                      "ready_samples": 0, "status": "BLOCKED",
                                      "missing": "separate independent D panel, complete GT and condition adapters"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_new(args.output_dir / "real5_input_manifest.json", manifest)
    write_new(args.output_dir / "metric_sample_counts.json", counts)
    print(json.dumps({"manifest": str(args.output_dir / "real5_input_manifest.json"),
                      "counts": str(args.output_dir / "metric_sample_counts.json"),
                      "mass": dict(status_mass), "friction": dict(status_friction)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
