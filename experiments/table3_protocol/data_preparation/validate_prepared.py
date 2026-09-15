"""Validate frozen provenance, sample inventory and fail-closed scoring gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/c4_real5"))
from evaluate import _read_json, validate_gt, validate_manifest  # noqa: E402


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-score-ready", action="store_true")
    args = parser.parse_args()
    manifest = _read_json(HERE / "real5_input_manifest.json")
    counts = _read_json(HERE / "metric_sample_counts.json")
    registry = yaml.safe_load((HERE / "gt_registry.yaml").read_text(encoding="utf-8"))
    d_panel = yaml.safe_load((HERE / "panel_d_manifest.yaml").read_text(encoding="utf-8"))
    policy = yaml.safe_load((HERE / "qc_policy.yaml").read_text(encoding="utf-8"))
    gt_path = ROOT / manifest["table5_gt"]
    source_path = ROOT / manifest["original_manifest"]
    dataset_root = ROOT / "data/external/kandukuri_ev_realphys"
    if digest(gt_path) != manifest["table5_gt_sha256"] or digest(gt_path) != registry["real5"]["gt_sha256"]:
        raise ValueError("GT_SHA_MISMATCH")
    if (digest(source_path) != manifest["original_manifest_sha256"] or
            digest(ROOT / manifest["audit_input"]) != manifest["audit_sha256"] or
            digest(ROOT / manifest["qc_policy"]) != manifest["qc_policy_sha256"] or
            digest(ROOT / manifest["mask_generator"]) != manifest["mask_generator_sha256"] or
            digest(ROOT / manifest["friction_tracker"]) != manifest["friction_tracker_sha256"]):
        raise ValueError("INPUT_LINEAGE_MISMATCH")
    source = _read_json(source_path)
    gt = validate_gt(_read_json(gt_path))
    mapping = validate_manifest(source, gt, dataset_root)
    if (manifest["archive_sha256"] != registry["real5"]["archive_sha256"] or
            manifest["archive_sha256"] != policy["real5_scope"]["archive_sha256"] or
            len(mapping) != 25 or len(manifest["records"]) != 25):
        raise ValueError("DATASET_MAPPING_UNVERIFIED")
    seen = set()
    for row in manifest["records"]:
        sid, oid = row["sequence_id"], row["object_id"]
        if (sid in seen or mapping[sid] != oid or row["bop_object_id"] != gt[oid]["bop_object_id"] or
                row["fixed_mass_frame"] != 15 or row["mass"]["eligible"] or
                row["friction"]["eligible"] or "tuna" in oid.lower()):
            raise ValueError(f"MANIFEST_ROW_INVALID: {sid}")
        seen.add(sid)
        crop = row["mass"].get("crop")
        if crop and digest(ROOT / crop) != row["mass"]["crop_sha256"]:
            raise ValueError(f"CROP_SHA_MISMATCH: {sid}")
    mass = Counter(row["mass"]["status"] for row in manifest["records"])
    friction = Counter(row["friction"]["status"] for row in manifest["records"])
    if (dict(mass) != counts["mass_input_status_counts"] or
            dict(friction) != counts["friction_input_status_counts"] or
            d_panel["samples"] or registry["independent_manipulation_panel"]["gt"]):
        raise ValueError("COUNT_OR_D_GT_MISMATCH")
    robot = _read_json(ROOT / "configs/robot_spec.json")
    payload = {ee["ee_id"]: ee["payload_kg"] for ee in robot["ee_pool"]}
    if payload != {"2F": 5.0, "3F": 2.5, "vac": 0.5}:
        raise ValueError("PAYLOAD_SPEC_CHANGED: revalidate the D protocol")
    ready = all(x["status"] not in {"BLOCKED", "TARGET_SEMANTICS_MISMATCH"}
                for x in counts["metrics"].values())
    summary = {"provenance": "VERIFIED", "sequences": len(seen), "objects": len(gt),
               "mass_status": dict(mass), "friction_status": dict(friction),
               "table3_score_ready": ready, "d_panel": d_panel["status"]}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_score_ready and not ready:
        raise SystemExit("BLOCKED: do not run Table 3 scoring with these inputs/GT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
