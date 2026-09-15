"""Validate the simulator-only D_SIM manifest/GT contract."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    manifest = yaml.safe_load((root / "sim_manifest.yaml").read_text(encoding="utf-8"))
    gt = yaml.safe_load((root / "sim_gt.yaml").read_text(encoding="utf-8"))
    if manifest.get("panel") != "D_SIM" or gt.get("panel") != "D_SIM":
        errors.append("PANEL_NOT_D_SIM")
    samples = manifest.get("samples", [])
    gt_samples = gt.get("samples", [])
    if len(samples) != 5 or len(gt_samples) != 5:
        errors.append("EXPECTED_FIVE_SIM_SAMPLES")
    mids = {s.get("sample_id") for s in samples}
    gids = {s.get("sample_id") for s in gt_samples}
    if mids != gids or len(mids) != 5:
        errors.append("MANIFEST_GT_SET_MISMATCH")
    for row in gt_samples:
        mass = row.get("mass_gt_kg")
        if not isinstance(mass, (int, float)) or not math.isfinite(mass) or mass <= 0:
            errors.append(f"INVALID_MASS:{row.get('sample_id')}")
        if row.get("mass_gt_source") != "simulator_hidden_state_compiled_subtree_mass":
            errors.append(f"INVALID_MASS_SOURCE:{row.get('sample_id')}")
        poses = row.get("suction_pose_pair", {}).get("poses", [])
        if len(poses) != 2 or any(len(p.get("trial_success", [])) != 5 for p in poses):
            errors.append(f"INVALID_SUCTION_TRIALS:{row.get('sample_id')}")
        for ee_id, data in row.get("ee_trial_gt", {}).items():
            if len(data.get("trial_success", [])) != 5:
                errors.append(f"INVALID_FEASIBILITY_TRIALS:{row.get('sample_id')}:{ee_id}")
        clearance = row.get("clearance_gt", {})
        if clearance.get("gt_source") != "simulator_geometry" or not clearance.get("reference_frame"):
            errors.append(f"INVALID_CLEARANCE_SOURCE:{row.get('sample_id')}")
    if not any(row.get("critical_subset_member") for row in gt_samples):
        errors.append("CRIT_SUBSET_EMPTY")
    for row in samples:
        if "mass_gt_kg" in row or "trial_success" in row or "allowed_ee_ids" in row:
            errors.append(f"GT_LEAKED_IN_MANIFEST:{row.get('sample_id')}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    errors = validate(args.root)
    if errors:
        raise SystemExit("SIM_GT_INVALID: " + ", ".join(errors))
    print("SIM_GT_VALID")


if __name__ == "__main__":
    main()

