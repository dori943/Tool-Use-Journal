"""Score D_SIM predictions using the registered pure Table III scorers.

The command is intentionally prediction-input only.  It does not create a
prediction and it refuses to score absent/incomplete condition adapters.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "c4_table3"))
from scoring import score_binary_macro, score_clearance, score_critical_mass, score_da, score_mass_acc, score_suction_pf, score_independent_trials  # noqa: E402


def _read_predictions(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def score(root: Path, predictions_path: Path) -> dict:
    gt = yaml.safe_load((root / "sim_gt.yaml").read_text(encoding="utf-8"))
    samples = gt["samples"]
    predictions = _read_predictions(predictions_path)
    mass_sources = {s.get("mass_gt_source") for s in samples}
    suction_sources = {p.get("gt_source") for s in samples for p in s.get("suction_pose_pair", {}).get("poses", [])}
    feas_sources = {t.get("gt_source") for s in samples for t in s.get("ee_trial_gt", {}).values()}
    clearance_sources = {s.get("clearance_gt", {}).get("gt_source") for s in samples}
    if len(mass_sources) != 1 or None in mass_sources or len(suction_sources) != 1 or None in suction_sources or len(feas_sources) != 1 or None in feas_sources or len(clearance_sources) != 1 or None in clearance_sources:
        raise ValueError("MIXED_OR_MISSING_SIM_PROXY_GT_SOURCES")
    mass_source = next(iter(mass_sources))
    suction_source = next(iter(suction_sources))
    feas_source = next(iter(feas_sources))
    clearance_source = next(iter(clearance_sources))
    by_metric = {}
    mass_preds = [p for p in predictions if p.get("metric") == "Mass_Acc"]
    by_metric["Mass_Acc"] = score_mass_acc(
        [{"sample_id": s["sample_id"], "object_id": s["object_instance_id"],
          "mass_gt_kg": s["mass_gt_kg"], "payload_applicability_verified": True,
          "gt_source": mass_source} for s in samples], mass_preds,
        {"2F": 5.0, "3F": 2.5, "vac": 0.5},
        required_gt_source=mass_source)
    suction_expected = []
    for s in samples:
        for pose in s["suction_pose_pair"]["poses"]:
            suction_expected.append({"unit_id": pose["pose_id"], "object_id": s["object_instance_id"],
                                     "gt_label": pose["gt_label"], "trial_success": pose["trial_success"],
                                     "gt_source": suction_source})
    suction_preds = [p for p in predictions if p.get("metric") == "Suction_Acc"]
    by_metric["Suction_Acc"] = score_independent_trials(
        suction_expected, suction_preds, required_gt_source=suction_source)
    pairs = [{"object_id": s["object_instance_id"], "poses": [
        {"pose_id": p["pose_id"], "gt_label": p["gt_label"], "trial_success": p["trial_success"], "gt_source": suction_source}
        for p in s["suction_pose_pair"]["poses"]]} for s in samples]
    by_metric["Suction_PF"] = score_suction_pf(
        pairs, [p for p in predictions if p.get("metric") == "Suction_PF"],
        required_gt_source=suction_source)
    by_metric["Clearance_RelErr"] = score_clearance(
        [{"unit_id": f'{s["sample_id"]}::clearance', "object_id": s["object_instance_id"],
          "gt_margin_mm": s["clearance_gt"]["signed_margin_gt_mm"], "reference_frame": "sim_table_frame",
          "gt_source": clearance_source} for s in samples],
        [p for p in predictions if p.get("metric") == "Clearance_RelErr"],
        required_gt_source=clearance_source)
    feas_expected = []
    for s in samples:
        for ee_id, trial in s["ee_trial_gt"].items():
            feas_expected.append({"unit_id": f'{s["sample_id"]}::{ee_id}', "object_id": s["object_instance_id"],
                                  "trial_success": trial["trial_success"], "gt_source": feas_source})
    by_metric["Feasibility_Acc"] = score_independent_trials(
        feas_expected, [p for p in predictions if p.get("metric") == "Feasibility_Acc"],
        required_gt_source=feas_source)
    da_expected = [{"unit_id": f'{s["sample_id"]}::decision', "object_id": s["object_instance_id"],
                    "allowed_ee_ids": s["decision_gt"]["allowed_ee_ids"],
                    "independent_trial_and_constraint_evidence_ids": s["decision_gt"]["independent_trial_and_constraint_evidence_ids"],
                    "objective_frozen": True} for s in samples]
    by_metric["DA"] = score_da(da_expected, [p for p in predictions if p.get("metric") == "DA"])
    crit_expected = [{"unit_id": s["sample_id"], "object_id": s["object_instance_id"],
                      "mass_gt_kg": s["mass_gt_kg"], "gt_source": mass_source} for s in samples]
    by_metric["Crit"] = score_critical_mass(
        crit_expected, [p for p in predictions if p.get("metric") == "Crit"],
        required_gt_source=mass_source)
    return by_metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score(args.root, args.predictions)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
