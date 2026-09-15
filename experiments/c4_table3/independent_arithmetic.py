"""Independent arithmetic audit of preserved historical C4 files only.

This does not make them a Table 3 condition result or prove the separately
stored result.json came from the prediction run.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "configs/c4_real5_gt.json"
PRED = ROOT / "output/c4_real5/20260914T025828Z/predictions.json"
RESULT = ROOT / "output/c4_real5/20260914T030346Z/result.json"


def main() -> None:
    gt = {r["object_id"]: r for r in json.loads(GT.read_text(encoding="utf-8"))["objects"]}
    pred = json.loads(PRED.read_text(encoding="utf-8"))
    historical = json.loads(RESULT.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in pred["predictions"]:
        grouped[row["object_id"]].append(row)
    if set(grouped) != set(gt):
        raise ValueError("historical predictions lack an object")
    errors_mass, errors_friction = [], []
    for oid, truth in gt.items():
        mass = statistics.median(p["mass_est_kg"] for p in grouped[oid])
        friction = statistics.median(p["friction_est"] for p in grouped[oid])
        errors_mass.append(abs(mass-truth["mass_gt_kg"])/truth["mass_gt_kg"])
        errors_friction.append(abs(friction-truth["friction_gt"]))
    values = {"Mass MnRE": statistics.mean(errors_mass),
              "Friction MAE-Real5": statistics.mean(errors_friction)}
    if any(abs(value-historical[key]) > 1e-12 for key, value in values.items()):
        raise ValueError("historical arithmetic does not match stored result")
    print(json.dumps({"status": "HISTORICAL_ARITHMETIC_ONLY",
                      "prediction_count": len(pred["predictions"]),
                      "object_count": len(gt), **values,
                      "table3_assignment": None,
                      "note": "different run directories and no prediction hash in result; no lineage proof"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
