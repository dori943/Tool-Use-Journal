"""Independent consistency checks for a blocked R validation bundle."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from audit_r_inputs import ROOT, sha256


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    run = args.run_dir.resolve()
    expected_files = ("config_snapshot.json", "gt_snapshot.json", "condition_matrix.csv",
                      "metric_specific_manifest.csv", "mass_qc_report.md",
                      "friction_semantics_report.md", "raw_predictions.jsonl",
                      "per_object_summary.csv", "per_repeat_summary.csv",
                      "overall_summary.json", "failures.csv", "table3.csv", "table3.md", "run.log")
    assert all((run / name).is_file() for name in expected_files)
    rows = read_csv(run / "metric_specific_manifest.csv")
    assert len(rows) == len({r["sequence_id"] for r in rows}) == 25
    assert Counter(r["object_id"] for r in rows) == {
        "003_cracker_box": 5, "006_mustard_bottle": 5,
        "019_pitcher_base": 5, "021_bleach_cleanser": 5, "025_mug": 5}
    counts = Counter(r["mass_qc_status"] for r in rows)
    assert counts == {"APPROVED": 6, "RESELECTED": 14, "EXCLUDED": 5}
    assert Counter(r["friction_qc_status"] for r in rows) == {"QC_UNREVIEWED": 25}
    assert all(r["semantics_status"] == "TARGET_SEMANTICS_UNVERIFIED" for r in rows)
    assert all(r["selected_friction_window"] for r in rows)
    for r in rows:
        if r["mass_qc_status"] == "EXCLUDED":
            assert not r["selected_mass_frame"] and not r["mass_crop"]
        else:
            assert r["selected_mass_frame"] and sha256(ROOT / r["mass_crop"]) == r["mass_crop_sha256"]
    object_rows = read_csv(run / "per_object_summary.csv")
    assert len(object_rows) == 5
    for r in object_rows:
        expected_count = sum(m["object_id"] == r["object_id"] and m["mass_qc_status"] != "EXCLUDED"
                             for m in rows)
        assert int(r["mass_approved_sequences"]) == expected_count
        assert not r["mass_median_kg"] and not r["friction_median"]
    overall = json.loads((run / "overall_summary.json").read_text(encoding="utf-8"))
    assert overall["ready_cell_count"] == overall["actual_model_api_calls"] == 0
    assert overall["official_mass_MnRE"] is None and overall["official_friction_MAE_Real5"] is None
    assert not (run / "raw_predictions.jsonl").read_bytes()
    table = read_csv(run / "table3.csv")
    assert len(table) == 54 and Counter(r["status"] for r in table) == {"BLOCKED": 42, "-": 12}
    assert all(not r["value"] for r in table)
    assert sha256(run / "gt_snapshot.json") == sha256(ROOT / "configs/c4_real5_gt.json")
    mapping = json.loads((run / "mapping_verification.json").read_text(encoding="utf-8"))
    assert len(mapping["rows"]) == 25 and mapping["numeric_gt_used"] is False
    result = {"status": "PASS", "required_files": len(expected_files),
              "unique_sequences": 25, "mass_approved": 20, "friction_approved": 0,
              "mapping_verified": 25, "ready_cells": 0, "model_calls": 0,
              "score_values_present": False}
    target = run / "bundle_validation.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != result:
            raise ValueError("stored validation result differs from current bundle")
    else:
        with target.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print(result)


if __name__ == "__main__":
    main()
