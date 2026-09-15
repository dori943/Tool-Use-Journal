"""Focused, synthetic arithmetic tests only; fixtures are never experiment predictions."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from run import ROOT, build_cells, canonical_sha, load_existing_predictions, read_matrix, resume_check
from scoring import (safe_input, score_binary_macro, score_clearance,
                     score_critical_mass, score_da, score_independent_trials,
                     score_mass_acc, score_real5, score_repeats,
                     score_suction_pf)


class Table3ScoringTests(unittest.TestCase):
    @staticmethod
    def real5_fixture():
        gt = {f"obj{i}": {"mass_gt_kg": float(i), "friction_gt": i/10}
              for i in range(1, 6)}
        expected = [{"input_id": f"s{i}_{j}", "object_id": f"obj{i}"}
                    for i in range(1, 6) for j in range(3)]
        predicted = [{"input_id": row["input_id"], "object_id": row["object_id"],
                      "parse_status": "OK", "value": gt[row["object_id"]]["mass_gt_kg"]}
                     for row in expected]
        return gt, expected, predicted

    def test_real5_median_then_equal_object_macro_and_mean_sensitivity(self):
        gt, expected, predicted = self.real5_fixture()
        predicted[2]["value"] = 10.0  # one long-tail sequence for obj1
        primary = score_real5("Mass_MnRE", expected, predicted, gt)
        sensitivity = score_real5("Mass_MnRE", expected, predicted, gt,
                                  object_aggregate="mean")
        self.assertEqual(primary["status"], "COMPLETE")
        self.assertAlmostEqual(primary["score"], 0.0)
        self.assertEqual(primary["aggregation"], "primary_object_median_prediction")
        self.assertAlmostEqual(sensitivity["score"], ((1+1+10)/3-1)/5)
        self.assertEqual(sensitivity["aggregation"], "sensitivity_object_mean_prediction")
        self.assertEqual(primary["expected_sequences"], 15)
        self.assertIsNotNone(primary["per_object"][0]["sequence_prediction_std_ddof1"])

    def test_real5_missing_sequence_or_object_is_incomplete_without_imputation(self):
        gt, expected, predicted = self.real5_fixture()
        missing_one = score_real5("Mass_MnRE", expected, predicted[:-1], gt)
        self.assertEqual(missing_one["status"], "INCOMPLETE")
        self.assertIsNone(missing_one["score"])
        self.assertEqual(missing_one["coverage"], 14/15)
        self.assertEqual(missing_one["failure_count"], 1)
        missing_object = score_real5("Mass_MnRE", expected, predicted[:-3], gt)
        self.assertEqual(missing_object["missing_objects"], ["obj5"])
        with self.assertRaisesRegex(ValueError, "DUPLICATE"):
            score_real5("Mass_MnRE", expected, predicted + [predicted[0]], gt)

    def test_friction_mae_object_macro_and_object_std(self):
        gt, expected, _ = self.real5_fixture()
        predicted = [{"input_id": r["input_id"], "object_id": r["object_id"],
                      "parse_status": "OK", "value": gt[r["object_id"]]["friction_gt"]}
                     for r in expected]
        for row in predicted:
            if row["object_id"] == "obj1":
                row["value"] += .1
        result = score_real5("Friction_MAE", expected, predicted, gt)
        self.assertAlmostEqual(result["score"], .02)
        self.assertAlmostEqual(result["object_error_std_ddof1"],
                               math.sqrt((.08**2 + 4*.02**2)/4))
        self.assertEqual(result["per_object"][0]["sequence_prediction_std_ddof1"], 0)

    def test_accuracy_keeps_fixed_denominator_and_object_macro(self):
        expected = [{"unit_id": "a", "object_id": "A", "gt_label": True},
                    {"unit_id": "b1", "object_id": "B", "gt_label": True},
                    {"unit_id": "b2", "object_id": "B", "gt_label": False},
                    {"unit_id": "b3", "object_id": "B", "gt_label": True}]
        predictions = [{"input_id": "a", "parse_status": "OK", "value": True},
                       {"input_id": "b1", "parse_status": "FAILED", "value": None},
                       {"input_id": "b2", "parse_status": "OK", "value": True}]
        result = score_binary_macro(expected, predictions)
        self.assertEqual(result["score"], .5)  # 1/1 and 0/3, not micro 1/4
        self.assertEqual(result["denominator"], 4)
        self.assertEqual(result["failed_prediction_count"], 2)
        self.assertEqual(result["coverage"], .5)

    def test_payload_strict_equality_and_three_ee_units(self):
        samples = [{"sample_id": "x", "object_id": "A", "mass_gt_kg": .5,
                    "payload_applicability_verified": True, "gt_source": "independent_scale"}]
        predicted = [{"input_id": "x", "parse_status": "OK", "value": .49}]
        result = score_mass_acc(samples, predicted, {"2F": 5.0, "3F": 2.5, "vac": .5})
        self.assertEqual(result["denominator"], 3)
        self.assertEqual(result["correct"], 2)
        with self.assertRaisesRegex(ValueError, "PAYLOAD_APPLICABILITY"):
            score_mass_acc([{**samples[0], "payload_applicability_verified": False}],
                           predicted, {"2F": 5.0, "3F": 2.5, "vac": .5})

    def test_suction_pf_requires_both_poses(self):
        feasible = {"pose_id": "A", "gt_label": True,
                    "gt_source": "independent_physical_trials", "trial_success": [True]*5}
        impossible = {"pose_id": "B", "gt_label": False,
                      "gt_source": "independent_physical_trials", "trial_success": [False]*5}
        pairs = [{"object_id": "obj", "poses": [feasible, impossible]}]
        result = score_suction_pf(pairs, [
            {"input_id": "A", "parse_status": "OK", "value": True},
            {"input_id": "B", "parse_status": "OK", "value": True}])
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["denominator_pairs"], 1)
        self.assertEqual(score_suction_pf(pairs, [
            {"input_id": "A", "parse_status": "OK", "value": True},
            {"input_id": "B", "parse_status": "OK", "value": False}])["score"], 1)
        self.assertEqual(score_suction_pf(pairs, [
            {"input_id": "A", "parse_status": "OK", "value": True}])["score"], 0)

    def test_independent_trials_not_proxy_label(self):
        expected = [{"unit_id": "s::vac", "object_id": "obj",
                     "gt_source": "independent_physical_trials",
                     "trial_success": [True, True, True, True, False]}]
        pred = [{"input_id": "s::vac", "parse_status": "OK", "value": True}]
        self.assertEqual(score_independent_trials(expected, pred)["score"], 1)
        with self.assertRaisesRegex(ValueError, "INDEPENDENT"):
            score_independent_trials([{**expected[0], "gt_source": "evaluate_ee_proxy"}], pred)

    def test_clearance_zero_and_object_median(self):
        expected = [{"unit_id": "a", "object_id": "obj", "gt_margin_mm": 0.0},
                    {"unit_id": "b", "object_id": "obj", "gt_margin_mm": 10.0}]
        expected = [{**r, "gt_source": "independent_caliper_or_3d", "reference_frame": "fixture"}
                    for r in expected]
        pred = [{"input_id": "a", "parse_status": "OK", "value": 5.0},
                {"input_id": "b", "parse_status": "OK", "value": 15.0}]
        self.assertAlmostEqual(score_clearance(expected, pred)["score"], .75)
        self.assertEqual(score_clearance(expected, pred[:-1])["status"], "INCOMPLETE")
        with self.assertRaisesRegex(ValueError, "PROTOCOL_FLOOR"):
            score_clearance(expected, pred, denominator_floor_mm=1.0)

    def test_da_multiple_allowed_and_explicit_no_feasible(self):
        expected = [{"unit_id": "a", "object_id": "obj", "allowed_ee_ids": ["2F", "3F"]},
                    {"unit_id": "b", "object_id": "obj", "allowed_ee_ids": []}]
        expected = [{**r, "independent_trial_and_constraint_evidence_ids": ["trial-1"],
                     "objective_frozen": True} for r in expected]
        pred = [{"input_id": "a", "parse_status": "OK", "value": "3F"},
                {"input_id": "b", "parse_status": "OK", "value": "NO_FEASIBLE_EE"}]
        self.assertEqual(score_da(expected, pred)["score"], 1)

    def test_crit_subset_and_no_zero_over_zero_score(self):
        expected = [{"unit_id": "a", "object_id": "obj", "mass_gt_kg": .5,
                     "gt_source": "independent_scale"}]
        pred = [{"input_id": "a", "parse_status": "OK", "value": .5}]
        self.assertEqual(score_critical_mass(expected, pred)["score"], 1)
        self.assertEqual(score_critical_mass([{**expected[0], "mass_gt_kg": .7}], pred)["status"],
                         "BLOCKED")

    def test_five_full_repeat_std_is_ddof1_and_deterministic_na(self):
        runs = [{"repeat_id": i, "status": "COMPLETE", "score": .1*(i+1)}
                for i in range(5)]
        result = score_repeats(runs, 5)
        self.assertAlmostEqual(result["score_mean"], .3)
        self.assertAlmostEqual(result["repeat_std_ddof1"], math.sqrt(.025))
        self.assertEqual(score_repeats(runs[:1], 5, deterministic=True)["repeat_std_display"], "N/A")
        self.assertEqual(score_repeats(runs[:-1], 5)["status"], "INCOMPLETE")

    def test_gt_leakage_allowlist_and_resume_hash(self):
        row = {"sequence_id": "s", "object_id": "obj", "fixed_mass_frame": 15,
               "mass_gt_kg": 999, "friction_gt": 999,
               "mass": {"rgb": "r", "depth": "d", "crop": "c", "crop_sha256": "hash"}}
        payload = safe_input(row, "Mass_MnRE")
        self.assertNotIn("mass_gt_kg", payload)
        self.assertNotIn("friction_gt", payload)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            source.write_text("{}", encoding="utf-8")
            run = root / "run"
            run.mkdir()
            (run / "manifest.json").write_bytes(source.read_bytes())
            config = {"version": 1}
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (run / "raw_predictions.jsonl").write_text("", encoding="utf-8")
            import hashlib
            hash_file = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            (run / "checkpoint.json").write_text(json.dumps({
                "state": "BLOCKED", "config_sha256": canonical_sha(config),
                "manifest_sha256": hash_file(source), "completed_keys": [],
                "prediction_file_sha256": hash_file(run / "raw_predictions.jsonl")}), encoding="utf-8")
            self.assertEqual(resume_check(run, config, source)["status"], "UNCHANGED")
            with self.assertRaisesRegex(ValueError, "CONFIG_OR_MANIFEST"):
                resume_check(run, {"version": 2}, source)

    def test_matrix_ready_cells_match_completed_static_real5(self):
        import yaml
        base = ROOT / "experiments/table3_protocol"
        matrix = read_matrix(base / "condition_matrix.csv")
        counts = json.loads((base / "data_preparation/metric_sample_counts.json").read_text(encoding="utf-8"))
        registry = yaml.safe_load((base / "metric_registry.yaml").read_text(encoding="utf-8"))
        cells = build_cells(matrix, counts, registry)
        self.assertEqual(len(cells), 54)
        self.assertEqual(sum(c["status"] == "READY" for c in cells), 4)
        self.assertTrue(any(c["status"] == "-" for c in cells))
        self.assertTrue(any(c["status"] == "BLOCKED" for c in cells))

    def test_checkpoint_rejects_duplicate_prediction_keys(self):
        record = {"condition_id": "siphy_adopted", "metric": "Mass_MnRE",
                  "input_id": "seq0", "object_id": "obj0", "repeat_id": 0,
                  "requested_seed": 100, "seed_sent_to_api": None,
                  "model_version": "test-only", "prompt_sha256": "f"*64,
                  "temperature": 0.0, "started_at_utc": "2026-01-01T00:00:00Z",
                  "finished_at_utc": "2026-01-01T00:00:01Z", "elapsed_s": 1.0,
                  "raw_response": "test fixture", "value": .1, "unit": "kg",
                  "parse_status": "OK", "failure_reason": None, "input_sha256": "a"*64}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw_predictions.jsonl"
            path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "DUPLICATE_CHECKPOINT"):
                load_existing_predictions(path)


if __name__ == "__main__":
    unittest.main()
