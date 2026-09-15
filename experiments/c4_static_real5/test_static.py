"""Focused static adapter/QC/evaluator contract tests; no model calls."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / "experiments/c4_table3")]
from evaluate_static import validate_manifest  # noqa: E402
from infer import FRICTION_PROMPT, parse_friction  # noqa: E402
from scoring import score_real5  # noqa: E402


class StaticContracts(unittest.TestCase):
    def test_visual_friction_parser(self):
        got = parse_friction('```json\n{"combined_friction_coefficient":0.37,"visual_basis":"plastic on table",'
                             '"uncertainty":"high"}\n```')
        self.assertEqual(got["value"], 0.37)
        for bad in ('{"combined_friction_coefficient":-1}',
                    '{"combined_friction_coefficient":"0.3"}',
                    '{"other_coefficient":0.3}'):
            with self.assertRaises((ValueError, KeyError)):
                parse_friction(bad)
        self.assertIn("visual prior", FRICTION_PROMPT)
        self.assertIn("object-table", FRICTION_PROMPT)

    def test_manifest_is_five_by_five_by_two_and_gt_free(self):
        files = sorted((ROOT / "output/c4_table3").glob("*_static_real5_prep/selected_static_inputs.csv"))
        self.assertTrue(files)
        with files[-1].open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        validate_manifest(rows)
        self.assertEqual(len(rows), 50)
        self.assertEqual(set(Counter(r["sequence_id"] for r in rows).values()), {2})
        self.assertEqual(set(Counter(r["object_id"] for r in rows).values()), {10})
        self.assertTrue(all(abs(int(a["frame_id"])-int(b["frame_id"])) >= 20
                            for sid in {r["sequence_id"] for r in rows}
                            for a, b in [[r for r in rows if r["sequence_id"] == sid]]))
        self.assertFalse(any("gt" in key.lower() or "friction_coefficient" in key.lower()
                             for key in rows[0]))
        self.assertEqual(len({r["input_sha256"] for r in rows}), 50)

    def test_median_then_object_macro(self):
        gt = {f"o{i}": {"mass_gt_kg": 1.0, "friction_gt": 0.2} for i in range(5)}
        expected = [{"input_id": f"o{i}_{j}", "object_id": f"o{i}"}
                    for i in range(5) for j in range(2)]
        preds = [{"input_id": f"o{i}_{j}", "object_id": f"o{i}", "parse_status": "OK",
                  "value": float(i+1) if j == 0 else float(i+3)}
                 for i in range(5) for j in range(2)]
        score = score_real5("Mass_MnRE", expected, preds, gt)
        # Each object median is i+2; errors are 1,2,3,4,5; macro is 3.
        self.assertEqual(score["score"], 3.0)
        self.assertEqual(score["status"], "COMPLETE")
        preds[0]["parse_status"] = "FAILED"
        self.assertIsNone(score_real5("Mass_MnRE", expected, preds, gt)["score"])


if __name__ == "__main__":
    unittest.main()
