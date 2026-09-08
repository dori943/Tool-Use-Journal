"""Unit tests for scripts/test_m6_full_lifecycle.py helpers (no live OpenAI)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "test_m6_full_lifecycle.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "test_m6_full_lifecycle_script",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(SCRIPT_PATH.is_file(), "smoke script missing")
class FullLifecycleScriptHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script_module()

    def test_mock_success_result(self):
        result = self.mod.build_mock_recovery_result(
            subgoal_id="SG1_s1_d1",
            outcome="success",
        )
        self.assertEqual(result["subgoal_id"], "SG1_s1_d1")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIsNone(result["failure_code"])

    def test_mock_fail_result_uses_null_failure_code(self):
        result = self.mod.build_mock_recovery_result(
            subgoal_id="SG9_x",
            outcome="fail",
        )
        self.assertEqual(result["subgoal_id"], "SG9_x")
        self.assertEqual(result["status"], "FAIL")
        self.assertIsNone(result["failure_code"])
        self.assertIn("Mock recovery failure", result["detail"])

    def test_invalid_outcome_rejected_by_helper(self):
        with self.assertRaises(ValueError):
            self.mod.build_mock_recovery_result(subgoal_id="SG1", outcome="passed")

    def test_argparse_rejects_invalid_outcome(self):
        with self.assertRaises(SystemExit):
            self.mod._parser().parse_args(["c1_1", "--outcome", "PASSED"])

    def test_phase_b_uses_phase_a_objects_not_fakes(self):
        """Ensure main wires Phase A outputs into process_recovery_outcome."""
        captured: dict = {}

        def fake_run_m6_e2e(**kwargs):
            return {
                "task_id": "demo",
                "subgoal_id": "SG_REAL",
                "failure_context": {
                    "subgoal": {"subgoal_id": "SG_REAL"},
                    "m5_result": {
                        "subgoal_id": "SG_REAL",
                        "status": "FAIL",
                        "phase": "planning",
                        "failure_code": "IK_FAILURE",
                        "detail": "initial",
                    },
                    "task": {"task_id": "demo"},
                    "task_plan": {},
                    "motion_plan": {},
                    "execution": {},
                    "scene": {"nodes": []},
                    "grounding": {},
                },
                "retrieved_experiences": [],
                "diagnosis": {
                    "failure_type": "PLANNING",
                    "failure_cause": {"code": "INVALID_APPROACH", "description": "x"},
                    "affected_module": "M5",
                    "confidence": 0.9,
                },
                "recovery": {
                    "recovery_category": "REPLAN_MOTION",
                    "action": {
                        "recovery_type": "CHANGE_APPROACH",
                        "target_module": "M5",
                        "target": {},
                        "parameters": {},
                    },
                    "routing": {
                        "restart_from": "M5",
                        "rerun_modules": ["M5"],
                        "invalidate": [],
                    },
                },
                "dispatch": {
                    "target_module": "M5",
                    "recovery_type": "CHANGE_APPROACH",
                    "request_path": "x",
                    "module_executed": False,
                },
                "selection": {},
                "summary": {},
            }

        def fake_process_recovery_outcome(**kwargs):
            captured.update(kwargs)
            return {
                "outcome": {"status": "PASS", "verification_result": None},
                "recovery_result": kwargs["recovery_result"],
                "experience": {
                    "experience_id": "RT-TEST",
                    "metadata": {"source": "runtime"},
                    "recovery_summary": {"outcome": {"status": "PASS"}},
                },
                "experience_id": "RT-TEST",
                "appended": True,
                "memory_path": str(kwargs["memory_path"]),
            }

        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.json"
            memory_path.write_text(
                json.dumps(
                    {
                        "object_knowledge": {"objects": {}},
                        "failure_recovery_experience": {"experiences": []},
                        "schema_version": "1.0",
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(self.mod, "run_m6_e2e", side_effect=fake_run_m6_e2e), mock.patch.object(
                self.mod, "process_recovery_outcome", side_effect=fake_process_recovery_outcome
            ), mock.patch.object(self.mod, "count_experiences", side_effect=[0, 1]):
                code = self.mod.main(
                    [
                        "demo",
                        "--outcome",
                        "success",
                        "--memory",
                        str(memory_path),
                        "--backend",
                        "mock",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertIs(captured["diagnosis"]["failure_cause"]["code"], "INVALID_APPROACH")
        self.assertEqual(captured["failure_context"]["subgoal"]["subgoal_id"], "SG_REAL")
        self.assertEqual(captured["recovery"]["action"]["recovery_type"], "CHANGE_APPROACH")
        self.assertEqual(captured["recovery_result"]["subgoal_id"], "SG_REAL")
        self.assertEqual(captured["recovery_result"]["status"], "SUCCESS")
        # Must not invent a fake diagnosis from IK_FAILURE.
        self.assertNotEqual(
            captured["diagnosis"]["failure_cause"]["code"],
            captured["failure_context"]["m5_result"]["failure_code"],
        )


if __name__ == "__main__":
    unittest.main()
