"""Tests for scripts/run.py M6 orchestration helpers (M1–M5 untouched)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_PY = REPO_ROOT / "scripts" / "run.py"


def _load_run_module():
    spec = importlib.util.spec_from_file_location("tuj_run_py_m6_integration", RUN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(RUN_PY.is_file(), "scripts/run.py missing")
class RunPyM6HelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runmod = _load_run_module()

    def test_downstream_chain_expansion(self):
        self.assertEqual(
            self.runmod.planned_downstream_chain("M1"),
            ["M1", "M2", "M3", "M4", "M5"],
        )
        self.assertEqual(
            self.runmod.planned_downstream_chain("M2"),
            ["M2", "M3", "M4", "M5"],
        )
        self.assertEqual(
            self.runmod.planned_downstream_chain("M3"),
            ["M3", "M4", "M5"],
        )
        self.assertEqual(
            self.runmod.planned_downstream_chain("M4"),
            ["M4", "M5"],
        )
        self.assertEqual(self.runmod.planned_downstream_chain("M5"), ["M5"])

    def test_read_success_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subgoal_result.json"
            path.write_text(
                json.dumps(
                    {
                        "subgoal_id": "SG1_s1_d1",
                        "status": "SUCCESS",
                        "phase": None,
                        "failure_code": None,
                        "detail": None,
                    }
                ),
                encoding="utf-8",
            )
            result = self.runmod.read_m5_subgoal_result(path)
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(result["subgoal_id"], "SG1_s1_d1")

    def test_read_fail_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subgoal_result.json"
            path.write_text(
                json.dumps(
                    {
                        "subgoal_id": "SG1_s1_d1",
                        "status": "FAIL",
                        "phase": "planning",
                        "failure_code": "IK_FAILURE",
                        "detail": "x",
                    }
                ),
                encoding="utf-8",
            )
            result = self.runmod.read_m5_subgoal_result(path)
            self.assertEqual(result["status"], "FAIL")

    def test_missing_result_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                self.runmod.read_m5_subgoal_result(Path(tmp) / "subgoal_result.json")
            )

    def test_invalid_status_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subgoal_result.json"
            path.write_text(
                json.dumps(
                    {
                        "subgoal_id": "SG1_s1_d1",
                        "status": "FAILED",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(self.runmod.M5SubgoalResultError):
                self.runmod.read_m5_subgoal_result(path)

    def test_success_bypasses_m6(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "c1_1"
            m5 = out / "m5"
            m5.mkdir(parents=True)
            (m5 / "subgoal_result.json").write_text(
                json.dumps(
                    {
                        "subgoal_id": "SG1_s1_d1",
                        "status": "SUCCESS",
                        "phase": None,
                        "failure_code": None,
                        "detail": None,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(skip_m6=False, memory="none")
            with mock.patch("tuj.m6_diagnosis.run_m6_e2e") as mocked:
                result = self.runmod.maybe_invoke_m6_after_m5("c1_1", out, args)
                mocked.assert_not_called()
            self.assertFalse(result["invoked"])
            self.assertEqual(result["reason"], "success")
            self.assertFalse(result["runtime_experience_written"])

    def test_missing_does_not_fallback_to_m5_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "c1_1"
            m5 = out / "m5"
            m5.mkdir(parents=True)
            (m5 / "m5_failure.json").write_text("[{}]", encoding="utf-8")
            args = SimpleNamespace(skip_m6=False, memory="none")
            with mock.patch("tuj.m6_diagnosis.run_m6_e2e") as mocked:
                result = self.runmod.maybe_invoke_m6_after_m5("c1_1", out, args)
                mocked.assert_not_called()
            self.assertEqual(result["reason"], "missing_subgoal_result")
            self.assertFalse(result["invoked"])

    def test_fail_calls_m6_with_detail_id_and_skips_process_recovery_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "c1_1"
            m5 = out / "m5"
            m5.mkdir(parents=True)
            (m5 / "subgoal_result.json").write_text(
                json.dumps(
                    {
                        "subgoal_id": "SG1_s1_d1",
                        "status": "FAIL",
                        "phase": "planning",
                        "failure_code": "IK_FAILURE",
                        "detail": "x",
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(skip_m6=False, memory="none")
            fake_phase_a = {
                "subgoal_id": "SG1_s1_d1",
                "failure_context": {},
                "diagnosis": {
                    "failure_type": "PLANNING",
                    "failure_cause": {"code": "INVALID_APPROACH"},
                    "affected_module": "M5",
                },
                "recovery": {
                    "action": {
                        "recovery_type": "CHANGE_APPROACH",
                        "target_module": "M5",
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
                    "request_path": str(out / "m6" / "recovery_request.json"),
                    "module_executed": False,
                },
            }
            with mock.patch("tuj.m6_diagnosis.run_m6_e2e", return_value=fake_phase_a) as mocked_e2e, mock.patch(
                "tuj.m6_diagnosis.process_recovery_outcome"
            ) as mocked_outcome:
                result = self.runmod.maybe_invoke_m6_after_m5("c1_1", out, args)
                mocked_e2e.assert_called_once()
                kwargs = mocked_e2e.call_args.kwargs
                self.assertEqual(kwargs["subgoal_id"], "SG1_s1_d1")
                self.assertEqual(kwargs["task_id"], "c1_1")
                mocked_outcome.assert_not_called()
            self.assertTrue(result["invoked"])
            self.assertEqual(result["planned_downstream_chain"], ["M5"])
            self.assertFalse(result["recovery_executed"])
            self.assertFalse(result["runtime_experience_written"])

    def test_fail_m3_plans_downstream_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "demo"
            m5 = out / "m5"
            m5.mkdir(parents=True)
            (m5 / "subgoal_result.json").write_text(
                json.dumps(
                    {
                        "subgoal_id": "SG1_s1_d1",
                        "status": "FAIL",
                        "phase": "planning",
                        "failure_code": "IK_FAILURE",
                        "detail": "x",
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(skip_m6=False, memory="none")
            fake_phase_a = {
                "subgoal_id": "SG1_s1_d1",
                "recovery": {
                    "routing": {
                        "restart_from": "M3",
                        "rerun_modules": ["M3"],
                        "invalidate": [],
                    }
                },
                "dispatch": {
                    "target_module": "M3",
                    "recovery_type": "REMEASURE_PROPERTY",
                    "request_path": "x",
                    "module_executed": False,
                },
            }
            with mock.patch("tuj.m6_diagnosis.run_m6_e2e", return_value=fake_phase_a):
                result = self.runmod.maybe_invoke_m6_after_m5("demo", out, args)
            self.assertEqual(
                result["planned_downstream_chain"],
                ["M3", "M4", "M5"],
            )


class SubgoalResultE2EGateTests(unittest.TestCase):
    def test_has_m5_failure_artifact_accepts_subgoal_result_fail(self):
        from tuj.m6_diagnosis.e2e_runner import has_m5_failure_artifact

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m5 = root / "m5"
            m5.mkdir()
            (m5 / "subgoal_result.json").write_text(
                json.dumps(
                    {
                        "subgoal_id": "SG1_s1_d1",
                        "status": "FAIL",
                        "phase": "planning",
                        "failure_code": "IK_FAILURE",
                        "detail": None,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(has_m5_failure_artifact(root))

    def test_has_m5_failure_artifact_ignores_success_subgoal_result(self):
        from tuj.m6_diagnosis.e2e_runner import has_m5_failure_artifact

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m5 = root / "m5"
            m5.mkdir()
            (m5 / "subgoal_result.json").write_text(
                json.dumps(
                    {
                        "subgoal_id": "SG1_s1_d1",
                        "status": "SUCCESS",
                        "phase": None,
                        "failure_code": None,
                        "detail": None,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(has_m5_failure_artifact(root))


if __name__ == "__main__":
    unittest.main()
