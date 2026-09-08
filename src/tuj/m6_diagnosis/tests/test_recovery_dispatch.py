"""Tests for Recovery Request builder and Dispatcher (Controller→M5)."""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from tuj.m6_diagnosis.diagnosis import validate_diagnosis_output
from tuj.m6_diagnosis.recovery_dispatcher import (
    RecoveryDispatchError,
    dispatch_recovery,
    validate_module_recovery_type,
)
from tuj.m6_diagnosis.recovery_request import RecoveryRequestError, build_recovery_request
from tuj.m6_diagnosis.recovery_router import (
    RecoveryValidationError,
    resolve_recovery_decision,
    validate_diagnosis_recovery_coherence,
    validate_recovery_output,
)
from tuj.m6_diagnosis.taxonomy import FAILURE_MODULES, RECOVERY_ACTION_MODULES


def _full_recovery(
    *,
    recovery_type: str,
    target_module: str,
    restart_from: str | None = None,
    rerun_modules: list[str] | None = None,
    category: str | None = None,
    target: dict | None = None,
    parameters: dict | None = None,
) -> dict:
    from tuj.m6_diagnosis.taxonomy import RECOVERY_VOCABULARY

    if category is None:
        for cat, actions in RECOVERY_VOCABULARY.items():
            if recovery_type in actions:
                category = cat
                break
    assert category is not None
    restart_from = restart_from or target_module
    rerun_modules = list(rerun_modules or [target_module])
    return {
        "decision_mode": "DIAGNOSIS_GUIDED",
        "guidance": {
            "experience_ids": [],
            "past_recoveries": [],
            "recovery_evidence": [],
            "selection": {
                "selected_experience_ids": [],
                "selection_count": 0,
                "selection_audit": [],
            },
        },
        "recovery_category": category,
        "action": {
            "recovery_type": recovery_type,
            "target_module": target_module,
            "target": target
            or {
                "subgoal_id": "SG3",
                "object_id": "spatula_03",
                "property": None,
                "relation": None,
                "ee_id": None,
                "tool_id": None,
            },
            "parameters": {} if parameters is None else parameters,
        },
        "routing": {
            "restart_from": restart_from,
            "rerun_modules": rerun_modules,
            "invalidate": [],
        },
        "outcome": {"status": None, "verification_result": None},
        "metadata": {"attempt": 1, "created_at": None},
    }


class ControllerToM5MigrationTests(unittest.TestCase):
    def test_execution_control_grasp_failure_affected_module_is_m5(self):
        validate_diagnosis_output(
            {
                "failure_type": "EXECUTION_CONTROL",
                "failure_cause": {"code": "GRASP_FAILURE", "description": "grasp"},
                "affected_module": "M5",
                "evidence": ["x"],
                "confidence": 0.8,
            }
        )
        self.assertEqual(FAILURE_MODULES["EXECUTION_CONTROL"], "M5")

    def test_execution_control_slip_maps_to_m5_retry_action(self):
        category, recovery_type, target_module = resolve_recovery_decision(
            "EXECUTION_CONTROL",
            "SLIP",
        )
        self.assertEqual(category, "RETRY_EXECUTION")
        self.assertEqual(recovery_type, "RETRY_ACTION")
        self.assertEqual(target_module, "M5")

    def test_m5_adjust_force_valid(self):
        validate_module_recovery_type("M5", "ADJUST_FORCE")
        self.assertEqual(RECOVERY_ACTION_MODULES["ADJUST_FORCE"], "M5")

    def test_m5_adjust_speed_valid(self):
        validate_module_recovery_type("M5", "ADJUST_SPEED")

    def test_m5_apply_offset_valid(self):
        recovery = _full_recovery(recovery_type="APPLY_OFFSET", target_module="M5")
        validate_recovery_output(recovery)
        validate_diagnosis_recovery_coherence(
            {"history": {"retry_count": 0, "previous_recoveries": []}},
            {
                "failure_type": "EXECUTION_CONTROL",
                "failure_cause": {"code": "SLIP"},
                "affected_module": "M5",
            },
            recovery,
        )

    def test_controller_retry_action_invalid(self):
        with self.assertRaises(RecoveryDispatchError):
            validate_module_recovery_type("Controller", "RETRY_ACTION")
        with self.assertRaises(RecoveryValidationError):
            validate_recovery_output(
                _full_recovery(recovery_type="RETRY_ACTION", target_module="Controller")
            )


class RecoveryRequestTests(unittest.TestCase):
    def test_build_m5_change_approach_request(self):
        recovery = _full_recovery(recovery_type="CHANGE_APPROACH", target_module="M5")
        request = build_recovery_request(recovery)
        self.assertEqual(
            request,
            {
                "recovery_type": "CHANGE_APPROACH",
                "target": {
                    "subgoal_id": "SG3",
                    "object_id": "spatula_03",
                    "property": None,
                    "relation": None,
                    "ee_id": None,
                    "tool_id": None,
                },
                "parameters": {},
            },
        )
        self.assertNotIn("target_module", request)

    def test_build_m3_remeasure_request(self):
        recovery = _full_recovery(
            recovery_type="REMEASURE_PROPERTY",
            target_module="M3",
            target={
                "subgoal_id": "SG2",
                "object_id": "obj_a",
                "property": "mass_kg",
                "relation": None,
                "ee_id": None,
                "tool_id": None,
            },
        )
        request = build_recovery_request(recovery)
        self.assertEqual(request["recovery_type"], "REMEASURE_PROPERTY")
        self.assertEqual(request["target"]["property"], "mass_kg")

    def test_build_m4_reselect_ee_request(self):
        recovery = _full_recovery(
            recovery_type="RESELECT_EE",
            target_module="M4",
            target={
                "subgoal_id": "SG1",
                "object_id": None,
                "property": None,
                "relation": None,
                "ee_id": "2F",
                "tool_id": None,
            },
        )
        request = build_recovery_request(recovery)
        self.assertEqual(request["recovery_type"], "RESELECT_EE")
        self.assertEqual(request["target"]["ee_id"], "2F")

    def test_missing_parameters_become_empty_object(self):
        recovery = _full_recovery(recovery_type="CHANGE_APPROACH", target_module="M5")
        del recovery["action"]["parameters"]
        request = build_recovery_request(recovery)
        self.assertEqual(request["parameters"], {})

    def test_null_target_fields_preserved(self):
        recovery = _full_recovery(recovery_type="REPLAN_MOTION", target_module="M5")
        request = build_recovery_request(recovery)
        self.assertIsNone(request["target"]["property"])
        self.assertIsNone(request["target"]["relation"])

    def test_missing_recovery_type_raises(self):
        recovery = _full_recovery(recovery_type="CHANGE_APPROACH", target_module="M5")
        recovery["action"]["recovery_type"] = None
        with self.assertRaises(RecoveryRequestError):
            build_recovery_request(recovery)

    def test_build_does_not_mutate_recovery(self):
        recovery = _full_recovery(recovery_type="CHANGE_APPROACH", target_module="M5")
        before = deepcopy(recovery)
        build_recovery_request(recovery)
        self.assertEqual(recovery, before)


class RecoveryDispatcherTests(unittest.TestCase):
    def test_dispatch_m5_writes_request_json(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp) / "c1_1"
            recovery = _full_recovery(
                recovery_type="CHANGE_APPROACH",
                target_module="M5",
                target={
                    "subgoal_id": "SG1_s1_d1",
                    "object_id": "obj_ladle_ladle",
                    "property": None,
                    "relation": None,
                    "ee_id": None,
                    "tool_id": None,
                },
            )
            result = dispatch_recovery(recovery, output_dir=output_dir, save=True)

            self.assertEqual(result["target_module"], "M5")
            self.assertEqual(result["recovery_type"], "CHANGE_APPROACH")
            self.assertFalse(result["module_executed"])
            path = Path(result["request_path"])
            self.assertTrue(path.is_file())
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["recovery_type"], "CHANGE_APPROACH")
            self.assertEqual(loaded["target"]["object_id"], "obj_ladle_ladle")
            self.assertEqual(loaded["parameters"], {})

    def test_invalid_module_recovery_type_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            recovery = _full_recovery(recovery_type="CHANGE_APPROACH", target_module="M3")
            # Force inconsistent target_module while keeping canonical category fields
            # invalid for validate_module_recovery_type.
            recovery["action"]["target_module"] = "M3"
            with self.assertRaises(RecoveryDispatchError):
                dispatch_recovery(recovery, output_dir=Path(temp), save=False)

    def test_missing_target_module_raises(self):
        recovery = _full_recovery(recovery_type="CHANGE_APPROACH", target_module="M5")
        recovery["action"]["target_module"] = None
        with self.assertRaises(RecoveryDispatchError):
            dispatch_recovery(recovery, output_dir=Path("."), save=False)

    def test_missing_recovery_type_raises(self):
        recovery = _full_recovery(recovery_type="CHANGE_APPROACH", target_module="M5")
        recovery["action"]["recovery_type"] = None
        with self.assertRaises(RecoveryDispatchError):
            dispatch_recovery(recovery, output_dir=Path("."), save=False)

    def test_dispatch_does_not_mutate_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            recovery = _full_recovery(recovery_type="REPLAN_MOTION", target_module="M5")
            before = deepcopy(recovery)
            dispatch_recovery(recovery, output_dir=Path(temp), save=False)
            self.assertEqual(recovery, before)


if __name__ == "__main__":
    unittest.main()
