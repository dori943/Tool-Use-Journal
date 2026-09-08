"""Tests for Recovery Outcome → Runtime Experience → M0 append."""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from tuj.m6_diagnosis.memory_adapter import MemoryAdapter, MemoryAppendError, append_experience
from tuj.m6_diagnosis.recovery_outcome import (
    RecoveryResultError,
    process_recovery_outcome,
    validate_recovery_result,
)
from tuj.m6_diagnosis.runtime_experience import (
    RuntimeExperienceBuilder,
    build_runtime_experience,
)
from tuj.m6_diagnosis.context_similarity import compute_context_similarity
from tuj.m6_diagnosis.retrieval_query import build_retrieval_query


def _failure_context(**overrides) -> dict:
    context = {
        "failure_id": "fail-runtime-test",
        "task": {"task_id": "c1_1", "instruction": "Use the ladle."},
        "subgoal": {
            "subgoal_id": "SG1_s1_d1",
            "parent_subgoal_id": "SG1_s1",
            "detail_id": "SG1_s1_d1",
            "description": "Acquire the ladle",
            "action_type": "acquire",
            "target_object_ids": ["obj_ladle"],
            "selected_object_id": "obj_ladle",
            "selected_object_class": "ladle",
            "preconditions": [],
            "postconditions": [],
            "invariants": [],
        },
        "scene": {
            "nodes": [{"id": "obj_ladle", "class": "ladle"}],
            "relations": [],
            "object_states": {},
        },
        "grounding": {},
        "task_plan": {"selected_ee": "2F", "selected_tool": "ladle"},
        "motion_plan": {"planning_status": "FAILED", "planning_error": None},
        "execution": {"controller_status": None},
        "m5_result": {
            "subgoal_id": "SG1_s1_d1",
            "status": "FAIL",
            "phase": "planning",
            "failure_code": "IK_FAILURE",
            "detail": "Initial failure",
        },
        "observation": {},
        "history": {
            "retry_count": 0,
            "previous_diagnoses": [],
            "previous_recoveries": [],
            "previous_outcomes": [],
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(context.get(key), dict):
            context[key] = {**context[key], **value}
        else:
            context[key] = value
    return context


def _diagnosis() -> dict:
    return {
        "failure_type": "PLANNING",
        "failure_cause": {
            "code": "INVALID_APPROACH",
            "description": "The approach is invalid.",
        },
        "affected_module": "M5",
        "confidence": 0.9,
        "evidence": [],
    }


def _recovery() -> dict:
    return {
        "recovery_category": "REPLAN_MOTION",
        "action": {
            "recovery_type": "CHANGE_APPROACH",
            "target_module": "M5",
            "target": {
                "subgoal_id": "SG1_s1_d1",
                "object_id": "obj_ladle",
                "property": None,
                "relation": None,
                "ee_id": None,
                "tool_id": None,
            },
            "parameters": {},
        },
        "routing": {
            "restart_from": "M5",
            "rerun_modules": ["M5"],
            "invalidate": [],
        },
    }


def _offline_seed(experience_id: str = "OFFLINE-SEED-001") -> dict:
    return {
        "experience_id": experience_id,
        "context_signature": {
            "task_id": None,
            "subgoal_id": None,
            "subgoal_description": "Acquire the ladle",
            "detail_id": None,
            "action_type": "acquire",
            "target": {"object_id": "obj_ladle", "object_class": "ladle"},
            "violated_predicates": [],
            "selected_ee": "2F",
            "selected_tool": "ladle",
            "execution_signature": {
                "motion_planning_status": "FAILED",
                "controller_status": None,
            },
        },
        "diagnosis_summary": {
            "source_failure_type": "offline label",
            "failure_type": "PLANNING",
            "failure_cause": {
                "code": "INVALID_APPROACH",
                "description": "seed",
            },
            "affected_module": "M5",
            "confidence": 0.5,
        },
        "recovery_summary": {
            "recovery_category": "REPLAN_MOTION",
            "action": {
                "target": {
                    "subgoal_id": None,
                    "object_id": None,
                    "property": None,
                    "relation": None,
                    "ee_id": None,
                    "tool_id": None,
                },
                "parameters": {},
                "recovery_type": "CHANGE_APPROACH",
            },
            "changes": [],
            "routing": {
                "restart_from": "M5",
                "rerun_modules": ["M5"],
                "invalidate": [],
            },
            "outcome": {"status": "NOT_EXECUTED", "verification_result": None},
        },
        "metadata": {
            "source": "offline_seed",
            "dataset": "fixture",
            "source_episode": "1",
            "created_at": None,
            "updated_at": None,
            "reuse_count": 0,
        },
    }


def _write_memory(path: Path, experiences: list[dict], *, object_knowledge=None) -> None:
    payload = {
        "object_knowledge": object_knowledge
        if object_knowledge is not None
        else {"objects": {"keep_me": {"props": {}}}},
        "failure_recovery_experience": {"experiences": experiences},
        "schema_version": "1.0",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class RecoveryResultValidationTests(unittest.TestCase):
    def test_valid_success(self):
        validated = validate_recovery_result(
            {
                "subgoal_id": "SG1_s1_d1",
                "status": "SUCCESS",
                "phase": "execution",
                "failure_code": None,
                "detail": None,
            },
            expected_subgoal_id="SG1_s1_d1",
        )
        self.assertEqual(validated["status"], "SUCCESS")

    def test_subgoal_mismatch_raises(self):
        with self.assertRaises(RecoveryResultError) as ctx:
            validate_recovery_result(
                {
                    "subgoal_id": "SG2_s1_d1",
                    "status": "SUCCESS",
                    "phase": "execution",
                    "failure_code": None,
                    "detail": None,
                },
                expected_subgoal_id="SG1_s1_d1",
            )
        self.assertIn("expected SG1_s1_d1, got SG2_s1_d1", str(ctx.exception))

    def test_failed_alias_rejected(self):
        with self.assertRaises(RecoveryResultError):
            validate_recovery_result(
                {"subgoal_id": "SG1_s1_d1", "status": "FAILED"},
                expected_subgoal_id="SG1_s1_d1",
            )

    def test_pass_alias_rejected(self):
        with self.assertRaises(RecoveryResultError):
            validate_recovery_result(
                {"subgoal_id": "SG1_s1_d1", "status": "PASS"},
                expected_subgoal_id="SG1_s1_d1",
            )

    def test_none_recovery_result_rejected(self):
        with self.assertRaises(RecoveryResultError) as ctx:
            validate_recovery_result(None, expected_subgoal_id="SG1_s1_d1")
        self.assertIn("required", str(ctx.exception).lower())


class RuntimeExperienceBuilderTests(unittest.TestCase):
    def test_runtime_pass_mapping(self):
        experience = build_runtime_experience(
            failure_context=_failure_context(),
            diagnosis=_diagnosis(),
            recovery=_recovery(),
            recovery_result={
                "subgoal_id": "SG1_s1_d1",
                "status": "SUCCESS",
                "phase": "execution",
                "failure_code": None,
                "detail": None,
            },
            experience_id="RT-TEST-PASS",
            created_at="2026-09-08T15:00:00+00:00",
        )
        self.assertEqual(experience["metadata"]["source"], "runtime")
        self.assertEqual(experience["recovery_summary"]["outcome"]["status"], "PASS")
        self.assertIsNone(experience["recovery_summary"]["outcome"]["verification_result"])
        self.assertEqual(experience["diagnosis_summary"]["failure_cause"]["code"], "INVALID_APPROACH")
        self.assertIsNone(experience["diagnosis_summary"]["source_failure_type"])
        signature = experience["context_signature"]
        self.assertEqual(signature["subgoal_id"], "SG1_s1")
        self.assertEqual(signature["subgoal_description"], "Acquire the ladle")
        self.assertEqual(signature["detail_id"], "SG1_s1_d1")
        self.assertEqual(signature["action_type"], "acquire")
        self.assertEqual(signature["violated_predicates"], [])
        self.assertEqual(
            signature["execution_signature"]["motion_planning_status"],
            "FAILED",
        )
        self.assertNotIn("target_module", experience["recovery_summary"]["action"])
        self.assertEqual(
            experience["recovery_summary"]["action"]["recovery_type"],
            "CHANGE_APPROACH",
        )
        # Initial m5_result must not overwrite diagnosis.
        self.assertNotEqual(
            experience["diagnosis_summary"]["failure_cause"]["code"],
            "IK_FAILURE",
        )

    def test_runtime_parent_detail_mapping_acquire(self):
        experience = build_runtime_experience(
            failure_context=_failure_context(
                subgoal={
                    "subgoal_id": "SG1_s1_d1",
                    "parent_subgoal_id": "SG1_s1",
                    "detail_id": "SG1_s1_d1",
                    "description": "대상 1개를 collection zone으로 쓸어 담는다",
                    "action_type": "acquire",
                }
            ),
            diagnosis=_diagnosis(),
            recovery=_recovery(),
            recovery_result={
                "subgoal_id": "SG1_s1_d1",
                "status": "SUCCESS",
                "phase": "execution",
                "failure_code": None,
                "detail": None,
            },
            experience_id="RT-CASE-A",
        )
        signature = experience["context_signature"]
        self.assertEqual(signature["subgoal_id"], "SG1_s1")
        self.assertEqual(
            signature["subgoal_description"],
            "대상 1개를 collection zone으로 쓸어 담는다",
        )
        self.assertEqual(signature["detail_id"], "SG1_s1_d1")
        self.assertEqual(signature["action_type"], "acquire")

    def test_runtime_parent_detail_mapping_sweep(self):
        experience = build_runtime_experience(
            failure_context=_failure_context(
                subgoal={
                    "subgoal_id": "SG1_s1_d2",
                    "parent_subgoal_id": "SG1_s1",
                    "detail_id": "SG1_s1_d2",
                    "description": "대상 1개를 collection zone으로 쓸어 담는다",
                    "action_type": "tool_act:sweep",
                }
            ),
            diagnosis=_diagnosis(),
            recovery=_recovery(),
            recovery_result={
                "subgoal_id": "SG1_s1_d2",
                "status": "FAIL",
                "phase": "planning",
                "failure_code": None,
                "detail": "still failing",
            },
            experience_id="RT-CASE-B",
        )
        signature = experience["context_signature"]
        self.assertEqual(signature["subgoal_id"], "SG1_s1")
        self.assertEqual(signature["detail_id"], "SG1_s1_d2")
        self.assertEqual(signature["action_type"], "tool_act:sweep")

    def test_recovery_validation_still_uses_detail_execution_id(self):
        context = _failure_context()
        self.assertEqual(context["subgoal"]["subgoal_id"], "SG1_s1_d1")
        experience = build_runtime_experience(
            failure_context=context,
            diagnosis=_diagnosis(),
            recovery=_recovery(),
            recovery_result={
                "subgoal_id": "SG1_s1_d1",
                "status": "SUCCESS",
                "phase": "execution",
                "failure_code": None,
                "detail": None,
            },
            experience_id="RT-CASE-D",
        )
        # Signature stores parent; lifecycle FC still identifies detail.
        self.assertEqual(experience["context_signature"]["subgoal_id"], "SG1_s1")
        self.assertEqual(experience["context_signature"]["detail_id"], "SG1_s1_d1")
        self.assertEqual(context["subgoal"]["subgoal_id"], "SG1_s1_d1")
        with self.assertRaises(RecoveryResultError):
            build_runtime_experience(
                failure_context=context,
                diagnosis=_diagnosis(),
                recovery=_recovery(),
                recovery_result={
                    "subgoal_id": "SG1_s1",
                    "status": "SUCCESS",
                    "phase": "execution",
                    "failure_code": None,
                    "detail": None,
                },
                experience_id="RT-CASE-D-BAD",
            )

    def test_runtime_fail_mapping(self):
        experience = build_runtime_experience(
            failure_context=_failure_context(),
            diagnosis=_diagnosis(),
            recovery=_recovery(),
            recovery_result={
                "subgoal_id": "SG1_s1_d1",
                "status": "FAIL",
                "phase": "planning",
                "failure_code": "IK_FAILURE",
                "detail": "Recovery did not resolve the failure",
            },
            experience_id="RT-TEST-FAIL",
        )
        self.assertEqual(experience["metadata"]["source"], "runtime")
        self.assertEqual(experience["recovery_summary"]["outcome"]["status"], "FAIL")

    def test_does_not_mutate_initial_m5_result(self):
        context = _failure_context()
        before = deepcopy(context["m5_result"])
        RuntimeExperienceBuilder().build(
            failure_context=context,
            diagnosis=_diagnosis(),
            recovery=_recovery(),
            recovery_result={
                "subgoal_id": "SG1_s1_d1",
                "status": "SUCCESS",
                "phase": "execution",
                "failure_code": None,
                "detail": None,
            },
            experience_id="RT-TEST-M5",
        )
        self.assertEqual(context["m5_result"], before)


class ProcessRecoveryOutcomeTests(unittest.TestCase):
    def test_runtime_pass_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.json"
            _write_memory(memory_path, [_offline_seed()])
            before = json.loads(memory_path.read_text(encoding="utf-8"))
            object_before = deepcopy(before["object_knowledge"])
            schema_before = before["schema_version"]
            n_before = len(before["failure_recovery_experience"]["experiences"])

            result = process_recovery_outcome(
                failure_context=_failure_context(),
                diagnosis=_diagnosis(),
                recovery=_recovery(),
                recovery_result={
                    "subgoal_id": "SG1_s1_d1",
                    "status": "SUCCESS",
                    "phase": "execution",
                    "failure_code": None,
                    "detail": None,
                },
                memory_path=memory_path,
            )

            after = json.loads(memory_path.read_text(encoding="utf-8"))
            experiences = after["failure_recovery_experience"]["experiences"]
            self.assertEqual(len(experiences), n_before + 1)
            self.assertEqual(result["outcome"]["status"], "PASS")
            self.assertTrue(result["appended"])
            self.assertEqual(experiences[0]["experience_id"], "OFFLINE-SEED-001")
            self.assertEqual(experiences[-1]["metadata"]["source"], "runtime")
            self.assertEqual(experiences[-1]["recovery_summary"]["outcome"]["status"], "PASS")
            self.assertEqual(experiences[-1]["context_signature"]["subgoal_id"], "SG1_s1")
            self.assertEqual(experiences[-1]["context_signature"]["detail_id"], "SG1_s1_d1")
            self.assertEqual(after["object_knowledge"], object_before)
            self.assertEqual(after["schema_version"], schema_before)

    def test_offline_seed_detail_id_null_and_retrieval_unaffected_by_detail_id(self):
        seed = _offline_seed()
        self.assertIsNone(seed["context_signature"]["detail_id"])
        self.assertIsNone(seed["context_signature"]["subgoal_id"])

        query = build_retrieval_query(_failure_context())
        with_detail = compute_context_similarity(
            query,
            {
                **seed["context_signature"],
                "detail_id": "SG1_s1_d1",
            },
        )
        without_detail = compute_context_similarity(
            query,
            {
                **seed["context_signature"],
                "detail_id": None,
            },
        )
        self.assertEqual(
            with_detail["context_similarity"],
            without_detail["context_similarity"],
        )
        self.assertEqual(
            with_detail["compared_field_count"],
            without_detail["compared_field_count"],
        )
        self.assertNotIn(
            "detail_id",
            with_detail["similarity_breakdown"]["compared_fields"],
        )

    def test_runtime_fail_also_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.json"
            _write_memory(memory_path, [_offline_seed()])
            n_before = 1

            result = process_recovery_outcome(
                failure_context=_failure_context(),
                diagnosis=_diagnosis(),
                recovery=_recovery(),
                recovery_result={
                    "subgoal_id": "SG1_s1_d1",
                    "status": "FAIL",
                    "phase": "planning",
                    "failure_code": "IK_FAILURE",
                    "detail": "Recovery did not resolve the failure",
                },
                memory_path=memory_path,
            )

            after = json.loads(memory_path.read_text(encoding="utf-8"))
            experiences = after["failure_recovery_experience"]["experiences"]
            self.assertEqual(len(experiences), n_before + 1)
            self.assertEqual(result["outcome"]["status"], "FAIL")
            self.assertEqual(experiences[-1]["recovery_summary"]["outcome"]["status"], "FAIL")
            self.assertEqual(experiences[-1]["metadata"]["source"], "runtime")

    def test_offline_seed_and_two_runtime_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.json"
            seeds = [_offline_seed("SEED-A"), _offline_seed("SEED-B")]
            _write_memory(memory_path, seeds)

            process_recovery_outcome(
                failure_context=_failure_context(),
                diagnosis=_diagnosis(),
                recovery=_recovery(),
                recovery_result={
                    "subgoal_id": "SG1_s1_d1",
                    "status": "SUCCESS",
                    "phase": "execution",
                    "failure_code": None,
                    "detail": None,
                },
                memory_path=memory_path,
            )
            process_recovery_outcome(
                failure_context=_failure_context(),
                diagnosis=_diagnosis(),
                recovery=_recovery(),
                recovery_result={
                    "subgoal_id": "SG1_s1_d1",
                    "status": "FAIL",
                    "phase": "planning",
                    "failure_code": "IK_FAILURE",
                    "detail": "still failing",
                },
                memory_path=memory_path,
            )

            after = json.loads(memory_path.read_text(encoding="utf-8"))
            experiences = after["failure_recovery_experience"]["experiences"]
            self.assertEqual(len(experiences), 4)
            self.assertEqual(experiences[0]["metadata"]["source"], "offline_seed")
            self.assertEqual(experiences[1]["metadata"]["source"], "offline_seed")
            sources = [item["metadata"]["source"] for item in experiences[2:]]
            self.assertEqual(sources, ["runtime", "runtime"])
            outcomes = [
                item["recovery_summary"]["outcome"]["status"] for item in experiences[2:]
            ]
            self.assertEqual(outcomes, ["PASS", "FAIL"])

    def test_subgoal_mismatch_does_not_modify_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.json"
            _write_memory(memory_path, [_offline_seed()])
            before = memory_path.read_text(encoding="utf-8")

            with self.assertRaises(RecoveryResultError):
                process_recovery_outcome(
                    failure_context=_failure_context(),
                    diagnosis=_diagnosis(),
                    recovery=_recovery(),
                    recovery_result={
                        "subgoal_id": "SG2_s1_d1",
                        "status": "SUCCESS",
                        "phase": "execution",
                        "failure_code": None,
                        "detail": None,
                    },
                    memory_path=memory_path,
                )

            self.assertEqual(memory_path.read_text(encoding="utf-8"), before)

    def test_invalid_status_does_not_modify_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.json"
            _write_memory(memory_path, [_offline_seed()])
            before = memory_path.read_text(encoding="utf-8")

            with self.assertRaises(RecoveryResultError):
                process_recovery_outcome(
                    failure_context=_failure_context(),
                    diagnosis=_diagnosis(),
                    recovery=_recovery(),
                    recovery_result={
                        "subgoal_id": "SG1_s1_d1",
                        "status": "FAILED",
                        "phase": "planning",
                        "failure_code": "IK_FAILURE",
                        "detail": None,
                    },
                    memory_path=memory_path,
                )

            self.assertEqual(memory_path.read_text(encoding="utf-8"), before)

    def test_no_recovery_result_does_not_create_not_executed_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.json"
            _write_memory(memory_path, [_offline_seed()])
            before = memory_path.read_text(encoding="utf-8")

            with self.assertRaises(RecoveryResultError):
                process_recovery_outcome(
                    failure_context=_failure_context(),
                    diagnosis=_diagnosis(),
                    recovery=_recovery(),
                    recovery_result=None,  # type: ignore[arg-type]
                    memory_path=memory_path,
                )

            after = json.loads(memory_path.read_text(encoding="utf-8"))
            self.assertEqual(memory_path.read_text(encoding="utf-8"), before)
            for item in after["failure_recovery_experience"]["experiences"]:
                self.assertNotEqual(item["metadata"]["source"], "runtime")

    def test_append_rejects_duplicate_experience_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.json"
            _write_memory(memory_path, [_offline_seed("DUP-ID")])
            experience = build_runtime_experience(
                failure_context=_failure_context(),
                diagnosis=_diagnosis(),
                recovery=_recovery(),
                recovery_result={
                    "subgoal_id": "SG1_s1_d1",
                    "status": "SUCCESS",
                    "phase": "execution",
                    "failure_code": None,
                    "detail": None,
                },
                experience_id="DUP-ID",
            )
            with self.assertRaises(MemoryAppendError):
                append_experience(experience, memory_path=memory_path)

    def test_reload_retrieval_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.json"
            _write_memory(memory_path, [_offline_seed()])

            process_recovery_outcome(
                failure_context=_failure_context(),
                diagnosis=_diagnosis(),
                recovery=_recovery(),
                recovery_result={
                    "subgoal_id": "SG1_s1_d1",
                    "status": "SUCCESS",
                    "phase": "execution",
                    "failure_code": None,
                    "detail": None,
                },
                memory_path=memory_path,
            )

            adapter = MemoryAdapter(memory_path=memory_path)
            retrieved = adapter.retrieve_experiences(
                _failure_context(),
                top_k=5,
                similarity_threshold=0.5,
            )
            self.assertGreaterEqual(len(retrieved), 1)
            ids = {
                (item.get("experience") or {}).get("experience_id")
                for item in retrieved
            }
            self.assertTrue(ids)
            # Offline seed and/or runtime should remain readable for ranking.
            loaded = adapter._load_experiences()
            self.assertEqual(len(loaded), 2)
            sources = {item["metadata"]["source"] for item in loaded}
            self.assertEqual(sources, {"offline_seed", "runtime"})


if __name__ == "__main__":
    unittest.main()
