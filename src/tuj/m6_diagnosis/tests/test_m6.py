import json
import tempfile
import unittest
from pathlib import Path

from tuj.m6 import DiagnoseRouter, FailureContextBuilder, MemoryAdapter
from tuj.m6.context_similarity import (
    compute_context_similarity,
    evaluate_all_candidates,
    rank_experiences,
)
from tuj.m6.diagnosis import (
    DIAGNOSIS_EVIDENCE_ALLOWED_KEYS,
    DiagnosisValidationError,
    MockFailureDiagnoser,
    apply_diagnosis_output,
    validate_diagnosis_output,
)
from tuj.m6.diagnosis_aware_selection import (
    DiagnosisAwareExperienceSelector,
    filter_recovery_evidence_by_ids,
)
from tuj.m6.evidence import prepare_diagnosis_evidence, prepare_recovery_evidence
from tuj.m6.recovery_router import (
    MockRecoveryRouter,
    RecoveryValidationError,
    apply_recovery_output,
    resolve_recovery_decision,
    validate_recovery_output,
)
from tuj.m6.retrieval_config import POSSIBLE_FIELD_COUNT, RetrievalConfig
from tuj.m6.retrieval_query import build_retrieval_query
from tuj.m6.schemas import empty_recovery


def _make_failure_context(**overrides) -> dict:
    context = {
        "failure_id": "failure-test",
        "task": {"task_id": "task-1", "instruction": "demo"},
        "subgoal": {
            "subgoal_id": "sg-1",
            "description": None,
            "action_type": None,
            "target_object_ids": [],
            "selected_object_id": None,
            "selected_object_class": None,
            "preconditions": [],
            "postconditions": [],
            "invariants": [],
        },
        "verification": {
            "result": "FAIL",
            "expected_state": [],
            "observed_state": [],
            "violated_predicates": [],
        },
        "scene": {"nodes": [], "relations": [], "object_states": {}},
        "grounding": {
            "physical_properties": {},
            "geometry": {},
            "metric_relations": {},
            "ee_feasibility": {},
            "confidence": {},
        },
        "task_plan": {
            "selected_ee": None,
            "selected_tool": None,
            "ee_candidates": [],
            "selection_score": None,
            "selection_reason": None,
            "final_order": [],
            "swap_plan": [],
        },
        "motion_plan": {
            "keyframes": [],
            "ik_result": None,
            "collision_result": None,
            "approach_pose": None,
            "trajectory": None,
            "planning_status": None,
            "planning_error": None,
        },
        "execution": {
            "executed_actions": [],
            "controller_status": None,
            "gripper": {
                "command": None,
                "position": None,
                "contact_detected": None,
                "force": None,
            },
            "timeout": False,
            "error": None,
        },
        "observation": {
            "before_image": None,
            "after_image": None,
            "before_scene": None,
            "after_scene": None,
        },
        "history": {
            "retry_count": 0,
            "previous_diagnoses": [],
            "previous_recoveries": [],
            "previous_outcomes": [],
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(context.get(key), dict):
            context[key].update(value)
        else:
            context[key] = value
    return context


def _make_experience(
    experience_id: str,
    *,
    subgoal_description=None,
    action_type=None,
    object_class=None,
    object_id=None,
    violated_predicates=None,
    selected_ee=None,
    selected_tool=None,
    motion_planning_status=None,
    controller_status=None,
    failure_type="PLANNING",
    failure_cause_code="INVALID_APPROACH",
    affected_module="M5",
    recovery_category="REPLAN_MOTION",
    recovery_action_type="CHANGE_APPROACH",
    outcome_status="NOT_EXECUTED",
    source="test_fixture",
):
    return {
        "experience_id": experience_id,
        "context_signature": {
            "task_id": None,
            "subgoal_id": None,
            "subgoal_description": subgoal_description,
            "action_type": action_type,
            "target": {"object_id": object_id, "object_class": object_class},
            "violated_predicates": violated_predicates or [],
            "selected_ee": selected_ee,
            "selected_tool": selected_tool,
            "execution_signature": {
                "motion_planning_status": motion_planning_status,
                "controller_status": controller_status,
            },
        },
        "diagnosis_summary": {
            "source_failure_type": "seed",
            "failure_type": failure_type,
            "failure_cause": {"code": failure_cause_code, "description": "seed cause"},
            "affected_module": affected_module,
            "confidence": 0.95,
        },
        "recovery_summary": {
            "recovery_category": recovery_category,
            "action": {"action_type": recovery_action_type},
            "changes": [],
            "routing": {"restart_from": None, "rerun_modules": [], "invalidate": []},
            "outcome": {"status": outcome_status, "verification_result": None},
        },
        "metadata": {"source": source},
    }


def _make_retrieved_item(experience: dict, **overrides) -> dict:
    item = {
        "experience": experience,
        "context_similarity": 1.0,
        "comparison_coverage": 1.0 / POSSIBLE_FIELD_COUNT,
        "compared_field_count": 1,
        "possible_field_count": POSSIBLE_FIELD_COUNT,
        "similarity_breakdown": {
            "matched_fields": [],
            "compared_fields": ["subgoal.action_type"],
            "ignored_fields": [],
            "field_scores": {"subgoal.action_type": 1.0},
        },
    }
    item.update(overrides)
    return item


class RetrievalQueryTests(unittest.TestCase):
    def test_build_retrieval_query_uses_explicit_target_fields_only(self):
        failure_context = _make_failure_context(
            subgoal={
                "description": "Acquire spatula",
                "action_type": "acquire",
                "selected_object_id": "spatula",
            },
            task_plan={"selected_ee": "2F"},
            execution={"controller_status": "SUCCESS"},
        )

        query = build_retrieval_query(failure_context)

        self.assertEqual(query["subgoal_description"], "Acquire spatula")
        self.assertEqual(query["action_type"], "acquire")
        self.assertEqual(query["target"]["object_id"], "spatula")
        self.assertIsNone(query["target"]["object_class"])
        self.assertEqual(query["selected_ee"], "2F")
        self.assertEqual(query["execution_signature"]["controller_status"], "SUCCESS")
        self.assertNotIn("failure_type", query)
        self.assertNotIn("recovery_summary", query)

    def test_selected_object_class_is_passed_to_retrieval_query(self):
        failure_context = _make_failure_context(
            subgoal={
                "selected_object_id": "spatula_03",
                "selected_object_class": "spatula",
            }
        )

        query = build_retrieval_query(failure_context)

        self.assertEqual(query["target"]["object_id"], "spatula_03")
        self.assertEqual(query["target"]["object_class"], "spatula")


class ContextSimilarityTests(unittest.TestCase):
    def test_missing_fields_are_ignored_not_penalized(self):
        query = build_retrieval_query(
            _make_failure_context(
                subgoal={"action_type": "acquire", "selected_object_id": "spatula"},
                task_plan={"selected_ee": "2F"},
            )
        )
        context_signature = {
            "subgoal_description": None,
            "action_type": "acquire",
            "target": {"object_id": None, "object_class": "spatula"},
            "violated_predicates": [],
            "selected_ee": None,
            "selected_tool": None,
            "execution_signature": {
                "motion_planning_status": None,
                "controller_status": None,
            },
        }

        result = compute_context_similarity(query, context_signature)

        self.assertEqual(result["context_similarity"], 1.0)
        self.assertIn("task_plan.selected_ee", result["similarity_breakdown"]["ignored_fields"])
        self.assertIn("target.object_id", result["similarity_breakdown"]["ignored_fields"])
        self.assertIn("target.object_class", result["similarity_breakdown"]["ignored_fields"])
        self.assertIn("subgoal.action_type", result["similarity_breakdown"]["matched_fields"])
        self.assertFalse(result["candidate_valid"])
        self.assertEqual(result["candidate_validity"]["reason"], "action_type_only_match")

    def test_object_id_and_object_class_are_not_cross_compared(self):
        query = build_retrieval_query(
            _make_failure_context(
                subgoal={"selected_object_class": "spatula"},
            )
        )
        context_signature = {
            "target": {"object_id": "spatula_03", "object_class": None},
        }

        result = compute_context_similarity(query, context_signature)

        self.assertIn("target.object_id", result["similarity_breakdown"]["ignored_fields"])
        self.assertIn("target.object_class", result["similarity_breakdown"]["ignored_fields"])
        self.assertEqual(result["similarity_breakdown"]["compared_fields"], [])

    def test_description_differences_do_not_change_similarity(self):
        base_subgoal = {
            "action_type": "acquire",
            "selected_object_class": "spatula",
        }
        query_a = build_retrieval_query(
            _make_failure_context(
                subgoal={**base_subgoal, "description": "Approach the wooden spatula"},
            )
        )
        query_b = build_retrieval_query(
            _make_failure_context(
                subgoal={**base_subgoal, "description": "Completely unrelated text"},
            )
        )
        context_signature = {
            "action_type": "acquire",
            "target": {"object_id": None, "object_class": "spatula"},
        }

        result_a = compute_context_similarity(query_a, context_signature)
        result_b = compute_context_similarity(query_b, context_signature)

        self.assertEqual(result_a, result_b)
        self.assertNotIn("subgoal.description", result_a["similarity_breakdown"]["field_scores"])

    def test_possible_field_count_is_eight(self):
        result = compute_context_similarity(
            build_retrieval_query(
                _make_failure_context(subgoal={"action_type": "acquire"}),
            ),
            {"action_type": "acquire"},
        )

        self.assertEqual(result["possible_field_count"], 8)
        self.assertEqual(POSSIBLE_FIELD_COUNT, 8)

    def test_string_exact_comparison_for_action_type(self):
        query = build_retrieval_query(
            _make_failure_context(subgoal={"action_type": "Acquire"})
        )
        context_signature = {"action_type": "acquire"}

        result = compute_context_similarity(query, context_signature)

        self.assertEqual(result["context_similarity"], 1.0)
        self.assertEqual(
            result["similarity_breakdown"]["field_scores"]["subgoal.action_type"],
            1.0,
        )

    def test_target_object_id_requires_both_sides(self):
        query = build_retrieval_query(
            _make_failure_context(subgoal={"selected_object_id": "obj-001"})
        )
        context_signature = {
            "target": {"object_id": "obj-001", "object_class": None},
        }

        result = compute_context_similarity(query, context_signature)

        self.assertEqual(result["context_similarity"], 1.0)
        self.assertIn("target.object_id", result["similarity_breakdown"]["matched_fields"])

    def test_violated_predicates_use_jaccard(self):
        query = build_retrieval_query(
            _make_failure_context(
                verification={"violated_predicates": ["holding(a)", "aligned(a)"]},
            )
        )
        context_signature = {
            "violated_predicates": ["holding(a)", "closed(gripper)"],
        }

        result = compute_context_similarity(query, context_signature)

        self.assertAlmostEqual(result["context_similarity"], 1.0 / 3.0)
        self.assertEqual(
            result["similarity_breakdown"]["compared_fields"],
            ["verification.violated_predicates"],
        )

    def test_comparison_coverage_is_separate_from_similarity(self):
        query = build_retrieval_query(
            _make_failure_context(
                subgoal={
                    "description": "Acquire spatula",
                    "action_type": "acquire",
                    "selected_object_class": "spatula",
                }
            )
        )
        context_signature = {
            "subgoal_description": "Different description",
            "action_type": "acquire",
            "target": {"object_id": None, "object_class": "spatula"},
        }

        result = compute_context_similarity(query, context_signature)

        self.assertEqual(result["context_similarity"], 1.0)
        self.assertEqual(result["compared_field_count"], 2)
        self.assertEqual(result["possible_field_count"], POSSIBLE_FIELD_COUNT)
        self.assertAlmostEqual(result["comparison_coverage"], 2 / POSSIBLE_FIELD_COUNT)

    def test_no_comparable_fields_yields_zero_similarity(self):
        query = build_retrieval_query(_make_failure_context())
        context_signature = {
            "subgoal_description": None,
            "action_type": None,
            "target": {"object_id": None, "object_class": None},
            "violated_predicates": [],
            "selected_ee": None,
            "selected_tool": None,
            "execution_signature": {
                "motion_planning_status": None,
                "controller_status": None,
            },
        }

        result = compute_context_similarity(query, context_signature)

        self.assertEqual(result["context_similarity"], 0.0)
        self.assertEqual(result["similarity_breakdown"]["compared_fields"], [])
        self.assertEqual(result["compared_field_count"], 0)

    def test_diagnosis_summary_does_not_affect_similarity(self):
        query = build_retrieval_query(
            _make_failure_context(subgoal={"action_type": "acquire", "selected_object_id": "spatula"})
        )
        signature_a = {
            "action_type": "acquire",
            "target": {"object_id": "spatula", "object_class": None},
        }
        signature_b = {
            "action_type": "acquire",
            "target": {"object_id": "spatula", "object_class": None},
        }

        score_a = compute_context_similarity(query, signature_a)["context_similarity"]
        score_b = compute_context_similarity(query, signature_b)["context_similarity"]
        self.assertEqual(score_a, score_b)

    def test_recovery_summary_does_not_affect_similarity(self):
        failure_context = _make_failure_context(
            subgoal={"action_type": "acquire", "selected_object_id": "spatula"},
        )
        experiences = [
            _make_experience(
                "exp-a",
                action_type="acquire",
                object_id="spatula",
                recovery_category="REPLAN_MOTION",
            ),
            _make_experience(
                "exp-b",
                action_type="acquire",
                object_id="spatula",
                recovery_category="RETRY_EXECUTION",
                recovery_action_type="RETRY_ACTION",
            ),
        ]

        ranked = rank_experiences(
            failure_context,
            experiences,
            similarity_threshold=0.0,
            top_k=2,
        )

        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["context_similarity"], ranked[1]["context_similarity"])

    def test_similarity_sorting_and_coverage_tie_breaker(self):
        failure_context = _make_failure_context(
            subgoal={
                "description": "Acquire spatula",
                "action_type": "acquire",
                "selected_object_class": "spatula",
            },
            task_plan={"selected_ee": "2F"},
        )
        experiences = [
            _make_experience(
                "exp-low-coverage",
                action_type="acquire",
                object_class="spatula",
            ),
            _make_experience(
                "exp-high-coverage",
                subgoal_description="Different description",
                action_type="acquire",
                object_class="spatula",
                selected_ee="2F",
            ),
        ]

        ranked = rank_experiences(
            failure_context,
            experiences,
            similarity_threshold=0.0,
            top_k=2,
        )

        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["experience"]["experience_id"], "exp-high-coverage")
        self.assertGreater(
            ranked[0]["compared_field_count"],
            ranked[1]["compared_field_count"],
        )


class CandidateValidityTests(unittest.TestCase):
    def test_action_type_only_match_is_rejected_from_top_k(self):
        failure_context = _make_failure_context(
            subgoal={"action_type": "acquire", "description": "Acquire the door handle"},
        )
        experiences = [
            _make_experience(
                "exp-door",
                subgoal_description="Grasp the door handle",
                action_type="acquire",
                object_class="door handle",
            )
        ]

        ranked = rank_experiences(
            failure_context,
            experiences,
            config=RetrievalConfig(similarity_threshold=0.5),
        )

        self.assertEqual(ranked, [])

    def test_description_only_match_does_not_retrieve_candidate(self):
        failure_context = _make_failure_context(
            subgoal={
                "description": "Move the left arm to approach and align the wooden spatula.",
                "action_type": "acquire",
            },
        )
        experiences = [
            _make_experience(
                "exp-description-only",
                subgoal_description="Move the left arm to approach and align the wooden spatula.",
                action_type="place",
                object_class="mug",
            )
        ]

        ranked = rank_experiences(
            failure_context,
            experiences,
            config=RetrievalConfig(similarity_threshold=0.5),
        )

        self.assertEqual(ranked, [])

    def test_action_type_plus_target_object_class_is_valid(self):
        failure_context = _make_failure_context(
            subgoal={
                "action_type": "acquire",
                "selected_object_class": "spatula",
            },
        )
        experiences = [
            _make_experience(
                "exp-spatula",
                action_type="acquire",
                object_class="spatula",
            )
        ]

        ranked = rank_experiences(
            failure_context,
            experiences,
            config=RetrievalConfig(similarity_threshold=0.5),
        )

        self.assertEqual(len(ranked), 1)
        self.assertTrue(ranked[0]["candidate_valid"])
        self.assertEqual(ranked[0]["candidate_validity"]["reason"], "context_specific_evidence_present")

    def test_target_object_class_mismatch_is_invalid(self):
        query = build_retrieval_query(
            _make_failure_context(
                subgoal={
                    "action_type": "acquire",
                    "selected_object_class": "spatula",
                    "description": "Approach the wooden spatula",
                }
            )
        )
        context_signature = {
            "subgoal_description": "Move the left arm to approach and align the spoon.",
            "action_type": "acquire",
            "target": {"object_id": None, "object_class": "spoon"},
        }

        result = compute_context_similarity(query, context_signature)

        self.assertFalse(result["candidate_valid"])
        self.assertEqual(result["candidate_validity"]["reason"], "target.object_class_mismatch")

    def test_candidate_validity_is_available_in_candidate_audit(self):
        failure_context = _make_failure_context(
            subgoal={"action_type": "acquire", "description": "Acquire the door handle"},
        )
        experiences = [
            _make_experience(
                "exp-door",
                subgoal_description="Grasp the door handle",
                action_type="acquire",
                object_class="door handle",
            )
        ]

        candidates = evaluate_all_candidates(
            failure_context,
            experiences,
            config=RetrievalConfig(similarity_threshold=0.5),
        )

        self.assertEqual(len(candidates), 1)
        self.assertFalse(candidates[0]["candidate_valid"])
        self.assertEqual(candidates[0]["candidate_validity"]["reason"], "action_type_only_match")


class DiagnosisTests(unittest.TestCase):
    def test_valid_canonical_diagnosis_passes(self):
        diagnosis_output = {
            "failure_type": "PLANNING",
            "failure_cause": {
                "code": "INVALID_APPROACH",
                "description": "valid diagnosis",
            },
            "affected_module": "M5",
            "evidence": ["motion plan invalid"],
            "confidence": 0.9,
        }

        validate_diagnosis_output(diagnosis_output)

    def test_invalid_failure_type_is_rejected(self):
        with self.assertRaises(DiagnosisValidationError):
            validate_diagnosis_output(
                {
                    "failure_type": "UNKNOWN_TYPE",
                    "failure_cause": {"code": "INVALID_APPROACH"},
                    "affected_module": "M5",
                    "evidence": [],
                    "confidence": 0.5,
                }
            )

    def test_invalid_failure_cause_is_rejected(self):
        with self.assertRaises(DiagnosisValidationError):
            validate_diagnosis_output(
                {
                    "failure_type": "PLANNING",
                    "failure_cause": {"code": "GRASP_FAILURE"},
                    "affected_module": "M5",
                    "evidence": [],
                    "confidence": 0.5,
                }
            )

    def test_failure_type_and_affected_module_mismatch_is_rejected(self):
        with self.assertRaises(DiagnosisValidationError):
            validate_diagnosis_output(
                {
                    "failure_type": "PLANNING",
                    "failure_cause": {"code": "INVALID_APPROACH"},
                    "affected_module": "Controller",
                    "evidence": [],
                    "confidence": 0.5,
                }
            )

    def test_confidence_range_is_validated(self):
        with self.assertRaises(DiagnosisValidationError):
            validate_diagnosis_output(
                {
                    "failure_type": "PLANNING",
                    "failure_cause": {"code": "INVALID_APPROACH"},
                    "affected_module": "M5",
                    "evidence": [],
                    "confidence": 1.5,
                }
            )

    def test_evidence_must_be_list(self):
        with self.assertRaises(DiagnosisValidationError):
            validate_diagnosis_output(
                {
                    "failure_type": "PLANNING",
                    "failure_cause": {"code": "INVALID_APPROACH"},
                    "affected_module": "M5",
                    "evidence": "not-a-list",
                    "confidence": 0.5,
                }
            )

    def test_diagnosis_evidence_excludes_recovery_information(self):
        experience = _make_experience("exp-1", action_type="acquire", object_class="spatula")
        evidence = prepare_diagnosis_evidence([_make_retrieved_item(experience)])[0]

        self.assertEqual(set(evidence.keys()), DIAGNOSIS_EVIDENCE_ALLOWED_KEYS)
        self.assertNotIn("past_recovery", evidence)
        self.assertNotIn("recovery_summary", evidence)

    def test_diagnosis_runs_without_retrieval_results(self):
        failure_context = _make_failure_context(
            subgoal={"action_type": "acquire", "selected_object_class": "card"},
        )

        result = MockFailureDiagnoser().diagnose(failure_context, [])

        validate_diagnosis_output(result)
        self.assertEqual(result["failure_type"], "PLANNING")

    def test_mock_diagnoser_does_not_copy_past_diagnosis_from_evidence(self):
        failure_context = _make_failure_context(subgoal={"action_type": "acquire"})
        evidence = [
            {
                "experience_id": "exp-past",
                "retrieval": {"context_similarity": 1.0},
                "past_context": {},
                "past_diagnosis": {
                    "failure_type": "EXECUTION_CONTROL",
                    "failure_cause": {
                        "code": "GRASP_FAILURE",
                        "description": "past only",
                    },
                    "affected_module": "Controller",
                    "confidence": 0.99,
                },
                "source": "offline_seed",
            }
        ]

        result = MockFailureDiagnoser().diagnose(failure_context, evidence)

        self.assertEqual(result["failure_type"], "PLANNING")
        self.assertEqual(result["failure_cause"]["code"], "INVALID_APPROACH")
        self.assertEqual(result["affected_module"], "M5")

    def test_router_applies_validated_diagnosis(self):
        result = DiagnoseRouter(failure_diagnoser=MockFailureDiagnoser()).run(
            {"verification": {"result": "FAIL"}}
        )

        self.assertEqual(result["diagnosis"]["failure_type"], "PLANNING")
        self.assertEqual(result["diagnosis"]["failure_cause"]["code"], "INVALID_APPROACH")
        self.assertEqual(result["diagnosis"]["affected_module"], "M5")
        self.assertEqual(result["diagnosis"]["evidence"], ["mock evidence"])
        self.assertEqual(result["diagnosis"]["confidence"], 0.9)


def _make_current_diagnosis(**overrides) -> dict:
    diagnosis = {
        "failure_type": "PLANNING",
        "failure_cause": {"code": "INVALID_APPROACH", "description": "current"},
        "affected_module": "M5",
        "evidence": [],
        "confidence": 0.9,
        "memory_context": {"retrieved_experiences": [], "diagnosis_evidence": []},
    }
    diagnosis.update(overrides)
    return diagnosis


class DiagnosisAwareSelectionTests(unittest.TestCase):
    def setUp(self):
        self.selector = DiagnosisAwareExperienceSelector()

    def test_exact_diagnosis_match_is_selected(self):
        retrieved = [
            _make_retrieved_item(
                _make_experience(
                    "exp-match",
                    failure_type="PLANNING",
                    failure_cause_code="INVALID_APPROACH",
                    affected_module="M5",
                )
            )
        ]

        result = self.selector.select(_make_current_diagnosis(), retrieved, [])

        self.assertEqual(result["selected_experience_ids"], ["exp-match"])
        self.assertEqual(result["selection_count"], 1)
        self.assertTrue(result["selection_audit"][0]["candidate_relevant"])

    def test_same_context_but_different_failure_cause_is_rejected(self):
        retrieved = [
            _make_retrieved_item(
                _make_experience(
                    "exp-match",
                    failure_type="PLANNING",
                    failure_cause_code="INVALID_APPROACH",
                    affected_module="M5",
                )
            ),
            _make_retrieved_item(
                _make_experience(
                    "exp-diff-cause",
                    failure_type="PLANNING",
                    failure_cause_code="COLLISION",
                    affected_module="M5",
                )
            ),
        ]

        result = self.selector.select(_make_current_diagnosis(), retrieved, [])

        self.assertEqual(result["selected_experience_ids"], ["exp-match"])
        rejected = next(
            entry for entry in result["selection_audit"] if entry["experience_id"] == "exp-diff-cause"
        )
        self.assertFalse(rejected["candidate_relevant"])
        self.assertEqual(rejected["relevance_reason"], "partial_diagnosis_match")

    def test_same_failure_type_but_different_cause_is_not_selected(self):
        retrieved = [
            _make_retrieved_item(
                _make_experience(
                    "exp-partial",
                    failure_type="PLANNING",
                    failure_cause_code="COLLISION",
                    affected_module="M5",
                )
            )
        ]

        result = self.selector.select(_make_current_diagnosis(), retrieved, [])

        self.assertEqual(result["selected_experience_ids"], [])
        self.assertEqual(result["selection_audit"][0]["relevance_reason"], "partial_diagnosis_match")

    def test_same_cause_but_inconsistent_affected_module_is_not_selected(self):
        retrieved = [
            _make_retrieved_item(
                _make_experience(
                    "exp-partial-module",
                    failure_type="PLANNING",
                    failure_cause_code="INVALID_APPROACH",
                    affected_module="Controller",
                )
            )
        ]

        result = self.selector.select(_make_current_diagnosis(), retrieved, [])

        self.assertEqual(result["selected_experience_ids"], [])
        self.assertEqual(result["selection_audit"][0]["relevance_reason"], "partial_diagnosis_match")

    def test_missing_past_diagnosis_fields_are_rejected_without_inference(self):
        experience = _make_experience("exp-missing")
        experience["diagnosis_summary"] = {
            "failure_type": "PLANNING",
            "failure_cause": {"code": None, "description": None},
            "affected_module": "M5",
        }
        retrieved = [_make_retrieved_item(experience)]

        result = self.selector.select(_make_current_diagnosis(), retrieved, [])

        self.assertEqual(result["selected_experience_ids"], [])
        self.assertEqual(
            result["selection_audit"][0]["relevance_reason"],
            "insufficient_diagnosis_information",
        )

    def test_empty_retrieval_produces_empty_selection(self):
        result = self.selector.select(_make_current_diagnosis(), [], [])

        self.assertEqual(result["selected_experience_ids"], [])
        self.assertEqual(result["selection_count"], 0)
        self.assertEqual(result["selection_audit"], [])

    def test_multiple_exact_diagnosis_matches_are_all_selected(self):
        retrieved = [
            _make_retrieved_item(
                _make_experience(
                    "exp-a",
                    failure_type="PLANNING",
                    failure_cause_code="INVALID_APPROACH",
                    affected_module="M5",
                )
            ),
            _make_retrieved_item(
                _make_experience(
                    "exp-b",
                    failure_type="PLANNING",
                    failure_cause_code="INVALID_APPROACH",
                    affected_module="M5",
                )
            ),
        ]

        result = self.selector.select(_make_current_diagnosis(), retrieved, [])

        self.assertEqual(sorted(result["selected_experience_ids"]), ["exp-a", "exp-b"])
        self.assertEqual(result["selection_count"], 2)

    def test_recovery_evidence_is_filtered_by_selected_experience_ids(self):
        retrieved = [
            _make_retrieved_item(
                _make_experience(
                    "exp-match",
                    failure_type="PLANNING",
                    failure_cause_code="INVALID_APPROACH",
                    affected_module="M5",
                )
            ),
            _make_retrieved_item(
                _make_experience(
                    "exp-reject",
                    failure_type="EXECUTION_CONTROL",
                    failure_cause_code="GRASP_FAILURE",
                    affected_module="Controller",
                )
            ),
        ]
        recovery_evidence = prepare_recovery_evidence(retrieved)

        result = self.selector.select(_make_current_diagnosis(), retrieved, recovery_evidence)

        self.assertEqual(result["selected_experience_ids"], ["exp-match"])
        self.assertEqual(
            [item["experience_id"] for item in result["selected_recovery_evidence"]],
            ["exp-match"],
        )
        self.assertEqual(
            filter_recovery_evidence_by_ids(recovery_evidence, result["selected_experience_ids"]),
            result["selected_recovery_evidence"],
        )

    def test_context_retrieval_without_diagnosis_match_is_diagnosis_guided(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            memory_path.write_text(
                json.dumps(
                    {
                        "failure_recovery_experience": {
                            "experiences": [
                                _make_experience(
                                    "exp-mismatch",
                                    action_type="acquire",
                                    object_class="spatula",
                                    failure_type="EXECUTION_CONTROL",
                                    failure_cause_code="GRASP_FAILURE",
                                    affected_module="Controller",
                                )
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            adapter = MemoryAdapter(
                memory_path=memory_path,
                config=RetrievalConfig(similarity_threshold=0.5),
            )
            pipeline_state = {
                "subgoal": {
                    "action_type": "acquire",
                    "selected_object_class": "spatula",
                },
                "verification": {"result": "FAIL"},
            }
            result = DiagnoseRouter(memory_adapter=adapter).run(pipeline_state)

        self.assertGreater(len(result["diagnosis"]["memory_context"]["retrieved_experiences"]), 0)
        self.assertEqual(result["recovery"]["decision_mode"], "DIAGNOSIS_GUIDED")
        self.assertEqual(result["recovery"]["guidance"]["experience_ids"], [])

    def test_diagnosis_match_sets_experience_guided(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            memory_path.write_text(
                json.dumps(
                    {
                        "failure_recovery_experience": {
                            "experiences": [
                                _make_experience(
                                    "exp-match",
                                    action_type="acquire",
                                    object_class="spatula",
                                    failure_type="PLANNING",
                                    failure_cause_code="INVALID_APPROACH",
                                    affected_module="M5",
                                )
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            adapter = MemoryAdapter(
                memory_path=memory_path,
                config=RetrievalConfig(similarity_threshold=0.5),
            )
            pipeline_state = {
                "subgoal": {
                    "action_type": "acquire",
                    "selected_object_class": "spatula",
                },
                "verification": {"result": "FAIL"},
            }
            result = DiagnoseRouter(memory_adapter=adapter).run(pipeline_state)

        self.assertEqual(result["recovery"]["decision_mode"], "EXPERIENCE_GUIDED")
        self.assertEqual(result["recovery"]["guidance"]["experience_ids"], ["exp-match"])

    def test_selection_does_not_modify_context_similarity(self):
        retrieved = [
            _make_retrieved_item(
                _make_experience(
                    "exp-match",
                    failure_type="PLANNING",
                    failure_cause_code="INVALID_APPROACH",
                    affected_module="M5",
                ),
                context_similarity=0.75,
                comparison_coverage=0.25,
            )
        ]
        before = json.loads(json.dumps(retrieved))

        self.selector.select(_make_current_diagnosis(), retrieved, [])

        self.assertEqual(retrieved[0]["context_similarity"], before[0]["context_similarity"])
        self.assertEqual(retrieved[0]["comparison_coverage"], before[0]["comparison_coverage"])

    def test_output_memory_json_is_not_modified_by_router(self):
        from hashlib import sha256

        from tuj.m6.memory_adapter import DEFAULT_MEMORY_PATH

        if not DEFAULT_MEMORY_PATH.exists():
            self.skipTest("output/memory.json is not available in this environment")

        before_digest = sha256(DEFAULT_MEMORY_PATH.read_bytes()).hexdigest()
        DiagnoseRouter().run({"verification": {"result": "FAIL"}})
        after_digest = sha256(DEFAULT_MEMORY_PATH.read_bytes()).hexdigest()

        self.assertEqual(before_digest, after_digest)


MOCK_SIMILAR_PIPELINE_STATE = {
    "failure_id": "failure-test-a-001",
    "task": {"task_id": "task-spatula-acquire", "instruction": "Acquire the wooden spatula."},
    "subgoal": {
        "subgoal_id": "sg-spatula-acquire",
        "description": "Approach the wooden spatula and get ready to grasp it.",
        "action_type": "acquire",
        "target_object_ids": ["spatula_03"],
        "selected_object_id": "spatula_03",
        "selected_object_class": "spatula",
        "postconditions": ["holding(spatula)"],
    },
    "verification": {
        "result": "FAIL",
        "expected_state": ["aligned(spatula)"],
        "observed_state": ["misaligned(spatula)"],
        "violated_predicates": [],
    },
    "task_plan": {"selected_ee": "2F", "selected_tool": None},
    "motion_plan": {"planning_status": None},
    "execution": {"controller_status": None},
}

MOCK_DISSIMILAR_PIPELINE_STATE = {
    "failure_id": "failure-test-b-001",
    "task": {"task_id": "task-card-extract", "instruction": "Extract the card from the slot."},
    "subgoal": {
        "subgoal_id": "sg-card-extract",
        "description": "Extract the loyalty card from the dispenser slot using the tool.",
        "action_type": "tool_act:extract",
        "target_object_ids": ["card"],
        "selected_object_id": "card",
        "selected_object_class": "card",
        "postconditions": ["holding(card)"],
    },
    "verification": {
        "result": "FAIL",
        "expected_state": ["holding(card)"],
        "observed_state": ["not_holding(card)"],
        "violated_predicates": ["holding(card)"],
    },
    "motion_plan": {
        "planning_status": "SUCCESS",
        "planning_error": None,
    },
    "execution": {
        "executed_actions": ["approach_target", "close_gripper"],
        "controller_status": "SUCCESS",
        "gripper": {
            "command": "close",
            "position": 0.0,
            "contact_detected": False,
            "force": 0.0,
        },
        "timeout": False,
        "error": None,
    },
}


class TestBContextTests(unittest.TestCase):
    def test_test_b_pipeline_state_expresses_grasp_failure_signals(self):
        self.assertEqual(
            MOCK_DISSIMILAR_PIPELINE_STATE["motion_plan"]["planning_status"],
            "SUCCESS",
        )
        execution = MOCK_DISSIMILAR_PIPELINE_STATE["execution"]
        self.assertEqual(execution["controller_status"], "SUCCESS")
        self.assertIn("close_gripper", execution["executed_actions"])
        self.assertFalse(execution["gripper"]["contact_detected"])
        self.assertEqual(execution["gripper"]["force"], 0.0)

        verification = MOCK_DISSIMILAR_PIPELINE_STATE["verification"]
        self.assertEqual(verification["result"], "FAIL")
        self.assertIn("holding(card)", verification["violated_predicates"])

    def test_failure_context_builder_preserves_test_b_execution_signals(self):
        context = FailureContextBuilder().build(MOCK_DISSIMILAR_PIPELINE_STATE)

        self.assertEqual(context["motion_plan"]["planning_status"], "SUCCESS")
        self.assertEqual(context["execution"]["controller_status"], "SUCCESS")
        self.assertIn("close_gripper", context["execution"]["executed_actions"])
        self.assertFalse(context["execution"]["gripper"]["contact_detected"])
        self.assertEqual(context["execution"]["gripper"]["force"], 0.0)
        self.assertEqual(context["verification"]["result"], "FAIL")
        self.assertIn("holding(card)", context["verification"]["violated_predicates"])


def _make_valid_recovery(**overrides) -> dict:
    recovery = empty_recovery()
    recovery.update(
        {
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
            "recovery_category": "REPLAN_MOTION",
            "action": {
                "action_type": "CHANGE_APPROACH",
                "target_module": "M5",
                "target": {
                    "subgoal_id": "sg-1",
                    "object_id": "obj-1",
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
            "outcome": {"status": None, "verification_result": None},
            "metadata": {"attempt": 1, "created_at": None},
        }
    )
    recovery.update(overrides)
    return recovery


class RecoveryRouterTests(unittest.TestCase):
    def test_valid_canonical_recovery_passes_validation(self):
        validate_recovery_output(_make_valid_recovery())

    def test_invalid_recovery_category_rejected(self):
        with self.assertRaises(RecoveryValidationError):
            validate_recovery_output(_make_valid_recovery(recovery_category="UNKNOWN_CATEGORY"))

    def test_invalid_action_type_rejected(self):
        recovery = _make_valid_recovery()
        recovery["action"]["action_type"] = "UNKNOWN_ACTION"
        with self.assertRaises(RecoveryValidationError):
            validate_recovery_output(recovery)

    def test_action_type_not_belonging_to_recovery_category_rejected(self):
        recovery = _make_valid_recovery(
            recovery_category="REPLAN_MOTION",
            action={
                "action_type": "RESELECT_EE",
                "target_module": "M4",
                "target": _make_valid_recovery()["action"]["target"],
                "parameters": {},
            },
            routing={"restart_from": "M4", "rerun_modules": ["M4"], "invalidate": []},
        )
        with self.assertRaises(RecoveryValidationError):
            validate_recovery_output(recovery)

    def test_target_module_mismatch_rejected(self):
        recovery = _make_valid_recovery()
        recovery["action"]["target_module"] = "M2"
        with self.assertRaises(RecoveryValidationError):
            validate_recovery_output(recovery)

    def test_rerun_modules_must_be_list(self):
        recovery = _make_valid_recovery()
        recovery["routing"]["rerun_modules"] = "M5"
        with self.assertRaises(RecoveryValidationError):
            validate_recovery_output(recovery)

    def test_clearly_inconsistent_routing_rejected(self):
        recovery = _make_valid_recovery(
            routing={"restart_from": "M5", "rerun_modules": ["M2"], "invalidate": []}
        )
        with self.assertRaises(RecoveryValidationError):
            validate_recovery_output(recovery)

    def test_invalid_decision_mode_rejected(self):
        with self.assertRaises(RecoveryValidationError):
            validate_recovery_output(_make_valid_recovery(decision_mode="UNKNOWN_MODE"))

    def test_experience_guided_preserves_selected_experience_ids(self):
        evidence = prepare_recovery_evidence(
            [
                _make_retrieved_item(
                    _make_experience(
                        "exp-match",
                        failure_type="PLANNING",
                        failure_cause_code="INVALID_APPROACH",
                        affected_module="M5",
                    )
                )
            ]
        )
        recovery = _make_valid_recovery(
            decision_mode="EXPERIENCE_GUIDED",
            guidance={
                "experience_ids": ["exp-match"],
                "past_recoveries": [],
                "recovery_evidence": evidence,
                "selection": {
                    "selected_experience_ids": ["exp-match"],
                    "selection_count": 1,
                    "selection_audit": [],
                },
            },
        )
        validate_recovery_output(recovery)

    def test_experience_guided_uses_only_filtered_recovery_evidence(self):
        selected = _make_retrieved_item(
            _make_experience(
                "exp-match",
                failure_type="PLANNING",
                failure_cause_code="INVALID_APPROACH",
                affected_module="M5",
            )
        )
        rejected = _make_retrieved_item(
            _make_experience(
                "exp-reject",
                failure_type="EXECUTION_CONTROL",
                failure_cause_code="GRASP_FAILURE",
                affected_module="Controller",
            )
        )
        all_evidence = prepare_recovery_evidence([selected, rejected])
        filtered = filter_recovery_evidence_by_ids(all_evidence, ["exp-match"])

        result = DiagnoseRouter(
            memory_adapter=_StubRetrievalAdapter([selected, rejected]),
        ).run({"verification": {"result": "FAIL"}})

        self.assertEqual(result["recovery"]["decision_mode"], "EXPERIENCE_GUIDED")
        self.assertEqual(result["recovery"]["guidance"]["experience_ids"], ["exp-match"])
        self.assertEqual(
            [item["experience_id"] for item in result["recovery"]["guidance"]["recovery_evidence"]],
            ["exp-match"],
        )
        self.assertEqual(result["recovery"]["guidance"]["recovery_evidence"], filtered)

    def test_diagnosis_guided_works_with_empty_recovery_evidence(self):
        result = MockRecoveryRouter().route(
            _make_failure_context(),
            {
                "failure_type": "PLANNING",
                "failure_cause": {"code": "INVALID_APPROACH"},
                "affected_module": "M5",
            },
            "DIAGNOSIS_GUIDED",
            [],
        )

        recovery = empty_recovery()
        recovery["decision_mode"] = "DIAGNOSIS_GUIDED"
        recovery["guidance"]["experience_ids"] = []
        recovery["guidance"]["recovery_evidence"] = []
        apply_recovery_output(recovery, result)

        self.assertEqual(recovery["recovery_category"], "REPLAN_MOTION")
        self.assertEqual(recovery["action"]["action_type"], "CHANGE_APPROACH")
        self.assertEqual(recovery["guidance"]["past_recoveries"], [])

    def test_invalid_approach_maps_to_replan_motion_change_approach_m5(self):
        category, action_type, target_module = resolve_recovery_decision(
            "PLANNING",
            "INVALID_APPROACH",
        )

        self.assertEqual(category, "REPLAN_MOTION")
        self.assertEqual(action_type, "CHANGE_APPROACH")
        self.assertEqual(target_module, "M5")

    def test_test_a_results_in_experience_guided_recovery(self):
        from tuj.m6.memory_adapter import DEFAULT_MEMORY_PATH

        if not DEFAULT_MEMORY_PATH.exists():
            self.skipTest("output/memory.json is not available in this environment")

        result = DiagnoseRouter(
            memory_adapter=MemoryAdapter(
                config=RetrievalConfig.create_default(similarity_threshold=0.5, top_k=3)
            )
        ).run(MOCK_SIMILAR_PIPELINE_STATE, top_k=3)

        self.assertEqual(result["recovery"]["decision_mode"], "EXPERIENCE_GUIDED")
        self.assertEqual(result["recovery"]["guidance"]["experience_ids"], ["VF-SPAT-007"])
        self.assertEqual(result["recovery"]["recovery_category"], "REPLAN_MOTION")
        self.assertEqual(result["recovery"]["action"]["action_type"], "CHANGE_APPROACH")
        self.assertEqual(result["recovery"]["action"]["target_module"], "M5")
        self.assertEqual(result["recovery"]["routing"]["restart_from"], "M5")
        self.assertEqual(result["recovery"]["routing"]["rerun_modules"], ["M5"])

    def test_test_b_results_in_diagnosis_guided_recovery(self):
        from tuj.m6.memory_adapter import DEFAULT_MEMORY_PATH

        if not DEFAULT_MEMORY_PATH.exists():
            self.skipTest("output/memory.json is not available in this environment")

        result = DiagnoseRouter(
            memory_adapter=MemoryAdapter(
                config=RetrievalConfig.create_default(similarity_threshold=0.5, top_k=3)
            )
        ).run(MOCK_DISSIMILAR_PIPELINE_STATE, top_k=3)

        self.assertEqual(result["recovery"]["decision_mode"], "DIAGNOSIS_GUIDED")
        self.assertEqual(result["recovery"]["guidance"]["experience_ids"], [])
        self.assertEqual(result["recovery"]["recovery_category"], "REPLAN_MOTION")
        self.assertEqual(result["recovery"]["action"]["action_type"], "CHANGE_APPROACH")
        self.assertEqual(result["recovery"]["action"]["target_module"], "M5")

    def test_recovery_router_does_not_modify_retrieval_similarity_data(self):
        retrieved = [
            _make_retrieved_item(
                _make_experience(
                    "exp-match",
                    action_type="acquire",
                    object_class="spatula",
                    failure_type="PLANNING",
                    failure_cause_code="INVALID_APPROACH",
                    affected_module="M5",
                ),
                context_similarity=0.75,
                comparison_coverage=0.25,
            )
        ]
        before = json.loads(json.dumps(retrieved))

        DiagnoseRouter(memory_adapter=_StubRetrievalAdapter(retrieved)).run(
            {
                "subgoal": {"action_type": "acquire", "selected_object_class": "spatula"},
                "verification": {"result": "FAIL"},
            }
        )

        self.assertEqual(retrieved[0]["context_similarity"], before[0]["context_similarity"])
        self.assertEqual(retrieved[0]["comparison_coverage"], before[0]["comparison_coverage"])

    def test_recovery_router_does_not_modify_diagnosis_result(self):
        retrieved = [
            _make_retrieved_item(
                _make_experience(
                    "exp-match",
                    action_type="acquire",
                    object_class="spatula",
                    failure_type="PLANNING",
                    failure_cause_code="INVALID_APPROACH",
                    affected_module="M5",
                )
            )
        ]
        expected_diagnosis = {
            "failure_type": "PLANNING",
            "failure_cause": {"code": "INVALID_APPROACH", "description": "Mock diagnosis for standalone test."},
            "affected_module": "M5",
            "evidence": ["mock evidence"],
            "confidence": 0.9,
        }

        result = DiagnoseRouter(memory_adapter=_StubRetrievalAdapter(retrieved)).run(
            {"verification": {"result": "FAIL"}}
        )

        self.assertEqual(result["diagnosis"]["failure_type"], expected_diagnosis["failure_type"])
        self.assertEqual(
            result["diagnosis"]["failure_cause"]["code"],
            expected_diagnosis["failure_cause"]["code"],
        )
        self.assertEqual(result["diagnosis"]["affected_module"], expected_diagnosis["affected_module"])
        self.assertEqual(result["diagnosis"]["evidence"], expected_diagnosis["evidence"])
        self.assertEqual(result["diagnosis"]["confidence"], expected_diagnosis["confidence"])

    def test_output_memory_json_is_not_modified_by_recovery_router(self):
        from hashlib import sha256

        from tuj.m6.memory_adapter import DEFAULT_MEMORY_PATH

        if not DEFAULT_MEMORY_PATH.exists():
            self.skipTest("output/memory.json is not available in this environment")

        before_digest = sha256(DEFAULT_MEMORY_PATH.read_bytes()).hexdigest()
        DiagnoseRouter().run(MOCK_SIMILAR_PIPELINE_STATE, top_k=3)
        after_digest = sha256(DEFAULT_MEMORY_PATH.read_bytes()).hexdigest()

        self.assertEqual(before_digest, after_digest)


class _StubRetrievalAdapter:
    def __init__(self, retrieved):
        self._retrieved = retrieved

    def retrieve_experiences(self, failure_context, top_k=3):
        return list(self._retrieved)


class EvidenceTests(unittest.TestCase):
    def test_diagnosis_evidence_excludes_recovery_information(self):
        experience = _make_experience("exp-1", action_type="acquire", object_class="spatula")
        retrieved = [_make_retrieved_item(experience)]

        evidence = prepare_diagnosis_evidence(retrieved)[0]

        self.assertEqual(evidence["experience_id"], "exp-1")
        self.assertIn("past_context", evidence)
        self.assertIn("past_diagnosis", evidence)
        self.assertNotIn("past_recovery", evidence)
        self.assertNotIn("recovery_summary", evidence)
        self.assertNotIn("recovery_category", evidence)

    def test_recovery_evidence_preserves_recovery_source_and_outcome(self):
        experience = _make_experience(
            "exp-2",
            action_type="acquire",
            object_class="spatula",
            recovery_category="REPLAN_MOTION",
            recovery_action_type="CHANGE_APPROACH",
            outcome_status="NOT_EXECUTED",
            source="offline_seed",
        )
        retrieved = [_make_retrieved_item(experience)]

        evidence = prepare_recovery_evidence(retrieved)[0]

        self.assertEqual(evidence["experience_id"], "exp-2")
        self.assertEqual(evidence["source"], "offline_seed")
        self.assertEqual(evidence["past_recovery"]["recovery_category"], "REPLAN_MOTION")
        self.assertEqual(evidence["past_recovery"]["action"]["action_type"], "CHANGE_APPROACH")
        self.assertEqual(evidence["outcome"]["status"], "NOT_EXECUTED")


class MemoryAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.memory_path = Path(self.temp_dir.name) / "memory.json"
        self.experiences = [
            _make_experience(
                "exp-similar",
                subgoal_description="Acquire the spatula",
                action_type="acquire",
                object_class="spatula",
            ),
            _make_experience(
                "exp-partial",
                action_type="acquire",
                object_class="spatula",
            ),
            _make_experience(
                "exp-different",
                action_type="place",
                object_class="mug",
            ),
        ]
        self.memory_path.write_text(
            json.dumps({"failure_recovery_experience": {"experiences": self.experiences}}),
            encoding="utf-8",
        )
        self.adapter = MemoryAdapter(
            memory_path=self.memory_path,
            config=RetrievalConfig(similarity_threshold=0.5),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_similar_context_is_retrieved(self):
        failure_context = _make_failure_context(
            subgoal={
                "description": "Acquire the spatula",
                "action_type": "acquire",
                "selected_object_id": "spatula_03",
                "selected_object_class": "spatula",
            }
        )

        retrieved = self.adapter.retrieve_experiences(failure_context, top_k=3)

        self.assertGreaterEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0]["experience"]["experience_id"], "exp-similar")
        self.assertEqual(retrieved[0]["context_similarity"], 1.0)
        self.assertIn("comparison_coverage", retrieved[0])
        self.assertIn("field_scores", retrieved[0]["similarity_breakdown"])

    def test_threshold_excludes_low_similarity_experiences(self):
        failure_context = _make_failure_context(
            subgoal={
                "description": "Acquire the spatula with a stable grasp",
                "action_type": "acquire",
                "selected_object_id": "spatula",
                "selected_object_class": "spatula",
            },
        )
        low_match_memory = Path(self.temp_dir.name) / "memory_low_match.json"
        low_match_memory.write_text(
            json.dumps(
                {
                    "failure_recovery_experience": {
                        "experiences": [
                            _make_experience(
                                "exp-low-match",
                                subgoal_description="Acquire the spatula with a stable grasp",
                                action_type="place",
                                object_class="mug",
                            )
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        adapter = MemoryAdapter(
            memory_path=low_match_memory,
            config=RetrievalConfig(similarity_threshold=0.75),
        )

        retrieved = adapter.retrieve_experiences(failure_context, top_k=3)

        self.assertEqual(retrieved, [])


class M6Tests(unittest.TestCase):
    def test_builder_maps_known_fields_and_preserves_defaults(self):
        state = {
            "failure_id": "f-1",
            "task": {"task_id": "t-1", "unknown": "ignored"},
            "verification": {"result": "FAIL"},
            "execution": {"controller_status": "COMPLETED", "gripper": {"force": 2.0}},
            "unknown_section": {"ignored": True},
        }
        context = FailureContextBuilder().build(state)

        self.assertEqual(context["failure_id"], "f-1")
        self.assertEqual(context["task"], {"task_id": "t-1", "instruction": None})
        self.assertEqual(context["verification"]["result"], "FAIL")
        self.assertEqual(context["execution"]["gripper"]["force"], 2.0)

    def test_router_with_empty_context_keeps_retrieval_empty_and_runs_diagnosis(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            memory_path.write_text(
                json.dumps(
                    {
                        "failure_recovery_experience": {
                            "experiences": [
                                _make_experience(
                                    "exp-seed",
                                    action_type="acquire",
                                    object_class="spatula",
                                )
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            adapter = MemoryAdapter(memory_path=memory_path)
            result = DiagnoseRouter(memory_adapter=adapter).run({"verification": {"result": "FAIL"}})

        self.assertEqual(result["diagnosis"]["memory_context"]["retrieved_experiences"], [])
        self.assertEqual(result["diagnosis"]["memory_context"]["diagnosis_evidence"], [])
        self.assertEqual(result["recovery"]["decision_mode"], "DIAGNOSIS_GUIDED")
        self.assertEqual(result["recovery"]["guidance"]["experience_ids"], [])
        self.assertEqual(result["diagnosis"]["failure_type"], "PLANNING")
        self.assertEqual(result["diagnosis"]["failure_cause"]["code"], "INVALID_APPROACH")
        self.assertEqual(result["diagnosis"]["affected_module"], "M5")

    def test_router_runs_diagnosis_when_retrieval_succeeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            memory_path.write_text(
                json.dumps(
                    {
                        "failure_recovery_experience": {
                            "experiences": [
                                _make_experience(
                                    "exp-seed",
                                    subgoal_description="Acquire the spatula",
                                    action_type="acquire",
                                    object_class="spatula",
                                )
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            adapter = MemoryAdapter(
                memory_path=memory_path,
                config=RetrievalConfig(similarity_threshold=0.5),
            )
            pipeline_state = {
                "subgoal": {
                    "description": "Acquire the spatula",
                    "action_type": "acquire",
                    "selected_object_id": "spatula",
                    "selected_object_class": "spatula",
                },
                "verification": {"result": "FAIL"},
            }
            result = DiagnoseRouter(memory_adapter=adapter).run(pipeline_state)

        self.assertEqual(result["recovery"]["decision_mode"], "EXPERIENCE_GUIDED")
        self.assertEqual(result["recovery"]["guidance"]["experience_ids"], ["exp-seed"])
        self.assertEqual(len(result["recovery"]["guidance"]["recovery_evidence"]), 1)
        self.assertEqual(result["diagnosis"]["failure_type"], "PLANNING")
        self.assertEqual(result["diagnosis"]["failure_cause"]["code"], "INVALID_APPROACH")
        self.assertEqual(result["diagnosis"]["affected_module"], "M5")

    def test_router_uses_diagnosis_guided_when_stub_retrieval_lacks_diagnosis_match(self):
        class StubAdapter:
            def retrieve_experiences(self, failure_context, top_k=3):
                return [
                    _make_retrieved_item(
                        _make_experience(
                            "exp-1",
                            failure_type="EXECUTION_CONTROL",
                            failure_cause_code="GRASP_FAILURE",
                            affected_module="Controller",
                        )
                    )
                ]

        result = DiagnoseRouter(memory_adapter=StubAdapter()).run({})

        self.assertEqual(result["recovery"]["decision_mode"], "DIAGNOSIS_GUIDED")
        self.assertEqual(result["recovery"]["guidance"]["experience_ids"], [])
        self.assertEqual(result["diagnosis"]["failure_type"], "PLANNING")

    def test_rank_experiences_skips_entries_without_comparable_fields(self):
        experiences = [
            _make_experience("exp-empty"),
            _make_experience("exp-match", action_type="acquire", object_id="spatula"),
        ]
        failure_context = _make_failure_context(
            subgoal={"action_type": "acquire", "selected_object_id": "spatula"},
        )

        ranked = rank_experiences(
            failure_context,
            experiences,
            top_k=3,
            similarity_threshold=0.5,
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["experience"]["experience_id"], "exp-match")


if __name__ == "__main__":
    unittest.main()
