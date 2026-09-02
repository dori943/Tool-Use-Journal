import json
import unittest
from hashlib import sha256
from types import SimpleNamespace
from unittest import mock

from pydantic import ValidationError

from tuj.m6.memory_adapter import DEFAULT_MEMORY_PATH
from tuj.m6.openai_recovery_router import (
    GeneratedRecoveryAction,
    GeneratedRecoveryDecision,
    MissingOpenAIAPIKeyError,
    OpenAIRecoveryRouter,
    _parsed_recovery_to_output,
    build_openai_recovery_input,
)
from tuj.m6.prompts import build_recovery_router_payload
from tuj.m6.recovery_config import create_recovery_router, get_recovery_router_backend
from tuj.m6.recovery_router import (
    MockRecoveryRouter,
    RecoveryAPIError,
    RecoveryResponseError,
    RecoveryValidationError,
    validate_recovery_output,
)
from tuj.m6.schemas import empty_recovery


def _make_failure_context(**overrides) -> dict:
    context = {
        "failure_id": "failure-test",
        "task": {"task_id": "task-1", "instruction": "demo"},
        "subgoal": {
            "subgoal_id": "sg-1",
            "description": "Acquire object",
            "action_type": "acquire",
            "selected_object_id": "obj-1",
            "selected_object_class": "spatula",
        },
        "verification": {"result": "FAIL", "violated_predicates": []},
        "scene": {"nodes": [], "relations": [], "object_states": {}},
        "grounding": {},
        "task_plan": {"selected_ee": "2F", "selected_tool": None},
        "motion_plan": {"planning_status": "SUCCESS"},
        "execution": {"controller_status": "SUCCESS"},
        "history": {"retry_count": 0},
        "observation": {
            "before_image": None,
            "after_image": None,
            "before_scene": None,
            "after_scene": None,
        },
    }
    context.update(overrides)
    return context


def _make_diagnosis(**overrides) -> dict:
    diagnosis = {
        "failure_type": "PLANNING",
        "failure_cause": {
            "code": "INVALID_APPROACH",
            "description": "Invalid approach to target.",
        },
        "affected_module": "M5",
        "evidence": ["motion plan invalid"],
        "confidence": 0.85,
    }
    diagnosis.update(overrides)
    return diagnosis


def _make_recovery_evidence(experience_id: str = "exp-1") -> dict:
    return {
        "experience_id": experience_id,
        "retrieval": {"context_similarity": 0.9},
        "past_diagnosis": {
            "failure_type": "PLANNING",
            "failure_cause": {"code": "INVALID_APPROACH"},
            "affected_module": "M5",
        },
        "past_recovery": {
            "recovery_category": "REPLAN_MOTION",
            "action": {"action_type": "CHANGE_APPROACH"},
            "routing": {"restart_from": "M5", "rerun_modules": ["M5"]},
        },
        "outcome": {"status": "NOT_EXECUTED", "verification_result": None},
        "source": "offline_seed",
    }


def _valid_generated_recovery(**overrides) -> GeneratedRecoveryDecision:
    payload = {
        "recovery_category": "REPLAN_MOTION",
        "action": {
            "action_type": "CHANGE_APPROACH",
            "target_module": "M5",
            "target": {
                "subgoal_id": "sg-1",
                "object_id": "obj-1",
                "property": None,
                "relation": None,
                "ee_id": "2F",
                "tool_id": None,
            },
        },
        "routing": {
            "restart_from": "M5",
            "rerun_modules": ["M5"],
            "invalidate": [],
        },
    }
    payload.update(overrides)
    return GeneratedRecoveryDecision.model_validate(payload)


class _FakeResponsesClient:
    def __init__(self, parsed=None, *, error: Exception | None = None):
        self.parsed = parsed
        self.error = error
        self.last_kwargs = None

    def parse(self, **kwargs):
        self.last_kwargs = kwargs
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            id="resp-test",
            status="completed",
            output_parsed=self.parsed,
        )


class _FakeOpenAIClient:
    def __init__(self, responses_client: _FakeResponsesClient):
        self.responses = responses_client


class OpenAIRecoveryRouterTests(unittest.TestCase):
    def test_canonical_structured_recovery_response_parses(self):
        router = OpenAIRecoveryRouter(
            client=_FakeOpenAIClient(_FakeResponsesClient(_valid_generated_recovery()))
        )

        result = router.route(_make_failure_context(), _make_diagnosis(), "DIAGNOSIS_GUIDED", [])

        self.assertEqual(result["recovery_category"], "REPLAN_MOTION")
        self.assertEqual(result["action"]["action_type"], "CHANGE_APPROACH")
        self.assertEqual(result["action"]["target_module"], "M5")

    def test_existing_recovery_validator_is_called(self):
        router = OpenAIRecoveryRouter(
            client=_FakeOpenAIClient(_FakeResponsesClient(_valid_generated_recovery()))
        )

        with mock.patch(
            "tuj.m6.openai_recovery_router.validate_recovery_output",
            wraps=validate_recovery_output,
        ) as validator:
            router.route(_make_failure_context(), _make_diagnosis(), "DIAGNOSIS_GUIDED", [])

        validator.assert_called_once()

    def test_invalid_recovery_category_rejected(self):
        router = OpenAIRecoveryRouter(
            client=_FakeOpenAIClient(
                _FakeResponsesClient(
                    _valid_generated_recovery(recovery_category="UNKNOWN_CATEGORY")
                )
            )
        )

        with self.assertRaises(RecoveryResponseError):
            router.route(_make_failure_context(), _make_diagnosis(), "DIAGNOSIS_GUIDED", [])

    def test_invalid_action_type_rejected(self):
        parsed = _valid_generated_recovery()
        parsed = GeneratedRecoveryDecision.model_validate(
            {
                **parsed.model_dump(),
                "action": {
                    **parsed.action.model_dump(),
                    "action_type": "UNKNOWN_ACTION",
                },
            }
        )
        router = OpenAIRecoveryRouter(client=_FakeOpenAIClient(_FakeResponsesClient(parsed)))

        with self.assertRaises(RecoveryResponseError):
            router.route(_make_failure_context(), _make_diagnosis(), "DIAGNOSIS_GUIDED", [])

    def test_category_action_mismatch_rejected(self):
        parsed = _valid_generated_recovery(
            recovery_category="REPLAN_MOTION",
            action={
                "action_type": "RESELECT_EE",
                "target_module": "M4",
                "target": _valid_generated_recovery().action.target.model_dump(),
            },
            routing={"restart_from": "M4", "rerun_modules": ["M4"], "invalidate": []},
        )
        router = OpenAIRecoveryRouter(client=_FakeOpenAIClient(_FakeResponsesClient(parsed)))

        with self.assertRaises(RecoveryResponseError):
            router.route(_make_failure_context(), _make_diagnosis(), "DIAGNOSIS_GUIDED", [])

    def test_target_module_mismatch_rejected(self):
        parsed = _valid_generated_recovery(
            action={
                "action_type": "CHANGE_APPROACH",
                "target_module": "M2",
                "target": _valid_generated_recovery().action.target.model_dump(),
            },
            routing={"restart_from": "M2", "rerun_modules": ["M2"], "invalidate": []},
        )
        router = OpenAIRecoveryRouter(client=_FakeOpenAIClient(_FakeResponsesClient(parsed)))

        with self.assertRaises(RecoveryResponseError):
            router.route(_make_failure_context(), _make_diagnosis(), "DIAGNOSIS_GUIDED", [])

    def test_routing_mismatch_rejected(self):
        parsed = _valid_generated_recovery(
            routing={"restart_from": "M5", "rerun_modules": ["M2"], "invalidate": []}
        )
        router = OpenAIRecoveryRouter(client=_FakeOpenAIClient(_FakeResponsesClient(parsed)))

        with self.assertRaises(RecoveryResponseError):
            router.route(_make_failure_context(), _make_diagnosis(), "DIAGNOSIS_GUIDED", [])

    def test_malformed_openai_response_rejected(self):
        router = OpenAIRecoveryRouter(client=_FakeOpenAIClient(_FakeResponsesClient(None)))

        with self.assertRaises(RecoveryResponseError):
            router.route(_make_failure_context(), _make_diagnosis(), "DIAGNOSIS_GUIDED", [])

    def test_api_exception_wrapped_correctly(self):
        router = OpenAIRecoveryRouter(
            client=_FakeOpenAIClient(
                _FakeResponsesClient(None, error=RuntimeError("network down"))
            )
        )

        with self.assertRaises(RecoveryAPIError):
            router.route(_make_failure_context(), _make_diagnosis(), "DIAGNOSIS_GUIDED", [])

    def test_missing_openai_api_key_handled(self):
        router = OpenAIRecoveryRouter()
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(MissingOpenAIAPIKeyError):
                router.route(_make_failure_context(), _make_diagnosis(), "DIAGNOSIS_GUIDED", [])

    def test_mock_backend_remains_unchanged(self):
        result = MockRecoveryRouter().route(
            _make_failure_context(),
            _make_diagnosis(failure_type="EXECUTION_CONTROL", failure_cause={"code": "GRASP_FAILURE", "description": "grasp failed"}, affected_module="Controller"),
            "DIAGNOSIS_GUIDED",
            [],
        )

        self.assertEqual(result["recovery_category"], "RETRY_EXECUTION")
        self.assertEqual(result["action"]["action_type"], "RETRY_ACTION")
        self.assertEqual(result["action"]["target_module"], "Controller")

    def test_experience_guided_request_contains_filtered_recovery_evidence(self):
        evidence = [_make_recovery_evidence("exp-match")]
        payload = build_recovery_router_payload(
            _make_failure_context(),
            _make_diagnosis(),
            "EXPERIENCE_GUIDED",
            evidence,
        )

        self.assertEqual(
            [item["experience_id"] for item in payload["filtered_recovery_evidence"]],
            ["exp-match"],
        )

    def test_experience_guided_includes_past_recovery_information(self):
        evidence = [_make_recovery_evidence("exp-match")]
        payload = build_recovery_router_payload(
            _make_failure_context(),
            _make_diagnosis(),
            "EXPERIENCE_GUIDED",
            evidence,
        )

        past_recovery = payload["filtered_recovery_evidence"][0]["past_recovery"]
        self.assertEqual(past_recovery["recovery_category"], "REPLAN_MOTION")
        self.assertEqual(past_recovery["action"]["action_type"], "CHANGE_APPROACH")

    def test_diagnosis_guided_works_with_empty_recovery_evidence(self):
        router = OpenAIRecoveryRouter(
            client=_FakeOpenAIClient(
                _FakeResponsesClient(
                    _valid_generated_recovery(
                        recovery_category="RETRY_EXECUTION",
                        action={
                            "action_type": "RETRY_ACTION",
                            "target_module": "Controller",
                            "target": {
                                "subgoal_id": "sg-1",
                                "object_id": "obj-1",
                                "property": None,
                                "relation": None,
                                "ee_id": "2F",
                                "tool_id": None,
                            },
                        },
                        routing={
                            "restart_from": "Controller",
                            "rerun_modules": ["Controller"],
                            "invalidate": [],
                        },
                    )
                )
            )
        )

        result = router.route(
            _make_failure_context(),
            _make_diagnosis(
                failure_type="EXECUTION_CONTROL",
                failure_cause={"code": "GRASP_FAILURE", "description": "grasp failed"},
                affected_module="Controller",
            ),
            "DIAGNOSIS_GUIDED",
            [],
        )

        self.assertEqual(result["recovery_category"], "RETRY_EXECUTION")
        self.assertEqual(result["past_recoveries"], [])

    def test_openai_schema_does_not_include_decision_mode(self):
        self.assertNotIn("decision_mode", GeneratedRecoveryDecision.model_fields)

    def test_openai_schema_does_not_include_guidance_or_outcome(self):
        self.assertNotIn("guidance", GeneratedRecoveryDecision.model_fields)
        self.assertNotIn("outcome", GeneratedRecoveryDecision.model_fields)

    def test_openai_router_output_does_not_include_outcome(self):
        router = OpenAIRecoveryRouter(
            client=_FakeOpenAIClient(_FakeResponsesClient(_valid_generated_recovery()))
        )

        result = router.route(_make_failure_context(), _make_diagnosis(), "DIAGNOSIS_GUIDED", [])

        self.assertNotIn("outcome", result)
        self.assertNotIn("decision_mode", result)
        self.assertNotIn("guidance", result)

    def test_current_diagnosis_is_included_in_request(self):
        diagnosis = _make_diagnosis()
        payload = build_recovery_router_payload(
            _make_failure_context(),
            diagnosis,
            "DIAGNOSIS_GUIDED",
            [],
        )

        self.assertEqual(payload["current_diagnosis"]["failure_type"], diagnosis["failure_type"])
        self.assertEqual(
            payload["current_diagnosis"]["failure_cause"]["code"],
            diagnosis["failure_cause"]["code"],
        )

    def test_fixed_recovery_vocabulary_is_included(self):
        payload = build_recovery_router_payload(
            _make_failure_context(),
            _make_diagnosis(),
            "DIAGNOSIS_GUIDED",
            [],
        )

        self.assertIn("REPLAN_MOTION", payload["fixed_recovery_vocabulary"]["recovery_categories_and_actions"])
        self.assertIn("CHANGE_APPROACH", payload["fixed_recovery_vocabulary"]["recovery_categories_and_actions"]["REPLAN_MOTION"])

    def test_routing_constraints_are_included(self):
        payload = build_recovery_router_payload(
            _make_failure_context(),
            _make_diagnosis(),
            "DIAGNOSIS_GUIDED",
            [],
        )

        constraints = payload["canonical_routing_constraints"]
        self.assertEqual(constraints["recovery_action_to_target_module"]["CHANGE_APPROACH"], "M5")
        self.assertIn("M5", constraints["valid_routing_modules"])

    def test_output_memory_json_is_not_modified(self):
        if not DEFAULT_MEMORY_PATH.exists():
            self.skipTest("output/memory.json is not available in this environment")

        before_digest = sha256(DEFAULT_MEMORY_PATH.read_bytes()).hexdigest()
        router = OpenAIRecoveryRouter(
            client=_FakeOpenAIClient(_FakeResponsesClient(_valid_generated_recovery()))
        )
        router.route(_make_failure_context(), _make_diagnosis(), "DIAGNOSIS_GUIDED", [])
        after_digest = sha256(DEFAULT_MEMORY_PATH.read_bytes()).hexdigest()

        self.assertEqual(before_digest, after_digest)

    def test_create_recovery_router_defaults_to_mock(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            router = create_recovery_router()

        self.assertIsInstance(router, MockRecoveryRouter)
        self.assertEqual(get_recovery_router_backend(), "mock")

    def test_build_openai_recovery_input_is_text_only(self):
        _, content = build_openai_recovery_input(
            _make_failure_context(),
            _make_diagnosis(),
            "DIAGNOSIS_GUIDED",
            [],
        )

        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "input_text")

    def test_generated_recovery_action_schema_has_no_parameters_field(self):
        self.assertNotIn("parameters", GeneratedRecoveryAction.model_fields)

    def test_parsed_openai_response_merge_adds_empty_parameters(self):
        merged = _parsed_recovery_to_output(_valid_generated_recovery())

        self.assertEqual(merged["action"]["parameters"], {})

    def test_extra_unexpected_field_is_rejected(self):
        payload = _valid_generated_recovery().model_dump()
        payload["unexpected_field"] = "not_allowed"

        with self.assertRaises(ValidationError):
            GeneratedRecoveryDecision.model_validate(payload)

        action_payload = _valid_generated_recovery().action.model_dump()
        action_payload["parameters"] = {"offset": 1.0}

        with self.assertRaises(ValidationError):
            GeneratedRecoveryAction.model_validate(action_payload)


if __name__ == "__main__":
    unittest.main()
