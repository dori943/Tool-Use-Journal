import base64
import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tuj.m6.diagnosis import (
    DiagnosisAPIError,
    DiagnosisResponseError,
    DiagnosisValidationError,
    MockFailureDiagnoser,
    validate_diagnosis_output,
)
from tuj.m6.diagnosis_config import create_failure_diagnoser, get_diagnoser_backend
from tuj.m6.image_utils import ImagePathError, local_image_to_data_url
from tuj.m6.memory_adapter import DEFAULT_MEMORY_PATH
from tuj.m6.openai_vlm_diagnoser import (
    GeneratedFailureDiagnosis,
    MissingOpenAIAPIKeyError,
    OpenAIVLMFailureDiagnoser,
    build_openai_diagnosis_input,
)
from tuj.m6.prompts import build_failure_diagnosis_payload


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
        "motion_plan": {"planning_status": "FAILED"},
        "execution": {"controller_status": None},
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


def _valid_generated_diagnosis(**overrides) -> GeneratedFailureDiagnosis:
    payload = {
        "failure_type": "PLANNING",
        "failure_cause": {
            "code": "INVALID_APPROACH",
            "description": "Approach direction collided with obstacle.",
        },
        "affected_module": "M5",
        "evidence": ["motion plan reported invalid approach"],
        "confidence": 0.87,
    }
    payload.update(overrides)
    return GeneratedFailureDiagnosis.model_validate(payload)


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


class OpenAIVLMDiagnosisTests(unittest.TestCase):
    def test_openai_diagnoser_returns_canonical_structured_output(self):
        fake_responses = _FakeResponsesClient(_valid_generated_diagnosis())
        diagnoser = OpenAIVLMFailureDiagnoser(client=_FakeOpenAIClient(fake_responses))

        result = diagnoser.diagnose(_make_failure_context(), [])

        self.assertEqual(result["failure_type"], "PLANNING")
        self.assertEqual(result["failure_cause"]["code"], "INVALID_APPROACH")
        self.assertEqual(result["affected_module"], "M5")
        self.assertEqual(result["confidence"], 0.87)

    def test_validator_runs_after_api_response(self):
        fake_responses = _FakeResponsesClient(_valid_generated_diagnosis())
        diagnoser = OpenAIVLMFailureDiagnoser(client=_FakeOpenAIClient(fake_responses))

        with mock.patch(
            "tuj.m6.openai_vlm_diagnoser.validate_diagnosis_output",
            wraps=validate_diagnosis_output,
        ) as validator:
            diagnoser.diagnose(_make_failure_context(), [])

        validator.assert_called_once()

    def test_invalid_failure_type_response_rejected(self):
        fake_responses = _FakeResponsesClient(
            _valid_generated_diagnosis(failure_type="UNKNOWN_TYPE", affected_module="M5")
        )
        diagnoser = OpenAIVLMFailureDiagnoser(client=_FakeOpenAIClient(fake_responses))

        with self.assertRaises(DiagnosisResponseError):
            diagnoser.diagnose(_make_failure_context(), [])

    def test_invalid_failure_cause_response_rejected(self):
        parsed = _valid_generated_diagnosis()
        parsed = GeneratedFailureDiagnosis.model_validate(
            {
                **parsed.model_dump(),
                "failure_cause": {
                    "code": "GRASP_FAILURE",
                    "description": "wrong cause for planning",
                },
            }
        )
        diagnoser = OpenAIVLMFailureDiagnoser(
            client=_FakeOpenAIClient(_FakeResponsesClient(parsed))
        )

        with self.assertRaises(DiagnosisResponseError):
            diagnoser.diagnose(_make_failure_context(), [])

    def test_affected_module_mismatch_rejected(self):
        fake_responses = _FakeResponsesClient(
            _valid_generated_diagnosis(affected_module="Controller")
        )
        diagnoser = OpenAIVLMFailureDiagnoser(client=_FakeOpenAIClient(fake_responses))

        with self.assertRaises(DiagnosisResponseError):
            diagnoser.diagnose(_make_failure_context(), [])

    def test_missing_openai_api_key_is_handled(self):
        diagnoser = OpenAIVLMFailureDiagnoser()
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(MissingOpenAIAPIKeyError):
                diagnoser.diagnose(_make_failure_context(), [])

    def test_api_exception_is_handled(self):
        fake_responses = _FakeResponsesClient(
            None,
            error=RuntimeError("network down"),
        )
        diagnoser = OpenAIVLMFailureDiagnoser(client=_FakeOpenAIClient(fake_responses))

        with self.assertRaises(DiagnosisAPIError):
            diagnoser.diagnose(_make_failure_context(), [])

    def test_malformed_response_is_handled(self):
        fake_responses = _FakeResponsesClient(None)
        diagnoser = OpenAIVLMFailureDiagnoser(client=_FakeOpenAIClient(fake_responses))

        with self.assertRaises(DiagnosisResponseError):
            diagnoser.diagnose(_make_failure_context(), [])

    def test_diagnosis_evidence_excludes_recovery_information(self):
        evidence = [
            {
                "experience_id": "exp-1",
                "retrieval": {"context_similarity": 1.0},
                "past_context": {"action_type": "acquire"},
                "past_diagnosis": {
                    "failure_type": "PLANNING",
                    "failure_cause": {"code": "INVALID_APPROACH"},
                    "affected_module": "M5",
                },
                "source": "offline_seed",
                "past_recovery": {"recovery_category": "REPLAN_MOTION"},
                "outcome": {"status": "SUCCESS"},
            }
        ]
        payload = build_failure_diagnosis_payload(_make_failure_context(), evidence)
        serialized = json.dumps(payload)

        self.assertIn("past_diagnosis", serialized)
        self.assertNotIn("past_recovery", serialized)
        self.assertNotIn('"outcome"', serialized)

    def test_text_only_request_has_no_image_inputs(self):
        _, content, image_count = build_openai_diagnosis_input(_make_failure_context(), [])

        self.assertEqual(image_count, 0)
        self.assertFalse(any(part.get("type") == "input_image" for part in content))

    def test_before_image_only_request_has_one_image_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "before.png"
            image_path.write_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                )
            )
            context = _make_failure_context(
                observation={"before_image": str(image_path), "after_image": None}
            )

            _, content, image_count = build_openai_diagnosis_input(context, [])

        self.assertEqual(image_count, 1)
        image_parts = [part for part in content if part.get("type") == "input_image"]
        self.assertEqual(len(image_parts), 1)
        self.assertTrue(image_parts[0]["image_url"].startswith("data:image/png;base64,"))

    def test_before_and_after_images_create_two_image_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            before_path = Path(temp_dir) / "before.jpg"
            after_path = Path(temp_dir) / "after.jpeg"
            png_bytes = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
            )
            before_path.write_bytes(png_bytes)
            after_path.write_bytes(png_bytes)
            context = _make_failure_context(
                observation={
                    "before_image": str(before_path),
                    "after_image": str(after_path),
                }
            )

            _, content, image_count = build_openai_diagnosis_input(context, [])

        self.assertEqual(image_count, 2)
        self.assertEqual(
            sum(1 for part in content if part.get("type") == "input_image"),
            2,
        )

    def test_past_diagnosis_is_included_as_supplementary_evidence(self):
        evidence = [
            {
                "experience_id": "exp-1",
                "retrieval": {"context_similarity": 0.9},
                "past_context": {"action_type": "acquire"},
                "past_diagnosis": {
                    "failure_type": "PLANNING",
                    "failure_cause": {"code": "INVALID_APPROACH"},
                    "affected_module": "M5",
                },
                "source": "offline_seed",
            }
        ]
        payload = build_failure_diagnosis_payload(_make_failure_context(), evidence)

        self.assertEqual(
            payload["diagnosis_side_past_experience_evidence"][0]["past_diagnosis"]["failure_type"],
            "PLANNING",
        )

    def test_past_recovery_is_not_included_in_prompt_payload(self):
        evidence = [
            {
                "experience_id": "exp-1",
                "retrieval": {"context_similarity": 0.9},
                "past_context": {},
                "past_diagnosis": {
                    "failure_type": "PLANNING",
                    "failure_cause": {"code": "INVALID_APPROACH"},
                    "affected_module": "M5",
                },
                "source": "offline_seed",
                "past_recovery": {"recovery_category": "REPLAN_MOTION"},
            }
        ]
        payload = build_failure_diagnosis_payload(_make_failure_context(), evidence)

        self.assertNotIn("past_recovery", payload["diagnosis_side_past_experience_evidence"][0])

    def test_client_injection_avoids_network_calls(self):
        fake_responses = _FakeResponsesClient(_valid_generated_diagnosis())
        fake_client = _FakeOpenAIClient(fake_responses)
        diagnoser = OpenAIVLMFailureDiagnoser(client=fake_client)

        diagnoser.diagnose(_make_failure_context(), [])

        self.assertIsNotNone(fake_responses.last_kwargs)
        self.assertIs(diagnoser._openai_client(), fake_client)

    def test_mock_backend_still_behaves_as_before(self):
        result = MockFailureDiagnoser().diagnose(_make_failure_context(), [])

        validate_diagnosis_output(result)
        self.assertEqual(result["failure_type"], "PLANNING")
        self.assertEqual(result["failure_cause"]["code"], "INVALID_APPROACH")
        self.assertEqual(result["affected_module"], "M5")

    def test_output_memory_json_is_not_modified(self):
        if not DEFAULT_MEMORY_PATH.exists():
            self.skipTest("output/memory.json is not available in this environment")

        before_digest = sha256(DEFAULT_MEMORY_PATH.read_bytes()).hexdigest()
        fake_responses = _FakeResponsesClient(_valid_generated_diagnosis())
        diagnoser = OpenAIVLMFailureDiagnoser(client=_FakeOpenAIClient(fake_responses))
        diagnoser.diagnose(_make_failure_context(), [])
        after_digest = sha256(DEFAULT_MEMORY_PATH.read_bytes()).hexdigest()

        self.assertEqual(before_digest, after_digest)

    def test_create_failure_diagnoser_defaults_to_mock(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            diagnoser = create_failure_diagnoser()

        self.assertIsInstance(diagnoser, MockFailureDiagnoser)
        self.assertEqual(get_diagnoser_backend(), "mock")

    def test_unsupported_image_path_raises_clear_error(self):
        context = _make_failure_context(
            observation={"before_image": "/missing/before.png", "after_image": None}
        )

        with self.assertRaises(DiagnosisAPIError):
            OpenAIVLMFailureDiagnoser(
                client=_FakeOpenAIClient(_FakeResponsesClient(_valid_generated_diagnosis()))
            ).diagnose(context, [])

    def test_local_image_to_data_url_rejects_missing_path(self):
        with self.assertRaises(ImagePathError):
            local_image_to_data_url("/path/does/not/exist.png")


if __name__ == "__main__":
    unittest.main()
