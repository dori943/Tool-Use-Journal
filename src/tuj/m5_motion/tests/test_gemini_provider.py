"""Gemini transport keeps M5's existing contract and credential boundary."""
from types import SimpleNamespace
import json
import traceback

import pytest

from tuj.m5_motion.gemini_provider import (
    GeminiKeyframeProvider, GeminiKeyframeProviderConfig, MissingGeminiAPIKeyError,
)
from tuj.m5_motion.vlm_provider import GeneratedKeyframeBatch, OpenAIKeyframeProviderError
from tuj.m5_motion.tests.test_vlm_provider import _request, _batch


class Chat:
    def __init__(self, batch):
        self.batch, self.calls = batch, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(id="gemini-request-1", choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.batch.model_dump_json() if self.batch else None), finish_reason="stop")])


def provider(batch, **config):
    chat = Chat(batch)
    client = SimpleNamespace(chat=SimpleNamespace(completions=chat))
    return GeminiKeyframeProvider(GeminiKeyframeProviderConfig(candidate_count=2, **config),
                                  client=client), chat


def test_gemini_structured_output_validates_and_caches_without_secrets(tmp_path):
    p, chat = provider(_batch(), cache_dir=tmp_path)
    first = p.generate(_request())
    assert p.generate(_request()) == first
    assert len(chat.calls) == 1
    assert chat.calls[0]["response_format"] == {"type": "json_object"}
    assert '"additionalProperties":false' in chat.calls[0]["messages"][0]["content"]
    assert "must-not-leave-the-process" not in str(chat.calls)
    assert first.provenance.metadata["provider"] == "gemini"
    assert first.candidates[0].provenance.generator_id == "GEMINI_KEYFRAME_STRATEGY_JSON_V2"
    assert first.candidates[0].provenance.provider_request_id == "gemini-request-1"


def test_gemini_unknown_frames_rejected_before_planning():
    p, _ = provider(_batch(frame_ref="object:invented"))
    with pytest.raises(OpenAIKeyframeProviderError, match="unknown object frame"):
        p.generate(_request())


def test_missing_gemini_key_does_not_use_openai_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "different-provider-key")
    p = GeminiKeyframeProvider(GeminiKeyframeProviderConfig(candidate_count=2))
    with pytest.raises(MissingGeminiAPIKeyError):
        p.generate(_request())


def test_transport_exception_does_not_leak_credentials():
    p, chat = provider(_batch())
    def fail(**kwargs):
        raise RuntimeError("credential-example-in-sdk-error")
    chat.create = fail
    with pytest.raises(OpenAIKeyframeProviderError) as caught:
        p.generate(_request())
    assert "credential-example" not in "".join(traceback.format_exception(caught.value))


def test_refusal_is_not_accepted_as_a_plan():
    p, _ = provider(None)
    with pytest.raises(OpenAIKeyframeProviderError, match="no parsed output"):
        p.generate(_request())


def test_environment_output_budget_and_reasoning_are_sent(monkeypatch):
    monkeypatch.setenv("GEMINI_KEYFRAME_REASONING_EFFORT", "low")
    monkeypatch.setenv("GEMINI_KEYFRAME_MAX_OUTPUT_TOKENS", "32000")
    p, chat = provider(_batch())
    p.config = GeminiKeyframeProviderConfig.from_environment(candidate_count=2)
    p.generate(_request())
    assert chat.calls[0]["reasoning_effort"] == "low"
    assert chat.calls[0]["max_tokens"] == 32000
    explicit = GeminiKeyframeProviderConfig.from_environment(max_output_tokens=24000)
    assert explicit.max_output_tokens == 24000


@pytest.mark.parametrize("name,value", [
    ("GEMINI_KEYFRAME_REASONING_EFFORT", "unsupported"),
    ("GEMINI_KEYFRAME_MAX_OUTPUT_TOKENS", "0"),
    ("GEMINI_KEYFRAME_MAX_OUTPUT_TOKENS", "invalid"),
])
def test_invalid_environment_generation_config_fails_early(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        GeminiKeyframeProviderConfig.from_environment()


def test_invalid_json_gets_one_model_repair_before_geometry_validation(tmp_path):
    p, chat = provider(_batch(), cache_dir=tmp_path)
    valid = chat.create
    def first_invalid(**kwargs):
        response = valid(**kwargs)
        if len(chat.calls) == 1:
            data = _batch().model_dump(mode="json")
            data["candidates"][0]["keyframes"] = data["candidates"][0]["keyframes"][:1]
            response.choices[0].message.content = json.dumps(data)
        return response
    chat.create = first_invalid
    artifact = p.generate(_request())
    assert len(chat.calls) == 2
    assert "too_short" in chat.calls[1]["messages"][-1]["content"]
    assert len(artifact.candidates) == 2
    assert p.generate(_request()) == artifact
    assert len(chat.calls) == 2


def test_invalid_repair_stops_without_cache_or_input_values(tmp_path):
    p, chat = provider(_batch(), cache_dir=tmp_path)
    valid = chat.create
    def invalid(**kwargs):
        response = valid(**kwargs)
        response.choices[0].message.content = '{"candidates":"private-invalid-input"}'
        return response
    chat.create = invalid
    with pytest.raises(OpenAIKeyframeProviderError, match="after one repair") as caught:
        p.generate(_request())
    assert len(chat.calls) == 2
    assert "private-invalid-input" not in str(caught.value)
    assert not list(tmp_path.glob("*.json"))
