import pytest

from tuj.m5_motion.vlm_provider import OpenAIKeyframeProviderConfig


def test_response_budget_is_opt_in_and_explicit_configuration_wins(monkeypatch):
    monkeypatch.delenv('OPENAI_KEYFRAME_MAX_OUTPUT_TOKENS', raising=False)
    assert OpenAIKeyframeProviderConfig.from_environment().max_output_tokens == 16000
    monkeypatch.setenv('OPENAI_KEYFRAME_MAX_OUTPUT_TOKENS', '32000')
    assert OpenAIKeyframeProviderConfig.from_environment().max_output_tokens == 32000
    assert OpenAIKeyframeProviderConfig.from_environment(max_output_tokens=24000).max_output_tokens == 24000


@pytest.mark.parametrize('budget', ['0', '-1', 'not-an-integer', '32000.5'])
def test_invalid_response_budget_rejected_before_api_call(monkeypatch, budget):
    monkeypatch.setenv('OPENAI_KEYFRAME_MAX_OUTPUT_TOKENS', budget)
    with pytest.raises(ValueError):
        OpenAIKeyframeProviderConfig.from_environment()
