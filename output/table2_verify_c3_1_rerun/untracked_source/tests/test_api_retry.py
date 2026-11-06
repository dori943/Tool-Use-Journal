# -*- coding: utf-8 -*-
"""외부 API 일시 장애: 요청 단위 재시도, 소진 시 API_UNAVAILABLE, 분류는 infrastructure."""
import pytest

from tuj.common import api_retry as ar
from tuj.m5_motion.failure_taxonomy import classify_failure


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setenv("TUJ_API_RETRIES", "3")
    monkeypatch.setenv("TUJ_API_BACKOFF_S", "0")
    ar.api_retry_log.clear()


class _E503(Exception):
    status_code = 503

    def __init__(self):
        super().__init__("This model is currently experiencing high demand")


class _E400(Exception):
    status_code = 400


def _client(side_effects):
    calls = {"n": 0}

    class Completions:
        def create(self, **kwargs):
            calls["n"] += 1
            effect = side_effects[min(calls["n"], len(side_effects)) - 1]
            if isinstance(effect, Exception):
                raise effect
            return effect

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()
        api_key = "k"

    return Client(), calls


def test_recovers_from_transient_503():
    client, calls = _client([_E503(), _E503(), "ok"])
    wrapped = ar.with_api_retry(client, "t")
    assert wrapped.chat.completions.create(model="m") == "ok"
    assert calls["n"] == 3
    assert wrapped.api_key == "k"                       # 나머지 속성은 그대로 위임
    assert ar.api_retry_log[-1]["recovered_after"] == 2


def test_permanent_error_is_not_retried():
    client, calls = _client([_E400("bad request")])
    with pytest.raises(_E400):
        ar.with_api_retry(client).chat.completions.create()
    assert calls["n"] == 1


def test_exhausted_retries_raise_api_unavailable():
    client, calls = _client([_E503()])
    with pytest.raises(ar.TransientAPIError) as info:
        ar.with_api_retry(client, "m5_gemini").chat.completions.create()
    assert calls["n"] == 4 and info.value.status_code == 503
    assert str(info.value).startswith("API_UNAVAILABLE")


def test_wrapping_is_idempotent():
    client, _ = _client(["ok"])
    once = ar.with_api_retry(client)
    assert ar.with_api_retry(once) is once


@pytest.mark.parametrize("detail", [
    "OpenAIKeyframeProviderError: Gemini request failed (HTTP 503): This model is currently experiencing high demand.",
    "OpenAIKeyframeProviderError: API_UNAVAILABLE: m5_gemini.chat.completions.create: 7회 호출 모두 일시 장애 (HTTP 503)",
    "OpenAI keyframe request failed (RateLimitError): too many requests",
])
def test_api_outage_is_infrastructure_not_planning(detail):
    r = classify_failure(detail)
    assert r["failure_stage"] == "infrastructure" and r["failure_code"] == "API_UNAVAILABLE"


@pytest.mark.parametrize("detail,stage", [
    ("ValueError: PLACEMENT_SPACE_EXHAUSTED: no unoccupied seat", "planning"),
    ("Gemini request failed (HTTP 400): Unsupported parameter", "planning"),
])
def test_real_failures_keep_their_stage(detail, stage):
    assert classify_failure(detail)["failure_stage"] == stage
