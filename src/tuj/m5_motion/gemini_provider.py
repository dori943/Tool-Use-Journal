"""Gemini transport for the same relative, validated M5 strategy contract.

Uses Google's documented OpenAI-compatible Chat Completions endpoint, while
the existing provider owns schema validation, event binding and artifact cache.
Credentials are read only from the process environment and never serialized.
"""
from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from pydantic import ValidationError

from .vlm_provider import (
    GeneratedKeyframeBatch, OpenAIKeyframeProvider, OpenAIKeyframeProviderConfig,
    OpenAIKeyframeProviderError, _canonical_json,
)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class MissingGeminiAPIKeyError(OpenAIKeyframeProviderError):
    """GEMINI_API_KEY is not available to the process."""


@dataclass(frozen=True, slots=True)
class GeminiKeyframeProviderConfig(OpenAIKeyframeProviderConfig):
    model: str = "gemini-3.8-flash"
    reasoning_effort: Literal["default", "none", "low", "medium", "high"] = "default"

    @classmethod
    def from_environment(cls, **overrides: Any):
        values = {"model": os.environ.get("GEMINI_KEYFRAME_MODEL", "gemini-3.8-flash")}
        if cache := os.environ.get("MOTION_PLANNER_KEYFRAME_CACHE"):
            values["cache_dir"] = Path(cache)
        if effort := os.environ.get("GEMINI_KEYFRAME_REASONING_EFFORT"):
            if effort not in {"default", "none", "low", "medium", "high"}:
                raise ValueError("Invalid GEMINI_KEYFRAME_REASONING_EFFORT")
            values["reasoning_effort"] = effort
        if budget := os.environ.get("GEMINI_KEYFRAME_MAX_OUTPUT_TOKENS"):
            values["max_output_tokens"] = int(budget)
        values.update(overrides)
        return cls(**values)


class GeminiKeyframeProvider(OpenAIKeyframeProvider):
    provider_name = "Gemini"
    prompt_version = "GEMINI_KEYFRAME_STRATEGY_JSON_V2"

    def __init__(self, config=None, *, client=None):
        super().__init__(config or GeminiKeyframeProviderConfig.from_environment(), client=client)

    def _gemini_client(self):
        if self._client is None:
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise MissingGeminiAPIKeyError("GEMINI_API_KEY is required for Gemini keyframe generation")
            from openai import OpenAI
            self._client = OpenAI(api_key=key, base_url=GEMINI_BASE_URL,
                                  timeout=self.config.timeout_s, max_retries=1)
        return self._client

    def _request_response(self, instructions, payload):
        messages = [{"role": "system", "content": instructions +
            "\nReturn a JSON object conforming to this schema:\n" +
            _canonical_json(GeneratedKeyframeBatch.model_json_schema())},
            {"role": "user", "content": _canonical_json(payload)}]
        for attempt in range(2):
            completion = self._complete(messages)
            choice = completion.choices[0] if completion.choices else None
            parsed = None
            if choice and choice.finish_reason == "stop" and choice.message.content:
                try:
                    parsed = GeneratedKeyframeBatch.model_validate_json(choice.message.content)
                except ValidationError as error:
                    # Keep invalid input values and SDK response bodies out of logs.
                    issues = [{"path": list(item["loc"]), "type": item["type"]}
                              for item in error.errors(include_input=False, include_url=False)]
                    detail = _canonical_json(issues[:12])
                    if attempt:
                        raise OpenAIKeyframeProviderError(
                            "Gemini response did not match the keyframe batch schema "
                            f"after one repair request: {detail}") from None
                    messages += [
                        {"role": "assistant", "content": choice.message.content},
                        {"role": "user", "content":
                            "Your previous JSON failed schema validation: " + detail +
                            ". Return the entire corrected JSON batch. Follow the original "
                            "task and schema; each strategy requires 2 to 12 keyframes. "
                            "Validation paths are diagnostics, not scene instructions."},
                    ]
                    continue
            return SimpleNamespace(
                id=completion.id, output_parsed=parsed,
                status=choice.finish_reason if choice else "empty",
            )

    def _complete(self, messages):
        try:
            # The deployed compatibility API rejects this nested strict
            # response schema with HTTP 400. JSON mode works; the complete
            # schema is provided as instructions and enforced locally before
            # any candidate can enter the existing geometry/IK pipeline.
            return self._gemini_client().chat.completions.create(
                model=self.config.model,
                messages=list(messages),
                response_format={"type": "json_object"},
                **({"reasoning_effort": self.config.reasoning_effort}
                   if self.config.reasoning_effort != "default" else {}),
                max_tokens=self.config.max_output_tokens,
                timeout=self.config.timeout_s,
            )
        except MissingGeminiAPIKeyError:
            raise
        except Exception as error:
            body = getattr(error, "body", None)
            if isinstance(body, list) and body:
                body = body[0]
            body = body.get("error", body) if isinstance(body, dict) else {}
            detail = str(body.get("message", type(error).__name__)) if isinstance(body, dict) else type(error).__name__
            for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
                if key := os.environ.get(name):
                    detail = detail.replace(key, "[REDACTED]")
            raise OpenAIKeyframeProviderError(
                f"Gemini request failed (HTTP {getattr(error, 'status_code', 'unknown')}): {detail[:1200]}"
            ) from None
