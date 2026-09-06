"""Token metering wrapper around an OpenAI-compatible client.

Production ``SiPhyBackend`` does not return usage. Without modifying production,
we inject a thin client proxy that records ``usage`` from each
``chat.completions.create`` response.

Logical VLM call (SiPhy design): 1 successful ``estimate`` / object.
Actual API attempts: up to ``MAX_TRIES`` (3) inside ``_propose`` on parse/API errors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenMeter:
    api_attempt_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    usage_events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, usage: Any) -> None:
        if usage is None:
            self.usage_events.append({"usage": None})
            return
        # OpenAI / Gemini OpenAI-compat: prompt_tokens / completion_tokens / total_tokens
        inp = getattr(usage, "prompt_tokens", None)
        if inp is None:
            inp = getattr(usage, "input_tokens", None)
        out = getattr(usage, "completion_tokens", None)
        if out is None:
            out = getattr(usage, "output_tokens", None)
        tot = getattr(usage, "total_tokens", None)
        if inp is not None:
            self.input_tokens += int(inp)
        if out is not None:
            self.output_tokens += int(out)
        if tot is not None:
            self.total_tokens += int(tot)
        self.usage_events.append(
            {
                "prompt_tokens": inp,
                "completion_tokens": out,
                "total_tokens": tot,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens if self.usage_events else None,
            "output_tokens": self.output_tokens if self.usage_events else None,
            # Prefer API-reported total (may exceed input+output due to reasoning tokens).
            "total_tokens": self.total_tokens if self.usage_events else None,
            "api_attempt_count": self.api_attempt_count,
            "usage_events": list(self.usage_events),
        }


class _CompletionsProxy:
    def __init__(self, inner_completions: Any, meter: TokenMeter):
        self._inner = inner_completions
        self._meter = meter

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self._meter.api_attempt_count += 1
        response = self._inner.create(*args, **kwargs)
        self._meter.record(getattr(response, "usage", None))
        return response


class _ChatProxy:
    def __init__(self, inner_chat: Any, meter: TokenMeter):
        self.completions = _CompletionsProxy(inner_chat.completions, meter)


class MeteredOpenAIClient:
    """Duck-types OpenAI client for ``SiPhyBackend`` (chat.completions.create + base_url)."""

    def __init__(self, inner_client: Any, meter: TokenMeter | None = None):
        self._inner = inner_client
        self.meter = meter or TokenMeter()
        self.base_url = getattr(inner_client, "base_url", "")
        self.chat = _ChatProxy(inner_client.chat, self.meter)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
