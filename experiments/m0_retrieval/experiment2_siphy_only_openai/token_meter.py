"""Token metering + gpt-5.x API compatibility for Experiment 2 OpenAI.

Production ``SiPhyBackend`` does not return usage. Without modifying production,
we inject a thin client proxy that:

  1. Records ``usage`` from each ``chat.completions.create`` response.
  2. Strips / remaps optional kwargs that some OpenAI models reject.
  3. For GPT-5.x only, enlarges the completion budget because production
     SiPhy uses max_tokens=500, which can cause GPT-5.x to terminate with
     finish_reason="length" before emitting the final JSON.

Does NOT alter messages, system prompt, image payload, or response parsing.
Does NOT modify production ``siphy_backend.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Experiment-only GPT-5.x completion budget
# ---------------------------------------------------------------------------

GPT5_MAX_COMPLETION_TOKENS = 4096


@dataclass
class TokenMeter:
    api_attempt_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    usage_events: list[dict[str, Any]] = field(default_factory=list)
    sanitized_kwargs_log: list[dict[str, Any]] = field(default_factory=list)

    def record(self, usage: Any) -> None:
        """Accumulate token usage reported by the OpenAI API.

        Important:
        ``total_tokens`` is taken directly from the API response.
        It is NOT recomputed as input + output because some models may include
        additional token categories in the API-reported total.
        """

        if usage is None:
            self.usage_events.append({"usage": None})
            return

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
            "input_tokens": (
                self.input_tokens if self.usage_events else None
            ),
            "output_tokens": (
                self.output_tokens if self.usage_events else None
            ),
            "total_tokens": (
                self.total_tokens if self.usage_events else None
            ),
            "api_attempt_count": self.api_attempt_count,
            "usage_events": list(self.usage_events),
            "sanitized_kwargs_log": list(self.sanitized_kwargs_log),
        }


def _is_gpt5_model(model: str) -> bool:
    """Return True for GPT-5 family models such as gpt-5.6."""

    ms = str(model or "").lower()
    return ms.startswith("gpt-5")


def _model_needs_max_completion_tokens(model: str) -> bool:
    """Return True for models that use max_completion_tokens."""

    ms = str(model or "").lower()

    if ms.startswith(("o1", "o3", "o4")):
        return True

    if ms.startswith("gpt-5"):
        return True

    return False


def sanitize_create_kwargs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (sanitized_kwargs, audit) without changing prompt semantics.

    Experiment-only compatibility rules:

    1. Drop ``temperature`` entirely.
       GPT-5.6 may reject explicitly supplied non-default temperature values.

    2. GPT-5.x:
       Production SiPhy uses ``max_tokens=500``. GPT-5.x may consume this
       completion budget before emitting the final JSON, causing
       ``finish_reason=length``.

       Therefore, for this experiment only:

           max_tokens=500
               ↓
           max_completion_tokens=4096

       This changes only the maximum allowed completion budget.
       It does NOT change the prompt, messages, image input, parser, or
       production SiPhy reasoning semantics.

    3. o1/o3/o4:
       Preserve the original budget but rename ``max_tokens`` to
       ``max_completion_tokens`` for API compatibility.
    """

    out = dict(kwargs)

    audit: dict[str, Any] = {
        "dropped": [],
        "renamed": {},
        "overridden": {},
    }

    # ---------------------------------------------------------------
    # temperature compatibility
    # ---------------------------------------------------------------

    if "temperature" in out:
        original_temperature = out.pop("temperature", None)

        audit["dropped"].append(
            {
                "name": "temperature",
                "value": original_temperature,
            }
        )

    model = str(out.get("model") or "")

    # ---------------------------------------------------------------
    # GPT-5.x
    # ---------------------------------------------------------------

    if _is_gpt5_model(model):
        original_max_tokens = out.pop("max_tokens", None)
        original_max_completion_tokens = out.get(
            "max_completion_tokens"
        )

        out["max_completion_tokens"] = GPT5_MAX_COMPLETION_TOKENS

        audit["overridden"]["completion_budget"] = {
            "model": model,
            "original_max_tokens": original_max_tokens,
            "original_max_completion_tokens": (
                original_max_completion_tokens
            ),
            "new_max_completion_tokens": (
                GPT5_MAX_COMPLETION_TOKENS
            ),
            "reason": (
                "Production SiPhy uses max_tokens=500. "
                "GPT-5.x may consume the completion budget before "
                "emitting the final JSON, causing "
                "finish_reason=length. For this experiment only, "
                "the completion budget is enlarged without modifying "
                "production SiPhy semantics."
            ),
        }

        return out, audit

    # ---------------------------------------------------------------
    # Other models requiring max_completion_tokens
    # ---------------------------------------------------------------

    if _model_needs_max_completion_tokens(model):
        if (
            "max_tokens" in out
            and "max_completion_tokens" not in out
        ):
            original_value = out.pop("max_tokens")

            out["max_completion_tokens"] = original_value

            audit["renamed"]["max_tokens"] = {
                "to": "max_completion_tokens",
                "value": original_value,
            }

    return out, audit


class _CompletionsProxy:
    def __init__(
        self,
        inner_completions: Any,
        meter: TokenMeter,
    ):
        self._inner = inner_completions
        self._meter = meter

    def create(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        sanitized, audit = sanitize_create_kwargs(kwargs)

        if (
            audit["dropped"]
            or audit["renamed"]
            or audit["overridden"]
        ):
            self._meter.sanitized_kwargs_log.append(audit)

        # Counts every real API attempt, including production retries.
        self._meter.api_attempt_count += 1

        response = self._inner.create(
            *args,
            **sanitized,
        )

        # Record actual API-provided token usage.
        self._meter.record(
            getattr(response, "usage", None)
        )

        return response


class _ChatProxy:
    def __init__(
        self,
        inner_chat: Any,
        meter: TokenMeter,
    ):
        self.completions = _CompletionsProxy(
            inner_chat.completions,
            meter,
        )


class MeteredOpenAIClient:
    """Duck-types OpenAI client for production ``SiPhyBackend``.

    Exposes:
        client.chat.completions.create(...)
        client.base_url

    while transparently metering token usage and applying only
    experiment-local API compatibility handling.
    """

    def __init__(
        self,
        inner_client: Any,
        meter: TokenMeter | None = None,
    ):
        self._inner = inner_client
        self.meter = meter or TokenMeter()

        self.base_url = getattr(
            inner_client,
            "base_url",
            "",
        )

        self.chat = _ChatProxy(
            inner_client.chat,
            self.meter,
        )

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        return getattr(
            self._inner,
            name,
        )