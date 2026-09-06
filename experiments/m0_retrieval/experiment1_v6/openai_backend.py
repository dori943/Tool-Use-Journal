"""Experiment-1 v6 OpenAI client (does NOT modify production / Gemini backends).

Uses official ``openai`` Python SDK (chat.completions + vision image_url).
Reads OPENAI_API_KEY. Model name comes from CLI (--model) unchanged.
"""
from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


API_KEY_ENV = "OPENAI_API_KEY"
PROVIDER = "openai"
# Soft default for dry-run display only; live runs require explicit --model.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


@dataclass
class OpenAIUsage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    raw: dict[str, Any]
    api_attempt_count: int = 1


@dataclass
class OpenAIResponse:
    text: str
    usage: OpenAIUsage
    sdk: str = "openai"


def require_api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set. "
            f"Set it before running Experiment 1 v6 OpenAI inference."
        )
    return key


def detect_sdk() -> str:
    import openai

    return f"openai-{getattr(openai, '__version__', 'unknown')}"


def _usage_from_response(usage: Any, *, attempts: int = 1) -> OpenAIUsage:
    """Read API usage.total_tokens as-is (do not recompute as input+output)."""
    if usage is None:
        return OpenAIUsage(None, None, None, {}, api_attempt_count=attempts)
    inp = getattr(usage, "prompt_tokens", None)
    out = getattr(usage, "completion_tokens", None)
    tot = getattr(usage, "total_tokens", None)
    raw = {
        "prompt_tokens": inp,
        "completion_tokens": out,
        "total_tokens": tot,
    }
    return OpenAIUsage(
        int(inp) if inp is not None else None,
        int(out) if out is not None else None,
        int(tot) if tot is not None else None,
        raw,
        api_attempt_count=attempts,
    )


def _image_data_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None:
        mime = "image/png"
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


class OpenAIClient:
    """Thin wrapper: one chat.completions.create → text + usage."""

    def __init__(self, model: str, *, max_retries: int = 0):
        from openai import OpenAI

        self.model = model
        self.max_retries = max(0, int(max_retries))
        self.sdk = detect_sdk()
        self._client = OpenAI(api_key=require_api_key())

    def generate_json(
        self,
        system_prompt: str,
        user_text: str,
        *,
        image_path: Path | None = None,
    ) -> OpenAIResponse:
        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        if image_path is not None:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(Path(image_path))},
                }
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        last_err: Exception | None = None
        attempts = 0
        max_attempts = 1 + self.max_retries
        while attempts < max_attempts:
            attempts += 1
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                usage = _usage_from_response(getattr(resp, "usage", None), attempts=attempts)
                return OpenAIResponse(text=text, usage=usage, sdk=self.sdk)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        raise RuntimeError(
            f"OpenAI chat.completions failed after {attempts} attempt(s): {last_err}"
        ) from last_err
