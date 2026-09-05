"""Experiment-1 Gemini client (does NOT modify production SiPhy / OpenAI).

Supports either installed package:
  - google-genai  (preferred: ``from google import genai``)
  - google-generativeai (legacy: ``import google.generativeai as genai``)

Install (user must run; production deps are not modified here):
  pip install google-genai
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
API_KEY_ENV = "GEMINI_API_KEY"
PROVIDER = "gemini"


@dataclass
class GeminiUsage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    raw: dict[str, Any]


@dataclass
class GeminiResponse:
    text: str
    usage: GeminiUsage
    sdk: str


def require_api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set. "
            f"Set it before running Experiment 1 Gemini inference "
            f"(do not hard-code the key)."
        )
    return key


def detect_sdk() -> str:
    """Return 'google-genai' | 'google-generativeai' | raise ImportError."""
    try:
        from google import genai  # noqa: F401

        return "google-genai"
    except Exception:
        pass
    try:
        import google.generativeai  # noqa: F401

        return "google-generativeai"
    except Exception as exc:
        raise ImportError(
            "No Gemini SDK found. Install one of:\n"
            "  pip install google-genai\n"
            "  # or\n"
            "  pip install google-generativeai\n"
            f"Original import error: {exc}"
        ) from exc


def _usage_from_metadata(meta: Any) -> GeminiUsage:
    """Map Gemini usage_metadata fields. Do not invent total from input+output."""
    if meta is None:
        return GeminiUsage(None, None, None, {})
    inp = (
        getattr(meta, "prompt_token_count", None)
        or getattr(meta, "input_token_count", None)
        or (meta.get("prompt_token_count") if isinstance(meta, dict) else None)
    )
    out = (
        getattr(meta, "candidates_token_count", None)
        or getattr(meta, "output_token_count", None)
        or (meta.get("candidates_token_count") if isinstance(meta, dict) else None)
    )
    total = (
        getattr(meta, "total_token_count", None)
        or (meta.get("total_token_count") if isinstance(meta, dict) else None)
    )
    raw = {}
    for name in (
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "input_token_count",
        "output_token_count",
        "cached_content_token_count",
    ):
        val = getattr(meta, name, None)
        if val is None and isinstance(meta, dict):
            val = meta.get(name)
        if val is not None:
            raw[name] = val
    return GeminiUsage(
        int(inp) if inp is not None else None,
        int(out) if out is not None else None,
        int(total) if total is not None else None,
        raw,
    )


def _extract_text(response: Any) -> str:
    text = getattr(response, "text", None) or ""
    if text:
        return text
    if getattr(response, "candidates", None):
        try:
            parts = response.candidates[0].content.parts
            return "".join(getattr(p, "text", "") or "" for p in parts)
        except Exception:
            return ""
    return ""


class GeminiClient:
    """Thin wrapper: exactly one generate_content call → text + usage."""

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or require_api_key()
        self.sdk = detect_sdk()
        self._client = None
        if self.sdk == "google-genai":
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        else:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path | None = None,
    ) -> GeminiResponse:
        """Single model call requesting JSON (optional PNG crop). No retries."""
        if self.sdk == "google-genai":
            from google.genai import types

            parts: list[Any] = []
            if image_path is not None:
                data = Path(image_path).read_bytes()
                parts.append(types.Part.from_bytes(data=data, mime_type="image/png"))
            parts.append(types.Part.from_text(text=user_prompt))
            response = self._client.models.generate_content(
                model=self.model,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                ),
            )
            usage = _usage_from_metadata(getattr(response, "usage_metadata", None))
            return GeminiResponse(text=_extract_text(response), usage=usage, sdk=self.sdk)

        import google.generativeai as genai

        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        )
        content: list[Any] = [user_prompt]
        if image_path is not None:
            from PIL import Image

            content = [Image.open(image_path), user_prompt]
        response = model.generate_content(content)
        usage = _usage_from_metadata(getattr(response, "usage_metadata", None))
        return GeminiResponse(text=_extract_text(response), usage=usage, sdk=self.sdk)
