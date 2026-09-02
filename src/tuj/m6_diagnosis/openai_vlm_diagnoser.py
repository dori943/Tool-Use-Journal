"""OpenAI VLM-backed failure diagnoser for M6."""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .diagnosis import (
    DiagnosisAPIError,
    DiagnosisResponseError,
    DiagnosisValidationError,
    validate_diagnosis_output,
)
from .diagnosis_config import get_diagnosis_model
from .image_utils import ImagePathError, resolve_observation_images
from .prompts import build_failure_diagnosis_instructions, build_failure_diagnosis_text

logger = logging.getLogger(__name__)


class MissingOpenAIAPIKeyError(DiagnosisAPIError):
    """OPENAI_API_KEY is not available to the process."""


class _OpenAIResponsesClient(Protocol):
    def parse(self, **kwargs: Any) -> Any: ...


class _OpenAIClient(Protocol):
    @property
    def responses(self) -> _OpenAIResponsesClient: ...


class GeneratedFailureCause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GeneratedFailureDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_type: str = Field(min_length=1)
    failure_cause: GeneratedFailureCause
    affected_module: str = Field(min_length=1)
    evidence: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


def build_openai_diagnosis_input(
    failure_context: dict,
    diagnosis_evidence: list[dict],
) -> tuple[str, list[dict[str, Any]], int]:
    """Build OpenAI Responses API input parts and return image count."""
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": build_failure_diagnosis_text(failure_context, diagnosis_evidence),
        }
    ]
    observation = failure_context.get("observation") or {}
    for label, image_url in resolve_observation_images(observation):
        content.append(
            {
                "type": "input_text",
                "text": f"Observation image: {label}",
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": image_url,
                "detail": "auto",
            }
        )
    image_count = sum(1 for part in content if part.get("type") == "input_image")
    return build_failure_diagnosis_instructions(), content, image_count


def _parsed_diagnosis_to_output(parsed: GeneratedFailureDiagnosis) -> dict:
    return {
        "failure_type": parsed.failure_type,
        "failure_cause": {
            "code": parsed.failure_cause.code,
            "description": parsed.failure_cause.description,
        },
        "affected_module": parsed.affected_module,
        "evidence": list(parsed.evidence),
        "confidence": parsed.confidence,
    }


class OpenAIVLMFailureDiagnoser:
    """OpenAI Responses API diagnoser with structured output and taxonomy validation."""

    def __init__(self, model: str | None = None, client: _OpenAIClient | None = None):
        self.model = get_diagnosis_model(model)
        self._client = client

    def _openai_client(self) -> _OpenAIClient:
        if self._client is not None:
            return self._client
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingOpenAIAPIKeyError(
                "OPENAI_API_KEY is required for OpenAI VLM failure diagnosis"
            )
        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover - packaging guard
            raise DiagnosisAPIError(
                "install openai>=2,<3 to enable OpenAI VLM failure diagnosis"
            ) from error
        self._client = OpenAI()
        return self._client

    def diagnose(self, failure_context: dict, diagnosis_evidence: list[dict]) -> dict:
        try:
            instructions, content, image_count = build_openai_diagnosis_input(
                failure_context,
                diagnosis_evidence,
            )
        except ImagePathError as error:
            raise DiagnosisAPIError(str(error)) from error

        logger.debug(
            "openai diagnosis request backend=openai model=%s image_count=%s diagnosis_evidence_count=%s",
            self.model,
            image_count,
            len(diagnosis_evidence),
        )

        try:
            response = self._openai_client().responses.parse(
                model=self.model,
                instructions=instructions,
                input=[{"role": "user", "content": content}],
                text_format=GeneratedFailureDiagnosis,
                store=False,
            )
        except MissingOpenAIAPIKeyError:
            raise
        except Exception as error:  # noqa: BLE001 - SDK error surface varies
            raise DiagnosisAPIError(
                f"OpenAI Responses request failed ({type(error).__name__})"
            ) from error

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            response_id = str(getattr(response, "id", "unknown"))
            status = str(getattr(response, "status", "unknown"))
            raise DiagnosisResponseError(
                f"OpenAI response {response_id!r} had no parsed output (status={status})"
            )
        if not isinstance(parsed, GeneratedFailureDiagnosis):
            try:
                parsed = GeneratedFailureDiagnosis.model_validate(parsed)
            except ValidationError as error:
                raise DiagnosisResponseError(
                    "OpenAI response did not match the failure diagnosis schema"
                ) from error

        diagnosis_output = _parsed_diagnosis_to_output(parsed)
        try:
            validate_diagnosis_output(diagnosis_output)
        except DiagnosisValidationError as error:
            raise DiagnosisResponseError(str(error)) from error

        logger.debug(
            "openai diagnosis result failure_type=%s failure_cause=%s affected_module=%s",
            diagnosis_output["failure_type"],
            diagnosis_output["failure_cause"]["code"],
            diagnosis_output["affected_module"],
        )
        return diagnosis_output
