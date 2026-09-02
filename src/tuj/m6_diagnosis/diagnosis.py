"""Failure diagnosis interface, validation, and mock implementation for M6."""

from __future__ import annotations

import logging
from typing import Protocol

from .taxonomy import FAILURE_MODULES, FAILURE_VOCABULARY

logger = logging.getLogger(__name__)

DIAGNOSIS_EVIDENCE_ALLOWED_KEYS = frozenset(
    {
        "experience_id",
        "retrieval",
        "past_context",
        "past_diagnosis",
        "source",
    }
)


class DiagnosisValidationError(ValueError):
    """Raised when a diagnosis output violates canonical taxonomy constraints."""


class DiagnosisAPIError(RuntimeError):
    """Raised when the OpenAI diagnosis backend fails before a valid diagnosis."""


class DiagnosisResponseError(DiagnosisAPIError):
    """Raised when the OpenAI diagnosis response is malformed or fails validation."""


def validate_diagnosis_output(diagnosis_output: dict) -> None:
    """Validate canonical consistency of a diagnoser output payload."""
    failure_type = diagnosis_output.get("failure_type")
    failure_cause = diagnosis_output.get("failure_cause") or {}
    cause_code = failure_cause.get("code")
    affected_module = diagnosis_output.get("affected_module")
    evidence = diagnosis_output.get("evidence")
    confidence = diagnosis_output.get("confidence")

    if failure_type not in FAILURE_VOCABULARY:
        raise DiagnosisValidationError(f"invalid failure_type: {failure_type!r}")

    if cause_code not in FAILURE_VOCABULARY[failure_type]:
        raise DiagnosisValidationError(
            f"invalid failure_cause.code {cause_code!r} for failure_type {failure_type!r}"
        )

    expected_module = FAILURE_MODULES.get(failure_type)
    if affected_module != expected_module:
        raise DiagnosisValidationError(
            f"affected_module {affected_module!r} does not match failure_type {failure_type!r}"
        )

    if not isinstance(evidence, list):
        raise DiagnosisValidationError("evidence must be a list")

    if confidence is None or not isinstance(confidence, (int, float)):
        raise DiagnosisValidationError("confidence must be a number")
    if not 0.0 <= float(confidence) <= 1.0:
        raise DiagnosisValidationError("confidence must be between 0.0 and 1.0")


class FailureDiagnoser(Protocol):
    def diagnose(self, failure_context: dict, diagnosis_evidence: list[dict]) -> dict:
        """Return canonical diagnosis fields without memory_context."""


class MockFailureDiagnoser:
    """Standalone mock diagnoser with fixed canonical output."""

    def diagnose(self, failure_context: dict, diagnosis_evidence: list[dict]) -> dict:
        logger.debug(
            "mock diagnosis failure_id=%s evidence_count=%s",
            failure_context.get("failure_id"),
            len(diagnosis_evidence),
        )
        return {
            "failure_type": "PLANNING",
            "failure_cause": {
                "code": "INVALID_APPROACH",
                "description": "Mock diagnosis for standalone test.",
            },
            "affected_module": "M5",
            "evidence": ["mock evidence"],
            "confidence": 0.9,
        }


def apply_diagnosis_output(diagnosis: dict, diagnosis_output: dict) -> dict:
    """Validate and merge diagnoser output into the diagnosis schema."""
    validate_diagnosis_output(diagnosis_output)
    diagnosis["failure_type"] = diagnosis_output["failure_type"]
    diagnosis["failure_cause"] = dict(diagnosis_output.get("failure_cause") or {})
    diagnosis["affected_module"] = diagnosis_output["affected_module"]
    diagnosis["evidence"] = list(diagnosis_output.get("evidence") or [])
    diagnosis["confidence"] = diagnosis_output["confidence"]
    logger.debug(
        "diagnosis applied failure_type=%s failure_cause=%s affected_module=%s",
        diagnosis["failure_type"],
        diagnosis["failure_cause"].get("code"),
        diagnosis["affected_module"],
    )
    return diagnosis
