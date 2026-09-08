"""Validate post-recovery M5 subgoal_result, map outcomes, and orchestrate M0 append."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .memory_adapter import DEFAULT_MEMORY_PATH, MemoryAdapter, append_experience

VALID_RECOVERY_RESULT_STATUSES = frozenset({"SUCCESS", "FAIL"})

# M5 subgoal_result.status → M0 recovery_summary.outcome.status
RECOVERY_RESULT_TO_OUTCOME = {
    "SUCCESS": "PASS",
    "FAIL": "FAIL",
}


class RecoveryResultError(ValueError):
    """Raised when a recovery_result violates the M5 subgoal_result contract."""


def validate_recovery_result(
    recovery_result: Any,
    *,
    expected_subgoal_id: str | None,
) -> dict[str, Any]:
    """Validate and return a shallow-normalized recovery_result dict.

    Accepts only the M5 ``subgoal_result.json`` contract. Does **not** normalize
    aliases such as FAILED→FAIL or PASS→SUCCESS.
    """
    if recovery_result is None:
        raise RecoveryResultError(
            "recovery_result is required; runtime experiences are not created "
            "without an executed recovery result"
        )
    if not isinstance(recovery_result, Mapping):
        raise RecoveryResultError("recovery_result must be a JSON object/dict")

    result = dict(recovery_result)

    subgoal_id = result.get("subgoal_id")
    if not isinstance(subgoal_id, str) or not subgoal_id.strip():
        raise RecoveryResultError("recovery_result.subgoal_id is required")
    subgoal_id = subgoal_id.strip()

    status = result.get("status")
    if not isinstance(status, str) or not status.strip():
        raise RecoveryResultError("recovery_result.status is required")
    status = status.strip()
    if status not in VALID_RECOVERY_RESULT_STATUSES:
        raise RecoveryResultError(
            "recovery_result.status must be 'SUCCESS' or 'FAIL' "
            f"(got {status!r}); aliases such as FAILED/PASS/OK are rejected"
        )

    if expected_subgoal_id is None or (
        isinstance(expected_subgoal_id, str) and not expected_subgoal_id.strip()
    ):
        raise RecoveryResultError(
            "failure_context.subgoal.subgoal_id is required to validate recovery_result"
        )
    expected = expected_subgoal_id.strip()
    if subgoal_id != expected:
        raise RecoveryResultError(
            f"Recovery result subgoal mismatch: expected {expected}, got {subgoal_id}"
        )

    return {
        "subgoal_id": subgoal_id,
        "status": status,
        "phase": result.get("phase"),
        "failure_code": result.get("failure_code"),
        "detail": result.get("detail"),
    }


def outcome_from_recovery_result(validated_result: Mapping[str, Any]) -> dict[str, Any]:
    """Map validated M5 status to M0 outcome (SUCCESS→PASS, FAIL→FAIL)."""
    status = validated_result.get("status")
    if status not in RECOVERY_RESULT_TO_OUTCOME:
        raise RecoveryResultError(
            f"cannot map recovery_result.status {status!r} to M0 outcome"
        )
    return {
        "status": RECOVERY_RESULT_TO_OUTCOME[status],
        "verification_result": None,
    }


def expected_subgoal_id_from_context(failure_context: Mapping[str, Any] | None) -> str | None:
    if not isinstance(failure_context, Mapping):
        return None
    subgoal = failure_context.get("subgoal") or {}
    if not isinstance(subgoal, Mapping):
        return None
    subgoal_id = subgoal.get("subgoal_id")
    if isinstance(subgoal_id, str) and subgoal_id.strip():
        return subgoal_id.strip()
    return None


def process_recovery_outcome(
    *,
    failure_context: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    recovery: Mapping[str, Any],
    recovery_result: Mapping[str, Any],
    memory_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate recovery_result, build a runtime experience, and append to M0.

    Does not mutate ``failure_context.m5_result`` (initial failure trigger).
    Does not re-execute modules. Intended for Phase B after external orchestration.
    """
    # Lazy import avoids circular dependency with runtime_experience.
    from .runtime_experience import build_runtime_experience

    path = Path(memory_path) if memory_path is not None else DEFAULT_MEMORY_PATH
    expected_subgoal = expected_subgoal_id_from_context(failure_context)
    validated = validate_recovery_result(
        recovery_result,
        expected_subgoal_id=expected_subgoal,
    )
    outcome = outcome_from_recovery_result(validated)

    adapter = MemoryAdapter(memory_path=path)
    existing_ids = {
        item.get("experience_id")
        for item in adapter._load_experiences()
        if isinstance(item, dict) and isinstance(item.get("experience_id"), str)
    }

    experience = build_runtime_experience(
        failure_context=failure_context,
        diagnosis=diagnosis,
        recovery=recovery,
        recovery_result=validated,
        existing_ids=existing_ids,
    )
    stored = append_experience(experience, memory_path=path)

    return {
        "outcome": dict(outcome),
        "recovery_result": deepcopy(validated),
        "experience": stored,
        "experience_id": stored.get("experience_id"),
        "memory_path": str(path),
        "appended": True,
    }
