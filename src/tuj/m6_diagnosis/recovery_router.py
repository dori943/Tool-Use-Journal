"""Recovery routing interface, validation, and mock implementation for M6."""

from __future__ import annotations

import logging
from typing import Protocol

from .taxonomy import (
    FAILURE_CAUSE_TO_RECOVERY_ACTION,
    FAILURE_TYPE_TO_RECOVERY_CATEGORY,
    RECOVERY_ACTION_MODULES,
    RECOVERY_CATEGORY_DEFAULT_ACTION,
    RECOVERY_ROUTING_PROFILES,
    RECOVERY_VOCABULARY,
    VALID_ROUTING_MODULES,
)

logger = logging.getLogger(__name__)

VALID_DECISION_MODES = frozenset({"EXPERIENCE_GUIDED", "DIAGNOSIS_GUIDED"})


class RecoveryValidationError(ValueError):
    """Raised when a recovery output violates canonical taxonomy constraints."""


class RecoveryAPIError(RuntimeError):
    """Raised when the OpenAI recovery backend fails before a valid recovery decision."""


class RecoveryResponseError(RecoveryAPIError):
    """Raised when the OpenAI recovery response is malformed or fails validation."""


def _category_for_action(action_type: str) -> str | None:
    for category, actions in RECOVERY_VOCABULARY.items():
        if action_type in actions:
            return category
    return None


def resolve_recovery_decision(failure_type: str, cause_code: str) -> tuple[str, str, str]:
    """Map diagnosis fields to canonical recovery category, action, and target module."""
    recovery_category = FAILURE_TYPE_TO_RECOVERY_CATEGORY.get(failure_type)
    if recovery_category is None:
        raise RecoveryValidationError(f"unsupported failure_type for recovery: {failure_type!r}")

    action_type = FAILURE_CAUSE_TO_RECOVERY_ACTION.get(cause_code)
    if action_type is None:
        action_type = RECOVERY_CATEGORY_DEFAULT_ACTION[recovery_category]
    elif action_type not in RECOVERY_VOCABULARY.get(recovery_category, ()):
        mapped_category = _category_for_action(action_type)
        if mapped_category is not None:
            recovery_category = mapped_category

    if action_type not in RECOVERY_VOCABULARY.get(recovery_category, ()):
        raise RecoveryValidationError(
            f"action_type {action_type!r} does not belong to recovery_category {recovery_category!r}"
        )

    target_module = RECOVERY_ACTION_MODULES.get(action_type)
    if target_module is None:
        raise RecoveryValidationError(f"missing target_module mapping for action_type {action_type!r}")

    return recovery_category, action_type, target_module


def build_recovery_routing(action_type: str, target_module: str) -> dict:
    """Build canonical routing fields for a recovery action."""
    profile = RECOVERY_ROUTING_PROFILES.get(action_type)
    if profile is not None:
        return {
            "restart_from": profile["restart_from"],
            "rerun_modules": list(profile["rerun_modules"]),
            "invalidate": list(profile.get("invalidate") or []),
        }
    return {
        "restart_from": target_module,
        "rerun_modules": [target_module],
        "invalidate": [],
    }


def build_recovery_target(failure_context: dict) -> dict:
    """Extract canonical action target fields from the current failure context."""
    subgoal = failure_context.get("subgoal") or {}
    task_plan = failure_context.get("task_plan") or {}
    return {
        "subgoal_id": subgoal.get("subgoal_id"),
        "object_id": subgoal.get("selected_object_id"),
        "property": None,
        "relation": None,
        "ee_id": task_plan.get("selected_ee"),
        "tool_id": task_plan.get("selected_tool"),
    }


def build_past_recoveries(recovery_evidence: list[dict]) -> list[dict]:
    """Summarize selected past recoveries for experience-guided guidance."""
    past_recoveries: list[dict] = []
    for item in recovery_evidence:
        past_recovery = item.get("past_recovery") or {}
        action = past_recovery.get("action") or {}
        past_recoveries.append(
            {
                "experience_id": item.get("experience_id"),
                "recovery_category": past_recovery.get("recovery_category"),
                "action_type": action.get("action_type"),
                "routing": dict(past_recovery.get("routing") or {}),
            }
        )
    return past_recoveries


def validate_recovery_output(recovery: dict) -> None:
    """Validate canonical consistency of a full recovery payload."""
    decision_mode = recovery.get("decision_mode")
    guidance = recovery.get("guidance") or {}
    experience_ids = guidance.get("experience_ids")
    recovery_evidence = guidance.get("recovery_evidence")
    recovery_category = recovery.get("recovery_category")
    action = recovery.get("action") or {}
    action_type = action.get("action_type")
    target_module = action.get("target_module")
    routing = recovery.get("routing") or {}
    restart_from = routing.get("restart_from")
    rerun_modules = routing.get("rerun_modules")
    invalidate = routing.get("invalidate")

    if decision_mode not in VALID_DECISION_MODES:
        raise RecoveryValidationError(f"invalid decision_mode: {decision_mode!r}")

    if not isinstance(experience_ids, list):
        raise RecoveryValidationError("guidance.experience_ids must be a list")

    if not isinstance(recovery_evidence, list):
        raise RecoveryValidationError("guidance.recovery_evidence must be a list")

    if recovery_category not in RECOVERY_VOCABULARY:
        raise RecoveryValidationError(f"invalid recovery_category: {recovery_category!r}")

    if action_type not in RECOVERY_VOCABULARY[recovery_category]:
        raise RecoveryValidationError(
            f"action_type {action_type!r} does not belong to recovery_category {recovery_category!r}"
        )

    expected_module = RECOVERY_ACTION_MODULES.get(action_type)
    if target_module != expected_module:
        raise RecoveryValidationError(
            f"target_module {target_module!r} does not match action_type {action_type!r}"
        )

    if not isinstance(rerun_modules, list):
        raise RecoveryValidationError("routing.rerun_modules must be a list")

    if restart_from not in VALID_ROUTING_MODULES:
        raise RecoveryValidationError(f"invalid routing.restart_from: {restart_from!r}")

    for module in rerun_modules:
        if module not in VALID_ROUTING_MODULES:
            raise RecoveryValidationError(f"invalid routing.rerun_modules entry: {module!r}")

    if not isinstance(invalidate, list):
        raise RecoveryValidationError("routing.invalidate must be a list")

    for module in invalidate:
        if module not in VALID_ROUTING_MODULES:
            raise RecoveryValidationError(f"invalid routing.invalidate entry: {module!r}")

    profile = RECOVERY_ROUTING_PROFILES.get(action_type)
    if profile is not None:
        if restart_from != profile["restart_from"]:
            raise RecoveryValidationError(
                f"routing.restart_from {restart_from!r} inconsistent with action_type {action_type!r}"
            )
        if list(rerun_modules) != list(profile["rerun_modules"]):
            raise RecoveryValidationError(
                f"routing.rerun_modules {rerun_modules!r} inconsistent with action_type {action_type!r}"
            )
        if list(invalidate) != list(profile.get("invalidate") or []):
            raise RecoveryValidationError(
                f"routing.invalidate {invalidate!r} inconsistent with action_type {action_type!r}"
            )
    else:
        if target_module not in rerun_modules:
            raise RecoveryValidationError(
                f"target_module {target_module!r} must appear in routing.rerun_modules"
            )
        if restart_from != target_module:
            raise RecoveryValidationError(
                f"routing.restart_from {restart_from!r} must match action.target_module {target_module!r}"
            )

    if decision_mode == "EXPERIENCE_GUIDED":
        evidence_ids = {
            item.get("experience_id")
            for item in recovery_evidence
            if item.get("experience_id") is not None
        }
        declared_ids = set(experience_ids)
        if not declared_ids:
            raise RecoveryValidationError(
                "EXPERIENCE_GUIDED recovery requires non-empty guidance.experience_ids"
            )
        if evidence_ids != declared_ids:
            raise RecoveryValidationError(
                "guidance.experience_ids must match guidance.recovery_evidence experience_id set"
            )


class RecoveryRouter(Protocol):
    def route(
        self,
        failure_context: dict,
        diagnosis: dict,
        decision_mode: str,
        recovery_evidence: list[dict],
    ) -> dict:
        """Return recovery decision fields without guidance selection metadata."""


class MockRecoveryRouter:
    """Standalone mock recovery router driven by canonical diagnosis-to-recovery mapping."""

    def route(
        self,
        failure_context: dict,
        diagnosis: dict,
        decision_mode: str,
        recovery_evidence: list[dict],
    ) -> dict:
        failure_type = diagnosis.get("failure_type")
        failure_cause = diagnosis.get("failure_cause") or {}
        cause_code = failure_cause.get("code")

        logger.debug(
            "mock recovery route failure_id=%s decision_mode=%s evidence_count=%s",
            failure_context.get("failure_id"),
            decision_mode,
            len(recovery_evidence),
        )

        recovery_category, action_type, target_module = resolve_recovery_decision(
            failure_type,
            cause_code,
        )
        routing = build_recovery_routing(action_type, target_module)

        recovery_output = {
            "recovery_category": recovery_category,
            "action": {
                "action_type": action_type,
                "target_module": target_module,
                "target": build_recovery_target(failure_context),
                "parameters": {},
            },
            "routing": routing,
        }

        if decision_mode == "EXPERIENCE_GUIDED":
            recovery_output["past_recoveries"] = build_past_recoveries(recovery_evidence)
        else:
            recovery_output["past_recoveries"] = []

        return recovery_output


def apply_recovery_output(recovery: dict, recovery_output: dict) -> dict:
    """Merge router output into the recovery schema and validate."""
    recovery["recovery_category"] = recovery_output["recovery_category"]
    recovery["action"] = dict(recovery_output.get("action") or {})
    recovery["routing"] = dict(recovery_output.get("routing") or {})
    recovery["guidance"]["past_recoveries"] = list(recovery_output.get("past_recoveries") or [])
    validate_recovery_output(recovery)
    logger.debug(
        "recovery applied category=%s action_type=%s target_module=%s restart_from=%s",
        recovery["recovery_category"],
        recovery["action"].get("action_type"),
        recovery["action"].get("target_module"),
        recovery["routing"].get("restart_from"),
    )
    return recovery
