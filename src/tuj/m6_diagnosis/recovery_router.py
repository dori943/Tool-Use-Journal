"""Recovery routing interface, validation, and mock implementation for M6."""

from __future__ import annotations

import logging
from typing import Protocol

from .taxonomy import (
    FAILURE_CAUSE_TO_RECOVERY_ACTION,
    FAILURE_MODULES,
    FAILURE_TYPE_TO_RECOVERY_CATEGORY,
    RECOVERY_ACTION_MODULES,
    RECOVERY_CATEGORY_DEFAULT_ACTION,
    RECOVERY_ROUTING_PROFILES,
    RECOVERY_VOCABULARY,
    VALID_ROUTING_MODULES,
)

logger = logging.getLogger(__name__)

VALID_DECISION_MODES = frozenset({"EXPERIENCE_GUIDED", "DIAGNOSIS_GUIDED"})

# Local preferred recovery categories per failure_type (reuse taxonomy mappings;
# allow closely related same-module categories where taxonomy already implies them).
_LOCAL_RECOVERY_CATEGORIES = {
    "TASK_DECOMPOSITION": frozenset({"REPLAN_SUBGOAL"}),
    "PERCEPTION_GROUNDING": frozenset({"REOBSERVE", "UPDATE_SCENE"}),
    "METRIC_REASONING": frozenset({"REMEASURE"}),
    "EE_SELECTION": frozenset({"RESELECT_EE"}),
    "PLANNING": frozenset({"REPLAN_MOTION"}),
    "EXECUTION_CONTROL": frozenset({"RETRY_EXECUTION"}),
    "ENVIRONMENT_CHANGE": frozenset({"UPDATE_SCENE", "REOBSERVE"}),
}


class RecoveryValidationError(ValueError):
    """Raised when a recovery output violates canonical taxonomy constraints."""


class RecoveryCoherenceError(RecoveryValidationError):
    """Raised when recovery is canonically valid but inconsistent with diagnosis/context."""


class RecoveryAPIError(RuntimeError):
    """Raised when the OpenAI recovery backend fails before a valid recovery decision."""


class RecoveryResponseError(RecoveryAPIError):
    """Raised when the OpenAI recovery response is malformed or fails validation."""


def _category_for_recovery_type(recovery_type: str) -> str | None:
    for category, actions in RECOVERY_VOCABULARY.items():
        if recovery_type in actions:
            return category
    return None


def _history_retry_count(failure_context: dict) -> int:
    history = failure_context.get("history") or {}
    retry_count = history.get("retry_count")
    if isinstance(retry_count, bool):
        return int(retry_count)
    if isinstance(retry_count, int):
        return max(retry_count, 0)
    if isinstance(retry_count, float) and retry_count.is_integer():
        return max(int(retry_count), 0)
    return 0


def _previous_recoveries(failure_context: dict) -> list[dict]:
    history = failure_context.get("history") or {}
    previous = history.get("previous_recoveries") or []
    if not isinstance(previous, list):
        return []
    return [item for item in previous if isinstance(item, dict)]


def _previous_outcomes(failure_context: dict) -> list:
    history = failure_context.get("history") or {}
    previous = history.get("previous_outcomes") or []
    return list(previous) if isinstance(previous, list) else []


def is_first_recovery_attempt(failure_context: dict) -> bool:
    """True when history shows no prior recovery attempts."""
    return _history_retry_count(failure_context) == 0 and not _previous_recoveries(
        failure_context
    )


def _outcome_status(value) -> str | None:
    if isinstance(value, dict):
        status = value.get("status")
        return status if isinstance(status, str) else None
    if isinstance(value, str):
        return value
    return None


def _is_local_past_recovery(item: dict, *, preferred_category: str | None, affected_module: str | None) -> bool:
    category = item.get("recovery_category")
    action = normalize_recovery_action(item.get("action"))
    recovery_type = action.get("recovery_type") or item.get("recovery_type")
    target_module = action.get("target_module") or item.get("target_module")
    if preferred_category and category == preferred_category:
        return True
    if preferred_category and recovery_type in RECOVERY_VOCABULARY.get(preferred_category, ()):
        return True
    if affected_module and target_module == affected_module:
        return True
    return False


def has_prior_local_recovery_failure(failure_context: dict, diagnosis: dict) -> bool:
    """True when history records a failed local recovery for the diagnosed module/category."""
    failure_type = diagnosis.get("failure_type")
    preferred_category = FAILURE_TYPE_TO_RECOVERY_CATEGORY.get(failure_type)
    affected_module = diagnosis.get("affected_module") or FAILURE_MODULES.get(failure_type)
    previous = _previous_recoveries(failure_context)
    outcomes = _previous_outcomes(failure_context)

    for index, item in enumerate(previous):
        if not _is_local_past_recovery(
            item,
            preferred_category=preferred_category,
            affected_module=affected_module,
        ):
            continue
        status = _outcome_status(item.get("outcome"))
        if status is None and index < len(outcomes):
            status = _outcome_status(outcomes[index])
        if status == "FAIL":
            return True
    return False


def escalation_is_justified(failure_context: dict, diagnosis: dict) -> bool:
    """ESCALATE_REPLAN is allowed only with explicit history-based justification.

    Runtime FAIL entries in recovery_evidence alone do NOT justify escalation.
    """
    if is_first_recovery_attempt(failure_context):
        return False
    # After at least one attempt, require a prior local recovery FAIL.
    return has_prior_local_recovery_failure(failure_context, diagnosis)


def local_recovery_categories_for_failure(failure_type: str) -> frozenset[str]:
    return _LOCAL_RECOVERY_CATEGORIES.get(failure_type, frozenset())


def validate_diagnosis_recovery_coherence(
    failure_context: dict,
    diagnosis: dict,
    recovery: dict,
) -> None:
    """Validate semantic coherence between diagnosis/context and recovery decision.

    Does not overwrite recovery. Raises RecoveryCoherenceError on violation.
    """
    failure_type = diagnosis.get("failure_type")
    affected_module = diagnosis.get("affected_module")
    expected_module = FAILURE_MODULES.get(failure_type)
    preferred_category = FAILURE_TYPE_TO_RECOVERY_CATEGORY.get(failure_type)
    local_categories = local_recovery_categories_for_failure(failure_type)

    recovery_category = recovery.get("recovery_category")
    action = normalize_recovery_action(recovery.get("action"))
    recovery_type = action.get("recovery_type")
    target_module = action.get("target_module")
    routing = recovery.get("routing") or {}

    if failure_type not in _LOCAL_RECOVERY_CATEGORIES:
        raise RecoveryCoherenceError(
            f"unsupported failure_type for coherence checks: {failure_type!r}"
        )

    if affected_module is None:
        affected_module = expected_module
    if expected_module is not None and affected_module != expected_module:
        raise RecoveryCoherenceError(
            f"diagnosis.affected_module {affected_module!r} inconsistent with "
            f"failure_type {failure_type!r}"
        )

    if recovery_category == "ESCALATE_REPLAN":
        if not escalation_is_justified(failure_context, diagnosis):
            raise RecoveryCoherenceError(
                "ESCALATE_REPLAN is not justified: first-attempt or missing prior "
                "local recovery FAIL evidence requires a local recovery in the "
                f"affected module ({affected_module}); preferred category is "
                f"{preferred_category}"
            )
        return

    if recovery_category not in local_categories:
        raise RecoveryCoherenceError(
            f"recovery_category {recovery_category!r} is not a local recovery for "
            f"failure_type {failure_type!r} / affected_module {affected_module!r}; "
            f"allowed local categories={sorted(local_categories)}"
        )

    if affected_module is not None and target_module != affected_module:
        raise RecoveryCoherenceError(
            f"action.target_module {target_module!r} does not match diagnosed "
            f"affected_module {affected_module!r} for local recovery"
        )

    restart_from = routing.get("restart_from")
    rerun_modules = routing.get("rerun_modules") or []
    if affected_module is not None:
        if restart_from != affected_module:
            raise RecoveryCoherenceError(
                f"routing.restart_from {restart_from!r} escalates beyond diagnosed "
                f"affected_module {affected_module!r} without ESCALATE_REPLAN justification"
            )
        if affected_module not in rerun_modules:
            raise RecoveryCoherenceError(
                f"routing.rerun_modules {rerun_modules!r} must include diagnosed "
                f"affected_module {affected_module!r}"
            )

    # Prefer that recovery_type belongs to the preferred local category family.
    if preferred_category and recovery_type not in RECOVERY_VOCABULARY.get(
        preferred_category, ()
    ):
        # Still allow other local-family categories listed in local_categories.
        owning = _category_for_recovery_type(recovery_type)
        if owning not in local_categories:
            raise RecoveryCoherenceError(
                f"recovery_type {recovery_type!r} is not local for failure_type "
                f"{failure_type!r}"
            )


def resolve_recovery_decision(failure_type: str, cause_code: str) -> tuple[str, str, str]:
    """Map diagnosis fields to canonical recovery category, type, and target module."""
    recovery_category = FAILURE_TYPE_TO_RECOVERY_CATEGORY.get(failure_type)
    if recovery_category is None:
        raise RecoveryValidationError(f"unsupported failure_type for recovery: {failure_type!r}")

    recovery_type = FAILURE_CAUSE_TO_RECOVERY_ACTION.get(cause_code)
    if recovery_type is None:
        recovery_type = RECOVERY_CATEGORY_DEFAULT_ACTION[recovery_category]
    elif recovery_type not in RECOVERY_VOCABULARY.get(recovery_category, ()):
        mapped_category = _category_for_recovery_type(recovery_type)
        if mapped_category is not None:
            recovery_category = mapped_category

    if recovery_type not in RECOVERY_VOCABULARY.get(recovery_category, ()):
        raise RecoveryValidationError(
            f"recovery_type {recovery_type!r} does not belong to recovery_category {recovery_category!r}"
        )

    target_module = RECOVERY_ACTION_MODULES.get(recovery_type)
    if target_module is None:
        raise RecoveryValidationError(
            f"missing target_module mapping for recovery_type {recovery_type!r}"
        )

    return recovery_category, recovery_type, target_module


def build_recovery_routing(recovery_type: str, target_module: str) -> dict:
    """Build canonical routing fields for a recovery action."""
    profile = RECOVERY_ROUTING_PROFILES.get(recovery_type)
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


def normalize_recovery_action(action: dict | None) -> dict:
    """Normalize recovery action keys; accept legacy action_type on read only."""
    normalized = dict(action or {})
    if "recovery_type" not in normalized and "action_type" in normalized:
        normalized["recovery_type"] = normalized.pop("action_type")
    elif "action_type" in normalized:
        # Canonical key wins; drop legacy duplicate if both somehow exist.
        normalized.pop("action_type", None)
    return normalized


def build_past_recoveries(recovery_evidence: list[dict]) -> list[dict]:
    """Summarize selected past recoveries for experience-guided guidance.

    Preserves outcome/source so logs can distinguish Runtime PASS/FAIL from
    Offline NOT_EXECUTED priors without changing retrieval or selection policy.
    """
    past_recoveries: list[dict] = []
    for item in recovery_evidence:
        past_recovery = item.get("past_recovery") or {}
        action = normalize_recovery_action(past_recovery.get("action"))
        outcome = item.get("outcome")
        past_recoveries.append(
            {
                "experience_id": item.get("experience_id"),
                "recovery_category": past_recovery.get("recovery_category"),
                "recovery_type": action.get("recovery_type"),
                "routing": dict(past_recovery.get("routing") or {}),
                "outcome": dict(outcome) if isinstance(outcome, dict) else {},
                "source": item.get("source"),
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
    action = normalize_recovery_action(recovery.get("action"))
    recovery_type = action.get("recovery_type")
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

    if recovery_type not in RECOVERY_VOCABULARY[recovery_category]:
        raise RecoveryValidationError(
            f"recovery_type {recovery_type!r} does not belong to recovery_category {recovery_category!r}"
        )

    expected_module = RECOVERY_ACTION_MODULES.get(recovery_type)
    if target_module != expected_module:
        raise RecoveryValidationError(
            f"target_module {target_module!r} does not match recovery_type {recovery_type!r}"
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

    profile = RECOVERY_ROUTING_PROFILES.get(recovery_type)
    if profile is not None:
        if restart_from != profile["restart_from"]:
            raise RecoveryValidationError(
                f"routing.restart_from {restart_from!r} inconsistent with recovery_type {recovery_type!r}"
            )
        if list(rerun_modules) != list(profile["rerun_modules"]):
            raise RecoveryValidationError(
                f"routing.rerun_modules {rerun_modules!r} inconsistent with recovery_type {recovery_type!r}"
            )
        if list(invalidate) != list(profile.get("invalidate") or []):
            raise RecoveryValidationError(
                f"routing.invalidate {invalidate!r} inconsistent with recovery_type {recovery_type!r}"
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

        recovery_category, recovery_type, target_module = resolve_recovery_decision(
            failure_type,
            cause_code,
        )
        routing = build_recovery_routing(recovery_type, target_module)

        recovery_output = {
            "recovery_category": recovery_category,
            "action": {
                "recovery_type": recovery_type,
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
    recovery["action"] = normalize_recovery_action(recovery_output.get("action"))
    recovery["routing"] = dict(recovery_output.get("routing") or {})
    recovery["guidance"]["past_recoveries"] = list(recovery_output.get("past_recoveries") or [])
    validate_recovery_output(recovery)
    logger.debug(
        "recovery applied category=%s recovery_type=%s target_module=%s restart_from=%s",
        recovery["recovery_category"],
        recovery["action"].get("recovery_type"),
        recovery["action"].get("target_module"),
        recovery["routing"].get("restart_from"),
    )
    return recovery
