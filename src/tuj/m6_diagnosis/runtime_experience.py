"""Build compact M0 Failure-Recovery Experiences from runtime recovery outcomes."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from .recovery_outcome import (
    RecoveryResultError,
    expected_subgoal_id_from_context,
    outcome_from_recovery_result,
    validate_recovery_result,
)
from .recovery_router import normalize_recovery_action
from .retrieval_query import _resolve_target


def _utc_now_iso() -> str:
    """Match M0 object_knowledge timestamp convention (UTC ISO-8601 seconds)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(value: Any, *, fallback: str = "unknown") -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return fallback
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return slug or fallback


def generate_runtime_experience_id(
    *,
    failure_context: Mapping[str, Any],
    existing_ids: set[str] | None = None,
    now: datetime | None = None,
) -> str:
    """Build a collision-resistant runtime experience_id.

    Format: ``RT-<task>-<subgoal>-<YYYYMMDDTHHMMSSZ>`` with numeric suffix on clash.
    """
    task = (failure_context.get("task") or {}) if isinstance(failure_context, Mapping) else {}
    subgoal = (failure_context.get("subgoal") or {}) if isinstance(failure_context, Mapping) else {}
    task_slug = _slug(task.get("task_id") if isinstance(task, Mapping) else None, fallback="task")
    # Prefer detail execution id in experience_id (failure unit), not parent id.
    id_source = None
    if isinstance(subgoal, Mapping):
        id_source = subgoal.get("detail_id") or subgoal.get("subgoal_id")
    subgoal_slug = _slug(id_source, fallback="subgoal")
    stamp_dt = now or datetime.now(timezone.utc)
    stamp = stamp_dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"RT-{task_slug}-{subgoal_slug}-{stamp}"
    known = existing_ids or set()
    if base not in known:
        return base
    for index in range(2, 10_000):
        candidate = f"{base}-{index}"
        if candidate not in known:
            return candidate
    raise RuntimeError(f"unable to allocate unique experience_id near {base}")


def _compact_recovery_action(action: Mapping[str, Any] | None) -> dict[str, Any]:
    """Store seed-compatible action fields (recovery_type; no target_module)."""
    normalized = normalize_recovery_action(deepcopy(action) if action else {})
    target = normalized.get("target") if isinstance(normalized.get("target"), Mapping) else {}
    parameters = normalized.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    return {
        "target": {
            "subgoal_id": target.get("subgoal_id"),
            "object_id": target.get("object_id"),
            "property": target.get("property"),
            "relation": target.get("relation"),
            "ee_id": target.get("ee_id"),
            "tool_id": target.get("tool_id"),
        },
        "parameters": dict(parameters),
        "recovery_type": normalized.get("recovery_type"),
    }


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _resolve_signature_parent_detail(subgoal: Mapping[str, Any]) -> tuple[Any, Any]:
    """Map Failure Context subgoal fields onto M0 parent/detail signature ids.

    Failure Context keeps ``subgoal_id`` as the M5 execution unit (often a detail
    id) for lifecycle validation. M0 stores parent ``subgoal_id`` separately from
    ``detail_id``. Parent/detail values come from adapter-resolved M2 fields only;
    never parse ids from strings.
    """
    parent_id = subgoal.get("parent_subgoal_id")
    detail_id = subgoal.get("detail_id")
    exec_id = subgoal.get("subgoal_id")

    if _present(parent_id):
        signature_subgoal_id = parent_id.strip() if isinstance(parent_id, str) else parent_id
    elif not _present(detail_id):
        # No explicit detail row — execution id is parent-level when present.
        signature_subgoal_id = exec_id if _present(exec_id) else None
    else:
        # detail_id known but parent missing: do not invent a parent id.
        signature_subgoal_id = None

    if detail_id is None and "detail_id" not in subgoal:
        signature_detail_id = None
    elif isinstance(detail_id, str) and detail_id.strip():
        signature_detail_id = detail_id.strip()
    else:
        signature_detail_id = detail_id if detail_id is None else None

    return signature_subgoal_id, signature_detail_id


def build_context_signature(failure_context: Mapping[str, Any]) -> dict[str, Any]:
    """Compact retrieval signature from the *initial* failure context only."""
    if not isinstance(failure_context, Mapping):
        failure_context = {}
    task = failure_context.get("task") or {}
    subgoal = failure_context.get("subgoal") or {}
    task_plan = failure_context.get("task_plan") or {}
    motion_plan = failure_context.get("motion_plan") or {}
    execution = failure_context.get("execution") or {}
    if not isinstance(task, Mapping):
        task = {}
    if not isinstance(subgoal, Mapping):
        subgoal = {}
    if not isinstance(task_plan, Mapping):
        task_plan = {}
    if not isinstance(motion_plan, Mapping):
        motion_plan = {}
    if not isinstance(execution, Mapping):
        execution = {}

    signature_subgoal_id, signature_detail_id = _resolve_signature_parent_detail(subgoal)

    return {
        "task_id": task.get("task_id"),
        "subgoal_id": signature_subgoal_id,
        "subgoal_description": subgoal.get("description"),
        "detail_id": signature_detail_id,
        # Detail-level action_type from Failure Context (M2 detail.action_type).
        "action_type": subgoal.get("action_type"),
        "target": _resolve_target(dict(failure_context)),
        # Compatibility only — Failure Context no longer carries predicates.
        "violated_predicates": [],
        "selected_ee": task_plan.get("selected_ee"),
        "selected_tool": task_plan.get("selected_tool"),
        "execution_signature": {
            "motion_planning_status": motion_plan.get("planning_status"),
            "controller_status": execution.get("controller_status"),
        },
    }


def build_diagnosis_summary(diagnosis: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(diagnosis, Mapping):
        diagnosis = {}
    failure_cause = diagnosis.get("failure_cause") or {}
    if not isinstance(failure_cause, Mapping):
        failure_cause = {}
    return {
        # Offline provenance only; runtime has no external source label.
        "source_failure_type": None,
        "failure_type": diagnosis.get("failure_type"),
        "failure_cause": {
            "code": failure_cause.get("code"),
            "description": failure_cause.get("description"),
        },
        "affected_module": diagnosis.get("affected_module"),
        "confidence": diagnosis.get("confidence"),
    }


def build_recovery_summary(
    recovery: Mapping[str, Any] | None,
    *,
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(recovery, Mapping):
        recovery = {}
    action = recovery.get("action") if isinstance(recovery.get("action"), Mapping) else {}
    routing = recovery.get("routing") if isinstance(recovery.get("routing"), Mapping) else {}
    return {
        "recovery_category": recovery.get("recovery_category"),
        "action": _compact_recovery_action(action),
        "changes": [],
        "routing": {
            "restart_from": routing.get("restart_from"),
            "rerun_modules": list(routing.get("rerun_modules") or []),
            "invalidate": list(routing.get("invalidate") or []),
        },
        "outcome": {
            "status": outcome.get("status"),
            "verification_result": None,
        },
    }


class RuntimeExperienceBuilder:
    """Assemble one M0-compatible experience from Phase-B recovery inputs."""

    def build(
        self,
        *,
        failure_context: Mapping[str, Any],
        diagnosis: Mapping[str, Any],
        recovery: Mapping[str, Any],
        recovery_result: Mapping[str, Any],
        experience_id: str | None = None,
        existing_ids: set[str] | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        if recovery_result is None:
            raise RecoveryResultError(
                "recovery_result is required; runtime experiences are not created "
                "without an executed recovery result"
            )

        expected_subgoal = expected_subgoal_id_from_context(failure_context)
        validated = validate_recovery_result(
            recovery_result,
            expected_subgoal_id=expected_subgoal,
        )
        outcome = outcome_from_recovery_result(validated)
        timestamp = created_at if created_at is not None else _utc_now_iso()
        exp_id = experience_id or generate_runtime_experience_id(
            failure_context=failure_context,
            existing_ids=existing_ids,
        )

        return {
            "experience_id": exp_id,
            "context_signature": build_context_signature(failure_context),
            "diagnosis_summary": build_diagnosis_summary(diagnosis),
            "recovery_summary": build_recovery_summary(recovery, outcome=outcome),
            "metadata": {
                "source": "runtime",
                "dataset": None,
                "source_episode": None,
                "created_at": timestamp,
                "updated_at": timestamp,
                "reuse_count": 0,
            },
        }


def build_runtime_experience(
    *,
    failure_context: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    recovery: Mapping[str, Any],
    recovery_result: Mapping[str, Any],
    experience_id: str | None = None,
    existing_ids: set[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    return RuntimeExperienceBuilder().build(
        failure_context=failure_context,
        diagnosis=diagnosis,
        recovery=recovery,
        recovery_result=recovery_result,
        experience_id=experience_id,
        existing_ids=existing_ids,
        created_at=created_at,
    )
