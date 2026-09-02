"""Build a retrieval query from the current failure context."""

from __future__ import annotations

from typing import Any


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def _first_present_object_id(subgoal: dict) -> str | None:
    selected = subgoal.get("selected_object_id")
    if _is_present(selected):
        return selected
    for item in subgoal.get("target_object_ids") or []:
        if _is_present(item):
            return item
    return None


def _first_present_object_class(subgoal: dict) -> str | None:
    selected = subgoal.get("selected_object_class")
    if _is_present(selected):
        return selected
    return None


def build_retrieval_query(failure_context: dict) -> dict:
    """Extract only retrieval-relevant fields from the failure context."""
    subgoal = failure_context.get("subgoal") or {}
    verification = failure_context.get("verification") or {}
    task_plan = failure_context.get("task_plan") or {}
    motion_plan = failure_context.get("motion_plan") or {}
    execution = failure_context.get("execution") or {}

    return {
        "subgoal_description": subgoal.get("description"),
        "action_type": subgoal.get("action_type"),
        "target": {
            "object_id": _first_present_object_id(subgoal),
            "object_class": _first_present_object_class(subgoal),
        },
        "violated_predicates": list(verification.get("violated_predicates") or []),
        "selected_ee": task_plan.get("selected_ee"),
        "selected_tool": task_plan.get("selected_tool"),
        "execution_signature": {
            "motion_planning_status": motion_plan.get("planning_status"),
            "controller_status": execution.get("controller_status"),
        },
    }
