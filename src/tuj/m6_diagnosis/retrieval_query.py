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


def _class_from_scene(failure_context: dict, object_id: str) -> str | None:
    """Join object_id to an explicit scene node class; never parse the id string."""
    scene = failure_context.get("scene") or {}
    for node in scene.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        if node.get("id") == object_id:
            node_class = node.get("class")
            return node_class if _is_present(node_class) else None
    return None


def _resolve_target(failure_context: dict) -> dict[str, str | None]:
    """Resolve retrieval target from the object currently being manipulated.

    Priority for object_id:
    1. subgoal.selected_object_id
    2. first present subgoal.target_object_ids entry
    3. null

    object_class comes only from an explicit source matching that choice:
    - selected_object_id → subgoal.selected_object_class (or null)
    - target_object_ids fallback → scene.nodes join (or null)
    """
    subgoal = failure_context.get("subgoal") or {}
    selected_id = subgoal.get("selected_object_id")
    if _is_present(selected_id):
        selected_class = subgoal.get("selected_object_class")
        return {
            "object_id": selected_id,
            "object_class": selected_class if _is_present(selected_class) else None,
        }

    for item in subgoal.get("target_object_ids") or []:
        if _is_present(item):
            return {
                "object_id": item,
                "object_class": _class_from_scene(failure_context, item),
            }

    return {"object_id": None, "object_class": None}


def build_retrieval_query(failure_context: dict) -> dict:
    """Extract only retrieval-relevant fields from the failure context."""
    subgoal = failure_context.get("subgoal") or {}
    task_plan = failure_context.get("task_plan") or {}
    motion_plan = failure_context.get("motion_plan") or {}
    execution = failure_context.get("execution") or {}

    # Failure Context no longer carries verification / violated_predicates.
    # Keep an empty list for retrieval-query schema compatibility with M0
    # context_signature.violated_predicates; empty → treated as missing and
    # excluded from similarity comparison (see context_similarity._is_present).
    return {
        "subgoal_description": subgoal.get("description"),
        "action_type": subgoal.get("action_type"),
        "target": _resolve_target(failure_context),
        "violated_predicates": [],
        "selected_ee": task_plan.get("selected_ee"),
        "selected_tool": task_plan.get("selected_tool"),
        "execution_signature": {
            "motion_planning_status": motion_plan.get("planning_status"),
            "controller_status": execution.get("controller_status"),
        },
    }
