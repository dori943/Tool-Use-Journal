"""Action-semantic helpers kept separate from motion-goal representation.

``MotionGoal`` describes only the geometric representation consumed by the
motion planner.  Resource effects such as acquiring, releasing, or exchanging
an end effector remain properties of ``MotionTask.action_type`` (or an explicit
``metadata.operation`` supplied by the Task Planner).

The small compatibility vocabulary below is intentionally isolated in one
module.  Adding a new manipulation action must not require extending
``GoalType`` or changing the motion-goal schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_action(value: object) -> str:
    """Return a stable upper-snake representation for an action label."""

    return str(value or "").strip().upper().replace(":", "_").replace("-", "_")


def task_operation(task: Any) -> str:
    """Read the grounded execution operation, falling back to action_type."""

    metadata = getattr(task, "metadata", {})
    operation = metadata.get("operation") if isinstance(metadata, Mapping) else None
    return normalize_action(operation or getattr(task, "action_type", ""))


def is_ee_exchange_task(task: Any) -> bool:
    return task_operation(task) in {
        "EE_ATTACH",
        "EE_EXCHANGE",
        "EXCHANGE_EE",
        "INITIAL_ATTACH_EE",
        "TERMINAL_RESTORE_EE",
    }


def is_acquire_task(task: Any) -> bool:
    return is_acquire_action(task_operation(task))


def is_acquire_action(value: object) -> bool:
    operation = normalize_action(value)
    return operation in {
        "ACQUIRE",
        "GRASP",
        "PICK",
        "PICK_OBJECT",
        "PICK_TOOL",
    } or operation.startswith("PICK_")


def is_release_task(task: Any) -> bool:
    return is_release_action(task_operation(task))


def is_release_action(value: object) -> bool:
    operation = normalize_action(value)
    return operation in {
        "PLACE",
        "RELEASE",
        "RETURN_TOOL",
        "TERMINAL_RETURN_TOOL",
    } or operation.startswith("PLACE_")


def attaches_target(task: Any) -> bool:
    """Whether successful execution should attach the task target."""

    metadata = getattr(task, "metadata", {})
    if isinstance(metadata, Mapping) and "attach_target" in metadata:
        return bool(metadata["attach_target"])
    return is_acquire_task(task) and task_operation(task) != "PICK_TOOL"


def detaches_target(task: Any) -> bool:
    """Whether successful execution should detach the task target."""

    metadata = getattr(task, "metadata", {})
    if isinstance(metadata, Mapping) and "detach_target" in metadata:
        return bool(metadata["detach_target"])
    return is_release_task(task) and task_operation(task) == "PLACE"


__all__ = [
    "attaches_target",
    "detaches_target",
    "is_acquire_action",
    "is_acquire_task",
    "is_ee_exchange_task",
    "is_release_action",
    "is_release_task",
    "normalize_action",
    "task_operation",
]
