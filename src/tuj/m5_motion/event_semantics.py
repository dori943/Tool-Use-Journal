"""Semantic predicates shared by motion-plan construction and execution."""

from __future__ import annotations

from enum import Enum


OBJECT_STATE_MUTATING_EVENT_NAMES = frozenset(
    {
        "GRIPPER_OPEN",
        "GRIPPER_CLOSE",
        "SUCTION_ON",
        "SUCTION_OFF",
        "ATTACH_OBJECT",
        "DETACH_OBJECT",
    }
)


def requires_endpoint_convergence(event_type: str | Enum) -> bool:
    """Return whether an object-state event must wait for endpoint settling."""

    value = event_type.value if isinstance(event_type, Enum) else event_type
    return str(value) in OBJECT_STATE_MUTATING_EVENT_NAMES
