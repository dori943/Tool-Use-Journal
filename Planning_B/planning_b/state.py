"""Immutable, hashable search state."""

from __future__ import annotations

from dataclasses import dataclass
from planning_b.conditions import FluentKey


@dataclass(frozen=True, slots=True)
class SearchState:
    """A node of the unified state space.

    States with identical completed/EE/tool fields are still distinct if group
    bindings, symbolic facts, or the predicted scene differ. Object ownership
    is represented by symbolic ``holding(object)`` facts.
    """

    completed_subgoals: frozenset[str]
    current_ee: str
    held_tool: str | None
    # EE committed per group_id, as a sorted tuple of (group_id, ee) pairs.
    group_ee_bindings: tuple[tuple[str, str], ...]
    symbolic_facts: frozenset[FluentKey]
    # Predicted-scene identity for geometric caching; part of state identity.
    scene_signature: str
    # Only used in explicit rack mode (reserved/unknown/temporary slots).
    rack_signature: tuple[tuple[str, str], ...] | None = None


def with_group_binding(
    bindings: tuple[tuple[str, str], ...], group_id: str, ee: str
) -> tuple[tuple[str, str], ...]:
    merged = dict(bindings)
    merged[group_id] = ee
    return tuple(sorted(merged.items()))
