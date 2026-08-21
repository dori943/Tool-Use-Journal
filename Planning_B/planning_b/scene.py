"""Predicted scene state for lazy geometric checks.

Geometric checks must run against the scene *after* preceding subgoal effects,
not the initial scene. The MVP updater derives a deterministic signature from
the completed subgoals and symbolic facts; a real geometry-state updater can
be plugged in through the same protocol.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from planning_b.candidate_provider import Candidate
from planning_b.conditions import FluentKey
from planning_b.models import PlannerAOutput, Subgoal


@dataclass(frozen=True, slots=True)
class SceneState:
    signature: str
    completed: tuple[str, ...] = ()


class SceneStateUpdater(Protocol):
    def initial_scene(self, planner_a: PlannerAOutput) -> SceneState: ...

    def apply_subgoal_effects(
        self,
        scene: SceneState,
        subgoal: Subgoal,
        candidate: Candidate,
        next_facts: frozenset[FluentKey],
    ) -> SceneState: ...


class SymbolicSceneUpdater:
    """Signature = hash(sorted completed subgoals + sorted symbolic facts)."""

    def initial_scene(self, planner_a: PlannerAOutput) -> SceneState:
        signature = planner_a.initial_state.scene_signature or "initial"
        return SceneState(signature=signature, completed=())

    def apply_subgoal_effects(
        self,
        scene: SceneState,
        subgoal: Subgoal,
        candidate: Candidate,
        next_facts: frozenset[FluentKey],
    ) -> SceneState:
        completed = tuple(sorted(set(scene.completed) | {subgoal.subgoal_id}))
        payload = json.dumps(
            [
                list(completed),
                sorted([name, list(args)] for name, args in next_facts),
            ],
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return SceneState(signature=f"sym-{digest}", completed=completed)
