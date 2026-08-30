"""Failure recovery: no-good learning + re-search from the executed state.

Rules: (1) the actual execution state becomes the new initial state,
(2) completed subgoals are never re-executed, (3) the failed candidate or
transition joins a no-good set scoped to the failure kind, (4) the same
Dijkstra planner runs again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from tuj.m3_taskplanner.conditions import CheckerRegistry
from tuj.m3_taskplanner.feasibility import GeometricFeasibilityChecker
from tuj.m3_taskplanner.models import (
    ExecutionState,
    FailureFeedback,
    FailureType,
    TaskPlannerRequest,
)
from tuj.m3_taskplanner.planner import plan
from tuj.m3_taskplanner.scene import SceneStateUpdater
from tuj.m3_taskplanner.serialization import CostVectorModel, PlanningResult, ReplanTrace
from tuj.m3_taskplanner.suitability import SuitabilityScorer


@dataclass(slots=True)
class NoGoodSet:
    """Bans learned from execution failures.

    - ``global_candidates``: candidate ids banned in every scene
      (scene-independent failures, e.g. EE-tool incompatibility discovered at
      execution time).
    - ``scene_candidates``: (scene_signature, candidate_id) pairs banned only
      for that predicted scene.
    - ``transitions``: banned transition signatures
      (current_ee, held_tool, candidate_id), optionally prefixed with a scene
      signature.
    """

    global_candidates: set[str] = field(default_factory=set)
    scene_candidates: set[tuple[str, str]] = field(default_factory=set)
    transitions: set[tuple[str, ...]] = field(default_factory=set)

    def describe(self) -> list[str]:
        out = [f"global:{cid}" for cid in sorted(self.global_candidates)]
        out += [
            f"scene:{scene}:{cid}" for scene, cid in sorted(self.scene_candidates)
        ]
        out += ["transition:" + "|".join(sig) for sig in sorted(self.transitions)]
        return out


_DEFAULT_SCOPE: dict[FailureType, str] = {
    FailureType.CANDIDATE_INVALID: "global",
    FailureType.RESOURCE_UNAVAILABLE: "global",
    FailureType.TRANSITION_INVALID: "transition",
    FailureType.ATTACHMENT_FAILED: "transition",
    FailureType.SCENE_CHANGED: "none",
}


def apply_failure_feedback(
    no_goods: NoGoodSet,
    feedback: FailureFeedback,
    execution_state: ExecutionState | None = None,
) -> list[str]:
    """Add the failure to the no-good set; returns descriptions of additions.

    The scope follows the feedback (or the failure-type default): a failure
    that only holds in one scene bans (scene, candidate); scene-independent
    failures ban the candidate globally; transition failures ban the
    transition signature. SCENE_CHANGED adds no ban — replanning from the new
    state is enough.
    """
    scope = feedback.scope or _DEFAULT_SCOPE[feedback.failure_type]
    added: list[str] = []
    scene = feedback.scene_signature or (
        execution_state.scene_signature if execution_state is not None else None
    )
    if scope == "none":
        return added
    if scope == "transition":
        if feedback.transition_signature:
            sig = tuple(feedback.transition_signature)
            no_goods.transitions.add(sig)
            added.append("transition:" + "|".join(sig))
        elif feedback.candidate_id is not None:
            # No signature supplied; fall back to the tightest available ban.
            if scene is not None:
                no_goods.scene_candidates.add((scene, feedback.candidate_id))
                added.append(f"scene:{scene}:{feedback.candidate_id}")
            else:
                no_goods.global_candidates.add(feedback.candidate_id)
                added.append(f"global:{feedback.candidate_id}")
        return added
    if feedback.candidate_id is None:
        return added
    if scope == "scene" and scene is not None:
        no_goods.scene_candidates.add((scene, feedback.candidate_id))
        added.append(f"scene:{scene}:{feedback.candidate_id}")
    else:
        no_goods.global_candidates.add(feedback.candidate_id)
        added.append(f"global:{feedback.candidate_id}")
    return added


def _merge_group_bindings_from_previous(
    execution_state: ExecutionState, previous_result: PlanningResult | None
) -> ExecutionState:
    """Groups with an executed subgoal stay bound to the EE actually used."""
    if previous_result is None or previous_result.selected_plan is None:
        return execution_state
    completed = set(execution_state.completed_subgoals)
    bindings = dict(execution_state.group_ee_bindings)
    for assignment in previous_result.selected_plan.candidate_assignments:
        if (
            assignment.group_id is not None
            and assignment.subgoal_id in completed
            and assignment.group_id not in bindings
        ):
            bindings[assignment.group_id] = assignment.ee
    return execution_state.model_copy(update={"group_ee_bindings": bindings})


def replan(
    request: TaskPlannerRequest,
    execution_state: ExecutionState,
    failure_feedback: FailureFeedback,
    previous_result: PlanningResult | None = None,
    *,
    no_goods: NoGoodSet | None = None,
    geometry_checker: GeometricFeasibilityChecker | None = None,
    scene_updater: SceneStateUpdater | None = None,
    checker_registry: CheckerRegistry | None = None,
    suitability_scorer: SuitabilityScorer | Literal["default"] | None = "default",
) -> PlanningResult:
    no_goods = no_goods if no_goods is not None else NoGoodSet()
    added = apply_failure_feedback(no_goods, failure_feedback, execution_state)
    execution_state = _merge_group_bindings_from_previous(
        execution_state, previous_result
    )

    result = plan(
        request,
        geometry_checker=geometry_checker,
        scene_updater=scene_updater,
        checker_registry=checker_registry,
        suitability_scorer=suitability_scorer,
        no_goods=no_goods,
        execution_state=execution_state,
    )

    newly_selected: str | None = None
    new_cost: CostVectorModel | None = None
    if result.selected_plan is not None:
        new_cost = result.selected_plan.cost_vector
        if failure_feedback.subgoal_id is not None:
            for assignment in result.selected_plan.candidate_assignments:
                if assignment.subgoal_id == failure_feedback.subgoal_id:
                    newly_selected = assignment.candidate_id
                    break
    result.replan_trace = ReplanTrace(
        previous_subgoal_id=failure_feedback.subgoal_id,
        previous_candidate_id=failure_feedback.candidate_id,
        failure_reason=failure_feedback.failure_type.value,
        added_no_goods=added,
        newly_selected_candidate_id=newly_selected,
        new_cost_vector=new_cost,
    )
    return result
