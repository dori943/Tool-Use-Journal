"""IK-branch-preserving strategy search.

This module intentionally finds the first fully connected realization.  It does
not optimize task cost or change the Task Planner's selected order, EE, Tool, or
Grasp.  Joint distance and branch continuity only provide deterministic search
ordering so a locally attractive IK solution cannot prematurely seal a plan.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from tuj.m5_motion.kinematics import IKResult, IKSolutionSet
from tuj.m5_motion.schema import KeyframePlanCandidate, RelativeKeyframeSpec

JointConfig = tuple[float, ...]


def wrapped_joint_delta(source: Sequence[float], target: Sequence[float]) -> np.ndarray:
    """Shortest angular displacement for deterministic ordering/interpolation."""
    if len(source) != len(target):
        raise ValueError("joint configurations must have the same DOF")
    delta = np.asarray(target, dtype=float) - np.asarray(source, dtype=float)
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def wrapped_joint_distance(source: Sequence[float], target: Sequence[float]) -> float:
    return float(np.linalg.norm(wrapped_joint_delta(source, target)))


@dataclass(frozen=True, slots=True)
class EdgePlanResult:
    valid: bool
    joint_path: tuple[JointConfig, ...] = ()
    failure_code: str | None = None
    detail: str = ""
    min_clearance_m: float | None = None

    def __post_init__(self) -> None:
        if self.valid and len(self.joint_path) < 2:
            raise ValueError("a valid edge must contain at least two joint states")
        if not self.valid and not self.failure_code:
            raise ValueError("an invalid edge must include a failure_code")


class EdgePlanner(Protocol):
    def plan(
        self,
        source: JointConfig,
        target: JointConfig,
        source_keyframe: RelativeKeyframeSpec | None,
        target_keyframe: RelativeKeyframeSpec,
    ) -> EdgePlanResult: ...


@dataclass(frozen=True, slots=True)
class SelectedIKNode:
    keyframe: RelativeKeyframeSpec
    solution: IKResult


@dataclass(frozen=True, slots=True)
class ConnectedStrategy:
    strategy_id: str
    nodes: tuple[SelectedIKNode, ...]
    edges: tuple[EdgePlanResult, ...]
    edge_evaluations: int

    @property
    def joint_path(self) -> tuple[JointConfig, ...]:
        merged: list[JointConfig] = []
        for edge in self.edges:
            if not merged:
                merged.extend(edge.joint_path)
            else:
                merged.extend(edge.joint_path[1:])
        return tuple(merged)


@dataclass(frozen=True, slots=True)
class RejectedEdge:
    source_keyframe_id: str
    target_keyframe_id: str
    source_branch_id: str
    target_branch_id: str
    failure_code: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class BranchSelectionResult:
    connected: ConnectedStrategy | None
    rejected_edges: tuple[RejectedEdge, ...] = ()
    failure_code: str | None = None
    detail: str = ""

    @property
    def solved(self) -> bool:
        return self.connected is not None


class FirstFeasibleBranchSelector:
    """Depth-first layered graph search with deterministic branch ordering."""

    def __init__(
        self,
        *,
        max_edge_evaluations: int = 256,
        timeout_s: float = 10.0,
    ) -> None:
        if max_edge_evaluations < 1:
            raise ValueError("max_edge_evaluations must be positive")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._max_edge_evaluations = max_edge_evaluations
        self._timeout_s = timeout_s

    def select(
        self,
        strategy: KeyframePlanCandidate,
        start_joint_config: Sequence[float],
        solution_sets: Sequence[IKSolutionSet],
        edge_planner: EdgePlanner,
    ) -> BranchSelectionResult:
        if len(solution_sets) != len(strategy.keyframes):
            raise ValueError("one IK solution set is required per keyframe")
        empty = [
            keyframe.keyframe_id
            for keyframe, solutions in zip(strategy.keyframes, solution_sets)
            if not solutions.solutions
        ]
        if empty:
            return BranchSelectionResult(
                connected=None,
                failure_code="NO_IK_BRANCH",
                detail=f"keyframes without IK solutions: {empty}",
            )

        started = time.monotonic()
        start_q = tuple(float(value) for value in start_joint_config)
        dof = len(start_q)
        if any(len(solution.qpos) != dof for group in solution_sets for solution in group.solutions):
            raise ValueError("all IK solutions must match the start-state DOF")

        rejected: list[RejectedEdge] = []
        evaluations = 0

        def timed_out() -> bool:
            return time.monotonic() - started >= self._timeout_s

        def ordered_solutions(
            solutions: Sequence[IKResult], previous: IKResult | None, previous_q: JointConfig
        ) -> list[IKResult]:
            previous_branch = previous.branch_id if previous is not None else ""
            return sorted(
                solutions,
                key=lambda solution: (
                    0 if previous_branch and solution.branch_id == previous_branch else 1,
                    wrapped_joint_distance(previous_q, solution.qpos),
                    solution.branch_id,
                    tuple(round(value, 10) for value in solution.qpos),
                ),
            )

        def visit(
            layer_index: int,
            previous_q: JointConfig,
            previous_node: SelectedIKNode | None,
            nodes: list[SelectedIKNode],
            edges: list[EdgePlanResult],
        ) -> tuple[list[SelectedIKNode], list[EdgePlanResult]] | None:
            nonlocal evaluations
            if layer_index == len(strategy.keyframes):
                return list(nodes), list(edges)
            if evaluations >= self._max_edge_evaluations or timed_out():
                return None

            keyframe = strategy.keyframes[layer_index]
            previous_solution = previous_node.solution if previous_node else None
            for solution in ordered_solutions(
                solution_sets[layer_index].solutions,
                previous_solution,
                previous_q,
            ):
                if evaluations >= self._max_edge_evaluations or timed_out():
                    return None
                edge = edge_planner.plan(
                    previous_q,
                    solution.qpos,
                    previous_node.keyframe if previous_node else None,
                    keyframe,
                )
                evaluations += 1
                if not edge.valid:
                    rejected.append(
                        RejectedEdge(
                            source_keyframe_id=(
                                previous_node.keyframe.keyframe_id
                                if previous_node
                                else "CURRENT_STATE"
                            ),
                            target_keyframe_id=keyframe.keyframe_id,
                            source_branch_id=(
                                previous_node.solution.branch_id
                                if previous_node
                                else "CURRENT"
                            ),
                            target_branch_id=solution.branch_id,
                            failure_code=edge.failure_code or "EDGE_INVALID",
                            detail=edge.detail,
                        )
                    )
                    continue
                node = SelectedIKNode(keyframe=keyframe, solution=solution)
                nodes.append(node)
                edges.append(edge)
                downstream = visit(
                    layer_index + 1,
                    tuple(edge.joint_path[-1]),
                    node,
                    nodes,
                    edges,
                )
                if downstream is not None:
                    return downstream
                edges.pop()
                nodes.pop()
            return None

        chosen = visit(0, start_q, None, [], [])
        if chosen is None:
            exhausted = evaluations >= self._max_edge_evaluations or timed_out()
            return BranchSelectionResult(
                connected=None,
                rejected_edges=tuple(rejected),
                failure_code="SEARCH_BUDGET_EXHAUSTED" if exhausted else "NO_CONNECTED_SEQUENCE",
                detail=f"evaluated {evaluations} branch edges",
            )
        nodes, edges = chosen
        return BranchSelectionResult(
            connected=ConnectedStrategy(
                strategy_id=strategy.strategy_id,
                nodes=tuple(nodes),
                edges=tuple(edges),
                edge_evaluations=evaluations,
            ),
            rejected_edges=tuple(rejected),
        )


StateValidator = Callable[[JointConfig, RelativeKeyframeSpec], bool]


@dataclass(slots=True)
class InterpolatingEdgePlanner:
    """Deterministic joint interpolation with per-sample validity checks.

    This remains the direct JOINT connector and a test seam.  Production
    CARTESIAN and SAMPLING_BASED edges use the implementations in
    ``motion_planner.path_planning`` through the same protocol.
    """

    state_validator: StateValidator
    max_joint_step_rad: float = 0.05
    wrap_joints: bool = True

    def __post_init__(self) -> None:
        if self.max_joint_step_rad <= 0:
            raise ValueError("max_joint_step_rad must be positive")

    def plan(
        self,
        source: JointConfig,
        target: JointConfig,
        source_keyframe: RelativeKeyframeSpec | None,
        target_keyframe: RelativeKeyframeSpec,
    ) -> EdgePlanResult:
        delta = (
            wrapped_joint_delta(source, target)
            if self.wrap_joints
            else np.asarray(target, dtype=float) - np.asarray(source, dtype=float)
        )
        steps = max(1, int(math.ceil(float(np.max(np.abs(delta))) / self.max_joint_step_rad)))
        source_values = np.asarray(source, dtype=float)
        path: list[JointConfig] = []
        for index in range(steps + 1):
            state = source_values + delta * (index / steps)
            config = tuple(float(value) for value in state)
            if not self.state_validator(config, target_keyframe):
                return EdgePlanResult(
                    valid=False,
                    failure_code="INTERPOLATED_STATE_INVALID",
                    detail=f"invalid sample {index}/{steps}",
                )
            path.append(config)
        return EdgePlanResult(valid=True, joint_path=tuple(path))


def filter_ik_solutions(
    solutions: IKSolutionSet,
    keyframe: RelativeKeyframeSpec,
    state_validator: StateValidator,
) -> IKSolutionSet:
    """Prune colliding endpoint states while retaining every valid branch."""
    valid_items: list[IKResult] = []
    rejection_summaries: list[str] = []
    check = getattr(state_validator, "check", None)
    for solution in solutions.solutions:
        if callable(check):
            report = check(solution.qpos, keyframe)
            accepted = bool(report.valid)
            if not accepted and len(rejection_summaries) < 3:
                rejection_summaries.append(
                    f"{solution.branch_id}: {report.failure_code}: "
                    f"{report.detail}"
                )
        else:
            accepted = bool(state_validator(solution.qpos, keyframe))
        if accepted:
            valid_items.append(solution)
    valid = tuple(valid_items)
    rejection_detail = (
        f"; rejected examples: {' | '.join(rejection_summaries)}"
        if rejection_summaries
        else ""
    )
    return IKSolutionSet(
        solutions=valid,
        best_position_error_m=solutions.best_position_error_m,
        best_orientation_error_rad=solutions.best_orientation_error_rad,
        attempted_seeds=solutions.attempted_seeds,
        solver_id=solutions.solver_id,
        enumeration_complete=solutions.enumeration_complete,
        detail=(
            f"{len(valid)}/{len(solutions.solutions)} IK branches passed state validity"
            f"{rejection_detail}"
        ),
    )
