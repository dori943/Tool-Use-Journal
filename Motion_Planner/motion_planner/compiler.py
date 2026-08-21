"""Compile frozen strategy candidates into the first connected joint realization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from motion_planner.geometry import GeometryResolutionError, RelativePoseResolver
from motion_planner.kinematics import IKSolutionSet, UR5eKinematics
from motion_planner.schema import KeyframePlanCandidate, Pose, WorldSnapshot
from motion_planner.strategy import (
    BranchSelectionResult,
    ConnectedStrategy,
    EdgePlanner,
    FirstFeasibleBranchSelector,
    StateValidator,
    filter_ik_solutions,
)


@dataclass(frozen=True, slots=True)
class ResolvedKeyframe:
    keyframe_id: str
    pose: Pose
    ik_solutions: IKSolutionSet


@dataclass(frozen=True, slots=True)
class StrategyAttempt:
    strategy_id: str
    resolved_keyframes: tuple[ResolvedKeyframe, ...] = ()
    selection: BranchSelectionResult | None = None
    failure_code: str | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class StrategyCompilationResult:
    connected: ConnectedStrategy | None
    attempts: tuple[StrategyAttempt, ...]

    @property
    def solved(self) -> bool:
        return self.connected is not None


class FirstFeasibleStrategyCompiler:
    """Hierarchical backtracking: strategy -> pose -> IK branch -> edge."""

    def __init__(
        self,
        kinematics: UR5eKinematics,
        *,
        position_tolerance_m: float = 5e-3,
        orientation_tolerance_rad: float = 5e-2,
        branch_selector: FirstFeasibleBranchSelector | None = None,
    ) -> None:
        self._kinematics = kinematics
        self._position_tolerance_m = position_tolerance_m
        self._orientation_tolerance_rad = orientation_tolerance_rad
        self._selector = branch_selector or FirstFeasibleBranchSelector()

    def compile(
        self,
        world: WorldSnapshot,
        candidates: Sequence[KeyframePlanCandidate],
        *,
        start_joint_config: Sequence[float],
        state_validator: StateValidator,
        edge_planner: EdgePlanner,
    ) -> StrategyCompilationResult:
        resolver = RelativePoseResolver(world)
        attempts: list[StrategyAttempt] = []
        for strategy in candidates:
            resolved: list[ResolvedKeyframe] = []
            solution_sets: list[IKSolutionSet] = []
            failure_code: str | None = None
            failure_detail = ""
            for keyframe in strategy.keyframes:
                try:
                    pose = resolver.resolve(keyframe)
                except GeometryResolutionError as error:
                    failure_code = "KEYFRAME_GEOMETRY_INVALID"
                    failure_detail = f"{keyframe.keyframe_id}: {error}"
                    break
                solutions = self._kinematics.solve_all_ik(
                    pose.position_m,
                    pose.orientation_xyzw,
                    position_tolerance_m=self._position_tolerance_m,
                    orientation_tolerance_rad=self._orientation_tolerance_rad,
                )
                valid_solutions = filter_ik_solutions(
                    solutions, keyframe, state_validator
                )
                resolved.append(
                    ResolvedKeyframe(
                        keyframe_id=keyframe.keyframe_id,
                        pose=pose,
                        ik_solutions=valid_solutions,
                    )
                )
                solution_sets.append(valid_solutions)
                if not valid_solutions.solved:
                    failure_code = (
                        "NO_VALID_IK_BRANCH"
                        if valid_solutions.enumeration_complete
                        else "IK_SEARCH_EXHAUSTED"
                    )
                    failure_detail = (
                        f"{keyframe.keyframe_id}: {valid_solutions.detail}"
                    )
                    break

            if failure_code is not None:
                attempts.append(
                    StrategyAttempt(
                        strategy_id=strategy.strategy_id,
                        resolved_keyframes=tuple(resolved),
                        failure_code=failure_code,
                        detail=failure_detail,
                    )
                )
                continue

            selection = self._selector.select(
                strategy,
                start_joint_config,
                solution_sets,
                edge_planner,
            )
            attempts.append(
                StrategyAttempt(
                    strategy_id=strategy.strategy_id,
                    resolved_keyframes=tuple(resolved),
                    selection=selection,
                    failure_code=selection.failure_code,
                    detail=selection.detail,
                )
            )
            if selection.connected is not None:
                return StrategyCompilationResult(
                    connected=selection.connected,
                    attempts=tuple(attempts),
                )
        return StrategyCompilationResult(connected=None, attempts=tuple(attempts))
