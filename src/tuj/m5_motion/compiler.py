"""Compile frozen strategy candidates into the first connected joint realization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from tuj.m5_motion.attachment_retarget import (
    AttachmentRetargetError,
    retarget_resolved_pose,
)
from tuj.m5_motion.contact_keyframe_validation import (
    ContactKeyframeGeometryError,
    canonicalize_held_tool_axis_for_contact,
    canonicalize_sweep_strategy_heights,
    validate_sweep_keyframe_strategy,
)
from tuj.m5_motion.geometry import GeometryResolutionError, RelativePoseResolver
from tuj.m5_motion.grasp_geometry import (
    MULTI_FINGER_PLACE_RETREAT_CLEARANCE,
    MULTI_FINGER_TABLETOP_ENCLOSURE,
    multi_finger_place_retreat_standoff_candidates,
    multi_finger_tabletop_lift_standoff_candidates,
    multi_finger_tabletop_pre_grasp_standoff_candidates,
)
from tuj.m5_motion.kinematics import IKSolutionSet, UR5eKinematics
from tuj.m5_motion.schema import (
    KeyframePlanCandidate,
    KeyframeType,
    MotionPlanRequest,
    Pose,
    RelativeKeyframeSpec,
    WorldSnapshot,
)
from tuj.m5_motion.strategy import (
    BranchSelectionResult,
    ConnectedStrategy,
    EdgePlanner,
    FirstFeasibleBranchSelector,
    StateValidator,
    filter_ik_solutions,
)


def _enclosure_standoff_offset_candidates(
    keyframe: RelativeKeyframeSpec,
) -> tuple[float, ...]:
    """Preferred enclosure standoff, then shorter reach fallbacks when in scope."""

    primary = float(keyframe.offset_along_approach_m)
    source = keyframe.metadata.get("contact_geometry_source")
    if (
        source == MULTI_FINGER_PLACE_RETREAT_CLEARANCE
        and keyframe.keyframe_type is KeyframeType.RETREAT
    ):
        # Empty-EE post-DETACH retreat: prefer finger clearance, then longer
        # bounded extensions if the preferred TCP is still collision-invalid.
        return multi_finger_place_retreat_standoff_candidates(primary)
    if source != MULTI_FINGER_TABLETOP_ENCLOSURE:
        return (primary,)
    if keyframe.keyframe_type is KeyframeType.PRE_GRASP:
        return multi_finger_tabletop_pre_grasp_standoff_candidates(primary)
    if keyframe.keyframe_type is KeyframeType.LIFT:
        return multi_finger_tabletop_lift_standoff_candidates(primary)
    if keyframe.keyframe_type is KeyframeType.GRASP and primary > 1e-9:
        # Enclosure contact is offset 0; a leftover high GRASP standoff hits the
        # same outer-workspace IK cliff as PRE/LIFT.
        return (primary, 0.0)
    return (primary,)


@dataclass(frozen=True, slots=True)
class ResolvedKeyframe:
    keyframe_id: str
    pose: Pose
    ik_solutions: IKSolutionSet


@dataclass(frozen=True, slots=True)
class KeyframeIKDiagnostic:
    keyframe_id: str
    raw_ik_count: int
    valid_ik_count: int
    attempted_seeds: int
    best_position_error_m: float
    best_orientation_error_rad: float
    solver_id: str
    solver_failure_code: str | None
    solver_detail: str
    validity_detail: str


@dataclass(frozen=True, slots=True)
class StrategyAttempt:
    strategy_id: str
    resolved_keyframes: tuple[ResolvedKeyframe, ...] = ()
    selection: BranchSelectionResult | None = None
    failure_code: str | None = None
    detail: str = ""
    ik_diagnostics: tuple[KeyframeIKDiagnostic, ...] = ()


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
        request: MotionPlanRequest | None = None,
    ) -> StrategyCompilationResult:
        resolver = RelativePoseResolver(world)
        attempts: list[StrategyAttempt] = []
        for strategy in candidates:
            resolved: list[ResolvedKeyframe] = []
            solution_sets: list[IKSolutionSet] = []
            diagnostics: list[KeyframeIKDiagnostic] = []
            adapted_keyframes: list[RelativeKeyframeSpec] = []
            failure_code: str | None = None
            failure_detail = ""
            strategy_keyframes = list(strategy.keyframes)
            if request is not None:
                # Contact tool_act (sweep): repair the held-tool attitude and
                # working height, then reject task-unrelated geometry before
                # spending an IK search on it.
                strategy_keyframes = canonicalize_sweep_strategy_heights(
                    request,
                    [
                        canonicalize_held_tool_axis_for_contact(
                            request, keyframe, resolver=resolver
                        )
                        for keyframe in strategy_keyframes
                    ],
                    resolver=resolver,
                )
                try:
                    validate_sweep_keyframe_strategy(
                        request, strategy_keyframes, resolver=resolver
                    )
                except ContactKeyframeGeometryError as error:
                    attempts.append(
                        StrategyAttempt(
                            strategy_id=strategy.strategy_id,
                            failure_code="KEYFRAME_GEOMETRY_INVALID",
                            detail=str(error),
                        )
                    )
                    continue
            for keyframe in strategy_keyframes:
                ik_options = {
                    "position_tolerance_m": self._position_tolerance_m,
                    "orientation_tolerance_rad": self._orientation_tolerance_rad,
                }
                if bool(
                    keyframe.metadata.get("preserve_endpoint_continuity", False)
                ):
                    # Contact-continuation callers can require the current
                    # branch as an explicit endpoint candidate. Applying this
                    # policy to unrelated transfer / grasp keyframes changes
                    # previously validated global branch selection.
                    ik_options["seed_qpos"] = start_joint_config

                preferred_offset = float(keyframe.offset_along_approach_m)
                offset_candidates = _enclosure_standoff_offset_candidates(keyframe)
                chosen_keyframe = keyframe
                chosen_pose: Pose | None = None
                chosen_solutions: IKSolutionSet | None = None
                chosen_valid: IKSolutionSet | None = None
                geometry_error: Exception | None = None

                for offset in offset_candidates:
                    if math.isclose(offset, preferred_offset, abs_tol=1e-9):
                        candidate_keyframe = keyframe
                    else:
                        adapted_meta = {
                            **keyframe.metadata,
                            "tabletop_enclosure_standoff_adapted_m": offset,
                            "tabletop_enclosure_standoff_preferred_m": preferred_offset,
                        }
                        if keyframe.keyframe_type is KeyframeType.PRE_GRASP:
                            adapted_meta[
                                "tabletop_enclosure_pre_standoff_adapted_m"
                            ] = offset
                            adapted_meta[
                                "tabletop_enclosure_pre_standoff_preferred_m"
                            ] = preferred_offset
                        candidate_keyframe = keyframe.model_copy(
                            update={
                                "offset_along_approach_m": offset,
                                "metadata": adapted_meta,
                            }
                        )
                    try:
                        pose = retarget_resolved_pose(
                            world,
                            candidate_keyframe,
                            resolver.resolve(candidate_keyframe),
                        )
                    except (GeometryResolutionError, AttachmentRetargetError) as error:
                        geometry_error = error
                        break
                    solutions = self._kinematics.solve_all_ik(
                        pose.position_m,
                        pose.orientation_xyzw,
                        **ik_options,
                    )
                    valid_solutions = filter_ik_solutions(
                        solutions, candidate_keyframe, state_validator
                    )
                    chosen_keyframe = candidate_keyframe
                    chosen_pose = pose
                    chosen_solutions = solutions
                    chosen_valid = valid_solutions
                    if valid_solutions.solved:
                        break

                if geometry_error is not None:
                    failure_code = (
                        "ATTACHMENT_RETARGET_INVALID"
                        if isinstance(geometry_error, AttachmentRetargetError)
                        else "KEYFRAME_GEOMETRY_INVALID"
                    )
                    failure_detail = f"{keyframe.keyframe_id}: {geometry_error}"
                    break
                assert (
                    chosen_pose is not None
                    and chosen_solutions is not None
                    and chosen_valid is not None
                )
                diagnostics.append(
                    KeyframeIKDiagnostic(
                        keyframe_id=keyframe.keyframe_id,
                        raw_ik_count=len(chosen_solutions.solutions),
                        valid_ik_count=len(chosen_valid.solutions),
                        attempted_seeds=chosen_solutions.attempted_seeds,
                        best_position_error_m=chosen_solutions.best_position_error_m,
                        best_orientation_error_rad=(
                            chosen_solutions.best_orientation_error_rad
                        ),
                        solver_id=chosen_solutions.solver_id,
                        solver_failure_code=chosen_solutions.failure_code,
                        solver_detail=chosen_solutions.detail,
                        validity_detail=chosen_valid.detail,
                    )
                )
                resolved.append(
                    ResolvedKeyframe(
                        keyframe_id=keyframe.keyframe_id,
                        pose=chosen_pose,
                        ik_solutions=chosen_valid,
                    )
                )
                solution_sets.append(chosen_valid)
                adapted_keyframes.append(chosen_keyframe)
                if not chosen_valid.solved:
                    if not chosen_solutions.solved:
                        failure_code = (
                            "TARGET_OUTSIDE_REACH_ENVELOPE"
                            if chosen_solutions.failure_code
                            == "TARGET_OUTSIDE_CONSERVATIVE_REACH"
                            else "NO_VALID_IK_BRANCH"
                            if chosen_solutions.enumeration_complete
                            else "IK_SEARCH_EXHAUSTED"
                        )
                    elif "COLLISION_" in chosen_valid.detail.upper():
                        failure_code = "COLLISION_FILTERED_ALL"
                    else:
                        failure_code = (
                            "NO_VALID_IK_BRANCH"
                            if chosen_valid.enumeration_complete
                            else "IK_SEARCH_EXHAUSTED"
                        )
                    failure_detail = (
                        f"{keyframe.keyframe_id}: {chosen_valid.detail}"
                    )
                    break

            if failure_code is not None:
                attempts.append(
                    StrategyAttempt(
                        strategy_id=strategy.strategy_id,
                        resolved_keyframes=tuple(resolved),
                        failure_code=failure_code,
                        detail=failure_detail,
                        ik_diagnostics=tuple(diagnostics),
                    )
                )
                continue

            selection_strategy = (
                strategy
                if adapted_keyframes == strategy_keyframes
                else strategy.model_copy(update={"keyframes": adapted_keyframes})
            )
            selection = self._selector.select(
                selection_strategy,
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
                    ik_diagnostics=tuple(diagnostics),
                )
            )
            if selection.connected is not None:
                return StrategyCompilationResult(
                    connected=selection.connected,
                    attempts=tuple(attempts),
                )
        return StrategyCompilationResult(connected=None, attempts=tuple(attempts))
