"""Compile frozen strategy candidates into the first connected joint realization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tuj.m5_motion.attachment_retarget import (
    AttachmentRetargetError,
    retarget_resolved_pose,
)
from tuj.m5_motion.geometry import GeometryResolutionError, RelativePoseResolver
from tuj.m5_motion.kinematics import IKSolutionSet, UR5eKinematics
from tuj.m5_motion.schema import KeyframePlanCandidate, Pose, WorldSnapshot
from tuj.m5_motion.strategy import (
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
    ) -> StrategyCompilationResult:
        resolver = RelativePoseResolver(world)
        attempts: list[StrategyAttempt] = []
        for strategy in candidates:
            resolved: list[ResolvedKeyframe] = []
            solution_sets: list[IKSolutionSet] = []
            diagnostics: list[KeyframeIKDiagnostic] = []
            failure_code: str | None = None
            failure_detail = ""
            for keyframe in strategy.keyframes:
                try:
                    pose = retarget_resolved_pose(
                        world,
                        keyframe,
                        resolver.resolve(keyframe),
                    )
                except (GeometryResolutionError, AttachmentRetargetError) as error:
                    failure_code = (
                        "ATTACHMENT_RETARGET_INVALID"
                        if isinstance(error, AttachmentRetargetError)
                        else "KEYFRAME_GEOMETRY_INVALID"
                    )
                    failure_detail = f"{keyframe.keyframe_id}: {error}"
                    break
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
                solutions = self._kinematics.solve_all_ik(
                    pose.position_m,
                    pose.orientation_xyzw,
                    **ik_options,
                )
                valid_solutions = filter_ik_solutions(
                    solutions, keyframe, state_validator
                )
                diagnostics.append(
                    KeyframeIKDiagnostic(
                        keyframe_id=keyframe.keyframe_id,
                        raw_ik_count=len(solutions.solutions),
                        valid_ik_count=len(valid_solutions.solutions),
                        attempted_seeds=solutions.attempted_seeds,
                        best_position_error_m=solutions.best_position_error_m,
                        best_orientation_error_rad=solutions.best_orientation_error_rad,
                        solver_id=solutions.solver_id,
                        solver_failure_code=solutions.failure_code,
                        solver_detail=solutions.detail,
                        validity_detail=valid_solutions.detail,
                    )
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
                    if not solutions.solved:
                        failure_code = (
                            "TARGET_OUTSIDE_REACH_ENVELOPE"
                            if solutions.failure_code
                            == "TARGET_OUTSIDE_CONSERVATIVE_REACH"
                            else "NO_VALID_IK_BRANCH"
                            if solutions.enumeration_complete
                            else "IK_SEARCH_EXHAUSTED"
                        )
                    elif "COLLISION_" in valid_solutions.detail.upper():
                        failure_code = "COLLISION_FILTERED_ALL"
                    else:
                        failure_code = (
                            "NO_VALID_IK_BRANCH"
                            if valid_solutions.enumeration_complete
                            else "IK_SEARCH_EXHAUSTED"
                        )
                    # 0912: 기각된 전략의 키프레임은 계획 파일에 남지 않아서
                    # "어느 자세를 시도했는지"를 사후에 볼 방법이 없었다.
                    # c2_2 의 place_on 은 전략 11개가 전부 같은 관통값으로
                    # 떨어졌는데, 그 자세가 받침의 윗면에서 온 것인지 중심에서
                    # 온 것인지 가릴 수가 없었다. 시도한 자세와 그 자세를 만든
                    # 기준 프레임/앵커를 실패 기록에 같이 남긴다.
                    try:
                        position = [
                            round(float(v), 6)
                            for v in pose.position_m
                        ]
                    except Exception:
                        position = None
                    frame_ref = getattr(keyframe, "frame_ref", None)
                    anchor = getattr(keyframe, "anchor", None)
                    failure_detail = (
                        f"{keyframe.keyframe_id}: {valid_solutions.detail}"
                        f"; attempted eef position_m={position}"
                        f" frame_ref={frame_ref} anchor={anchor}"
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
                    ik_diagnostics=tuple(diagnostics),
                )
            )
            if selection.connected is not None:
                return StrategyCompilationResult(
                    connected=selection.connected,
                    attempts=tuple(attempts),
                )
        return StrategyCompilationResult(connected=None, attempts=tuple(attempts))
