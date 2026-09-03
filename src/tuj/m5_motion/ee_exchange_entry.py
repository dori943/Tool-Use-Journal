"""Collision-checked positioning at a stored EE exchange-entry state.

Return trajectories are commissioned from one exact joint state per mounted
end effector.  An ``EE_EXCHANGE_ENTRY`` request moves the currently mounted EE
to that stored state before the fail-closed precomputed return/attach plan is
bound.  Only this preparation leg is planned dynamically; the rack operation
itself remains precomputed.
"""

from __future__ import annotations

import enum
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from tuj.m5_motion.path_planning import RRTConnectEdgePlanner, validate_joint_segment
from tuj.m5_motion.precomputed_ee_attach import (
    CollisionChecker,
    EEAttachPathFailureCode,
    PrecomputedEEPathError,
    _robot_model,
    compute_rack_signature,
    compute_workcell_signature,
    normalize_ee_id,
)
from tuj.m5_motion.precomputed_ee_exchange import (
    EEReturnTrajectoryTemplate,
    PrecomputedEEReturnRegistry,
)
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    CollisionContext,
    InterpolationType,
    KeyframePlannerType,
    KeyframeType,
    ModuleName,
    MotionPlan,
    MotionPlanRequest,
    RelativeKeyframeSpec,
    RobotState,
    SegmentType,
    TrajectoryProcessingStep,
    TrajectorySegment,
)
from tuj.m5_motion.task_semantics import task_operation
from tuj.m5_motion.trajectory_processing import (
    QuinticTimeParameterizer,
    TrajectoryProcessingError,
    deterministic_shortcut,
)


def is_ee_exchange_entry_request(request: MotionPlanRequest) -> bool:
    """Whether this is the explicit attached-EE exchange preparation leg."""

    return task_operation(request.task) == "EE_EXCHANGE_ENTRY"


class EEExchangeEntryFailureCode(str, enum.Enum):
    INVALID_REQUEST = "EE_EXCHANGE_ENTRY_INVALID_REQUEST"
    PATH_NOT_FOUND = "EE_EXCHANGE_ENTRY_PATH_NOT_FOUND"
    DYNAMICS_INVALID = "EE_EXCHANGE_ENTRY_DYNAMICS_INVALID"
    FINAL_COLLISION_CHECK_FAILED = "EE_EXCHANGE_ENTRY_FINAL_COLLISION_CHECK_FAILED"


class EEExchangeEntryPlanningError(RuntimeError):
    def __init__(
        self,
        failure_code: EEExchangeEntryFailureCode,
        detail: str,
        *,
        trajectory_id: str | None = None,
    ) -> None:
        self.failure_code = failure_code
        self.detail = detail
        self.trajectory_id = trajectory_id
        super().__init__(f"{failure_code.value}: {detail}")


class JointPositionLimitsProvider(Protocol):
    @property
    def joint_limits_rad(self) -> Sequence[tuple[float, float]]: ...


def _minimum_clearance(
    path: Sequence[Sequence[float]],
    keyframe: RelativeKeyframeSpec,
    collision_checker: CollisionChecker,
    max_joint_step_rad: float,
) -> float | None:
    minimum: float | None = None
    for source, target in zip(path, path[1:]):
        report = validate_joint_segment(
            source,
            target,
            keyframe,
            collision_checker,  # type: ignore[arg-type]
            max_joint_step_rad=max_joint_step_rad,
            wrap_joints=False,
        )
        if not report.valid:
            raise EEExchangeEntryPlanningError(
                EEExchangeEntryFailureCode.FINAL_COLLISION_CHECK_FAILED,
                f"{report.failure_code or 'COLLISION'}: {report.detail}",
            )
        if report.min_clearance_m is not None:
            minimum = (
                report.min_clearance_m
                if minimum is None
                else min(minimum, report.min_clearance_m)
            )
    return minimum


class EEExchangeEntryPlanner:
    """Plan current attached-EE state to its commissioned return start state."""

    def __init__(
        self,
        return_registry: PrecomputedEEReturnRegistry,
        *,
        joint_position_limits_rad: Sequence[tuple[float, float]],
        log: Any = print,
    ) -> None:
        limits = tuple(
            (float(lower), float(upper))
            for lower, upper in joint_position_limits_rad
        )
        if not limits or any(
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower >= upper
            for lower, upper in limits
        ):
            raise ValueError("exchange-entry joint position limits are invalid")
        self.return_registry = return_registry
        self.joint_position_limits_rad = limits
        self._log = log

    @staticmethod
    def _source(request: MotionPlanRequest) -> str:
        raw = (
            request.task.metadata.get("entry_ee")
            or request.task.metadata.get("from_ee")
            or request.task.ee
        )
        try:
            return normalize_ee_id(raw)
        except ValueError as error:
            raise EEExchangeEntryPlanningError(
                EEExchangeEntryFailureCode.INVALID_REQUEST,
                str(error),
            ) from error

    def load(self, request: MotionPlanRequest) -> EEReturnTrajectoryTemplate:
        environment = request.world.metadata.get("environment_name")
        if not isinstance(environment, str) or not environment:
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.WORKCELL_SIGNATURE_MISMATCH,
                "exchange-entry request has no environment_name",
            )
        return self.return_registry.load(environment, self._source(request))

    def _validate_request(
        self,
        request: MotionPlanRequest,
        template: EEReturnTrajectoryTemplate,
    ) -> str:
        if not is_ee_exchange_entry_request(request):
            raise EEExchangeEntryPlanningError(
                EEExchangeEntryFailureCode.INVALID_REQUEST,
                "request is not EE_EXCHANGE_ENTRY",
                trajectory_id=template.trajectory_id,
            )
        source = self._source(request)
        state = request.world.robot_state
        try:
            physical = normalize_ee_id(
                request.world.metadata.get("physical_active_ee")
            )
            requested = normalize_ee_id(request.task.ee)
        except ValueError as error:
            raise EEExchangeEntryPlanningError(
                EEExchangeEntryFailureCode.INVALID_REQUEST,
                str(error),
                trajectory_id=template.trajectory_id,
            ) from error
        if physical != source or requested != source:
            raise EEExchangeEntryPlanningError(
                EEExchangeEntryFailureCode.INVALID_REQUEST,
                "request, stored entry, and physically mounted EE do not match",
                trajectory_id=template.trajectory_id,
            )
        if state.attached_object_id is not None or state.held_tool_id is not None:
            raise EEExchangeEntryPlanningError(
                EEExchangeEntryFailureCode.INVALID_REQUEST,
                "exchange-entry motion requires an empty mounted end effector",
                trajectory_id=template.trajectory_id,
            )
        if state.joint_names != template.joint_names:
            raise EEExchangeEntryPlanningError(
                EEExchangeEntryFailureCode.INVALID_REQUEST,
                "joint name or order differs from the stored exchange-entry",
                trajectory_id=template.trajectory_id,
            )
        if len(self.joint_position_limits_rad) != len(template.joint_names):
            raise EEExchangeEntryPlanningError(
                EEExchangeEntryFailureCode.INVALID_REQUEST,
                "joint position limit DOF differs from the stored exchange-entry",
                trajectory_id=template.trajectory_id,
            )
        for name, value, (lower, upper) in zip(
            template.joint_names,
            template.start_joint_positions_rad,
            self.joint_position_limits_rad,
        ):
            if value < lower - 1e-9 or value > upper + 1e-9:
                raise EEExchangeEntryPlanningError(
                    EEExchangeEntryFailureCode.INVALID_REQUEST,
                    f"stored exchange-entry for joint {name} exceeds its limit",
                    trajectory_id=template.trajectory_id,
                )
        missing_dynamics = set(template.joint_names) - set(
            request.constraints.joint_limits
        )
        if missing_dynamics:
            raise EEExchangeEntryPlanningError(
                EEExchangeEntryFailureCode.DYNAMICS_INVALID,
                f"request has no dynamic limits for {sorted(missing_dynamics)}",
                trajectory_id=template.trajectory_id,
            )
        return source

    @staticmethod
    def _validate_workcell(
        request: MotionPlanRequest,
        template: EEReturnTrajectoryTemplate,
        contexts: Mapping[str, CollisionContext],
    ) -> None:
        try:
            selected = {
                context_id: contexts[context_id]
                for context_id in template.collision_model_versions
            }
        except KeyError as error:
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                f"required exchange-entry collision context is unavailable: {error.args[0]}",
                trajectory_id=template.trajectory_id,
            ) from error
        if template.robot_model != _robot_model(request.world):
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                "robot model differs from the stored exchange-entry",
                trajectory_id=template.trajectory_id,
            )
        if template.rack_signature != compute_rack_signature(request.world):
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.WORKCELL_SIGNATURE_MISMATCH,
                "rack/dock signature differs from the stored exchange-entry",
                trajectory_id=template.trajectory_id,
            )
        versions = {
            context_id: context.collision_model_version
            for context_id, context in selected.items()
        }
        if versions != template.collision_model_versions:
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                "collision models differ from the stored exchange-entry",
                trajectory_id=template.trajectory_id,
            )
        if template.start_eef_pose is None:
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                "stored exchange-entry has no mounted-EE canonical EEF pose",
                trajectory_id=template.trajectory_id,
            )
        canonical_world = request.world.model_copy(deep=True)
        canonical_world.robot_state.joint_positions_rad = list(
            template.start_joint_positions_rad
        )
        canonical_world.robot_state.joint_velocities_rad_s = [
            0.0
        ] * len(template.joint_names)
        canonical_world.robot_state.eef_pose = template.start_eef_pose.model_copy(
            deep=True
        )
        if compute_workcell_signature(canonical_world, selected) != template.workcell_signature:
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.WORKCELL_SIGNATURE_MISMATCH,
                "static workcell differs from the stored exchange-entry",
                trajectory_id=template.trajectory_id,
            )

    def plan(
        self,
        request: MotionPlanRequest,
        *,
        collision_contexts: Mapping[str, CollisionContext],
        collision_checker: CollisionChecker,
        template: EEReturnTrajectoryTemplate | None = None,
    ) -> MotionPlan:
        selected = template or self.load(request)
        source = self._validate_request(request, selected)
        self._validate_workcell(request, selected, collision_contexts)
        context_id = f"ee-attached:{source}"
        context = collision_contexts.get(context_id)
        if context is None or context.active_ee != source:
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                f"attached collision context for {source} is unavailable",
                trajectory_id=selected.trajectory_id,
            )
        keyframe = RelativeKeyframeSpec(
            keyframe_id=f"{request.task.subgoal_id}:exchange-entry",
            keyframe_type=KeyframeType.CUSTOM,
            frame_ref="world",
            anchor="exchange-entry",
            approach_axis_xyz=(0.0, 0.0, 1.0),
            planner=KeyframePlannerType.SAMPLING_BASED,
            collision_context_id=context_id,
            metadata={
                "source_ee": source,
                "return_trajectory_id": selected.trajectory_id,
            },
        )
        start = tuple(float(value) for value in request.world.robot_state.joint_positions_rad)
        target = tuple(float(value) for value in selected.start_joint_positions_rad)
        direct = validate_joint_segment(
            start,
            target,
            keyframe,
            collision_checker,  # type: ignore[arg-type]
            max_joint_step_rad=request.constraints.max_joint_path_step_rad,
            wrap_joints=False,
        )
        planner_name = "DIRECT_JOINT"
        if direct.valid:
            geometric_path: tuple[tuple[float, ...], ...] = (start, target)
        else:
            rrt = RRTConnectEdgePlanner(
                collision_checker,  # type: ignore[arg-type]
                self.joint_position_limits_rad,
                random_seed=request.options.random_seed,
                max_iterations=request.options.rrt_max_iterations,
                timeout_s=request.options.allowed_planning_time_s,
                extension_step_rad=request.options.rrt_extension_step_rad,
                validation_step_rad=request.constraints.max_joint_path_step_rad,
                goal_bias=request.options.rrt_goal_bias,
                wrap_joints=False,
            )
            result = rrt.plan(start, target, None, keyframe)
            if not result.valid:
                raise EEExchangeEntryPlanningError(
                    EEExchangeEntryFailureCode.PATH_NOT_FOUND,
                    f"{result.failure_code or 'RRT_CONNECT_FAILED'}: {result.detail}",
                    trajectory_id=selected.trajectory_id,
                )
            geometric_path = result.joint_path
            planner_name = "RRT_CONNECT"
            if request.options.simplify_path:
                geometric_path = deterministic_shortcut(
                    geometric_path,
                    lambda left, right: validate_joint_segment(
                        left,
                        right,
                        keyframe,
                        collision_checker,  # type: ignore[arg-type]
                        max_joint_step_rad=request.constraints.max_joint_path_step_rad,
                        wrap_joints=False,
                    ).valid,
                )
        minimum_clearance = _minimum_clearance(
            geometric_path,
            keyframe,
            collision_checker,
            request.constraints.max_joint_path_step_rad,
        )
        try:
            timed = QuinticTimeParameterizer(
                sample_dt_s=request.options.interpolation_dt_s
            ).parameterize(
                selected.joint_names,
                geometric_path,
                request.constraints.joint_limits,
                velocity_scaling=request.constraints.velocity_scaling,
                acceleration_scaling=request.constraints.acceleration_scaling,
                jerk_scaling=request.constraints.jerk_scaling,
            )
        except TrajectoryProcessingError as error:
            raise EEExchangeEntryPlanningError(
                EEExchangeEntryFailureCode.DYNAMICS_INVALID,
                str(error),
                trajectory_id=selected.trajectory_id,
            ) from error
        final_validator = getattr(collision_checker, "final_segment_validator", None)
        if callable(final_validator) and not final_validator(timed.waypoints, context):
            report = getattr(collision_checker, "last_path_collision_check", None)
            detail = (
                f"{report.failure_code}: {report.detail}"
                if report is not None
                else "timed trajectory failed final collision validation"
            )
            raise EEExchangeEntryPlanningError(
                EEExchangeEntryFailureCode.FINAL_COLLISION_CHECK_FAILED,
                detail,
                trajectory_id=selected.trajectory_id,
            )
        digest = hashlib.sha256(
            (
                f"{request.request_id}|{request.world.scene.signature}|"
                f"{selected.trajectory_id}|exchange-entry"
            ).encode("utf-8")
        ).hexdigest()[:24]
        moved = max(abs(left - right) for left, right in zip(start, target)) > 1e-9
        segment = TrajectorySegment(
            segment_id=f"{request.task.subgoal_id}:exchange-entry",
            segment_type=SegmentType.EE_EXCHANGE,
            start_time_s=0.0,
            end_time_s=timed.duration_s,
            interpolation=InterpolationType.QUINTIC,
            waypoints=list(timed.waypoints),
            collision_checked=True,
            min_clearance_m=minimum_clearance,
            collision_context_before=context,
            collision_context_after=context,
            processing_steps=(
                [TrajectoryProcessingStep.RAW_PATH]
                + (
                    [TrajectoryProcessingStep.SHORTCUT]
                    if planner_name == "RRT_CONNECT" and request.options.simplify_path
                    else []
                )
                + [
                    TrajectoryProcessingStep.TIME_PARAMETERIZATION,
                    TrajectoryProcessingStep.FINAL_COLLISION_CHECK,
                    TrajectoryProcessingStep.DYNAMICS_CHECK,
                ]
            ),
            metadata={
                "source": "dynamic-exchange-entry",
                "planner": planner_name,
                "return_trajectory_id": selected.trajectory_id,
            },
        )
        plan = MotionPlan(
            plan_id=f"motion-plan:{digest}",
            request_id=request.request_id,
            provenance=ArtifactProvenance(
                artifact_id=f"motion-plan-artifact:{digest}",
                artifact_type="MotionPlan",
                produced_by=ModuleName.MOTION_PLANNER,
                invocation_id=f"ee-exchange-entry:{request.request_id}",
                input_artifact_ids=[request.provenance.artifact_id],
                metadata={
                    "source": "dynamic-exchange-entry",
                    "return_trajectory_id": selected.trajectory_id,
                },
            ),
            scene_signature=request.world.scene.signature,
            robot_id=request.world.robot_state.robot_id,
            joint_names=list(selected.joint_names),
            duration_s=timed.duration_s,
            segments=[segment],
            events=[],
            expected_final_state=RobotState(
                robot_id=request.world.robot_state.robot_id,
                joint_names=list(selected.joint_names),
                joint_positions_rad=list(target),
                joint_velocities_rad_s=[0.0] * len(target),
                eef_pose=selected.start_eef_pose.model_copy(deep=True),
                gripper=(
                    request.world.robot_state.gripper.model_copy(deep=True)
                    if request.world.robot_state.gripper is not None
                    else None
                ),
                attached_object_id=None,
                held_tool_id=None,
            ),
            metadata={
                "source": "dynamic-exchange-entry",
                "selection_policy": "STORED_EE_EXCHANGE_ENTRY_REQUIRED",
                "source_active_ee": source,
                "target_active_ee": source,
                "return_trajectory_id": selected.trajectory_id,
                "planner": planner_name,
                "entry_move_skipped": not moved,
                "workcell_signature_validation": "PASS",
                "collision_validation": "PASS",
                "dynamic_planner_invoked": moved,
            },
        )
        self._log(
            f"[M5][EE_ENTRY] hit: {source} trajectory={selected.trajectory_id}"
        )
        self._log(f"[M5][EE_ENTRY] planner={planner_name}")
        self._log("[M5][EE_ENTRY] workcell_signature=PASS")
        self._log("[M5][EE_ENTRY] collision_validation=PASS")
        self._log(f"[M5][EE_ENTRY] dynamic_planner_invoked={str(moved).lower()}")
        return plan


__all__ = [
    "EEExchangeEntryFailureCode",
    "EEExchangeEntryPlanner",
    "EEExchangeEntryPlanningError",
    "is_ee_exchange_entry_request",
]
