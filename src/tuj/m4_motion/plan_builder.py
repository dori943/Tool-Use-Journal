"""Build a versioned MotionPlan from one connected strategy realization."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from tuj.m4_motion.schema import (
    ArtifactProvenance,
    CollisionContext,
    EventType,
    GripperMode,
    GripperState,
    InterpolationType,
    KeyframeEventType,
    KeyframeType,
    ModuleName,
    MotionPlan,
    MotionPlanRequest,
    Pose,
    RobotState,
    SegmentType,
    TrajectoryEvent,
    TrajectoryProcessingStep,
    TrajectorySegment,
    TrajectoryWaypoint,
)
from tuj.m4_motion.strategy import ConnectedStrategy
from tuj.m4_motion.trajectory_processing import (
    QuinticTimeParameterizer,
    clamp_joint_limit_roundoff,
    deviation_bounded_shortcut,
    unwrap_joint_path,
)

FinalSegmentValidator = Callable[
    [tuple[TrajectoryWaypoint, ...], CollisionContext], bool
]
ForwardPose = Callable[
    [Sequence[float]],
    tuple[
        tuple[float, float, float],
        tuple[float, float, float, float],
    ],
]


class MotionPlanBuildError(ValueError):
    pass


def _segment_type(keyframe_type: KeyframeType) -> SegmentType:
    mapping = {
        KeyframeType.PRE_GRASP: SegmentType.APPROACH,
        KeyframeType.GRASP: SegmentType.GRASP,
        KeyframeType.LIFT: SegmentType.LIFT,
        KeyframeType.TRANSFER: SegmentType.TRANSFER,
        KeyframeType.PRE_PLACE: SegmentType.PLACE,
        KeyframeType.PLACE: SegmentType.PLACE,
        KeyframeType.RETREAT: SegmentType.RETREAT,
        KeyframeType.EE_UNDOCK_STAGING: SegmentType.EE_UNDOCK,
        KeyframeType.EE_PRE_UNDOCK: SegmentType.EE_UNDOCK,
        KeyframeType.EE_UNDOCK: SegmentType.EE_UNDOCK,
        KeyframeType.EE_DOCK_STAGING: SegmentType.EE_DOCK,
        KeyframeType.EE_PRE_DOCK: SegmentType.EE_DOCK,
        KeyframeType.EE_DOCK: SegmentType.EE_DOCK,
        KeyframeType.CUSTOM: SegmentType.CUSTOM,
    }
    return mapping[keyframe_type]


def _scene_signature(context: CollisionContext) -> tuple[object, ...]:
    return (
        context.scene_state_id or context.context_id,
        context.active_ee,
        tuple(sorted(context.attached_object_ids)),
        context.collision_model_version,
    )


def _finite_metadata_number(
    value: object,
    *,
    field: str,
    allow_zero: bool = True,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (float(value) < 0.0 if allow_zero else float(value) <= 0.0)
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise MotionPlanBuildError(
            f"{field} must be a finite {qualifier} number"
        )
    return float(value)


def _finite_metadata_signed_number(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MotionPlanBuildError(f"{field} must be a finite number")
    return float(value)


def _finite_metadata_position(value: object, *, field: str) -> list[float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise MotionPlanBuildError(f"{field} must contain exactly 3 numbers")
    result: list[float] = []
    for component in value:
        if (
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not math.isfinite(float(component))
        ):
            raise MotionPlanBuildError(f"{field} must contain finite numbers")
        result.append(float(component))
    return result


def _physical_tool_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Validate and preserve M4 runtime metadata needed for physical tool use."""

    result: dict[str, object] = {}
    specifications: tuple[
        tuple[str, tuple[str, ...], tuple[str, ...]], ...
    ] = (
        (
            "physical_tool_control",
            (
                "target_clearance_m",
                "clearance_tolerance_m",
                "max_table_penetration_m",
                "gain",
                "rate_m_s",
                "max_offset_m",
                "activation_band_m",
                "max_joint_offset_rad",
            ),
            ("max_table_penetration_m",),
        ),
        (
            "physical_tool_settle",
            (
                "target_clearance_m",
                "xy_tolerance_m",
                "clearance_tolerance_m",
                "max_table_penetration_m",
                "max_tool_speed_m_s",
            ),
            ("max_table_penetration_m",),
        ),
    )
    for metadata_name, numeric_fields, zero_allowed_fields in specifications:
        raw = metadata.get(metadata_name)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise MotionPlanBuildError(
                f"keyframe {metadata_name} metadata must be a mapping"
            )
        allowed_fields = set(numeric_fields)
        if metadata_name == "physical_tool_settle":
            allowed_fields.add("target_position_m")
        unknown_fields = set(raw) - allowed_fields
        if unknown_fields:
            raise MotionPlanBuildError(
                f"unknown {metadata_name} fields: "
                f"{sorted(str(value) for value in unknown_fields)}"
            )
        missing_fields = allowed_fields - set(raw)
        if missing_fields:
            raise MotionPlanBuildError(
                f"{metadata_name} is missing fields: "
                f"{sorted(missing_fields)}"
            )
        validated: dict[str, object] = {
            name: _finite_metadata_number(
                raw[name],
                field=f"{metadata_name} {name}",
                allow_zero=name in zero_allowed_fields,
            )
            for name in numeric_fields
        }
        if metadata_name == "physical_tool_settle":
            validated["target_position_m"] = _finite_metadata_position(
                raw["target_position_m"],
                field="physical_tool_settle target_position_m",
            )
        result[metadata_name] = validated

    raw_target = metadata.get("physical_tool_target_position_m")
    if raw_target is not None:
        result["physical_tool_target_position_m"] = _finite_metadata_position(
            raw_target,
            field="physical_tool_target_position_m",
        )
    raw_push = metadata.get("physical_push_control")
    if raw_push is not None:
        if not isinstance(raw_push, Mapping):
            raise MotionPlanBuildError(
                "keyframe physical_push_control metadata must be a mapping"
            )
        allowed_push_fields = {
            "push_axis_world",
            "contact_offset_local_m",
            # Backward-compatible input alias for older C1_1 artifacts. The
            # normalized controller metadata below is surface-agnostic.
            "rim_contact_offset_local_m",
            "block_support_m",
            "contact_penetration_m",
            "max_correction_m",
            "contact_plan_time_scale",
            "reacquire_timeout_s",
            "max_reacquire_attempts",
            "contact_height_offset_from_block_center_m",
            "contact_height_target_m",
            "block_support_center_z_m",
            "block_support_tolerance_m",
            "contact_height_gain",
            "contact_height_rate_m_s",
            "contact_height_max_offset_m",
            "contact_height_max_downward_offset_m",
        }
        required_push_fields = allowed_push_fields - {
            "contact_offset_local_m",
            "rim_contact_offset_local_m",
        }
        unknown_push_fields = set(raw_push) - allowed_push_fields
        missing_push_fields = required_push_fields - set(raw_push)
        if unknown_push_fields:
            raise MotionPlanBuildError(
                "unknown physical_push_control fields: "
                f"{sorted(str(value) for value in unknown_push_fields)}"
            )
        if missing_push_fields:
            raise MotionPlanBuildError(
                "physical_push_control is missing fields: "
                f"{sorted(missing_push_fields)}"
            )
        contact_offset_fields = {
            name
            for name in (
                "contact_offset_local_m",
                "rim_contact_offset_local_m",
            )
            if name in raw_push
        }
        if len(contact_offset_fields) != 1:
            raise MotionPlanBuildError(
                "physical_push_control requires exactly one contact offset field"
            )
        push_axis = _finite_metadata_position(
            raw_push["push_axis_world"],
            field="physical_push_control push_axis_world",
        )
        axis_norm = math.sqrt(sum(value * value for value in push_axis))
        if axis_norm <= 1e-9:
            raise MotionPlanBuildError(
                "physical_push_control push_axis_world must be non-zero"
            )
        result["physical_push_control"] = {
            "push_axis_world": [value / axis_norm for value in push_axis],
            "contact_offset_local_m": _finite_metadata_position(
                raw_push[next(iter(contact_offset_fields))],
                field="physical_push_control contact_offset_local_m",
            ),
            "block_support_m": _finite_metadata_number(
                raw_push["block_support_m"],
                field="physical_push_control block_support_m",
                allow_zero=False,
            ),
            "contact_penetration_m": _finite_metadata_number(
                raw_push["contact_penetration_m"],
                field="physical_push_control contact_penetration_m",
            ),
            "max_correction_m": _finite_metadata_number(
                raw_push["max_correction_m"],
                field="physical_push_control max_correction_m",
                allow_zero=False,
            ),
            "contact_plan_time_scale": _finite_metadata_number(
                raw_push["contact_plan_time_scale"],
                field="physical_push_control contact_plan_time_scale",
                allow_zero=False,
            ),
            "reacquire_timeout_s": _finite_metadata_number(
                raw_push["reacquire_timeout_s"],
                field="physical_push_control reacquire_timeout_s",
                allow_zero=False,
            ),
            "contact_height_offset_from_block_center_m": (
                _finite_metadata_signed_number(
                    raw_push["contact_height_offset_from_block_center_m"],
                    field=(
                        "physical_push_control "
                        "contact_height_offset_from_block_center_m"
                    ),
                )
            ),
            "contact_height_target_m": _finite_metadata_number(
                raw_push["contact_height_target_m"],
                field="physical_push_control contact_height_target_m",
                allow_zero=False,
            ),
            "block_support_center_z_m": _finite_metadata_number(
                raw_push["block_support_center_z_m"],
                field="physical_push_control block_support_center_z_m",
                allow_zero=False,
            ),
            "block_support_tolerance_m": _finite_metadata_number(
                raw_push["block_support_tolerance_m"],
                field="physical_push_control block_support_tolerance_m",
                allow_zero=False,
            ),
            "contact_height_gain": _finite_metadata_number(
                raw_push["contact_height_gain"],
                field="physical_push_control contact_height_gain",
                allow_zero=False,
            ),
            "contact_height_rate_m_s": _finite_metadata_number(
                raw_push["contact_height_rate_m_s"],
                field="physical_push_control contact_height_rate_m_s",
                allow_zero=False,
            ),
            "contact_height_max_offset_m": _finite_metadata_number(
                raw_push["contact_height_max_offset_m"],
                field="physical_push_control contact_height_max_offset_m",
                allow_zero=False,
            ),
            "contact_height_max_downward_offset_m": _finite_metadata_number(
                raw_push["contact_height_max_downward_offset_m"],
                field=(
                    "physical_push_control "
                    "contact_height_max_downward_offset_m"
                ),
                allow_zero=False,
            ),
        }
        raw_max_reacquire_attempts = raw_push["max_reacquire_attempts"]
        if (
            isinstance(raw_max_reacquire_attempts, bool)
            or not isinstance(raw_max_reacquire_attempts, int)
            or raw_max_reacquire_attempts <= 0
        ):
            raise MotionPlanBuildError(
                "physical_push_control max_reacquire_attempts must be a "
                "positive integer"
            )
        result["physical_push_control"]["max_reacquire_attempts"] = (
            raw_max_reacquire_attempts
        )
        if result["physical_push_control"]["contact_plan_time_scale"] > 1.0:
            raise MotionPlanBuildError(
                "physical_push_control contact_plan_time_scale must be at most 1"
            )
    return result


class MotionPlanBuilder:
    """Finalize timing, events, and segment-scoped collision state."""

    def __init__(
        self,
        *,
        time_parameterizer: QuinticTimeParameterizer | None = None,
        forward_pose: ForwardPose | None = None,
    ) -> None:
        self._time_parameterizer = time_parameterizer or QuinticTimeParameterizer()
        self._forward_pose = forward_pose

    def _apply_cartesian_speed_limit(
        self,
        waypoints: tuple[TrajectoryWaypoint, ...],
        *,
        segment_start_s: float,
        duration_s: float,
        max_speed_m_s: float | None,
    ) -> tuple[tuple[TrajectoryWaypoint, ...], float]:
        if self._forward_pose is None:
            if max_speed_m_s is not None:
                raise MotionPlanBuildError(
                    "max_cartesian_speed_m_s requires forward kinematics"
                )
            return waypoints, duration_s
        result = [waypoint.model_copy(deep=True) for waypoint in waypoints]
        positions: list[tuple[float, float, float]] = []
        for waypoint in result:
            position, orientation = self._forward_pose(
                waypoint.joint_positions_rad
            )
            waypoint.eef_pose = Pose(
                frame_id="world",
                position_m=position,
                orientation_xyzw=orientation,
            )
            positions.append(position)
        scale = 1.0
        if max_speed_m_s is not None:
            for left, right, left_position, right_position in zip(
                result,
                result[1:],
                positions,
                positions[1:],
            ):
                dt = right.time_from_start_s - left.time_from_start_s
                distance = sum(
                    (right_value - left_value) ** 2
                    for left_value, right_value in zip(
                        left_position, right_position
                    )
                ) ** 0.5
                scale = max(scale, distance / (max_speed_m_s * dt))
        if scale > 1.0:
            for waypoint in result:
                waypoint.time_from_start_s = segment_start_s + scale * (
                    waypoint.time_from_start_s - segment_start_s
                )
                if waypoint.joint_velocities_rad_s is not None:
                    waypoint.joint_velocities_rad_s = [
                        value / scale
                        for value in waypoint.joint_velocities_rad_s
                    ]
                if waypoint.joint_accelerations_rad_s2 is not None:
                    waypoint.joint_accelerations_rad_s2 = [
                        value / (scale * scale)
                        for value in waypoint.joint_accelerations_rad_s2
                    ]
            duration_s *= scale
        return tuple(result), duration_s

    def build(
        self,
        request: MotionPlanRequest,
        connected: ConnectedStrategy,
        *,
        plan_id: str,
        provenance: ArtifactProvenance,
        collision_contexts: Mapping[str, CollisionContext],
        initial_collision_context_id: str,
        final_segment_validator: FinalSegmentValidator,
        joint_position_limits_rad: Sequence[tuple[float, float]] | None = None,
    ) -> MotionPlan:
        if provenance.produced_by is not ModuleName.MOTION_PLANNER:
            raise MotionPlanBuildError("MotionPlan provenance must be produced by MOTION_PLANNER")
        if len(connected.nodes) != len(connected.edges):
            raise MotionPlanBuildError("each selected keyframe requires one incoming edge")
        if initial_collision_context_id not in collision_contexts:
            raise MotionPlanBuildError("initial collision context is not registered")
        joint_names = request.world.robot_state.joint_names
        if set(request.constraints.joint_limits) != set(joint_names):
            raise MotionPlanBuildError(
                "joint_limits must contain exactly the request robot joint_names"
            )

        current_context = collision_contexts[initial_collision_context_id]
        current_attached_object_id = (
            request.world.robot_state.attached_object_id
        )
        current_held_tool_id = request.world.robot_state.held_tool_id
        if request.task.action_type == "PICK_TOOL":
            current_held_tool_id = request.task.goal.target_object_id
        elif request.task.action_type in {
            "RETURN_TOOL",
            "TERMINAL_RETURN_TOOL",
        }:
            current_held_tool_id = None
        current_gripper = (
            request.world.robot_state.gripper.model_copy(deep=True)
            if request.world.robot_state.gripper is not None
            else GripperState()
        )
        clock = 0.0
        continuous_joint_reference = tuple(
            float(value)
            for value in request.world.robot_state.joint_positions_rad
        )
        segments: list[TrajectorySegment] = []
        events: list[TrajectoryEvent] = []

        for index, (node, edge) in enumerate(zip(connected.nodes, connected.edges)):
            if not edge.valid:
                raise MotionPlanBuildError("connected strategy contains an invalid edge")
            movement_context = current_context
            context_id = node.keyframe.collision_context_id
            if context_id is not None:
                if context_id not in collision_contexts:
                    raise MotionPlanBuildError(
                        f"unknown collision context {context_id!r}"
                    )
                movement_context = collision_contexts[context_id]
            if _scene_signature(current_context) != _scene_signature(movement_context):
                raise MotionPlanBuildError(
                    f"keyframe {node.keyframe.keyframe_id!r} changes physical scene "
                    "state without an event"
                )

            # Dense Cartesian IK and collision-validation samples are geometric
            # checks, not intended rest points.  Reduce only points that stay
            # close to the original path, and validate the resulting trajectory
            # against the real segment collision context.  Tighten the bound on
            # failure before falling back to the original path.
            shortcut_applied = False
            timed_waypoints: tuple[TrajectoryWaypoint, ...] | None = None
            timed_duration: float | None = None
            shortcut_tolerances = (
                request.constraints.max_joint_path_step_rad * 0.15,
                request.constraints.max_joint_path_step_rad * 0.075,
                0.0,
            )
            continuous_path = unwrap_joint_path(
                edge.joint_path,
                start_reference=continuous_joint_reference,
                joint_limits_rad=joint_position_limits_rad,
            )
            if joint_position_limits_rad is not None:
                continuous_path = clamp_joint_limit_roundoff(
                    continuous_path,
                    joint_position_limits_rad,
                )
            for shortcut_tolerance in shortcut_tolerances:
                geometric_path = deviation_bounded_shortcut(
                    continuous_path,
                    max_deviation_rad=shortcut_tolerance,
                )
                timed = self._time_parameterizer.parameterize(
                    joint_names,
                    geometric_path,
                    request.constraints.joint_limits,
                    velocity_scaling=request.constraints.velocity_scaling,
                    acceleration_scaling=request.constraints.acceleration_scaling,
                    jerk_scaling=request.constraints.jerk_scaling,
                    start_time_s=clock,
                )
                candidate_waypoints, candidate_duration = (
                    self._apply_cartesian_speed_limit(
                        timed.waypoints,
                        segment_start_s=clock,
                        duration_s=timed.duration_s,
                        max_speed_m_s=request.constraints.max_cartesian_speed_m_s,
                    )
                )
                if final_segment_validator(candidate_waypoints, movement_context):
                    timed_waypoints = candidate_waypoints
                    timed_duration = candidate_duration
                    shortcut_applied = len(geometric_path) < len(continuous_path)
                    break
            if timed_waypoints is None or timed_duration is None:
                validator_owner = getattr(
                    final_segment_validator, "__self__", None
                )
                last_check = getattr(
                    validator_owner, "last_path_collision_check", None
                )
                if last_check is None:
                    last_check = getattr(
                        final_segment_validator,
                        "last_path_collision_check",
                        None,
                    )
                detail = (
                    f": {last_check.failure_code}: {last_check.detail}"
                    if last_check is not None and not last_check.valid
                    else ""
                )
                raise MotionPlanBuildError(
                    f"final validation failed for "
                    f"{node.keyframe.keyframe_id!r}{detail}"
                )

            after_context = movement_context
            after_id = node.keyframe.collision_context_after_events_id
            if after_id is not None:
                if after_id not in collision_contexts:
                    raise MotionPlanBuildError(
                        f"unknown post-event collision context {after_id!r}"
                    )
                after_context = collision_contexts[after_id]

            movement_end = clock + timed_duration
            raw_hold_duration = node.keyframe.metadata.get(
                "hold_duration_after_s", 0.0
            )
            if (
                isinstance(raw_hold_duration, bool)
                or not isinstance(raw_hold_duration, (int, float))
                or not math.isfinite(float(raw_hold_duration))
                or float(raw_hold_duration) < 0.0
            ):
                raise MotionPlanBuildError(
                    "keyframe hold_duration_after_s must be finite and "
                    "non-negative"
                )
            hold_duration = float(raw_hold_duration)
            physical_tool_metadata = _physical_tool_metadata(
                node.keyframe.metadata
            )
            raw_tracking_settle = node.keyframe.metadata.get("tracking_settle")
            tracking_settle: dict[str, float | int] | None = None
            if raw_tracking_settle is not None:
                if not isinstance(raw_tracking_settle, Mapping):
                    raise MotionPlanBuildError(
                        "keyframe tracking_settle metadata must be a mapping"
                    )
                allowed_settle_fields = {
                    "joint_tolerance_rad",
                    "eef_tolerance_m",
                    "max_wait_s",
                    "required_consecutive_ticks",
                }
                unknown_settle_fields = set(raw_tracking_settle) - (
                    allowed_settle_fields
                )
                if unknown_settle_fields:
                    raise MotionPlanBuildError(
                        "unknown tracking_settle fields: "
                        f"{sorted(str(value) for value in unknown_settle_fields)}"
                    )
                tracking_settle = {}
                for name in ("joint_tolerance_rad", "eef_tolerance_m"):
                    raw_value = raw_tracking_settle.get(name)
                    if raw_value is None:
                        continue
                    if (
                        isinstance(raw_value, bool)
                        or not isinstance(raw_value, (int, float))
                        or not math.isfinite(float(raw_value))
                        or float(raw_value) <= 0.0
                    ):
                        raise MotionPlanBuildError(
                            f"tracking_settle {name} must be finite and positive"
                        )
                    tracking_settle[name] = float(raw_value)
                if not any(
                    name in tracking_settle
                    for name in ("joint_tolerance_rad", "eef_tolerance_m")
                ):
                    raise MotionPlanBuildError(
                        "tracking_settle requires a joint or EEF tolerance"
                    )
                raw_max_wait = raw_tracking_settle.get("max_wait_s", 2.0)
                if (
                    isinstance(raw_max_wait, bool)
                    or not isinstance(raw_max_wait, (int, float))
                    or not math.isfinite(float(raw_max_wait))
                    or float(raw_max_wait) <= 0.0
                ):
                    raise MotionPlanBuildError(
                        "tracking_settle max_wait_s must be finite and positive"
                    )
                raw_required_ticks = raw_tracking_settle.get(
                    "required_consecutive_ticks", 3
                )
                if (
                    isinstance(raw_required_ticks, bool)
                    or not isinstance(raw_required_ticks, int)
                    or raw_required_ticks <= 0
                ):
                    raise MotionPlanBuildError(
                        "tracking_settle required_consecutive_ticks must be a "
                        "positive integer"
                    )
                tracking_settle["max_wait_s"] = float(raw_max_wait)
                tracking_settle["required_consecutive_ticks"] = (
                    raw_required_ticks
                )
            raw_event_offsets = node.keyframe.metadata.get(
                "event_time_offsets_s", {}
            )
            if not isinstance(raw_event_offsets, Mapping):
                raise MotionPlanBuildError(
                    "keyframe event_time_offsets_s metadata must be a mapping"
                )
            event_offsets: dict[str, float] = {}
            declared_event_names = {
                event_type.value for event_type in node.keyframe.events_after
            }
            for raw_name, raw_offset in raw_event_offsets.items():
                name = str(raw_name)
                if name not in declared_event_names:
                    raise MotionPlanBuildError(
                        f"event offset {name!r} has no matching events_after entry"
                    )
                if (
                    isinstance(raw_offset, bool)
                    or not isinstance(raw_offset, (int, float))
                    or not math.isfinite(float(raw_offset))
                    or float(raw_offset) < 0.0
                    or float(raw_offset) > hold_duration
                ):
                    raise MotionPlanBuildError(
                        f"event offset for {name!r} must be within the "
                        "post-motion hold interval"
                    )
                event_offsets[name] = float(raw_offset)

            segment_end = movement_end + hold_duration
            if hold_duration > 0.0:
                held = timed_waypoints[-1].model_copy(deep=True)
                held.time_from_start_s = segment_end
                held.joint_velocities_rad_s = [0.0] * len(joint_names)
                held.joint_accelerations_rad_s2 = [0.0] * len(joint_names)
                timed_waypoints = (*timed_waypoints, held)
            segment_id = f"{connected.strategy_id}:{index}:{node.keyframe.keyframe_id}"
            segments.append(
                TrajectorySegment(
                    segment_id=segment_id,
                    segment_type=_segment_type(node.keyframe.keyframe_type),
                    start_time_s=clock,
                    end_time_s=segment_end,
                    interpolation=InterpolationType.QUINTIC,
                    waypoints=list(timed_waypoints),
                    collision_checked=True,
                    min_clearance_m=edge.min_clearance_m,
                    collision_context_before=movement_context,
                    collision_context_after=after_context,
                    processing_steps=(
                        [TrajectoryProcessingStep.RAW_PATH]
                        + (
                            [TrajectoryProcessingStep.SHORTCUT]
                            if shortcut_applied
                            else []
                        )
                        + [
                            TrajectoryProcessingStep.TIME_PARAMETERIZATION,
                            TrajectoryProcessingStep.FINAL_COLLISION_CHECK,
                            TrajectoryProcessingStep.DYNAMICS_CHECK,
                        ]
                    ),
                    metadata={
                        "strategy_id": connected.strategy_id,
                        "keyframe_id": node.keyframe.keyframe_id,
                        "ik_branch_id": node.solution.branch_id,
                        "planner": node.keyframe.planner.value,
                        "motion_end_time_s": movement_end,
                        "hold_duration_after_s": hold_duration,
                        **(
                            {
                                "target_block_id": node.keyframe.metadata[
                                    "target_block_id"
                                ]
                            }
                            if "target_block_id" in node.keyframe.metadata
                            else {}
                        ),
                        **(
                            {"tracking_settle": tracking_settle}
                            if tracking_settle is not None
                            else {}
                        ),
                        **physical_tool_metadata,
                    },
                )
            )
            event_target = node.keyframe.metadata.get("event_target_id")
            if event_target is None:
                event_target = node.keyframe.metadata.get("ee")
            raw_event_parameters = node.keyframe.metadata.get(
                "event_parameters", {}
            )
            if not isinstance(raw_event_parameters, Mapping):
                raise MotionPlanBuildError(
                    "keyframe event_parameters metadata must be a mapping"
                )
            for event_index, event_type in enumerate(node.keyframe.events_after):
                raw_parameters = raw_event_parameters.get(event_type.value, {})
                if not isinstance(raw_parameters, Mapping):
                    raise MotionPlanBuildError(
                        f"event parameters for {event_type.value} must be a mapping"
                    )
                events.append(
                    TrajectoryEvent(
                        event_id=(
                            f"{node.keyframe.keyframe_id}:event:{event_index}:"
                            f"{event_type.value}"
                        ),
                        time_from_start_s=(
                            movement_end
                            + event_offsets.get(event_type.value, hold_duration)
                        ),
                        event_type=EventType(event_type.value),
                        target_id=str(event_target) if event_target is not None else None,
                        command=raw_parameters.get("command"),
                        parameters={
                            str(key): value
                            for key, value in raw_parameters.items()
                            if key != "command"
                        },
                    )
                )
                if event_type is KeyframeEventType.ATTACH_OBJECT:
                    if event_target is None:
                        raise MotionPlanBuildError(
                            "ATTACH_OBJECT keyframe requires event_target_id"
                        )
                    current_attached_object_id = str(event_target)
                    current_gripper = GripperState(mode=GripperMode.HOLDING)
                elif event_type is KeyframeEventType.DETACH_OBJECT:
                    if (
                        event_target is not None
                        and current_attached_object_id is not None
                        and str(event_target) != current_attached_object_id
                    ):
                        raise MotionPlanBuildError(
                            "DETACH_OBJECT target does not match attached object"
                        )
                    current_attached_object_id = None
                elif event_type in {
                    KeyframeEventType.GRIPPER_CLOSE,
                    KeyframeEventType.SUCTION_ON,
                }:
                    current_gripper = GripperState(mode=GripperMode.CLOSED)
                elif event_type in {
                    KeyframeEventType.GRIPPER_OPEN,
                    KeyframeEventType.SUCTION_OFF,
                }:
                    current_gripper = GripperState(mode=GripperMode.OPEN)
            clock = segment_end
            continuous_joint_reference = continuous_path[-1]
            current_context = after_context

        final_joint_state = list(continuous_joint_reference)
        initial_state = request.world.robot_state
        expected_final_state = RobotState(
            robot_id=initial_state.robot_id,
            joint_names=list(joint_names),
            joint_positions_rad=final_joint_state,
            joint_velocities_rad_s=[0.0] * len(joint_names),
            eef_pose=(
                segments[-1].waypoints[-1].eef_pose
                if segments[-1].waypoints[-1].eef_pose is not None
                else None
            ),
            gripper=current_gripper,
            attached_object_id=(
                current_attached_object_id
            ),
            held_tool_id=current_held_tool_id,
        )
        return MotionPlan(
            plan_id=plan_id,
            request_id=request.request_id,
            provenance=provenance,
            scene_signature=request.world.scene.signature,
            robot_id=initial_state.robot_id,
            joint_names=list(joint_names),
            duration_s=clock,
            segments=segments,
            events=events,
            expected_final_state=expected_final_state,
            metadata={
                "selection_policy": "FIRST_FEASIBLE_CONNECTED_SEQUENCE",
                "strategy_id": connected.strategy_id,
                "ik_branch_ids": [node.solution.branch_id for node in connected.nodes],
                "edge_evaluations": connected.edge_evaluations,
            },
        )
