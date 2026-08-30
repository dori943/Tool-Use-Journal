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
                raise MotionPlanBuildError(
                    f"final validation failed for {node.keyframe.keyframe_id!r}"
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
                        parameters={
                            str(key): value for key, value in raw_parameters.items()
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
