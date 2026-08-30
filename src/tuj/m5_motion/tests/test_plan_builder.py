"""Connected branch sequences become time-parameterized, scene-aware plans."""

from __future__ import annotations

from tuj.m5_motion.kinematics import IKResult
from tuj.m5_motion.plan_builder import MotionPlanBuilder
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    CollisionContext,
    EventType,
    GoalType,
    GripperMode,
    JointDynamicLimit,
    KeyframeEventType,
    KeyframePlannerType,
    KeyframeType,
    ModuleName,
    MotionConstraints,
    MotionGoal,
    MotionPlanRequest,
    MotionTask,
    Pose,
    RelativeKeyframeSpec,
    RobotState,
    SceneRef,
    WorldSnapshot,
)
from tuj.m5_motion.strategy import (
    ConnectedStrategy,
    EdgePlanResult,
    SelectedIKNode,
)


def _request() -> MotionPlanRequest:
    return MotionPlanRequest(
        request_id="request-1",
        provenance=ArtifactProvenance(
            artifact_id="request-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="task-planner-1",
        ),
        world=WorldSnapshot(
            scene=SceneRef(signature="scene-1"),
            robot_state=RobotState(
                robot_id="ur5e-1",
                joint_names=["j1", "j2"],
                joint_positions_rad=[0.0, 0.0],
            ),
        ),
        task=MotionTask(
            task_id="task-1",
            subgoal_id="move-1",
            action_type="move",
            ee="2f",
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_pose=Pose(
                    frame_id="world",
                    position_m=(0.1, 0.2, 0.3),
                    orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
            ),
        ),
        constraints=MotionConstraints(
            joint_limits={
                "j1": JointDynamicLimit(
                    max_velocity_rad_s=1.0,
                    max_acceleration_rad_s2=2.0,
                    max_jerk_rad_s3=10.0,
                ),
                "j2": JointDynamicLimit(
                    max_velocity_rad_s=1.0,
                    max_acceleration_rad_s2=2.0,
                    max_jerk_rad_s3=10.0,
                ),
            }
        ),
    )


def _node(keyframe, q, branch="B1") -> SelectedIKNode:
    return SelectedIKNode(
        keyframe=keyframe,
        solution=IKResult(
            solved=True,
            qpos=q,
            position_error_m=0.0,
            orientation_error_rad=0.0,
            branch_id=branch,
        ),
    )


def test_plan_builder_applies_event_scoped_collision_state() -> None:
    grasp = RelativeKeyframeSpec(
        keyframe_id="grasp",
        keyframe_type=KeyframeType.GRASP,
        frame_ref="object:obj1",
        anchor="center",
        approach_axis_xyz=(0.0, 0.0, 1.0),
        planner=KeyframePlannerType.CARTESIAN,
        events_after=[KeyframeEventType.ATTACH_OBJECT],
        collision_context_id="grasp-contact",
        collision_context_after_events_id="object-attached",
        metadata={
            "event_target_id": "obj1",
            "event_parameters": {
                "ATTACH_OBJECT": {
                    "attachment_mode": "BREAKABLE_WELD",
                    "max_weld_force_n": 55.0,
                }
            },
        },
    )
    lift = RelativeKeyframeSpec(
        keyframe_id="lift",
        keyframe_type=KeyframeType.LIFT,
        frame_ref="object:obj1",
        anchor="center",
        approach_axis_xyz=(0.0, 0.0, 1.0),
        planner=KeyframePlannerType.CARTESIAN,
        collision_context_id="object-attached",
    )
    connected = ConnectedStrategy(
        strategy_id="pick-top",
        nodes=(_node(grasp, (0.2, 0.1)), _node(lift, (0.4, 0.2))),
        edges=(
            EdgePlanResult(valid=True, joint_path=((0.0, 0.0), (0.2, 0.1))),
            EdgePlanResult(valid=True, joint_path=((0.2, 0.1), (0.4, 0.2))),
        ),
        edge_evaluations=3,
    )
    contact = CollisionContext(
        context_id="grasp-contact",
        scene_state_id="object-world",
        active_ee="2f",
        allowed_collision_pairs=[("left_finger", "obj1")],
        collision_model_version="model-world",
    )
    attached = CollisionContext(
        context_id="object-attached",
        scene_state_id="object-attached",
        active_ee="2f",
        attached_object_ids=["obj1"],
        touch_links=["left_finger", "right_finger"],
        collision_model_version="model-attached",
    )

    plan = MotionPlanBuilder().build(
        _request(),
        connected,
        plan_id="plan-1",
        provenance=ArtifactProvenance(
            artifact_id="plan-artifact",
            artifact_type="MotionPlan",
            produced_by=ModuleName.MOTION_PLANNER,
            invocation_id="motion-1",
            input_artifact_ids=["request-artifact"],
        ),
        collision_contexts={
            contact.context_id: contact,
            attached.context_id: attached,
        },
        initial_collision_context_id=contact.context_id,
        final_segment_validator=lambda waypoints, context: bool(waypoints),
    )

    assert plan.events[0].event_type is EventType.ATTACH_OBJECT
    assert plan.events[0].parameters == {
        "attachment_mode": "BREAKABLE_WELD",
        "max_weld_force_n": 55.0,
    }
    assert plan.segments[0].collision_context_after == attached
    assert plan.segments[1].collision_context_before == attached
    assert plan.expected_final_state.attached_object_id == "obj1"
    assert plan.expected_final_state.gripper is not None
    assert plan.expected_final_state.gripper.mode is GripperMode.HOLDING
    assert plan.metadata["selection_policy"] == "FIRST_FEASIBLE_CONNECTED_SEQUENCE"


def test_plan_builder_holds_pose_between_close_and_attach_events() -> None:
    grasp = RelativeKeyframeSpec(
        keyframe_id="settled-grasp",
        keyframe_type=KeyframeType.GRASP,
        frame_ref="object:obj1",
        anchor="center",
        approach_axis_xyz=(0.0, 0.0, 1.0),
        planner=KeyframePlannerType.CARTESIAN,
        events_after=[
            KeyframeEventType.GRIPPER_CLOSE,
            KeyframeEventType.ATTACH_OBJECT,
        ],
        collision_context_id="grasp-contact",
        collision_context_after_events_id="object-attached",
        metadata={
            "event_target_id": "obj1",
            "hold_duration_after_s": 0.75,
            "tracking_settle": {
                "joint_tolerance_rad": 0.02,
                "eef_tolerance_m": 0.01,
                "max_wait_s": 3.0,
                "required_consecutive_ticks": 5,
            },
            "physical_tool_control": {
                "target_clearance_m": 0.0015,
                "clearance_tolerance_m": 0.003,
                "max_table_penetration_m": 0.001,
                "gain": 4.0,
                "rate_m_s": 0.03,
                "max_offset_m": 0.03,
                "activation_band_m": 0.02,
                "max_joint_offset_rad": 0.15,
            },
            "physical_tool_settle": {
                "target_position_m": [0.1, 0.2, 0.3],
                "target_clearance_m": 0.0015,
                "xy_tolerance_m": 0.015,
                "clearance_tolerance_m": 0.003,
                "max_table_penetration_m": 0.001,
                "max_tool_speed_m_s": 0.02,
            },
            "physical_tool_target_position_m": [0.4, 0.5, 0.6],
            "physical_push_control": {
                "push_axis_world": [2.0, 0.0, 0.0],
                "contact_offset_local_m": [0.0, 0.09, 0.0],
                "block_support_m": 0.01,
                "contact_penetration_m": 0.001,
                "max_correction_m": 0.08,
                "contact_plan_time_scale": 0.5,
                "reacquire_timeout_s": 8.0,
                "max_reacquire_attempts": 2,
                "contact_height_offset_from_block_center_m": 0.0,
                "contact_height_target_m": 0.806,
                "block_support_center_z_m": 0.806,
                "block_support_tolerance_m": 0.001,
                "contact_height_gain": 2.0,
                "contact_height_rate_m_s": 0.03,
                "contact_height_max_offset_m": 0.02,
                "contact_height_max_downward_offset_m": 0.002,
            },
            "event_time_offsets_s": {
                "GRIPPER_CLOSE": 0.0,
                "ATTACH_OBJECT": 0.75,
            },
            "event_parameters": {
                "GRIPPER_CLOSE": {"command": 0.25},
            },
        },
    )

    connected = ConnectedStrategy(
        strategy_id="settled-pick",
        nodes=(_node(grasp, (0.2, 0.1)),),
        edges=(
            EdgePlanResult(
                valid=True,
                joint_path=((0.0, 0.0), (0.2, 0.1)),
            ),
        ),
        edge_evaluations=1,
    )
    contact = CollisionContext(
        context_id="grasp-contact",
        scene_state_id="object-world",
        active_ee="2f",
        collision_model_version="model-world",
    )
    attached = CollisionContext(
        context_id="object-attached",
        scene_state_id="object-attached",
        active_ee="2f",
        attached_object_ids=["obj1"],
        collision_model_version="model-attached",
    )

    plan = MotionPlanBuilder().build(
        _request(),
        connected,
        plan_id="settled-pick-plan",
        provenance=ArtifactProvenance(
            artifact_id="settled-pick-artifact",
            artifact_type="MotionPlan",
            produced_by=ModuleName.MOTION_PLANNER,
            invocation_id="settled-pick",
        ),
        collision_contexts={
            contact.context_id: contact,
            attached.context_id: attached,
        },
        initial_collision_context_id=contact.context_id,
        final_segment_validator=lambda waypoints, context: bool(waypoints),
    )

    segment = plan.segments[0]
    motion_end = float(segment.metadata["motion_end_time_s"])
    assert segment.end_time_s == motion_end + 0.75
    assert segment.waypoints[-1].time_from_start_s == segment.end_time_s
    assert segment.waypoints[-1].joint_positions_rad == (
        segment.waypoints[-2].joint_positions_rad
    )
    assert segment.metadata["tracking_settle"] == {
        "joint_tolerance_rad": 0.02,
        "eef_tolerance_m": 0.01,
        "max_wait_s": 3.0,
        "required_consecutive_ticks": 5,
    }
    assert segment.metadata["physical_tool_control"]["gain"] == 4.0
    assert segment.metadata["physical_tool_settle"]["target_position_m"] == [
        0.1,
        0.2,
        0.3,
    ]
    assert segment.metadata["physical_tool_target_position_m"] == [
        0.4,
        0.5,
        0.6,
    ]
    assert segment.metadata["physical_push_control"] == {
        "push_axis_world": [1.0, 0.0, 0.0],
        "contact_offset_local_m": [0.0, 0.09, 0.0],
        "block_support_m": 0.01,
        "contact_penetration_m": 0.001,
        "max_correction_m": 0.08,
        "contact_plan_time_scale": 0.5,
        "reacquire_timeout_s": 8.0,
        "max_reacquire_attempts": 2,
        "contact_height_offset_from_block_center_m": 0.0,
        "contact_height_target_m": 0.806,
        "block_support_center_z_m": 0.806,
        "block_support_tolerance_m": 0.001,
        "contact_height_gain": 2.0,
        "contact_height_rate_m_s": 0.03,
        "contact_height_max_offset_m": 0.02,
        "contact_height_max_downward_offset_m": 0.002,
    }
    assert plan.events[0].event_type is EventType.GRIPPER_CLOSE
    assert plan.events[0].command == 0.25
    assert plan.events[0].parameters == {}
    assert plan.events[0].time_from_start_s == motion_end
    assert plan.events[1].event_type is EventType.ATTACH_OBJECT
    assert plan.events[1].time_from_start_s == segment.end_time_s


def test_plan_builder_scales_timing_to_cartesian_speed_limit() -> None:
    request = _request()
    request.constraints.max_cartesian_speed_m_s = 0.05
    keyframe = RelativeKeyframeSpec(
        keyframe_id="move",
        keyframe_type=KeyframeType.CUSTOM,
        frame_ref="world",
        anchor="origin",
        approach_axis_xyz=(0.0, 0.0, 1.0),
        planner=KeyframePlannerType.JOINT,
    )
    connected = ConnectedStrategy(
        strategy_id="limited",
        nodes=(_node(keyframe, (0.1, 0.0)),),
        edges=(
            EdgePlanResult(
                valid=True,
                joint_path=((0.0, 0.0), (0.1, 0.0)),
            ),
        ),
        edge_evaluations=1,
    )
    context = CollisionContext(
        context_id="default",
        collision_model_version="model",
    )
    builder = MotionPlanBuilder(
        forward_pose=lambda q: (
            (float(q[0]), float(q[1]), 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    plan = builder.build(
        request,
        connected,
        plan_id="speed-limited",
        provenance=ArtifactProvenance(
            artifact_id="speed-limited-artifact",
            artifact_type="MotionPlan",
            produced_by=ModuleName.MOTION_PLANNER,
            invocation_id="speed-limited",
        ),
        collision_contexts={context.context_id: context},
        initial_collision_context_id=context.context_id,
        final_segment_validator=lambda waypoints, selected: bool(waypoints),
    )

    assert plan.duration_s >= 2.0
    assert all(
        waypoint.eef_pose is not None
        for waypoint in plan.segments[0].waypoints
    )
