"""MOVE_TO_WORKSPACE targets must be valid under the mounted EE."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tuj.m5_motion.move_to_workspace import (
    MoveToWorkspaceFailureCode,
    MoveToWorkspacePlanner,
    MoveToWorkspacePlanningError,
    append_safe_rack_exit_plan,
    commissioned_reverse_workspace_path,
    select_mounted_workspace_target,
)
from tuj.m5_motion.precomputed_ee_attach import (
    EEAttachTrajectorySegmentTemplate,
    EEAttachTrajectoryTemplate,
)
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    CollisionContext,
    GoalType,
    InterpolationType,
    JointDynamicLimit,
    KeyframePlannerType,
    KeyframeType,
    ModuleName,
    MotionConstraints,
    MotionGoal,
    MotionPlan,
    MotionPlanRequest,
    MotionTask,
    Pose,
    RelativeKeyframeSpec,
    RobotState,
    SceneRef,
    SegmentType,
    TrajectorySegment,
    TrajectoryWaypoint,
    WorldSnapshot,
)


JOINT_NAMES = [
    "robot0_shoulder_pan_joint",
    "robot0_shoulder_lift_joint",
    "robot0_elbow_joint",
    "robot0_wrist_1_joint",
    "robot0_wrist_2_joint",
    "robot0_wrist_3_joint",
]


def _q(value: float) -> list[float]:
    return [value, -1.0, 1.0, -1.0, -1.5, 0.0]


def _pose(z: float) -> Pose:
    return Pose(
        frame_id="world",
        position_m=(0.0, 0.0, z),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )


def _waypoint(time_s: float, joints: list[float], z: float) -> TrajectoryWaypoint:
    return TrajectoryWaypoint(
        time_from_start_s=time_s,
        joint_positions_rad=list(joints),
        joint_velocities_rad_s=[0.0] * 6,
        joint_accelerations_rad_s2=[0.0] * 6,
        eef_pose=_pose(z),
    )


def _template() -> EEAttachTrajectoryTemplate:
    # bare start (0) invalid mounted; staging mid (1) valid; staging/retreat
    # end (2) valid. Dock uses 3.0 so it is not on the 2→1 interpolation.
    staging = EEAttachTrajectorySegmentTemplate(
        segment_id="seg:new-staging",
        segment_type=SegmentType.EE_DOCK,
        collision_context_before="bare-flange",
        collision_context_after="bare-flange",
        waypoints=[
            _waypoint(0.0, _q(0.0), 1.0),
            _waypoint(1.0, _q(1.0), 1.1),
            _waypoint(2.0, _q(2.0), 1.2),
        ],
    )
    dock = EEAttachTrajectorySegmentTemplate(
        segment_id="seg:new-dock",
        segment_type=SegmentType.EE_DOCK,
        collision_context_before="ee-attached-dock-contact:vac",
        collision_context_after="ee-attached-dock-contact:vac",
        waypoints=[
            _waypoint(2.0, _q(2.0), 1.2),
            _waypoint(3.0, _q(3.0), 1.05),
        ],
    )
    retreat = EEAttachTrajectorySegmentTemplate(
        segment_id="seg:new-retreat",
        segment_type=SegmentType.EE_DOCK,
        collision_context_before="ee-attached:vac",
        collision_context_after="ee-attached:vac",
        waypoints=[
            _waypoint(3.0, _q(3.0), 1.05),
            _waypoint(4.0, _q(2.0), 1.2),
        ],
    )
    return EEAttachTrajectoryTemplate(
        schema_version="1.0",
        trajectory_id="test-bare-to-vac",
        environment_name="TestEnv",
        robot_model="UR5e",
        source_active_ee=None,
        target_active_ee="vac",
        joint_names=list(JOINT_NAMES),
        start_joint_positions_rad=_q(0.0),
        start_eef_pose=_pose(1.0),
        workcell_signature="workcell",
        rack_signature="rack",
        collision_model_versions={
            "bare-flange": "v1",
            "ee-attached-dock-contact:vac": "v1",
            "ee-attached:vac": "v1",
        },
        segments=[staging, dock, retreat],
    )


@dataclass
class _Report:
    valid: bool
    detail: str = ""
    min_clearance_m: float | None = 0.01
    failure_code: str | None = None


class _Checker:
    """Reject bare start (0.0) and dock (3.0); accept staging mid/end."""

    def __init__(self) -> None:
        self.invalid = {0.0, 3.0}

    def check(self, joints, keyframe) -> _Report:  # noqa: ANN001
        del keyframe
        value = float(joints[0])
        if value in self.invalid:
            return _Report(
                valid=False,
                detail=f"joint0={value} collides",
                min_clearance_m=-0.01,
                failure_code="COLLISION",
            )
        return _Report(valid=True)

    def final_segment_validator(self, waypoints, context) -> bool:  # noqa: ANN001
        del waypoints, context
        return True


def _keyframe() -> RelativeKeyframeSpec:
    return RelativeKeyframeSpec(
        keyframe_id="k",
        keyframe_type=KeyframeType.CUSTOM,
        frame_ref="world",
        anchor="workspace",
        approach_axis_xyz=(0.0, 0.0, 1.0),
        planner=KeyframePlannerType.SAMPLING_BASED,
        collision_context_id="ee-attached:vac",
    )


def _request(*, joints: list[float], z: float) -> MotionPlanRequest:
    world = WorldSnapshot(
        scene=SceneRef(signature="scene"),
        robot_state=RobotState(
            robot_id="ur5e_0",
            joint_names=list(JOINT_NAMES),
            joint_positions_rad=list(joints),
            joint_velocities_rad_s=[0.0] * 6,
            eef_pose=_pose(z),
        ),
        metadata={
            "environment_name": "TestEnv",
            "physical_active_ee": "vac",
            "robot_model": "UR5e",
        },
    )
    return MotionPlanRequest(
        request_id="request:workspace",
        provenance=ArtifactProvenance(
            artifact_id="request-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="test",
        ),
        world=world,
        task=MotionTask(
            task_id="task",
            subgoal_id="SG",
            action_type="MOVE_TO_WORKSPACE",
            ee="vac",
            goal=MotionGoal(goal_type=GoalType.POSE),
            metadata={"operation": "MOVE_TO_WORKSPACE"},
        ),
        constraints=MotionConstraints(
            joint_limits={
                name: JointDynamicLimit(
                    max_velocity_rad_s=1.0,
                    max_acceleration_rad_s2=2.0,
                    max_jerk_rad_s3=5.0,
                )
                for name in JOINT_NAMES
            },
            max_joint_path_step_rad=0.5,
        ),
    )


def test_select_mounted_workspace_target_skips_invalid_bare_start() -> None:
    template = _template()
    target, pose, source = select_mounted_workspace_target(
        template, _keyframe(), _Checker()
    )
    assert source == "nearest-mounted-valid-attach-waypoint"
    assert target[0] == pytest.approx(1.0)
    assert pose is not None
    assert pose.position_m[2] == pytest.approx(1.1)


def test_commissioned_reverse_prefers_staging_side_duplicate() -> None:
    template = _template()
    start = tuple(_q(2.0))
    path = commissioned_reverse_workspace_path(
        start,
        template,
        _keyframe(),
        _Checker(),
        max_joint_step_rad=0.5,
    )
    assert len(path) >= 2
    assert path[0][0] == pytest.approx(2.0)
    assert path[-1][0] == pytest.approx(1.0)
    assert all(sample[0] != pytest.approx(3.0) for sample in path)


def test_planner_uses_commissioned_reverse_not_bare_start() -> None:
    template = _template()
    request = _request(joints=_q(2.0), z=1.2)
    context = CollisionContext(
        context_id="ee-attached:vac",
        active_ee="vac",
        collision_model_version="v1",
    )
    planner = MoveToWorkspacePlanner(
        registry=SimpleNamespace(),
        joint_position_limits_rad=[(-3.14, 3.14)] * 6,
        log=lambda *_a, **_k: None,
    )
    plan = planner.plan(
        request,
        collision_contexts={"ee-attached:vac": context},
        collision_checker=_Checker(),
        template=template,
    )
    assert plan.segments[0].metadata["planner"] == "COMMISSIONED_REVERSE"
    assert plan.metadata["workspace_target_source"] == "commissioned-reverse"
    assert plan.expected_final_state.joint_positions_rad[0] == pytest.approx(1.0)
    assert plan.expected_final_state.joint_positions_rad[0] != pytest.approx(0.0)


def test_short_reverse_continues_to_mounted_workspace_target() -> None:
    """A reverse that cannot leave the rack corridor must not finalize there."""

    staging = EEAttachTrajectorySegmentTemplate(
        segment_id="seg:staging",
        segment_type=SegmentType.EE_DOCK,
        collision_context_before="bare-flange",
        collision_context_after="bare-flange",
        waypoints=[
            _waypoint(0.0, _q(0.0), 1.0),
            _waypoint(1.0, _q(0.5), 1.05),
            _waypoint(2.0, _q(1.0), 1.1),
            _waypoint(3.0, _q(2.0), 1.2),
        ],
    )
    template = EEAttachTrajectoryTemplate(
        schema_version="1.0",
        trajectory_id="test-bare-to-vac-short-reverse",
        environment_name="TestEnv",
        robot_model="UR5e",
        source_active_ee=None,
        target_active_ee="vac",
        joint_names=list(JOINT_NAMES),
        start_joint_positions_rad=_q(0.0),
        start_eef_pose=_pose(1.0),
        workcell_signature="workcell",
        rack_signature="rack",
        collision_model_versions={
            "bare-flange": "v1",
            "ee-attached:vac": "v1",
        },
        segments=[staging],
    )

    class _ShortReverseChecker:
        """Bare start and mid corridor invalid; workspace mid and retreat valid."""

        def check(self, joints, keyframe) -> _Report:  # noqa: ANN001
            del keyframe
            value = float(joints[0])
            if value <= 0.0 + 1e-9 or abs(value - 1.0) <= 1e-9:
                return _Report(
                    valid=False,
                    detail=f"joint0={value} collides",
                    min_clearance_m=-0.01,
                    failure_code="COLLISION",
                )
            return _Report(valid=True)

        def final_segment_validator(self, waypoints, context) -> bool:  # noqa: ANN001
            del waypoints, context
            return True

    request = _request(joints=_q(2.0), z=1.2)
    request.constraints.max_joint_path_step_rad = 5.0
    context = CollisionContext(
        context_id="ee-attached:vac",
        active_ee="vac",
        collision_model_version="v1",
    )
    planner = MoveToWorkspacePlanner(
        registry=SimpleNamespace(),
        joint_position_limits_rad=[(-3.14, 3.14)] * 6,
        log=lambda *_a, **_k: None,
    )
    plan = planner.plan(
        request,
        collision_contexts={"ee-attached:vac": context},
        collision_checker=_ShortReverseChecker(),
        template=template,
    )
    assert plan.segments[0].metadata["planner"] != "COMMISSIONED_REVERSE"
    assert plan.metadata["workspace_target_source"] == (
        "nearest-mounted-valid-attach-waypoint"
    )
    assert plan.expected_final_state.joint_positions_rad[0] == pytest.approx(0.5)
    assert plan.metadata.get("safe_rack_exit") is True


def test_invalid_start_fails_closed() -> None:
    template = _template()
    request = _request(joints=_q(0.0), z=1.0)
    context = CollisionContext(
        context_id="ee-attached:vac",
        active_ee="vac",
        collision_model_version="v1",
    )
    planner = MoveToWorkspacePlanner(
        registry=SimpleNamespace(),
        joint_position_limits_rad=[(-3.14, 3.14)] * 6,
        log=lambda *_a, **_k: None,
    )
    with pytest.raises(MoveToWorkspacePlanningError) as raised:
        planner.plan(
            request,
            collision_contexts={"ee-attached:vac": context},
            collision_checker=_Checker(),
            template=template,
        )
    assert raised.value.failure_code == MoveToWorkspaceFailureCode.PATH_NOT_FOUND


def test_append_safe_rack_exit_shifts_segments_and_final_state() -> None:
    context = CollisionContext(
        context_id="ee-attached:vac",
        active_ee="vac",
        collision_model_version="v1",
    )

    def _plan(*, joints: list[float], duration: float, plan_id: str) -> MotionPlan:
        return MotionPlan(
            plan_id=plan_id,
            request_id="request:exchange",
            provenance=ArtifactProvenance(
                artifact_id=f"artifact:{plan_id}",
                artifact_type="MotionPlan",
                produced_by=ModuleName.MOTION_PLANNER,
                invocation_id=plan_id,
            ),
            scene_signature="scene",
            robot_id="ur5e_0",
            joint_names=list(JOINT_NAMES),
            duration_s=duration,
            segments=[
                TrajectorySegment(
                    segment_id=f"{plan_id}:seg",
                    segment_type=SegmentType.TRANSFER,
                    start_time_s=0.0,
                    end_time_s=duration,
                    interpolation=InterpolationType.LINEAR,
                    waypoints=[
                        TrajectoryWaypoint(
                            time_from_start_s=0.0,
                            joint_positions_rad=list(joints),
                        ),
                        TrajectoryWaypoint(
                            time_from_start_s=duration,
                            joint_positions_rad=list(joints),
                        ),
                    ],
                    collision_checked=True,
                    collision_context_before=context,
                    collision_context_after=context,
                    metadata={"planner": "TEST"},
                )
            ],
            expected_final_state=RobotState(
                robot_id="ur5e_0",
                joint_names=list(JOINT_NAMES),
                joint_positions_rad=list(joints),
                joint_velocities_rad_s=[0.0] * 6,
            ),
            metadata={"source": "precomputed"},
        )

    merged = append_safe_rack_exit_plan(
        _plan(joints=_q(2.0), duration=1.0, plan_id="exchange"),
        _plan(joints=_q(0.5), duration=2.0, plan_id="exit"),
    )
    assert merged.duration_s == pytest.approx(3.0)
    assert len(merged.segments) == 2
    assert merged.segments[1].start_time_s == pytest.approx(1.0)
    assert merged.segments[1].metadata["safe_rack_exit"] is True
    assert merged.expected_final_state.joint_positions_rad[0] == pytest.approx(0.5)
    assert merged.metadata["safe_rack_exit"] is True
