"""Trajectory contract, validation, and checked-in schema tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from task_planner.models import GraspSpec

from motion_planner.schema import (
    SCHEMA_VERSION,
    ArtifactProvenance,
    CollisionContext,
    EventType,
    ExecutionReport,
    ExecutionStatus,
    FailureObservation,
    FreeObjectPose,
    GoalType,
    KeyframePlanArtifact,
    KeyframePlanCandidate,
    KeyframePlannerType,
    KeyframeType,
    ModuleName,
    MotionGoal,
    MotionPlan,
    MotionPlanRequest,
    MotionTask,
    Pose,
    RelativeKeyframeSpec,
    RecoveryDirective,
    RetryPolicy,
    RobotState,
    RootCause,
    SceneRef,
    SegmentType,
    SimulationConfig,
    SimulationMetrics,
    SimulationRun,
    StrategyGenerationProvenance,
    StrategyGeneratorKind,
    TrajectoryEvent,
    TrajectorySegment,
    TrajectoryWaypoint,
    WorldSnapshot,
    motion_planner_json_schema,
)


def _provenance(
    artifact_id: str,
    artifact_type: str,
    module: ModuleName,
    invocation_id: str,
    inputs: tuple[str, ...] = (),
) -> ArtifactProvenance:
    return ArtifactProvenance(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        produced_by=module,
        invocation_id=invocation_id,
        input_artifact_ids=list(inputs),
    )


def _pose(z: float = 0.3) -> Pose:
    return Pose(
        frame_id="world",
        position_m=(0.1, 0.2, z),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )


def _state(positions=(0.0, 0.0)) -> RobotState:
    return RobotState(
        robot_id="ur5e-1",
        joint_names=["shoulder", "elbow"],
        joint_positions_rad=list(positions),
    )


def _grasp() -> GraspSpec:
    return GraspSpec(
        grasp_id="g-object-1",
        owner_kind="object",
        owner_id="obj1",
        pose=_pose().model_dump(mode="json"),
        score=0.9,
        source="affordance-r1",
    )


def _plan() -> MotionPlan:
    segment = TrajectorySegment(
        segment_id="approach-1",
        segment_type=SegmentType.APPROACH,
        start_time_s=0.0,
        end_time_s=1.0,
        collision_checked=True,
        waypoints=[
            TrajectoryWaypoint(
                time_from_start_s=0.0,
                joint_positions_rad=[0.0, 0.0],
            ),
            TrajectoryWaypoint(
                time_from_start_s=1.0,
                joint_positions_rad=[0.2, -0.1],
            ),
        ],
    )
    return MotionPlan(
        plan_id="plan-1",
        request_id="request-1",
        provenance=_provenance(
            "artifact-plan-1",
            "MotionPlan",
            ModuleName.MOTION_PLANNER,
            "motion-invocation-1",
            ("artifact-request-1",),
        ),
        scene_signature="scene-1",
        robot_id="ur5e-1",
        joint_names=["shoulder", "elbow"],
        duration_s=1.0,
        segments=[segment],
        events=[
            TrajectoryEvent(
                event_id="close-1",
                time_from_start_s=1.0,
                event_type=EventType.GRIPPER_CLOSE,
                target_id="obj1",
            )
        ],
        expected_final_state=_state((0.2, -0.1)),
    )


def test_motion_plan_request_carries_grounded_task_and_world() -> None:
    request = MotionPlanRequest(
        request_id="request-1",
        provenance=_provenance(
            "artifact-request-1",
            "MotionPlanRequest",
            ModuleName.TASK_PLANNER,
            "task-planner-invocation-1",
        ),
        world=WorldSnapshot(
            scene=SceneRef(signature="scene-1"),
            robot_state=_state(),
            objects={"obj1": {"pose": _pose().model_dump(mode="json")}},
        ),
        task=MotionTask(
            task_id="task-1",
            subgoal_id="pick-obj1",
            action_type="pick",
            ee="2f",
            target_ids=["obj1"],
            grasp=_grasp(),
            goal=MotionGoal(
                goal_type=GoalType.PICK,
                target_object_id="obj1",
                approach_direction=(0.0, 0.0, -1.0),
                approach_distance_m=0.1,
            ),
        ),
    )
    assert request.schema_version == SCHEMA_VERSION
    assert request.task.grasp is not None
    assert request.world.scene.signature == "scene-1"


def test_pick_task_requires_grasp() -> None:
    with pytest.raises(ValidationError, match="PICK task requires"):
        MotionTask(
            task_id="task-1",
            subgoal_id="pick-obj1",
            action_type="pick",
            ee="2f",
            goal=MotionGoal(goal_type=GoalType.PICK),
        )


def test_robot_state_rejects_mismatched_joint_dimensions() -> None:
    with pytest.raises(ValidationError, match="length must match joint_names"):
        RobotState(
            robot_id="ur5e-1",
            joint_names=["j1", "j2"],
            joint_positions_rad=[0.0],
        )


def test_collision_context_keeps_free_and_attached_objects_disjoint() -> None:
    with pytest.raises(ValidationError, match="must be disjoint"):
        CollisionContext(
            context_id="invalid",
            attached_object_ids=["obj1"],
            free_object_poses=[
                FreeObjectPose(
                    object_id="obj1",
                    free_joint_name="obj1_free",
                    pose=_pose(),
                )
            ],
            collision_model_version="model-v1",
        )


def test_pose_rejects_a_non_normalized_quaternion() -> None:
    with pytest.raises(ValidationError, match="must be normalized"):
        Pose(
            frame_id="world",
            position_m=(0.0, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 2.0),
        )


def test_keyframe_candidate_is_relative_and_provenance_frozen() -> None:
    candidate = KeyframePlanCandidate(
        strategy_id="top-down",
        keyframes=[
            RelativeKeyframeSpec(
                keyframe_id="pre",
                keyframe_type=KeyframeType.PRE_GRASP,
                frame_ref="object:obj1",
                anchor="top_center",
                approach_axis_xyz=(0.0, 0.0, 1.0),
                offset_along_approach_m=0.1,
                planner=KeyframePlannerType.CARTESIAN,
            ),
            RelativeKeyframeSpec(
                keyframe_id="grasp",
                keyframe_type=KeyframeType.GRASP,
                frame_ref="object:obj1",
                anchor="top_center",
                approach_axis_xyz=(0.0, 0.0, 1.0),
                planner=KeyframePlannerType.CARTESIAN,
            ),
        ],
        provenance=StrategyGenerationProvenance(
            generator_kind=StrategyGeneratorKind.VLM,
            generator_id="keyframe-generator-v1",
            input_hash="input-hash",
            model_id="vlm-model",
            prompt_hash="prompt-hash",
            temperature=0.0,
        ),
    )
    artifact = KeyframePlanArtifact(
        artifact_id="keyframes-1",
        provenance=_provenance(
            "keyframes-1",
            "KeyframePlanArtifact",
            ModuleName.MOTION_PLANNER,
            "keyframe-invocation-1",
        ),
        scene_signature="scene-1",
        subgoal_id="pick-obj1",
        candidates=[candidate],
    )
    assert artifact.candidates[0].keyframes[0].frame_ref == "object:obj1"


def test_motion_plan_rejects_discontinuous_collision_contexts() -> None:
    context_a = CollisionContext(
        context_id="object-world",
        active_ee="2f",
        collision_model_version="model-1",
    )
    context_b = CollisionContext(
        context_id="object-attached",
        active_ee="2f",
        attached_object_ids=["obj1"],
        collision_model_version="model-2",
    )
    first = TrajectorySegment(
        segment_id="first",
        segment_type=SegmentType.GRASP,
        start_time_s=0.0,
        end_time_s=1.0,
        collision_checked=True,
        collision_context_after=context_a,
        waypoints=[
            TrajectoryWaypoint(time_from_start_s=0.0, joint_positions_rad=[0.0, 0.0]),
            TrajectoryWaypoint(time_from_start_s=1.0, joint_positions_rad=[0.1, 0.0]),
        ],
    )
    second = TrajectorySegment(
        segment_id="second",
        segment_type=SegmentType.LIFT,
        start_time_s=1.0,
        end_time_s=2.0,
        collision_checked=True,
        collision_context_before=context_b,
        waypoints=[
            TrajectoryWaypoint(time_from_start_s=1.0, joint_positions_rad=[0.1, 0.0]),
            TrajectoryWaypoint(time_from_start_s=2.0, joint_positions_rad=[0.2, 0.0]),
        ],
    )
    with pytest.raises(ValidationError, match="must preserve scene state"):
        MotionPlan(
            plan_id="plan-context",
            request_id="request-1",
            provenance=_provenance(
                "artifact-plan-context",
                "MotionPlan",
                ModuleName.MOTION_PLANNER,
                "motion-invocation-context",
            ),
            scene_signature="scene-1",
            robot_id="ur5e-1",
            joint_names=["shoulder", "elbow"],
            duration_s=2.0,
            segments=[first, second],
            expected_final_state=_state((0.2, 0.0)),
        )


def test_trajectory_waypoint_times_are_strictly_increasing() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        TrajectorySegment(
            segment_id="bad",
            segment_type=SegmentType.CUSTOM,
            start_time_s=0.0,
            end_time_s=1.0,
            collision_checked=False,
            waypoints=[
                TrajectoryWaypoint(
                    time_from_start_s=0.0,
                    joint_positions_rad=[0.0, 0.0],
                ),
                TrajectoryWaypoint(
                    time_from_start_s=0.0,
                    joint_positions_rad=[0.1, 0.1],
                ),
            ],
        )


def test_motion_plan_validates_joint_dof_and_event_timeline() -> None:
    plan = _plan()
    assert plan.duration_s == 1.0
    assert plan.events[0].time_from_start_s == plan.duration_s


def test_failed_execution_report_requires_failure_observation() -> None:
    with pytest.raises(ValidationError, match="requires a failure observation"):
        ExecutionReport(
            report_id="report-1",
            run_id="run-1",
            plan_id="plan-1",
            provenance=_provenance(
                "artifact-report-1",
                "ExecutionReport",
                ModuleName.SIMULATOR,
                "sim-invocation-1",
                ("artifact-plan-1",),
            ),
            status=ExecutionStatus.FAILED,
            metrics=SimulationMetrics(
                executed_duration_s=0.5,
                max_joint_tracking_error_rad=0.1,
            ),
        )


def test_plan_can_be_embedded_in_an_internal_simulation_run() -> None:
    request = SimulationRun(
        run_id="run-1",
        provenance=_provenance(
            "artifact-run-1",
            "SimulationRun",
            ModuleName.SIMULATOR,
            "sim-invocation-1",
            ("artifact-plan-1",),
        ),
        plan=_plan(),
        config=SimulationConfig(render=False, random_seed=7),
    )
    assert request.plan.plan_id == "plan-1"
    assert request.config.random_seed == 7


def test_recovery_directive_targets_only_the_root_cause_module() -> None:
    cause = RootCause(
        module=ModuleName.GRASP_PLANNER,
        invocation_id="grasp-invocation-1",
        artifact_id="artifact-grasp-1",
        cause_code="OBJECT_SLIP",
        confidence=0.95,
        evidence_refs=["trace://run-1/contact"],
        explanation="object detached after the close event",
    )
    directive = RecoveryDirective(
        directive_id="retry-1",
        source_report_id="report-1",
        provenance=_provenance(
            "artifact-retry-1",
            "RecoveryDirective",
            ModuleName.RECOVERY_ORCHESTRATOR,
            "recovery-invocation-1",
            ("artifact-report-1",),
        ),
        root_cause=cause,
        target_module=ModuleName.GRASP_PLANNER,
        restart_from_artifact_id="artifact-grasp-input-1",
        invalidated_artifact_ids=["artifact-grasp-1", "artifact-plan-1"],
        retry_policy=RetryPolicy(
            current_attempt=1,
            max_attempts=3,
            parameter_overrides={"grasp_candidate_rank": 2},
        ),
    )
    assert directive.target_module is ModuleName.GRASP_PLANNER
    assert directive.invalidated_artifact_ids == [
        "artifact-grasp-1",
        "artifact-plan-1",
    ]


def test_recovery_directive_rejects_a_different_target_module() -> None:
    with pytest.raises(ValidationError, match="must match root_cause.module"):
        RecoveryDirective(
            directive_id="retry-1",
            source_report_id="report-1",
            provenance=_provenance(
                "artifact-retry-1",
                "RecoveryDirective",
                ModuleName.RECOVERY_ORCHESTRATOR,
                "recovery-invocation-1",
            ),
            root_cause=RootCause(
                module=ModuleName.MOTION_PLANNER,
                invocation_id="motion-invocation-1",
                artifact_id="artifact-plan-1",
                cause_code="COLLISION",
                confidence=1.0,
                explanation="trajectory collided during transfer",
            ),
            target_module=ModuleName.TASK_PLANNER,
            restart_from_artifact_id="artifact-request-1",
        )


def test_json_schema_exposes_trajectory_and_simulation_contracts() -> None:
    schema = motion_planner_json_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["x-schema-version"] == SCHEMA_VERSION
    definitions = schema["$defs"]
    for name in schema["x-contract-roots"]:
        assert name in definitions
    for name in (
        "KeyframePlanArtifact",
        "KeyframePlanCandidate",
        "RelativeKeyframeSpec",
        "CollisionContext",
        "FreeObjectPose",
        "MotionPlan",
        "TrajectorySegment",
        "TrajectoryWaypoint",
        "TrajectoryEvent",
        "SimulationMetrics",
        "ArtifactProvenance",
        "RootCause",
        "RecoveryDirective",
    ):
        assert name in definitions
    assert schema["x-operations"]["generate_trajectory"] == {
        "input": "MotionPlanRequest",
        "emits": "MotionPlan",
        "visibility": "internal",
    }
    assert schema["x-result-routing"]["return_to_task_planner"] is False


def test_checked_in_json_schema_matches_runtime_contract() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "motion_planner.schema.json"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == (
        motion_planner_json_schema()
    )
