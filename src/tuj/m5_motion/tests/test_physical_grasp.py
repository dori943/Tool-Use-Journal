from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import tuj.m5_motion.physical_grasp as physical_grasp_module
from tuj.m5_motion.grasp_geometry import (
    TWO_FINGER_OPPOSED_CONTACT,
    TwoFingerOpposedContactBinder,
    annotate_support_clearance,
    bind_grasp_geometry,
    support_clearance_context,
)
from tuj.m5_motion.physical_grasp import (
    ContactFrictionRetentionMonitor,
    PhysicalGraspControllerTrajectoryPlayer,
    PhysicalGraspMonitor,
    RetargetedAcquireKeyframeProvider,
    RuntimeGraspParameters,
    _runtime_grasp_parameters,
    with_contact_friction_grasp,
    with_contact_friction_release,
)
from tuj.m5_motion.profiles import (
    GraspValidationProfile,
    PhysicalGraspProfile,
)
from tuj.m5_motion.schema import (
    AttachedObjectTransform,
    ArtifactProvenance,
    ExecutionReport,
    ExecutionStatus,
    GoalType,
    KeyframeEventType,
    KeyframePlanArtifact,
    KeyframePlanCandidate,
    KeyframePlannerType,
    KeyframeType,
    ModuleName,
    MotionGoal,
    MotionPlan,
    MotionPlanRequest,
    MotionTask,
    RelativeKeyframeSpec,
    RobotState,
    SceneRef,
    SegmentType,
    SimulationConfig,
    SimulationMetrics,
    SimulationRun,
    StrategyGenerationProvenance,
    StrategyGeneratorKind,
    TrajectorySegment,
    TrajectoryWaypoint,
    WorldSnapshot,
)
from tuj.m5_motion.tool_use_journal_runtime import (
    AttachmentContactMetrics,
    ToolUseJournalAttachmentBroken,
    ToolUseJournalControllerTrajectoryPlayer,
)


def _artifact() -> KeyframePlanArtifact:
    strategy_provenance = StrategyGenerationProvenance(
        generator_kind=StrategyGeneratorKind.TEMPLATE,
        generator_id="fixture",
        input_hash="input",
    )
    keyframes = [
        RelativeKeyframeSpec(
            keyframe_id="pre",
            keyframe_type=KeyframeType.PRE_GRASP,
            frame_ref="object:tool",
            anchor="center",
            approach_axis_xyz=(0.0, 0.0, 1.0),
            planner=KeyframePlannerType.CARTESIAN,
        ),
        RelativeKeyframeSpec(
            keyframe_id="grasp",
            keyframe_type=KeyframeType.GRASP,
            frame_ref="object:tool",
            anchor="center",
            approach_axis_xyz=(0.0, 0.0, 1.0),
            planner=KeyframePlannerType.CARTESIAN,
            events_after=[
                KeyframeEventType.GRIPPER_CLOSE,
                KeyframeEventType.ATTACH_OBJECT,
            ],
            metadata={
                "event_parameters": {
                    "ATTACH_OBJECT": {"attachment_mode": "KINEMATIC"}
                }
            },
        ),
        RelativeKeyframeSpec(
            keyframe_id="lift",
            keyframe_type=KeyframeType.LIFT,
            frame_ref="object:tool",
            anchor="center",
            approach_axis_xyz=(0.0, 0.0, 1.0),
            offset_along_approach_m=0.1,
            planner=KeyframePlannerType.CARTESIAN,
        ),
    ]
    return KeyframePlanArtifact(
        artifact_id="keyframes",
        provenance=ArtifactProvenance(
            artifact_id="keyframes-artifact",
            artifact_type="KeyframePlanArtifact",
            produced_by=ModuleName.MOTION_PLANNER,
            invocation_id="fixture",
        ),
        scene_signature="scene",
        subgoal_id="pick",
        candidates=[
            KeyframePlanCandidate(
                strategy_id="strategy",
                keyframes=keyframes,
                provenance=strategy_provenance,
            )
        ],
    )


def _simulation_run() -> SimulationRun:
    state = RobotState(
        robot_id="robot",
        joint_names=["j1"],
        joint_positions_rad=[0.0],
    )
    plan = MotionPlan(
        plan_id="plan",
        request_id="request",
        provenance=ArtifactProvenance(
            artifact_id="plan-artifact",
            artifact_type="MotionPlan",
            produced_by=ModuleName.MOTION_PLANNER,
            invocation_id="fixture-plan",
        ),
        scene_signature="scene",
        robot_id="robot",
        joint_names=["j1"],
        duration_s=0.1,
        segments=[
            TrajectorySegment(
                segment_id="segment",
                segment_type=SegmentType.CUSTOM,
                start_time_s=0.0,
                end_time_s=0.1,
                waypoints=[
                    TrajectoryWaypoint(
                        time_from_start_s=0.0,
                        joint_positions_rad=[0.0],
                    ),
                    TrajectoryWaypoint(
                        time_from_start_s=0.1,
                        joint_positions_rad=[0.0],
                    ),
                ],
                collision_checked=True,
            )
        ],
        expected_final_state=state,
    )
    return SimulationRun(
        run_id="run",
        provenance=ArtifactProvenance(
            artifact_id="run-artifact",
            artifact_type="SimulationRun",
            produced_by=ModuleName.MOTION_PLANNER,
            invocation_id="fixture-run",
        ),
        plan=plan,
        config=SimulationConfig(),
    )


def test_failed_retention_changes_successful_trajectory_report_to_failed(
    monkeypatch,
) -> None:
    run = _simulation_run()
    trajectory_report = ExecutionReport(
        report_id="report",
        run_id=run.run_id,
        plan_id=run.plan.plan_id,
        provenance=ArtifactProvenance(
            artifact_id="report-artifact",
            artifact_type="ExecutionReport",
            produced_by=ModuleName.MOTION_PLANNER,
            invocation_id="fixture-report",
        ),
        status=ExecutionStatus.SUCCESS,
        final_robot_state=run.plan.expected_final_state,
        metrics=SimulationMetrics(
            executed_duration_s=0.1,
            max_joint_tracking_error_rad=0.0,
        ),
    )
    monkeypatch.setattr(
        ToolUseJournalControllerTrajectoryPlayer,
        "execute",
        lambda self, run, report_id=None: trajectory_report,
    )
    runtime = SimpleNamespace(
        held_tool_id=None,
        attached_object_id=None,
        mark_contact_friction_object_as_tool=lambda object_id: pytest.fail(
            f"failed grasp unexpectedly marked {object_id!r} as held"
        ),
    )
    monitor = SimpleNamespace(
        object_id="part",
        summary=lambda: {
            "status": "FAILED",
            "object_id": "part",
            "contact_formation": {"status": "FAILED"},
            "grasp_retention": {"status": "FAILED"},
        },
    )
    player = PhysicalGraspControllerTrajectoryPlayer(
        runtime,
        monitor=monitor,
    )

    report = player.execute(run)

    assert report.status is ExecutionStatus.FAILED
    assert report.failure is not None
    assert report.failure.code == "GRASP_RETENTION_FAILED"
    assert report.final_robot_state.held_tool_id is None
    assert report.metadata["physical_grasp_execution_succeeded"] is False
    assert report.metadata["trajectory_execution_status"] == "SUCCESS"


def test_transport_retention_monitor_clears_held_state_after_contact_loss(
    monkeypatch,
) -> None:
    transform = AttachedObjectTransform(
        object_id="part",
        free_joint_name="part_free",
        reference_kind="body",
        reference_name="right_hand",
        position_in_reference_m=(0.0, 0.0, 0.1),
        orientation_in_reference_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    monkeypatch.setattr(
        physical_grasp_module,
        "_contact_friction_transform",
        lambda runtime, object_id: transform,
    )
    runtime = SimpleNamespace(
        _held_tool_id="part",
        _contact_friction_retention=None,
        object_contact_metrics=lambda object_id: AttachmentContactMetrics(),
    )
    monitor = ContactFrictionRetentionMonitor(
        runtime=runtime,
        object_id="part",
        profile=PhysicalGraspProfile(),
        reference_transform=transform,
        last_valid_contact_time_s=0.0,
    )
    runtime._contact_friction_retention = monitor

    monitor.after_tick(0.05)
    with pytest.raises(ToolUseJournalAttachmentBroken) as caught:
        monitor.after_tick(0.07)

    assert "CONTACT_FRICTION_CONTACT_LOST" in caught.value.observation.reasons
    assert runtime._held_tool_id is None
    assert runtime._contact_friction_retention is None


def test_transport_retention_monitor_rejects_relative_pose_drift(
    monkeypatch,
) -> None:
    reference = AttachedObjectTransform(
        object_id="part",
        free_joint_name="part_free",
        reference_kind="body",
        reference_name="right_hand",
        position_in_reference_m=(0.0, 0.0, 0.1),
        orientation_in_reference_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    shifted = reference.model_copy(
        update={"position_in_reference_m": (0.01, 0.0, 0.1)}
    )
    monkeypatch.setattr(
        physical_grasp_module,
        "_contact_friction_transform",
        lambda runtime, object_id: shifted,
    )
    contact = AttachmentContactMetrics(
        contact_count=2,
        normal_force_n=1.0,
        contact_groups=("left_fingerpad", "right_fingerpad"),
    )
    runtime = SimpleNamespace(
        _held_tool_id="part",
        _contact_friction_retention=None,
        object_contact_metrics=lambda object_id: contact,
    )
    monitor = ContactFrictionRetentionMonitor(
        runtime=runtime,
        object_id="part",
        profile=PhysicalGraspProfile(),
        reference_transform=reference,
        last_valid_contact_time_s=0.0,
    )
    runtime._contact_friction_retention = monitor

    with pytest.raises(ToolUseJournalAttachmentBroken) as caught:
        monitor.after_tick(0.01)

    assert "CONTACT_FRICTION_TRANSLATION_SLIP" in caught.value.observation.reasons
    assert runtime._held_tool_id is None


def test_contact_friction_decorator_removes_synthetic_attachment() -> None:
    profile = PhysicalGraspProfile(
        grasp_hold_duration_s=1.25,
        lift_hold_duration_s=2.0,
    )

    result = with_contact_friction_grasp(_artifact(), profile=profile)

    grasp = result.candidates[0].keyframes[1]
    lift = result.candidates[0].keyframes[2]
    assert KeyframeEventType.ATTACH_OBJECT not in grasp.events_after
    assert grasp.events_after == [KeyframeEventType.GRIPPER_CLOSE]
    assert grasp.metadata["grasp_execution_mode"] == "CONTACT_FRICTION"
    assert grasp.metadata["hold_duration_after_s"] == 1.25
    assert grasp.metadata["event_parameters"] == {
        "GRIPPER_CLOSE": {"command": 1.0}
    }
    assert lift.metadata["physical_retention_hold"] is True
    assert lift.metadata["hold_duration_after_s"] == 2.0
    assert result.artifact_id != "keyframes"


def test_contact_friction_rejects_support_blocked_grasp_candidate() -> None:
    artifact = _artifact()
    safe = artifact.candidates[0]
    blocked = safe.model_copy(
        update={
            "strategy_id": "support-blocked",
            "metadata": {
                "support_collision_risk": True,
                "support_collision_reasons": [
                    "SUPPORT_BLOCKS_UNDER_APPROACH"
                ],
            },
        }
    )
    artifact.candidates = [blocked, safe]

    result = with_contact_friction_grasp(
        artifact,
        profile=PhysicalGraspProfile(),
    )

    assert [candidate.strategy_id for candidate in result.candidates] == [
        "strategy"
    ]
    rejected = result.provenance.metadata["rejected_candidates"]
    assert rejected == [
        {
            "strategy_id": "support-blocked",
            "reason": (
                "SUPPORT_COLLISION_RISK:SUPPORT_BLOCKS_UNDER_APPROACH"
            ),
        }
    ]


def test_frozen_acquire_artifact_retargets_current_scene_and_tool() -> None:
    request = MotionPlanRequest(
        request_id="request-ladle",
        provenance=ArtifactProvenance(
            artifact_id="request-ladle-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="fixture",
        ),
        world=WorldSnapshot(
            scene=SceneRef(signature="current-scene"),
            robot_state=RobotState(
                robot_id="robot",
                joint_names=["joint"],
                joint_positions_rad=[0.0],
            ),
            objects={"ladle": {"anchors": {"center": [0.0, 0.0, 0.0]}}},
        ),
        task=MotionTask(
            task_id="pick-ladle",
            subgoal_id="pick-ladle",
            action_type="PICK_TOOL",
            ee="2F",
            tool="ladle",
            target_ids=["ladle"],
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_object_id="ladle",
            ),
        ),
    )

    result = RetargetedAcquireKeyframeProvider(_artifact()).generate(request)

    assert result.scene_signature == "current-scene"
    assert result.subgoal_id == "pick-ladle"
    assert all(
        keyframe.frame_ref == "object:ladle"
        for keyframe in result.candidates[0].keyframes
    )
    grasp = result.candidates[0].keyframes[1]
    assert grasp.metadata["event_target_id"] == "ladle"
    assert result.provenance.metadata["source_object_frame"] == "object:tool"
    assert result.provenance.metadata["target_object_frame"] == "object:ladle"


def _supported_object_request(*, center_z_m: float = 0.805) -> MotionPlanRequest:
    return MotionPlanRequest(
        request_id="request-supported-object",
        provenance=ArtifactProvenance(
            artifact_id="request-supported-object-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="fixture",
        ),
        world=WorldSnapshot(
            scene=SceneRef(signature="scene"),
            robot_state=RobotState(
                robot_id="robot",
                joint_names=["joint"],
                joint_positions_rad=[0.0],
            ),
            objects={
                "tool": {
                    "pose": {
                        "position_m": [-0.4, -0.1, center_z_m],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "dimensions_m": [0.18, 0.18, 0.01],
                    "anchors": {
                        "center": [0.0, 0.0, 0.0],
                        "top": [0.0, 0.0, 0.005],
                        "bottom": [0.0, 0.0, -0.005],
                    },
                }
            },
            obstacles=[
                {
                    "obstacle_id": "table_collision",
                    "aabb_min_m": [-0.65, -0.8, 0.75],
                    "aabb_max_m": [0.65, 0.8, 0.8],
                }
            ],
        ),
        task=MotionTask(
            task_id="pick-supported-object",
            subgoal_id="pick",
            action_type="PICK_TOOL",
            ee="2F",
            tool="tool",
            target_ids=["tool"],
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_object_id="tool",
            ),
        ),
    )


@pytest.mark.parametrize("object_id", ["plate", "box", "new_object_17"])
@pytest.mark.parametrize("dimensions", [[0.18, 0.18, 0.01], [0.08, 0.08, 0.08], [0.04, 0.06, 0.2]])
def test_generic_binding_preserves_candidates_for_any_bbox(object_id, dimensions) -> None:
    request = _supported_object_request()
    record = request.world.objects.pop("tool")
    record["dimensions_m"] = dimensions
    record["pose"]["position_m"][2] = 0.8 + dimensions[2] / 2.0
    request.world.objects[object_id] = record
    request.task.goal.target_object_id = object_id
    request.task.tool = object_id
    request.task.target_ids = [object_id]
    artifact = _artifact()
    for keyframe in artifact.candidates[0].keyframes:
        keyframe.frame_ref = f"object:{object_id}"
    world_before = request.world.model_dump()
    artifact_before = artifact.model_dump()

    result = bind_grasp_geometry(artifact, request)

    assert len(result.candidates) == len(artifact.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.strategy_id == artifact.candidates[0].strategy_id
    for old, new in zip(artifact.candidates[0].keyframes, candidate.keyframes):
        assert old.model_dump(exclude={"metadata"}) == new.model_dump(exclude={"metadata"})
    assert candidate.metadata["under_clearance_m"] == pytest.approx(0.0)
    assert candidate.metadata["object_top_clearance_m"] == pytest.approx(dimensions[2])
    assert candidate.metadata["support_clearance_status"] == "SCREENED"
    assert "edge_height_m" not in candidate.metadata
    assert "grasp_preshape_aperture_m" not in request.task.metadata
    assert request.world.model_dump() == world_before
    assert artifact.model_dump() == artifact_before
    assert result.artifact_id != artifact.artifact_id


def test_generic_annotation_preserves_source_candidate_order_and_directions() -> None:
    request = _supported_object_request()
    artifact = _artifact()
    sideways = artifact.candidates[0].model_copy(deep=True)
    sideways.strategy_id = "source-sideways"
    for keyframe in sideways.keyframes:
        keyframe.approach_axis_xyz = (0.6, 0.8, 0.0)
        keyframe.roll_rad = 0.3
    artifact.candidates.append(sideways)

    result = bind_grasp_geometry(artifact, request)

    assert [candidate.strategy_id for candidate in result.candidates] == [
        "strategy", "source-sideways",
    ]
    for before, after in zip(artifact.candidates, result.candidates):
        assert [frame.model_dump(exclude={"metadata"}) for frame in before.keyframes] == [
            frame.model_dump(exclude={"metadata"}) for frame in after.keyframes
        ]


@pytest.mark.parametrize("override", ["task", "candidate", "keyframe"])
def test_clearance_requirements_use_keyframe_then_candidate_then_task(override) -> None:
    request = _supported_object_request()
    request.task.metadata["grasp_clearance_requirements"] = {
        "required_grasp_clearance_m": 0.01,
    }
    artifact = _artifact()
    if override in {"candidate", "keyframe"}:
        artifact.candidates[0].metadata["required_grasp_clearance_m"] = 0.004
    if override == "keyframe":
        artifact.candidates[0].keyframes[1].metadata["required_grasp_clearance_m"] = 0.02

    result = annotate_support_clearance(artifact, request).candidates[0]

    assert result.metadata["support_clearance_status"] == (
        "SCREENED" if override == "candidate" else "BLOCKED"
    )


def test_contact_clearance_uses_contact_point_not_bbox_top() -> None:
    request = _supported_object_request(center_z_m=0.9)
    request.world.objects["tool"]["dimensions_m"] = [0.04, 0.06, 0.2]
    artifact = _artifact()
    artifact.candidates[0].keyframes[1].metadata.update({
        "contact_center_local_m": [0.0, 0.0, -0.09],
        "required_contact_clearance_m": 0.02,
    })

    candidate = annotate_support_clearance(artifact, request).candidates[0]

    assert candidate.metadata["object_top_clearance_m"] == pytest.approx(0.2)
    assert candidate.metadata["grasp_point_clearance_m"] == pytest.approx(0.1)
    assert candidate.metadata["contact_point_clearance_m"] == pytest.approx(0.01)
    assert candidate.metadata["support_collision_reasons"] == [
        "INSUFFICIENT_CONTACT_POINT_CLEARANCE"
    ]


def test_missing_contact_geometry_remains_unknown() -> None:
    request = _supported_object_request()
    artifact = _artifact()
    artifact.candidates[0].metadata["required_contact_clearance_m"] = 0.002

    candidate = annotate_support_clearance(artifact, request).candidates[0]

    assert candidate.metadata["support_clearance_status"] == "UNKNOWN"
    assert candidate.metadata["support_collision_risk"] is None
    assert candidate.metadata["support_unknown_reasons"] == ["CONTACT_POINT_UNAVAILABLE"]
    assert "contact_point_clearance_m" not in candidate.metadata


@pytest.mark.parametrize("missing", ["support", "dimensions", "orientation", "center"])
def test_missing_support_evidence_is_not_treated_as_safe(missing) -> None:
    request = _supported_object_request()
    if missing == "support":
        request.world.obstacles = []
    elif missing == "dimensions":
        del request.world.objects["tool"]["dimensions_m"]
    elif missing == "orientation":
        request.world.objects["tool"]["pose"]["orientation_xyzw"] = [0, 0, 0, 0]
    else:
        request.world.objects["tool"]["anchors"]["center"] = [0, 0, float("nan")]

    candidate = annotate_support_clearance(_artifact(), request).candidates[0]

    assert candidate.metadata["support_clearance_status"] == "UNKNOWN"
    assert candidate.metadata["support_collision_risk"] is None
    assert candidate.metadata["requires_gripper_envelope_check"] is True


def test_incomplete_explicit_relation_does_not_fabricate_zero_gap() -> None:
    request = _supported_object_request()
    record = request.world.objects["tool"]
    del record["dimensions_m"]
    record["support"] = {"support_id": "support", "surface_z_m": 0.8}

    assert support_clearance_context(record, request, "tool") is None


def test_rotated_bbox_uses_world_vertical_extent() -> None:
    request = _supported_object_request(center_z_m=0.85)
    record = request.world.objects["tool"]
    record["dimensions_m"] = [0.04, 0.1, 0.02]
    record["pose"]["orientation_xyzw"] = [2**-0.5, 0.0, 0.0, 2**-0.5]

    support = support_clearance_context(record, request, "tool")

    assert support is not None
    assert support.under_clearance_m == pytest.approx(0.0)
    assert support.object_top_clearance_m == pytest.approx(0.1)
    assert support.vertical_extent_m == pytest.approx(0.1)


def test_infers_support_from_another_object_without_class_matching() -> None:
    request = _supported_object_request()
    request.world.obstacles = []
    request.world.objects["arbitrary_support"] = {
        "pose": {"position_m": [-0.4, -0.1, 0.75], "orientation_xyzw": [0, 0, 0, 1]},
        "dimensions_m": [0.4, 0.4, 0.1],
    }
    record = request.world.objects["tool"]

    support = support_clearance_context(record, request, "tool")

    assert support is not None
    assert support.support_id == "arbitrary_support"
    assert support.source == "world.objects.obb"
    request.world.objects["arbitrary_support"]["collision_enabled"] = False
    assert support_clearance_context(record, request, "tool") is None


def test_world_frame_grasp_is_screened_and_earlier_failure_is_preserved() -> None:
    request = _supported_object_request()
    artifact = _artifact()
    low = artifact.candidates[0].keyframes[1]
    low.frame_ref = "world"
    low.offset_along_approach_m = 0.79
    high = low.model_copy(update={
        "keyframe_id": "second-grasp",
        "offset_along_approach_m": 0.9,
    })
    artifact.candidates[0].keyframes.insert(2, high)

    candidate = annotate_support_clearance(artifact, request).candidates[0]

    assert candidate.metadata["support_clearance_status"] == "BLOCKED"
    assert candidate.metadata["support_collision_risk"] is True
    per_grasp = candidate.metadata["grasp_support_measurements"]
    assert per_grasp["grasp"]["grasp_point_clearance_m"] == pytest.approx(-0.01)
    assert per_grasp["second-grasp"]["support_clearance_status"] == "SCREENED"


@pytest.mark.parametrize("value", [-0.01, float("nan"), "bad", True])
def test_invalid_clearance_requirements_are_rejected(value) -> None:
    request = _supported_object_request()
    artifact = _artifact()
    artifact.candidates[0].metadata["required_under_clearance_m"] = value

    candidate = annotate_support_clearance(artifact, request).candidates[0]

    assert candidate.metadata["support_clearance_status"] == "BLOCKED"
    assert "INVALID_CLEARANCE_REQUIREMENT:required_under_clearance_m" in (
        candidate.metadata["support_collision_reasons"]
    )


def test_retired_object_specific_provider_is_not_silently_enabled() -> None:
    request = _supported_object_request()
    request.task.metadata["grasp_geometry_provider"] = "TWO_FINGER_THIN_PLATE"

    with pytest.raises(ValueError, match="unsupported grasp geometry provider"):
        bind_grasp_geometry(_artifact(), request)


def test_generic_support_filter_enforces_declared_grasp_clearances() -> None:
    request = _supported_object_request()
    artifact = _artifact()
    artifact.candidates[0].metadata = {
        "uses_underside_contact": True,
        "required_under_clearance_m": 0.004,
        "required_grasp_clearance_m": 0.012,
    }

    result = annotate_support_clearance(artifact, request)

    candidate = result.candidates[0]
    assert candidate.metadata["support_collision_risk"] is True
    assert candidate.metadata["support_collision_reasons"] == [
        "INSUFFICIENT_GRASP_POINT_CLEARANCE",
        "INSUFFICIENT_UNDER_CLEARANCE",
    ]
    assert candidate.metadata["under_clearance_m"] == pytest.approx(0.0)
    assert candidate.metadata["object_top_clearance_m"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="no feasible GRASP/LIFT candidate"):
        with_contact_friction_grasp(
            result,
            profile=PhysicalGraspProfile(),
        )


def test_support_screen_accepts_explicit_support_relation() -> None:
    request = _supported_object_request(center_z_m=1.2)
    request.world.obstacles = []
    request.world.metadata["support_relations"] = {
        "tool": {
            "support_id": "countertop",
            "surface_z_m": 1.195,
            "object_bottom_z_m": 1.195,
        }
    }

    result = annotate_support_clearance(_artifact(), request)

    candidate = result.candidates[0]
    assert candidate.metadata["support_id"] == "countertop"
    assert candidate.metadata["support_detection_source"] == (
        "world.support_relations"
    )


@pytest.mark.parametrize("has_anchors", [True, False])
def test_opposed_contact_binder_places_finger_center_on_feature(has_anchors) -> None:
    request = MotionPlanRequest(
        request_id="request-handle",
        provenance=ArtifactProvenance(
            artifact_id="request-handle-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="fixture",
        ),
        world=WorldSnapshot(
            scene=SceneRef(signature="scene"),
            robot_state=RobotState(
                robot_id="robot",
                joint_names=["joint"],
                joint_positions_rad=[0.0],
            ),
            objects={
                "tool": {
                    "pose": {
                        "position_m": [0.0, 0.0, 0.0],
                        "orientation_xyzw": [
                            0.0,
                            0.0,
                            -2**-0.5,
                            2**-0.5,
                        ],
                    },
                    "anchors": {"center": [0.0, 0.0, 0.0]},
                    "physical_metadata": {
                        "grasp_feature": {
                            "provider": TWO_FINGER_OPPOSED_CONTACT,
                            "contact_center_local_m": [0.001, -0.04, -0.02],
                            "closing_axis_local_xyz": [1.0, 0.0, 0.0],
                            "approach_axis_local_xyz": [0.0, 0.0, 1.0],
                            "contact_span_m": 0.015,
                            "grip_site_to_contact_center_m": 0.01,
                            "approach_distance_m": 0.08,
                            "lift_distance_m": 0.12,
                        }
                    },
                }
            },
        ),
        task=MotionTask(
            task_id="pick-tool",
            subgoal_id="pick",
            action_type="PICK_TOOL",
            ee="2F",
            tool="tool",
            target_ids=["tool"],
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_object_id="tool",
            ),
        ),
    )

    if not has_anchors:
        del request.world.objects["tool"]["anchors"]

    result = TwoFingerOpposedContactBinder().bind(_artifact(), request)

    candidate = result.candidates[0]
    pre, grasp, lift = candidate.keyframes
    assert candidate.metadata["geometry_binder"] == TWO_FINGER_OPPOSED_CONTACT
    assert request.world.objects["tool"]["anchors"]["2f_opposed_contact"] == [
        0.001,
        -0.04,
        -0.03,
    ]
    assert pre.offset_along_approach_m == pytest.approx(0.08)
    assert grasp.offset_along_approach_m == 0.0
    assert lift.offset_along_approach_m == pytest.approx(0.12)
    assert grasp.roll_rad == pytest.approx(np.pi / 2.0)


def test_support_clearance_guides_non_plate_opposed_contact_grasp() -> None:
    request = MotionPlanRequest(
        request_id="request-supported-box",
        provenance=ArtifactProvenance(
            artifact_id="request-supported-box-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="fixture",
        ),
        world=WorldSnapshot(
            scene=SceneRef(signature="scene"),
            robot_state=RobotState(
                robot_id="robot",
                joint_names=["joint"],
                joint_positions_rad=[0.0],
            ),
            objects={
                "tool": {
                    "pose": {
                        "position_m": [0.0, 0.0, 0.9],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "dimensions_m": [0.04, 0.06, 0.2],
                    "anchors": {"center": [0.0, 0.0, 0.0]},
                    "physical_metadata": {
                        "grasp_feature": {
                            "provider": TWO_FINGER_OPPOSED_CONTACT,
                            "contact_center_local_m": [0.0, 0.0, 0.05],
                            "closing_axis_local_xyz": [1.0, 0.0, 0.0],
                            "approach_axis_local_xyz": [0.0, 0.0, 1.0],
                            "contact_span_m": 0.03,
                            "grip_site_to_contact_center_m": 0.01,
                        }
                    },
                }
            },
            obstacles=[
                {
                    "obstacle_id": "table_collision",
                    "aabb_min_m": [-0.5, -0.5, 0.75],
                    "aabb_max_m": [0.5, 0.5, 0.8],
                }
            ],
        ),
        task=MotionTask(
            task_id="pick-supported-box",
            subgoal_id="pick",
            action_type="PICK_TOOL",
            ee="2F",
            tool="tool",
            target_ids=["tool"],
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_object_id="tool",
            ),
        ),
    )

    support = support_clearance_context(
        request.world.objects["tool"], request, "tool"
    )
    assert support is not None
    assert support.support_id == "table_collision"
    assert support.under_clearance_m == pytest.approx(0.0)
    assert support.object_top_clearance_m == pytest.approx(0.2)
    assert support.vertical_extent_m == pytest.approx(0.2)

    generic_result = bind_grasp_geometry(_artifact(), request)
    generic_candidate = generic_result.candidates[0]
    assert generic_candidate.metadata["support_aware"] is True
    assert generic_candidate.metadata["object_top_clearance_m"] == pytest.approx(0.2)

    result = TwoFingerOpposedContactBinder().bind(_artifact(), request)

    candidate = result.candidates[0]
    grasp = candidate.keyframes[1]
    assert candidate.metadata["support_aware"] is True
    assert candidate.metadata["support_blocks_under_grasp"] is True
    assert "support_compatible_approach_modes" not in candidate.metadata
    assert candidate.metadata["support_clearance_status"] == "SCREENED"
    assert candidate.metadata["support_approach_mode"] == "FROM_ABOVE"
    assert candidate.metadata["support_collision_risk"] is False
    assert grasp.metadata["grasp_point_clearance_m"] == pytest.approx(0.14)
    assert grasp.metadata["contact_point_clearance_m"] == pytest.approx(0.15)


def test_contact_friction_release_replaces_detach_without_duplicate_open() -> None:
    artifact = _artifact()
    place = artifact.candidates[0].keyframes[1]
    place.keyframe_type = KeyframeType.PLACE
    place.events_after = [
        KeyframeEventType.DETACH_OBJECT,
        KeyframeEventType.GRIPPER_OPEN,
    ]
    place.metadata = {
        "event_parameters": {
            "DETACH_OBJECT": {"object_id": "tool"},
        }
    }

    result = with_contact_friction_release(artifact)

    place = result.candidates[0].keyframes[1]
    assert place.events_after == [KeyframeEventType.GRIPPER_OPEN]
    assert place.metadata["grasp_execution_mode"] == "CONTACT_FRICTION"
    assert place.metadata["event_parameters"] == {
        "GRIPPER_OPEN": {"command": -1.0}
    }


class _Runtime:
    def __init__(self) -> None:
        raw_data = SimpleNamespace(
            xpos=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            xmat=np.asarray([np.eye(3).reshape(-1), np.eye(3).reshape(-1)]),
        )
        self.env = SimpleNamespace(
            obj_body_id={"tool": 1},
            sim=SimpleNamespace(data=SimpleNamespace(_data=raw_data)),
        )
        self.active_ee = "2F"
        self.grasp_engaged = True
        self.force_targets: list[dict[str, float]] = []
        self.hold_count = 0

    def object_contact_metrics(self, object_id: str):
        assert object_id == "tool"
        return SimpleNamespace(
            contact_count=2,
            normal_force_n=10.0,
            tangential_force_n=0.0,
            total_force_n=10.0,
            contact_groups=("left_finger", "right_finger"),
        )

    def set_finger_gripper_force_target(self, **kwargs):
        self.force_targets.append(kwargs)

    def hold_gripper_position(self) -> None:
        self.hold_count += 1

    def _grasp_reference(self, env):
        assert env is self.env
        return "site", "grip_site", np.zeros(3), np.eye(3)

    def _object_free_joint(self, env, object_id: str):
        assert env is self.env
        assert object_id == "tool"
        return 1, 0, "tool_free_joint"


def test_monitor_requires_formation_lift_and_final_hold() -> None:
    runtime = _Runtime()
    profile = PhysicalGraspProfile(
        validation=GraspValidationProfile(
            minimum_lift_m=0.05,
            contact_loss_grace_s=0.06,
            required_contact_ticks=2,
            contact_freeze_ticks=2,
            final_hold_duration_s=0.10,
            minimum_normal_force_n=0.1,
        )
    )
    monitor = PhysicalGraspMonitor(
        runtime=runtime,
        object_id="tool",
        profile=profile,
        runtime_parameters=RuntimeGraspParameters(
            maximum_grip_force_n=80.0,
            sliding_friction=1.0,
            retention_target_normal_force_n=10.0,
        ),
        initial_position_m=(0.0, 0.0, 0.0),
    )

    monitor.sample(0.00)
    monitor.sample(0.02)
    runtime.env.sim.data._data.xpos[1, 2] = 0.06
    monitor.sample(0.04)
    monitor.sample(0.16)

    validation = monitor.summary()
    assert validation["status"] == "SUCCESS"
    assert validation["contact_formation"]["status"] == "SUCCESS"
    assert validation["grasp_retention"]["status"] == "SUCCESS"
    assert validation["max_lift_m"] == 0.06
    assert runtime.force_targets
    assert runtime.hold_count == 1


@pytest.mark.parametrize("object_id", ["plate", "box", "unknown_object"])
@pytest.mark.parametrize("dimensions", [[181.8, 181.8, 11.1], [50, 50, 50], [20, 20, 300]])
def test_runtime_force_does_not_infer_contact_torque_from_bbox(
    object_id: str, dimensions: list[float]
) -> None:
    metadata = {
        "mass_kg": 0.2,
        "friction": [0.8, 0.005, 0.0001],
        "full_size_mm": dimensions,
    }
    runtime = SimpleNamespace(
        active_ee="2F",
        env=SimpleNamespace(
            robot_spec={"ee_pool": [{"ee_id": "2F", "grip_force_n": 80.0}]},
            get_tool_physical_metadata=lambda object_id: metadata,
        ),
    )

    parameters = _runtime_grasp_parameters(
        runtime,
        object_id,
        PhysicalGraspProfile(),
    )

    assert parameters.retention_model == "WEIGHT_WITH_RETENTION_FLOOR"
    assert parameters.retention_target_normal_force_n == pytest.approx(10.0)


def test_runtime_parameters_do_not_undercut_retention_force_floor() -> None:
    metadata = {
        "mass_kg": 0.026,
        "minimum_retention_force_n": 30.0,
        "retention_control_mode": "POSITION_HOLD",
        "friction": [0.95, 0.3, 0.1],
        "full_size_mm": [63.5, 208.1, 110.2],
    }
    runtime = SimpleNamespace(
        active_ee="2F",
        env=SimpleNamespace(
            robot_spec={"ee_pool": [{"ee_id": "2F", "grip_force_n": 80.0}]},
            get_tool_physical_metadata=lambda object_id: metadata,
        ),
    )

    parameters = _runtime_grasp_parameters(
        runtime,
        "ladle",
        PhysicalGraspProfile(),
    )

    assert parameters.retention_model == "WEIGHT_WITH_RETENTION_FLOOR"
    assert parameters.retention_target_normal_force_n == pytest.approx(30.0)
    assert parameters.retention_control_mode == "POSITION_HOLD"


def test_c1_profile_keys_migrate_to_generic_profile() -> None:
    profile = PhysicalGraspProfile.from_mapping(
        {
            "pick_grasp_hold_s": 1.5,
            "pick_gripper_close_rate": 0.8,
            "pick_gripper_closure_actuator_kp": 45.0,
            "pick_min_lift_m": 0.075,
            "pick_required_contact_ticks": 7,
            "pick_final_hold_s": 3.0,
        }
    )

    assert profile.grasp_hold_duration_s == 1.5
    assert profile.force.close_command == 0.8
    assert profile.force.closure_actuator_kp == 45.0
    assert profile.validation.minimum_lift_m == 0.075
    assert profile.validation.required_contact_ticks == 7
    assert profile.validation.final_hold_duration_s == 3.0
