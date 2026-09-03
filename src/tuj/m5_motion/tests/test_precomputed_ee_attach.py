"""Initial EE attach uses a validated trajectory and never silently replans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from tuj.m5_motion.precomputed_ee_attach import (
    EEAttachPathFailureCode,
    EEAttachPolicy,
    EEAttachTrajectoryEventTemplate,
    EEAttachTrajectorySegmentTemplate,
    EEAttachTrajectoryTemplate,
    PrecomputedEEAttachPlanner,
    PrecomputedEEAttachRegistry,
    PrecomputedEEPathError,
    compute_rack_signature,
    compute_workcell_signature,
)
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    CollisionContext,
    EventType,
    GoalType,
    JointDynamicLimit,
    ModuleName,
    MotionConstraints,
    MotionGoal,
    MotionPlanRequest,
    MotionTask,
    Pose,
    RobotState,
    SceneRef,
    SegmentType,
    TrajectoryWaypoint,
    WorldSnapshot,
)
from tuj.m5_motion.tool_use_journal_planning import (
    ToolUseJournalMotionRequestPlanner,
)


JOINT_NAMES = [
    "robot0_shoulder_pan_joint",
    "robot0_shoulder_lift_joint",
    "robot0_elbow_joint",
    "robot0_wrist_1_joint",
    "robot0_wrist_2_joint",
    "robot0_wrist_3_joint",
]


def _world() -> WorldSnapshot:
    rack = {
        ee: {
            "dock_pose": {
                "frame_id": "world",
                "position_m": [index * 0.1, -0.4, 1.0],
                "orientation_xyzw": [1.0, 0.0, 0.0, 0.0],
            },
            "approach_axis_xyz": [0.0, 0.0, 1.0],
        }
        for index, ee in enumerate(("2F", "3F", "vac"), start=1)
    }
    return WorldSnapshot(
        scene=SceneRef(signature="scene:current"),
        robot_state=RobotState(
            robot_id="ur5e_0",
            joint_names=JOINT_NAMES,
            joint_positions_rad=[0.0] * 6,
            joint_velocities_rad_s=[0.0] * 6,
            eef_pose=Pose(
                frame_id="world",
                position_m=(-0.34, -0.025, 1.1265),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
        ),
        rack=rack,
        metadata={
            "environment_name": "C1_1_LegoSweep",
            "physical_active_ee": None,
            "declared_active_ee": None,
            "source_revision": "fixed-workcell-revision",
            "rack_collision_policy": "PROMOTE_IN_PLANNER_COPY",
        },
    )


def _request(target: str = "2F") -> MotionPlanRequest:
    world = _world()
    return MotionPlanRequest(
        request_id=f"request:bare-to:{target}",
        provenance=ArtifactProvenance(
            artifact_id=f"request-artifact:{target}",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="test",
        ),
        world=world,
        task=MotionTask(
            task_id=f"attach:{target}",
            subgoal_id=f"attach:{target}",
            action_type="EE_ATTACH",
            ee=target,
            goal=MotionGoal(goal_type=GoalType.POSE, target_object_id=target),
            metadata={"from_ee": None, "to_ee": target},
        ),
        constraints=MotionConstraints(
            max_joint_path_step_rad=0.02,
            joint_limits={
                name: JointDynamicLimit(
                    max_velocity_rad_s=1.0,
                    max_acceleration_rad_s2=2.0,
                    max_jerk_rad_s3=10.0,
                )
                for name in JOINT_NAMES
            },
        ),
    )


def _contexts(target: str) -> dict[str, CollisionContext]:
    return {
        "bare-flange": CollisionContext(
            context_id="bare-flange",
            scene_state_id="bare-flange",
            collision_model_version="bare-model-v1",
        ),
        f"bare-flange-dock-contact:{target}": CollisionContext(
            context_id=f"bare-flange-dock-contact:{target}",
            scene_state_id="bare-flange",
            allowed_collision_pairs=[("qc_master", target)],
            collision_model_version="bare-model-v1",
        ),
        f"ee-attached:{target}": CollisionContext(
            context_id=f"ee-attached:{target}",
            scene_state_id=f"ee-attached:{target}",
            active_ee=target,
            collision_model_version=f"attached-{target}-model-v1",
        ),
        f"ee-attached-dock-contact:{target}": CollisionContext(
            context_id=f"ee-attached-dock-contact:{target}",
            scene_state_id=f"ee-attached:{target}",
            active_ee=target,
            allowed_collision_pairs=[(target, f"rack_support:{target}")],
            collision_model_version=f"attached-{target}-model-v1",
        ),
    }


def _waypoint(time_s: float, position: float) -> TrajectoryWaypoint:
    return TrajectoryWaypoint(
        time_from_start_s=time_s,
        joint_positions_rad=[position] * 6,
        joint_velocities_rad_s=[0.0] * 6,
        joint_accelerations_rad_s2=[0.0] * 6,
    )


def _template(
    request: MotionPlanRequest,
    contexts: dict[str, CollisionContext],
    *,
    events: bool = True,
) -> EEAttachTrajectoryTemplate:
    target = request.task.ee
    positions = (0.0, 0.1, 0.2, 0.3, 0.2)
    before = (
        "bare-flange",
        "bare-flange",
        f"bare-flange-dock-contact:{target}",
        f"ee-attached-dock-contact:{target}",
    )
    after = (
        "bare-flange",
        "bare-flange",
        f"ee-attached-dock-contact:{target}",
        f"ee-attached:{target}",
    )
    return EEAttachTrajectoryTemplate(
        trajectory_id=f"ur5e-bare-to-{target}-test-v1",
        environment_name="C1_1_LegoSweep",
        robot_model="UR5e",
        target_active_ee=target,
        joint_names=JOINT_NAMES,
        start_joint_positions_rad=[0.0] * 6,
        workcell_signature=compute_workcell_signature(request.world, contexts),
        rack_signature=compute_rack_signature(request.world),
        collision_model_versions={
            context_id: context.collision_model_version
            for context_id, context in contexts.items()
        },
        segments=[
            EEAttachTrajectorySegmentTemplate(
                segment_id=segment_id,
                segment_type=(
                    SegmentType.RETREAT if index == 3 else SegmentType.EE_DOCK
                ),
                collision_context_before=before[index],
                collision_context_after=after[index],
                waypoints=[
                    _waypoint(float(index), positions[index]),
                    _waypoint(float(index + 1), positions[index + 1]),
                ],
            )
            for index, segment_id in enumerate(
                (
                    "home-to-staging",
                    "staging-to-pre-dock",
                    "pre-dock-to-dock",
                    "dock-to-retreat",
                )
            )
        ],
        events=(
            [
                EEAttachTrajectoryEventTemplate(
                    time_from_start_s=3.0,
                    event_type=EventType.TOOL_LOCK,
                    target_id=target,
                ),
                EEAttachTrajectoryEventTemplate(
                    time_from_start_s=3.0,
                    event_type=EventType.VERIFY_TOOL_LOCK,
                    target_id=target,
                ),
            ]
            if events
            else []
        ),
    )


def _write_template(root: Path, template: EEAttachTrajectoryTemplate) -> Path:
    destination = (
        root
        / template.environment_name
        / f"bare_to_{template.target_active_ee}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.model_dump_json(indent=2), encoding="utf-8")
    return destination


@dataclass
class _CollisionResult:
    valid: bool
    failure_code: str | None = None
    detail: str = ""
    min_clearance_m: float | None = 0.1


class _CollisionChecker:
    def __init__(self, *, fail_at: float | None = None) -> None:
        self.fail_at = fail_at
        self.checked: list[tuple[float, ...]] = []

    def check(self, joint_config, keyframe=None, *, context=None, context_id=None):
        values = tuple(float(value) for value in joint_config)
        self.checked.append(values)
        if self.fail_at is not None and abs(values[0] - self.fail_at) < 1e-9:
            return _CollisionResult(
                valid=False,
                failure_code="TEST_COLLISION",
                detail="collision between stored waypoints",
            )
        return _CollisionResult(valid=True)


def _planner(
    root: Path,
    *,
    checker: _CollisionChecker | None = None,
) -> tuple[PrecomputedEEAttachPlanner, _CollisionChecker]:
    selected_checker = checker or _CollisionChecker()
    return (
        PrecomputedEEAttachPlanner(
            PrecomputedEEAttachRegistry(root),
            joint_position_limits_rad=[(-6.3, 6.3)] * 6,
            log=lambda _: None,
        ),
        selected_checker,
    )


@pytest.mark.parametrize(
    ("stored_target", "lookup_target"),
    [
        ("2F", "2f"),
        ("3F", "3F"),
        ("vac", "vac"),
        ("vac", "Vac"),
        ("vac", "vacuum"),
    ],
)
def test_registry_lookup_and_alias_normalization(
    tmp_path: Path,
    stored_target: str,
    lookup_target: str,
) -> None:
    request = _request(stored_target)
    template = _template(request, _contexts(stored_target))
    _write_template(tmp_path, template)

    loaded = PrecomputedEEAttachRegistry(tmp_path).load(
        "C1_1_LegoSweep", lookup_target
    )

    assert loaded.trajectory_id == template.trajectory_id
    assert loaded.target_active_ee == stored_target


def test_registry_rejects_unknown_target(tmp_path: Path) -> None:
    with pytest.raises(PrecomputedEEPathError) as captured:
        PrecomputedEEAttachRegistry(tmp_path).load("C1_1_LegoSweep", "welder")
    assert (
        captured.value.failure_code
        is EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_NOT_FOUND
    )


def test_start_joint_mismatch_is_fail_closed(tmp_path: Path) -> None:
    request = _request("2F")
    contexts = _contexts("2F")
    _write_template(tmp_path, _template(request, contexts))
    request.world.robot_state.joint_positions_rad[0] = 0.02
    planner, checker = _planner(tmp_path)

    with pytest.raises(PrecomputedEEPathError) as captured:
        planner.plan(
            request,
            collision_contexts=contexts,
            collision_checker=checker,
        )
    assert captured.value.failure_code is EEAttachPathFailureCode.START_STATE_MISMATCH


def test_workcell_signature_mismatch_is_fail_closed(tmp_path: Path) -> None:
    request = _request("2F")
    contexts = _contexts("2F")
    _write_template(tmp_path, _template(request, contexts))
    request.world.rack["2F"]["dock_pose"]["position_m"][0] += 0.001
    planner, checker = _planner(tmp_path)

    with pytest.raises(PrecomputedEEPathError) as captured:
        planner.plan(
            request,
            collision_contexts=contexts,
            collision_checker=checker,
        )
    assert (
        captured.value.failure_code
        is EEAttachPathFailureCode.WORKCELL_SIGNATURE_MISMATCH
    )


def test_robot_base_pose_mismatch_is_fail_closed(tmp_path: Path) -> None:
    request = _request("2F")
    contexts = _contexts("2F")
    _write_template(tmp_path, _template(request, contexts))
    assert request.world.robot_state.eef_pose is not None
    request.world.robot_state.eef_pose.position_m = (-0.33, -0.025, 1.1265)
    planner, checker = _planner(tmp_path)

    with pytest.raises(PrecomputedEEPathError) as captured:
        planner.plan(
            request,
            collision_contexts=contexts,
            collision_checker=checker,
        )
    assert (
        captured.value.failure_code
        is EEAttachPathFailureCode.WORKCELL_SIGNATURE_MISMATCH
    )


def test_collision_model_version_mismatch_is_stale(tmp_path: Path) -> None:
    request = _request("2F")
    contexts = _contexts("2F")
    _write_template(tmp_path, _template(request, contexts))
    contexts["ee-attached:2F"] = contexts["ee-attached:2F"].model_copy(
        update={"collision_model_version": "attached-2F-model-v2"}
    )
    planner, checker = _planner(tmp_path)

    with pytest.raises(PrecomputedEEPathError) as captured:
        planner.plan(
            request,
            collision_contexts=contexts,
            collision_checker=checker,
        )
    assert (
        captured.value.failure_code
        is EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE
    )


def test_interpolated_path_collision_is_detected(tmp_path: Path) -> None:
    request = _request("2F")
    contexts = _contexts("2F")
    _write_template(tmp_path, _template(request, contexts))
    planner, checker = _planner(tmp_path, checker=_CollisionChecker(fail_at=0.06))

    with pytest.raises(PrecomputedEEPathError) as captured:
        planner.plan(
            request,
            collision_contexts=contexts,
            collision_checker=checker,
        )
    assert (
        captured.value.failure_code
        is EEAttachPathFailureCode.PRECOMPUTED_PATH_COLLISION
    )
    assert any(abs(state[0] - 0.06) < 1e-9 for state in checker.checked)


def test_lock_and_verify_events_are_required(tmp_path: Path) -> None:
    request = _request("2F")
    contexts = _contexts("2F")
    _write_template(tmp_path, _template(request, contexts, events=False))
    planner, checker = _planner(tmp_path)

    with pytest.raises(PrecomputedEEPathError) as captured:
        planner.plan(
            request,
            collision_contexts=contexts,
            collision_checker=checker,
        )
    assert captured.value.failure_code is EEAttachPathFailureCode.LOCK_EVENT_MISSING


def test_precomputed_hit_does_not_call_dynamic_pipeline(tmp_path: Path) -> None:
    request = _request("2F")
    contexts = _contexts("2F")
    _write_template(tmp_path, _template(request, contexts))
    precomputed, checker = _planner(tmp_path)

    class DynamicPipelineMustNotRun:
        def plan(self, *args, **kwargs):
            raise AssertionError("dynamic planner was called")

    class Factory:
        compiler = SimpleNamespace(environment_name="C1_1_LegoSweep")

        def prepare_precomputed_ee_attach(self, incoming):
            assert incoming is request
            return contexts, checker

    planner = ToolUseJournalMotionRequestPlanner(
        DynamicPipelineMustNotRun(),  # type: ignore[arg-type]
        Factory(),  # type: ignore[arg-type]
        precomputed_ee_attach_planner=precomputed,
        ee_attach_policy=EEAttachPolicy.PRECOMPUTED_REQUIRED,
        log=lambda _: None,
    )

    plan = planner(request)

    assert plan.metadata["source"] == "precomputed"
    assert plan.metadata["dynamic_planner_invoked"] is False
    assert "edge_evaluations" not in plan.metadata
    assert [event.event_type for event in plan.events] == [
        EventType.TOOL_LOCK,
        EventType.VERIFY_TOOL_LOCK,
    ]
    assert plan.segments[-1].collision_context_after.active_ee == "2F"


def test_dynamic_fallback_requires_explicit_policy(tmp_path: Path) -> None:
    request = _request("2F")
    calls: list[str] = []

    class DynamicPipeline:
        def plan(self, incoming, *, collision_context_factory):
            calls.append(incoming.request_id)
            return "dynamic-result"

    class Factory:
        compiler = SimpleNamespace(environment_name="C1_1_LegoSweep")

        def prepare_precomputed_ee_attach(self, incoming):
            return _contexts("2F"), _CollisionChecker()

    missing = PrecomputedEEAttachPlanner(
        PrecomputedEEAttachRegistry(tmp_path),
        log=lambda _: None,
    )
    required = ToolUseJournalMotionRequestPlanner(
        DynamicPipeline(),  # type: ignore[arg-type]
        Factory(),  # type: ignore[arg-type]
        precomputed_ee_attach_planner=missing,
        ee_attach_policy=EEAttachPolicy.PRECOMPUTED_REQUIRED,
        log=lambda _: None,
    )
    with pytest.raises(PrecomputedEEPathError):
        required(request)
    assert calls == []

    fallback = ToolUseJournalMotionRequestPlanner(
        DynamicPipeline(),  # type: ignore[arg-type]
        Factory(),  # type: ignore[arg-type]
        precomputed_ee_attach_planner=missing,
        ee_attach_policy=EEAttachPolicy.PRECOMPUTED_OR_PLAN,
        log=lambda _: None,
    )
    assert fallback(request) == "dynamic-result"
    assert calls == [request.request_id]
