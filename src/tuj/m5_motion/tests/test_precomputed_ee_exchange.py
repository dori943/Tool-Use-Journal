"""EE return and re-attach are composed without dynamic planning."""

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
from tuj.m5_motion.precomputed_ee_exchange import (
    PrecomputedEEExchangePlanner,
    PrecomputedEEReturnRegistry,
    derive_return_template_from_attach,
)
from tuj.m5_motion.ee_exchange_entry import (
    EEExchangeEntryFailureCode,
    EEExchangeEntryPlanner,
    EEExchangeEntryPlanningError,
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
from tuj.m5_motion.tool_use_journal_planning import ToolUseJournalMotionRequestPlanner


JOINT_NAMES = [
    "robot0_shoulder_pan_joint",
    "robot0_shoulder_lift_joint",
    "robot0_elbow_joint",
    "robot0_wrist_1_joint",
    "robot0_wrist_2_joint",
    "robot0_wrist_3_joint",
]


def _world(*, active_ee: str | None, position: float) -> WorldSnapshot:
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
            joint_positions_rad=[position] * 6,
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
            "physical_active_ee": active_ee,
            "declared_active_ee": active_ee,
            "source_revision": "fixed-workcell-revision",
            "rack_collision_policy": "PROMOTE_IN_PLANNER_COPY",
        },
    )


def _contexts(source: str, target: str) -> dict[str, CollisionContext]:
    contexts = {
        "bare-flange": CollisionContext(
            context_id="bare-flange",
            scene_state_id="bare-flange",
            collision_model_version="bare-model-v1",
        ),
        f"ee-attached:{source}": CollisionContext(
            context_id=f"ee-attached:{source}",
            scene_state_id=f"ee-attached:{source}",
            active_ee=source,
            collision_model_version=f"attached-{source}-model-v1",
        ),
        f"ee-attached-dock-contact:{source}": CollisionContext(
            context_id=f"ee-attached-dock-contact:{source}",
            scene_state_id=f"ee-attached:{source}",
            active_ee=source,
            allowed_collision_pairs=[(source, f"rack_support:{source}")],
            collision_model_version=f"attached-{source}-model-v1",
        ),
        f"bare-flange-dock-contact:{target}": CollisionContext(
            context_id=f"bare-flange-dock-contact:{target}",
            scene_state_id="bare-flange",
            allowed_collision_pairs=[("qc_master", target)],
            collision_model_version="bare-model-v1",
        ),
        f"ee-attached-dock-contact:{target}": CollisionContext(
            context_id=f"ee-attached-dock-contact:{target}",
            scene_state_id=f"ee-attached:{target}",
            active_ee=target,
            allowed_collision_pairs=[(target, f"rack_support:{target}")],
            collision_model_version=f"attached-{target}-model-v1",
        ),
        f"ee-attached:{target}": CollisionContext(
            context_id=f"ee-attached:{target}",
            scene_state_id=f"ee-attached:{target}",
            active_ee=target,
            collision_model_version=f"attached-{target}-model-v1",
        ),
    }
    contexts[f"bare-flange-dock-contact:{source}"] = CollisionContext(
        context_id=f"bare-flange-dock-contact:{source}",
        scene_state_id="bare-flange",
        allowed_collision_pairs=[("qc_master", source)],
        collision_model_version="bare-model-v1",
    )
    return contexts


def _waypoint(time_s: float, position: float) -> TrajectoryWaypoint:
    return TrajectoryWaypoint(
        time_from_start_s=time_s,
        joint_positions_rad=[position] * 6,
        joint_velocities_rad_s=[0.0] * 6,
        joint_accelerations_rad_s2=[0.0] * 6,
        eef_pose=Pose(
            frame_id="world",
            position_m=(-0.34, -0.025, 1.1265),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
    )


def _attach_template(
    target: str, world: WorldSnapshot, contexts: dict[str, CollisionContext]
) -> EEAttachTrajectoryTemplate:
    selected = {
        key: value
        for key, value in contexts.items()
        if key == "bare-flange" or key.endswith(f":{target}")
    }
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
        workcell_signature=compute_workcell_signature(world, selected),
        rack_signature=compute_rack_signature(world),
        collision_model_versions={
            key: value.collision_model_version for key, value in selected.items()
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
        events=[
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
        ],
    )


def _request(source: str = "2F", target: str = "3F") -> MotionPlanRequest:
    world = _world(active_ee=source, position=0.2)
    return MotionPlanRequest(
        request_id=f"request:{source}-to-{target}",
        provenance=ArtifactProvenance(
            artifact_id=f"request-artifact:{source}-to-{target}",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="test",
        ),
        world=world,
        task=MotionTask(
            task_id=f"exchange:{source}-to-{target}",
            subgoal_id=f"exchange:{source}-to-{target}",
            action_type="EE_EXCHANGE",
            ee=target,
            target_ids=[source, target],
            goal=MotionGoal(goal_type=GoalType.POSE, target_object_id=target),
            metadata={"from_ee": source, "to_ee": target},
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


def _write_attach(root: Path, template: EEAttachTrajectoryTemplate) -> None:
    destination = root / template.environment_name / f"bare_to_{template.target_active_ee}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.model_dump_json(indent=2), encoding="utf-8")


@dataclass
class _CollisionResult:
    valid: bool = True
    min_clearance_m: float = 0.1
    failure_code: str | None = None
    detail: str = ""


class _CollisionChecker:
    def check(self, joint_config, keyframe=None, *, context=None, context_id=None):
        return _CollisionResult()

    def __call__(self, joint_config, keyframe):
        return self.check(joint_config, keyframe).valid

    def final_segment_validator(self, waypoints, context):
        return all(
            self.check(item.joint_positions_rad, context=context).valid
            for item in waypoints
        )


def _setup(tmp_path: Path):
    request = _request()
    contexts = _contexts("2F", "3F")
    bare_world = _world(active_ee=None, position=0.0)
    attach_source = _attach_template("2F", bare_world, contexts)
    attach_target = _attach_template("3F", bare_world, contexts)
    returned = derive_return_template_from_attach(
        attach_source,
        request.world,
        contexts,
        trajectory_id="ur5e-2F-to-bare-test-v1",
    )
    destination = tmp_path / "C1_1_LegoSweep" / "2F_to_bare.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(returned.model_dump_json(indent=2), encoding="utf-8")
    _write_attach(tmp_path, attach_target)
    attach_planner = PrecomputedEEAttachPlanner(
        PrecomputedEEAttachRegistry(tmp_path),
        joint_position_limits_rad=[(-6.3, 6.3)] * 6,
        log=lambda _: None,
    )
    exchange = PrecomputedEEExchangePlanner(
        PrecomputedEEReturnRegistry(tmp_path),
        attach_planner,
        log=lambda _: None,
    )
    return request, contexts, returned, exchange


def _entry_request(request: MotionPlanRequest, *, position: float) -> MotionPlanRequest:
    source = str(request.task.metadata["from_ee"])
    target = str(request.task.metadata["to_ee"])
    entry = request.model_copy(deep=True)
    entry.request_id = f"request:{source}-exchange-entry"
    entry.provenance.artifact_id = f"request-artifact:{source}-exchange-entry"
    entry.world.robot_state.joint_positions_rad = [position] * 6
    entry.task = MotionTask(
        task_id=f"exchange-entry:{source}",
        subgoal_id=f"exchange-entry:{source}",
        action_type="EE_EXCHANGE_ENTRY",
        ee=source,
        target_ids=[source],
        goal=MotionGoal(goal_type=GoalType.POSE),
        metadata={
            "entry_ee": source,
            "from_ee": source,
            "next_ee": target,
        },
    )
    return entry


def test_exchange_entry_moves_to_stored_return_start_without_generic_pipeline(
    tmp_path: Path,
) -> None:
    request, contexts, returned, _ = _setup(tmp_path)
    entry_request = _entry_request(request, position=0.0)
    checker = _CollisionChecker()
    entry_planner = EEExchangeEntryPlanner(
        PrecomputedEEReturnRegistry(tmp_path),
        joint_position_limits_rad=[(-6.3, 6.3)] * 6,
        log=lambda _: None,
    )

    class DynamicPipelineMustNotRun:
        def plan(self, *args, **kwargs):
            raise AssertionError("generic dynamic pipeline was called")

    class Factory:
        compiler = SimpleNamespace(environment_name="C1_1_LegoSweep")

        def prepare_ee_exchange_entry(self, incoming):
            assert incoming is entry_request
            return contexts, checker

    planner = ToolUseJournalMotionRequestPlanner(
        DynamicPipelineMustNotRun(),  # type: ignore[arg-type]
        Factory(),  # type: ignore[arg-type]
        ee_exchange_entry_planner=entry_planner,
        log=lambda _: None,
    )

    plan = planner(entry_request)

    assert plan.metadata["selection_policy"] == "STORED_EE_EXCHANGE_ENTRY_REQUIRED"
    assert plan.metadata["planner"] == "DIRECT_JOINT"
    assert plan.metadata["target_active_ee"] == "2F"
    assert plan.metadata["dynamic_planner_invoked"] is True
    assert plan.expected_final_state.joint_positions_rad == pytest.approx(
        returned.start_joint_positions_rad
    )
    assert plan.segments[0].collision_context_before.active_ee == "2F"
    assert plan.events == []


def test_exchange_entry_uses_rrt_when_direct_joint_path_is_blocked(
    tmp_path: Path,
) -> None:
    request, contexts, returned, _ = _setup(tmp_path)
    entry_request = _entry_request(request, position=0.0)

    class DiagonalBarrier(_CollisionChecker):
        def check(self, joint_config, keyframe=None, *, context=None, context_id=None):
            blocked = (
                0.07 < joint_config[0] < 0.13
                and 0.07 < joint_config[1] < 0.13
            )
            return _CollisionResult(valid=not blocked)

    plan = EEExchangeEntryPlanner(
        PrecomputedEEReturnRegistry(tmp_path),
        joint_position_limits_rad=[(-1.0, 1.0)] * 6,
        log=lambda _: None,
    ).plan(
        entry_request,
        collision_contexts=contexts,
        collision_checker=DiagonalBarrier(),
        template=returned,
    )

    assert plan.metadata["planner"] == "RRT_CONNECT"
    assert plan.expected_final_state.joint_positions_rad == pytest.approx([0.2] * 6)


def test_exchange_entry_fails_closed_when_no_collision_free_path_exists(
    tmp_path: Path,
) -> None:
    request, contexts, returned, _ = _setup(tmp_path)
    entry_request = _entry_request(request, position=0.0)
    entry_request.options.rrt_max_iterations = 5
    entry_request.options.allowed_planning_time_s = 0.1

    class TargetBlocked(_CollisionChecker):
        def check(self, joint_config, keyframe=None, *, context=None, context_id=None):
            return _CollisionResult(valid=joint_config[0] < 0.15)

    with pytest.raises(EEExchangeEntryPlanningError) as captured:
        EEExchangeEntryPlanner(
            PrecomputedEEReturnRegistry(tmp_path),
            joint_position_limits_rad=[(-1.0, 1.0)] * 6,
            log=lambda _: None,
        ).plan(
            entry_request,
            collision_contexts=contexts,
            collision_checker=TargetBlocked(),
            template=returned,
        )

    assert captured.value.failure_code is EEExchangeEntryFailureCode.PATH_NOT_FOUND


def test_exchange_composes_return_and_attach_without_dynamic_planner(tmp_path: Path) -> None:
    request, contexts, _, exchange = _setup(tmp_path)
    checker = _CollisionChecker()

    class DynamicPipelineMustNotRun:
        def plan(self, *args, **kwargs):
            raise AssertionError("dynamic planner was called")

    class Factory:
        compiler = SimpleNamespace(environment_name="C1_1_LegoSweep")

        def prepare_precomputed_ee_exchange(self, incoming):
            assert incoming is request
            return contexts, checker

    planner = ToolUseJournalMotionRequestPlanner(
        DynamicPipelineMustNotRun(),  # type: ignore[arg-type]
        Factory(),  # type: ignore[arg-type]
        precomputed_ee_exchange_planner=exchange,
        ee_attach_policy=EEAttachPolicy.PRECOMPUTED_REQUIRED,
        log=lambda _: None,
    )

    plan = planner(request)

    assert plan.metadata["selection_policy"] == "PRECOMPUTED_EE_EXCHANGE"
    assert plan.metadata["dynamic_planner_invoked"] is False
    assert plan.metadata["target_active_ee"] == "3F"
    assert len(plan.segments) == 8
    assert [event.event_type for event in plan.events] == [
        EventType.TOOL_UNLOCK,
        EventType.VERIFY_TOOL_RELEASE,
        EventType.TOOL_LOCK,
        EventType.VERIFY_TOOL_LOCK,
    ]
    assert plan.segments[3].collision_context_after.active_ee is None
    assert plan.segments[-1].collision_context_after.active_ee == "3F"


def test_return_start_mismatch_is_fail_closed(tmp_path: Path) -> None:
    request, contexts, _, exchange = _setup(tmp_path)
    request.world.robot_state.joint_positions_rad[0] += 0.02

    with pytest.raises(PrecomputedEEPathError) as captured:
        exchange.plan(
            request,
            collision_contexts=contexts,
            collision_checker=_CollisionChecker(),
        )
    assert captured.value.failure_code is EEAttachPathFailureCode.START_STATE_MISMATCH


def test_return_static_signature_ignores_allowed_controller_tracking_error(
    tmp_path: Path,
) -> None:
    request, contexts, _, exchange = _setup(tmp_path)
    request.world.robot_state.joint_positions_rad[0] += 0.005
    pose = request.world.robot_state.eef_pose
    assert pose is not None
    request.world.robot_state.eef_pose = pose.model_copy(
        update={"position_m": (pose.position_m[0] + 0.002, *pose.position_m[1:])}
    )

    plan = exchange.plan(
        request,
        collision_contexts=contexts,
        collision_checker=_CollisionChecker(),
    )

    assert plan.metadata["workcell_signature_validation"] == "PASS"
    assert plan.metadata["dynamic_planner_invoked"] is False


def test_return_release_events_are_required(tmp_path: Path) -> None:
    request, contexts, returned, exchange = _setup(tmp_path)
    returned.events = []
    path = tmp_path / "C1_1_LegoSweep" / "2F_to_bare.json"
    path.write_text(returned.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(PrecomputedEEPathError) as captured:
        exchange.plan(
            request,
            collision_contexts=contexts,
            collision_checker=_CollisionChecker(),
        )
    assert captured.value.failure_code is EEAttachPathFailureCode.RELEASE_EVENT_MISSING


def test_return_attach_joint_seam_must_be_exact(tmp_path: Path) -> None:
    request, contexts, returned, exchange = _setup(tmp_path)
    returned.segments[-1].waypoints[-1].joint_positions_rad[0] = 0.001
    path = tmp_path / "C1_1_LegoSweep" / "2F_to_bare.json"
    path.write_text(returned.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(PrecomputedEEPathError) as captured:
        exchange.plan(
            request,
            collision_contexts=contexts,
            collision_checker=_CollisionChecker(),
        )
    assert (
        captured.value.failure_code
        is EEAttachPathFailureCode.TRANSITION_SEAM_MISMATCH
    )


def test_missing_return_never_silently_falls_back(tmp_path: Path) -> None:
    request = _request()
    calls: list[str] = []

    class DynamicPipeline:
        def plan(self, incoming, *, collision_context_factory):
            calls.append(incoming.request_id)
            return "dynamic"

    attach = PrecomputedEEAttachPlanner(
        PrecomputedEEAttachRegistry(tmp_path), log=lambda _: None
    )
    exchange = PrecomputedEEExchangePlanner(
        PrecomputedEEReturnRegistry(tmp_path), attach, log=lambda _: None
    )
    planner = ToolUseJournalMotionRequestPlanner(
        DynamicPipeline(),  # type: ignore[arg-type]
        SimpleNamespace(
            compiler=SimpleNamespace(environment_name="C1_1_LegoSweep")
        ),  # type: ignore[arg-type]
        precomputed_ee_exchange_planner=exchange,
        ee_attach_policy=EEAttachPolicy.PRECOMPUTED_REQUIRED,
        log=lambda _: None,
    )

    with pytest.raises(PrecomputedEEPathError):
        planner(request)
    assert calls == []
