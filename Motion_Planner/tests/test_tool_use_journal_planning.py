from __future__ import annotations

from types import SimpleNamespace

from task_planner.models import GraspSpec

from motion_planner.schema import (
    ArtifactProvenance,
    AttachedObjectTransform,
    GoalType,
    JointDynamicLimit,
    KeyframeEventType,
    KeyframePlanArtifact,
    KeyframePlanCandidate,
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
    StrategyGenerationProvenance,
    StrategyGeneratorKind,
    WorldSnapshot,
)
from motion_planner.tool_use_journal_planning import (
    ToolUseJournalCollisionContextFactory,
    WorkcellMotionRequestRouter,
    attached_object_transform_from_state,
)


class _Registry:
    def __call__(self, joint_config, keyframe):
        del joint_config, keyframe
        return True

    @staticmethod
    def final_segment_validator(waypoints, context):
        return bool(waypoints) and context is not None


class _Compiler:
    environment_name = "C2_1_ObjectSorting"

    def __init__(self) -> None:
        self.calls = []

    @staticmethod
    def model_version_for(active_ee):
        return f"model:{active_ee}"

    def build_collision_registry(self, contexts, **kwargs):
        self.calls.append((dict(contexts), kwargs))
        return _Registry()


def _provenance(artifact_id: str, artifact_type: str) -> ArtifactProvenance:
    return ArtifactProvenance(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        produced_by=ModuleName.MOTION_PLANNER,
        invocation_id="test",
    )


def _world(*, attached: AttachedObjectTransform | None = None) -> WorldSnapshot:
    return WorldSnapshot(
        scene=SceneRef(signature="scene-1"),
        robot_state=RobotState(
            robot_id="ur5e",
            joint_names=["j1", "j2"],
            joint_positions_rad=[0.0, 0.0],
            attached_object_id=(attached.object_id if attached else None),
        ),
        objects={
            "bottle": {
                "object_id": "bottle",
                "free_joint_name": "bottle_free",
                "pose": {
                    "position_m": [0.4, 0.0, 0.2],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "dimensions_m": [0.05, 0.05, 0.2],
            },
            "other": {
                "object_id": "other",
                "free_joint_name": "other_free",
                "pose": {
                    "frame_id": "world",
                    "position_m": [0.6, 0.0, 0.1],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
        metadata={
            "environment_name": "C2_1_ObjectSorting",
            "physical_active_ee": "2F",
            "attached_object_transforms": (
                {attached.object_id: attached.model_dump(mode="json")}
                if attached
                else {}
            ),
        },
    )


def _constraints() -> MotionConstraints:
    return MotionConstraints(
        joint_limits={
            name: JointDynamicLimit(
                max_velocity_rad_s=1.0,
                max_acceleration_rad_s2=2.0,
            )
            for name in ("j1", "j2")
        }
    )


def _request(goal: MotionGoal, *, attached=None) -> MotionPlanRequest:
    grasp = None
    if goal.goal_type is GoalType.PICK:
        grasp = GraspSpec(
            grasp_id="grasp-1",
            owner_kind="object",
            owner_id="bottle",
        )
    return MotionPlanRequest(
        request_id=f"request-{goal.goal_type.value.lower()}",
        provenance=ArtifactProvenance(
            artifact_id="request-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="task-planner",
        ),
        world=_world(attached=attached),
        task=MotionTask(
            task_id="task-1",
            subgoal_id="subgoal-1",
            action_type=goal.goal_type.value,
            ee="2F",
            target_ids=["bottle"],
            grasp=grasp,
            goal=goal,
            allowed_touch_objects=["table_collision"],
        ),
        constraints=_constraints(),
    )


def _keyframe(
    identifier: str,
    kind: KeyframeType,
    *,
    events: tuple[KeyframeEventType, ...] = (),
) -> RelativeKeyframeSpec:
    return RelativeKeyframeSpec(
        keyframe_id=identifier,
        keyframe_type=kind,
        frame_ref="object:bottle",
        anchor="center",
        approach_axis_xyz=(0.0, 0.0, 1.0),
        offset_along_approach_m=(0.1 if kind is KeyframeType.PRE_GRASP else 0.0),
        planner=KeyframePlannerType.CARTESIAN,
        events_after=list(events),
        metadata={"event_target_id": "bottle"} if events else {},
    )


def _artifact(keyframes) -> KeyframePlanArtifact:
    return KeyframePlanArtifact(
        artifact_id="source-keyframes",
        provenance=_provenance("source-keyframe-artifact", "KeyframePlanArtifact"),
        scene_signature="scene-1",
        subgoal_id="subgoal-1",
        candidates=[
            KeyframePlanCandidate(
                strategy_id="strategy-1",
                keyframes=list(keyframes),
                provenance=StrategyGenerationProvenance(
                    generator_kind=StrategyGeneratorKind.TEMPLATE,
                    generator_id="test",
                    input_hash="input",
                ),
            )
        ],
    )


def _factory(compiler=None):
    return ToolUseJournalCollisionContextFactory(
        compiler or _Compiler(),
        attachment_reference_name="robot0_right_hand",
    )


def test_pick_binds_contact_then_candidate_specific_attachment_context() -> None:
    request = _request(
        MotionGoal(goal_type=GoalType.PICK, target_object_id="bottle")
    )
    request.task.metadata["support_collision_selectors"] = ["table*"]
    source = _artifact(
        (
            _keyframe("pre", KeyframeType.PRE_GRASP),
            _keyframe(
                "grasp",
                KeyframeType.GRASP,
                events=(
                    KeyframeEventType.GRIPPER_CLOSE,
                    KeyframeEventType.ATTACH_OBJECT,
                ),
            ),
            _keyframe("lift", KeyframeType.LIFT),
            _keyframe("transfer", KeyframeType.TRANSFER),
        )
    )

    setup = _factory().prepare(request, source)
    pre, grasp, lift, transfer = setup.keyframe_artifact.candidates[0].keyframes

    assert source.candidates[0].keyframes[0].collision_context_id is None
    assert pre.collision_context_id == setup.initial_collision_context_id
    assert grasp.collision_context_id.startswith("grasp-contact:bottle:")
    assert grasp.collision_context_after_events_id.startswith(
        "object-attached-release:bottle:"
    )
    assert lift.collision_context_id.startswith("object-attached-release:bottle:")
    assert transfer.collision_context_id.startswith("object-attached:bottle:")
    assert lift.collision_context_id == grasp.collision_context_after_events_id
    release = setup.collision_contexts[lift.collision_context_id]
    assert ("bottle", "table*") in release.allowed_collision_pairs
    attached = setup.collision_contexts[transfer.collision_context_id]
    assert attached.attached_object_ids == ["bottle"]
    assert attached.attached_object_transforms[0].reference_name == (
        "robot0_right_hand"
    )
    assert {item.object_id for item in attached.free_object_poses} == {"other"}
    contact = setup.collision_contexts[grasp.collision_context_id]
    assert ("2F", "bottle") in contact.allowed_collision_pairs
    assert setup.keyframe_artifact.provenance.input_artifact_ids == [
        "source-keyframe-artifact"
    ]


def test_place_binds_detach_and_stationary_target_pose_for_retreat() -> None:
    attached = AttachedObjectTransform(
        object_id="bottle",
        free_joint_name="bottle_free",
        reference_kind="body",
        reference_name="robot0_right_hand",
        position_in_reference_m=(0.0, 0.0, 0.1),
        orientation_in_reference_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    target_pose = Pose(
        frame_id="world",
        position_m=(0.7, 0.1, 0.15),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    request = _request(
        MotionGoal(
            goal_type=GoalType.PLACE,
            target_object_id="bottle",
            target_pose=target_pose,
            target_region_id="table_collision",
        ),
        attached=attached,
    )
    source = _artifact(
        (
            _keyframe("transfer", KeyframeType.TRANSFER),
            _keyframe(
                "place",
                KeyframeType.PLACE,
                events=(
                    KeyframeEventType.DETACH_OBJECT,
                    KeyframeEventType.GRIPPER_OPEN,
                ),
            ),
            _keyframe("retreat", KeyframeType.RETREAT),
        )
    )

    setup = _factory().prepare(request, source)
    transfer, place, retreat = setup.keyframe_artifact.candidates[0].keyframes

    assert transfer.collision_context_id == setup.initial_collision_context_id
    assert place.collision_context_id.startswith("place-contact:bottle:")
    assert place.collision_context_after_events_id.startswith(
        "object-detached:bottle:"
    )
    assert retreat.collision_context_id == place.collision_context_after_events_id
    detached = setup.collision_contexts[retreat.collision_context_id]
    assert detached.attached_object_ids == []
    bottle = next(
        item for item in detached.free_object_poses if item.object_id == "bottle"
    )
    assert bottle.pose == target_pose
    contact = setup.collision_contexts[place.collision_context_id]
    assert ("bottle", "table_collision") in contact.allowed_collision_pairs


def test_default_motion_gets_explicit_context_on_every_keyframe() -> None:
    request = _request(
        MotionGoal(
            goal_type=GoalType.POSE,
            target_pose=Pose(
                frame_id="world",
                position_m=(0.5, 0.0, 0.3),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
        )
    )
    source = _artifact(
        (
            _keyframe("a", KeyframeType.CUSTOM),
            _keyframe("b", KeyframeType.CUSTOM),
        )
    )

    setup = _factory().prepare(request, source)

    assert {
        item.collision_context_id
        for item in setup.keyframe_artifact.candidates[0].keyframes
    } == {setup.initial_collision_context_id}


def test_workcell_router_selects_environment_binding() -> None:
    calls = []
    router = WorkcellMotionRequestRouter(
        {"C2_1_ObjectSorting": lambda request: calls.append(request.request_id)}
    )
    request = _request(
        MotionGoal(
            goal_type=GoalType.POSE,
            target_pose=Pose(
                frame_id="world",
                position_m=(0.5, 0.0, 0.3),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
        )
    )

    router(request)

    assert calls == [request.request_id]


def test_runtime_attachment_state_converts_to_contract_quaternion() -> None:
    transform = attached_object_transform_from_state(
        SimpleNamespace(
            object_id="bottle",
            free_joint_name="bottle_free",
            reference_kind="site",
            reference_name="grip_site",
            position_in_reference_m=(0.0, 0.0, 0.1),
            rotation_in_reference=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        )
    )

    assert transform.object_id == "bottle"
    assert transform.reference_kind == "site"
    assert transform.orientation_in_reference_xyzw == (0.0, 0.0, 0.0, 1.0)
