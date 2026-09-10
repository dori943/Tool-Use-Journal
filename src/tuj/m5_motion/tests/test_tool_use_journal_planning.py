from __future__ import annotations
import pytest

from types import SimpleNamespace

from tuj.m4_taskplanner.models import GraspSpec

from tuj.m5_motion.schema import (
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
from tuj.m5_motion.ee_exchange import EEExchangeKeyframeProvider
from tuj.m5_motion.tool_use_journal_planning import (
    ToolUseJournalCollisionBindingError,
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

    @staticmethod
    def build_ee_exchange_contexts(*, from_ee, to_ee):
        from tuj.m5_motion.ee_exchange import EEExchangeTemplateGenerator

        return EEExchangeTemplateGenerator().build_collision_contexts(
            from_ee=from_ee,
            to_ee=to_ee,
            bare_flange_model_version="model:None",
            attached_model_versions={to_ee: f"model:{to_ee}"},
        )

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
        rack={
            "2F": {
                "dock_pose": {
                    "position_m": [0.2, -0.5, 0.4],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "approach_axis_xyz": [1.0, 0.0, 0.0],
            }
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


def _request(
    goal: MotionGoal,
    *,
    action_type: str = "MOVE",
    attached=None,
) -> MotionPlanRequest:
    grasp = None
    if action_type.upper() == "PICK":
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
            action_type=action_type,
            ee="2F",
            target_ids=["bottle"],
            grasp=grasp,
            goal=goal,
            allowed_touch_objects=["table_collision"],
        ),
        constraints=_constraints(),
    )


def _exchange_entry_request() -> MotionPlanRequest:
    request = _request(
        MotionGoal(goal_type=GoalType.POSE, target_object_id="2F")
    )
    request.task = MotionTask(
        task_id="exchange-entry-2F",
        subgoal_id="exchange-entry-2F",
        action_type="EE_EXCHANGE_ENTRY",
        ee="2F",
        target_ids=["2F"],
        goal=MotionGoal(goal_type=GoalType.POSE),
        metadata={
            "entry_ee": "2F",
            "from_ee": "2F",
            "next_ee": "3F",
        },
    )
    return request


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
        MotionGoal(goal_type=GoalType.POSE, target_object_id="bottle"),
        action_type="PICK",
    )
    request.task.metadata["support_collision_selectors"] = ["table_collision"]
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
    assert lift.collision_context_after_events_id == transfer.collision_context_id
    assert lift.metadata["validate_endpoint_after_context"] is True
    release = setup.collision_contexts[lift.collision_context_id]
    assert ("bottle", "table_collision") in release.allowed_collision_pairs
    assert release.metadata["post_segment_validation_context_id"] == (
        transfer.collision_context_id
    )
    assert release.metadata["bounded_collision_allowances"] == [
        {
            "selectors": ["bottle", "table_collision"],
            "minimum_distance_m": -0.001,
        }
    ]
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


def test_pick_infers_exact_initial_support_for_only_the_separation_edge() -> None:
    request = _request(
        MotionGoal(goal_type=GoalType.POSE, target_object_id="bottle"),
        action_type="PICK",
    )
    request.world.obstacles = [
        {
            "obstacle_id": "fixture_surface",
            "aabb_min_m": [0.0, -0.5, 0.0],
            "aabb_max_m": [1.0, 0.5, 0.1],
            "collision_enabled": True,
        }
    ]
    source = _artifact(
        (
            _keyframe("pre", KeyframeType.PRE_GRASP),
            _keyframe(
                "grasp",
                KeyframeType.GRASP,
                events=(KeyframeEventType.ATTACH_OBJECT,),
            ),
            _keyframe("retreat", KeyframeType.RETREAT),
            _keyframe("transfer", KeyframeType.TRANSFER),
        )
    )

    setup = _factory().prepare(request, source)
    _, grasp, retreat, transfer = setup.keyframe_artifact.candidates[0].keyframes

    assert "support_collision_selectors" not in request.task.metadata
    assert retreat.collision_context_id.startswith(
        "object-attached-release:bottle:"
    )
    release = setup.collision_contexts[retreat.collision_context_id]
    assert release.allowed_collision_pairs == [("bottle", "fixture_surface")]
    assert release.metadata["support_separation"] == {
        "policy": "AUTO_INITIAL_SUPPORT_V1",
        "detection_source": "world.obstacles.aabb",
        "support_initial_clearance_m": pytest.approx(0.0),
        "support_horizontal_overlap_ratio": pytest.approx(1.0),
        "support_min_horizontal_overlap_ratio": pytest.approx(0.5),
        "maximum_penetration_m": pytest.approx(0.001),
        "target_selector": "bottle",
        "support_selectors": ["fixture_surface"],
    }
    assert grasp.collision_context_after_events_id == retreat.collision_context_id
    assert retreat.collision_context_after_events_id == transfer.collision_context_id
    assert transfer.collision_context_id.startswith("object-attached:bottle:")


def test_pick_does_not_infer_support_from_a_sliver_overlap() -> None:
    request = _request(
        MotionGoal(goal_type=GoalType.POSE, target_object_id="bottle"),
        action_type="PICK",
    )
    request.world.obstacles = [
        {
            "obstacle_id": "neighbor-edge",
            "aabb_min_m": [0.424, -0.025, 0.0],
            "aabb_max_m": [0.6, 0.025, 0.1],
            "collision_enabled": True,
        }
    ]
    source = _artifact(
        (
            _keyframe(
                "grasp",
                KeyframeType.GRASP,
                events=(KeyframeEventType.ATTACH_OBJECT,),
            ),
            _keyframe("lift", KeyframeType.LIFT),
        )
    )

    setup = _factory().prepare(request, source)
    grasp, lift = setup.keyframe_artifact.candidates[0].keyframes

    assert grasp.collision_context_after_events_id.startswith(
        "object-attached:bottle:"
    )
    assert lift.collision_context_id == grasp.collision_context_after_events_id
    assert not any(
        context_id.startswith("object-attached-release:bottle:")
        for context_id in setup.collision_contexts
    )


def test_pick_rejects_pattern_support_selectors() -> None:
    request = _request(
        MotionGoal(goal_type=GoalType.POSE, target_object_id="bottle"),
        action_type="PICK",
    )
    request.task.metadata["support_collision_selectors"] = ["table*"]
    source = _artifact(
        (
            _keyframe(
                "grasp",
                KeyframeType.GRASP,
                events=(KeyframeEventType.ATTACH_OBJECT,),
            ),
            _keyframe("lift", KeyframeType.LIFT),
        )
    )

    with pytest.raises(
        ToolUseJournalCollisionBindingError,
        match="must use exact selectors",
    ):
        _factory().prepare(request, source)


def test_pick_rejects_initial_support_penetration_over_one_millimeter() -> None:
    request = _request(
        MotionGoal(goal_type=GoalType.POSE, target_object_id="bottle"),
        action_type="PICK",
    )
    request.world.obstacles = [
        {
            "obstacle_id": "badly-overlapping-support",
            "aabb_min_m": [0.0, -0.5, 0.0],
            "aabb_max_m": [1.0, 0.5, 0.1015],
            "collision_enabled": True,
        }
    ]
    source = _artifact(
        (
            _keyframe(
                "grasp",
                KeyframeType.GRASP,
                events=(KeyframeEventType.ATTACH_OBJECT,),
            ),
            _keyframe("lift", KeyframeType.LIFT),
        )
    )

    with pytest.raises(
        ToolUseJournalCollisionBindingError,
        match="initial support penetration exceeds the 1 mm hard limit",
    ):
        _factory().prepare(request, source)


def test_explicit_empty_pick_support_policy_disables_inference() -> None:
    request = _request(
        MotionGoal(goal_type=GoalType.POSE, target_object_id="bottle"),
        action_type="PICK",
    )
    request.task.metadata["support_collision_selectors"] = []
    request.world.obstacles = [
        {
            "obstacle_id": "fixture_surface",
            "aabb_min_m": [0.0, -0.5, 0.0],
            "aabb_max_m": [1.0, 0.5, 0.1],
            "collision_enabled": True,
        }
    ]
    source = _artifact(
        (
            _keyframe(
                "grasp",
                KeyframeType.GRASP,
                events=(KeyframeEventType.ATTACH_OBJECT,),
            ),
            _keyframe("lift", KeyframeType.LIFT),
        )
    )

    setup = _factory().prepare(request, source)
    grasp, lift = setup.keyframe_artifact.candidates[0].keyframes

    assert grasp.collision_context_after_events_id.startswith(
        "object-attached:bottle:"
    )
    assert lift.collision_context_id == grasp.collision_context_after_events_id
    assert lift.collision_context_after_events_id is None
    assert all(
        not context.allowed_collision_pairs
        for context_id, context in setup.collision_contexts.items()
        if context_id.startswith("object-attached:")
    )


@pytest.mark.parametrize(
    "grasp_mode",
    ["CONTACT_FRICTION", "contact_friction", "contact-friction"],
)
def test_contact_friction_pick_keeps_target_free_after_gripper_close(
    grasp_mode,
) -> None:
    request = _request(
        MotionGoal(goal_type=GoalType.POSE, target_object_id="bottle"),
        action_type="PICK",
    )
    request.task.metadata["grasp_execution_mode"] = grasp_mode
    source = _artifact(
        (
            _keyframe("pre", KeyframeType.PRE_GRASP),
            _keyframe(
                "grasp",
                KeyframeType.GRASP,
                events=(KeyframeEventType.GRIPPER_CLOSE,),
            ),
            _keyframe("lift", KeyframeType.LIFT),
        )
    )

    setup = _factory().prepare(request, source)
    pre, grasp, lift = setup.keyframe_artifact.candidates[0].keyframes

    assert pre.collision_context_id == setup.initial_collision_context_id
    assert grasp.collision_context_id.startswith(
        "physical-grasp-contact:bottle:"
    )
    assert lift.collision_context_id == grasp.collision_context_id
    assert grasp.collision_context_after_events_id is None
    assert all(
        not context.attached_object_ids
        for context in setup.collision_contexts.values()
    )
    assert (
        "2F",
        "bottle",
    ) in setup.collision_contexts[grasp.collision_context_id].allowed_collision_pairs


@pytest.mark.parametrize("release_contact", [False, True])
def test_place_binds_detach_and_stationary_target_pose_for_retreat(release_contact) -> None:
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
            goal_type=GoalType.POSE,
            target_object_id="bottle",
            target_pose=target_pose,
            target_region_id="table_collision",
        ),
        action_type="PLACE",
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

    if release_contact:
        source.candidates[0].keyframes[1].metadata["allow_release_contact"] = True
        source.candidates[0].keyframes.append(_keyframe("clear", KeyframeType.RETREAT))
    setup = _factory().prepare(request, source)
    transfer, place, retreat, *clear = setup.keyframe_artifact.candidates[0].keyframes

    assert transfer.collision_context_id == setup.initial_collision_context_id
    assert place.collision_context_id.startswith("place-contact:bottle:")
    assert place.collision_context_after_events_id.startswith(
        "object-release-contact:bottle:" if release_contact else "object-detached:bottle:"
    )
    assert retreat.collision_context_id == place.collision_context_after_events_id
    detached = setup.collision_contexts[retreat.collision_context_id]
    assert detached.attached_object_ids == []
    bottle = next(
        item for item in detached.free_object_poses if item.object_id == "bottle"
    )
    # The untagged keyframe puts the reference at bottle center (.4, 0, .2).
    # Its captured +.1m attachment offset survives release; goal pose is stale.
    assert bottle.pose.position_m == pytest.approx((0.4, 0.0, 0.3))
    assert bottle.pose.orientation_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))
    if release_contact:
        assert detached.allowed_collision_pairs
        assert clear[0].collision_context_id != retreat.collision_context_id
        assert not setup.collision_contexts[clear[0].collision_context_id].allowed_collision_pairs
    contact = setup.collision_contexts[place.collision_context_id]
    assert ("bottle", "table_collision") in contact.allowed_collision_pairs


def test_contact_friction_place_uses_collision_proxy_then_opens_gripper() -> None:
    target_pose = Pose(
        frame_id="world",
        position_m=(0.7, 0.1, 0.15),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    request = _request(
        MotionGoal(
            goal_type=GoalType.POSE,
            target_object_id="bottle",
            target_pose=target_pose,
            target_region_id="table_collision",
        ),
        action_type="PLACE",
    )
    request.world.robot_state.held_tool_id = "bottle"
    request.world.metadata["contact_friction_held_objects"] = {
        "bottle": {
            "object_id": "bottle",
            "free_joint_name": "bottle_free",
            "reference_kind": "body",
            "reference_name": "robot0_right_hand",
            "position_in_reference_m": [0.0, 0.0, 0.1],
            "orientation_in_reference_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    }
    source = _artifact(
        (
            _keyframe("transfer", KeyframeType.TRANSFER),
            _keyframe(
                "place",
                KeyframeType.PLACE,
                events=(KeyframeEventType.GRIPPER_OPEN,),
            ),
            _keyframe("retreat", KeyframeType.RETREAT),
        )
    )

    setup = _factory().prepare(request, source)
    transfer, place, retreat = setup.keyframe_artifact.candidates[0].keyframes

    initial = setup.collision_contexts[setup.initial_collision_context_id]
    assert initial.metadata["attachment_proxy"] == "CONTACT_FRICTION"
    assert initial.attached_object_ids == ["bottle"]
    assert transfer.collision_context_id == setup.initial_collision_context_id
    assert place.collision_context_id.startswith("place-contact:bottle:")
    assert place.collision_context_after_events_id.startswith(
        "object-detached:bottle:"
    )
    assert retreat.collision_context_id == place.collision_context_after_events_id


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


def test_initial_ee_attach_binds_bare_flange_as_initial_context() -> None:
    request = _request(
        MotionGoal(goal_type=GoalType.POSE, target_object_id="2F")
    )
    request.world.metadata["physical_active_ee"] = None
    request.world.metadata["declared_active_ee"] = None
    request.task.action_type = "EE_ATTACH"
    request.task.target_ids = ["2F"]
    request.task.metadata = {"from_ee": None, "to_ee": "2F"}
    compiler = _Compiler()
    artifact = EEExchangeKeyframeProvider().generate(request)

    setup = _factory(compiler).prepare(request, artifact)

    assert setup.initial_collision_context_id == "bare-flange"
    assert set(setup.collision_contexts) == {
        "bare-flange",
        "bare-flange-dock-contact:2F",
        "ee-attached-dock-contact:2F",
        "ee-attached:2F",
    }
    assert compiler.calls[0][1]["default_active_ee"] is None


def test_exchange_entry_uses_only_the_current_attached_ee_for_motion() -> None:
    request = _exchange_entry_request()
    compiler = _Compiler()

    contexts, registry = _factory(compiler).prepare_ee_exchange_entry(request)

    assert "ee-attached:2F" in contexts
    assert contexts["ee-attached:2F"].active_ee == "2F"
    assert compiler.calls[0][1]["default_active_ee"] == "2F"
    assert registry is not None


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
