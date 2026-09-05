from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.m5_motion.schema import (
    ArtifactProvenance, AttachedObjectTransform, CollisionContext, MotionGoal,
    MotionPlanRequest, MotionTask, Pose, RobotState, SceneRef, WorldSnapshot,
)
from tuj.m5_motion.scripted_grasps.registry import (
    ENABLED_ENTRIES, ENTRIES, EXPERIMENTAL_INTEGRATION, PENDING_INTEGRATION,
    integration_status, resolve,
)
from tuj.m5_motion.scripted_grasps.frames import transform
from tuj.m5_motion.tool_use_journal_runtime import (
    ToolUseJournalEERuntime, ToolUseJournalKinematicTrajectoryPlayer, ToolUseJournalRuntimeError,
)
from tuj.m5_motion.tool_use_journal_planning import ToolUseJournalCollisionContextFactory


def request_for(entry=ENTRIES[0], action="PICK", object_id=None):
    target = object_id or entry.object_id
    return MotionPlanRequest(request_id="request",
        provenance=ArtifactProvenance(artifact_id="request", artifact_type="MotionPlanRequest",
            produced_by="TASK_PLANNER", invocation_id="test"),
        world=WorldSnapshot(scene=SceneRef(signature="before"),
            robot_state=RobotState(robot_id="robot", joint_names=["j1"], joint_positions_rad=[0.]),
            metadata={"environment_name": entry.environment, "physical_active_ee": entry.ee}),
        task=MotionTask(task_id="task", subgoal_id="pick", action_type=action, ee=entry.ee,
            target_ids=[target], goal=MotionGoal(goal_type="POSE", target_object_id=target),
            metadata={"attach_target": False}))


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.object_id)
def test_exact_scene_object_and_ee_dispatch(entry):
    for request in (request_for(entry), request_for(entry, object_id=f"obj_{entry.object_id}_{entry.object_id}")):
        if entry.object_id in PENDING_INTEGRATION:
            with pytest.raises(ScriptedGraspUnavailable, match="NOT_VALIDATED"):
                resolve(request)
        else:
            assert resolve(request) == entry
    assert callable(entry.function())
    assert entry.recipe().ee_id == entry.ee
    request = request_for(entry, action="MOVE")
    assert resolve(request) is None
    request = request_for(entry)
    request.world.metadata["environment_name"] = "different-scene"
    assert resolve(request) is None


def test_excluded_unknown_and_wrong_hand():
    assert len(ENTRIES) == 14
    assert not {"tongs", "ladle"} & {e.object_id for e in ENTRIES}
    assert resolve(request_for(object_id="plate_large")) is None
    assert resolve(request_for(object_id="tongs")) is None
    request = request_for()
    request.task.ee = "3F"
    with pytest.raises(ValueError, match="EE_MISMATCH"):
        resolve(request)


def test_plate_is_routable_but_explicitly_experimental():
    plate = ENTRIES[0]
    assert plate.object_id == "plate"
    assert plate in ENABLED_ENTRIES
    assert plate.object_id in EXPERIMENTAL_INTEGRATION
    assert plate.object_id not in PENDING_INTEGRATION
    assert integration_status(plate) == "EXPERIMENTAL"
    assert resolve(request_for(plate)) == plate


@pytest.mark.parametrize("entry", [e for e in ENTRIES if e.driver == "catalog"], ids=lambda e: e.object_id)
def test_catalog_targets_follow_body_translation_rotation_and_center_offset(entry):
    from tuj.m5_motion.scripted_grasps.catalog_types import build_catalog_targets
    recipe = entry.recipe()
    center = np.array([.012, -.003, .006])
    original = transform([.3, -.1, .8], rotation=np.eye(3))
    moved = transform([-.2, .4, .85], rotation=Rotation.from_euler("z", 37, degrees=True).as_matrix())
    a = build_catalog_targets(original, center, recipe.expected_size_m, recipe)
    b = build_catalog_targets(moved, center, recipe.expected_size_m, recipe)
    np.testing.assert_allclose(np.linalg.inv(original) @ a["GRASP"], np.linalg.inv(moved) @ b["GRASP"], atol=1e-12)
    np.testing.assert_allclose(b["T_WC"][:3, 3], moved[:3, 3] + moved[:3, :3] @ center)
    np.testing.assert_allclose(b["LIFT"][:3, 3] - b["GRASP"][:3, 3], [0, 0, recipe.lift_distance_m])


def test_collision_proxy_uses_observed_transform_without_runtime_attachment():
    request = request_for()
    held = AttachedObjectTransform(object_id="plate", free_joint_name="plate_free",
        reference_kind="site", reference_name="grip", position_in_reference_m=(.03, -.002, .01),
        orientation_in_reference_xyzw=(0, 0, 0, 1))
    request.world.robot_state.held_tool_id = "plate"
    request.world.metadata["contact_friction_held_objects"] = {"plate": held.model_dump()}
    request.world.objects = {"plate": {"pose": Pose(frame_id="world", position_m=(0, 0, 0), orientation_xyzw=(0, 0, 0, 1)).model_dump(), "free_joint_name": "plate_free"},
                             "other": {"pose": Pose(frame_id="world", position_m=(1, 0, 0), orientation_xyzw=(0, 0, 0, 1)).model_dump(), "free_joint_name": "other_free"}}
    factory = ToolUseJournalCollisionContextFactory(SimpleNamespace(model_version_for=lambda ee: "fixture"), attachment_reference_name="grip")
    context = factory._base_context(request, active_ee="2F")
    assert context.attached_object_transforms == [held]
    assert context.metadata["attachment_proxy"] == "CONTACT_FRICTION"
    assert request.world.robot_state.attached_object_id is None
    assert [p.object_id for p in context.free_object_poses] == ["other"]


def runtime_stub():
    runtime = ToolUseJournalEERuntime.__new__(ToolUseJournalEERuntime)
    runtime._closed = False
    runtime._env = SimpleNamespace(obj_body_id={"plate": 1})
    runtime._attachment = None
    runtime._active_ee = "2F"
    runtime._held_tool_id = None
    runtime._grasp_engaged = True
    runtime._gripper_command = 1.
    runtime.scripted_grasp_retention = SimpleNamespace(entry=ENTRIES[0])
    return runtime


def test_contact_resource_does_not_create_attachment_and_open_clears_retention():
    runtime = runtime_stub()
    runtime.mark_contact_friction_object_as_tool("plate")
    assert runtime.attachment is None
    assert runtime.held_tool_id == "plate"
    runtime.command_gripper(engaged=False, suction=False)
    assert runtime.held_tool_id is None
    assert runtime.scripted_grasp_retention is None
    assert not runtime.grasp_engaged


def test_detach_event_physically_opens_contact_grasp():
    from tuj.m5_motion.schema import TrajectoryEvent
    runtime = runtime_stub()
    runtime.mark_contact_friction_object_as_tool("plate")
    player = ToolUseJournalKinematicTrajectoryPlayer(runtime)
    event = TrajectoryEvent(event_id="release", event_type="DETACH_OBJECT", time_from_start_s=0., target_id="plate")
    player._execute_event(event)
    assert runtime.held_tool_id is None and runtime.gripper_command < 0.
    assert runtime.scripted_grasp_retention is None


def test_proxy_context_requires_matching_live_contact_resource():
    runtime = runtime_stub()
    runtime.mark_contact_friction_object_as_tool("plate")
    player = ToolUseJournalKinematicTrajectoryPlayer(runtime)
    context = CollisionContext(context_id="held", active_ee="2F", attached_object_ids=["plate"],
        collision_model_version="fixture",
        metadata={"attachment_proxy": "CONTACT_FRICTION"})
    player._verify_runtime_context(context, label="start")
    runtime.scripted_grasp_retention = None
    with pytest.raises(ToolUseJournalRuntimeError, match="attached objects"):
        player._verify_runtime_context(context, label="start")


def test_script_dispatch_is_lazy_and_next_request_receives_actual_joints(monkeypatch, tmp_path):
    from tuj.m5_motion.scripted_grasps import live, context
    request = request_for(ENTRIES[4])
    actual = request.world.model_copy(deep=True)
    monkeypatch.setattr(live, "snapshot", lambda runtime, previous=None: actual.model_copy(deep=True))
    calls = []
    def grasp(runtime, entry, output, **kwargs):
        calls.append(kwargs["request"].world.robot_state.joint_positions_rad)
        actual.robot_state.joint_positions_rad = [.37]
        actual.scene.signature = "measured-after-grasp"
        return {"status": "SUCCESS"}
    monkeypatch.setattr(context, "execute_grasp", grasp)
    def planner_factory(env, repository, **kwargs):
        def planner(next_request):
            assert next_request.world.robot_state.joint_positions_rad == [.37]
            assert next_request.world.scene.signature == "measured-after-grasp"
            raise RuntimeError("ordinary planner reached")
        return planner
    session = live.ScriptedGraspSession(SimpleNamespace(env=object()), tmp_path, tmp_path / "run", planner_factory=planner_factory)
    session.execute_request(request, completed_subgoal="pick")
    assert calls == [[0.]]
    request.task.action_type = "MOVE"
    with pytest.raises(RuntimeError, match="ordinary planner reached"):
        session.execute_request(request)
    assert [r["route"] for r in session.records] == ["SCRIPTED_GRASP", "M5_MOTION_PLAN"]


def test_failed_grasp_is_recorded_and_never_completes_subgoal(monkeypatch, tmp_path):
    from tuj.m5_motion.scripted_grasps import live, context
    request = request_for(ENTRIES[4])
    monkeypatch.setattr(live, "snapshot", lambda runtime, previous=None: request.world.model_copy(deep=True))
    def fail(*args, **kwargs):
        raise RuntimeError("CONTACT_NOT_STABLE")
    monkeypatch.setattr(context, "execute_grasp", fail)
    session = live.ScriptedGraspSession(SimpleNamespace(), tmp_path, tmp_path)
    with pytest.raises(RuntimeError, match="CONTACT_NOT_STABLE"):
        session.execute_request(request, completed_subgoal="pick")
    assert session.world.scene.completed_subgoals == []
    manifest = json.loads((tmp_path / "live-execution-manifest.json").read_text())
    assert manifest["status"] == "FAILED"
    assert manifest["steps"][0]["status"] == "FAILED"


def test_failed_experimental_plate_keeps_route_and_status(monkeypatch, tmp_path):
    from tuj.m5_motion.scripted_grasps import live, context
    request = request_for(ENTRIES[0])
    monkeypatch.setattr(live, "snapshot", lambda runtime, previous=None: request.world.model_copy(deep=True))
    monkeypatch.setattr(context, "execute_grasp",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("PRE_LIFT_CONTACT_NOT_STABLE")))
    session = live.ScriptedGraspSession(SimpleNamespace(), tmp_path, tmp_path)
    with pytest.raises(RuntimeError, match="PRE_LIFT_CONTACT_NOT_STABLE"):
        session.execute_request(request)
    step = json.loads((tmp_path / "live-execution-manifest.json").read_text())["steps"][0]
    assert step["route"] == "SCRIPTED_GRASP"
    assert step["object_id"] == "plate"
    assert step["integration_status"] == "EXPERIMENTAL"


def test_place_switches_contact_proxy_to_free_object_after_release():
    from tuj.m5_motion.tests.test_tool_use_journal_planning import _request, _artifact, _keyframe, _factory
    from tuj.m5_motion.schema import KeyframeType, KeyframeEventType
    held = AttachedObjectTransform(object_id="bottle", free_joint_name="bottle_free",
        reference_name="robot0_right_hand", position_in_reference_m=(.03, .01, -.02),
        orientation_in_reference_xyzw=(0, 0, 0, 1))
    target = Pose(frame_id="world", position_m=(.4, 0, .2), orientation_xyzw=(0, 0, 0, 1))
    request = _request(MotionGoal(goal_type="POSE", target_object_id="bottle", target_pose=target), action_type="PLACE")
    request.world.robot_state.held_tool_id = "bottle"
    request.world.metadata["contact_friction_held_objects"] = {"bottle": held.model_dump()}
    request.task.allowed_touch_objects = ["table_collision"]
    artifact = _artifact((
        _keyframe("transfer", KeyframeType.TRANSFER),
        _keyframe("place", KeyframeType.PLACE, events=(KeyframeEventType.DETACH_OBJECT, KeyframeEventType.GRIPPER_OPEN)),
        _keyframe("retreat", KeyframeType.RETREAT),
    ))
    setup = _factory().prepare(request, artifact)
    transfer, place, retreat = setup.keyframe_artifact.candidates[0].keyframes
    before = setup.collision_contexts[place.collision_context_id]
    after = setup.collision_contexts[retreat.collision_context_id]
    assert before.metadata["attachment_proxy"] == "CONTACT_FRICTION"
    assert before.attached_object_transforms == [held]
    assert ("bottle", "table_collision") in before.allowed_collision_pairs
    assert after.attached_object_ids == []
    assert next(p for p in after.free_object_poses if p.object_id == "bottle").pose == target


def test_experimental_plate_routes_to_grasp_without_planner_fallback(monkeypatch, tmp_path):
    from tuj.m5_motion.scripted_grasps import live, context
    request = request_for()
    monkeypatch.setattr(live, "snapshot", lambda runtime, previous=None: request.world.model_copy(deep=True))
    calls = []
    def grasp(*args, **kwargs):
        calls.append(kwargs["request"].task.goal.target_object_id)
        return {"status": "SUCCESS", "metrics": {}}
    def forbidden(*args, **kwargs):
        pytest.fail("experimental plate must not fall back to the ordinary planner")
    monkeypatch.setattr(context, "execute_grasp", grasp)
    session = live.ScriptedGraspSession(SimpleNamespace(), tmp_path, tmp_path, planner_factory=forbidden)
    record = session.execute_request(request)
    assert calls == ["plate"]
    assert record["route"] == "SCRIPTED_GRASP"
    assert record["integration_status"] == "EXPERIMENTAL"
