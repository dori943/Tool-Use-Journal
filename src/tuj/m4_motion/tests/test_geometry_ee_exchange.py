"""VLM keyframes stay symbolic and rack exchange stays template-driven."""

from __future__ import annotations

import pytest

from tuj.m4_motion.ee_exchange import (
    EEExchangeTemplateGenerator,
    RoutedKeyframeStrategyProvider,
)
from tuj.m4_motion.geometry import RelativePoseResolver
from tuj.m4_motion.schema import (
    KeyframeEventType,
    KeyframePlannerType,
    KeyframeType,
    RelativeKeyframeSpec,
    RobotState,
    SceneRef,
    WorldSnapshot,
    ArtifactProvenance,
    GoalType,
    ModuleName,
    MotionGoal,
    MotionPlanRequest,
    MotionTask,
)


def _world() -> WorldSnapshot:
    return WorldSnapshot(
        scene=SceneRef(signature="scene-1"),
        robot_state=RobotState(
            robot_id="ur5e-1",
            joint_names=["j1", "j2"],
            joint_positions_rad=[0.0, 0.0],
        ),
        objects={
            "bottle": {
                "pose": {
                    "position_m": [0.4, -0.1, 0.5],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "dimensions_m": [0.08, 0.08, 0.2],
            }
        },
        rack={
            "2f": {
                "dock_pose": {
                    "position_m": [0.2, -0.5, 0.4],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "approach_axis_xyz": [1.0, 0.0, 0.0],
            },
            "vacuum": {
                "dock_pose": {
                    "position_m": [0.2, -0.3, 0.4],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "approach_axis_xyz": [1.0, 0.0, 0.0],
            },
        },
    )


def test_relative_keyframe_resolves_from_object_anchor() -> None:
    keyframe = RelativeKeyframeSpec(
        keyframe_id="pre-grasp",
        keyframe_type=KeyframeType.PRE_GRASP,
        frame_ref="object:bottle",
        anchor="top_center",
        approach_axis_xyz=(0.0, 0.0, 1.0),
        offset_along_approach_m=0.1,
        planner=KeyframePlannerType.CARTESIAN,
    )

    pose = RelativePoseResolver(_world()).resolve(keyframe)

    assert pose.position_m == pytest.approx((0.4, -0.1, 0.7))
    assert sum(value * value for value in pose.orientation_xyzw) == pytest.approx(1.0)


def test_ee_exchange_template_has_explicit_model_transitions() -> None:
    candidate = EEExchangeTemplateGenerator().generate(
        _world(),
        subgoal_id="swap-1",
        from_ee="2f",
        to_ee="vacuum",
    )

    assert len(candidate.keyframes) == 8
    undock = next(
        item for item in candidate.keyframes if item.keyframe_type is KeyframeType.EE_UNDOCK
    )
    dock = next(
        item for item in candidate.keyframes if item.keyframe_type is KeyframeType.EE_DOCK
    )
    assert undock.events_after == [
        KeyframeEventType.TOOL_UNLOCK,
        KeyframeEventType.VERIFY_TOOL_RELEASE,
    ]
    assert undock.collision_context_id == "ee-attached-dock-contact:2f"
    assert undock.collision_context_after_events_id == "bare-flange"
    assert dock.collision_context_id == "bare-flange-dock-contact:vacuum"
    assert dock.collision_context_after_events_id == "ee-attached:vacuum"
    assert all(item.frame_ref.startswith("rack:") for item in candidate.keyframes)


def test_initial_ee_attach_starts_bare_and_skips_undock() -> None:
    generator = EEExchangeTemplateGenerator()

    candidate = generator.generate(
        _world(),
        subgoal_id="initial-attach",
        from_ee=None,
        to_ee="vacuum",
    )
    contexts = generator.build_collision_contexts(
        from_ee=None,
        to_ee="vacuum",
    )

    assert len(candidate.keyframes) == 4
    assert candidate.keyframes[0].collision_context_id == "bare-flange"
    assert all(
        KeyframeEventType.TOOL_UNLOCK not in item.events_after
        for item in candidate.keyframes
    )
    assert any(
        KeyframeEventType.TOOL_LOCK in item.events_after
        for item in candidate.keyframes
    )
    assert set(contexts) == {
        "bare-flange",
        "bare-flange-dock-contact:vacuum",
        "ee-attached:vacuum",
    }


def test_ee_exchange_contexts_limit_contact_relaxation_to_dock_segments() -> None:
    contexts = EEExchangeTemplateGenerator().build_collision_contexts(
        from_ee="2f",
        to_ee="vacuum",
    )

    old_free = contexts["ee-attached:2f"]
    old_dock = contexts["ee-attached-dock-contact:2f"]
    bare = contexts["bare-flange"]
    new_dock = contexts["bare-flange-dock-contact:vacuum"]

    assert old_free.scene_state_id == old_dock.scene_state_id
    assert old_free.collision_model_version == old_dock.collision_model_version
    assert old_free.allowed_collision_pairs == []
    assert old_dock.allowed_collision_pairs == [("2f", "rack_support:2f")]
    assert bare.scene_state_id == new_dock.scene_state_id
    assert bare.collision_model_version == new_dock.collision_model_version
    assert new_dock.allowed_collision_pairs == [("qc_master", "vacuum")]
    assert contexts["ee-attached:vacuum"].active_ee == "vacuum"


def test_routed_provider_uses_template_without_calling_default() -> None:
    class _Default:
        def generate(self, request):
            raise AssertionError("default provider must not handle EE exchange")

    request = MotionPlanRequest(
        request_id="exchange-request",
        provenance=ArtifactProvenance(
            artifact_id="exchange-request-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="test",
        ),
        world=_world(),
        task=MotionTask(
            task_id="exchange-task",
            subgoal_id="exchange-subgoal",
            action_type="EE_EXCHANGE",
            ee="vacuum",
            target_ids=["2f", "vacuum"],
            goal=MotionGoal(goal_type=GoalType.POSE, target_object_id="vacuum"),
            metadata={"from_ee": "2f", "to_ee": "vacuum"},
        ),
    )

    artifact = RoutedKeyframeStrategyProvider(_Default()).generate(request)

    assert artifact.scene_signature == "scene-1"
    assert len(artifact.candidates) == 1
    assert artifact.candidates[0].metadata == {
        "from_ee": "2f",
        "to_ee": "vacuum",
    }


def test_routed_provider_accepts_initial_attach_without_from_ee() -> None:
    class _Default:
        def generate(self, request):
            raise AssertionError("default provider must not handle initial EE attach")

    request = MotionPlanRequest(
        request_id="attach-request",
        provenance=ArtifactProvenance(
            artifact_id="attach-request-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="test",
        ),
        world=_world(),
        task=MotionTask(
            task_id="attach-task",
            subgoal_id="attach-subgoal",
            action_type="EE_ATTACH",
            ee="vacuum",
            target_ids=["vacuum"],
            goal=MotionGoal(goal_type=GoalType.POSE, target_object_id="vacuum"),
            metadata={"from_ee": None, "to_ee": "vacuum"},
        ),
    )

    artifact = RoutedKeyframeStrategyProvider(_Default()).generate(request)

    assert len(artifact.candidates[0].keyframes) == 4
    assert artifact.candidates[0].metadata["from_ee"] is None
