"""OpenAI keyframe generation stays structured, relative, and cacheable."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tuj.m4_motion.schema import (
    ArtifactProvenance,
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
    WorldSnapshot,
    KeyframeEventType,
)
from tuj.m3_taskplanner.models import GraspSpec
from tuj.m4_motion.vlm_provider import (
    GeneratedKeyframe,
    GeneratedKeyframeBatch,
    GeneratedStrategy,
    MissingOpenAIAPIKeyError,
    OpenAIKeyframeProvider,
    OpenAIKeyframeProviderConfig,
    OpenAIKeyframeProviderError,
)


def _request() -> MotionPlanRequest:
    return MotionPlanRequest(
        request_id="request-pick-bottle",
        provenance=ArtifactProvenance(
            artifact_id="request-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="task-planner-1",
        ),
        world=WorldSnapshot(
            scene=SceneRef(signature="scene-1"),
            robot_state=RobotState(
                robot_id="ur5e",
                joint_names=["j1", "j2"],
                joint_positions_rad=[0.0, 0.0],
            ),
            objects={
                "bottle": {
                    "pose": {
                        "position_m": [0.45, 0.0, 0.2],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "dimensions_m": [0.06, 0.06, 0.2],
                    "api_key": "must-not-leave-the-process",
                }
            },
            obstacles=[{"id": "table", "kind": "box"}],
        ),
        task=MotionTask(
            task_id="pick-bottle",
            subgoal_id="sg-pick",
            action_type="PICK",
            ee="2F",
            target_ids=["bottle"],
            # The provider tests proposal generation; the grasp-planner-owned
            # structured grasp is exercised by the contract tests.
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_pose=Pose(
                    frame_id="world",
                    position_m=(0.45, 0.0, 0.2),
                    orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
            ),
        ),
        constraints=MotionConstraints(
            joint_limits={
                "j1": JointDynamicLimit(
                    max_velocity_rad_s=1.0,
                    max_acceleration_rad_s2=2.0,
                ),
                "j2": JointDynamicLimit(
                    max_velocity_rad_s=1.0,
                    max_acceleration_rad_s2=2.0,
                ),
            }
        ),
    )


def _batch(*, frame_ref: str = "object:bottle") -> GeneratedKeyframeBatch:
    return GeneratedKeyframeBatch(
        candidates=[
            GeneratedStrategy(
                strategy_id="top",
                rationale="Top approach with a vertical retreat.",
                keyframes=[
                    GeneratedKeyframe(
                        keyframe_id="pre",
                        keyframe_type="PRE_GRASP",
                        frame_ref=frame_ref,
                        anchor="top_center",
                        approach_axis_xyz=[0.0, 0.0, 1.0],
                        tool_axis_to_align="-z",
                        offset_along_approach_m=0.12,
                        roll_rad=0.0,
                        planner="CARTESIAN",
                    ),
                    GeneratedKeyframe(
                        keyframe_id="grasp",
                        keyframe_type="GRASP",
                        frame_ref=frame_ref,
                        anchor="top_center",
                        approach_axis_xyz=[0.0, 0.0, 1.0],
                        tool_axis_to_align="-z",
                        offset_along_approach_m=0.01,
                        roll_rad=0.0,
                        planner="CARTESIAN",
                    ),
                    GeneratedKeyframe(
                        keyframe_id="lift",
                        keyframe_type="LIFT",
                        frame_ref=frame_ref,
                        anchor="top_center",
                        approach_axis_xyz=[0.0, 0.0, 1.0],
                        tool_axis_to_align="-z",
                        offset_along_approach_m=0.15,
                        roll_rad=0.0,
                        planner="CARTESIAN",
                    ),
                ],
            ),
            GeneratedStrategy(
                strategy_id="side",
                rationale="Side approach provides a distinct IK family.",
                keyframes=[
                    GeneratedKeyframe(
                        keyframe_id="pre",
                        keyframe_type="PRE_GRASP",
                        frame_ref=frame_ref,
                        anchor="center",
                        approach_axis_xyz=[1.0, 0.0, 0.0],
                        tool_axis_to_align="-z",
                        offset_along_approach_m=0.12,
                        roll_rad=1.57079632679,
                        planner="SAMPLING_BASED",
                    ),
                    GeneratedKeyframe(
                        keyframe_id="grasp",
                        keyframe_type="GRASP",
                        frame_ref=frame_ref,
                        anchor="center",
                        approach_axis_xyz=[1.0, 0.0, 0.0],
                        tool_axis_to_align="-z",
                        offset_along_approach_m=0.01,
                        roll_rad=1.57079632679,
                        planner="CARTESIAN",
                    ),
                    GeneratedKeyframe(
                        keyframe_id="retreat",
                        keyframe_type="RETREAT",
                        frame_ref=frame_ref,
                        anchor="center",
                        approach_axis_xyz=[1.0, 0.0, 0.0],
                        tool_axis_to_align="-z",
                        offset_along_approach_m=0.15,
                        roll_rad=1.57079632679,
                        planner="CARTESIAN",
                    ),
                ],
            ),
        ]
    )


class _FakeResponses:
    def __init__(self, parsed) -> None:
        self.parsed = parsed
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp_test_123",
            status="completed",
            output_parsed=self.parsed,
        )


class _FakeClient:
    def __init__(self, parsed) -> None:
        self.responses = _FakeResponses(parsed)


def test_structured_openai_response_becomes_validated_artifact(tmp_path) -> None:
    client = _FakeClient(_batch())
    provider = OpenAIKeyframeProvider(
        OpenAIKeyframeProviderConfig(
            model="gpt-test",
            candidate_count=2,
            cache_dir=tmp_path,
        ),
        client=client,
    )

    artifact = provider.generate(_request())

    assert artifact.scene_signature == "scene-1"
    assert artifact.subgoal_id == "sg-pick"
    assert len(artifact.candidates) == 2
    assert artifact.candidates[0].provenance.model_id == "gpt-test"
    assert artifact.candidates[0].provenance.provider_request_id == "resp_test_123"
    call = client.responses.calls[0]
    assert call["text_format"] is GeneratedKeyframeBatch
    assert call["store"] is False
    assert "must-not-leave-the-process" not in call["input"]


def test_identical_request_reuses_frozen_artifact_cache(tmp_path) -> None:
    client = _FakeClient(_batch())
    provider = OpenAIKeyframeProvider(
        OpenAIKeyframeProviderConfig(
            model="gpt-test",
            candidate_count=2,
            cache_dir=tmp_path,
        ),
        client=client,
    )

    first = provider.generate(_request())
    second = provider.generate(_request())

    assert second == first
    assert len(client.responses.calls) == 1


def test_unknown_generated_frame_fails_before_ik() -> None:
    provider = OpenAIKeyframeProvider(
        OpenAIKeyframeProviderConfig(model="gpt-test", candidate_count=2),
        client=_FakeClient(_batch(frame_ref="object:not-in-scene")),
    )

    with pytest.raises(OpenAIKeyframeProviderError, match="unknown object frame"):
        provider.generate(_request())


def test_invalid_candidate_is_dropped_when_another_candidate_is_valid() -> None:
    batch = _batch()
    invalid_keyframes = list(batch.candidates[0].keyframes)
    invalid_keyframes[0] = invalid_keyframes[0].model_copy(
        update={"frame_ref": "object:not-in-scene"}
    )
    batch = batch.model_copy(
        update={
            "candidates": [
                batch.candidates[0].model_copy(
                    update={"keyframes": invalid_keyframes}
                ),
                batch.candidates[1],
            ]
        }
    )
    provider = OpenAIKeyframeProvider(
        OpenAIKeyframeProviderConfig(model="gpt-test", candidate_count=2),
        client=_FakeClient(batch),
    )

    artifact = provider.generate(_request())

    assert [candidate.strategy_id for candidate in artifact.candidates] == [
        "sg-pick:side"
    ]
    assert artifact.provenance.metadata["rejected_candidate_count"] == 1
    assert "unknown object frame" in artifact.provenance.metadata[
        "rejected_candidates"
    ][0]


def test_missing_api_key_fails_without_network(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIKeyframeProvider(
        OpenAIKeyframeProviderConfig(model="gpt-test", candidate_count=2)
    )

    with pytest.raises(MissingOpenAIAPIKeyError):
        provider.generate(_request())


def test_environment_configuration_uses_safe_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_KEYFRAME_MODEL", "gpt-env-test")
    monkeypatch.setenv("MOTION_PLANNER_KEYFRAME_CACHE", str(tmp_path))

    config = OpenAIKeyframeProviderConfig.from_environment(candidate_count=2)

    assert config.model == "gpt-env-test"
    assert config.cache_dir == tmp_path


def test_pick_keyframe_gets_deterministic_grasp_and_attach_events() -> None:
    request = _request()
    request.task.grasp = GraspSpec(
        grasp_id="grasp-1",
        owner_kind="object",
        owner_id="bottle",
    )
    request.task.goal = MotionGoal(
        goal_type=GoalType.POSE,
        target_object_id="bottle",
    )
    provider = OpenAIKeyframeProvider(
        OpenAIKeyframeProviderConfig(model="gpt-test", candidate_count=2),
        client=_FakeClient(_batch()),
    )

    artifact = provider.generate(request)

    grasp = artifact.candidates[0].keyframes[1]
    assert grasp.events_after == [
        KeyframeEventType.GRIPPER_CLOSE,
        KeyframeEventType.ATTACH_OBJECT,
    ]
    assert grasp.metadata["event_target_id"] == "bottle"
