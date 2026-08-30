"""Opt-in network smoke test for the real OpenAI Responses API."""

from __future__ import annotations

import os

import pytest

from tuj.m4_motion.schema import (
    ArtifactProvenance,
    GoalType,
    ModuleName,
    MotionGoal,
    MotionPlanRequest,
    MotionTask,
    Pose,
    RobotState,
    SceneRef,
    WorldSnapshot,
)
from tuj.m4_motion.vlm_provider import (
    OpenAIKeyframeProvider,
    OpenAIKeyframeProviderConfig,
)


_LIVE = bool(os.environ.get("OPENAI_API_KEY")) and os.environ.get(
    "RUN_OPENAI_LIVE_TEST"
) == "1"


@pytest.mark.skipif(
    not _LIVE,
    reason="set OPENAI_API_KEY and RUN_OPENAI_LIVE_TEST=1 for the network smoke test",
)
def test_real_openai_keyframe_generation() -> None:
    request = MotionPlanRequest(
        request_id="live-openai-keyframe-smoke",
        provenance=ArtifactProvenance(
            artifact_id="live-request-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="live-smoke",
        ),
        world=WorldSnapshot(
            scene=SceneRef(signature="live-scene-v1"),
            robot_state=RobotState(
                robot_id="ur5e",
                joint_names=[f"joint_{index}" for index in range(6)],
                joint_positions_rad=[0.0, -1.2, 1.4, -1.7, -1.2, 0.0],
            ),
            objects={
                "bottle": {
                    "pose": {
                        "position_m": [0.45, 0.0, 0.2],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "dimensions_m": [0.06, 0.06, 0.20],
                }
            },
            obstacles=[
                {
                    "id": "table",
                    "shape": "box",
                    "center_m": [0.45, 0.0, -0.025],
                    "size_m": [0.8, 0.8, 0.05],
                }
            ],
        ),
        task=MotionTask(
            task_id="live-pick",
            subgoal_id="live-pick-bottle",
            action_type="PICK",
            ee="2F",
            target_ids=["bottle"],
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_pose=Pose(
                    frame_id="world",
                    position_m=(0.45, 0.0, 0.2),
                    orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
            ),
        ),
    )
    config = OpenAIKeyframeProviderConfig.from_environment(candidate_count=2)

    artifact = OpenAIKeyframeProvider(config).generate(request)

    assert len(artifact.candidates) == 2
    assert all(len(candidate.keyframes) >= 2 for candidate in artifact.candidates)
    assert all(
        keyframe.frame_ref in {"world", "object:bottle"}
        for candidate in artifact.candidates
        for keyframe in candidate.keyframes
    )
