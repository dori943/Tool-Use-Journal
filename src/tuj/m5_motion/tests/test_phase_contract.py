from __future__ import annotations

import pytest

from tuj.m5_motion.phase_contract import (
    KeyframePhaseContractError,
    validate_keyframe_phase_contract,
)
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    GoalType,
    KeyframeEventType,
    KeyframePlanArtifact,
    KeyframePlanCandidate,
    KeyframePlannerType,
    KeyframeType,
    ModuleName,
    MotionGoal,
    MotionPlanRequest,
    MotionTask,
    RelativeKeyframeSpec,
    RobotState,
    SceneRef,
    StrategyGenerationProvenance,
    StrategyGeneratorKind,
    WorldSnapshot,
)


def _provenance(identifier: str, artifact_type: str) -> ArtifactProvenance:
    return ArtifactProvenance(
        artifact_id=identifier,
        artifact_type=artifact_type,
        produced_by=ModuleName.MOTION_PLANNER,
        invocation_id="phase-test",
    )


def _request(action: str) -> MotionPlanRequest:
    return MotionPlanRequest(
        request_id="request",
        provenance=_provenance("request-artifact", "MotionPlanRequest"),
        world=WorldSnapshot(
            scene=SceneRef(signature="scene"),
            robot_state=RobotState(
                robot_id="robot",
                joint_names=["j1"],
                joint_positions_rad=[0.0],
            ),
        ),
        task=MotionTask(
            task_id="task",
            subgoal_id="subgoal",
            action_type=action,
            ee="2F",
            target_ids=["object"],
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_object_id="object",
            ),
        ),
    )


def _keyframe(
    identifier: str,
    kind: KeyframeType,
    *events: KeyframeEventType,
) -> RelativeKeyframeSpec:
    return RelativeKeyframeSpec(
        keyframe_id=identifier,
        keyframe_type=kind,
        frame_ref="world",
        anchor="center",
        approach_axis_xyz=(0.0, 0.0, 1.0),
        planner=KeyframePlannerType.CARTESIAN,
        events_after=list(events),
    )


def _artifact(*keyframes: RelativeKeyframeSpec) -> KeyframePlanArtifact:
    return KeyframePlanArtifact(
        artifact_id="keyframes",
        provenance=_provenance("keyframe-artifact", "KeyframePlanArtifact"),
        scene_signature="scene",
        subgoal_id="subgoal",
        candidates=[
            KeyframePlanCandidate(
                strategy_id="strategy",
                keyframes=list(keyframes),
                provenance=StrategyGenerationProvenance(
                    generator_kind=StrategyGeneratorKind.TEMPLATE,
                    generator_id="test",
                    input_hash="hash",
                ),
            )
        ],
    )


def test_transport_rejects_place_and_release_effects() -> None:
    artifact = _artifact(
        _keyframe("transfer", KeyframeType.TRANSFER),
        _keyframe(
            "place",
            KeyframeType.PLACE,
            KeyframeEventType.DETACH_OBJECT,
            KeyframeEventType.GRIPPER_OPEN,
        ),
    )

    with pytest.raises(KeyframePhaseContractError, match="TRANSPORT"):
        validate_keyframe_phase_contract(_request("transport"), artifact)


def test_release_requires_detach_before_open_and_retreat_after_place() -> None:
    invalid = _artifact(
        _keyframe("transfer", KeyframeType.TRANSFER),
        _keyframe(
            "place",
            KeyframeType.PLACE,
            KeyframeEventType.GRIPPER_OPEN,
            KeyframeEventType.DETACH_OBJECT,
        ),
        _keyframe("retreat", KeyframeType.RETREAT),
    )
    with pytest.raises(KeyframePhaseContractError, match="before opening"):
        validate_keyframe_phase_contract(_request("place"), invalid)

    valid = _artifact(
        _keyframe("transfer", KeyframeType.TRANSFER),
        _keyframe(
            "place",
            KeyframeType.PLACE,
            KeyframeEventType.DETACH_OBJECT,
            KeyframeEventType.GRIPPER_OPEN,
        ),
        _keyframe("retreat", KeyframeType.RETREAT),
    )
    validate_keyframe_phase_contract(_request("place"), valid)
