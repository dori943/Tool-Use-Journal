"""Generated candidates flow through IK selection into a final MotionPlan."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tuj.m4_motion.kinematics import IKResult, IKSolutionSet
from tuj.m4_motion.pipeline import CollisionPlanningSetup, MotionPlanningPipeline
from tuj.m4_motion.schema import (
    ArtifactProvenance,
    CollisionContext,
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
)
from tuj.m4_motion.vlm_provider import (
    GeneratedKeyframe,
    GeneratedKeyframeBatch,
    GeneratedStrategy,
    OpenAIKeyframeProvider,
    OpenAIKeyframeProviderConfig,
)


def _request() -> MotionPlanRequest:
    return MotionPlanRequest(
        request_id="request-1",
        provenance=ArtifactProvenance(
            artifact_id="request-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="task-planner",
        ),
        world=WorldSnapshot(
            scene=SceneRef(signature="scene-1"),
            robot_state=RobotState(
                robot_id="robot",
                joint_names=["j1", "j2"],
                joint_positions_rad=[0.0, 0.0],
            ),
            objects={
                "target": {
                    "pose": {
                        "position_m": [0.4, 0.0, 0.2],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                }
            },
        ),
        task=MotionTask(
            task_id="task-1",
            subgoal_id="sg-1",
            action_type="MOVE",
            ee="2F",
            target_ids=["target"],
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_pose=Pose(
                    frame_id="world",
                    position_m=(0.4, 0.0, 0.2),
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


def _keyframe(identifier: str, axis: list[float]) -> GeneratedKeyframe:
    return GeneratedKeyframe(
        keyframe_id=identifier,
        keyframe_type="CUSTOM",
        frame_ref="object:target",
        anchor="center",
        approach_axis_xyz=axis,
        tool_axis_to_align="+z",
        offset_along_approach_m=0.05,
        roll_rad=0.0,
        planner="JOINT",
    )


class _FakeResponses:
    def parse(self, **kwargs):
        del kwargs
        return SimpleNamespace(
            id="resp_pipeline",
            status="completed",
            output_parsed=GeneratedKeyframeBatch(
                candidates=[
                    GeneratedStrategy(
                        strategy_id="blocked",
                        rationale="This proposal is rejected by robot validation.",
                        keyframes=[
                            _keyframe("blocked-a", [0.0, 0.0, 1.0]),
                            _keyframe("blocked-b", [0.0, 0.0, 1.0]),
                        ],
                    ),
                    GeneratedStrategy(
                        strategy_id="connected",
                        rationale="This proposal has a connected joint realization.",
                        keyframes=[
                            _keyframe("connected-a", [1.0, 0.0, 0.0]),
                            _keyframe("connected-b", [1.0, 0.0, 0.0]),
                        ],
                    ),
                ]
            ),
        )


class _FakeClient:
    responses = _FakeResponses()


class _FakeKinematics:
    def __init__(self):
        self.endpoint_seeds = []

    def solve_all_ik(self, position, orientation, **kwargs):
        del orientation
        self.endpoint_seeds.append(kwargs.get("seed_qpos"))
        q0 = round(float(position[0]), 3)
        return IKSolutionSet(
            solutions=(
                IKResult(
                    solved=True,
                    qpos=(q0, 0.2),
                    position_error_m=0.0,
                    orientation_error_rad=0.0,
                    branch_id="B1",
                ),
            ),
            enumeration_complete=True,
            attempted_seeds=1,
        )


def test_openai_candidates_are_robot_filtered_and_finalized() -> None:
    provider = OpenAIKeyframeProvider(
        OpenAIKeyframeProviderConfig(model="gpt-test", candidate_count=2),
        client=_FakeClient(),
    )
    kinematics = _FakeKinematics()
    pipeline = MotionPlanningPipeline(provider, kinematics)
    context = CollisionContext(
        context_id="default",
        active_ee="2F",
        collision_model_version="test-model",
    )

    result = pipeline.plan(
        _request(),
        state_validator=lambda q, keyframe: "blocked" not in keyframe.keyframe_id,
        collision_contexts={context.context_id: context},
        initial_collision_context_id=context.context_id,
        final_segment_validator=lambda waypoints, selected_context: bool(waypoints)
        and selected_context.context_id == "default",
    )

    assert result.compilation.connected is not None
    assert result.compilation.connected.strategy_id.endswith(":connected")
    assert len(result.compilation.attempts) == 2
    assert result.compilation.attempts[0].failure_code == "NO_VALID_IK_BRANCH"
    assert result.plan.metadata["strategy_id"].endswith(":connected")
    assert len(result.plan.segments) == 2
    assert result.plan.duration_s > 0
    assert result.plan.provenance.input_artifact_ids == [
        "request-artifact",
        result.keyframe_artifact.provenance.artifact_id,
    ]
    assert kinematics.endpoint_seeds
    assert all(seed is None for seed in kinematics.endpoint_seeds)


def test_pipeline_accepts_one_artifact_aware_collision_factory() -> None:
    provider = OpenAIKeyframeProvider(
        OpenAIKeyframeProviderConfig(model="gpt-test", candidate_count=2),
        client=_FakeClient(),
    )
    kinematics = _FakeKinematics()
    pipeline = MotionPlanningPipeline(provider, kinematics)
    context = CollisionContext(
        context_id="factory-default",
        active_ee="2F",
        collision_model_version="test-model",
    )

    class Factory:
        def prepare(self, request, artifact):
            del request
            bound = artifact.model_copy(deep=True)
            for candidate in bound.candidates:
                for keyframe in candidate.keyframes:
                    keyframe.collision_context_id = context.context_id
                    keyframe.metadata["preserve_endpoint_continuity"] = True
            return CollisionPlanningSetup(
                keyframe_artifact=bound,
                state_validator=lambda q, keyframe: True,
                collision_contexts={context.context_id: context},
                initial_collision_context_id=context.context_id,
                final_segment_validator=lambda waypoints, selected: (
                    bool(waypoints) and selected.context_id == context.context_id
                ),
            )

    result = pipeline.plan(_request(), collision_context_factory=Factory())

    assert result.plan.segments
    assert all(
        segment.collision_context_before == context
        for segment in result.plan.segments
    )
    assert kinematics.endpoint_seeds
    assert all(
        tuple(seed) == (0.0, 0.0)
        for seed in kinematics.endpoint_seeds
        if seed is not None
    )
    assert all(seed is not None for seed in kinematics.endpoint_seeds)

    with pytest.raises(ValueError, match="cannot be combined"):
        pipeline.plan(
            _request(),
            collision_context_factory=Factory(),
            state_validator=lambda q, keyframe: True,
        )
