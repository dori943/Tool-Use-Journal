"""Generated candidates flow through IK selection into a final MotionPlan."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tuj.m5_motion.kinematics import IKResult, IKSolutionSet
from tuj.m5_motion.pipeline import (
    CollisionPlanningSetup,
    DebugKeyframeStrategyProvider,
    MotionPlanningPipeline,
    MotionPlanningPipelineError,
    _EdgeContextMaterializingPlanner,
)
from tuj.m5_motion.strategy import EdgePlanResult
from tuj.m5_motion.plan_builder import MotionPlanBuilder
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    CollisionContext,
    GoalType,
    JointDynamicLimit,
    KeyframeEventType,
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
from tuj.m5_motion.vlm_provider import (
    GeneratedKeyframe,
    GeneratedKeyframeBatch,
    GeneratedStrategy,
    OpenAIKeyframeProvider,
    OpenAIKeyframeProviderConfig,
)
from tuj.m5_motion.trajectory_processing import TrajectoryProcessingError


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

    @staticmethod
    def forward_pose_world(qpos):
        return (tuple(qpos[:2]) + (0.3,), (0.0, 0.0, 0.0, 1.0))


class _OutsideReachKinematics:
    def solve_all_ik(self, position, orientation, **kwargs):
        del position, orientation, kwargs
        return IKSolutionSet(
            best_position_error_m=0.25,
            best_orientation_error_rad=0.0,
            attempted_seeds=0,
            failure_code="TARGET_OUTSIDE_CONSERVATIVE_REACH",
            detail="target is 0.25 m beyond the conservative reach envelope",
        )


class _MixedFailureKinematics(_FakeKinematics):
    def solve_all_ik(self, position, orientation, **kwargs):
        if float(position[0]) > 0.44:
            return IKSolutionSet(
                attempted_seeds=8,
                failure_code="IK_NUMERICAL_NON_CONVERGENCE",
                detail="no full-pose IK solution converged from deterministic seeds",
            )
        return super().solve_all_ik(position, orientation, **kwargs)


class _CollisionRejectingValidator:
    def __call__(self, qpos, keyframe):
        del qpos, keyframe
        return False

    def check(self, qpos, keyframe):
        del qpos, keyframe
        return SimpleNamespace(
            valid=False,
            failure_code="COLLISION_MARGIN_VIOLATION",
            detail="held object intersects packing-box wall",
        )


class _RepairAwareCollisionValidator:
    @staticmethod
    def _valid(keyframe) -> bool:
        return "repaired" in keyframe.keyframe_id

    def __call__(self, qpos, keyframe):
        del qpos
        return self._valid(keyframe)

    def check(self, qpos, keyframe):
        del qpos
        valid = self._valid(keyframe)
        return SimpleNamespace(
            valid=valid,
            failure_code=None if valid else "COLLISION_MARGIN_VIOLATION",
            detail=(
                ""
                if valid
                else "fixture_geom_a <-> fixture_geom_b clearance 0.001000 m "
                "is below required 0.005000 m"
            ),
        )


class _UnstructuredCollisionValidator:
    def __call__(self, qpos, keyframe):
        del qpos, keyframe
        return False

    def check(self, qpos, keyframe):
        del qpos, keyframe
        return SimpleNamespace(
            valid=False,
            failure_code="COLLISION_MARGIN_VIOLATION",
            detail="collision validator returned no numeric observation",
        )


class _FeedbackAwareProvider:
    supports_collision_feedback = True

    def __init__(self, *, repair_on_attempt: int | None = 1) -> None:
        self.calls = []
        self.repair_on_attempt = repair_on_attempt
        self.delegate = OpenAIKeyframeProvider(
            OpenAIKeyframeProviderConfig(model="gpt-test", candidate_count=2),
            client=_FakeClient(),
        )

    def generate(self, request):
        feedback = request.task.metadata.get("collision_repair_feedback")
        self.calls.append(feedback)
        artifact = self.delegate.generate(request)
        if feedback is None or self.repair_on_attempt is None:
            return artifact
        if int(feedback["repair_attempt"]) < self.repair_on_attempt:
            return artifact
        repaired = artifact.model_copy(deep=True)
        for candidate in repaired.candidates:
            for keyframe in candidate.keyframes:
                keyframe.keyframe_id = f"repaired:{keyframe.keyframe_id}"
        return repaired


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


def test_pipeline_preserves_raw_ik_diagnostics_and_reach_failure_code() -> None:
    provider = OpenAIKeyframeProvider(
        OpenAIKeyframeProviderConfig(model="gpt-test", candidate_count=2),
        client=_FakeClient(),
    )
    pipeline = MotionPlanningPipeline(provider, _OutsideReachKinematics())
    context = CollisionContext(
        context_id="default",
        active_ee="2F",
        collision_model_version="test-model",
    )

    with pytest.raises(MotionPlanningPipelineError) as caught:
        pipeline.plan(
            _request(),
            state_validator=lambda q, keyframe: True,
            collision_contexts={context.context_id: context},
            initial_collision_context_id=context.context_id,
            final_segment_validator=lambda waypoints, selected_context: True,
        )

    attempt = caught.value.compilation.attempts[0]
    diagnostic = attempt.ik_diagnostics[0]
    assert attempt.failure_code == "TARGET_OUTSIDE_REACH_ENVELOPE"
    assert diagnostic.raw_ik_count == 0
    assert diagnostic.valid_ik_count == 0
    assert diagnostic.attempted_seeds == 0
    assert diagnostic.solver_failure_code == "TARGET_OUTSIDE_CONSERVATIVE_REACH"
    assert diagnostic.best_position_error_m == pytest.approx(0.25)
    assert "conservative reach envelope" in diagnostic.solver_detail
    assert "conservative reach envelope" in diagnostic.validity_detail


def test_pipeline_debug_dump_preserves_generated_bound_and_failed_pose(tmp_path) -> None:
    provider = OpenAIKeyframeProvider(
        OpenAIKeyframeProviderConfig(model="gpt-test", candidate_count=2),
        client=_FakeClient(),
    )
    provider = DebugKeyframeStrategyProvider(provider, tmp_path)
    pipeline = MotionPlanningPipeline(
        provider, _OutsideReachKinematics(), debug_dir=tmp_path
    )
    context = CollisionContext(
        context_id="default",
        active_ee="2F",
        collision_model_version="test-model",
    )

    with pytest.raises(MotionPlanningPipelineError):
        pipeline.plan(
            _request(),
            state_validator=lambda q, keyframe: True,
            collision_contexts={context.context_id: context},
            initial_collision_context_id=context.context_id,
            final_segment_validator=lambda waypoints, selected_context: True,
        )

    generated_path, planning_path = sorted(tmp_path.glob("*.json"))
    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    planning = json.loads(planning_path.read_text(encoding="utf-8"))
    assert generated["stage"] == "VLM_GENERATED"
    raw = generated["strategies"][0]["keyframes"][0]
    assert raw["index"] == 0
    assert raw["phase"] == "CUSTOM"
    assert raw["frame_ref"] == "object:target"
    assert raw["raw_offset_m"] == pytest.approx(0.05)
    assert raw["raw_position_world_m"] == pytest.approx([0.4, 0.0, 0.25])

    assert planning["stage"] == "BINDER_AND_COMPILER"
    assert planning["binder_strategies"][0]["keyframes"][0][
        "resolver_world_pose"
    ]["position_m"] == pytest.approx([0.4, 0.0, 0.25])
    assert planning["final_strategies"][0]["keyframes"][0][
        "final_tcp_world_pose"
    ]["position_m"] == pytest.approx([0.4, 0.0, 0.25])
    failure = planning["compilation_attempts"][0]
    assert failure["failure_code"] == "TARGET_OUTSIDE_REACH_ENVELOPE"
    assert failure["failed_keyframe_id"] == raw["keyframe_id"]
    assert failure["failed_keyframe_index"] == 0
    assert failure["failed_keyframe_phase"] == "CUSTOM"
    assert failure["resolved_final_tcp_keyframes"][0][
        "final_tcp_world_pose"
    ]["position_m"] == pytest.approx([0.4, 0.0, 0.25])


def test_attachment_context_materializes_from_selected_source_branch_fk() -> None:
    observed = []

    class Delegate:
        def plan(self, source, target, source_keyframe, target_keyframe):
            del source_keyframe, target_keyframe
            return EdgePlanResult(valid=True, joint_path=(source, target))

    wrapper = _EdgeContextMaterializingPlanner(
        Delegate(),
        _FakeKinematics(),
        lambda keyframe, pose: observed.append((keyframe.keyframe_id, pose)),
    )
    source_keyframe = SimpleNamespace(
        keyframe_id="actual-grasp",
        events_after=[KeyframeEventType.ATTACH_OBJECT],
    )
    target_keyframe = SimpleNamespace()

    result = wrapper.plan(
        (0.397, 0.002),
        (0.5, 0.1),
        source_keyframe,
        target_keyframe,
    )

    assert result.valid
    assert observed[0][0] == "actual-grasp"
    assert observed[0][1].position_m == pytest.approx((0.397, 0.002, 0.3))


def test_pipeline_distinguishes_collision_filtered_ik_branches() -> None:
    provider = OpenAIKeyframeProvider(
        OpenAIKeyframeProviderConfig(model="gpt-test", candidate_count=2),
        client=_FakeClient(),
    )
    pipeline = MotionPlanningPipeline(provider, _FakeKinematics())
    context = CollisionContext(
        context_id="default",
        active_ee="2F",
        collision_model_version="test-model",
    )

    with pytest.raises(MotionPlanningPipelineError) as caught:
        pipeline.plan(
            _request(),
            state_validator=_CollisionRejectingValidator(),
            collision_contexts={context.context_id: context},
            initial_collision_context_id=context.context_id,
            final_segment_validator=lambda waypoints, selected_context: True,
        )

    attempt = caught.value.compilation.attempts[0]
    assert attempt.failure_code == "COLLISION_FILTERED_ALL"
    assert attempt.ik_diagnostics[0].raw_ik_count == 1
    assert attempt.ik_diagnostics[0].valid_ik_count == 0


def test_pipeline_uses_structured_collision_feedback_repair_batch() -> None:
    provider = _FeedbackAwareProvider()
    original = _request()
    pipeline = MotionPlanningPipeline(provider, _FakeKinematics())
    context = CollisionContext(
        context_id="default",
        active_ee="2F",
        collision_model_version="test-model",
    )

    result = pipeline.plan(
        original,
        state_validator=_RepairAwareCollisionValidator(),
        collision_contexts={context.context_id: context},
        initial_collision_context_id=context.context_id,
        final_segment_validator=lambda waypoints, selected_context: True,
    )

    assert len(provider.calls) == 2
    assert provider.calls[0] is None
    feedback = provider.calls[1]
    assert feedback["contract_version"] == "COLLISION_REPAIR_V1"
    assert feedback["repair_attempt"] == 1
    assert feedback["maximum_repair_attempts"] == 2
    assert feedback["required_collision_margin_m"] == pytest.approx(0.005)
    assert feedback["failed_strategies"]
    observation = feedback["failed_strategies"][0][
        "collision_observations"
    ][0]
    assert observation == {
        "geometry_a": "fixture_geom_a",
        "geometry_b": "fixture_geom_b",
        "measured_clearance_m": pytest.approx(0.001),
        "required_clearance_m": pytest.approx(0.005),
    }
    assert feedback["failed_strategies"][0]["source_repair_attempt"] == 0
    assert "collision_repair_feedback" not in original.task.metadata
    assert result.compilation.connected is not None
    assert len(result.compilation.attempts) == 3
    assert all(
        attempt.failure_code == "COLLISION_FILTERED_ALL"
        for attempt in result.compilation.attempts[:2]
    )
    assert result.keyframe_artifact.candidates[0].provenance.attempt_index == 2


def test_pipeline_repairs_collision_candidates_in_mixed_failure_batch() -> None:
    provider = _FeedbackAwareProvider()
    pipeline = MotionPlanningPipeline(provider, _MixedFailureKinematics())
    context = CollisionContext(
        context_id="default",
        active_ee="2F",
        collision_model_version="test-model",
    )

    result = pipeline.plan(
        _request(),
        state_validator=_RepairAwareCollisionValidator(),
        collision_contexts={context.context_id: context},
        initial_collision_context_id=context.context_id,
        final_segment_validator=lambda waypoints, selected_context: True,
    )

    assert len(provider.calls) == 2
    feedback = provider.calls[1]
    assert feedback is not None
    assert [
        item["strategy_id"] for item in feedback["failed_strategies"]
    ] == ["sg-1:blocked"]
    assert feedback["failed_strategies"][0]["collision_observations"]
    assert all(
        "connected" not in item["strategy_id"]
        for item in feedback["failed_strategies"]
    )
    assert [
        attempt.failure_code for attempt in result.compilation.attempts[:2]
    ] == ["COLLISION_FILTERED_ALL", "IK_SEARCH_EXHAUSTED"]
    assert result.compilation.connected is not None


def test_pipeline_does_not_repair_without_numeric_collision_observation() -> None:
    provider = _FeedbackAwareProvider()
    pipeline = MotionPlanningPipeline(provider, _FakeKinematics())
    context = CollisionContext(
        context_id="default",
        active_ee="2F",
        collision_model_version="test-model",
    )

    with pytest.raises(MotionPlanningPipelineError):
        pipeline.plan(
            _request(),
            state_validator=_UnstructuredCollisionValidator(),
            collision_contexts={context.context_id: context},
            initial_collision_context_id=context.context_id,
            final_segment_validator=lambda waypoints, selected_context: True,
        )

    assert provider.calls == [None]


def test_pipeline_second_repair_uses_bounded_cumulative_feedback() -> None:
    provider = _FeedbackAwareProvider(repair_on_attempt=2)
    original = _request()
    original_copy = original.model_copy(deep=True)
    pipeline = MotionPlanningPipeline(provider, _FakeKinematics())
    context = CollisionContext(
        context_id="default",
        active_ee="2F",
        collision_model_version="test-model",
    )

    result = pipeline.plan(
        original,
        state_validator=_RepairAwareCollisionValidator(),
        collision_contexts={context.context_id: context},
        initial_collision_context_id=context.context_id,
        final_segment_validator=lambda waypoints, selected_context: True,
    )

    assert len(provider.calls) == 3
    second_feedback = provider.calls[2]
    assert second_feedback["repair_attempt"] == 2
    assert second_feedback["maximum_repair_attempts"] == 2
    assert len(second_feedback["failed_strategies"]) == 4
    assert {
        strategy["source_repair_attempt"]
        for strategy in second_feedback["failed_strategies"]
    } == {0, 1}
    assert len(second_feedback["failed_strategies"]) <= 8
    assert original == original_copy
    assert "collision_repair_feedback" not in original.task.metadata
    assert result.keyframe_artifact.candidates[0].provenance.attempt_index == 3


def test_pipeline_collision_feedback_retry_exhaustion_combines_batches() -> None:
    provider = _FeedbackAwareProvider(repair_on_attempt=None)
    pipeline = MotionPlanningPipeline(provider, _FakeKinematics())
    context = CollisionContext(
        context_id="default",
        active_ee="2F",
        collision_model_version="test-model",
    )

    with pytest.raises(
        MotionPlanningPipelineError,
        match="collision-feedback repair exhausted after 2 additional batch",
    ) as caught:
        pipeline.plan(
            _request(),
            state_validator=_RepairAwareCollisionValidator(),
            collision_contexts={context.context_id: context},
            initial_collision_context_id=context.context_id,
            final_segment_validator=lambda waypoints, selected_context: True,
        )

    assert len(provider.calls) == 3
    assert caught.value.compilation is not None
    assert len(caught.value.compilation.attempts) == 6
    assert all(
        attempt.failure_code == "COLLISION_FILTERED_ALL"
        for attempt in caught.value.compilation.attempts
    )


def test_pipeline_retries_after_connected_strategy_fails_final_validation() -> None:
    provider = OpenAIKeyframeProvider(
        OpenAIKeyframeProviderConfig(model="gpt-test", candidate_count=2),
        client=_FakeClient(),
    )
    pipeline = MotionPlanningPipeline(provider, _FakeKinematics())
    context = CollisionContext(
        context_id="default",
        active_ee="2F",
        collision_model_version="test-model",
    )

    result = pipeline.plan(
        _request(),
        state_validator=lambda q, keyframe: True,
        collision_contexts={context.context_id: context},
        initial_collision_context_id=context.context_id,
        final_segment_validator=lambda waypoints, selected_context: (
            bool(waypoints)
            and selected_context.context_id == context.context_id
            and waypoints[-1].joint_positions_rad[0] > 0.42
        ),
    )

    assert result.compilation.connected is not None
    assert result.compilation.connected.strategy_id.endswith(":connected")
    assert [attempt.failure_code for attempt in result.compilation.attempts] == [
        "FINAL_VALIDATION_FAILED",
        None,
    ]
    assert result.compilation.attempts[0].strategy_id.endswith(":blocked")


@pytest.mark.parametrize(
    "build_error",
    [
        TrajectoryProcessingError("invalid time parameterization"),
        ValueError("invalid motion plan payload"),
    ],
)
def test_pipeline_retries_after_non_build_validation_error(build_error) -> None:
    provider = OpenAIKeyframeProvider(
        OpenAIKeyframeProviderConfig(model="gpt-test", candidate_count=2),
        client=_FakeClient(),
    )
    delegate = MotionPlanBuilder()

    class FailFirstBuilder:
        def __init__(self) -> None:
            self.calls = 0

        def build(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise build_error
            return delegate.build(*args, **kwargs)

    pipeline = MotionPlanningPipeline(
        provider,
        _FakeKinematics(),
        plan_builder=FailFirstBuilder(),
    )
    context = CollisionContext(
        context_id="default",
        active_ee="2F",
        collision_model_version="test-model",
    )

    result = pipeline.plan(
        _request(),
        state_validator=lambda q, keyframe: True,
        collision_contexts={context.context_id: context},
        initial_collision_context_id=context.context_id,
        final_segment_validator=lambda waypoints, selected_context: True,
    )

    assert [attempt.failure_code for attempt in result.compilation.attempts] == [
        "FINAL_VALIDATION_FAILED",
        None,
    ]
    assert type(build_error).__name__ in result.compilation.attempts[0].detail


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
