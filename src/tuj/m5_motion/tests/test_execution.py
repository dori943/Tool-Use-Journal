from __future__ import annotations

import json
from dataclasses import dataclass, field

from tuj.m5_motion.execution import (
    ExecutionAcceptance,
    GoalEvaluation,
    GoalEvaluationStatus,
    SelectedPlanSimulationOrchestrator,
    SequenceExecutionStatus,
    SimulationArtifactStore,
)
from tuj.m5_motion.orchestration import SelectedPlanPlanningResult
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    ExecutionReport,
    ExecutionStatus,
    FailureObservation,
    GoalType,
    InterpolationType,
    ModuleName,
    MotionConstraints,
    MotionGoal,
    MotionPlan,
    MotionPlanRequest,
    MotionTask,
    RobotState,
    SceneRef,
    SegmentType,
    SimulationConfig,
    SimulationMetrics,
    TrajectorySegment,
    TrajectoryWaypoint,
    WorldSnapshot,
)
from tuj.m4_taskplanner.models import GraspSpec


def _provenance(artifact_id: str, artifact_type: str, module: ModuleName):
    return ArtifactProvenance(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        produced_by=module,
        invocation_id=f"invocation:{artifact_id}",
    )


def _world(position: float, *, signature: str) -> WorldSnapshot:
    return WorldSnapshot(
        scene=SceneRef(signature=signature),
        robot_state=RobotState(
            robot_id="robot",
            joint_names=["j1"],
            joint_positions_rad=[position],
        ),
    )


def _request(
    index: int,
    *,
    start: float,
    action_type: str = "MOVE",
    target: float | None = None,
) -> MotionPlanRequest:
    is_pick = action_type.upper() == "PICK"
    goal = (
        MotionGoal(goal_type=GoalType.POSE, target_object_id="part")
        if is_pick
        else
        MotionGoal(
            goal_type=GoalType.JOINT,
            target_joint_positions_rad=[target if target is not None else start + 0.1],
        )
    )
    return MotionPlanRequest(
        request_id=f"request-{index}",
        provenance=_provenance(
            f"request-artifact-{index}", "MotionPlanRequest", ModuleName.TASK_PLANNER
        ),
        world=_world(start, signature=f"scene-{index}"),
        task=MotionTask(
            task_id=f"task-{index}",
            subgoal_id=f"subgoal-{index}",
            action_type=action_type,
            ee="2F",
            target_ids=["part"] if is_pick else [],
            grasp=(
                GraspSpec(
                    grasp_id="grasp-part",
                    owner_kind="object",
                    owner_id="part",
                )
                if is_pick
                else None
            ),
            goal=goal,
        ),
        constraints=MotionConstraints(),
    )


def _plan(
    request: MotionPlanRequest,
    *,
    start: float,
    end: float,
    attached_object_id: str | None = None,
) -> MotionPlan:
    return MotionPlan(
        plan_id=f"plan:{request.request_id}",
        request_id=request.request_id,
        provenance=_provenance(
            f"plan-artifact:{request.request_id}",
            "MotionPlan",
            ModuleName.MOTION_PLANNER,
        ),
        scene_signature=request.world.scene.signature,
        robot_id="robot",
        joint_names=["j1"],
        duration_s=0.1,
        segments=[
            TrajectorySegment(
                segment_id=f"segment:{request.request_id}",
                segment_type=SegmentType.CUSTOM,
                start_time_s=0.0,
                end_time_s=0.1,
                interpolation=InterpolationType.LINEAR,
                waypoints=[
                    TrajectoryWaypoint(
                        time_from_start_s=0.0,
                        joint_positions_rad=[start],
                    ),
                    TrajectoryWaypoint(
                        time_from_start_s=0.1,
                        joint_positions_rad=[end],
                    ),
                ],
                collision_checked=True,
            )
        ],
        expected_final_state=RobotState(
            robot_id="robot",
            joint_names=["j1"],
            joint_positions_rad=[end],
            attached_object_id=attached_object_id,
        ),
    )


def _planning(*pairs: tuple[MotionPlanRequest, MotionPlan]):
    return SelectedPlanPlanningResult(
        requests=tuple(pair[0] for pair in pairs),
        plans=tuple(pair[1] for pair in pairs),
        final_world=pairs[-1][0].world,
    )


@dataclass
class _FakePlayer:
    final_states: dict[str, RobotState] = field(default_factory=dict)
    failures: set[str] = field(default_factory=set)
    run_ids: list[str] = field(default_factory=list)

    def execute(self, run, *, report_id=None):
        self.run_ids.append(run.run_id)
        failed = run.plan.plan_id in self.failures
        return ExecutionReport(
            report_id=report_id or f"report:{run.run_id}",
            run_id=run.run_id,
            plan_id=run.plan.plan_id,
            provenance=_provenance(
                f"report-artifact:{run.run_id}",
                "ExecutionReport",
                ModuleName.SIMULATOR,
            ),
            status=ExecutionStatus.FAILED if failed else ExecutionStatus.SUCCESS,
            final_robot_state=(
                None
                if failed
                else self.final_states.get(
                    run.plan.plan_id, run.plan.expected_final_state
                )
            ),
            metrics=SimulationMetrics(
                executed_duration_s=run.plan.duration_s,
                max_joint_tracking_error_rad=0.0,
            ),
            failure=(
                FailureObservation(
                    code="TEST_FAILURE",
                    category="SIMULATION_RUNTIME",
                    message="synthetic execution failure",
                )
                if failed
                else None
            ),
            metadata={"final_active_ee": "2F"},
        )


def test_sequence_executes_in_order_and_persists_goal_aware_manifest(tmp_path) -> None:
    request_0 = _request(0, start=0.0, target=0.1)
    request_1 = _request(1, start=0.1, target=0.2)
    plan_0 = _plan(request_0, start=0.0, end=0.1)
    plan_1 = _plan(request_1, start=0.1, end=0.2)
    player = _FakePlayer()
    orchestrator = SelectedPlanSimulationOrchestrator(
        player,
        config=SimulationConfig(),
        store=SimulationArtifactStore(tmp_path),
    )

    result = orchestrator.execute(
        _planning((request_0, plan_0), (request_1, plan_1))
    )

    assert result.status is SequenceExecutionStatus.SUCCESS
    assert result.successful is True
    assert len(result.runs) == len(result.reports) == 2
    assert [goal.status for goal in result.goal_evaluations] == [
        GoalEvaluationStatus.SATISFIED,
        GoalEvaluationStatus.SATISFIED,
    ]
    assert result.reports[0].metadata["goal_evaluation"]["status"] == "SATISFIED"
    assert result.final_world.robot_state.joint_positions_rad == [0.2]
    assert result.manifest_path is not None
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "SUCCESS"
    assert manifest["successful"] is True
    assert len(manifest["run_files"]) == 2
    assert all(path.is_file() for path in result.report_paths)


def test_execution_success_does_not_hide_a_failed_pick_goal() -> None:
    request = _request(0, start=0.0, action_type="PICK")
    plan = _plan(request, start=0.0, end=0.1, attached_object_id=None)

    result = SelectedPlanSimulationOrchestrator(
        _FakePlayer(), config=SimulationConfig()
    ).execute(_planning((request, plan)))

    assert result.reports[0].status is ExecutionStatus.SUCCESS
    assert result.status is SequenceExecutionStatus.GOAL_FAILED
    assert result.successful is False
    assert result.goal_evaluations[0].status is GoalEvaluationStatus.FAILED
    assert "not attached" in result.detail


def test_sequence_stops_after_execution_failure() -> None:
    request_0 = _request(0, start=0.0, target=0.1)
    request_1 = _request(1, start=0.1, target=0.2)
    plan_0 = _plan(request_0, start=0.0, end=0.1)
    plan_1 = _plan(request_1, start=0.1, end=0.2)
    player = _FakePlayer(failures={plan_0.plan_id})

    result = SelectedPlanSimulationOrchestrator(
        player, config=SimulationConfig()
    ).execute(_planning((request_0, plan_0), (request_1, plan_1)))

    assert result.status is SequenceExecutionStatus.EXECUTION_FAILED
    assert result.failed_index == 0
    assert len(result.runs) == 1
    assert len(player.run_ids) == 1


def test_sequence_rejects_large_observed_state_drift_before_next_plan() -> None:
    request_0 = _request(0, start=0.0, target=0.1)
    request_1 = _request(1, start=0.1, target=0.2)
    plan_0 = _plan(request_0, start=0.0, end=0.1)
    plan_1 = _plan(request_1, start=0.1, end=0.2)
    drifted = plan_0.expected_final_state.model_copy(
        update={"joint_positions_rad": [0.15]}
    )
    player = _FakePlayer(final_states={plan_0.plan_id: drifted})
    acceptance = ExecutionAcceptance(
        max_start_joint_error_rad=0.01,
        max_final_joint_error_rad=0.10,
        require_goal_verification=False,
    )

    result = SelectedPlanSimulationOrchestrator(
        player,
        config=SimulationConfig(),
        acceptance=acceptance,
    ).execute(_planning((request_0, plan_0), (request_1, plan_1)))

    assert result.status is SequenceExecutionStatus.STATE_DIVERGED
    assert result.failed_index == 1
    assert len(result.reports) == 1
    assert "starts" in result.detail


def test_sequence_level_goal_can_reject_individually_successful_plans() -> None:
    request = _request(0, start=0.0, target=0.1)
    plan = _plan(request, start=0.0, end=0.1)

    def final_goal(planning, reports, world):
        del planning, reports, world
        return GoalEvaluation(
            request_id="sequence",
            goal_type="TASK",
            status=GoalEvaluationStatus.FAILED,
            detail="only 2 of 12 blocks reached the collection zone",
            observed={"inside": 2, "total": 12},
        )

    result = SelectedPlanSimulationOrchestrator(
        _FakePlayer(),
        config=SimulationConfig(),
        sequence_goal_evaluator=final_goal,
    ).execute(_planning((request, plan)))

    assert result.reports[0].status is ExecutionStatus.SUCCESS
    assert result.status is SequenceExecutionStatus.GOAL_FAILED
    assert result.goal_evaluations[-1].request_id == "sequence"
    assert result.successful is False
