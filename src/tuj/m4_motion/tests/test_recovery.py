from __future__ import annotations

import pytest

from tuj.m4_motion.execution import (
    GoalEvaluation,
    GoalEvaluationStatus,
    SelectedPlanExecutionResult,
    SequenceExecutionStatus,
)
from tuj.m4_motion.orchestration import SelectedPlanPlanningResult
from tuj.m4_motion.recovery import (
    RecoveryAttributionError,
    RecoveryExecutionError,
    RecoveryOrchestrator,
)
from tuj.m4_motion.schema import (
    ArtifactProvenance,
    ExecutionReport,
    ExecutionStatus,
    FailureObservation,
    GoalType,
    ModuleName,
    MotionGoal,
    MotionPlan,
    MotionPlanRequest,
    MotionTask,
    RobotState,
    SceneRef,
    SegmentType,
    SimulationConfig,
    SimulationMetrics,
    SimulationRun,
    TrajectorySegment,
    TrajectoryWaypoint,
    WorldSnapshot,
)


def _provenance(
    artifact_id: str,
    artifact_type: str,
    module: ModuleName,
    *inputs: str,
    attempt: int = 1,
) -> ArtifactProvenance:
    return ArtifactProvenance(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        produced_by=module,
        invocation_id=f"invocation:{artifact_id}",
        input_artifact_ids=list(inputs),
        attempt=attempt,
    )


def _chain(code: str, *, observed=None, plan_attempt: int = 1):
    request = MotionPlanRequest(
        request_id="request",
        provenance=_provenance(
            "request-artifact",
            "MotionPlanRequest",
            ModuleName.TASK_PLANNER,
            "selected-plan-artifact",
        ),
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
            action_type="MOVE",
            ee="2F",
            goal=MotionGoal(
                goal_type=GoalType.JOINT,
                target_joint_positions_rad=[0.1],
            ),
        ),
    )
    plan = MotionPlan(
        plan_id="plan",
        request_id=request.request_id,
        provenance=_provenance(
            "plan-artifact",
            "MotionPlan",
            ModuleName.MOTION_PLANNER,
            request.provenance.artifact_id,
            attempt=plan_attempt,
        ),
        scene_signature="scene",
        robot_id="robot",
        joint_names=["j1"],
        duration_s=0.1,
        segments=[
            TrajectorySegment(
                segment_id="segment",
                segment_type=SegmentType.CUSTOM,
                start_time_s=0.0,
                end_time_s=0.1,
                waypoints=[
                    TrajectoryWaypoint(
                        time_from_start_s=0.0, joint_positions_rad=[0.0]
                    ),
                    TrajectoryWaypoint(
                        time_from_start_s=0.1, joint_positions_rad=[0.1]
                    ),
                ],
                collision_checked=True,
            )
        ],
        expected_final_state=RobotState(
            robot_id="robot",
            joint_names=["j1"],
            joint_positions_rad=[0.1],
        ),
    )
    run = SimulationRun(
        run_id="run",
        provenance=_provenance(
            "run-artifact", "SimulationRun", ModuleName.SIMULATOR, "plan-artifact"
        ),
        plan=plan,
        config=SimulationConfig(),
    )
    report = ExecutionReport(
        report_id="report",
        run_id=run.run_id,
        plan_id=plan.plan_id,
        provenance=_provenance(
            "report-artifact",
            "ExecutionReport",
            ModuleName.SIMULATOR,
            "run-artifact",
            "plan-artifact",
        ),
        status=ExecutionStatus.FAILED,
        metrics=SimulationMetrics(
            executed_duration_s=0.05,
            max_joint_tracking_error_rad=0.1,
        ),
        failure=FailureObservation(
            code=code,
            category="SIMULATION_RUNTIME",
            message=f"failure {code}",
            observed=dict(observed or {}),
        ),
    )
    return request, plan, run, report


def test_collision_failure_restarts_motion_planner_input_only() -> None:
    request, plan, run, report = _chain("EXECUTION_COLLISION")

    directive = RecoveryOrchestrator().directive(
        request=request,
        plan=plan,
        run=run,
        report=report,
        parameter_overrides={"random_seed": 7},
    )

    assert directive.target_module is ModuleName.MOTION_PLANNER
    assert directive.restart_from_artifact_id == "request-artifact"
    assert set(directive.invalidated_artifact_ids) == {
        "plan-artifact",
        "run-artifact",
        "report-artifact",
    }
    assert directive.retry_policy.parameter_overrides == {"random_seed": 7}


def test_runtime_failure_restarts_simulation_from_the_same_plan() -> None:
    request, plan, run, report = _chain("PLAYBACK_RUNTIME_FAILED")

    directive = RecoveryOrchestrator().directive(
        request=request, plan=plan, run=run, report=report
    )

    assert directive.target_module is ModuleName.SIMULATOR
    assert directive.root_cause.artifact_id == "run-artifact"
    assert directive.restart_from_artifact_id == "plan-artifact"
    assert set(directive.invalidated_artifact_ids) == {
        "run-artifact",
        "report-artifact",
    }


def test_unknown_failure_is_not_attributed_optimistically() -> None:
    request, plan, run, report = _chain("UNCLASSIFIED_FAILURE")

    with pytest.raises(RecoveryAttributionError, match="no unambiguous"):
        RecoveryOrchestrator().directive(
            request=request, plan=plan, run=run, report=report
        )


def test_explicit_controller_evidence_can_target_registered_handler() -> None:
    request, plan, run, report = _chain(
        "UNCLASSIFIED_FAILURE",
        observed={"root_cause_module": "CONTROLLER"},
    )
    controller_artifact = _provenance(
        "controller-artifact",
        "ControllerCommand",
        ModuleName.CONTROLLER,
        "plan-artifact",
    )
    handled = []
    orchestrator = RecoveryOrchestrator(
        {ModuleName.CONTROLLER: lambda directive: handled.append(directive.directive_id)}
    )
    directive = orchestrator.directive(
        request=request,
        plan=plan,
        run=run,
        report=report,
        additional_artifacts=[controller_artifact],
    )

    result = orchestrator.execute(directive)

    assert directive.target_module is ModuleName.CONTROLLER
    assert directive.root_cause.confidence == 1.0
    assert handled == [directive.directive_id]
    assert result.value is None


def test_recovery_execution_enforces_retry_budget_and_handler_registration() -> None:
    request, plan, run, report = _chain(
        "EXECUTION_COLLISION", plan_attempt=2
    )
    orchestrator = RecoveryOrchestrator()
    exhausted = orchestrator.directive(
        request=request,
        plan=plan,
        run=run,
        report=report,
        max_attempts=2,
    )
    with pytest.raises(RecoveryExecutionError, match="budget exhausted"):
        orchestrator.execute(exhausted)

    retryable = orchestrator.directive(
        request=request,
        plan=plan,
        run=run,
        report=report,
        max_attempts=3,
    )
    with pytest.raises(RecoveryExecutionError, match="no recovery handler"):
        orchestrator.execute(retryable)


def test_sequence_level_goal_failure_restarts_task_planner() -> None:
    request, plan, run, failed_report = _chain("IGNORED")
    report = ExecutionReport.model_validate(
        {
            **failed_report.model_dump(mode="json"),
            "status": "SUCCESS",
            "final_robot_state": plan.expected_final_state.model_dump(mode="json"),
            "failure": None,
        }
    )
    planning = SelectedPlanPlanningResult(
        requests=(request,),
        plans=(plan,),
        final_world=request.world,
    )
    result = SelectedPlanExecutionResult(
        planning=planning,
        status=SequenceExecutionStatus.GOAL_FAILED,
        runs=(run,),
        reports=(report,),
        goal_evaluations=(
            GoalEvaluation(
                request_id="sequence",
                goal_type="TASK",
                status=GoalEvaluationStatus.FAILED,
                detail="only 2 of 12 blocks reached the collection zone",
                observed={"inside": 2, "total": 12},
            ),
        ),
        final_world=request.world,
        failed_index=0,
    )

    directive = RecoveryOrchestrator().directive_for_sequence(result)

    assert directive.target_module is ModuleName.TASK_PLANNER
    assert directive.restart_from_artifact_id == "selected-plan-artifact"
    assert directive.root_cause.cause_code == "GOAL_NOT_REACHED"
