from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from tuj.m5_motion.closed_loop_contact import (
    ClosedLoopContactExecutor,
    ClosedLoopContactStatus,
    ContactStepObservation,
)
from tuj.m5_motion.contact_evaluation import (
    CompositeGoalEvaluator,
    RegionContainmentEvaluator,
    TaskAwareGoalEvaluator,
    ToolClearanceEvaluator,
)
from tuj.m5_motion.execution import GoalEvaluationStatus
from tuj.m5_motion.profiles import (
    ContactExecutionProfile,
    PushPlanningProfile,
    TaskRecoveryProfile,
)
from tuj.m5_motion.push_to_region import (
    PushToRegionStrategyProvider,
    target_fully_inside_region,
)
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    ContactManipulationSpec,
    ContactSurfaceType,
    ExecutionReport,
    ExecutionStatus,
    GoalType,
    ModuleName,
    MotionGoal,
    MotionPlanRequest,
    MotionTask,
    Pose,
    RobotState,
    SceneRef,
    SimulationMetrics,
    StrategyGeneratorKind,
    WorldSnapshot,
)
from tuj.m5_motion.tool_affordance import (
    CircularPlateAffordanceProvider,
    select_contact_patch,
)


def _provenance(identifier: str, artifact_type: str) -> ArtifactProvenance:
    return ArtifactProvenance(
        artifact_id=identifier,
        artifact_type=artifact_type,
        produced_by=ModuleName.MOTION_PLANNER,
        invocation_id="contact-test",
    )


def _world(*, target_x_m: float = 0.24) -> WorldSnapshot:
    return WorldSnapshot(
        scene=SceneRef(signature="contact-scene"),
        robot_state=RobotState(
            robot_id="robot",
            joint_names=["j1"],
            joint_positions_rad=[0.0],
        ),
        objects={
            "plate": {
                "pose": {
                    "frame_id": "world",
                    "position_m": [0.0, -0.25, 0.08],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "dimensions_m": [0.24, 0.24, 0.02],
                "collision_geometry_refs": ["plate_collision"],
            },
            "block": {
                "pose": {
                    "frame_id": "world",
                    "position_m": [target_x_m, 0.0, 0.02],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "dimensions_m": [0.02, 0.02, 0.02],
            },
            "region": {
                "pose": {
                    "frame_id": "world",
                    "position_m": [0.0, 0.0, 0.005],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "dimensions_m": [0.20, 0.20, 0.01],
            },
        },
    )


def _request(*, target_x_m: float = 0.24) -> MotionPlanRequest:
    return MotionPlanRequest(
        request_id="contact-request",
        provenance=_provenance("contact-request-artifact", "MotionPlanRequest"),
        world=_world(target_x_m=target_x_m),
        task=MotionTask(
            task_id="contact-task",
            subgoal_id="push-block",
            action_type="push_to_region",
            ee="2F",
            tool="plate",
            target_ids=["block"],
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_region_id="region",
            ),
            contact=ContactManipulationSpec(
                primitive="PUSH",
                contact_surface=ContactSurfaceType.BROAD_FACE,
                path_pattern="RADIAL",
                maintain_contact=True,
            ),
        ),
    )


def _report() -> ExecutionReport:
    return ExecutionReport(
        report_id="report",
        run_id="run",
        plan_id="plan",
        provenance=_provenance("report-artifact", "ExecutionReport"),
        status=ExecutionStatus.SUCCESS,
        final_robot_state=RobotState(
            robot_id="robot",
            joint_names=["j1"],
            joint_positions_rad=[0.0],
        ),
        metrics=SimulationMetrics(
            executed_duration_s=1.0,
            max_joint_tracking_error_rad=0.0,
        ),
        metadata={"minimum_tool_clearance_m": 0.02},
    )


def test_circular_tool_affordance_selects_requested_surface() -> None:
    world = _world()
    patches = CircularPlateAffordanceProvider().patches("plate", world)

    patch = select_contact_patch(
        patches,
        ContactManipulationSpec(
            primitive="SWEEP",
            contact_surface=ContactSurfaceType.BROAD_FACE,
        ),
    )

    assert len(patches) == 6
    assert patch.surface_type is ContactSurfaceType.BROAD_FACE
    assert patch.collision_geometry_refs == ["plate_collision"]


def test_push_to_region_is_grounded_from_geometry_and_contact_patch() -> None:
    request = _request()
    artifact = PushToRegionStrategyProvider(
        CircularPlateAffordanceProvider(),
    ).generate(request)

    candidate = artifact.candidates[0]
    assert len(candidate.keyframes) == 4
    assert candidate.provenance.generator_kind is StrategyGeneratorKind.TASK_GEOMETRY
    assert candidate.metadata["contact_patch"]["surface_type"] == "BROAD_FACE"
    assert candidate.metadata["target_order"] == ["block"]
    assert all(
        keyframe.metadata["contact_patch_id"].startswith("plate:broad-face:")
        for keyframe in candidate.keyframes
    )
    assert sum(
        bool(record.get("virtual_reference_frame"))
        for record in request.world.objects.values()
    ) == 4


@dataclass
class _CheckpointBackend:
    checkpoints: int = 0
    restored: list[object] = field(default_factory=list)

    def checkpoint(self) -> object:
        self.checkpoints += 1
        return f"checkpoint-{self.checkpoints}"

    def restore(self, checkpoint: object) -> None:
        self.restored.append(checkpoint)


def test_closed_loop_contact_rolls_back_and_retries_with_shorter_step() -> None:
    backend = _CheckpointBackend()
    seen: list[tuple[float, int, bool]] = []

    def run_step(distance_m: float, attempt: int, reacquire: bool):
        seen.append((distance_m, attempt, reacquire))
        if attempt == 1:
            return ContactStepObservation(
                execution_succeeded=True,
                goal_satisfied=False,
                progress_m=0.0,
                contact_maintained=False,
            )
        return ContactStepObservation(
            execution_succeeded=True,
            goal_satisfied=True,
            progress_m=distance_m,
            contact_maintained=True,
        )

    result = ClosedLoopContactExecutor(
        backend,
        contact_profile=ContactExecutionProfile(maintain_contact=True),
        push_profile=PushPlanningProfile(
            nominal_step_distance_m=0.04,
            minimum_step_distance_m=0.01,
            retry_distance_scale=0.5,
        ),
        recovery_profile=TaskRecoveryProfile(maximum_execution_attempts=3),
    ).run(run_step)

    assert result.status is ClosedLoopContactStatus.SUCCESS
    assert seen == [(0.04, 1, False), (0.02, 2, True)]
    assert backend.restored == ["checkpoint-1"]
    assert result.attempts[0].state_rolled_back is True


def test_composite_goal_requires_region_containment_and_tool_clearance() -> None:
    request = _request(target_x_m=0.04)
    evaluator = CompositeGoalEvaluator(
        [
            RegionContainmentEvaluator(inset_margin_m=0.003),
            ToolClearanceEvaluator(minimum_clearance_m=0.01),
        ]
    )

    evaluation = evaluator.evaluate(request, _report(), request.world)

    assert target_fully_inside_region(
        request.world,
        target_id="block",
        region_id="region",
        inset_margin_m=0.003,
    )
    assert evaluation.status is GoalEvaluationStatus.SATISFIED

    outside_world = _world(target_x_m=0.24)
    failed = evaluator.evaluate(request, _report(), outside_world)
    assert failed.status is GoalEvaluationStatus.FAILED


def test_task_aware_goal_routes_physical_regions_and_conceptual_tool_rest() -> None:
    region_request = _request(target_x_m=0.04)
    evaluator = TaskAwareGoalEvaluator()

    region_result = evaluator.evaluate(
        region_request,
        _report(),
        region_request.world,
    )
    assert region_result.status is GoalEvaluationStatus.SATISFIED
    assert region_result.observed["region_id"] == "region"

    rest_pose = Pose(
        frame_id="world",
        position_m=(0.0, -0.25, 0.08),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    transport = region_request.model_copy(deep=True)
    transport.task.action_type = "transport"
    transport.task.target_ids = ["plate"]
    transport.task.goal = MotionGoal(
        goal_type=GoalType.POSE,
        target_pose=rest_pose,
        target_object_id="plate",
        target_region_id="tool_rest",
    )
    report = _report()
    report.final_robot_state.eef_pose = rest_pose

    rest_result = evaluator.evaluate(transport, report, transport.world)
    assert rest_result.status is GoalEvaluationStatus.SATISFIED


def test_profiles_reject_nonphysical_retry_configuration() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        PushPlanningProfile(
            nominal_step_distance_m=0.01,
            minimum_step_distance_m=0.02,
        )
