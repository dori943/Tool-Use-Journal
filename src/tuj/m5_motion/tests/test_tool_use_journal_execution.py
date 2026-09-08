from __future__ import annotations

from types import SimpleNamespace

import pytest

from tuj.m5_motion.attachment_retarget import attachment_transform
from tuj.m5_motion.schema import (
    AttachedObjectTransform,
    ArtifactProvenance,
    CollisionContext,
    ExecutionReport,
    ExecutionStatus,
    GoalType,
    ModuleName,
    MotionGoal,
    MotionPlan,
    MotionPlanRequest,
    MotionTask,
    RobotState,
    SceneRef,
    SegmentType,
    SimulationMetrics,
    TrajectorySegment,
    TrajectoryWaypoint,
    WorldSnapshot,
)
import tuj.m5_motion.tool_use_journal_execution as execution_module
from tuj.m5_motion.tool_use_journal_execution import (
    ToolUseJournalExecutionAdapter,
)
from tuj.m5_motion.tool_use_journal_runtime import (
    ToolUseJournalKinematicTrajectoryPlayer,
)


def _provenance(artifact_id: str, artifact_type: str, module: ModuleName):
    return ArtifactProvenance(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        produced_by=module,
        invocation_id=f"invocation:{artifact_id}",
    )


def _request_and_plan(*, with_context: bool = True):
    request = MotionPlanRequest(
        request_id="request",
        provenance=_provenance(
            "request-artifact", "MotionPlanRequest", ModuleName.TASK_PLANNER
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
    context = (
        CollisionContext(
            context_id="attached-2f",
            active_ee="2F",
            collision_model_version="fixture",
        )
        if with_context
        else None
    )
    plan = MotionPlan(
        plan_id="plan",
        request_id=request.request_id,
        provenance=_provenance(
            "plan-artifact", "MotionPlan", ModuleName.MOTION_PLANNER
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
                collision_context_before=context,
                collision_context_after=context,
            )
        ],
        expected_final_state=RobotState(
            robot_id="robot",
            joint_names=["j1"],
            joint_positions_rad=[0.1],
        ),
    )
    return request, plan


class _Compiler:
    def __init__(self):
        self.calls = []
        self.registry = SimpleNamespace(check=lambda *args, **kwargs: None)

    def build_collision_registry(self, contexts, **kwargs):
        self.calls.append((contexts, kwargs))
        return self.registry


def test_adapter_rebuilds_exact_plan_contexts_and_derives_runtime_config() -> None:
    request, plan = _request_and_plan()
    compiler = _Compiler()
    runtime = SimpleNamespace(
        env=SimpleNamespace(model_timestep=0.002, control_timestep=0.02)
    )
    adapter = ToolUseJournalExecutionAdapter(
        runtime,
        compiler=compiler,
        controller=False,
        max_duration_padding_s=2.0,
        random_seed=7,
    )

    player = adapter.player(request, plan, 0)
    config = adapter.config(request, plan, 0)

    assert isinstance(player, ToolUseJournalKinematicTrajectoryPlayer)
    assert player._collision_probe is compiler.registry
    assert set(compiler.calls[0][0]) == {"attached-2f"}
    assert compiler.calls[0][1]["default_active_ee"] == "2F"
    assert config.physics_timestep_s == 0.002
    assert config.control_timestep_s == 0.02
    assert config.max_duration_s == pytest.approx(2.1)
    assert config.random_seed == 7
    assert adapter.collision_probe(request, plan, 1) is compiler.registry
    assert len(compiler.calls) == 1


def test_adapter_refuses_unscoped_collision_execution() -> None:
    request, plan = _request_and_plan(with_context=False)
    runtime = SimpleNamespace(
        env=SimpleNamespace(model_timestep=0.002, control_timestep=0.02)
    )
    adapter = ToolUseJournalExecutionAdapter(
        runtime, compiler=_Compiler(), controller=False
    )

    with pytest.raises(ValueError, match="no event-scoped collision contexts"):
        adapter.player(request, plan, 0)


def test_world_snapshot_normalizes_and_carries_contact_friction_transform(
    monkeypatch,
) -> None:
    request, _ = _request_and_plan()
    request.task.tool = "part"
    final_state = request.world.robot_state.model_copy(
        update={"held_tool_id": "part"}
    )
    raw_transform = {
        "object_id": "part",
        "free_joint_name": "part_free",
        "reference_kind": "body",
        "reference_name": "right_hand",
        "position_in_reference_m": [0.0, 0.0, 0.1],
        "orientation_in_reference_xyzw": [0.0, 0.0, 0.0, 1.0],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }

    class _EnvironmentAdapter:
        def __init__(self, env):
            del env

        def world_snapshot(self, **kwargs):
            del kwargs
            return request.world.model_copy(deep=True)

    monkeypatch.setattr(
        execution_module, "ToolUseJournalEnvironmentAdapter", _EnvironmentAdapter
    )
    runtime = SimpleNamespace(
        env=object(), attachment=None, attached_object_id=None
    )
    adapter = ToolUseJournalExecutionAdapter(runtime, compiler=_Compiler())
    report = ExecutionReport(
        report_id="report:pick",
        run_id="run:pick",
        plan_id="plan:pick",
        provenance=_provenance(
            "report-artifact:pick", "ExecutionReport", ModuleName.MOTION_PLANNER
        ),
        status=ExecutionStatus.SUCCESS,
        final_robot_state=final_state,
        metrics=SimulationMetrics(
            executed_duration_s=0.1,
            max_joint_tracking_error_rad=0.0,
        ),
        metadata={
            "physical_grasp_execution_succeeded": True,
            "physical_grasp_transform": raw_transform,
        },
    )

    picked_world = adapter.world_snapshot(request, report)
    stored = picked_world.metadata["contact_friction_held_objects"]["part"]
    assert "rotation_matrix" not in stored
    assert AttachedObjectTransform.model_validate(stored).object_id == "part"

    transport_request = request.model_copy(
        update={"world": picked_world}, deep=True
    )
    transport_report = report.model_copy(
        update={
            "report_id": "report:transport",
            "final_robot_state": final_state,
            "metadata": {},
        }
    )
    transported_world = adapter.world_snapshot(
        transport_request, transport_report
    )
    assert transported_world.metadata["contact_friction_held_objects"] == {
        "part": stored
    }
    assert attachment_transform(transported_world, "part").object_id == "part"

    released_report = report.model_copy(
        update={
            "report_id": "report:release",
            "final_robot_state": final_state.model_copy(
                update={"held_tool_id": None}
            ),
            "metadata": {},
        }
    )
    released_world = adapter.world_snapshot(transport_request, released_report)
    assert released_world.metadata["contact_friction_held_objects"] == {}
