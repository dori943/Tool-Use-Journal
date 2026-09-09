from __future__ import annotations

from types import SimpleNamespace

import pytest

from tuj.m5_motion.live_execution import (
    LivePlanExecutionError,
    LivePlanExecutionSession,
)
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    GoalType,
    InterpolationType,
    ModuleName,
    MotionGoal,
    MotionPlan,
    MotionPlanRequest,
    MotionTask,
    RobotState,
    SceneRef,
    SegmentType,
    TrajectorySegment,
    TrajectoryWaypoint,
    WorldSnapshot,
)


def _world(x: float = 0.0) -> WorldSnapshot:
    return WorldSnapshot(
        scene=SceneRef(signature=f"scene:{x}"),
        robot_state=RobotState(
            robot_id="robot",
            joint_names=["j1"],
            joint_positions_rad=[x],
        ),
        objects={
            "part": {
                "pose": {
                    "frame_id": "world",
                    "position_m": [x, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        },
    )


def _request(index: int, world: WorldSnapshot) -> MotionPlanRequest:
    return MotionPlanRequest(
        request_id=f"request:{index}",
        provenance=ArtifactProvenance(
            artifact_id=f"request-artifact:{index}",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id=f"request:{index}",
        ),
        world=world,
        task=MotionTask(
            task_id="task",
            subgoal_id=f"contact:{index}",
            action_type="tool_act",
            ee="2F",
            target_ids=["part"],
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_object_id="part",
            ),
        ),
    )


def _plan(request: MotionPlanRequest) -> MotionPlan:
    start = request.world.robot_state.joint_positions_rad[0]
    end = start + 0.1
    return MotionPlan(
        plan_id=f"plan:{request.request_id}",
        request_id=request.request_id,
        provenance=ArtifactProvenance(
            artifact_id=f"plan-artifact:{request.request_id}",
            artifact_type="MotionPlan",
            produced_by=ModuleName.MOTION_PLANNER,
            invocation_id=f"plan:{request.request_id}",
        ),
        scene_signature=request.world.scene.signature,
        robot_id="robot",
        joint_names=["j1"],
        duration_s=1.0,
        segments=[
            TrajectorySegment(
                segment_id=f"segment:{request.request_id}",
                segment_type=SegmentType.CUSTOM,
                start_time_s=0.0,
                end_time_s=1.0,
                interpolation=InterpolationType.LINEAR,
                waypoints=[
                    TrajectoryWaypoint(
                        time_from_start_s=0.0,
                        joint_positions_rad=[start],
                    ),
                    TrajectoryWaypoint(
                        time_from_start_s=1.0,
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
        ),
    )


class _Runtime:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class _Adapter:
    def __init__(self, seen: list[object], *, successful: bool) -> None:
        self.seen = seen
        self.successful = successful

    def execute(self, planning, *, store):
        del store
        request = planning.requests[0]
        self.seen.append(request)
        observed = request.world.model_copy(deep=True)
        old_x = float(observed.objects["part"]["pose"]["position_m"][0])
        observed.objects["part"]["pose"]["position_m"][0] = old_x + 1.0
        observed.scene.signature = f"observed:{len(self.seen)}"
        status = "SUCCESS" if self.successful else "GOAL_FAILED"
        return SimpleNamespace(
            successful=self.successful,
            status=SimpleNamespace(value=status),
            detail=("ok" if self.successful else "grounded goal failed"),
            final_world=observed,
            manifest_path=None,
            runs=[SimpleNamespace(run_id=f"run:{len(self.seen)}")],
            reports=[SimpleNamespace(report_id=f"report:{len(self.seen)}")],
            goal_evaluations=[SimpleNamespace(status=SimpleNamespace(value=status))],
        )


def test_live_session_returns_observed_world_and_reuses_runtime(tmp_path) -> None:
    runtime = _Runtime()
    seen_runtime: list[object] = []
    seen_requests: list[object] = []

    def factory(current_runtime, repository, **kwargs):
        del repository, kwargs
        seen_runtime.append(current_runtime)
        return _Adapter(seen_requests, successful=True)

    session = LivePlanExecutionSession(
        runtime,
        tmp_path,
        tmp_path / "simulation",
        seed=0,
        controller=True,
        realtime_factor=0.0,
        render=False,
        adapter_factory=factory,
    )
    first_request = _request(1, _world())
    first_observed = session(first_request, _plan(first_request))
    second_request = _request(2, first_observed)
    second_observed = session(second_request, _plan(second_request))
    session.complete(video_hold_seconds=0.0, viewer_hold_seconds=0.0)
    session.close()

    assert seen_runtime == [runtime, runtime]
    assert seen_requests[1].world.objects["part"]["pose"]["position_m"][0] == 1.0
    assert second_observed.objects["part"]["pose"]["position_m"][0] == 2.0
    assert session.status == "SUCCESS"
    assert session.run_count == 2
    assert session.report_count == 2
    assert session.manifest_path.is_file()
    assert runtime.close_count == 1


def test_live_session_fails_closed_after_execution_failure(tmp_path) -> None:
    runtime = _Runtime()
    seen_requests: list[object] = []
    session = LivePlanExecutionSession(
        runtime,
        tmp_path,
        tmp_path / "simulation",
        seed=0,
        controller=True,
        realtime_factor=0.0,
        render=False,
        adapter_factory=lambda *args, **kwargs: _Adapter(
            seen_requests, successful=False
        ),
    )
    request = _request(1, _world())

    with pytest.raises(LivePlanExecutionError, match="grounded goal failed"):
        session(request, _plan(request))
    with pytest.raises(RuntimeError, match="not accepting plans"):
        session(_request(2, _world()), _plan(_request(2, _world())))
    session.close()

    assert len(seen_requests) == 1
    assert session.status == "FAILED"
    assert session.records[0]["status"] == "FAILED"
    assert runtime.close_count == 1
