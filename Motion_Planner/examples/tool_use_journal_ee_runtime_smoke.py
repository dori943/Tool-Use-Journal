"""Headless smoke test for Tool-Use-Journal runtime EE model exchange.

This intentionally uses stationary arm waypoints.  It verifies event ordering,
model replacement, state transfer, rack visibility, and ExecutionReport output;
it is not a dock-motion reachability test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TASK_PLANNER = PROJECT.parent / "Task_Planner"
for path in (PROJECT, TASK_PLANNER):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from motion_planner.schema import (  # noqa: E402
    ArtifactProvenance,
    CollisionContext,
    EventType,
    ModuleName,
    MotionPlan,
    RobotState,
    SegmentType,
    SimulationConfig,
    SimulationRun,
    TrajectoryEvent,
    TrajectorySegment,
    TrajectoryWaypoint,
)
from motion_planner.tool_use_journal import (  # noqa: E402
    ToolUseJournalEnvironmentAdapter,
)
from motion_planner.tool_use_journal_runtime import (  # noqa: E402
    ToolUseJournalControllerTrajectoryPlayer,
    ToolUseJournalEERuntime,
    ToolUseJournalKinematicTrajectoryPlayer,
)


def _provenance(
    artifact_id: str, artifact_type: str, module: ModuleName
) -> ArtifactProvenance:
    return ArtifactProvenance(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        produced_by=module,
        invocation_id=f"{artifact_id}:invocation",
    )


def _stationary_exchange_run(
    runtime: ToolUseJournalEERuntime,
    *,
    from_ee: str,
    to_ee: str,
) -> SimulationRun:
    state = ToolUseJournalEnvironmentAdapter(
        runtime.env
    ).world_snapshot().robot_state
    attached_from = CollisionContext(
        context_id=f"ee-attached:{from_ee}",
        active_ee=from_ee,
        collision_model_version=f"runtime-smoke:{from_ee}",
    )
    bare = CollisionContext(
        context_id="bare-flange",
        collision_model_version="runtime-smoke:bare",
    )
    attached_to = CollisionContext(
        context_id=f"ee-attached:{to_ee}",
        active_ee=to_ee,
        collision_model_version=f"runtime-smoke:{to_ee}",
    )
    q = list(state.joint_positions_rad)
    first = TrajectorySegment(
        segment_id="stationary-undock-event",
        segment_type=SegmentType.EE_UNDOCK,
        start_time_s=0.0,
        end_time_s=0.1,
        collision_checked=False,
        collision_context_before=attached_from,
        collision_context_after=bare,
        waypoints=[
            TrajectoryWaypoint(time_from_start_s=0.0, joint_positions_rad=q),
            TrajectoryWaypoint(time_from_start_s=0.1, joint_positions_rad=q),
        ],
    )
    second = TrajectorySegment(
        segment_id="stationary-dock-event",
        segment_type=SegmentType.EE_DOCK,
        start_time_s=0.1,
        end_time_s=0.2,
        collision_checked=False,
        collision_context_before=bare,
        collision_context_after=attached_to,
        waypoints=[
            TrajectoryWaypoint(time_from_start_s=0.1, joint_positions_rad=q),
            TrajectoryWaypoint(time_from_start_s=0.2, joint_positions_rad=q),
        ],
    )
    plan = MotionPlan(
        plan_id="tool-use-journal-ee-runtime-smoke-plan",
        request_id="tool-use-journal-ee-runtime-smoke-request",
        provenance=_provenance(
            "runtime-smoke-plan-artifact",
            "MotionPlan",
            ModuleName.MOTION_PLANNER,
        ),
        scene_signature="runtime-smoke-scene",
        robot_id=state.robot_id,
        joint_names=list(state.joint_names),
        duration_s=0.2,
        segments=[first, second],
        events=[
            TrajectoryEvent(
                event_id="unlock",
                time_from_start_s=0.1,
                event_type=EventType.TOOL_UNLOCK,
                target_id=from_ee,
            ),
            TrajectoryEvent(
                event_id="verify-release",
                time_from_start_s=0.1,
                event_type=EventType.VERIFY_TOOL_RELEASE,
                target_id=from_ee,
            ),
            TrajectoryEvent(
                event_id="lock",
                time_from_start_s=0.2,
                event_type=EventType.TOOL_LOCK,
                target_id=to_ee,
            ),
            TrajectoryEvent(
                event_id="verify-lock",
                time_from_start_s=0.2,
                event_type=EventType.VERIFY_TOOL_LOCK,
                target_id=to_ee,
            ),
        ],
        expected_final_state=RobotState(
            robot_id=state.robot_id,
            joint_names=list(state.joint_names),
            joint_positions_rad=q,
            joint_velocities_rad_s=[0.0] * len(q),
        ),
    )
    return SimulationRun(
        run_id="tool-use-journal-ee-runtime-smoke",
        provenance=_provenance(
            "runtime-smoke-run-artifact",
            "SimulationRun",
            ModuleName.SIMULATOR,
        ),
        plan=plan,
        config=SimulationConfig(render=False),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument(
        "--env",
        choices=("C1_1_LegoSweep", "C2_1_ObjectSorting"),
        default="C2_1_ObjectSorting",
    )
    parser.add_argument("--from-ee", choices=("2F", "3F", "vac"), default="2F")
    parser.add_argument("--to-ee", choices=("2F", "3F", "vac"), default="vac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controller-kp", type=float, default=50.0)
    parser.add_argument(
        "--controller-damping-ratio", type=float, default=1.0
    )
    parser.add_argument(
        "--controller",
        action="store_true",
        help="use robosuite absolute joint-position torque control",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional path for the ExecutionReport JSON",
    )
    args = parser.parse_args()
    if args.from_ee == args.to_ee:
        parser.error("--from-ee and --to-ee must differ")

    runtime_factory = (
        ToolUseJournalEERuntime.from_repository_for_controller
        if args.controller
        else ToolUseJournalEERuntime.from_repository
    )
    controller_options = (
        {
            "joint_position_kp": args.controller_kp,
            "joint_position_damping_ratio": args.controller_damping_ratio,
        }
        if args.controller
        else {}
    )
    runtime = runtime_factory(
        args.repository,
        args.env,
        active_ee=args.from_ee,
        seed=args.seed,
        **controller_options,
    )
    try:
        run = _stationary_exchange_run(
            runtime,
            from_ee=args.from_ee,
            to_ee=args.to_ee,
        )
        player_type = (
            ToolUseJournalControllerTrajectoryPlayer
            if args.controller
            else ToolUseJournalKinematicTrajectoryPlayer
        )
        report = player_type(runtime).execute(run)
        payload = report.model_dump_json(indent=2)
        print(payload)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        return 0 if report.status.value == "SUCCESS" else 1
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
