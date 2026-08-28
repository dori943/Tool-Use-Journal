"""Replay one saved MotionPlan in a visible Tool-Use-Journal MuJoCo window."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
TASK_PLANNER = WORKSPACE / "Task_Planner"
for package_root in (PROJECT, TASK_PLANNER):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from motion_planner.schema import (  # noqa: E402
    ArtifactProvenance,
    ModuleName,
    MotionPlan,
    SimulationConfig,
    SimulationRun,
)
from motion_planner.tool_use_journal_runtime import (  # noqa: E402
    ToolUseJournalControllerTrajectoryPlayer,
    ToolUseJournalEERuntime,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("motion_plan", type=Path)
    parser.add_argument("--env", default="C1_1_LegoSweep")
    parser.add_argument("--active-ee", default="2F")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--hold-seconds", type=float, default=10.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--controller-kp", type=float, default=50.0)
    parser.add_argument(
        "--controller-damping-ratio", type=float, default=1.0
    )
    args = parser.parse_args()

    if args.realtime_factor <= 0.0:
        parser.error("--realtime-factor must be positive for visible playback")
    if args.hold_seconds < 0.0:
        parser.error("--hold-seconds must be non-negative")

    plan_path = args.motion_plan.resolve()
    plan = MotionPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    record_video = args.video is not None
    runtime = ToolUseJournalEERuntime.from_repository_for_controller(
        args.repository.resolve(),
        args.env,
        active_ee=args.active_ee,
        seed=args.seed,
        joint_position_kp=args.controller_kp,
        joint_position_damping_ratio=args.controller_damping_ratio,
        ignore_done=True,
        use_camera_obs=False,
        has_renderer=not record_video,
        has_offscreen_renderer=record_video,
        render_camera=args.camera,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
    )
    video_writer: cv2.VideoWriter | None = None
    try:
        if record_video:
            video_path = args.video.resolve()
            video_path.parent.mkdir(parents=True, exist_ok=True)
            frames_per_second = 1.0 / float(runtime.env.control_timestep)
            video_writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                frames_per_second,
                (args.width, args.height),
            )
            if not video_writer.isOpened():
                raise RuntimeError(f"could not open video writer for {video_path}")

            def record_frame() -> None:
                rgb = runtime.env.sim.render(
                    camera_name=args.camera,
                    width=args.width,
                    height=args.height,
                )[::-1]
                video_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            runtime.env.render = record_frame

        run = SimulationRun(
            run_id=f"viewer:{plan.plan_id}",
            provenance=ArtifactProvenance(
                artifact_id=f"viewer-run:{plan.plan_id}",
                artifact_type="SimulationRun",
                produced_by=ModuleName.SIMULATOR,
                invocation_id=f"viewer:{plan.plan_id}",
                input_artifact_ids=[plan.provenance.artifact_id],
                metadata={"source_motion_plan": str(plan_path)},
            ),
            plan=plan,
            config=SimulationConfig(
                physics_timestep_s=float(runtime.env.model_timestep),
                control_timestep_s=float(runtime.env.control_timestep),
                realtime_factor=args.realtime_factor,
                max_duration_s=float(plan.duration_s) + 5.0,
                terminate_on_collision=True,
                render=True,
                random_seed=args.seed,
            ),
        )
        report = ToolUseJournalControllerTrajectoryPlayer(runtime).execute(run)
        print(
            json.dumps(
                {
                    "plan_id": plan.plan_id,
                    "status": report.status.value,
                    "executed_duration_s": report.metrics.executed_duration_s,
                    "final_attached_object_id": (
                        report.final_robot_state.attached_object_id
                    ),
                    "failure": (
                        report.failure.model_dump(mode="json")
                        if report.failure is not None
                        else None
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        if args.report is not None:
            report_path = args.report.resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                report.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
        deadline = time.monotonic() + args.hold_seconds
        while time.monotonic() < deadline:
            runtime.env.render()
            time.sleep(0.02)
        return 0 if report.status.value == "SUCCESS" else 2
    finally:
        if video_writer is not None:
            video_writer.release()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
