"""Replay saved MotionPlans in one Tool-Use-Journal runtime or record MP4."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

SRC = Path(__file__).resolve().parents[3]
REPOSITORY = SRC.parent
for package_root in (REPOSITORY, SRC):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from tuj.m4_motion.schema import (  # noqa: E402
    ArtifactProvenance,
    ModuleName,
    MotionPlan,
    SimulationConfig,
    SimulationRun,
)
from tuj.m4_motion.tool_use_journal_runtime import (  # noqa: E402
    ToolUseJournalControllerTrajectoryPlayer,
    ToolUseJournalEERuntime,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("motion_plans", type=Path, nargs="+")
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
    parser.add_argument(
        "--camera-eye",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Override the fixed camera world position in metres.",
    )
    parser.add_argument(
        "--camera-look-at",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="World point aimed at by --camera-eye.",
    )
    parser.add_argument("--camera-fovy", type=float)
    parser.add_argument("--preshape-close-pulses", type=int, default=0)
    parser.add_argument("--preshape-aperture-m", type=float)
    parser.add_argument("--preshape-settle-ticks", type=int, default=25)
    parser.add_argument("--gripper-actuator-kp", type=float)
    parser.add_argument("--max-grip-force-n", type=float)
    parser.add_argument(
        "--video-fps",
        type=float,
        default=20.0,
        help="Output FPS and simulated-time capture rate.",
    )
    parser.add_argument(
        "--controller-kp-after-first",
        type=float,
        help="Retune the live joint controller after the first plan.",
    )
    parser.add_argument(
        "--controller-damping-ratio-after-first",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()

    if args.realtime_factor < 0.0:
        parser.error("--realtime-factor must be non-negative")
    if args.hold_seconds < 0.0:
        parser.error("--hold-seconds must be non-negative")
    if args.video_fps <= 0.0:
        parser.error("--video-fps must be positive")
    if (args.camera_eye is None) != (args.camera_look_at is None):
        parser.error("--camera-eye and --camera-look-at must be supplied together")
    if args.camera_fovy is not None and not 1.0 < args.camera_fovy < 179.0:
        parser.error("--camera-fovy must be between 1 and 179 degrees")
    if args.preshape_close_pulses < 0 or args.preshape_settle_ticks < 0:
        parser.error("gripper pre-shape values must be non-negative")
    if args.preshape_close_pulses and args.preshape_aperture_m is not None:
        parser.error("choose pulse or metric-aperture pre-shaping, not both")
    if args.preshape_aperture_m is not None and not (
        0.0 < args.preshape_aperture_m < 0.085
    ):
        parser.error("--preshape-aperture-m must be within (0, 0.085)")
    if args.gripper_actuator_kp is not None and args.gripper_actuator_kp <= 0.0:
        parser.error("--gripper-actuator-kp must be positive")
    if args.max_grip_force_n is not None and args.max_grip_force_n <= 0.0:
        parser.error("--max-grip-force-n must be positive")

    plan_paths = [path.resolve() for path in args.motion_plans]
    plans = [
        MotionPlan.model_validate_json(path.read_text(encoding="utf-8"))
        for path in plan_paths
    ]
    record_video = args.video is not None
    runtime = ToolUseJournalEERuntime.from_repository_for_controller(
        args.repository.resolve(),
        args.env,
        active_ee=args.active_ee,
        seed=args.seed,
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
    reports = []
    try:
        model: mujoco.MjModel = runtime.env.sim.model._model
        data: mujoco.MjData = runtime.env.sim.data._data
        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera
        )
        if camera_id < 0:
            raise RuntimeError(f"camera {args.camera!r} is absent")
        if args.camera_eye is not None and args.camera_look_at is not None:
            eye = np.asarray(args.camera_eye, dtype=float)
            target = np.asarray(args.camera_look_at, dtype=float)
            forward = target - eye
            forward /= np.linalg.norm(forward)
            world_up = np.asarray((0.0, 0.0, 1.0), dtype=float)
            right = np.cross(forward, world_up)
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            rotation = np.column_stack((right, up, -forward))
            quaternion_wxyz = np.empty(4, dtype=float)
            mujoco.mju_mat2Quat(
                quaternion_wxyz,
                np.ascontiguousarray(rotation.reshape(9)),
            )
            model.cam_pos[camera_id] = eye
            model.cam_quat[camera_id] = quaternion_wxyz
        if args.camera_fovy is not None:
            model.cam_fovy[camera_id] = args.camera_fovy
        mujoco.mj_forward(model, data)

        if record_video:
            video_path = args.video.resolve()
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.video_fps,
                (args.width, args.height),
            )
            if not video_writer.isOpened():
                raise RuntimeError(f"could not open video writer for {video_path}")

            capture_credit = 0.0

            def write_frame() -> None:
                rgb = runtime.env.sim.render(
                    camera_name=args.camera,
                    width=args.width,
                    height=args.height,
                )[::-1]
                video_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            def record_frame() -> None:
                nonlocal capture_credit
                capture_credit += (
                    args.video_fps * float(runtime.env.control_timestep)
                )
                if capture_credit + 1e-12 < 1.0:
                    return
                capture_credit -= 1.0
                write_frame()

            runtime.env.render = record_frame
            write_frame()

        gripper_actuators: tuple[str, ...] = ()
        if args.gripper_actuator_kp is not None:
            gripper_actuators = runtime.set_finger_gripper_actuator_gains(
                kp=args.gripper_actuator_kp,
                max_grip_force_n=args.max_grip_force_n,
            )

        if args.preshape_close_pulses or args.preshape_aperture_m is not None:
            preshape_player = ToolUseJournalControllerTrajectoryPlayer(runtime)
            if args.preshape_aperture_m is not None:
                preshape_player.preshape_finger_gripper_to_aperture(
                    target_aperture_m=args.preshape_aperture_m,
                    final_settle_ticks=args.preshape_settle_ticks,
                )
            else:
                preshape_player.preshape_finger_gripper(
                    close_pulses=args.preshape_close_pulses,
                    settle_ticks=args.preshape_settle_ticks,
                )
            if record_video:
                write_frame()

        for index, (plan_path, plan) in enumerate(
            zip(plan_paths, plans, strict=True)
        ):
            adaptive_settle_budget_s = sum(
                float(
                    segment.metadata.get("tracking_settle", {}).get(
                        "max_wait_s", 0.0
                    )
                )
                for segment in plan.segments
                if isinstance(segment.metadata.get("tracking_settle"), dict)
            )
            run = SimulationRun(
                run_id=f"viewer:{index}:{plan.plan_id}",
                provenance=ArtifactProvenance(
                    artifact_id=f"viewer-run:{index}:{plan.plan_id}",
                    artifact_type="SimulationRun",
                    produced_by=ModuleName.SIMULATOR,
                    invocation_id=f"viewer:{index}:{plan.plan_id}",
                    input_artifact_ids=[plan.provenance.artifact_id],
                    metadata={"source_motion_plan": str(plan_path)},
                ),
                plan=plan,
                config=SimulationConfig(
                    physics_timestep_s=float(runtime.env.model_timestep),
                    control_timestep_s=float(runtime.env.control_timestep),
                    realtime_factor=args.realtime_factor,
                    max_duration_s=(
                        float(plan.duration_s)
                        + adaptive_settle_budget_s
                        + 5.0
                    ),
                    terminate_on_collision=True,
                    render=True,
                    random_seed=args.seed,
                ),
            )
            report = ToolUseJournalControllerTrajectoryPlayer(runtime).execute(
                run
            )
            reports.append(report)
            if report.status.value != "SUCCESS":
                break
            if (
                index == 0
                and len(plans) > 1
                and args.controller_kp_after_first is not None
            ):
                runtime.set_joint_position_controller_gains(
                    kp=args.controller_kp_after_first,
                    damping_ratio=(
                        args.controller_damping_ratio_after_first
                    ),
                )
        print(
            json.dumps(
                {
                    "plan_ids": [plan.plan_id for plan in plans],
                    "statuses": [report.status.value for report in reports],
                    "executed_durations_s": [
                        report.metrics.executed_duration_s for report in reports
                    ],
                    "final_attached_object_id": runtime.attached_object_id,
                    "camera": {
                        "name": args.camera,
                        "width": args.width,
                        "height": args.height,
                        "eye_m": args.camera_eye,
                        "look_at_m": args.camera_look_at,
                        "fovy_deg": float(model.cam_fovy[camera_id]),
                    },
                    "gripper_actuator_control": {
                        "actuators": list(gripper_actuators),
                        "kp": args.gripper_actuator_kp,
                        "max_grip_force_n": args.max_grip_force_n,
                    },
                    "failures": [
                        (
                            report.failure.model_dump(mode="json")
                            if report.failure is not None
                            else None
                        )
                        for report in reports
                    ],
                    "video": str(args.video.resolve()) if record_video else None,
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
                json.dumps(
                    [report.model_dump(mode="json") for report in reports],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        deadline = time.monotonic() + args.hold_seconds
        while time.monotonic() < deadline:
            runtime.env.render()
            time.sleep(0.02)
        return (
            0
            if len(reports) == len(plans)
            and all(report.status.value == "SUCCESS" for report in reports)
            else 2
        )
    finally:
        if video_writer is not None:
            video_writer.release()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
