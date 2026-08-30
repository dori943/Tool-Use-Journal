"""Run a headless C1_1 MuJoCo contact-dynamics task preview.

This consumes Task Planner's selected EE/tool and uses the real C1_1 MuJoCo
objects, masses, friction, contacts, and free-body dynamics. The selected plate
is moved kinematically behind each remaining block and pushed toward the
collection zone. This is deliberately labelled a preview: the current Motion
Planner still has no grasp-pose generator, and this legacy preview does not
invoke the object-attach runtime or controller-backed trajectory player. The
robot therefore remains stationary and plate-to-gripper attachment is not
simulated here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

SRC = Path(__file__).resolve().parents[3]
REPOSITORY = SRC.parent
for path in (REPOSITORY, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tuj.m4_motion.tool_use_journal import (  # noqa: E402
    ToolUseJournalEnvironmentAdapter,
    make_tool_use_journal_env,
)


def _selected_resources(payload: dict[str, Any]) -> tuple[str, str]:
    if payload.get("status") != "SUCCESS":
        raise ValueError("Task Planner result is not SUCCESS")
    selected = payload.get("selected_plan")
    if not isinstance(selected, dict):
        raise ValueError("Task Planner result has no selected_plan")
    assignments = selected.get("candidate_assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("Task Planner selected_plan has no candidate_assignments")
    ee_values = {
        item.get("ee") for item in assignments if isinstance(item, dict)
    }
    tool_values = {
        item.get("tool")
        for item in assignments
        if isinstance(item, dict) and item.get("tool")
    }
    if None in ee_values or len(ee_values) != 1 or len(tool_values) != 1:
        raise ValueError("preview requires exactly one selected EE and tool")
    return str(next(iter(ee_values))), str(next(iter(tool_values)))


def _gk_mass(repository: Path, selected_tool: str) -> float | None:
    payload = json.loads(
        (repository / "output" / "c1_1" / "gk_SG1.json").read_text(
            encoding="utf-8"
        )
    )
    for raw_id, node in (payload.get("nodes") or {}).items():
        if str(raw_id).endswith("_" + selected_tool):
            value = node.get("mass_kg") if isinstance(node, dict) else None
            return float(value) if isinstance(value, (int, float)) else None
    return None


def _body_position(data: mujoco.MjData, body_id: int) -> list[float]:
    return [round(float(value), 6) for value in data.xpos[body_id]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--render-stride", type=int, default=80)
    args = parser.parse_args()

    plan_payload = json.loads(args.plan.read_text(encoding="utf-8"))
    selected_ee, selected_tool = _selected_resources(plan_payload)
    repository = args.repository.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "c1_1_contact_preview.mp4"
    final_frame_path = output_dir / "c1_1_final.png"
    report_path = output_dir / "c1_1_preview_report.json"

    env = make_tool_use_journal_env(
        repository,
        "C1_1_LegoSweep",
        active_ee=selected_ee,
        seed=args.seed,
        ignore_done=True,
        use_camera_obs=True,
        has_offscreen_renderer=True,
        camera_names="agentview",
        camera_heights=args.height,
        camera_widths=args.width,
        render_camera="agentview",
    )
    writer: cv2.VideoWriter | None = None
    try:
        env.reset()
        adapter = ToolUseJournalEnvironmentAdapter(env)
        adapter.require_physical_ee(selected_ee)
        model: mujoco.MjModel = env.sim.model._model
        data: mujoco.MjData = env.sim.data._data
        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview"
        )
        if camera_id >= 0:
            model.cam_fovy[camera_id] = 70.0
            mujoco.mj_forward(model, data)

        if selected_tool not in {"light_plate", "heavy_plate"}:
            raise ValueError(
                f"selected tool {selected_tool!r} is not a C1_1 plate"
            )
        tool_object = getattr(env, selected_tool)
        tool_joint = tool_object.joints[0]
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, tool_joint
        )
        if joint_id < 0:
            raise ValueError(f"tool joint {tool_joint!r} is absent")
        qpos_address = int(model.jnt_qposadr[joint_id])
        qvel_address = int(model.jnt_dofadr[joint_id])
        timestep = float(model.opt.timestep)

        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.width, args.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {video_path}")

        simulation_steps = 0
        rendered_frames = 0
        phase = "INITIAL"

        def render_frame(*, force: bool = False) -> None:
            nonlocal rendered_frames
            if not force and simulation_steps % args.render_stride != 0:
                return
            rgb = env.sim.render(
                camera_name="agentview",
                height=args.height,
                width=args.width,
            )[::-1].copy()
            cv2.rectangle(rgb, (0, 0), (args.width, 54), (0, 0, 0), -1)
            cv2.putText(
                rgb,
                f"C1_1 | EE={selected_ee} | tool={selected_tool}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                rgb,
                f"{phase} | KINEMATIC TOOL CONTACT PREVIEW",
                (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 220, 80),
                1,
                cv2.LINE_AA,
            )
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            rendered_frames += 1

        def step_once() -> None:
            nonlocal simulation_steps
            mujoco.mj_step(model, data)
            simulation_steps += 1
            render_frame()

        def tool_pose() -> np.ndarray:
            return np.asarray(data.qpos[qpos_address : qpos_address + 7]).copy()

        def move_tool(target_xyz: tuple[float, float, float], speed: float) -> None:
            start = tool_pose()
            target = start.copy()
            target[:3] = target_xyz
            distance = float(np.linalg.norm(target[:3] - start[:3]))
            steps = max(2, int(distance / (speed * timestep)))
            velocity = (target[:3] - start[:3]) / (steps * timestep)
            for index in range(1, steps + 1):
                alpha = index / steps
                data.qpos[qpos_address : qpos_address + 7] = (
                    (1.0 - alpha) * start + alpha * target
                )
                data.qvel[qvel_address : qvel_address + 6] = 0.0
                data.qvel[qvel_address : qvel_address + 3] = velocity
                mujoco.mj_forward(model, data)
                step_once()
            data.qvel[qvel_address : qvel_address + 6] = 0.0

        for _ in range(100):
            step_once()
        initial_positions = {
            block.name: _body_position(data, env.obj_body_id[block.name])
            for block in env.blocks
        }
        table_tool_z = float(tool_pose()[2])
        zone_center = np.asarray(env.collection_zone_center, dtype=float)
        zone_size = np.asarray(env.collection_zone_size, dtype=float)

        def safely_inside(position_xy: np.ndarray) -> bool:
            margin = 0.012
            return bool(
                abs(position_xy[0] - zone_center[0])
                <= zone_size[0] / 2.0 - margin
                and abs(position_xy[1] - zone_center[1])
                <= zone_size[1] / 2.0 - margin
            )

        for _ in range(12):
            render_frame(force=True)

        pushes: list[dict[str, Any]] = []
        for round_index in range(4):
            remaining_at_round_start = [
                block
                for block in env.blocks
                if not safely_inside(
                    np.asarray(
                        data.xpos[env.obj_body_id[block.name], :2], dtype=float
                    )
                )
            ]
            if not remaining_at_round_start:
                break
            for block in env.blocks:
                body_id = env.obj_body_id[block.name]
                block_xy = np.asarray(data.xpos[body_id, :2], dtype=float)
                if safely_inside(block_xy):
                    continue
                delta = block_xy - zone_center
                norm = float(np.linalg.norm(delta))
                direction = (
                    delta / norm if norm > 1e-9 else np.asarray((1.0, 0.0))
                )
                start_xy = block_xy + direction * 0.12
                end_xy = zone_center + direction * 0.10
                phase = f"PUSH {block.name} (round {round_index + 1})"
                move_tool((float(start_xy[0]), float(start_xy[1]), 0.95), 0.8)
                move_tool(
                    (float(start_xy[0]), float(start_xy[1]), table_tool_z),
                    0.3,
                )
                move_tool(
                    (float(end_xy[0]), float(end_xy[1]), table_tool_z),
                    0.10,
                )
                move_tool((float(end_xy[0]), float(end_xy[1]), 0.95), 0.6)
                for _ in range(20):
                    step_once()
                pushes.append(
                    {
                        "round": round_index + 1,
                        "block_id": block.name,
                        "start_xy_m": [float(value) for value in start_xy],
                        "end_xy_m": [float(value) for value in end_xy],
                    }
                )

        phase = "SETTLE / EVALUATE"
        for _ in range(300):
            step_once()
        final_positions = {
            block.name: _body_position(data, env.obj_body_id[block.name])
            for block in env.blocks
        }
        blocks_inside = [
            block.name
            for block in env.blocks
            if abs(data.xpos[env.obj_body_id[block.name], 0] - zone_center[0])
            <= zone_size[0] / 2.0
            and abs(data.xpos[env.obj_body_id[block.name], 1] - zone_center[1])
            <= zone_size[1] / 2.0
        ]
        phase = f"FINAL: {len(blocks_inside)}/{len(env.blocks)} IN ZONE"
        for _ in range(30):
            render_frame(force=True)
        final_rgb = env.sim.render(
            camera_name="agentview",
            height=args.height,
            width=args.width,
        )[::-1]
        cv2.imwrite(
            str(final_frame_path), cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
        )

        actual_tool_mass = float(
            env.get_tool_physical_metadata(selected_tool)["mass_kg"]
        )
        report = {
            "status": "SUCCESS" if len(blocks_inside) == len(env.blocks) else "FAILED",
            "task_id": "c1_1",
            "mode": "KINEMATIC_TOOL_CONTACT_PREVIEW",
            "task_planner_result": str(args.plan.resolve()),
            "selected_ee": selected_ee,
            "selected_tool": selected_tool,
            "gk_inferred_tool_mass_kg": _gk_mass(repository, selected_tool),
            "environment_tool_mass_kg": actual_tool_mass,
            "blocks_total": len(env.blocks),
            "blocks_in_collection_zone": len(blocks_inside),
            "block_ids_in_collection_zone": blocks_inside,
            "initial_block_positions_m": initial_positions,
            "final_block_positions_m": final_positions,
            "pushes": pushes,
            "simulation": {
                "engine": "MuJoCo",
                "physics_timestep_s": timestep,
                "seed": args.seed,
                "simulation_steps": simulation_steps,
                "rendered_frames": rendered_frames,
                "contact_dynamics_for_blocks": True,
                "selected_tool_motion": "KINEMATIC_FREE_JOINT_COMMAND",
                "robot_joint_motion_simulated": False,
                "grasp_or_attachment_simulated": False,
                "controller_tracking_simulated": False,
            },
            "artifacts": {
                "video": str(video_path),
                "final_frame": str(final_frame_path),
                "report": str(report_path),
            },
            "limitations": [
                "The selected plate is commanded kinematically; it is not attached to the gripper.",
                "This preview does not invoke controller-backed MotionPlan playback.",
                "Block motion and plate/block contacts use MuJoCo dynamics.",
            ],
        }
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["status"] == "SUCCESS" else 1
    finally:
        if writer is not None:
            writer.release()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
