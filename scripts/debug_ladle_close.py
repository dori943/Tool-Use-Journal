"""Close the mounted 2F at one planned ladle grasp pose and print contacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from tuj.m5_motion.tool_use_journal_runtime import (
    ToolUseJournalControllerTrajectoryPlayer,
    ToolUseJournalEERuntime,
)


def _geom_points(model, data, geom_id: int) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id >= 0:
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        points = np.asarray(model.mesh_vert[start : start + count], dtype=float)
    else:
        sx, sy, sz = np.asarray(model.geom_size[geom_id], dtype=float)
        points = np.asarray(
            [
                (x * sx, y * sy, z * sz)
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ]
        )
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    return np.asarray(data.geom_xpos[geom_id], dtype=float) + points @ rotation.T


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    grasp = next(
        segment for segment in plan["segments"]
        if segment["segment_type"] == "GRASP"
    )["waypoints"][-1]
    runtime = ToolUseJournalEERuntime.from_repository_for_controller(
        args.repository,
        "C1_1_LegoSweep",
        active_ee="2F",
        seed=0,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        render_camera="frontview",
    )
    try:
        env = runtime.env
        model = env.sim.model._model
        data = env.sim.data._data
        context = plan["segments"][0]["collision_context_before"]
        for free_pose in context["free_object_poses"]:
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                free_pose["free_joint_name"],
            )
            qpos_address = int(model.jnt_qposadr[joint_id])
            pose = free_pose["pose"]
            x, y, z, w = pose["orientation_xyzw"]
            data.qpos[qpos_address : qpos_address + 7] = [
                *pose["position_m"],
                w,
                x,
                y,
                z,
            ]
        for joint_name, value in zip(
            plan["joint_names"], grasp["joint_positions_rad"], strict=True
        ):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)
            dof_address = int(model.jnt_dofadr[joint_id])
            data.qvel[dof_address] = 0.0
        mujoco.mj_forward(model, data)

        player = ToolUseJournalControllerTrajectoryPlayer(runtime)
        actual = player.preshape_finger_gripper_to_aperture(
            target_aperture_m=0.080,
            tolerance_m=0.010,
            final_settle_ticks=25,
        )
        print("preshape aperture", actual)
        robot = env.robots[0]
        splits = robot.composite_controller._action_split_indexes
        arm_start, arm_end = splits["right"]
        grip_start, grip_end = splits["right_gripper"]
        action = np.zeros(int(robot.action_dim), dtype=float)
        action[arm_start:arm_end] = player._actual_joint_positions(
            env, tuple(str(name) for name in robot.robot_model.joints)
        )
        action[grip_start:grip_end] = 1.0
        runtime.command_gripper(engaged=True, suction=False, command=1.0)
        object_body = int(env.obj_body_id["ladle"])
        initial_z = float(data.xpos[object_body][2])
        print(
            "ladle pose before close",
            np.asarray(data.xpos[object_body], dtype=float).round(6).tolist(),
            np.asarray(data.xquat[object_body], dtype=float).round(6).tolist(),
        )
        site_name = robot.gripper["right"].important_sites["grip_site"]
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        for tick in range(100):
            player._advance_controller(action)
            contact = runtime.object_contact_metrics("ladle")
            if tick % 5 == 0 or contact.contact_count:
                print(
                    tick,
                    "aperture",
                    round(runtime.fingerpad_separation_m(), 6),
                    "contacts",
                    contact.contact_count,
                    contact.contact_groups,
                    "normal",
                    round(contact.normal_force_n, 6),
                    "object_dz",
                    round(float(data.xpos[object_body][2]) - initial_z, 6),
                )
            if tick in {0, 5, 10}:
                site_position = np.asarray(data.site_xpos[site_id], dtype=float)
                site_rotation = np.asarray(
                    data.site_xmat[site_id], dtype=float
                ).reshape(3, 3)
                body_position = np.asarray(data.xpos[object_body], dtype=float)
                body_rotation = np.asarray(
                    data.xmat[object_body], dtype=float
                ).reshape(3, 3)
                center_world = body_position + body_rotation @ np.asarray(
                    [0.0, 0.087, 0.047]
                )
                print(
                    "  contact_center_in_site",
                    ((center_world - site_position) @ site_rotation).round(6).tolist(),
                    "site_world",
                    site_position.round(6).tolist(),
                )
                if tick == 10:
                    for geom_name in (
                        "ladle_g7",
                        "ladle_g10",
                        "ladle_g12",
                        "ladle_g17",
                        "gripper0_right_left_fingerpad_collision",
                        "gripper0_right_right_fingerpad_collision",
                    ):
                        geom_id = mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
                        )
                        points = (
                            _geom_points(model, data, geom_id) - site_position
                        ) @ site_rotation
                        print(
                            " ",
                            geom_name,
                            "bounds_in_site",
                            np.min(points, axis=0).round(6).tolist(),
                            np.max(points, axis=0).round(6).tolist(),
                            "contact_bits",
                            int(model.geom_contype[geom_id]),
                            int(model.geom_conaffinity[geom_id]),
                        )
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
