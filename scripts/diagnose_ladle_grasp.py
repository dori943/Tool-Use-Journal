"""Print MuJoCo-local ladle and 2F finger geometry for grasp debugging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from tuj.m5_motion.tool_use_journal import make_tool_use_journal_env


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, obj, index) or f"#{index}"


def _mesh_world_vertices(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> np.ndarray | None:
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return None
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    return np.asarray(data.geom_xpos[geom_id], dtype=float) + vertices @ rotation.T


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--world", type=Path)
    args = parser.parse_args()
    env = make_tool_use_journal_env(
        args.repository,
        "C1_1_LegoSweep",
        active_ee="2F",
        has_renderer=False,
        ignore_done=True,
    )
    try:
        env.reset()
        model = env.sim.model._model
        data = env.sim.data._data
        if args.world is not None:
            world = json.loads(args.world.read_text(encoding="utf-8"))
            pose = world["objects"]["ladle"]["pose"]
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, "ladle_joint0"
            )
            qpos_address = int(model.jnt_qposadr[joint_id])
            x, y, z, w = pose["orientation_xyzw"]
            data.qpos[qpos_address : qpos_address + 7] = [
                *pose["position_m"],
                w,
                x,
                y,
                z,
            ]
            mujoco.mj_forward(model, data)
        if args.plan is not None:
            plan = json.loads(args.plan.read_text(encoding="utf-8"))
            grasp = next(
                segment for segment in plan["segments"]
                if segment["segment_type"] == "GRASP"
            )["waypoints"][-1]
            for joint_name, value in zip(
                plan["joint_names"], grasp["joint_positions_rad"], strict=True
            ):
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)
            mujoco.mj_forward(model, data)

        body_id = int(env.obj_body_id["ladle"])
        body_position = np.asarray(data.xpos[body_id], dtype=float)
        body_rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
        print("ladle body world", body_position.tolist())
        print("ladle body mass kg", float(model.body_mass[body_id]))

        component_vertices: list[np.ndarray] = []
        for geom_id in range(model.ngeom):
            name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not name.startswith("ladle_g"):
                continue
            world = _mesh_world_vertices(model, data, geom_id)
            if world is None:
                continue
            local = (world - body_position) @ body_rotation
            component_vertices.append(local)
            print(
                name,
                "min",
                np.min(local, axis=0).round(6).tolist(),
                "max",
                np.max(local, axis=0).round(6).tolist(),
                "center",
                np.mean(local, axis=0).round(6).tolist(),
                "world_z",
                [round(float(np.min(world[:, 2])), 6), round(float(np.max(world[:, 2])), 6)],
            )
            if name == "ladle_g2":
                for y_center in (-0.05, -0.04, -0.03, -0.02, 0.0, 0.02, 0.04):
                    selected = local[
                        np.abs(local[:, 1] - y_center) <= 0.005
                    ]
                    if len(selected):
                        print(
                            " ",
                            "slice_y",
                            y_center,
                            "min",
                            np.min(selected, axis=0).round(6).tolist(),
                            "max",
                            np.max(selected, axis=0).round(6).tolist(),
                            "mean",
                            np.mean(selected, axis=0).round(6).tolist(),
                        )
        vertices = np.concatenate(component_vertices, axis=0)
        print(
            "ladle union min/max",
            np.min(vertices, axis=0).round(6).tolist(),
            np.max(vertices, axis=0).round(6).tolist(),
        )

        grippers = env.robots[0].gripper
        gripper = grippers["right"]
        site_name = gripper.important_sites["grip_site"]
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        site_position = np.asarray(data.site_xpos[site_id], dtype=float)
        site_rotation = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
        print("grip site", site_name, site_position.round(6).tolist())
        print("grip axes columns", site_rotation.round(6).tolist())
        for geom_id in range(model.ngeom):
            name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if "fingerpad_collision" not in name:
                continue
            position = np.asarray(data.geom_xpos[geom_id], dtype=float)
            in_site = (position - site_position) @ site_rotation
            local_points = _mesh_world_vertices(model, data, geom_id)
            if local_points is None:
                geom_type = int(model.geom_type[geom_id])
                if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                    sx, sy, sz = np.asarray(model.geom_size[geom_id], dtype=float)
                    corners = np.asarray(
                        [
                            (x * sx, y * sy, z * sz)
                            for x in (-1.0, 1.0)
                            for y in (-1.0, 1.0)
                            for z in (-1.0, 1.0)
                        ]
                    )
                    geom_rotation = np.asarray(
                        data.geom_xmat[geom_id], dtype=float
                    ).reshape(3, 3)
                    local_points = position + corners @ geom_rotation.T
            bounds = ""
            if local_points is not None:
                points_in_site = (local_points - site_position) @ site_rotation
                bounds = (
                    f" bounds_in_grip_site="
                    f"{np.min(points_in_site, axis=0).round(6).tolist()}.."
                    f"{np.max(points_in_site, axis=0).round(6).tolist()}"
                )
            print(
                name,
                "world",
                position.round(6).tolist(),
                "in_grip_site",
                in_site.round(6).tolist(),
                "contact_bits",
                (int(model.geom_contype[geom_id]), int(model.geom_conaffinity[geom_id])),
                bounds,
            )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
