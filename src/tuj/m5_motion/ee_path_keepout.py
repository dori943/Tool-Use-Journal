"""Compute the EE-rack trajectory keep-out from the scene's MuJoCo model.

See :mod:`tuj.common.ee_keepout` for why placements avoid this volume.  The
templates are joint-space and replayed as stored, so forward-evaluating the
arm's own collision meshes in the current model gives exactly where the arm
will be, wherever the rack and robot base sit.  Mounted gripper geometry is
not included: the templates carry an EE only beside the rack, where nothing is
placed.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from tuj.common.ee_keepout import SweptArmHeightField


EE_PATH_KEEPOUT_METADATA_KEY = "ee_path_keepout"
# Spacing of surface samples on the arm meshes.  A flat face wider than the
# object (the wrist end cap over a mug) has no vertex above it, so faces are
# sampled as well as vertices.
SURFACE_SAMPLE_SPACING_M = 0.01
# Joint step between evaluated states.  A link 0.8 m from the pan axis moves
# 16 mm per step, inside sample spacing plus the placement buffer.
SWEEP_JOINT_STEP_RAD = 0.02
CELL_M = 0.005
# Half side of the XY square around the arm base the field covers.  UR5e
# reach is 0.85 m; the base collision mesh adds its own radius.
_ARM_REACH_BOUND_M = 1.3

# Fields are numpy arrays; requests carry only this registry key so they stay
# JSON-serializable and survive the deep copies M5 makes of them.
_REGISTRY: dict[str, SweptArmHeightField] = {}


def register_keepout(key: str, field: SweptArmHeightField) -> None:
    _REGISTRY[key] = field


def keepout_for_task(task: object) -> SweptArmHeightField | None:
    metadata = getattr(task, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    key = metadata.get(EE_PATH_KEEPOUT_METADATA_KEY)
    return _REGISTRY.get(key) if isinstance(key, str) else None


def arm_collision_geom_ids(model, naming_prefix: str) -> list[int]:
    """Collision geoms of the arm itself (``robot0_*``), not the mounted EE."""

    import mujoco

    return [
        geom_id
        for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
        .startswith(naming_prefix)
        and (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id]))
    ]


def _triangle_samples(v0, v1, v2, spacing: float) -> np.ndarray:
    longest = max(
        float(np.linalg.norm(v1 - v0)),
        float(np.linalg.norm(v2 - v1)),
        float(np.linalg.norm(v0 - v2)),
    )
    n = max(1, math.ceil(longest / spacing))
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    keep = (i + j) <= n
    a, b = i[keep, None] / n, j[keep, None] / n
    return v0 + a * (v1 - v0) + b * (v2 - v0)


def geom_local_samples(
    model, geom_id: int, spacing: float = SURFACE_SAMPLE_SPACING_M
) -> np.ndarray:
    """Surface samples in the geom frame (mesh faces, else local AABB surface)."""

    import mujoco

    if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = int(model.geom_dataid[geom_id])
        vadr, vnum = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fadr, fnum = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        verts = np.asarray(model.mesh_vert[vadr:vadr + vnum], dtype=float)
        faces = np.asarray(model.mesh_face[fadr:fadr + fnum], dtype=int)
        samples = np.vstack([
            verts,
            *(_triangle_samples(verts[a], verts[b], verts[c], spacing)
              for a, b, c in faces),
        ])
    else:
        aabb = np.asarray(model.geom_aabb[geom_id], dtype=float)
        center, half = aabb[:3], aabb[3:]
        axes = [
            np.linspace(-h, h, max(2, math.ceil(2 * h / spacing) + 1)) for h in half
        ]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        samples = center + grid[np.any(np.isclose(np.abs(grid), half), axis=1)]
    # Thin to one sample per half-spacing voxel; mesh faces overlap heavily.
    keys = np.floor(samples / (spacing * 0.5)).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return samples[np.sort(first)]


def dense_states(
    paths: Iterable[Sequence[Sequence[float]]], step_rad: float
) -> np.ndarray:
    """Interpolated joint states, de-duplicated across paths."""

    states = []
    for path in paths:
        q = np.asarray(path, dtype=float)
        if len(q) == 0:
            continue
        states.append(q[:1])
        for left, right in zip(q[:-1], q[1:]):
            n = max(1, math.ceil(float(np.max(np.abs(right - left))) / step_rad))
            t = (np.arange(1, n + 1) / n)[:, None]
            states.append(left + (right - left) * t)
    if not states:
        return np.zeros((0, 0))
    stacked = np.vstack(states)
    # A return path is its attach path reversed; evaluate each state once.
    _, first = np.unique(
        np.round(stacked / (step_rad * 0.25)), axis=0, return_index=True
    )
    return stacked[np.sort(first)]


def sweep_height_field(
    model,
    baseline_qpos: Sequence[float],
    joint_names: Sequence[str],
    states: np.ndarray,
    geom_ids: Sequence[int],
    *,
    cell_m: float = CELL_M,
) -> tuple[np.ndarray, np.ndarray]:
    """``(origin_xy, floor_z)`` of the arm surface over ``states``.

    ``geom_ids[0]`` anchors the grid and must be a link that stays near the
    base (the base mesh, first in model order).
    """

    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(baseline_qpos, dtype=float)
    addresses = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"joint {name!r} is absent from the scene model")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    local = [(g, geom_local_samples(model, g)) for g in geom_ids]
    mujoco.mj_kinematics(model, data)
    anchor = np.asarray(data.geom_xpos[geom_ids[0]][:2], dtype=float)
    origin = np.floor((anchor - _ARM_REACH_BOUND_M) / cell_m) * cell_m
    size = int(math.ceil(2 * _ARM_REACH_BOUND_M / cell_m)) + 1
    floor_z = np.full((size, size), math.inf)
    for q in states:
        data.qpos[addresses] = q
        mujoco.mj_kinematics(model, data)
        world = np.vstack([
            points @ np.asarray(data.geom_xmat[g]).reshape(3, 3).T + data.geom_xpos[g]
            for g, points in local
        ])
        index = np.floor((world[:, :2] - origin) / cell_m).astype(np.int64)
        if np.any(index < 0) or np.any(index >= size):
            raise ValueError("arm sample left the reach bound; enlarge it")
        np.minimum.at(floor_z, (index[:, 0], index[:, 1]), world[:, 2])
    return origin, floor_z


def rack_templates(
    environment_name: str,
    world: object,
    attach_registry: object,
    return_registry: object | None,
) -> list[object]:
    """Every stored attach/return template this rack can replay."""

    templates = []
    for ee in getattr(world, "rack", {}) or {}:
        for registry in (attach_registry, return_registry):
            if registry is None:
                continue
            try:
                templates.append(registry.load(environment_name, ee, world=world))
            except Exception:  # noqa: BLE001 - a missing template sweeps nothing
                continue
    return templates


def compute_swept_arm_height_field(
    env: object, templates: Sequence[object]
) -> SweptArmHeightField:
    """Forward-evaluate the arm meshes of ``env`` along ``templates``."""

    model = env.sim.model._model  # type: ignore[attr-defined]
    baseline = np.asarray(env.sim.data._data.qpos, dtype=float)  # type: ignore[attr-defined]
    robot = env.robots[0]  # type: ignore[attr-defined]
    geom_ids = arm_collision_geom_ids(model, str(robot.robot_model.naming_prefix))
    if not geom_ids or not templates:
        raise ValueError("no arm collision geoms or no rack templates to sweep")
    joint_names = list(templates[0].joint_names)
    if any(list(t.joint_names) != joint_names for t in templates):
        raise ValueError("rack templates disagree on joint names")
    states = dense_states(
        (
            [w.joint_positions_rad for w in segment.waypoints]
            for template in templates
            for segment in template.segments
        ),
        SWEEP_JOINT_STEP_RAD,
    )
    origin, floor_z = sweep_height_field(
        model, baseline, joint_names, states, geom_ids
    )
    return SweptArmHeightField(
        origin_xy=origin,
        cell_m=CELL_M,
        floor_z=floor_z,
        trajectory_ids=tuple(t.trajectory_id for t in templates),
    ).cropped()


def compute_rack_keepout(
    env: object,
    world: object,
    environment_name: str,
    registry_root,
) -> SweptArmHeightField:
    """Field for every template the shared registries resolve for this rack."""

    from tuj.m5_motion.precomputed_ee_attach import PrecomputedEEAttachRegistry
    from tuj.m5_motion.precomputed_ee_exchange import PrecomputedEEReturnRegistry

    templates = rack_templates(
        environment_name,
        world,
        PrecomputedEEAttachRegistry(registry_root),
        PrecomputedEEReturnRegistry(registry_root),
    )
    return compute_swept_arm_height_field(env, templates)
