"""Copy actual configuration into isolated collision data by named bindings."""

import mujoco
import numpy as np


def copy_runtime_configuration(source_model, source_data, target_model, target_data):
    """Fail closed if compiled collision geometry cannot bind the live state.

    This runs after planned context transforms. In particular, an attached free
    body is checked at its actual pose, including compliance or grasp slip.
    Neither the live data nor the stored planning context is modified.
    """
    if source_data is target_data:
        raise ValueError("runtime collision data must be isolated")
    qpos = target_data.qpos.copy()
    for joint in range(target_model.njnt):
        name = mujoco.mj_id2name(target_model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        live = -1 if name is None else mujoco.mj_name2id(
            source_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        kind = int(target_model.jnt_type[joint])
        if live < 0 or int(source_model.jnt_type[live]) != kind:
            raise ValueError(f"runtime joint binding unavailable: {name!r}")
        width = 7 if kind == int(mujoco.mjtJoint.mjJNT_FREE) else (
            4 if kind == int(mujoco.mjtJoint.mjJNT_BALL) else 1
        )
        src, dst = int(source_model.jnt_qposadr[live]), int(target_model.jnt_qposadr[joint])
        qpos[dst:dst + width] = source_data.qpos[src:src + width]
    positions, quaternions = target_data.mocap_pos.copy(), target_data.mocap_quat.copy()
    for body in range(target_model.nbody):
        dst = int(target_model.body_mocapid[body])
        if dst < 0:
            continue
        name = mujoco.mj_id2name(target_model, mujoco.mjtObj.mjOBJ_BODY, body)
        live = -1 if name is None else mujoco.mj_name2id(
            source_model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        src = -1 if live < 0 else int(source_model.body_mocapid[live])
        if src < 0:
            raise ValueError(f"runtime mocap binding unavailable: {name!r}")
        positions[dst], quaternions[dst] = source_data.mocap_pos[src], source_data.mocap_quat[src]
    if not all(np.isfinite(values).all() for values in (qpos, positions, quaternions)):
        raise ValueError("runtime configuration contains non-finite values")
    target_data.qpos[:] = qpos
    target_data.mocap_pos[:] = positions
    target_data.mocap_quat[:] = quaternions
