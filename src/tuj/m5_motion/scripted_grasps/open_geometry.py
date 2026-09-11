"""Resolve public gripper opening commands to constrained joint geometry."""
from copy import deepcopy
import numpy as np
import mujoco
from scipy.optimize import least_squares
from tuj.m5_motion.gripper_release import open_action_endpoint


def open_joint_positions(context):
    c = context
    cached = getattr(c, '_resolved_open_joint_positions', None)
    if cached is not None:
        return dict(cached)
    endpoint, _ = open_action_endpoint(c.gripper, -1.)
    hand = deepcopy(c.gripper)
    hand.current_action = endpoint.copy()
    formatted = np.asarray(hand.format_action(np.zeros(hand.dof)), dtype=float)
    if len(formatted) != len(c.gripper_actuator_ids):
        raise ValueError('OPEN_GEOMETRY_ACTUATOR_DIMENSION_MISMATCH')
    model = c.model
    probe = mujoco.MjData(model)
    probe.qpos[:] = c.data.qpos
    joint_ids = [model.joint(name).id for name in c.gripper.joints]
    driven = []
    for value, aid in zip(formatted, c.gripper_actuator_ids):
        if (model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT
                or not np.isclose(model.actuator_gear[aid, 0], 1.)
                or model.actuator_biastype[aid] != mujoco.mjtBias.mjBIAS_AFFINE
                or not np.isclose(model.actuator_biasprm[aid, 1], -model.actuator_gainprm[aid, 0])):
            raise ValueError('OPEN_GEOMETRY_UNSUPPORTED_POSITION_TRANSMISSION')
        jid = int(model.actuator_trnid[aid, 0])
        if jid not in joint_ids:
            raise ValueError('OPEN_GEOMETRY_ACTUATOR_OUTSIDE_HAND')
        lo, hi = model.actuator_ctrlrange[aid]
        target = float(lo + (value + 1.) * .5 * (hi - lo))
        if model.jnt_limited[jid] and not model.jnt_range[jid, 0] <= target <= model.jnt_range[jid, 1]:
            raise ValueError('OPEN_GEOMETRY_TARGET_OUTSIDE_JOINT_RANGE')
        probe.qpos[model.jnt_qposadr[jid]] = target
        driven.append(jid)
    passive = [jid for jid in joint_ids if jid not in driven]
    addresses = model.jnt_qposadr[passive]

    def residual(q):
        probe.qpos[addresses] = q
        mujoco.mj_forward(model, probe)
        return probe.efc_pos[np.asarray(probe.efc_type) == mujoco.mjtConstraint.mjCNSTR_EQUALITY].copy()

    if passive:
        lower = np.where(model.jnt_limited[passive], model.jnt_range[passive, 0], -np.inf)
        upper = np.where(model.jnt_limited[passive], model.jnt_range[passive, 1], np.inf)
        initial = np.clip(probe.qpos[addresses], lower, upper)
        if not len(residual(initial)):
            raise ValueError('OPEN_GEOMETRY_PASSIVE_CONSTRAINT_REQUIRED')
        solved = least_squares(residual, initial, bounds=(lower, upper), max_nfev=100,
                               ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if not solved.success or np.max(np.abs(residual(solved.x))) > 1e-8:
            raise ValueError('OPEN_GEOMETRY_LINKAGE_NOT_SOLVED')
    result = {name: float(probe.qpos[model.jnt_qposadr[model.joint(name).id]])
              for name in c.gripper.joints}
    c._resolved_open_joint_positions = result
    return dict(result)
