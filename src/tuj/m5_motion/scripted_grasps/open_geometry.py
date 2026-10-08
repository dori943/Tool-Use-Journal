"""Resolve public gripper opening commands to constrained joint geometry."""
from copy import deepcopy
import numpy as np
import mujoco
from scipy.optimize import least_squares
from tuj.m5_motion.gripper_release import open_action_endpoint


def _linkage_equality_ids(model, passive):
    """Equality rows the passive joints can actually satisfy.

    A scene weld or an unrelated joint constraint stays in ``efc_pos`` and does
    not move when the finger linkage does. Including it makes the open-pose
    solve stop on a residual the hand cannot change.
    """

    passive_ids = {int(jid) for jid in passive}
    tendon_joints = []
    for tendon_id in range(int(model.ntendon)):
        start = int(model.tendon_adr[tendon_id])
        count = int(model.tendon_num[tendon_id])
        joints = []
        for wrap in range(start, start + count):
            if int(model.wrap_type[wrap]) == int(mujoco.mjtWrap.mjWRAP_JOINT):
                joints.append(int(model.wrap_objid[wrap]))
        tendon_joints.append(joints)
    selected = []
    for equality_id in range(int(model.neq)):
        kind = int(model.eq_type[equality_id])
        objects = [int(model.eq_obj1id[equality_id]), int(model.eq_obj2id[equality_id])]
        objects = [item for item in objects if item >= 0]
        if kind == int(mujoco.mjtEq.mjEQ_JOINT):
            involved = passive_ids.intersection(objects)
        elif kind == int(mujoco.mjtEq.mjEQ_TENDON):
            involved = passive_ids.intersection(
                joint for tendon_id in objects for joint in tendon_joints[tendon_id]
            )
        else:
            involved = False
        if involved:
            selected.append(equality_id)
    return np.asarray(selected, dtype=int)


def open_joint_positions(context):
    c = context
    cached = getattr(c, '_resolved_open_joint_positions', None)
    if cached is not None:
        return dict(cached)
    if not c.gripper.joints:
        # Rigid tools do not change shape on release. Body transmissions apply
        # adhesion forces; other actuator kinds need an explicit geometry model.
        if any(c.model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_BODY
               for aid in c.gripper_actuator_ids):
            raise ValueError('OPEN_GEOMETRY_UNSUPPORTED_RIGID_TRANSMISSION')
        c._resolved_open_joint_positions = {}
        return {}
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
    linkage_ids = _linkage_equality_ids(model, passive)

    def residual(q):
        probe.qpos[addresses] = q
        mujoco.mj_forward(model, probe)
        equality = np.asarray(probe.efc_type) == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        if linkage_ids.size:
            equality &= np.isin(np.asarray(probe.efc_id), linkage_ids)
        return probe.efc_pos[equality].copy()

    if passive:
        lower = np.where(model.jnt_limited[passive], model.jnt_range[passive, 0], -np.inf)
        upper = np.where(model.jnt_limited[passive], model.jnt_range[passive, 1], np.inf)
        observed_passive = probe.qpos[addresses].copy()
        initial = np.clip(probe.qpos[addresses], lower, upper)
        if not len(residual(initial)):
            raise ValueError('OPEN_GEOMETRY_PASSIVE_CONSTRAINT_REQUIRED')
        # At a joint bound, the scaled gradient can vanish before the linkage
        # residual meets our geometric tolerance. Keep the residual gate below.
        solved = least_squares(residual, initial, bounds=(lower, upper), max_nfev=1000,
                               ftol=1e-12, xtol=1e-12, gtol=None)
        residual_max = float(np.max(np.abs(residual(solved.x))))
        if not solved.success or residual_max > 1e-8:
            observed_violation = np.maximum(
                np.maximum(lower - observed_passive, observed_passive - upper), 0.0
            )
            reference = np.asarray(model.qpos0[addresses], dtype=float)
            reference_violation = np.maximum(
                np.maximum(lower - reference, reference - upper), 0.0
            )
            # A tendon equality is written about the stored reference pose.
            # That pose can already sit outside the joint range, so the open
            # linkage may have to stop between the range and the reference.
            # Do not go farther out than either of those, and do not accept a
            # coupling whose only solution lies beyond both.
            allowed_violation = np.maximum(observed_violation, reference_violation)
            if np.any(allowed_violation > 1e-8):
                fallback = least_squares(
                    residual,
                    observed_passive,
                    max_nfev=1000,
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=None,
                )
                fallback_residual = float(
                    np.max(np.abs(residual(fallback.x)))
                )
                fallback_violation = np.maximum(
                    np.maximum(lower - fallback.x, fallback.x - upper), 0.0
                )
                if (
                    fallback.success
                    and fallback_residual <= 1e-8
                    and np.all(fallback_violation <= allowed_violation + 1e-8)
                ):
                    solved = fallback
                    residual_max = fallback_residual
                else:
                    raise ValueError(
                        'OPEN_GEOMETRY_LINKAGE_NOT_SOLVED: '
                        f'status={solved.status} nfev={solved.nfev} '
                        f'max_residual={residual_max:.3e} message={solved.message}'
                    )
            else:
                raise ValueError(
                    'OPEN_GEOMETRY_LINKAGE_NOT_SOLVED: '
                    f'status={solved.status} nfev={solved.nfev} '
                    f'max_residual={residual_max:.3e} message={solved.message}'
                )
    result = {name: float(probe.qpos[model.jnt_qposadr[model.joint(name).id]])
              for name in c.gripper.joints}
    c._resolved_open_joint_positions = result
    return dict(result)
