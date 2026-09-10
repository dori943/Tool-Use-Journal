"""Payload-derived base adhesion for a force-based, breakable grasp."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class AdhesionLoad:
    mass_kg: float
    base_force_n: float
    capacity_n: float
    command: float


def payload_adhesion_load(model: mujoco.MjModel, object_body: int, ee_body: int) -> AdhesionLoad:
    """Use weight as base suction, leaving dynamic retention to the grasp model.

    This is not a guarantee of dynamic load capacity. The physical attachment's
    contact, slip and wrench limits must still be enforced during execution.
    Model gains and controller ranges are never changed.
    """
    def subtree(root):
        if not 0 < root < model.nbody:
            raise ValueError("adhesion requires a bound object and mounted EE")
        result = {root}
        for body in range(root + 1, model.nbody):
            if int(model.body_parentid[body]) in result:
                result.add(body)
        return result

    payload, mounted = subtree(object_body), subtree(ee_body)
    if payload & mounted:
        raise ValueError("adhesion payload must be separate from mounted EE")
    mass = sum(float(model.body_mass[body]) for body in payload)
    force = mass * float(np.linalg.norm(model.opt.gravity))
    capacities = []
    for actuator in range(model.nu):
        if (int(model.actuator_trntype[actuator]) != int(mujoco.mjtTrn.mjTRN_BODY)
                or int(model.actuator_trnid[actuator, 0]) not in mounted):
            continue
        if (int(model.actuator_gaintype[actuator]) != int(mujoco.mjtGain.mjGAIN_FIXED)
                or int(model.actuator_biastype[actuator]) != int(mujoco.mjtBias.mjBIAS_NONE)
                or int(model.actuator_dyntype[actuator]) != int(mujoco.mjtDyn.mjDYN_NONE)):
            raise ValueError("payload adhesion requires a fixed native adhesion actuator")
        low, high = model.actuator_ctrlrange[actuator]
        capacity = float(high * model.actuator_gainprm[actuator, 0])
        if low != 0 or high <= 0 or capacity <= 0:
            raise ValueError("adhesion controller must map zero to off and positive force to on")
        limit = (float(model.actuator_forcerange[actuator, 1])
                 if model.actuator_forcelimited[actuator] else capacity)
        capacities.append((capacity, min(capacity, limit)))
    if len(capacities) != 1:
        raise ValueError("payload adhesion requires one bound native adhesion actuator")
    raw_capacity, capacity = capacities[0]
    if not np.isfinite([mass, force, capacity]).all() or force <= 0 or capacity <= 0:
        raise ValueError("positive payload weight and native adhesion capacity are required")
    if force > capacity:
        raise ValueError("payload weight exceeds native adhesion capacity")
    return AdhesionLoad(mass, force, capacity, 2 * force / raw_capacity - 1)
