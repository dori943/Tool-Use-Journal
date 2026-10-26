"""Read-only physical contact evidence for completed placement checks.

This telemetry never overrides a goal verdict or changes simulation state.
"""
import mujoco
import numpy as np


def measure_contacts(model, data, objects, checks):
    result = []
    for check in checks:
        target, region = check['target'], check['region']
        if target not in objects or region not in objects:
            result.append({'target': target, 'region': region, 'available': False,
                           'reason': 'missing measured object', 'diagnostic_only': True})
            continue
        groups = []
        for name in (target, region):
            ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                   for n in objects[name].get('geom_names', [])}
            groups.append({i for i in ids if i >= 0 and
                           (model.geom_contype[i] or model.geom_conaffinity[i])})
        samples = []
        for index, contact in enumerate(data.contact[:data.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            if a in groups[0] and b in groups[1]:
                sign = -1.
            elif b in groups[0] and a in groups[1]:
                sign = 1.
            else:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            normal = sign * np.asarray(contact.frame[:3])
            samples.append({'distance_m': float(contact.dist),
                            'normal_on_target_world': normal.tolist(),
                            'normal_force_n': float(force[0]),
                            'upward_force_n': float(force[0] * normal[2])})
        velocity = None
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                objects[target].get('body_name', ''))
        if body >= 0:
            value = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, value, 0)
            velocity = {'angular_rad_s': value[:3].tolist(), 'linear_m_s': value[3:].tolist()}
        result.append({'target': target, 'region': region,
                       'simulation_time_s': float(data.time),
                       'contacts': samples, 'target_velocity': velocity,
                       'diagnostic_only': True})
    return result


def placement_contacts(runtime, world, checks):
    from tuj.m5_motion.tool_use_journal import _raw_model_data, ToolUseJournalCompatibilityError
    try:
        model, data = _raw_model_data(runtime.env)
    except ToolUseJournalCompatibilityError:
        return {'available': False, 'reason': 'runtime has no native simulation data'}
    return measure_contacts(model, data, world.objects, checks)
