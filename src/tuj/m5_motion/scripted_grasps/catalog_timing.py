"""One clock contract for compiled physics, control ticks and measured holds."""
import math
import xml.etree.ElementTree as ET

RUNTIME_VERSION = 'catalog_v2_measured_time'


def timing_xml(xml, timestep_s, integrator):
    """Kitchen merges multiple option elements; the final one wins in MuJoCo."""
    root = ET.fromstring(xml)
    options = root.findall('option')
    if not options:
        options = [ET.SubElement(root, 'option')]
    for option in options:
        option.set('timestep', str(timestep_s))
        option.set('integrator', integrator)
    return ET.tostring(root, encoding='unicode')


def synchronize_timing(env, recipe):
    actual = float(env.sim.model.opt.timestep)
    if not math.isclose(actual, recipe.physics_timestep_s, abs_tol=1e-12):
        raise RuntimeError(f'COMPILED_TIMESTEP_MISMATCH: {actual}')
    if not math.isclose(float(env.control_timestep), .02, abs_tol=1e-12):
        raise RuntimeError('CATALOG_REQUIRES_50_HZ_CONTROL')
    env.model_timestep = actual
    return {'physics_timestep_s': actual, 'control_timestep_s': float(env.control_timestep),
            'runtime_version': RUNTIME_VERSION, 'clock_source': 'MUJOCO_DATA_TIME'}


def check_control_elapsed(start_s, end_s, expected_s):
    elapsed = end_s - start_s
    if not math.isclose(elapsed, expected_s, rel_tol=0., abs_tol=1e-8):
        raise RuntimeError(f'CONTROL_TIME_MISMATCH: expected {expected_s}, actual {elapsed}')
    return elapsed


def run_timed_hold(context, q, opening, duration_s):
    start = float(context.data.time)
    rows = []
    while float(context.data.time) - start < duration_s - 1e-9:
        rows.append(context.step(q, opening))
    return rows, float(context.data.time) - start
