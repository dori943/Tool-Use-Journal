"""Physical contact and restart tests, without a task completion shortcut."""
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

from environments.deformable_material import MaterialParameters
from environments.deformable_runtime import FlexMaterialRuntime, check_native_contact
from environments.objects.deformable_ellipsoid import DeformableEllipsoidObject


def make_runtime():
    obj = DeformableEllipsoidObject('sample', [.06, .06, .061],
                                   MaterialParameters(500, 20000, 1000, 1000))
    root = ET.Element('mujoco')
    ET.SubElement(root, 'option', timestep='.0005', integrator='implicitfast')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'geom', type='plane', size='.2 .2 .01')
    world.append(obj.get_obj())
    plate = ET.SubElement(world, 'body', name='press', mocap='true', pos='0 0 .085')
    ET.SubElement(plate, 'geom', type='box', size='.045 .045 .005')
    ET.SubElement(root, 'deformable').append(obj.flex)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    runtime = FlexMaterialRuntime(model, data, obj)
    runtime.reset([0, 0, .032], [1, 0, 0, 0])
    return runtime


def step(runtime):
    mujoco.mj_step1(runtime.model, runtime.data)
    runtime.prepare_forces()
    mujoco.mj_step2(runtime.model, runtime.data)


def test_contact_compression_leaves_residual_shape_compared_with_no_press():
    results = []
    for pressing in (False, True):
        runtime = make_runtime()
        model, data = runtime.model, runtime.data
        body = model.body('press').id
        mocap = int(model.body_mocapid[body])
        peak_force = 0.
        for _ in range(5000):
            t = float(data.time)
            z = (.085 if t < .3 else .085 - (t - .3) * .06 if t < 1.3
                 else .025 if t < 1.6 else min(.085, .025 + (t - 1.6) * .09))
            data.mocap_pos[mocap, 2] = z if pressing else .085
            step(runtime)
            for i in range(data.ncon):
                contact = data.contact[i]
                if any(g >= 0 and model.geom_bodyid[g] == body
                       for g in (contact.geom1, contact.geom2)):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, i, force)
                    peak_force = max(peak_force, abs(force[0]))
        results.append((runtime.metrics(), peak_force))
    control, pressed = results
    assert control[1] == 0
    assert pressed[1] > 1
    assert pressed[0]['height_m'] < control[0]['height_m'] - .005
    assert abs(pressed[0]['volume_ratio'] - 1) < .01


def test_restore_reproduces_physics_and_reset_clears_plastic_state():
    first = make_runtime()
    for _ in range(500):
        step(first)
    saved = first.state()
    second = make_runtime()
    second.restore(saved)
    for _ in range(50):
        step(first)
        step(second)
    np.testing.assert_allclose(first.data.qpos[first.qpos], second.data.qpos[second.qpos], atol=1e-9)
    np.testing.assert_allclose(first.material.plastic_gradient, second.material.plastic_gradient, atol=1e-9)
    second.reset([1, 2, 3], [1, 0, 0, 0])
    np.testing.assert_allclose(second.material.plastic_gradient, np.tile(np.eye(3), (len(second.material.tetrahedra), 1, 1)))
    np.testing.assert_allclose(second.metrics()['height_m'], .061)


def test_world_forces_rotate_with_material_without_spurious_plasticity():
    from scipy.spatial.transform import Rotation
    original, rotated = make_runtime(), make_runtime()
    rotation = Rotation.from_rotvec([.7, -.4, .2])
    quat = rotation.as_quat()[[3, 0, 1, 2]]
    rotated.reset([1, 2, 3], quat)
    deformation = np.diag([1.1, .95, .96])
    displacement = original.obj.reference_positions @ (deformation - np.eye(3)).T
    for runtime in (original, rotated):
        runtime.data.qpos[runtime.qpos] = displacement.reshape(-1)
        mujoco.mj_forward(runtime.model, runtime.data)
        runtime.prepare_forces()
    np.testing.assert_allclose(rotated.data.xfrc_applied[rotated.bodies, :3],
                               original.data.xfrc_applied[original.bodies, :3] @ rotation.as_matrix().T,
                               atol=1e-9)
    np.testing.assert_allclose(original.material.plastic_gradient, rotated.material.plastic_gradient, atol=1e-10)


def test_contact_query_identifies_flex_and_preserves_unrelated_geom_queries():
    runtime = make_runtime()
    for _ in range(500):
        step(runtime)
    assert check_native_contact(runtime.model, runtime.data, runtime.obj)
    assert not check_native_contact(runtime.model, runtime.data, 'missing_button')
    assert not check_native_contact(runtime.model, runtime.data, runtime.obj, 'missing_button')


def test_contact_force_monitor_rejects_overload_and_ignores_support_only():
    import pytest
    runtime = make_runtime()
    body = runtime.model.body('press').id
    geoms = np.flatnonzero(runtime.model.geom_bodyid == body)
    runtime.begin_contact_measurement(geoms, .01)
    for _ in range(500):
        step(runtime)
        runtime.observe_contacts()
    assert runtime.contact_measurement['impulse_ns'] == 0
    runtime.data.mocap_pos[int(runtime.model.body_mocapid[body]), 2] = .04
    with pytest.raises(ValueError, match='TOOL_CONTACT_FORCE_LIMIT'):
        for _ in range(50):
            step(runtime)
            runtime.observe_contacts()

def test_ee_contact_is_separate_from_tool_and_counts_toward_force_limit():
    import pytest
    runtime = make_runtime()
    body = runtime.model.body('press').id
    geoms = np.flatnonzero(runtime.model.geom_bodyid == body)
    runtime.begin_contact_measurement([], .01, ee_geoms=geoms)
    runtime.data.mocap_pos[int(runtime.model.body_mocapid[body]), 2] = .04
    with pytest.raises(ValueError, match='TOOL_CONTACT_FORCE_LIMIT'):
        for _ in range(100):
            step(runtime)
            runtime.observe_contacts()
    contact = runtime.contact_measurement
    assert contact['impulse_ns'] == 0
    assert contact['ee_impulse_ns'] > 0
    assert contact['combined_peak_force_n'] == contact['ee_peak_force_n']
    assert contact['unloaded_duration_s'] == 0
