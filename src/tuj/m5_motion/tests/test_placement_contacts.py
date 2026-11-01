import mujoco
import numpy as np
import pytest
from tuj.m5_motion.scripted_grasps.placement_contacts import measure_contacts


def test_contact_evidence_is_measured_and_does_not_mutate_state():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001"/>
      <worldbody><geom name="support" type="box" size=".2 .2 .01"/>
        <body name="fruit" pos="0 0 .06"><freejoint/>
          <geom name="fruit_geom" type="sphere" size=".05" mass=".1"/>
        </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    objects = {'fruit': {'geom_names': ['fruit_geom'], 'body_name': 'fruit'},
               'plate': {'geom_names': ['support']}}
    checks = [{'target': 'fruit', 'region': 'plate'}]
    before = data.qpos.copy(), data.qvel.copy(), data.time
    result = measure_contacts(model, data, objects, checks)[0]
    assert sum(c['upward_force_n'] for c in result['contacts']) == pytest.approx(.981, abs=.01)
    assert all(-.005 < c['distance_m'] <= 0 for c in result['contacts'])
    assert np.linalg.norm(result['target_velocity']['linear_m_s']) < 1e-5
    np.testing.assert_array_equal(data.qpos, before[0])
    np.testing.assert_array_equal(data.qvel, before[1])
    assert data.time == before[2]
    data.qpos[2] += .1
    mujoco.mj_forward(model, data)
    assert measure_contacts(model, data, objects, checks)[0]['contacts'] == []


def test_missing_geometry_evidence_does_not_mask_original_goal_failure():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    data = mujoco.MjData(model)
    result = measure_contacts(model, data, {}, [{'target': 'fruit', 'region': 'plate'}])
    assert result[0]['available'] is False
    assert result[0]['reason'] == 'missing measured object'


def wall_friction_scene(*, frictional=True, yaw=0.):
    from scipy.spatial.transform import Rotation
    quat = Rotation.from_euler('z', yaw).as_quat()[[3, 0, 1, 2]]
    rotation = Rotation.from_euler('z', yaw).as_matrix()
    positions = [rotation @ [x, 0., .2] for x in (-.0598, .0598)]
    vector = lambda value: ' '.join(map(str, value))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><option timestep=".001"/>
      <default><geom friction="1 0 0" condim="{3 if frictional else 1}" solref=".005 1"/></default>
      <worldbody>
      <geom name="left" type="box" pos="{vector(positions[0])}" quat="{vector(quat)}" size=".01 .2 .2"/>
      <geom name="right" type="box" pos="{vector(positions[1])}" quat="{vector(quat)}" size=".01 .2 .2"/>
      <body name="payload" pos="0 0 .2" quat="{vector(quat)}"><freejoint/>
        <geom name="payload_geom" type="box" size=".05 .04 .04" mass=".1"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    for _ in range(1000 if frictional else 100):
        mujoco.mj_step(model, data)
    objects = {'payload': {'geom_names': ['payload_geom'], 'body_name': 'payload'},
               'walls': {'geom_names': ['left', 'right']}}
    return model, data, objects


@pytest.mark.parametrize('yaw', [0., .7])
def test_vertical_wall_friction_is_upward_load_and_pair_reversal_changes_sign(yaw):
    model, data, objects = wall_friction_scene(yaw=yaw)
    before = data.qpos.copy(), data.qvel.copy(), data.time
    evidence = measure_contacts(model, data, objects, [
        {'target': 'payload', 'region': 'walls'},
        {'target': 'walls', 'region': 'payload'}])
    upward = sum(c['upward_force_n'] for c in evidence[0]['contacts'])
    assert upward == pytest.approx(.981, abs=.005)
    assert sum(c['normal_upward_force_n'] for c in evidence[0]['contacts']) == pytest.approx(0., abs=1e-8)
    assert sum(c['upward_force_n'] for c in evidence[1]['contacts']) == pytest.approx(-upward)
    for positive, negative in zip(evidence[0]['contacts'], evidence[1]['contacts']):
        np.testing.assert_allclose(positive['force_on_target_world_n'],
                                   -np.array(negative['force_on_target_world_n']))
        assert positive['force_basis'] == 'FULL_CONTACT_FORCE_WORLD'
    np.testing.assert_array_equal(data.qpos, before[0])
    np.testing.assert_array_equal(data.qvel, before[1])
    assert data.time == before[2]


def test_frictionless_vertical_contacts_have_no_upward_load():
    model, data, objects = wall_friction_scene(frictional=False)
    evidence = measure_contacts(model, data, objects, [{'target': 'payload', 'region': 'walls'}])[0]
    assert evidence['contacts']
    assert all(c['upward_force_n'] == 0. for c in evidence['contacts'])
    assert data.qvel[2] < -.9
