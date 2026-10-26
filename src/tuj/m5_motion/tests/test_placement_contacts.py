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
