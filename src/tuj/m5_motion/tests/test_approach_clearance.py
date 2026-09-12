from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from tuj.m5_motion.scripted_grasps.spoon_runtime import SpoonContext


def context(margin=.005):
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <worldbody>
        <body name="robot"><joint name="arm" type="slide" axis="1 0 0"/>
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <body name="finger"><joint name="preshape" type="slide" axis="1 0 0"/>
            <geom name="tip" type="box" size=".01 .01 .01"/>
          </body>
        </body>
        <geom name="obstacle" type="box" pos=".04 0 0" size=".01 .01 .01"/>
      </worldbody></mujoco>''')
    c = SpoonContext.__new__(SpoonContext)
    c.model, c.data, c.probe = model, mujoco.MjData(model), mujoco.MjData(model)
    c.mj = mujoco
    c.arm_ids = np.array([model.jnt_qposadr[model.joint('arm').id]])
    c.robot = SimpleNamespace(robot_model=SimpleNamespace(joints=['arm'], root_body='robot'))
    c.request_collision_margin_m = margin
    c.fixed_mount_pairs = set()
    c.robot_geoms = c.gripper_geoms = c.finger_geoms = {model.geom('tip').id}
    c.object_geoms = c.handle_geoms = {model.geom('obstacle').id}
    c.carried_pose = None
    mujoco.mj_forward(model, c.data)
    return c


@pytest.mark.parametrize('stage', ['PRE_GRASP', 'PRE_CLEARANCE'])
@pytest.mark.parametrize('q, expected', [(.014, True), (.017, False), (.020, False), (.023, False)])
def test_approach_honors_positive_request_clearance(stage, q, expected):
    c = context()
    c._prepare_approach_clearance(stage)
    assert c.valid_state([q], SimpleNamespace(keyframe_id=stage)) is expected
    if not expected:
        assert any(set(record['geoms']) == {'tip', 'obstacle'} for record in c.last_planning_collision)


@pytest.mark.parametrize('stage, margin', [('GRASP', .005), ('LIFT', .005), ('PRE_GRASP', 0.)])
def test_other_stages_and_zero_request_margin_keep_legacy_behavior(stage, margin):
    c = context(margin)
    c._prepare_approach_clearance(stage)
    assert c._approach_clearance is None
    assert c.valid_state([.017], SimpleNamespace(keyframe_id=stage))


def test_planning_rebuilds_actual_preshape_and_preserves_live_state():
    c = context()
    key = SimpleNamespace(keyframe_id='PRE_GRASP')
    c._prepare_approach_clearance('PRE_GRASP')
    first = c._approach_clearance
    assert c.valid_state([.010], key)
    c.data.qpos[c.model.jnt_qposadr[c.model.joint('preshape').id]] = .010
    c.data.qvel[:] = [.1, .2]
    c.data.time = 3.25
    c._prepare_approach_clearance('PRE_GRASP')
    assert c._approach_clearance is not first
    arrays = {name: getattr(c.data, name).copy() for name in ('qpos', 'qvel', 'ctrl')}
    model_arrays = {name: getattr(c.model, name).copy()
                    for name in ('geom_margin', 'flex_margin', 'geom_contype', 'geom_conaffinity')}
    assert not c.valid_state([.010], key)
    for name, value in arrays.items():
        np.testing.assert_array_equal(getattr(c.data, name), value)
    for name, value in model_arrays.items():
        np.testing.assert_array_equal(getattr(c.model, name), value)
    assert c.data.time == 3.25
    c._prepare_approach_clearance('GRASP')
    assert c._approach_clearance is None


@pytest.mark.parametrize('margin', [-.001, float('nan')])
def test_invalid_margin_fails_closed(margin):
    c = context(margin)
    with pytest.raises(ValueError):
        c._prepare_approach_clearance('PRE_GRASP')


def test_existing_contact_rejection_is_not_overridden_by_margin_validator():
    c = context()
    c._prepare_approach_clearance('PRE_GRASP')
    c.bad_contacts = lambda data, stage: [{'geoms': ['existing', 'violation']}]
    assert not c.valid_state([0.], SimpleNamespace(keyframe_id='PRE_GRASP'))
    assert c.last_planning_collision == [{'geoms': ['existing', 'violation']}]
