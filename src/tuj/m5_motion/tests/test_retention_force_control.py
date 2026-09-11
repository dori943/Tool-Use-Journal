from types import SimpleNamespace

import numpy as np
import pytest

from tuj.m5_motion.scripted_grasps.retention import GraspRetention


@pytest.mark.parametrize('attachment, constrained', [
    (None, False),
    (SimpleNamespace(object_id='held', mode='BREAKABLE_WELD'), False),
    (SimpleNamespace(object_id='other', mode='KINEMATIC'), False),
    (SimpleNamespace(object_id='held', mode='KINEMATIC'), True),
])
@pytest.mark.parametrize('explicit_hold', [False, True])
def test_retention_feedback_depends_on_actual_object_constraint(attachment, constrained, explicit_hold):
    env = object()
    retention = GraspRetention.__new__(GraspRetention)
    retention.entry = SimpleNamespace(ee='3F', driver='catalog', object_id='held')
    retention.commands = np.array([.1, .1, .1])
    retention.forces = dict(thumb=6., index=0., pinky=6.)
    retention.context = SimpleNamespace(
        runtime=SimpleNamespace(env=env, attachment=attachment), env=env,
        recipe=SimpleNamespace(three_finger_force_targets_n=(6., 3., 3.),
                               three_finger_force_gain=.002, hold_finger_positions=explicit_hold),
        three_finger_force_hold=True,
        gripper=SimpleNamespace(current_action=None),
        robot=SimpleNamespace(composite_controller=SimpleNamespace(
            _action_split_indexes={'right_gripper': (6, 9)})))
    action = retention.before_tick(np.ones(9))
    expected = [.1, .1, .1] if explicit_hold else ([.1, .094, .1] if constrained else [.1, .094, .106])
    np.testing.assert_allclose(retention.commands, expected)
    np.testing.assert_array_equal(action[:6], np.ones(6))
    np.testing.assert_array_equal(action[6:], np.zeros(3))
    if constrained and not explicit_hold:
        # Once contact returns, even a force below target must not keep closing.
        retention.forces = dict(thumb=1., index=1., pinky=1.)
        retention.before_tick(np.ones(9))
        np.testing.assert_allclose(retention.commands, expected)
