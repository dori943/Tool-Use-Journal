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
@pytest.mark.parametrize('bounded', [False, True])
def test_retention_feedback_depends_on_actual_object_constraint(attachment, constrained, explicit_hold, bounded):
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
    if bounded:
        retention.context.three_finger_hold_command_min = np.full(3, .096)
        retention.context.three_finger_hold_command_max = np.full(3, .104)
    action = retention.before_tick(np.ones(9))
    if explicit_hold and constrained:
        expected = [.1, .1, .1]
    elif explicit_hold or constrained:
        expected = [.1, .094, .1]
    else:
        expected = [.1, .094, .106]
    if bounded and not (explicit_hold and constrained):
        expected = np.clip(expected, .096, .104)
    np.testing.assert_allclose(retention.commands, expected)
    np.testing.assert_array_equal(action[:6], np.ones(6))
    np.testing.assert_array_equal(action[6:], np.zeros(3))
    if constrained and not explicit_hold:
        # Once contact returns, even a force below target must not keep closing.
        retention.forces = dict(thumb=1., index=1., pinky=1.)
        retention.before_tick(np.ones(9))
        np.testing.assert_allclose(retention.commands, expected)


def test_scene_instance_kinematic_hold_stays_frozen():
    env = object()
    retention = GraspRetention.__new__(GraspRetention)
    retention.entry = SimpleNamespace(
        ee="3F", driver="catalog", object_id="fruit", scene_object_id="fruit_a")
    retention.commands = np.array([.1, .1, .1])
    retention.forces = dict(thumb=6., index=0., pinky=6.)
    retention.context = SimpleNamespace(
        runtime=SimpleNamespace(
            env=env,
            attachment=SimpleNamespace(object_id="fruit_a", mode="KINEMATIC")),
        env=env,
        recipe=SimpleNamespace(
            three_finger_force_targets_n=(6., 3., 3.),
            three_finger_force_gain=.002,
            hold_finger_positions=True),
        three_finger_force_hold=True,
        gripper=SimpleNamespace(current_action=None),
        robot=SimpleNamespace(composite_controller=SimpleNamespace(
            _action_split_indexes={"right_gripper": (6, 9)})))
    retention.before_tick(np.ones(9))
    np.testing.assert_allclose(retention.commands, [.1, .1, .1])


def _hand_limit_retention(qpos, qvel):
    retention = GraspRetention.__new__(GraspRetention)
    data = SimpleNamespace(qpos=np.asarray(qpos, dtype=float), qvel=np.asarray(qvel, dtype=float))
    model = SimpleNamespace(
        jnt_limited=np.array([1]),
        jnt_range=np.array([[0.0, 1.51]]),
        jnt_qposadr=np.array([0]),
        jnt_dofadr=np.array([0]),
    )
    seen = []
    retention.context = SimpleNamespace(
        hand_joint_ids=[0],
        model=model,
        data=data,
        recipe=SimpleNamespace(maximum_joint_limit_error_rad=0.01),
        audit_hand_range=lambda: seen.append(float(data.qpos[0])),
    )
    return retention, data, seen


def test_retention_projects_a_finger_stop_spike_before_the_audit():
    retention, data, seen = _hand_limit_retention([1.533659], [2.0])
    retention.audit_substep()
    assert data.qpos[0] == pytest.approx(1.51)
    assert data.qvel[0] == 0.0
    assert seen == [pytest.approx(1.51)]


def test_retention_keeps_a_finger_residual_inside_the_allowance():
    retention, data, seen = _hand_limit_retention([1.515], [0.2])
    retention.audit_substep()
    assert data.qpos[0] == pytest.approx(1.515)
    assert data.qvel[0] == pytest.approx(0.2)
    assert seen == [pytest.approx(1.515)]
