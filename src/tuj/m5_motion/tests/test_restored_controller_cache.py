"""Named-state transfer must invalidate measured controller state, not its goals."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from robosuite.controllers.parts.generic.joint_pos import JointPositionController

from tuj.m5_motion.runtime_checkpoint import (
    RuntimeCheckpointError, restore_runtime_checkpoint, save_runtime_checkpoint,
)
from tuj.m5_motion.tests.test_tool_use_journal import ARM_JOINTS, _fake_env
from tuj.m5_motion.tool_use_journal_runtime import (
    ToolUseJournalControllerTrajectoryPlayer, ToolUseJournalEERuntime,
    ToolUseJournalRuntimeError, _capture_runtime_state, _restore_runtime_state,
)


class _ModelProxy:
    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    def joint_id2name(self, index):
        return mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, index)


class _DataProxy:
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        return getattr(self._data, name)


def controlled_env(ee):
    env = _fake_env(ee)
    model, data = env.sim.model._model, env.sim.data._data
    env.sim.model, env.sim.data = _ModelProxy(model), _DataProxy(data)
    env.sim.forward = lambda: mujoco.mj_forward(model, data)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
    controller = JointPositionController(
        env.sim, {'joints': ids, 'qpos': list(model.jnt_qposadr[ids]),
                  'qvel': list(model.jnt_dofadr[ids])},
        (-np.full(6, 150.), np.full(6, 150.)), input_type='absolute',
    )
    controller.goal_qpos = np.linspace(.01, .06, 6)
    robot = env.robots[0]
    robot.part_controllers = {'right': controller}
    robot.composite_controller = SimpleNamespace(_action_split_indexes={'right': (0, 6)})
    robot.action_dim = 6
    return env


def moved_state(env):
    c = env.robots[0].part_controllers['right']
    env.sim.data.qpos[c.qpos_index] = [.3, -.2, .15, -.4, .25, -.5]
    env.sim.data.qvel[c.qvel_index] = [.02, -.01, .03, -.02, .01, -.04]
    env.sim.forward()
    return _capture_runtime_state(env)


def first_hold_torque(runtime):
    """Use the production absolute action mapping and installed controller."""
    c = runtime.env.robots[0].part_controllers['right']
    desired = runtime.env.sim.data.qpos[c.qpos_index].copy()
    action = ToolUseJournalControllerTrajectoryPlayer(runtime)._controller_action(
        SimpleNamespace(joint_names=ARM_JOINTS), desired)
    c.set_goal(action)
    return c.run_controller().copy()


def assert_first_hold_uses_restored_measurements(runtime):
    c = runtime.env.robots[0].part_controllers['right']
    physical = _capture_runtime_state(runtime.env)
    first = first_hold_torque(runtime)
    np.testing.assert_array_equal(c.joint_pos, runtime.env.sim.data.qpos[c.qpos_index])
    np.testing.assert_array_equal(c.joint_vel, runtime.env.sim.data.qvel[c.qvel_index])
    c.update(force=True)
    np.testing.assert_allclose(first, first_hold_torque(runtime), rtol=0., atol=1e-12)
    assert _capture_runtime_state(runtime.env) == physical


def test_shared_restore_refreshes_first_torque_without_resetting_goal_or_limits():
    env = controlled_env('2F')
    runtime = ToolUseJournalEERuntime(env, controlled_env)
    try:
        c = env.robots[0].part_controllers['right']
        state = moved_state(env)
        goal, initial, limits = c.goal_qpos.copy(), c.initial_joint.copy(), c.actuator_max.copy()
        assert not c.new_update and not np.array_equal(c.joint_pos, env.sim.data.qpos[c.qpos_index])
        _restore_runtime_state(env, state)
        np.testing.assert_array_equal(c.goal_qpos, goal)
        np.testing.assert_array_equal(c.initial_joint, initial)
        np.testing.assert_array_equal(c.actuator_max, limits)
        assert_first_hold_uses_restored_measurements(runtime)
    finally:
        runtime.close()


def test_checkpoint_restore_refreshes_first_arm_torque(tmp_path):
    source = ToolUseJournalEERuntime(controlled_env('2F'), controlled_env)
    restored = ToolUseJournalEERuntime(controlled_env('2F'), controlled_env)
    try:
        moved_state(source.env)
        path = save_runtime_checkpoint(tmp_path/'arm.json', source)
        restore_runtime_checkpoint(path, restored)
        assert_first_hold_uses_restored_measurements(restored)
    finally:
        source.close(); restored.close()


def test_checkpoint_rollback_refreshes_measurements_after_failed_logical_restore(tmp_path, monkeypatch):
    source = ToolUseJournalEERuntime(controlled_env('2F'), controlled_env)
    restored = ToolUseJournalEERuntime(controlled_env('2F'), controlled_env)
    try:
        moved_state(source.env)
        path = save_runtime_checkpoint(tmp_path/'arm.json', source)
        before = _capture_runtime_state(restored.env)
        def reject(**kwargs):
            # A validator may legitimately read the newly restored state before rejecting it.
            restored.env.robots[0].part_controllers['right'].update(force=True)
            raise RuntimeError('deliberate logical rejection')
        monkeypatch.setattr(restored, 'restore_logical_state', reject)
        with pytest.raises(RuntimeCheckpointError, match='rolled back'):
            restore_runtime_checkpoint(path, restored)
        assert _capture_runtime_state(restored.env) == before
        assert_first_hold_uses_restored_measurements(restored)
    finally:
        source.close(); restored.close()


@pytest.mark.parametrize('failed_transition', [False, True])
def test_ee_transition_and_rollback_refresh_first_arm_torque(failed_transition):
    def factory(ee):
        return controlled_env('3F' if failed_transition and ee is None else ee)
    runtime = ToolUseJournalEERuntime(controlled_env('2F'), factory)
    try:
        before = moved_state(runtime.env)
        if failed_transition:
            with pytest.raises(ToolUseJournalRuntimeError, match='factory requested'):
                runtime.unlock('2F')
            assert runtime.active_ee == '2F'
        else:
            runtime.unlock('2F')
            assert runtime.active_ee is None
        after = _capture_runtime_state(runtime.env)
        for name in ARM_JOINTS:
            assert after.qpos_by_joint[name] == before.qpos_by_joint[name]
            assert after.qvel_by_joint[name] == before.qvel_by_joint[name]
        assert_first_hold_uses_restored_measurements(runtime)
    finally:
        runtime.close()
