"""Checkpoint continuity must include the hand's incremental target, not just ctrl."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from robosuite.controllers.parts.gripper.simple_grip import SimpleGripController
from robosuite.models.grippers import Robotiq85Gripper, JacoThreeFingerDexterousGripper

from tuj.m5_motion.runtime_checkpoint import (
    RuntimeCheckpointError, capture_runtime_checkpoint, restore_runtime_checkpoint,
    save_runtime_checkpoint,
)
from tuj.m5_motion.tests.test_elastic_checkpoint import elastic_runtime
from tuj.m5_motion.tests.test_tool_use_journal import ARM_JOINTS, _fake_env
from tuj.m5_motion.tool_use_journal_runtime import (
    ToolUseJournalControllerTrajectoryPlayer, ToolUseJournalEERuntime,
)


def runtime_with_hand(action, *, held=False, ee='2F'):
    runtime = (elastic_runtime()[0] if held else
               ToolUseJournalEERuntime(_fake_env(ee), _fake_env))
    cls = Robotiq85Gripper if ee == '2F' else JacoThreeFingerDexterousGripper
    hand = cls(idn='0_right')
    hand.current_action = np.array(action, dtype=float)
    robot = runtime.env.robots[0]
    robot.gripper['right'] = hand
    robot.part_controllers = {'right': SimpleNamespace(name='JOINT_POSITION', input_type='absolute')}
    robot.composite_controller = SimpleNamespace(
        _action_split_indexes={'right': (0, 6), 'right_gripper': (6, 6 + hand.dof)})
    robot.action_dim = 6 + hand.dof
    runtime.command_gripper(engaged=True, suction=False)
    return runtime


def next_gripper_ctrl(runtime):
    """Actual player action + installed robosuite formatter/controller, no physics."""
    player = ToolUseJournalControllerTrajectoryPlayer(runtime)
    action = player._controller_action(SimpleNamespace(joint_names=ARM_JOINTS), [0.] * 6)
    hand = runtime.env.robots[0].gripper['right']
    formatted = hand.format_action(action[6:])
    controller = object.__new__(SimpleGripController)
    controller.update = lambda: None  # The action mapping does not depend on measured joints.
    controller.goal_qvel = None
    controller.control_dim = controller.joint_dim = len(formatted)
    controller.use_action_scaling = True
    controller.input_min = controller.output_min = -np.ones(len(formatted))
    controller.input_max = controller.output_max = np.ones(len(formatted))
    controller.action_scale = controller.action_input_transform = controller.action_output_transform = None
    controller.actuator_min = np.zeros(len(formatted))
    controller.actuator_max = np.full(len(formatted), .8)
    controller.interpolator = None
    controller.set_goal(formatted)
    return controller.run_controller()


@pytest.mark.parametrize('action,command', [([1., 1.], 1.), ([.37, .37], 1.), ([.37, .37], 0.), ([0.], 1.)])
@pytest.mark.parametrize('held', [False, True])
def test_roundtrip_preserves_next_incremental_controller_output(tmp_path, action, command, held):
    source = runtime_with_hand(action, held=held)
    restored = runtime_with_hand([0.])
    try:
        source.command_gripper(engaged=True, suction=False, command=command)
        assert source.captured_gripper_action is None
        physical = source.env.sim.data._data.qpos.copy()
        internal = source.env.robots[0].gripper['right'].current_action.copy()
        path = save_runtime_checkpoint(tmp_path/'hand.json', source)
        np.testing.assert_array_equal(source.env.sim.data._data.qpos, physical)
        np.testing.assert_array_equal(source.env.robots[0].gripper['right'].current_action, internal)
        original_attachment = source.attachment
        restore_runtime_checkpoint(path, restored)
        assert restored.attachment == original_attachment
        assert restored.captured_gripper_action is None
        np.testing.assert_array_equal(restored.env.robots[0].gripper['right'].current_action, internal)
        for _ in range(2):
            np.testing.assert_array_equal(next_gripper_ctrl(restored), next_gripper_ctrl(source))
    finally:
        source.close(); restored.close()


def test_missing_accumulator_reproduces_23_control_drop_without_physics():
    held = runtime_with_hand([1., 1.])
    cold = runtime_with_hand([0.])
    try:
        # The same close command produces different next controls despite any
        # previously restored ctrl: the formatter's private state is an input.
        np.testing.assert_allclose(next_gripper_ctrl(held), [.8, .8])
        np.testing.assert_allclose(next_gripper_ctrl(cold), [.48, .48])
        np.testing.assert_allclose(next_gripper_ctrl(cold), [.56, .56])
    finally:
        held.close(); cold.close()


def test_asymmetric_three_finger_targets_survive_roundtrip(tmp_path):
    source = runtime_with_hand([-.2, .1, .6], ee='3F')
    restored = runtime_with_hand([0., 0., 0.], ee='3F')
    try:
        path = save_runtime_checkpoint(tmp_path/'three.json', source)
        restore_runtime_checkpoint(path, restored)
        np.testing.assert_array_equal(next_gripper_ctrl(source), next_gripper_ctrl(restored))
    finally:
        source.close(); restored.close()


def test_hold_override_and_live_accumulator_remain_distinct(tmp_path):
    source = runtime_with_hand([.7, .7])
    restored = runtime_with_hand([0.])
    try:
        source._captured_gripper_action = np.array([.3, .3])
        path = save_runtime_checkpoint(tmp_path/'override.json', source)
        restore_runtime_checkpoint(path, restored)
        np.testing.assert_array_equal(restored.env.robots[0].gripper['right'].current_action, [.7, .7])
        assert restored.captured_gripper_action == (.3, .3)
        np.testing.assert_array_equal(next_gripper_ctrl(source), next_gripper_ctrl(restored))
    finally:
        source.close(); restored.close()


@pytest.mark.parametrize('corruption', ['missing', 'nan', 'shape', 'width', 'identity', 'legacy_v1'])
def test_invalid_or_legacy_hand_state_rejected_before_mutation(tmp_path, corruption):
    source = runtime_with_hand([.7, .7])
    restored = runtime_with_hand([-.5])
    try:
        path = save_runtime_checkpoint(tmp_path/'bad.json', source)
        raw = json.loads(path.read_text())
        if corruption == 'missing': raw.pop('gripper_control_state')
        elif corruption == 'legacy_v1': raw['version'] = 1; raw.pop('gripper_control_state')
        elif corruption == 'identity': raw['gripper_control_state'][0]['gripper_class'] = 'DifferentHand'
        else:
            raw['gripper_control_state'][0]['current_action'] = {
                'nan': [float('nan')], 'shape': [[.3]], 'width': [.1, .2, .3]}[corruption]
        path.write_text(json.dumps(raw))
        before = capture_runtime_checkpoint(restored)
        with pytest.raises(RuntimeCheckpointError): restore_runtime_checkpoint(path, restored)
        assert capture_runtime_checkpoint(restored) == before
    finally:
        source.close(); restored.close()


def test_hand_accumulator_rolls_back_when_logical_restore_fails(tmp_path, monkeypatch):
    source = runtime_with_hand([.9, .9])
    restored = runtime_with_hand([-.4])
    try:
        path = save_runtime_checkpoint(tmp_path/'rollback.json', source)
        before = capture_runtime_checkpoint(restored)
        def reject(**kwargs):
            np.testing.assert_array_equal(restored.env.robots[0].gripper['right'].current_action, [.9, .9])
            raise RuntimeError('deliberate logical validation failure')
        monkeypatch.setattr(restored, 'restore_logical_state', reject)
        with pytest.raises(RuntimeCheckpointError, match='rolled back'):
            restore_runtime_checkpoint(path, restored)
        assert capture_runtime_checkpoint(restored) == before
    finally:
        source.close(); restored.close()


def test_current_checkpoint_with_no_stateful_hand_still_roundtrips(tmp_path):
    source = ToolUseJournalEERuntime(_fake_env(None), _fake_env)
    restored = ToolUseJournalEERuntime(_fake_env(None), _fake_env)
    try:
        path = save_runtime_checkpoint(tmp_path/'bare.json', source)
        assert json.loads(path.read_text())['gripper_control_state'] == []
        restore_runtime_checkpoint(path, restored)
        assert capture_runtime_checkpoint(restored) == capture_runtime_checkpoint(source)
    finally:
        source.close(); restored.close()
