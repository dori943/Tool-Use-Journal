from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts._vacuum import register_vacuum
from tuj.m5_motion.runtime_checkpoint import save_runtime_checkpoint, restore_runtime_checkpoint
from tuj.m5_motion.tool_use_journal_runtime import (
    ToolUseJournalEERuntime, ToolUseJournalControllerTrajectoryPlayer, ToolUseJournalRuntimeError,
)
from tuj.m5_motion.tests.test_tool_use_journal import _fake_env


@pytest.mark.parametrize("command", [-.9, -.5, 0., .4, 1.])
def test_vacuum_absolute_action_is_continuous_across_control_ticks(command):
    gripper = register_vacuum()()
    robot = NS(
        part_controllers={"right": NS(name="JOINT_POSITION", input_type="absolute")},
        composite_controller=NS(_action_split_indexes={"right": (0, 1), "right_gripper": (1, 2)}),
        robot_model=NS(joints=["arm"]), action_dim=2, gripper={"right": gripper},
    )
    runtime = NS(env=NS(robots=[robot]), gripper_command=command, _captured_gripper_action=None)
    player = ToolUseJournalControllerTrajectoryPlayer(runtime)
    actions = [gripper.format_action(player._controller_action(NS(joint_names=["arm"]), [0.])[1:]) for _ in range(20)]
    np.testing.assert_allclose(actions, np.full((20, 1), command))


def test_partial_suction_survives_checkpoint_and_off_really_disables(tmp_path):
    runtime = ToolUseJournalEERuntime(_fake_env("vac"), _fake_env)
    runtime.command_gripper(engaged=True, suction=True, command=-.9)
    checkpoint = tmp_path / "vacuum.json"
    save_runtime_checkpoint(checkpoint, runtime)
    restored = ToolUseJournalEERuntime(_fake_env("vac"), _fake_env)
    restore_runtime_checkpoint(checkpoint, restored)
    assert restored.gripper_command == -.9
    assert restored.grasp_engaged
    with pytest.raises(ToolUseJournalRuntimeError, match="suction-off"):
        restored.command_gripper(engaged=False, suction=True, command=-.9)
    assert restored.command_gripper(engaged=False, suction=True) == -1.
    assert not restored.grasp_engaged
    runtime.close()
    restored.close()


def test_negative_finger_closure_remains_invalid():
    runtime = ToolUseJournalEERuntime(_fake_env("2F"), _fake_env)
    with pytest.raises(ToolUseJournalRuntimeError, match="non-negative"):
        runtime.command_gripper(engaged=True, suction=False, command=-.9)
    runtime.close()
