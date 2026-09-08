"""Public controller quantization must never choose insufficient clearance."""
from types import SimpleNamespace
import numpy as np
import pytest
from tuj.m5_motion.tool_use_journal_runtime import (
    ToolUseJournalControllerTrajectoryPlayer, ToolUseJournalRuntimeError,
)


class QuantizedPlayer(ToolUseJournalControllerTrajectoryPlayer):
    def __init__(self, recovery=.032):
        self.aperture, self.recovery, self.pulses = .045, recovery, []
        robot = SimpleNamespace(action_dim=2, robot_model=SimpleNamespace(joints=['j']),
            composite_controller=SimpleNamespace(_action_split_indexes={'right':(0,1),'right_gripper':(1,2)}))
        runtime = SimpleNamespace(active_ee='2F', env=SimpleNamespace(robots=[robot]),
            fingerpad_separation_m=lambda:self.aperture, command_gripper=lambda **kw:None)
        super().__init__(runtime)

    def _actual_joint_positions(self, env, names):
        return np.zeros(1)

    def _advance_controller(self, action):
        if action[1]:
            self.pulses.append(action[1])
            self.aperture = .010 if action[1] > 0 else self.recovery
        return 0.


def test_wider_quantized_state_uses_reverse_public_pulse():
    player = QuantizedPlayer()
    aperture = player.preshape_finger_gripper_to_aperture(
        target_aperture_m=.03, tolerance_m=.001, allow_wider_discrete_state=True)
    assert aperture == .032
    assert player.pulses == [1., -1.]


def test_default_strict_mode_still_rejects_unreachable_aperture():
    with pytest.raises(ToolUseJournalRuntimeError, match='between discrete'):
        QuantizedPlayer().preshape_finger_gripper_to_aperture(target_aperture_m=.03)


def test_failed_recovery_cannot_accept_a_narrower_aperture():
    with pytest.raises(ToolUseJournalRuntimeError, match='quantization failed'):
        QuantizedPlayer(recovery=.015).preshape_finger_gripper_to_aperture(
            target_aperture_m=.03, allow_wider_discrete_state=True)
