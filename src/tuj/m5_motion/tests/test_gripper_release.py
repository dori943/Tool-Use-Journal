import numpy as np
import pytest

from tuj.m5_motion.gripper_release import GripperReleaseWait, open_action_endpoint


class IncrementalHand:
    speed = .01
    dof = 3

    def __init__(self, direction):
        self.direction = direction
        self.current_action = np.full(3, -direction, dtype=float)

    def format_action(self, action):
        self.current_action = np.clip(
            self.current_action - self.direction * self.speed * np.sign(action), -1., 1.)
        return self.current_action


@pytest.mark.parametrize('direction', [-1, 1])
def test_open_endpoint_uses_hand_sign_without_mutating_live_hand(direction):
    hand = IncrementalHand(direction)
    initial = hand.current_action.copy()
    target, ticks = open_action_endpoint(hand, -1)
    np.testing.assert_allclose(target, direction)
    np.testing.assert_array_equal(hand.current_action, initial)
    assert ticks == 201


def test_release_wait_requires_endpoint_and_fresh_contact_free_window():
    gate = GripperReleaseWait(np.ones(3), 0., 5., 3, 'tool')
    assert not gate.update(1., np.zeros(3), 0)
    assert not gate.update(2., np.ones(3), 1)
    assert not gate.update(2.02, np.ones(3), 0)
    assert not gate.update(2.04, np.ones(3), 0)
    assert not gate.update(2.06, np.ones(3), 1)
    assert not gate.update(2.08, np.ones(3), 0)
    assert not gate.update(2.10, np.ones(3), 0)
    assert gate.update(2.12, np.ones(3), 0)


@pytest.mark.parametrize('target,contacts', [(np.zeros(3),0),(np.ones(3),1)])
def test_stuck_release_is_bounded_in_execution_time(target, contacts):
    gate = GripperReleaseWait(np.ones(3), 10., 5., 3, 'tool')
    assert not gate.update(14.9, target, contacts)
    with pytest.raises(TimeoutError, match='GRIPPER_RELEASE_NOT_SETTLED'):
        gate.update(15., target, contacts)


def test_invalid_increment_speed_fails_closed():
    hand = IncrementalHand(1)
    hand.speed = 0
    with pytest.raises(ValueError, match='INVALID_SPEED'):
        open_action_endpoint(hand, -1)
