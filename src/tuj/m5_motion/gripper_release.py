"""Bounded release synchronization for incremental finger controllers."""
from copy import deepcopy
from dataclasses import dataclass, field
import math

import numpy as np


def open_action_endpoint(gripper, command):
    """Find the open target through a copy of the hand's public action API."""
    speed = float(gripper.speed)
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError('GRIPPER_RELEASE_INVALID_SPEED')
    probe = deepcopy(gripper)
    previous = np.asarray(probe.current_action, dtype=float).copy()
    action = np.full(probe.dof, float(command))
    # Robosuite normalized integrated targets lie in [-1, 1]. Include the
    # terminal repeated sample so saturation is observed, not presumed.
    for ticks in range(1, min(4096, math.ceil(2. / speed) + 3)):
        probe.format_action(action)
        current = np.asarray(probe.current_action, dtype=float).copy()
        if not np.isfinite(current).all():
            raise ValueError('GRIPPER_RELEASE_NONFINITE_TARGET')
        if np.allclose(current, previous, atol=1e-12, rtol=0.):
            return current, ticks
        previous = current
    raise ValueError('GRIPPER_RELEASE_ENDPOINT_UNAVAILABLE')


@dataclass
class GripperReleaseWait:
    target: np.ndarray
    started_at_s: float
    max_wait_s: float
    required_ticks: int
    object_id: str
    absolute_target: bool = False
    consecutive_ticks: int = 0
    observation: dict = field(default_factory=dict)

    def update(self, time_s, current_action, contact_count):
        target_reached = bool(np.allclose(
            current_action, self.target, atol=1e-9, rtol=0.))
        self.consecutive_ticks = (
            self.consecutive_ticks + 1 if target_reached and contact_count == 0 else 0)
        self.observation = {
            'object_id': self.object_id,
            'wait_duration_s': float(time_s - self.started_at_s),
            'open_target': self.target.tolist(),
            'current_target': np.asarray(current_action).tolist(),
            'target_reached': target_reached,
            'ee_object_contact_count': int(contact_count),
            'consecutive_ticks': self.consecutive_ticks,
            'max_wait_s': self.max_wait_s,
            'command_mode': 'SCRIPTED_POSITION_TARGET' if self.absolute_target else 'INCREMENTAL',
        }
        if self.consecutive_ticks >= self.required_ticks:
            return True
        if time_s - self.started_at_s >= self.max_wait_s:
            raise TimeoutError('GRIPPER_RELEASE_NOT_SETTLED')
        return False
