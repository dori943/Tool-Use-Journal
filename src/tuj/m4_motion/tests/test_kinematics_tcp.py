"""Active-EE TCP transforms stay consistent with the UR5e hand model."""

from __future__ import annotations

import numpy as np

from tuj.m4_motion.kinematics import (
    UR5eKinematics,
    _quaternion_xyzw_to_matrix,
)


def test_forward_pose_applies_fixed_tcp_transform() -> None:
    qpos = (-0.47, -1.735, 2.48, -2.275, -1.59, -1.991)
    hand = UR5eKinematics(base_pos=(0.0, 0.0, 0.0))
    tcp = UR5eKinematics(
        base_pos=(0.0, 0.0, 0.0),
        target_position_in_eef_m=(0.0, 0.0, 0.145),
        target_orientation_in_eef_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    hand_position, hand_orientation = hand.forward_pose_world(qpos)
    tcp_position, tcp_orientation = tcp.forward_pose_world(qpos)
    expected = np.asarray(hand_position) + _quaternion_xyzw_to_matrix(
        hand_orientation
    ) @ np.asarray((0.0, 0.0, 0.145))

    assert np.allclose(tcp_position, expected, atol=1e-12, rtol=0.0)
    assert np.allclose(tcp_orientation, hand_orientation, atol=1e-12, rtol=0.0)
