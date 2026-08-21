"""Path processing must populate a constrained time-axis trajectory."""

from __future__ import annotations

import pytest

from motion_planner.schema import JointDynamicLimit
from motion_planner.trajectory_processing import (
    QuinticTimeParameterizer,
    deviation_bounded_shortcut,
    deterministic_shortcut,
    unwrap_joint_path,
)


def test_deterministic_shortcut_uses_the_farthest_valid_connection() -> None:
    path = [(0.0,), (1.0,), (2.0,), (3.0,)]

    def valid(source, target):
        return not (source == (0.0,) and target == (3.0,))

    assert deterministic_shortcut(path, valid) == ((0.0,), (2.0,), (3.0,))


def test_deviation_bounded_shortcut_preserves_a_joint_space_corner() -> None:
    path = [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (1.0, 0.5),
        (1.0, 1.0),
    ]

    assert deviation_bounded_shortcut(
        path, max_deviation_rad=0.01
    ) == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))


def test_unwrap_joint_path_keeps_pi_boundary_continuous() -> None:
    path = [(3.10,), (-3.12,), (-3.00,)]

    unwrapped = unwrap_joint_path(path, start_reference=(3.05,))

    assert unwrapped[0] == pytest.approx((3.10,))
    assert unwrapped[1][0] > 3.10
    assert unwrapped[2][0] > unwrapped[1][0]
    assert max(
        abs(right[0] - left[0])
        for left, right in zip(unwrapped, unwrapped[1:])
    ) < 0.2


def test_quintic_parameterization_respects_velocity_and_acceleration_limits() -> None:
    limits = {
        "j1": JointDynamicLimit(
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=2.0,
            max_jerk_rad_s3=10.0,
        ),
        "j2": JointDynamicLimit(
            max_velocity_rad_s=0.8,
            max_acceleration_rad_s2=1.5,
            max_jerk_rad_s3=8.0,
        ),
    }
    result = QuinticTimeParameterizer(sample_dt_s=0.01).parameterize(
        ["j1", "j2"],
        [(0.0, 0.0), (1.0, -0.5)],
        limits,
    )

    times = [waypoint.time_from_start_s for waypoint in result.waypoints]
    assert all(right > left for left, right in zip(times, times[1:]))
    assert result.waypoints[0].joint_positions_rad == pytest.approx([0.0, 0.0])
    assert result.waypoints[-1].joint_positions_rad == pytest.approx([1.0, -0.5])
    assert max(abs(w.joint_velocities_rad_s[0]) for w in result.waypoints) <= 1.0 + 1e-9
    assert max(abs(w.joint_velocities_rad_s[1]) for w in result.waypoints) <= 0.8 + 1e-9
    assert max(abs(w.joint_accelerations_rad_s2[0]) for w in result.waypoints) <= 2.0 + 1e-9
    assert max(abs(w.joint_accelerations_rad_s2[1]) for w in result.waypoints) <= 1.5 + 1e-9
