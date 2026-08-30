"""Full-pose IK must retain deterministic UR branch alternatives."""

from __future__ import annotations

import pytest

from tuj.m4_motion import UR5eKinematics


@pytest.fixture(scope="module")
def kinematics() -> UR5eKinematics:
    return UR5eKinematics()


def test_full_pose_ik_returns_a_distinct_solution_set(kinematics) -> None:
    seed_q = (0.3, -1.2, 1.4, -1.7, -1.2, 0.4)
    position, orientation = kinematics.forward_pose_world(seed_q)

    result = kinematics.solve_all_ik(position, orientation)

    assert result.solved
    assert result.enumeration_complete is False
    assert len(result.solutions) >= 2
    assert len({solution.branch_id for solution in result.solutions}) >= 2
    assert all(solution.position_error_m <= 5e-3 for solution in result.solutions)
    assert all(solution.orientation_error_rad <= 5e-2 for solution in result.solutions)


def test_full_pose_ik_is_deterministic(kinematics) -> None:
    seed_q = (0.3, -1.2, 1.4, -1.7, -1.2, 0.4)
    position, orientation = kinematics.forward_pose_world(seed_q)

    first = kinematics.solve_all_ik(position, orientation)
    second = kinematics.solve_all_ik(position, orientation)

    assert first == second


def test_full_pose_ik_prefers_a_supplied_local_seed(kinematics) -> None:
    seed_q = (0.3, -1.2, 1.4, -1.7, -1.2, 0.4)
    position, orientation = kinematics.forward_pose_world(seed_q)

    result = kinematics.solve_all_ik(
        position,
        orientation,
        seed_qpos=seed_q,
    )

    assert result.solved
    assert max(
        abs(actual - expected)
        for actual, expected in zip(result.solutions[0].qpos, seed_q)
    ) <= 1e-9
