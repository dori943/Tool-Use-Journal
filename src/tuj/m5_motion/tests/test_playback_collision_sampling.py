import numpy as np
import pytest

from tuj.m5_motion.schema import TrajectoryWaypoint
from tuj.m5_motion.trajectory_processing import (
    interpolate_trajectory_joint_position, playback_collision_waypoints,
)
from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalControllerTrajectoryPlayer


def point(t, q, v=None):
    return TrajectoryWaypoint(time_from_start_s=t, joint_positions_rad=[q],
                              joint_velocities_rad_s=None if v is None else [v])


@pytest.mark.parametrize('velocities,expected', [((None, None), 0.), ((2., -2.), .5)])
def test_validation_and_controller_preserve_the_existing_interpolation(velocities, expected):
    points = (point(0., 0., velocities[0]), point(1., 0., velocities[1]))
    np.testing.assert_array_equal(interpolate_trajectory_joint_position(points, .5), [expected])
    for time in [-1., 0., .25, .5, .75, 1., 2.]:
        np.testing.assert_array_equal(
            ToolUseJournalControllerTrajectoryPlayer._desired_joint_position(points, time),
            interpolate_trajectory_joint_position(points, time))


def test_validation_keeps_knots_and_adds_absolute_grid_and_off_grid_midpoints():
    points = (point(.007, 0.), point(.027, .9))
    original = [p.model_dump() for p in points]
    samples = playback_collision_waypoints(points)
    times = [p.time_from_start_s for p in samples]
    assert times == [.007, .017, .02, .027]
    assert samples[0] is points[0] and samples[-1] is points[-1]
    assert samples[1].joint_positions_rad == pytest.approx([.45])
    assert [p.model_dump() for p in points] == original


def test_validation_deduplicates_and_retains_hold_and_single_state():
    points = (point(0., .1, 0.), point(.04, .1, 0.))
    samples = playback_collision_waypoints(points)
    assert [p.time_from_start_s for p in samples] == [0., .02, .04]
    assert all(p.joint_positions_rad == [.1] for p in samples)
    assert playback_collision_waypoints(points[:1]) == points[:1]
    assert playback_collision_waypoints(()) == ()
