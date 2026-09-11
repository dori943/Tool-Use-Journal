import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.m5_motion.scripted_grasps.container_release import rim_clearance_lift


@pytest.mark.parametrize('yaw', [0., .7, -1.4])
def test_open_hand_crossing_wall_is_lifted_above_rim(yaw):
    region = np.eye(4)
    region[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    region[:3, 3] = [2., -3., .9]
    destination = region.copy()
    hand = np.array([[-.11, -.02, .095], [.01, .14, .29]])
    lift = rim_clearance_lift(hand, destination, region, [0., 0., .11],
                              [.18, .24, .2], .21, .005)
    assert lift == pytest.approx(.12)
    assert hand[:, 2].min() + lift >= .21 + .005 - 1e-12


def test_hand_that_fits_keeps_floor_release():
    hand = np.array([[-.04, -.05, .095], [.04, .05, .2]])
    assert rim_clearance_lift(hand, np.eye(4), np.eye(4), [0., 0., .11],
                              [.18, .24, .2], .21, .005) == 0.
