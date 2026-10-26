from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.m5_motion.scripted_grasps.transport import _Grounding


def grounding(operation='PLACE_ON', height=.23, yaw=.7, kind='CONTAINER'):
    g = object.__new__(_Grounding)
    g.task = SimpleNamespace(action_type=operation, metadata={})
    g.record = {'packing_metadata': {'kind': kind, 'opening_top_z_m': .01 + height}}
    g.T_WR = np.eye(4)
    g.T_WR[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    g.T_WR[2, 3] = .92
    g.region_world = np.array([0., 0., .93 + height / 2.])
    g.region_half = np.array([.1, .13, height / 2.])
    g.half = np.array([.1, .13, .006])
    g.floor_top_world_z = lambda: .93
    g._occupants = lambda: [(np.array([0., 0.]), np.array([.03, .04]), 1.08, False)]
    return g


@pytest.mark.parametrize('height', [.15, .23, .31])
@pytest.mark.parametrize('yaw', [0., .7, -1.2])
def test_place_on_container_uses_actual_rim_not_contents(height, yaw):
    g = grounding(height=height, yaw=yaw)
    top, stacked = g.support_top_world_z([0., 0.])
    assert top == pytest.approx(.93 + height)
    assert not stacked


@pytest.mark.parametrize('operation,kind', [('PLACE', 'CONTAINER'), ('PLACE_ON', 'TRAY')])
def test_interior_packing_and_other_surfaces_keep_region_floor_support(operation, kind):
    g = grounding(operation=operation, kind=kind)
    assert g.support_top_world_z([0., 0.]) == (.93, False)


def test_higher_occupant_raises_travel_clearance_but_is_not_a_stacking_support():
    g = grounding()
    g._occupants = lambda: [(np.array([0., 0.]), np.array([.03, .04]), 1.3, False)]
    top, stacked = g.support_top_world_z([0., 0.])
    assert top == pytest.approx(1.16) and not stacked
    assert g.interior_top_world_z() == 1.3


@pytest.mark.parametrize('invalid', ['tilt', 'nan'])
def test_unknown_support_plane_fails_closed(invalid):
    g = grounding()
    if invalid == 'tilt':
        g.T_WR[:3, :3] = Rotation.from_euler('x', .2).as_matrix()
    else:
        g.record['packing_metadata']['opening_top_z_m'] = float('nan')
    with pytest.raises(ValueError, match='UPRIGHT_CONTAINER_RIM_REQUIRED'):
        g.support_top_world_z([0., 0.])
