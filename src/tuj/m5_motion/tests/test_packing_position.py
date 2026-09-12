from copy import copy
from itertools import product
from types import SimpleNamespace as NS
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.m5_motion.scripted_grasps import packing_position as module
from tuj.m5_motion.scripted_grasps.packing_occupancy import bounded_packing_xy


def scene(monkeypatch, yaw=0.):
    region = np.eye(4)
    region[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    region[:3, 3] = [.2, -.3, .8]
    position = region[:3, 3] + region[:3, :3] @ [.04, 0., .03]
    quat = Rotation.from_euler('z', yaw).as_quat()[[3, 0, 1, 2]]
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="target" pos="1 1 1"><freejoint/><geom type="box" size=".06 .01 .01"/></body>
      <body name="occupant" pos="{' '.join(map(str, position))}" quat="{' '.join(map(str, quat))}">
        <geom type="box" size=".02 .008 .03"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    context = NS(model=model, data=data, body_id=1, object_geoms={0},
                 env=NS(obj_body_id={'occupied': 2}), descendant=lambda child, root: child == root)
    points = np.array(list(product([-.06, .06], [-.01, .01], [-.01, .01])))
    g = NS(object_id='target', retention=NS(context=context),
           task=NS(action_type='TRANSPORT', goal=NS(target_region_id='box'), metadata={}),
           record={'packing_metadata': {'kind': 'CONTAINER', 'position_policy': module.POSITION_POLICY,
                                        'interior_center_m': [0, 0, .03],
                                        'interior_dimensions_m': [.2, .08, .06]}},
           request=NS(constraints=NS(collision_margin_m=.005, position_tolerance_m=.005)),
           T_WR=region, destination_rotation=region[:3, :3].copy(),
           center_in_body=np.zeros(3), packing_bounds=(points.min(0), points.max(0)),
           preferred_xy=np.zeros(2))
    monkeypatch.setattr(module, 'packing_occupants', lambda g: ['occupied'])
    monkeypatch.setattr('tuj.m5_motion.scripted_grasps.transport.transport_destination_center',
                        lambda g: g.T_WR[:3, 3] + g.T_WR[:3, :3] @ np.r_[
                            getattr(g, 'packing_body_xy', g.preferred_xy), .2])
    monkeypatch.setattr('tuj.m5_motion.scripted_grasps.container_orientation.packing_transport_has_ik',
                        lambda g: True)
    return g, points


@pytest.mark.parametrize('yaw', [0., .7, -1.1])
def test_real_geometry_selects_nearest_clear_position_without_advancing_state(monkeypatch, yaw):
    g, points = scene(monkeypatch, yaw)
    before = {k: getattr(g.retention.context.data, k).copy() for k in ('qpos', 'qvel', 'ctrl', 'act', 'xpos')}
    module.select_clear_packing_position(g)
    # Symmetric free sites are equally close; either side is physically valid.
    np.testing.assert_allclose(np.abs(g.packing_body_xy), [0., .02], atol=1e-12)
    e = g.task.metadata['packing_position_selection']
    assert e['initial_score_m'] > 0 and e['selected_score_m'] == 0.
    np.testing.assert_allclose(g.destination_rotation, g.T_WR[:3, :3])
    for k, expected in before.items():
        np.testing.assert_array_equal(getattr(g.retention.context.data, k), expected)
    assert g.retention.context.data.time == 0.


def test_clear_preferred_position_and_non_opt_in_keep_existing_behavior(monkeypatch):
    g, _ = scene(monkeypatch)
    g.preferred_xy[:] = [0., .025]
    module.select_clear_packing_position(g)
    assert not hasattr(g, 'packing_body_xy')
    assert g.task.metadata['packing_position_selection']['status'] == 'KEEP_CLEAR_PREFERRED_POSITION'
    g.record['packing_metadata'].pop('position_policy')
    monkeypatch.setattr(module, 'packing_overlap_preference', lambda *a, **k: pytest.fail('must not sample'))
    module.select_clear_packing_position(g)


def test_unreachable_nearest_zero_overlap_position_is_not_selected(monkeypatch):
    g, _ = scene(monkeypatch)
    checked = []
    def reachable(p):
        checked.append(p.packing_body_xy.copy())
        return p.packing_body_xy[1] < 0.
    monkeypatch.setattr('tuj.m5_motion.scripted_grasps.container_orientation.packing_transport_has_ik',
                        reachable)
    module.select_clear_packing_position(g)
    np.testing.assert_allclose(g.packing_body_xy, [0., -.02], atol=1e-12)
    assert any(xy[1] > 0. for xy in checked)


@pytest.mark.parametrize('zero_but_unreachable', [False, True])
def test_no_clear_reachable_alternative_keeps_original_fallback(monkeypatch, zero_but_unreachable):
    g, _ = scene(monkeypatch)
    if zero_but_unreachable:
        monkeypatch.setattr('tuj.m5_motion.scripted_grasps.container_orientation.packing_transport_has_ik', lambda g: False)
    else:
        monkeypatch.setattr(module, 'packing_overlap_preference', lambda *a, **k: .01)
    module.select_clear_packing_position(g)
    assert not hasattr(g, 'packing_body_xy') and g.retention.packing_position is None
    assert g.task.metadata['packing_position_selection']['status'] == 'NO_CLEAR_REACHABLE_POSITION_KEEP_EXISTING'


def test_retained_orientation_and_xy_follow_region_frame_and_recheck_bounds(monkeypatch):
    g, points = scene(monkeypatch, .7)
    module.select_clear_packing_position(g)
    place = copy(g)
    place.task = NS(action_type='PLACE', goal=NS(target_region_id='box'), metadata={})
    place.T_WR = g.T_WR.copy()
    place.T_WR[:3, :3] = Rotation.from_euler('z', -.8).as_matrix()
    assert module.restore_packing_position(place, points)
    np.testing.assert_allclose(place.destination_rotation, place.T_WR[:3, :3], atol=1e-12)
    np.testing.assert_allclose(place.packing_body_xy, g.packing_body_xy, atol=1e-12)
    place.task.goal.target_region_id = 'another_region'
    assert not module.restore_packing_position(place, points)
    place.task.goal.target_region_id = 'box'
    place.retention.packing_position['body_xy_in_region_m'] = [10., 10.]
    with pytest.raises(ValueError, match='OUTSIDE_BOUNDS'):
        module.restore_packing_position(place, points)


def test_roundoff_snapping_does_not_admit_physical_margin_violation(monkeypatch):
    g, _ = scene(monkeypatch)
    low, _ = module.packing_translation_bounds(g)
    np.testing.assert_array_equal(bounded_packing_xy(g, low[:2] - 1e-16), low[:2])
    with pytest.raises(ValueError, match='OUTSIDE_BOUNDS'):
        bounded_packing_xy(g, low[:2] - 1e-6)


def test_grid_budget_is_checked_before_allocating_xy_grid(monkeypatch):
    g, _ = scene(monkeypatch)
    monkeypatch.setattr(module, 'packing_overlap_preference', lambda *a, **k: .01)
    g.request.constraints.position_tolerance_m = 1e-12
    monkeypatch.setattr(module.np, 'linspace', lambda *a, **k: pytest.fail('grid must not allocate'))
    module.select_clear_packing_position(g)
    assert g.task.metadata['packing_position_selection']['status'] == 'GRID_BUDGET_EXCEEDED_KEEP_EXISTING'
    assert not hasattr(g, 'packing_body_xy') and g.retention.packing_position is None


def test_new_transport_invalidates_old_selected_position(monkeypatch):
    g, points = scene(monkeypatch)
    g.packing_body_xy = np.array([0., -.02])
    g.retention.packing_position = {'object_id': 'target', 'region_id': 'box',
        'body_xy_in_region_m': [0., -.02], 'orientation_in_region_xyzw': [0., 0., 0., 1.]}
    g.preferred_xy[:] = [0., .025]
    module.select_clear_packing_position(g)
    assert not hasattr(g, 'packing_body_xy') and g.retention.packing_position is None
    g.task.action_type = 'PLACE'
    assert not module.restore_packing_position(g, points)


def test_non_opt_in_destination_ignores_helper_position_attribute():
    from tuj.m5_motion.tests.test_container_orientation import fixture
    from tuj.m5_motion.scripted_grasps.container_orientation import configure_packing_orientation, packing_destination_center
    g, _ = fixture(.7)
    configure_packing_orientation(g)
    desired = np.array([4., -4., 1.5])
    expected = packing_destination_center(g, desired, place=True)
    g.packing_body_xy = np.array([99., 99.])
    np.testing.assert_array_equal(packing_destination_center(g, desired, place=True), expected)
