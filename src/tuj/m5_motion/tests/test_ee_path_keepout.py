"""Placements keep out from under the stored EE-rack trajectories (c3_2 mug_a)."""
import math

import numpy as np

from tuj.common.ee_keepout import (
    SweptArmHeightField, load_height_field, placement_margin_m, save_height_field,
)
from tuj.m1_scene.relations import container_layout
from tuj.m2_subgoal.ground import container_layout_clear_of_sweep
from tuj.m5_motion.ee_path_keepout import (
    EE_PATH_KEEPOUT_METADATA_KEY, dense_states, register_keepout,
)
from tuj.m5_motion.tests.test_scripted_grasps import ENTRIES
from tuj.m5_motion.tests.test_scripted_transport import (
    BODY, CENTER_IN_BODY, REGION, REGION_POSITION, _generic_held, _tray_request,
)


def _field(lo_xy, hi_xy, floor_z, cell=.005):
    """Arm reaches down to ``floor_z`` over [lo, hi]; never elsewhere nearby."""
    lo, hi = np.asarray(lo_xy, float), np.asarray(hi_xy, float)
    origin = lo - .2
    shape = np.ceil((hi - lo + .4) / cell).astype(int)
    z = np.full(tuple(shape), math.inf)
    a = np.floor((lo - origin) / cell).astype(int)
    b = np.floor((hi - origin) / cell).astype(int)
    z[a[0]:b[0] + 1, a[1]:b[1] + 1] = floor_z
    return SweptArmHeightField(origin_xy=origin, cell_m=cell, floor_z=z)


def test_height_field_clearance_counts_margin_on_every_side():
    f = _field((0., 0.), (.1, .1), 1.0)
    assert f.lowest_z_under((.05, .05), (.01, .01)) == 1.0
    assert f.lowest_z_under((.3, .3), (.01, .01)) == math.inf
    # Beside the arm cells: blocked while the margin still reaches them.
    assert f.clearance((.12, .05), (.01, .01), .99, .02) < 0.
    assert f.clearance((.16, .05), (.01, .01), .99, .02) == math.inf
    # Under the arm: a low object passes, a tall one does not.
    assert f.clearance((.05, .05), (.01, .01), .9, .015) > 0.
    assert f.clearance((.05, .05), (.01, .01), .99, .015) < 0.


def test_height_field_round_trips_and_shifts(tmp_path):
    f = _field((0., 0.), (.1, .1), 1.0).cropped()
    save_height_field(tmp_path / 'k.npz', f)
    g = load_height_field(tmp_path / 'k.npz').shifted((1., 2.))
    assert g.lowest_z_under((.05 - 1., .05 - 2.), (.001, .001)) == 1.0
    assert load_height_field(tmp_path / 'missing.npz') is None


def test_return_path_states_are_not_evaluated_twice():
    attach = [[0., 0.], [.1, 0.], [.1, .1]]
    states = dense_states([attach, attach[::-1]], .02)
    assert len(states) == len(dense_states([attach], .02))


def _members():
    return [
        {'id': 'plate', 'bbox_mm': [160., 160., 10.]},
        {'id': 'mug', 'bbox_mm': [80., 100., 100.]},
    ]


def _tray():
    return {'id': 'tray', 'center_mm': [0., 0., 920.], 'bbox_mm': [220., 360., 20.]}


def test_layout_row_order_and_mirror_move_rows_not_their_contents():
    base = container_layout(_members(), _tray())
    flipped = container_layout(_members(), _tray(), row_order=[1, 0], mirror_x=True)
    assert base['slots']['plate']['offset_mm'][1] < base['slots']['mug']['offset_mm'][1]
    assert flipped['slots']['plate']['offset_mm'][1] > flipped['slots']['mug']['offset_mm'][1]
    assert flipped['slots']['mug']['row'] == base['slots']['mug']['row']


def test_layout_moves_the_tall_row_out_from_under_the_sweep():
    """The default puts the mug row on +y, where the arm passes 60 mm up."""
    base = container_layout(_members(), _tray())
    mug_y = base['slots']['mug']['offset_mm'][1] / 1000.
    sweep = _field((-.2, mug_y - .06), (.2, .2), .99)
    layout = container_layout_clear_of_sweep(_members(), _tray(), sweep)
    assert layout['ee_sweep']['blocked'] == []
    assert layout['ee_sweep']['rearranged'] is True
    assert layout['slots']['mug']['offset_mm'][1] < 0.
    # The flat plate may stay under the arm.
    assert layout['ee_sweep']['clearance_mm']['plate'] > 0.


def test_layout_is_unchanged_without_a_sweep_or_when_already_clear():
    base = container_layout(_members(), _tray())
    assert container_layout_clear_of_sweep(_members(), _tray(), None) == base
    far = _field((5., 5.), (5.1, 5.1), .95)
    clear = container_layout_clear_of_sweep(_members(), _tray(), far)
    assert clear['slots'] == base['slots'] and not clear['ee_sweep']['rearranged']


def _slot_request(uv):
    request = _tray_request(.005)
    request.task.action_type = 'place'
    request.world.objects['tray']['dimensions_m'] = [.6, .4, .1]
    _generic_held(request, ENTRIES[4].object_id)
    request.task.metadata['action_parameters'] = {'placement_slot': {
        'region': 'obj_tray_tray', 'uv': list(uv), 'source': 'm2_container_layout'}}
    region_center = (REGION @ np.array([.02, 0, 0]) + REGION_POSITION)[:2]
    inner = np.array([.6, .4]) * .5 - .02
    slot = region_center + (REGION @ np.r_[np.asarray(uv) * inner, 0.])[:2]
    return request, slot


def _seated_center(request):
    hint = request.task.metadata['held_place_goal']
    rec = request.world.objects['tray']
    destination = REGION @ rec['anchors'][hint['anchor']] + REGION_POSITION
    return destination + BODY[:3, :3] @ CENTER_IN_BODY


def _bind(request, field, key):
    register_keepout(key, field)
    request.task.metadata[EE_PATH_KEEPOUT_METADATA_KEY] = key


def test_place_leaves_a_slot_that_lies_under_the_sweep():
    from tuj.m5_motion.scripted_grasps.transport import ground_held_place
    request, slot = _slot_request((.5, -.5))
    # Object top is .745 (floor .655 + .09); the arm passes 5 mm above it.
    _bind(request, _field(slot - .03, slot + .03, .75), 'test:slot-under-sweep')
    ground_held_place(request)
    center = _seated_center(request)
    note = request.task.metadata['ee_path_keepout_seat']
    assert note['clearance_m'] >= 0. and 'unavoidable' not in note
    assert 'slot_xy_in_sweep' in note
    assert np.linalg.norm(center[:2] - slot) > .03


def test_place_keeps_the_slot_when_the_sweep_is_elsewhere():
    from tuj.m5_motion.scripted_grasps.transport import ground_held_place
    request, slot = _slot_request((.5, -.5))
    _bind(request, _field(slot + .5, slot + .6, .75), 'test:sweep-elsewhere')
    ground_held_place(request)
    np.testing.assert_allclose(_seated_center(request)[:2], slot, atol=1e-9)
    assert 'slot_xy_in_sweep' not in request.task.metadata['ee_path_keepout_seat']


def test_place_reports_a_region_entirely_under_the_sweep():
    from tuj.m5_motion.scripted_grasps.transport import ground_held_place
    request, slot = _slot_request((.5, -.5))
    _bind(request, _field(slot - 1., slot + 1., .75), 'test:region-under-sweep')
    ground_held_place(request)
    # Unconstrained choice is kept; the conflict is recorded for the run log.
    np.testing.assert_allclose(_seated_center(request)[:2], slot, atol=1e-9)
    note = request.task.metadata['ee_path_keepout_seat']
    assert note['unavoidable'] is True and note['clearance_m'] < 0.


def test_placement_margin_includes_the_planner_collision_margin():
    assert placement_margin_m(.005) > .005


def test_column_fallback_layout_still_checks_every_slot_against_ee_sweep():
    from tuj.m1_scene.tests.test_container_layout import scene, assert_layout_clear
    members, tray = scene()
    sweep = _field((-.2, 0.), (.2, .25), .99)
    layout = container_layout_clear_of_sweep(members, tray, sweep)
    assert layout['layout_method'] == 'column_shelf_fallback'
    assert layout['pass']
    assert layout['ee_sweep']['blocked'] == []
    assert layout['ee_sweep']['rearranged']
    assert_layout_clear(layout, members, [196.9, 309.1], 10.)
