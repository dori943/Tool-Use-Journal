"""Frozen bbox allocation keeps existing margins and never claims physical fit."""
from copy import deepcopy
from itertools import combinations
import numpy as np
import pytest
from tuj.m1_scene.relations import container_layout


def scene():
    members = [{'id': oid, 'bbox_mm': [w, d, h]} for oid,w,d,h in [
        ('plate',164.6,165.3,9.), ('mug',85.6,125.4,97.4),
        ('fork',100.6,20.1,8.3), ('spoon',69.6,37.8,21.4)]]
    tray = {'id':'tray', 'center_mm':[0.,0.,920.], 'bbox_mm':[204.9,317.1,25.3]}
    return members,tray


def assert_layout_clear(layout, members, opening, gap):
    # Offsets are serialized at0.1mm precision by the unchanged public contract.
    tolerance=.101
    by_id={m['id']:m for m in members}
    assert set(layout['slots'])==set(by_id)
    assert all(slot['frame'] == 'world_xy' for slot in layout['slots'].values())
    for oid,slot in layout['slots'].items():
        center=np.array(slot['offset_mm']);half=np.array(by_id[oid]['bbox_mm'][:2])/2
        assert np.all(np.abs(center)+half<=np.array(opening)/2+tolerance)
    for a,b in combinations(by_id,2):
        ca=np.array(layout['slots'][a]['offset_mm']);cb=np.array(layout['slots'][b]['offset_mm'])
        half=(np.array(by_id[a]['bbox_mm'][:2])+by_id[b]['bbox_mm'][:2])/2
        assert np.max(np.abs(ca-cb)-half)>=gap-tolerance


def test_short_objects_share_column_without_changing_frozen_bbox_or_gap():
    members,tray=scene();before=deepcopy((members,tray))
    layout=container_layout(members,tray)
    assert layout['pass'] and layout['overflow']==[]
    assert layout['layout_method']=='column_shelf_fallback'
    assert layout['rows']==2 and layout['value_mm']==pytest.approx(8.4)
    assert layout['slots']['fork']['row']==layout['slots']['spoon']['row']
    assert_layout_clear(layout,members,[196.9,309.1],10.)
    assert (members,tray)==before
    assert container_layout(members,tray)==layout


@pytest.mark.parametrize('order,mirror', [([0,1],True),([1,0],False),([1,0],True)])
def test_fallback_keeps_row_permutation_and_mirror_geometry(order,mirror):
    members,tray=scene();base=container_layout(members,tray)
    layout=container_layout(members,tray,row_order=order,mirror_x=mirror)
    assert layout['pass']
    assert_layout_clear(layout,members,[196.9,309.1],10.)
    assert layout['slots']['fork']['row']==base['slots']['fork']['row']
    if mirror:
        for oid in base['slots']:
            assert layout['slots'][oid]['offset_mm'][0]==-base['slots'][oid]['offset_mm'][0]


@pytest.mark.parametrize('width,depth,gap', [(196.1,309.1,10.),(196.9,300.6,10.),(196.9,309.1,20.)])
def test_insufficient_width_depth_or_unchanged_gap_stays_overflow(width,depth,gap):
    members,tray=scene();tray['inner_bbox_mm']=[width,depth,20.]
    layout=container_layout(members,tray,gap_mm=gap)
    assert not layout['pass'] and layout['overflow']
    assert layout['layout_method']=='shelf'


def test_fitting_original_shelf_is_not_rearranged():
    members,tray=scene();members=members[:2]
    layout=container_layout(members,tray)
    assert layout['pass'] and layout['layout_method']=='shelf'
    assert layout['slots']['plate']['offset_mm']==[0.,-67.7]
    assert layout['slots']['mug']['offset_mm']==[0.,87.6]


def test_too_wide_item_is_not_rotated_to_fake_a_solution():
    members,tray=scene();members[2]['bbox_mm']=[200.,20.1,8.3]
    layout=container_layout(members,tray)
    assert not layout['pass'] and 'fork' in layout['overflow']
    assert members[2]['bbox_mm']==[200.,20.1,8.3]


def test_m2_ingest_preserves_world_axis_slot_contract():
    from tuj.m2_subgoal.ingest import assign_container_slots
    members, tray = scene()
    m2 = {'m2_subgoals': [
        {'subgoal_id': member['id'], 'kind': 'relocate', 'split_from': 'parent',
         'container_id': 'tray', 'target_ids': [member['id']],
         'details': [{'binding': {'?r': 'tray'}}]}
        for member in members]}
    assign_container_slots(m2, {'nodes': [tray, *members]})
    for subgoal in m2['m2_subgoals']:
        assert subgoal['placement_slot']['frame'] == 'world_xy'
        slot = subgoal['details'][0]['action_parameters']['placement_slot']
        assert slot['source'] == 'm2_container_layout'
        assert slot['frame'] == 'world_xy'
        assert slot['uv'] == subgoal['placement_slot']['uv']
