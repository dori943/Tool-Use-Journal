"""Regression coverage for source-plate loops and disturbed completed placements."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest

from tuj.m2_subgoal.assembly import normalize_assembly_plan
from tuj.m2_subgoal.pipeline import run_m2
from tuj.m5_motion.scripted_grasps.preservation import (
    PlacementPreservation, PlacementPreservationError,
)

TASK = '재료를 순서대로 쌓아 샌드위치를 만들어라'


def assembly_input():
    m1 = {'nodes': [{'id': 'source', 'class': 'bread_plate'},
                    {'id': 'serving', 'class': 'serving_plate'},
                    {'id': 'bread', 'class': 'bread', 'mass_kg': 2.,
                     'ee': {'vac': {'feasible': False}}}],
          'edges': [{'from': 'bread', 'to': 'source', 'type': 'on'}]}
    subs = [dict(subgoal_id='stage', kind='relocate', target_ids=['bread'], container_id='source'),
            dict(subgoal_id='assemble', kind='stack', target_ids=['bread'], container_id='source')]
    return m1, subs


def test_pipeline_removes_source_loop_and_keeps_estimates_unchanged():
    m1, subs = assembly_input()
    original = deepcopy((m1, subs))
    out = run_m2(TASK, m1, rough=SimpleNamespace(generate=lambda *_: subs))
    assert len(out['m2_subgoals']) == 1
    stack = out['m2_subgoals'][0]
    assert stack['container_id'] == 'serving'
    assert stack['details'][-1]['binding']['?base'] == 'serving'
    assert len(out['m2_semantic_corrections']) == 2
    assert (m1, subs) == original  # no physical priors or original plan mutated


def test_explicit_move_and_later_return_are_not_noop_staging():
    m1, subs = assembly_input()
    other = dict(subgoal_id='elsewhere', kind='relocate', target_ids=['bread'], container_id='elsewhere')
    out, _ = normalize_assembly_plan(TASK, [other, *subs], m1)
    assert [s['subgoal_id'] for s in out] == ['elsewhere', 'stage', 'assemble']
    out, _ = normalize_assembly_plan(TASK, [subs[1], subs[0]], m1)
    assert len(out) == 2


def test_unknown_source_relation_and_unrelated_tasks_are_not_repaired():
    m1, subs = assembly_input()
    assert normalize_assembly_plan('접시로 되돌려라', subs, m1) == (subs, [])
    assert normalize_assembly_plan('빵 접시에 샌드위치를 만들어라', subs, m1) == (subs, [])
    m1['edges'] = []
    assert normalize_assembly_plan(TASK, subs, m1) == (subs, [])


@pytest.mark.parametrize('serving_count', [0, 2])
def test_ambiguous_assembly_destination_is_rejected(serving_count):
    m1, subs = assembly_input()
    m1['nodes'] = [n for n in m1['nodes'] if n['class'] != 'serving_plate']
    m1['nodes'] += [dict(id=f'plate_{i}', **{'class': 'serving_plate'}) for i in range(serving_count)]
    with pytest.raises(ValueError, match='ASSEMBLY_DESTINATION_AMBIGUOUS'):
        normalize_assembly_plan(TASK, subs, m1)


def box(x, z, size):
    return {'pose': {'frame_id': 'world', 'position_m': [x, 0., z],
                     'orientation_xyzw': [0., 0., 0., 1.]}, 'dimensions_m': size}


def measured_world():
    return {'objects': {'plate': box(0., .5, [.3, .3, .02]),
                        'fruit': box(0., .53, [.04, .04, .04])},
            'robot_state': {'attached_object_id': None, 'held_tool_id': None}, 'metadata': {}}


def task(action='place', target='fruit', region='plate'):
    return SimpleNamespace(action_type=action, target_ids=[target], metadata={},
                           goal=SimpleNamespace(target_region_id=region))


def test_completed_placement_is_rechecked_after_unrelated_motion():
    monitor = PlacementPreservation()
    world = measured_world()
    assert monitor.observe(task(), world)[0]['passed'] is True
    assert monitor.observe(task('EE_EXCHANGE', 'other', None), world)[0]['passed'] is True
    world['objects']['fruit']['pose']['position_m'][0] = .2
    with pytest.raises(PlacementPreservationError, match='PLACEMENT_PRESERVATION_FAILED') as error:
        monitor.observe(task('place', 'mug', 'tray'), world)
    evidence = error.value.checks[0]['checks'][0]['evidence']
    assert evidence['maximum_overhang_m'] > .06
    assert evidence['tolerance_m'] == .0015


def test_explicit_regrasp_retires_only_that_objects_prior_placement():
    monitor = PlacementPreservation()
    world = measured_world()
    monitor.observe(task(), world)
    world['objects']['fruit']['pose']['position_m'][2] += .1
    world['robot_state']['held_tool_id'] = 'fruit'
    assert monitor.observe(task('acquire'), world) == []
    assert monitor.relations == {}


def test_moving_the_support_preserves_relative_placement():
    monitor = PlacementPreservation()
    world = measured_world()
    monitor.observe(task(), world)
    for obj in world['objects'].values():
        obj['pose']['position_m'][0] += .3
    assert monitor.observe(task('transport', 'plate', 'tray'), world)[0]['passed'] is True


@pytest.mark.parametrize('bad_state', ['missing', 'floating', 'held', 'overhang'])
def test_invalid_or_unknown_placement_is_never_registered_as_success(bad_state):
    monitor = PlacementPreservation()
    world = measured_world()
    if bad_state == 'missing':
        del world['objects']['fruit']['dimensions_m']
    elif bad_state == 'floating':
        world['objects']['fruit']['pose']['position_m'][2] += .02
    elif bad_state == 'held':
        world['robot_state']['attached_object_id'] = 'fruit'
    else:
        world['objects']['fruit']['pose']['position_m'][0] = .15
    with pytest.raises(PlacementPreservationError, match='PLACEMENT_GOAL_NOT_SATISFIED'):
        monitor.observe(task(), world)
    assert not monitor.relations


def test_missing_geometry_after_success_is_a_preservation_failure():
    monitor = PlacementPreservation()
    world = measured_world()
    monitor.observe(task(), world)
    del world['objects']['fruit']
    with pytest.raises(PlacementPreservationError, match='PLACEMENT_PRESERVATION_FAILED'):
        monitor.observe(task('MOVE', 'other'), world)


def test_live_session_records_disturbance_and_does_not_complete_subgoal(monkeypatch, tmp_path):
    from tuj.m5_motion.scripted_grasps import live
    from tuj.m5_motion import container_settle
    from tuj.m5_motion.tests.test_scripted_grasps import request_for, ENTRIES
    req = request_for(ENTRIES[4], action='MOVE')
    world = req.world.model_copy(deep=True)
    world.objects.update(measured_world()['objects'])
    world.robot_state.attached_object_id = None
    world.robot_state.held_tool_id = None
    runtime = SimpleNamespace(env=object())
    monkeypatch.setattr(live, 'snapshot', lambda *_: world.model_copy(deep=True))
    monkeypatch.setattr(live, 'resolve', lambda _: None)
    monkeypatch.setattr(live, '_unwrap_plan', lambda result, _: result)
    monkeypatch.setattr(container_settle, 'configure_container_settle', lambda *_: None)
    monkeypatch.setattr(live.MotionPlanStore, 'save_plan', lambda *_a, **_k: tmp_path / 'plan.json')
    def planner(_):
        return object()
    planner.collision_context_factory = SimpleNamespace(compiler=None)
    class Executor:
        def __init__(self, *a, **k):
            pass
        def execute(self, *a, **k):
            world.objects['fruit']['pose']['position_m'][0] = .2
            return SimpleNamespace(status=SimpleNamespace(value='SUCCESS'), successful=True)
    session = live.ScriptedGraspSession(runtime, tmp_path, tmp_path,
        planner_factory=lambda *_a, **_k: planner, executor_factory=Executor)
    session._placement_preservation.observe(task(), session.world)
    with pytest.raises(PlacementPreservationError):
        session.execute_request(req, completed_subgoal='later_move')
    manifest = json.loads((tmp_path / 'live-execution-manifest.json').read_text())
    assert manifest['status'] == 'FAILED'
    assert manifest['steps'][0]['status'] == 'FAILED'
    assert manifest['steps'][0]['execution_status'] == 'SUCCESS'
    assert 'later_move' not in manifest['final_world']['scene']['completed_subgoals']
    assert manifest['steps'][0]['placement_preservation'][0]['passed'] is False


def test_finger_adjustment_cannot_fall_back_onto_another_occupant(monkeypatch):
    import numpy as np
    from tuj.m5_motion.scripted_grasps import transport
    monkeypatch.setattr(transport, '_grip_beside_payload', lambda _: True)
    g = SimpleNamespace(half=np.array([.02, .02, .02]),
        region_world=np.zeros(3), request=SimpleNamespace(
            constraints=SimpleNamespace(collision_margin_m=.005)),
        _occupants=lambda: [(np.array([.05, 0.]), np.array([.02, .02]), .1, False)],
        usable_half_xy=lambda: np.ones(2), place_xy_away_from=lambda *a, **kw: None,
        seat_clear_of_occupants=lambda xy: False)
    with pytest.raises(ValueError, match='PLACEMENT_FINGER_SPACE_EXHAUSTED'):
        transport._clear_finger_overhang(g, np.zeros(2))
