"""Goal verdicts come from observed geometry/history, never M5 status alone."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.evaluation.task_goals import evaluate_task_goal, world_points
from tuj.evaluation.saved_runs import evaluate_saved_run, attach_live_goal_evaluation
from scripts.run import write_run_result
from scripts.run_ablation_batch import collect_record


def box(x=0., y=0., z=0., size=(.1, .1, .1), angle=0.):
    return {"pose": {"position_m": [x, y, z], "orientation_xyzw": Rotation.from_euler('z', angle, degrees=True).as_quat().tolist()},
            "dimensions_m": list(size)}


def world(objects):
    return {"objects": objects, "robot_state": {"attached_object_id": None, "held_tool_id": None}, "metadata": {}}


def sorting():
    objects = {"blue_tray": box(size=(2., 2., .02))}
    for i, name in enumerate(["spoon", "plate", "bread", "apple", "mug"]):
        objects[name] = box(y=(i - 2) * .25, z=.06)
    return world(objects)


def check(result, name):
    return next(c for c in result['checks'] if c['name'] == name)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_unknown_is_not_success_and_known_violation_dominates():
    missing = world({})
    assert evaluate_task_goal('c3_1', missing, missing)['goal_success'] is None
    missing['objects'].update(blue_tray=box(size=(1, 1, .1)), apple=box(x=2))
    result = evaluate_task_goal('c3_1', missing, missing)
    assert result['goal_success'] is False
    assert result['unknown_checks']


def test_rotation_footprint_not_just_center():
    state = sorting()
    state['objects']['spoon'] = box(x=.92, z=.06, size=(.3, .02, .1), angle=45)
    result = evaluate_task_goal('c3_1', state, state)
    assert check(result, 'spoon.inside.blue_tray')['passed'] is False
    corners = world_points(state['objects']['spoon'])
    assert corners[:, 0].max() == pytest.approx(.92 + .16 / 2**.5)


def test_sorting_passes_only_when_supported_and_released():
    state = sorting()
    assert evaluate_task_goal('c3_1', state, state)['goal_success'] is True
    floating = deepcopy(state)
    floating['objects']['apple']['pose']['position_m'][2] = 1
    assert evaluate_task_goal('c3_1', state, floating)['goal_success'] is False
    state['robot_state']['attached_object_id'] = 'mug'
    assert evaluate_task_goal('c3_1', state, state)['goal_success'] is False


def test_order_uses_actual_successful_history():
    state = sorting()
    names = ['spoon', 'plate', 'bread', 'apple', 'mug']
    events = [{'status': 'SUCCESS', 'action_type': action, 'target_ids': [name]} for name in names for action in ['acquire', 'place']]
    assert evaluate_task_goal('c2_1', state, state, history=events)['goal_success'] is True
    assert evaluate_task_goal('c2_1', state, state)['goal_success'] is None
    events[0]['target_ids'] = ['mug']
    assert evaluate_task_goal('c2_1', state, state, history=events)['goal_success'] is False
    events[0]['target_ids'] = None
    assert evaluate_task_goal('c2_1', state, state, history=events)['goal_success'] is None


def breakfast():
    objects = {}
    for side, offset in [('a', -1), ('b', 1)]:
        objects[f'tray_{side}'] = box(x=offset, size=(1, 1, .02))
        objects[f'plate_{side}'] = box(x=offset, z=.025, size=(.3, .3, .03))
        objects[f'mug_{side}'] = box(x=offset, y=.3, z=.06)
        objects[f'fork_{side}'] = box(x=offset, y=-.25, z=.02, size=(.2, .02, .02))
        objects[f'spoon_{side}'] = box(x=offset, y=-.35, z=.02, size=(.2, .02, .02))
        objects[f'bread_{side}'] = box(x=offset-.06, z=.065, size=(.05, .05, .05))
        objects[f'fruit_{side}'] = box(x=offset+.06, z=.065, size=(.05, .05, .05))
    return world(objects)


def test_breakfast_full_composition_and_neatness():
    state = breakfast()
    assert evaluate_task_goal('c3_2', state, state)['goal_success'] is True
    state['objects']['fruit_a']['pose']['position_m'][0] += .2
    assert 'fruit_a.inside.plate_a' in evaluate_task_goal('c3_2', state, state)['failed_checks']
    state = breakfast()
    state['objects']['spoon_a']['pose']['orientation_xyzw'] = Rotation.from_euler('z', 40, degrees=True).as_quat().tolist()
    assert 'utensils_a.parallel' in evaluate_task_goal('c3_2', state, state)['failed_checks']


def extraction():
    return world({'appliance_left': box(y=-.2, z=.15, size=(.5, .2, .3)),
        'appliance_right': box(y=.2, z=.15, size=(.5, .2, .3)),
        'card': box(z=.01, size=(.08, .03, .02))})


def test_card_requires_clearance_motion_and_remaining_on_table():
    start = extraction()
    end = deepcopy(start)
    end['objects']['card']['pose']['position_m'][0] = -.32
    assert evaluate_task_goal('c4_1', start, end)['goal_success'] is True
    end['objects']['card']['pose']['position_m'][2] -= .5
    assert evaluate_task_goal('c4_1', start, end)['goal_success'] is False
    end = deepcopy(start)
    for name in ['appliance_left', 'appliance_right']:
        end['objects'][name]['pose']['position_m'][0] = 2
    assert evaluate_task_goal('c4_1', start, end)['goal_success'] is False


def test_flatten_uses_measured_unloaded_reduction():
    state = world({'dough': {'deformable_metrics': {'initial_height_m': .1, 'height_m': .077, 'volume_ratio': 1.,
        'tool_contact': {'impulse_ns': 1., 'current_force_n': 0., 'unloaded_duration_s': 2., 'peak_force_n': 2., 'force_limit_n': 3., 'start_height_m': .09}},
        'flattening_configuration': {'height_reduction_fraction': [.3, .5], 'unloaded_observation_s': 1., 'volume_tolerance_fraction': .01}}})
    assert evaluate_task_goal('c1_2', state, state)['goal_success'] is False
    state['objects']['dough']['deformable_metrics']['height_m'] = .06
    assert evaluate_task_goal('c1_2', state, state)['goal_success'] is True
    del state['objects']['dough']['deformable_metrics']['tool_contact']
    assert evaluate_task_goal('c1_2', state, state)['goal_success'] is None


def test_cover_needs_measured_contact_not_geometry_alone():
    objects = {n: box(x=(i-2)*.2, z=0) for i,n in enumerate(['rolling_pin','baguette','whisk','cereal','milk'])}
    objects['packing_box'] = box(size=(2., 2., 2.))
    objects['packing_box']['packing_metadata'] = {'interior_dimensions_m': [1.8]*3, 'interior_center_m': [0.]*3, 'opening_top_z_m': .9}
    objects['lid'] = box(z=.91, size=(2., 2., .02))
    objects['lid']['solid_box_geometry'] = {'source': 'MUJOCO_BOX', 'half_size_m': [1.,1.,.01], 'center_in_body_m': [0.]*3, 'rotation_in_body': np.eye(3).tolist()}
    state = world(objects)
    assert evaluate_task_goal('c4_2', state, state)['goal_success'] is None
    evidence = {'succeeded': True, 'surface_support': {'contact_count': 2, 'upward_force_n': 1.}}
    assert evaluate_task_goal('c4_2', state, state, cover_support=evidence)['goal_success'] is True


def saved(tmp_path, status='SUCCESS'):
    initial = sorting()
    initial['metadata']['environment_name'] = 'C3_1_ObjectSorting'
    write(tmp_path/'m5/initial_world.json', initial)
    write(tmp_path/'m5/live/run-test/live-execution-manifest.json', {'status':status, 'final_world':initial, 'steps': []})
    write(tmp_path/'m5/m5_summary.json', {'status':status, 'live_run_dir': r'C:\old\m5\live\run-test'})
    return initial


def test_loader_relocates_windows_and_preserves_sources(tmp_path):
    saved(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*.json')}
    result = evaluate_saved_run(tmp_path, task='c3_1')
    assert result['goal_success'] is True
    assert result['source_sha256']
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*.json')}


def test_loader_cannot_reuse_stale_or_nonterminal_success(tmp_path):
    saved(tmp_path, 'IN_PROGRESS')
    assert evaluate_saved_run(tmp_path)['task_goal_status'] == 'INCOMPLETE'
    saved(tmp_path)
    write(tmp_path/'batch_record.json', {'exit_code': -9})
    assert evaluate_saved_run(tmp_path)['task_goal_status'] == 'INCOMPLETE'
    (tmp_path/'batch_record.json').unlink()
    (tmp_path/'m5/m5_summary.json').unlink()
    write(tmp_path/'m5/live/run-other/live-execution-manifest.json', {'status': 'SUCCESS'})
    assert evaluate_saved_run(tmp_path)['task_goal_status'] == 'UNKNOWN'


def test_live_hook_evaluates_failed_execution_and_overrides_stale_summary(tmp_path):
    saved(tmp_path, 'FAILED')
    summary = {'status': 'FAILED', 'live_run_dir': 'run-test', 'manifest': 'run-test/live-execution-manifest.json'}
    attach_live_goal_evaluation(tmp_path/'m5', summary)
    assert summary['task_goal_status'] == 'SATISFIED'
    assert summary['status'] == 'FAILED'
    assert Path(summary['task_goal_evaluation']).is_file()


@pytest.mark.parametrize('execution,goal,success,stage', [('SUCCESS','UNKNOWN',None,'evaluation'), ('SUCCESS','FAILED',False,'goal'), ('SUCCESS','SATISFIED',True,'none'), ('FAILED','SATISFIED',False,'execution')])
def test_result_separates_execution_from_goal(tmp_path, execution, goal, success, stage):
    write_run_result(tmp_path,'c3_1', SimpleNamespace(), method='without-reasoner', attempts=[{'m5_status':execution, 'task_goal_status':goal, 'failure_stage':'execution'}])
    rr=json.loads((tmp_path/'run_result.json').read_text())
    assert rr['success'] is success
    assert rr['failure_stage'] == stage
    write(tmp_path/'m5/m5_summary.json', {'status':execution, 'task_goal_status':'NOT_EVALUATED'})
    rec=collect_record(tmp_path, task='c3_1',seed=0,method='without-reasoner',exit_code=0,duration_s=0,started_at='now')
    assert rec['task_goal_status'] == goal
    assert rec['success'] is success


def test_interchangeable_breakfast_objects_can_swap_trays():
    state = breakfast()
    for kind in ['plate', 'bread', 'fork']:
        a, b = f'{kind}_a', f'{kind}_b'
        state['objects'][a], state['objects'][b] = state['objects'][b], state['objects'][a]
    assert evaluate_task_goal('c3_2', state, state)['goal_success'] is True


def test_failed_runtime_start_cannot_inherit_previous_manifest(tmp_path):
    saved(tmp_path)
    summary = {'status': 'FAILED'}
    attach_live_goal_evaluation(tmp_path/'m5', summary)
    assert summary['task_goal_status'] == 'UNKNOWN'
    assert summary['goal_success'] is None


@pytest.mark.parametrize('execution_fails', [False, True])
def test_live_cli_persists_terminal_goal_when_execution_completes_or_raises(tmp_path, monkeypatch, execution_fails):
    from tuj.m5_motion.scripted_grasps import cli
    from tuj.m5_motion import generic_runner
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime

    state = sorting()
    state['metadata']['environment_name'] = 'C3_1_ObjectSorting'
    final = deepcopy(state)
    final['objects']['apple']['pose']['position_m'][0] = 3.
    initial = SimpleNamespace(model_dump_json=lambda **kw: json.dumps(state))
    closed = []
    runtime = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(ToolUseJournalEERuntime, 'from_repository_for_controller', lambda *a, **kw: runtime)
    monkeypatch.setattr(generic_runner, '_runtime_environment_name', lambda w: 'C3_1_ObjectSorting')
    monkeypatch.setattr(generic_runner, '_runtime_active_ee', lambda w: '2F')
    monkeypatch.setattr(generic_runner, 'validate_selected_plan', lambda *a: {})
    monkeypatch.setattr(cli, 'execution_initial_world', lambda *a, **kw: initial)
    monkeypatch.setattr(cli, 'snapshot', lambda *a: initial)
    monkeypatch.setattr(cli, '_mujoco_simulation_time_s', lambda r: 0.)

    class Session:
        def __init__(self, runtime, repository, output, **kw):
            self.output, self.records = output, []
            self.status = 'FAILED' if execution_fails else 'SUCCESS'
        def save_manifest(self):
            p = self.output/'live-execution-manifest.json'
            write(p, {'status':self.status, 'final_world':final, 'steps':[]})
            return p
        def execute_selected_plan(self, *a, **kw):
            p = self.save_manifest()
            if execution_fails:
                raise RuntimeError('execution failed after measured placement')
            return p
    monkeypatch.setattr(cli, 'ScriptedGraspSession', Session)
    args = SimpleNamespace(video=None, headless=True, camera='agentview', initial_world=None,
                           seed=0, realtime_factor=0., hold_seconds=0.)
    output = tmp_path/'m5'
    output.mkdir()
    code = cli.execute_selected_plan_live(args, None, initial, None, None, tmp_path, output, 'test', {})
    summary = json.loads((output/'m5_summary.json').read_text())
    assert closed == [True]
    assert code == (2 if execution_fails else 0)
    assert summary['status'] == ('FAILED' if execution_fails else 'SUCCESS')
    assert summary['task_goal_status'] == 'FAILED'
    assert summary['goal_success'] is False
    assert Path(summary['task_goal_evaluation']).is_file()


def test_batch_summary_only_unknown_stays_unknown(tmp_path):
    write(tmp_path/'m5/m5_summary.json', {'status':'SUCCESS', 'task_goal_status':'UNKNOWN',
        'task_goal_policy':'task-goals-v1', 'task_goal_evaluation':'m5/task_goal_evaluation.json'})
    rec=collect_record(tmp_path, task='c3_1', seed=0, method='without-reasoner', exit_code=0, duration_s=0, started_at='now')
    assert rec['success'] is None
    assert rec['failure_stage'] == 'evaluation'
    assert rec['task_goal_policy'] == 'task-goals-v1'
    assert rec['reason_code'] == 'GOAL_EVIDENCE_MISSING'


def test_legacy_execution_only_success_not_counted_as_overall_goal_success(tmp_path):
    from scripts.run_ablation_batch import write_summaries
    write(tmp_path/'run_result.json', {'success':True, 'exec_success':True, 'goal_success':None,
        'm5_status':'SUCCESS', 'task_goal_status':'NOT_EVALUATED', 'failure_stage':'none'})
    rec=collect_record(tmp_path, task='c3_1', seed=0, method='without-reasoner', exit_code=0, duration_s=0, started_at='now')
    write_summaries(tmp_path, [rec])
    report=(tmp_path/'errors_summary.md').read_text()
    assert '| c3_1 | 1 | 0/1 |' in report
    assert '### UNKNOWN c3_1' in report


def test_sweep_all_blocks_required_and_piles_allowed():
    objects={'collection_zone_visual':box(size=(2,2,.02))}
    for i in range(12):
        objects[f'block_{i}']=box(x=-.5+i*.08, z=.025, size=(.04,.04,.03))
    state=world(objects)
    assert evaluate_task_goal('c1_1',state,state)['goal_success'] is True
    state['objects']['block_1']=box(x=-.5,z=.055,size=(.04,.04,.03))
    assert evaluate_task_goal('c1_1',state,state)['goal_success'] is True
    del state['objects']['block_11']
    assert evaluate_task_goal('c1_1',state,state)['goal_success'] is None


def test_sandwich_requires_bread_ends_and_supported_layers():
    names=['bread_a','turkey_1','cheese_1','tomato_slice','bread_b']
    objects={'serving_plate':box(size=(.3,.3,.02))}
    for i,name in enumerate(names):
        objects[name]=box(z=.02+i*.02,size=(.1,.1,.02))
    state=world(objects)
    assert evaluate_task_goal('c2_2',state,state)['goal_success'] is True
    objects['bread_b'],objects['cheese_1']=objects['cheese_1'],objects['bread_b']
    assert evaluate_task_goal('c2_2',state,state)['goal_success'] is False


def test_comparison_is_offline_preserves_inputs_and_refuses_overwrite(tmp_path):
    from scripts.evaluate_task_goals import evaluate_comparison
    baseline,current=tmp_path/'old',tmp_path/'new'
    saved(baseline/'c3_1/seed_0')
    saved(current/'c3_1/seed_0')
    before={p:p.read_bytes() for root in (baseline,current) for p in root.rglob('*.json')}
    output=tmp_path/'evaluation'
    totals=evaluate_comparison(baseline,current,output)
    assert totals['current']['end_to_end_success']==1
    assert before=={p:p.read_bytes() for p in before}
    assert json.loads((output/'preservation_check.json').read_text())['all_read_sources_unchanged']
    with pytest.raises(FileExistsError):
        evaluate_comparison(baseline,current,output)
