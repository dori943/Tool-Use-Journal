"""Coverage and failure semantics; no physical success is inferred from registration."""
import importlib.util
import json
from pathlib import Path

import pytest
from tuj.m5_motion.scripted_grasps.validate import record_controller_result, controller_validate_case

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('table2_verify', ROOT / 'scripts/verify_table2_coverage.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def test_unknown_and_unexecuted_are_not_physical_failures():
    rows = [dict(object='rolling_pin', ee='2F', inferred_feasible=True, status='PASS', grasp_attempted=True),
            dict(object='milk', ee='2F', inferred_feasible=None, status='EE_MISMATCH', grasp_attempted=False),
            dict(object='lid', ee='2F', inferred_feasible=False, status='ERROR', grasp_attempted=False)]
    res = verify.summarize_trials('c4_2', rows, ['2F'])['2F']
    assert res['coverage'] == '1/6'
    assert res['inferred'] == res['inferred_grasped'] == 1
    assert res['failed'] == []
    assert res['not_registered'] == ['milk']
    assert res['errors'] == ['lid']
    assert not res['matrix_complete']


def test_partial_inference_keeps_unknown_records_absent(tmp_path):
    path = tmp_path / 'm1.json'
    path.write_text(json.dumps({'nodes': [{'id':'obj_milk_milk', 'class':'milk',
                                          'ee':{'3F':{'feasible':True}}}]}))
    assert verify.reasoner_feasible('c4_2', path) == {'milk':['3F']}


@pytest.mark.parametrize('error_type, status', [('AttributeError','ERROR'),('ToolUseJournalRuntimeError','ERROR'),('ToolUseJournalAttachmentGeometryError','FAIL'),('GraspFailure','FAIL'),(None,'FAIL')])
def test_wrapped_driver_error_is_distinguished(error_type, status):
    report = {'failures':[]}
    record_controller_result(report, {'status':'FAILED','error_type':error_type,'failure_stage':'CLOSE',
                                      'failure_reason':'example','attachment_used':False})
    assert report['controller_status'] == status
    assert report['result']['failure_stage'] == 'CLOSE'


def test_acquisition_and_post_grasp_attachment_are_distinct():
    report = {'failures':[]}
    record_controller_result(report, {'status':'SUCCESS', 'acquisition_attachment_used':False,
                                      'attachment_used':True, 'attachment_phase':'POST_VALIDATED_GRASP'})
    assert report['controller_status'] == 'PASS'
    assert report['result']['acquisition_attachment_used'] is False
    assert report['result']['attachment_used'] is True


def test_unregistered_case_saves_report_without_simulation(tmp_path):
    report = controller_validate_case('unknown_test_object', '2F', tmp_path,
                                      environment='C4_2_DiagonalFitPacking')
    assert report['controller_status'] == 'NOT_REGISTERED'
    assert not report['simulation_started'] and not report['grasp_attempted']
    assert json.loads((tmp_path/'validation_report.json').read_text()) == report


def test_existing_output_is_rejected_before_any_execution(tmp_path):
    with pytest.raises(SystemExit):
        verify.main(['--out-root',str(tmp_path),'--doc',str(tmp_path/'doc.md')])


@pytest.mark.parametrize('driver_error', [None, 'AttributeError', 'GraspFailure'])
def test_controller_records_ticks_and_preserves_wrapped_errors(monkeypatch, tmp_path, driver_error):
    from types import SimpleNamespace
    import numpy as np
    from tuj.m5_motion.scripted_grasps import validate, context, profiles
    from tuj.m5_motion.scripted_grasps.runtime import GraspFailure
    from tuj.m5_motion import tool_use_journal, tool_use_journal_runtime, generic_runner

    data=SimpleNamespace(xpos=np.zeros((1,3)), xmat=np.eye(3).reshape(1,9),
                         qpos=np.zeros(1), qvel=np.zeros(1), ctrl=np.zeros(1), time=0.)
    env=SimpleNamespace(obj_body_id={'milk':0}, scripted_grasp_profile={})
    runtime=SimpleNamespace(env=env, scripted_render=False, close=lambda:None)
    adapter=SimpleNamespace(data=data, model=object())
    monkeypatch.setattr(tool_use_journal_runtime.ToolUseJournalEERuntime,
                        'from_repository_for_controller', lambda *a,**k:runtime)
    monkeypatch.setattr(tool_use_journal,'ToolUseJournalEnvironmentAdapter',lambda _:adapter)
    monkeypatch.setattr(tool_use_journal,'_object_record',lambda *a:{'pose':{}})
    monkeypatch.setattr(profiles,'settle_tool_use_journal_free_objects',lambda *a,**k:None)
    monkeypatch.setattr(validate,'static_validate_case',lambda *a,**k:{'object_id':'milk', 'ee':'3F',
                        'static_status':'FAIL','failures':['SYNTHETIC_STATIC_WARNING'],'warnings':[]})
    monkeypatch.setattr(validate,'resolve_case',lambda *a,**k:{'entry':SimpleNamespace(ee='3F',scene_object_id='milk')})
    recorder_events=[]
    class Recorder:
        def __init__(self,*a,**k): pass
        def hold_final_frame(self,*a): recorder_events.append('final')
        def close(self): recorder_events.append('closed')
    monkeypatch.setattr(generic_runner,'GenericSimulationVideoRecorder',Recorder)
    def execute(*args,**kwargs):
        assert runtime.scripted_render is True
        if driver_error:
            output=args[2]; output.mkdir()
            (output/'result.json').write_text(json.dumps({'status':'FAILED','error_type':driver_error,
                                                        'failure_stage':'CLOSE','failure_reason':'test'}))
            raise GraspFailure('wrapped error')
        return {'status':'SUCCESS','attachment_used':False}
    monkeypatch.setattr(context,'execute_grasp',execute)
    report=validate.controller_validate_case('milk','3F',tmp_path,video=tmp_path/'video.mp4')
    expected='PASS' if driver_error is None else 'ERROR' if driver_error=='AttributeError' else 'FAIL'
    assert report['controller_status']==expected
    assert report['static_failures']==['SYNTHETIC_STATIC_WARNING']
    assert 'SYNTHETIC_STATIC_WARNING' not in report['failures']
    assert report['simulation_started'] and report['grasp_attempted']
    assert (tmp_path/'initial_scene.json').exists()
    assert recorder_events[-1]=='closed'
    assert json.loads((tmp_path/'validation_report.json').read_text())['controller_status']==expected


def test_corrupt_worker_report_is_recorded_as_error_not_grasp_failure(monkeypatch,tmp_path):
    from types import SimpleNamespace
    output=tmp_path/'case'
    def worker(*args,**kwargs):
        (output/'validation_report.json').write_text('{broken')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(verify.subprocess,'run',worker)
    result=verify.run_case('c4_2','milk','2F',seed=0,video=False,camera='agentview',output=output,timeout_s=1)
    assert result['status']=='ERROR'
    assert result['failure_category']=='IMPLEMENTATION_OR_RUNTIME'
    assert not result['grasp_attempted']


@pytest.mark.parametrize('name,ee', [
    (f'{kind}_{suffix}', '3F')
    for kind in ('mug', 'fork', 'spoon', 'fruit', 'bread') for suffix in ('a', 'b')
] + [('bread_a', 'vac'), ('bread_b', 'vac')])
def test_production_filter_recognizes_instance_type_recipes(name, ee):
    assert verify.has_production_recipe(name, ee, verify.TASKS['c3_2'].cls)


def test_production_filter_respects_exact_instance_precedence(monkeypatch):
    from types import SimpleNamespace
    from tuj.m5_motion.scripted_grasps import registry
    rows = (SimpleNamespace(object_id='mug_a', ee='2F', environment='test'),
            SimpleNamespace(object_id='mug', ee='3F', environment='test'))
    monkeypatch.setattr(registry, 'ENTRIES', rows)
    monkeypatch.setattr(registry, 'ALTERNATIVE_ENTRIES', ())
    monkeypatch.setattr(registry, 'PENDING_INTEGRATION', {})
    assert verify.has_production_recipe('mug_a', '2F', 'test')
    assert not verify.has_production_recipe('mug_a', '3F', 'test')
    assert verify.has_production_recipe('mug_b', '3F', 'test')


@pytest.mark.parametrize('registered_only,experimental_only,expected', [
    (True, False, ['3F']), (False, True, ['2F', 'vac'])])
def test_run_task_filters_use_production_resolution(monkeypatch, tmp_path,
                                                   registered_only, experimental_only, expected):
    def fake_case(task, name, ee, **kwargs):
        return dict(object=name, ee=ee, status='FAIL', failures=['fixture'],
                    grasp_attempted=False)
    monkeypatch.setattr(verify, 'run_case', fake_case)
    result = verify.run_task('c3_2', seed=0, denominator='none', all_ees=True,
                            video=False, out_root=tmp_path, ee_ids=['2F', '3F', 'vac'],
                            objects=['mug_a'], registered_only=registered_only,
                            experimental_only=experimental_only)
    assert [row['ee'] for row in result['trials']] == expected
