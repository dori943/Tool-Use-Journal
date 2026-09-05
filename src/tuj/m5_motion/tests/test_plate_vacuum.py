"""The validated suction alternative must not change the default 2F contract."""
from types import SimpleNamespace
import pytest
from tuj.m5_motion.tests.test_scripted_grasps import request_for
from tuj.m5_motion.scripted_grasps.registry import ENTRIES, ALTERNATIVE_ENTRIES, resolve, integration_status
from tuj.m5_motion.scripted_grasps.objects.plate_vacuum import grasp_plate_vacuum
from tuj.m5_motion.scripted_grasps.profiles import configure_environment
from tuj.m5_motion.tests.test_scripted_task_constraints import make_request
from tuj.m5_motion.scripted_grasps.task_constraints import constrain_task_request


def test_plate_vacuum_is_selected_only_by_exact_scene_and_ee():
    request=request_for(ENTRIES[0])
    assert resolve(request)==ENTRIES[0]
    assert integration_status(resolve(request))=='EXPERIMENTAL'
    request.task.ee='vac'
    entry=resolve(request)
    assert entry==ALTERNATIVE_ENTRIES[0]
    assert entry.function() is grasp_plate_vacuum
    assert entry.recipe().object_id=='plate'
    assert integration_status(entry)=='VALIDATED'
    request.world.metadata['environment_name']='C2_1_ObjectSorting'
    assert resolve(request) is None


def test_plate_vacuum_does_not_accept_another_object():
    with pytest.raises(ValueError,match='WRONG_TARGET_OBJECT'):
        grasp_plate_vacuum(SimpleNamespace(),object_id='lid')


@pytest.mark.parametrize('ee',[None,'2F'])
def test_c1_1_native_and_2f_profiles_do_not_rebuild(ee):
    # A rebuild requires _load_model; this object deliberately has no such API.
    env=SimpleNamespace()
    assert configure_environment(env,'C1_1_LegoSweep',ee) is env
    assert env.scripted_grasp_profile['correction']['policy']=='NATIVE_HAND'


@pytest.mark.parametrize('feasible,expected',[(['vac'],['vac']),(['2F','vac'],['2F'])])
def test_task_preparation_preserves_grounded_vacuum_and_existing_default(feasible,expected):
    request=make_request(feasible)
    for subgoal in request.task_graph.subgoals:
        subgoal.tool_id='plate'
        subgoal.target_ids=['plate']
    prepared,changes=constrain_task_request(request,'C1_1_LegoSweep')
    assert all(s.feasible_ee==expected for s in prepared.task_graph.subgoals)
    assert all(s.feasible_ee==feasible for s in request.task_graph.subgoals)
    assert all(c['scripted_feasible_ee']==expected for c in changes)
