import pytest
from tuj.m4_taskplanner.models import InitialState, Subgoal, TaskGraph, TaskPlannerRequest, ResourceCatalog
from tuj.m5_motion.scripted_grasps.task_constraints import constrain_task_request


def test_scene_aliases_preserve_full_class_and_instance_names():
    from tuj.m5_motion.scripted_grasps.task_constraints import scene_id_aliases
    scene={'nodes':[
        {'id':'obj_rolling_pin_rolling_pin','class':'rolling_pin'},
        {'id':'obj_bread_a_bread_a','class':'bread_a'},
        {'id':'obj_plate_dinner_plate','class':'plate'},
        {'id':'obj_unknown_plate','class':'mug'},
        {'id':'obj_opaque'}]}
    assert scene_id_aliases(scene)=={'obj_rolling_pin_rolling_pin':'rolling_pin',
        'obj_bread_a_bread_a':'bread_a','obj_plate_dinner_plate':'dinner_plate'}


def make_request(ee):
    return TaskPlannerRequest(task_graph=TaskGraph(initial_state=InitialState(),subgoals=[
        Subgoal(subgoal_id='pick',action_type='PICK_TOOL',tool_id='spatula',target_ids=['spatula'],feasible_ee=ee),
        Subgoal(subgoal_id='flatten',action_type='tool_act',tool_id='spatula',target_ids=['dough'],feasible_ee=ee),
    ]),resource_catalog=ResourceCatalog())


def test_scripted_ee_contract_covers_acquire_and_use_without_mutating_input():
    original=make_request(['2F','3F'])
    constrained,changes=constrain_task_request(original,'C1_2_DoughFlatten')
    assert [s.feasible_ee for s in constrained.task_graph.subgoals]==[['3F'],['3F']]
    assert all(s.feasible_ee==['2F','3F'] for s in original.task_graph.subgoals)
    assert constrained.task_graph.subgoals[1].target_ids==['dough']
    assert len(changes)==2


def test_scripted_ee_contract_does_not_override_grounded_infeasibility():
    with pytest.raises(ValueError,match='SCRIPTED_GRASP_EE_INFEASIBLE'):
        constrain_task_request(make_request(['2F']),'C1_2_DoughFlatten')


def test_scripted_ee_contract_is_environment_scoped():
    constrained,changes=constrain_task_request(make_request(['2F','3F']),'C2_1_ObjectSorting')
    assert not changes
    assert constrained.task_graph.subgoals[0].feasible_ee==['2F','3F']
