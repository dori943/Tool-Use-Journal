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


def make_request(ee,tool_id='spatula'):
    return TaskPlannerRequest(task_graph=TaskGraph(initial_state=InitialState(),subgoals=[
        Subgoal(subgoal_id='pick',action_type='PICK_TOOL',tool_id=tool_id,target_ids=[tool_id],feasible_ee=ee),
        Subgoal(subgoal_id='flatten',action_type='tool_act',tool_id=tool_id,target_ids=['dough'],feasible_ee=ee),
    ]),resource_catalog=ResourceCatalog())


def test_scripted_ee_contract_covers_acquire_and_use_without_mutating_input():
    original=make_request(['2F','3F'])
    constrained,changes=constrain_task_request(original,'C1_2_DoughFlatten')
    assert [s.feasible_ee for s in constrained.task_graph.subgoals]==[['3F'],['3F']]
    assert all(s.feasible_ee==['2F','3F'] for s in original.task_graph.subgoals)
    assert constrained.task_graph.subgoals[1].target_ids==['dough']
    assert len(changes)==2
    assert all(c['source']=='scripted_grasp_registry' for c in changes)
    assert constrained.task_graph.subgoals[0].action_parameters[
        'execution_compatibility']['supported_ee']==['3F']


def test_scripted_ee_contract_does_not_override_grounded_infeasibility():
    with pytest.raises(ValueError,match='SCRIPTED_GRASP_EE_INFEASIBLE'):
        constrain_task_request(make_request(['2F']),'C1_2_DoughFlatten')


def test_scripted_ee_contract_is_environment_scoped():
    constrained,changes=constrain_task_request(make_request(['2F','3F']),'C2_1_ObjectSorting')
    assert not changes
    assert constrained.task_graph.subgoals[0].feasible_ee==['2F','3F']


def test_spoon_contract_keeps_both_m4_hand_candidates_and_selected_order():
    constrained,changes=constrain_task_request(
        make_request(['3F','2F'],tool_id='spoon'),'C2_1_ObjectSorting')
    assert [s.feasible_ee for s in constrained.task_graph.subgoals]==[
        ['3F','2F'],['3F','2F']]
    assert len(changes)==2
    contract=constrained.task_graph.subgoals[0].action_parameters[
        'execution_compatibility']
    assert contract['supported_ee']==['2F','3F']
    assert contract['selected_feasible_ee']==['3F','2F']


def test_c3_2_bread_instance_contract_keeps_3f_and_vac():
    """bread_a shares type-keyed recipes; M4 may keep either mounted EE."""
    constrained,changes=constrain_task_request(
        make_request(['vac','3F'],tool_id='bread_a'),
        'C3_2_BreakfastTrayPreparation')
    assert constrained.task_graph.subgoals[0].feasible_ee==['vac','3F']
    assert changes[0]['object_id']=='bread'
    assert changes[0]['scripted_feasible_ee']==['vac','3F']
    contract=constrained.task_graph.subgoals[0].action_parameters[
        'execution_compatibility']
    assert set(contract['supported_ee'])=={'3F','vac'}


@pytest.mark.parametrize(('environment','tool_id'),(
    ('C1_1_LegoSweep','plate'),
    ('C2_1_ObjectSorting','plate'),
    ('C3_1_ObjectSorting','plate'),
    ('C3_2_BreakfastTrayPreparation','plate_a'),
    ('C3_2_BreakfastTrayPreparation','plate_b'),
))
def test_plate_vac_contract_is_available_in_all_registered_tasks(environment,tool_id):
    constrained,changes=constrain_task_request(
        make_request(['2F','3F','vac'],tool_id=tool_id),environment)
    expected=['2F','vac'] if environment=='C1_1_LegoSweep' else ['vac']
    assert [s.feasible_ee for s in constrained.task_graph.subgoals]==[expected,expected]
    assert len(changes)==2
    contract=constrained.task_graph.subgoals[0].action_parameters[
        'execution_compatibility']
    assert contract['object_id']==tool_id
    assert contract['supported_ee']==expected


@pytest.mark.parametrize('target_id', ['cereal', 'obj_cereal_cereal'])
def test_cereal_empty_feasibility_is_not_revived_by_scripted_recipe(target_id):
    """A recipe's availability cannot make a rejected grasp physically feasible."""
    original = TaskPlannerRequest(
        task_graph=TaskGraph(
            initial_state=InitialState(),
            subgoals=[Subgoal(
                subgoal_id='SG1_s2_d1',
                action_type='acquire',
                target_ids=[target_id],
                feasible_ee=[],
                feasible_ee_source='gk',
            )],
        ),
        resource_catalog=ResourceCatalog.model_validate({
            'end_effectors': {ee: {} for ee in ('2F', '3F', 'vac')},
        }),
    )
    before = original.model_dump()

    constrained, changes = constrain_task_request(
        original, 'C4_2_DiagonalFitPacking')

    assert original.model_dump() == before
    assert constrained.task_graph.subgoals[0].feasible_ee == []
    assert len(changes) == 1
    assert changes[0]['grounded_feasible_ee'] == []
    assert changes[0]['scripted_feasible_ee'] == []
    contract = constrained.task_graph.subgoals[0].action_parameters[
        'execution_compatibility']
    assert contract['supported_ee'] == ['2F']
    assert contract['selected_feasible_ee'] == []

    from tuj.m4_taskplanner.diagnostics import PlanStatus, ReasonCode
    from tuj.m4_taskplanner.planner import plan

    result = plan(constrained)
    assert result.status is PlanStatus.INFEASIBLE_NO_CANDIDATE
    assert result.selected_plan is None
    assert result.search_stats.states_popped == 0
    assert [(r.subgoal_id, r.reason_code) for r in result.rejections] == [
        ('SG1_s2_d1', ReasonCode.EMPTY_FEASIBLE_EE)]
    assert original.model_dump() == before


def test_empty_tool_feasibility_stays_empty_for_acquire_and_use():
    original = make_request([])
    before = original.model_dump()

    constrained, changes = constrain_task_request(original, 'C1_2_DoughFlatten')

    assert [s.feasible_ee for s in constrained.task_graph.subgoals] == [[], []]
    assert all(change['scripted_feasible_ee'] == [] for change in changes)
    assert original.model_dump() == before
