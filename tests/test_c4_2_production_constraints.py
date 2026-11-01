"""Execution support must never turn a rejected physical EE into a candidate."""
from itertools import combinations
import pytest
from tuj.m4_taskplanner.models import InitialState, Subgoal, TaskGraph, TaskPlannerRequest, ResourceCatalog
from tuj.m5_motion.scripted_grasps.task_constraints import constrain_task_request

SUBSETS=[list(c) for size in range(4) for c in combinations(('2F','3F','vac'),size)]

@pytest.mark.parametrize('name', ['rolling_pin','baguette','cereal','milk'])
@pytest.mark.parametrize('grounded', SUBSETS)
def test_added_recipes_preserve_grounded_set_including_empty(name, grounded):
    request=TaskPlannerRequest(task_graph=TaskGraph(initial_state=InitialState(),subgoals=[
        Subgoal(subgoal_id='pick',action_type='PICK_TOOL',tool_id=name,
                target_ids=[name],feasible_ee=grounded),
    ]),resource_catalog=ResourceCatalog())
    original=request.model_dump()
    result,changes=constrain_task_request(request,'C4_2_DiagonalFitPacking')
    assert result.task_graph.subgoals[0].feasible_ee==grounded
    assert changes[0]['grounded_feasible_ee']==grounded
    assert set(result.task_graph.subgoals[0].action_parameters[
        'execution_compatibility']['supported_ee'])=={'2F','3F','vac'}
    assert request.model_dump()==original
