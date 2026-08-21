"""Replanning: no-good learning, scope handling, no re-execution."""

from __future__ import annotations

from task_planner.diagnostics import PlanStatus
from task_planner.models import ExecutionState, FailureFeedback, FailureType
from task_planner.planner import plan
from task_planner.replanning import NoGoodSet, apply_failure_feedback, replan

from conftest import cond, make_request, prop, sg


def _two_candidate_request():
    subgoals = [sg("S1", targets=["obj1"], feasible=["A"])]
    proposals = {
        "S1": [
            prop("S1-c1", "S1", "A", score=0.9),
            prop("S1-c2", "S1", "A", score=0.8),
        ]
    }
    return make_request(subgoals, proposals=proposals, initial_ee="A")


def _assignment(result, subgoal_id: str):
    assert result.selected_plan is not None
    return next(
        a
        for a in result.selected_plan.candidate_assignments
        if a.subgoal_id == subgoal_id
    )


def test_scene_failure_bans_candidate_for_scene_and_switches_candidate() -> None:
    request = _two_candidate_request()
    first = plan(request)
    assert first.status is PlanStatus.SUCCESS
    assert _assignment(first, "S1").candidate_id == "S1-c1"

    execution_state = ExecutionState(
        completed_subgoals=[],
        current_ee="A",
        facts=list(request.task_graph.initial_state.facts),
        scene_signature="initial",
    )
    feedback = FailureFeedback(
        failure_type=FailureType.CANDIDATE_INVALID,
        subgoal_id="S1",
        candidate_id="S1-c1",
        scene_signature="initial",
        scope="scene",
    )
    second = replan(request, execution_state, feedback, previous_result=first)
    assert second.status is PlanStatus.SUCCESS
    assert _assignment(second, "S1").candidate_id == "S1-c2"
    assert second.replan_trace is not None
    assert second.replan_trace.previous_candidate_id == "S1-c1"
    assert second.replan_trace.newly_selected_candidate_id == "S1-c2"
    assert second.replan_trace.added_no_goods == ["scene:initial:S1-c1"]
    assert second.replan_trace.new_cost_vector is not None


def test_scene_scoped_ban_does_not_apply_in_other_scenes() -> None:
    no_goods = NoGoodSet()
    feedback = FailureFeedback(
        failure_type=FailureType.CANDIDATE_INVALID,
        candidate_id="S1-c1",
        scene_signature="scene-x",
        scope="scene",
    )
    apply_failure_feedback(no_goods, feedback)
    assert ("scene-x", "S1-c1") in no_goods.scene_candidates
    assert "S1-c1" not in no_goods.global_candidates


def test_candidate_invalid_is_global_ban() -> None:
    request = _two_candidate_request()
    execution_state = ExecutionState(
        completed_subgoals=[],
        current_ee="A",
        facts=list(request.task_graph.initial_state.facts),
    )
    feedback = FailureFeedback(
        failure_type=FailureType.CANDIDATE_INVALID,
        subgoal_id="S1",
        candidate_id="S1-c1",
    )
    second = replan(request, execution_state, feedback)
    assert second.status is PlanStatus.SUCCESS
    assert _assignment(second, "S1").candidate_id == "S1-c2"
    assert second.replan_trace is not None
    assert second.replan_trace.added_no_goods == ["global:S1-c1"]


def test_completed_subgoals_are_not_reexecuted() -> None:
    subgoals = [
        sg(
            "S1",
            targets=["obj1"],
            feasible=["A"],
            estab=[cond("holding", "obj1")],
        ),
        sg(
            "S2",
            targets=["obj1"],
            feasible=["A"],
            pre=[cond("holding", "obj1", pass_=False)],
            estab=[cond("in_region", "obj1", "box")],
            destroy=[cond("holding", "obj1")],
            action="place",
        ),
    ]
    proposals = {
        "S1": [prop("S1-c1", "S1", "A")],
        "S2": [prop("S2-c1", "S2", "A")],
    }
    request = make_request(
        subgoals, edges=[("S1", "S2")], proposals=proposals, initial_ee="A"
    )
    execution_state = ExecutionState(
        completed_subgoals=["S1"],
        current_ee="A",
        facts=[cond("holding", "obj1", pass_=True)],
        scene_signature="mid-task",
    )
    feedback = FailureFeedback(failure_type=FailureType.SCENE_CHANGED)
    result = replan(request, execution_state, feedback)
    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    assert result.selected_plan.subgoal_order == ["S2"]
    assert result.replan_trace is not None
    assert result.replan_trace.added_no_goods == []


def test_transition_failure_bans_transition_signature() -> None:
    no_goods = NoGoodSet()
    feedback = FailureFeedback(
        failure_type=FailureType.TRANSITION_INVALID,
        transition_signature=["A", "", "S1-c1"],
    )
    added = apply_failure_feedback(no_goods, feedback)
    assert ("A", "", "S1-c1") in no_goods.transitions
    assert added == ["transition:A||S1-c1"]

    # A banned transition removes the edge for that exact context.
    request = _two_candidate_request()
    result = plan(request, no_goods=no_goods)
    assert result.status is PlanStatus.SUCCESS
    assert _assignment(result, "S1").candidate_id == "S1-c2"


def test_replanning_rejects_non_predecessor_closed_completed_prefix() -> None:
    request = make_request(
        [
            sg("S1", targets=["obj1"]),
            sg("S2", targets=["obj2"]),
        ],
        edges=[("S1", "S2")],
        proposals={
            "S1": [prop("S1-c1", "S1", "A")],
            "S2": [prop("S2-c1", "S2", "A")],
        },
    )
    execution_state = ExecutionState(
        completed_subgoals=["S2"],
        current_ee="A",
    )
    result = replan(
        request,
        execution_state,
        FailureFeedback(failure_type=FailureType.SCENE_CHANGED),
    )
    assert result.status is PlanStatus.INVALID_INPUT
