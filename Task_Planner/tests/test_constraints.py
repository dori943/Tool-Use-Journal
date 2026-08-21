"""Planner-A contract preservation and search-time choice resolution."""

from __future__ import annotations

from task_planner.constraints import PlannerAConstraintEngine
from task_planner.diagnostics import PlanStatus, ReasonCode
from task_planner.models import PlannerAConstraints
from task_planner.planner import plan

from conftest import cond, make_request, prop, sg


def test_mutex_blocks_second_consumer_until_resource_is_reestablished() -> None:
    contract = PlannerAConstraints.model_validate(
        {
            "mutex": [
                {
                    "a": "A",
                    "b": "B",
                    "condition": "C_hand_empty",
                    "predicate": "hand_empty",
                    "args": [],
                    "reestablished_by": ["R"],
                }
            ]
        }
    )
    engine = PlannerAConstraintEngine(contract)

    blocked = engine.blockers("B", frozenset({"A"}), frozenset())
    allowed = engine.blockers(
        "B", frozenset({"A", "R"}), frozenset({("hand_empty", ())})
    )

    assert [item.kind for item in blocked] == ["mutex"]
    assert allowed == ()


def test_open_condition_selects_a_completed_producer_and_records_trace() -> None:
    subgoals = [
        sg(
            "A_target",
            action="sense",
            pre=[cond("ready", pass_=False, condition_id="C_target_ready")],
        ),
        sg(
            "B_producer",
            action="sense",
            estab=[cond("ready", condition_id="C_b_ready")],
        ),
        sg(
            "C_producer",
            action="sense",
            estab=[cond("ready", condition_id="C_c_ready")],
        ),
    ]
    proposals = {
        item.subgoal_id: [prop(f"cand-{item.subgoal_id}", item.subgoal_id, "A")]
        for item in subgoals
    }
    request = make_request(subgoals, proposals=proposals)
    request.planner_a.constraints = PlannerAConstraints.model_validate(
        {
            "open_conditions": [
                {
                    "subgoal": "A_target",
                    "condition": "C_target_ready",
                    "candidates": ["B_producer", "C_producer"],
                }
            ]
        }
    )

    result = plan(request, suitability_scorer=None)

    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    assert result.selected_plan.subgoal_order.index("B_producer") < (
        result.selected_plan.subgoal_order.index("A_target")
    )
    open_trace = next(
        item
        for item in result.selected_plan.constraint_trace
        if item.constraint_type == "open_condition"
    )
    assert open_trace.status == "satisfied"
    assert open_trace.details["selected_producer"] == "B_producer"


def test_disjunctive_threat_is_demoted_after_causal_link() -> None:
    subgoals = [
        sg(
            "A_producer",
            action="sense",
            estab=[cond("protected", condition_id="C_est")],
        ),
        sg(
            "B_consumer",
            action="sense",
            pre=[cond("protected", pass_=False, condition_id="C_pre")],
            estab=[cond("can_threat", condition_id="C_can_threat")],
        ),
        sg(
            "C_threat",
            action="sense",
            pre=[
                cond(
                    "can_threat",
                    pass_=False,
                    condition_id="C_threat_ready",
                )
            ],
            destroy=[cond("protected", condition_id="C_destroy")],
        ),
    ]
    proposals = {
        item.subgoal_id: [prop(f"cand-{item.subgoal_id}", item.subgoal_id, "A")]
        for item in subgoals
    }
    request = make_request(
        subgoals,
        edges=[("A_producer", "B_consumer")],
        proposals=proposals,
    )
    request.planner_a.constraints = PlannerAConstraints.model_validate(
        {
            "disjunctive_threats": [
                {
                    "key": ["A_producer", "B_consumer", "C_threat", "C_pre"],
                    "link": ["A_producer", "B_consumer"],
                    "threat": "C_threat",
                    "condition": "C_pre",
                    "options": ["promotion", "demotion"],
                }
            ]
        }
    )

    result = plan(request, suitability_scorer=None)

    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    assert result.selected_plan.subgoal_order == [
        "A_producer",
        "B_consumer",
        "C_threat",
    ]
    threat_trace = next(
        item
        for item in result.selected_plan.constraint_trace
        if item.constraint_type == "disjunctive_threat"
    )
    assert threat_trace.selected_option == "demotion"


def test_unknown_contract_reference_is_rejected_before_search() -> None:
    subgoal = sg(
        "target",
        action="sense",
        pre=[cond("ready", pass_=False, condition_id="C_ready")],
    )
    request = make_request(
        [subgoal], proposals={"target": [prop("cand", "target", "A")]}
    )
    request.planner_a.constraints = PlannerAConstraints.model_validate(
        {
            "open_conditions": [
                {
                    "subgoal": "target",
                    "condition": "C_ready",
                    "candidates": ["missing_producer"],
                }
            ]
        }
    )

    result = plan(request, suitability_scorer=None)

    assert result.status is PlanStatus.INVALID_INPUT
    assert result.rejections[0].reason_code is ReasonCode.UNKNOWN_CONSTRAINT_REFERENCE
