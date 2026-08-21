"""Input validation: cycles, unestablishable, unknown ids, schema errors."""

from __future__ import annotations

from task_planner.diagnostics import PlanStatus, ReasonCode
from task_planner.planner import plan

from conftest import cond, make_request, prop, sg


def _codes(result) -> set[ReasonCode]:
    return {r.reason_code for r in result.rejections}


def test_declared_cycle_skips_search() -> None:
    request = make_request(
        [sg("S1"), sg("S2")],
        edges=[("S1", "S2")],
        proposals={"S1": [], "S2": []},
        order_kwargs={"cycle_detected": True},
    )
    result = plan(request)
    assert result.status is PlanStatus.INFEASIBLE_REDECOMPOSE
    assert ReasonCode.CYCLE_DETECTED in _codes(result)
    assert result.search_stats.states_popped == 0
    assert result.selected_plan is None


def test_actual_edge_cycle_detected_even_if_flag_false() -> None:
    request = make_request(
        [sg("S1"), sg("S2")],
        edges=[("S1", "S2"), ("S2", "S1")],
        proposals={"S1": [], "S2": []},
        order_kwargs={"cycle_detected": False},
    )
    result = plan(request)
    assert result.status is PlanStatus.INFEASIBLE_REDECOMPOSE
    assert ReasonCode.CYCLE_DETECTED in _codes(result)
    assert result.search_stats.states_popped == 0


def test_unestablishable_requests_redecompose() -> None:
    request = make_request(
        [sg("S1")],
        proposals={"S1": []},
        order_kwargs={"unestablishable": [["holding", "obj9"]]},
    )
    result = plan(request)
    assert result.status is PlanStatus.INFEASIBLE_REDECOMPOSE
    assert ReasonCode.UNESTABLISHABLE_PRECONDITIONS in _codes(result)
    assert result.search_stats.states_popped == 0


def test_unknown_predecessor_id_rejected() -> None:
    request = make_request(
        [sg("S1")],
        edges=[("S0", "S1")],
        proposals={"S1": []},
    )
    result = plan(request)
    assert result.status is PlanStatus.INVALID_INPUT
    assert ReasonCode.UNKNOWN_PREDECESSOR in _codes(result)


def test_unique_solution_with_multiple_feasible_ee_is_invalid() -> None:
    request = make_request(
        [sg("S1", feasible=["A", "B"], unique=True)],
        proposals={"S1": [prop("S1-c1", "S1", "A")]},
    )
    result = plan(request)
    assert result.status is PlanStatus.INVALID_INPUT
    assert ReasonCode.UNIQUE_SOLUTION_VIOLATION in _codes(result)


def test_establish_destroy_conflict_is_invalid() -> None:
    request = make_request(
        [
            sg(
                "S1",
                targets=["obj1"],
                estab=[cond("holding", "obj1")],
                destroy=[cond("holding", "obj1")],
            )
        ],
        proposals={"S1": [prop("S1-c1", "S1", "A")]},
    )
    result = plan(request)
    assert result.status is PlanStatus.INVALID_INPUT
    assert ReasonCode.EFFECT_CONFLICT in _codes(result)


def test_duplicate_subgoal_ids_rejected() -> None:
    request = make_request(
        [sg("S1"), sg("S1")],
        proposals={"S1": [prop("S1-c1", "S1", "A")]},
    )
    result = plan(request)
    assert result.status is PlanStatus.INVALID_INPUT
    assert ReasonCode.DUPLICATE_SUBGOAL_ID in _codes(result)


def test_malformed_edge_rejected() -> None:
    request = make_request(
        [sg("S1")],
        proposals={"S1": []},
        order_kwargs={},
    )
    request.task_graph.order_constraints.edges = [{"weird": "shape"}]
    result = plan(request)
    assert result.status is PlanStatus.INVALID_INPUT
    assert ReasonCode.MALFORMED_EDGE in _codes(result)


def test_unknown_initial_ee_rejected() -> None:
    request = make_request(
        [sg("S1")],
        proposals={"S1": [prop("S1-c1", "S1", "A")]},
        initial_ee="ghost",
    )
    result = plan(request)
    assert result.status is PlanStatus.INVALID_INPUT
    assert ReasonCode.UNKNOWN_EE in _codes(result)


def test_empty_feasible_ee_is_no_candidate() -> None:
    request = make_request(
        [sg("S1", feasible=[])],
        proposals={"S1": []},
    )
    result = plan(request)
    assert result.status is PlanStatus.INFEASIBLE_NO_CANDIDATE
    assert ReasonCode.EMPTY_FEASIBLE_EE in _codes(result)


def test_duplicate_candidate_ids_are_invalid_cache_identities() -> None:
    request = make_request(
        [
            sg("S1", targets=["obj1"]),
            sg("S2", targets=["obj2"]),
        ],
        proposals={
            "S1": [prop("duplicate", "S1", "A")],
            "S2": [prop("duplicate", "S2", "A")],
        },
    )
    result = plan(request)
    assert result.status is PlanStatus.INVALID_INPUT
    assert ReasonCode.DUPLICATE_CANDIDATE_ID in _codes(result)
