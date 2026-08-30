"""Group EE binding: same EE across the group, empty intersection pre-fail."""

from __future__ import annotations

from tuj.m3_taskplanner.diagnostics import PlanStatus, ReasonCode
from tuj.m3_taskplanner.planner import plan

from conftest import make_request, prop, sg


def test_group_uses_single_ee_even_when_cheaper_alternative_exists() -> None:
    # S2 (unordered vs S1) has candidates on A and B. S1 (same group) only has
    # B. The group binding must force S2 onto B as well.
    subgoals = [
        sg("S1", group="G", targets=["obj1"], feasible=["B"]),
        sg("S2", group="G", targets=["obj2"], feasible=["A", "B"]),
    ]
    proposals = {
        "S1": [prop("S1-cB", "S1", "B")],
        "S2": [
            prop("S2-cA", "S2", "A"),
            prop("S2-cB", "S2", "B"),
        ],
    }
    result = plan(
        make_request(subgoals, proposals=proposals, initial_ee="A")
    )
    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    assert result.selected_plan.group_ee_assignments == {"G": "B"}
    assert {a.ee for a in result.selected_plan.candidate_assignments} == {"B"}
    assert result.selected_plan.cost_vector.ee_switches == 1


def test_group_binding_fixed_by_first_executed_subgoal() -> None:
    # Both S1 and S2 could run on A or B; whatever the first one uses binds
    # the group, so the final assignment must be uniform.
    subgoals = [
        sg("S1", group="G", targets=["obj1"], feasible=["A", "B"]),
        sg("S2", group="G", targets=["obj2"], feasible=["A", "B"]),
    ]
    proposals = {
        "S1": [
            prop("S1-cA", "S1", "A"),
            prop("S1-cB", "S1", "B"),
        ],
        "S2": [
            prop("S2-cA", "S2", "A"),
            prop("S2-cB", "S2", "B"),
        ],
    }
    result = plan(make_request(subgoals, proposals=proposals, initial_ee="A"))
    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    ees = {a.ee for a in result.selected_plan.candidate_assignments}
    assert ees == {"A"}  # staying on the initial EE costs zero switches
    assert result.selected_plan.cost_vector.ee_switches == 0


def test_empty_group_intersection_fails_before_search() -> None:
    subgoals = [
        sg("S1", group="G", targets=["obj1"], feasible=["A"]),
        sg("S2", group="G", targets=["obj2"], feasible=["B"]),
    ]
    proposals = {
        "S1": [prop("S1-cA", "S1", "A")],
        "S2": [prop("S2-cB", "S2", "B")],
    }
    result = plan(make_request(subgoals, proposals=proposals))
    assert result.status is PlanStatus.INFEASIBLE_NO_CANDIDATE
    assert result.diagnostics.groups_without_common_ee == ["G"]
    assert any(
        r.reason_code is ReasonCode.GROUP_NO_COMMON_EE for r in result.rejections
    )
    assert result.search_stats.states_popped == 0
    assert result.selected_plan is None
