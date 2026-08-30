"""Terminal policy: tool return, EE restore, and their real costs."""

from __future__ import annotations

from tuj.m4_taskplanner.diagnostics import PlanStatus
from tuj.m4_taskplanner.models import PlanningPolicy, TerminalPolicy
from tuj.m4_taskplanner.planner import plan
from tuj.m4_taskplanner.transitions import P

from conftest import cond, make_request, prop, sg


def _tool_request(policy: PlanningPolicy | None = None, feasible=("A",), ee="A"):
    subgoals = [sg("S1", tool_id="t1", feasible=list(feasible), action="operate")]
    proposals = {"S1": [prop("S1-c1", "S1", ee, tool="t1")]}
    return make_request(
        subgoals, proposals=proposals, initial_ee="A", policy=policy
    )


def test_tool_returned_at_end_by_default_with_cost() -> None:
    result = plan(_tool_request())
    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    actions = [s.action for s in result.selected_plan.steps]
    assert actions[-1] == P.TERMINAL_RETURN_TOOL
    # pick (1) + terminal return (1) tool switches; terminal motion included.
    assert result.selected_plan.cost_vector.tool_switches == 2
    assert result.selected_plan.cost_vector.motion_cost == 2 + 1 + 2
    assert result.selected_plan.terminal_state["held_tool"] is None


def test_tool_kept_when_return_disabled() -> None:
    policy = PlanningPolicy(terminal=TerminalPolicy(return_tool_at_end=False))
    result = plan(_tool_request(policy))
    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    actions = [s.action for s in result.selected_plan.steps]
    assert P.TERMINAL_RETURN_TOOL not in actions
    assert result.selected_plan.cost_vector.tool_switches == 1
    assert result.selected_plan.terminal_state["held_tool"] == "t1"


def test_restore_initial_ee_adds_switch_and_step() -> None:
    policy = PlanningPolicy(
        terminal=TerminalPolicy(restore_initial_ee_at_end=True)
    )
    subgoals = [sg("S1", targets=["obj1"], feasible=["B"])]
    proposals = {"S1": [prop("S1-c1", "S1", "B")]}
    result = plan(
        make_request(subgoals, proposals=proposals, initial_ee="A", policy=policy)
    )
    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    actions = [s.action for s in result.selected_plan.steps]
    assert actions[-1] == P.TERMINAL_RESTORE_EE
    # A -> B for the subgoal, then B -> A restore: both count.
    assert result.selected_plan.cost_vector.ee_switches == 2
    assert result.selected_plan.terminal_state["current_ee"] == "A"


def test_restore_policy_has_no_target_when_initial_ee_is_absent() -> None:
    policy = PlanningPolicy(
        terminal=TerminalPolicy(restore_initial_ee_at_end=True)
    )
    subgoals = [sg("S1", targets=["obj1"], feasible=["B"])]
    proposals = {"S1": [prop("S1-c1", "S1", "B")]}

    result = plan(
        make_request(
            subgoals,
            proposals=proposals,
            initial_ee=None,
            policy=policy,
        )
    )

    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    actions = [step.action for step in result.selected_plan.steps]
    assert P.TERMINAL_RESTORE_EE not in actions
    assert result.selected_plan.cost_vector.ee_switches == 0
    assert result.selected_plan.terminal_state["current_ee"] == "B"


def test_terminal_cleanup_cost_is_not_hidden() -> None:
    with_return = plan(_tool_request())
    policy = PlanningPolicy(terminal=TerminalPolicy(return_tool_at_end=False))
    without_return = plan(_tool_request(policy))
    assert with_return.selected_plan is not None
    assert without_return.selected_plan is not None
    a = with_return.selected_plan.cost_vector
    b = without_return.selected_plan.cost_vector
    assert (a.tool_switches, a.motion_cost) > (b.tool_switches, b.motion_cost)


def test_restore_initial_ee_never_detaches_while_object_is_held() -> None:
    policy = PlanningPolicy(
        terminal=TerminalPolicy(
            return_tool_at_end=False,
            restore_initial_ee_at_end=True,
            require_empty_object_hand_at_end=False,
        )
    )
    subgoals = [
        sg(
            "S1",
            targets=["obj1"],
            feasible=["B"],
            estab=[cond("holding", "obj1")],
        )
    ]
    proposals = {"S1": [prop("S1-c1", "S1", "B")]}
    result = plan(
        make_request(subgoals, proposals=proposals, initial_ee="A", policy=policy)
    )
    assert result.status is PlanStatus.INFEASIBLE_NO_PLAN
    assert result.selected_plan is None


def test_terminal_does_not_return_tool_while_tool_holds_object() -> None:
    policy = PlanningPolicy(
        terminal=TerminalPolicy(require_empty_object_hand_at_end=False)
    )
    subgoals = [
        sg(
            "S1",
            targets=["obj1"],
            tool_id="t1",
            feasible=["A"],
            estab=[cond("holding", "obj1")],
            action="operate",
        )
    ]
    proposals = {
        "S1": [
            prop(
                "S1-c1",
                "S1",
                "A",
                tool="t1",
            )
        ]
    }
    result = plan(make_request(subgoals, proposals=proposals, policy=policy))
    assert result.status is PlanStatus.INFEASIBLE_NO_PLAN
    assert result.selected_plan is None
