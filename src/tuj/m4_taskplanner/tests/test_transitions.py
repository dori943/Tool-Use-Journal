"""Transition rules for fixed tools and EE changes."""

from __future__ import annotations

from tuj.m4_taskplanner.candidate_provider import Candidate
from tuj.m4_taskplanner.diagnostics import ReasonCode
from tuj.m4_taskplanner.models import PlanningPolicy
from tuj.m4_taskplanner.state import SearchState
from tuj.m4_taskplanner.transitions import P, TransitionContext, build_transition

from conftest import base_catalog


def make_state(
    ee: str | None = "A",
    tool: str | None = None,
    *,
    held_object: str | None = None,
) -> SearchState:
    facts = (
        frozenset({("holding", (held_object,))})
        if held_object is not None
        else frozenset()
    )
    return SearchState(
        completed_subgoals=frozenset(),
        current_ee=ee,
        held_tool=tool,
        group_ee_bindings=(),
        symbolic_facts=facts,
        scene_signature="initial",
    )


def make_candidate(
    ee: str = "A",
    tool: str | None = None,
    *,
    action_type: str | None = None,
) -> Candidate:
    return Candidate(
        candidate_id=f"c-{ee}-{tool}",
        subgoal_id="S1",
        ee=ee,
        tool=tool,
        action_type=action_type,
    )


def ctx(initial_ee: str | None = "A") -> TransitionContext:
    return TransitionContext(
        catalog=base_catalog(),
        policy=PlanningPolicy(),
        initial_ee=initial_ee,
    )


def actions(result) -> list[str]:
    return [primitive.action for primitive in result.primitives]


def test_same_ee_and_tool_keeps_resources() -> None:
    result = build_transition(make_state(tool="t1"), make_candidate(tool="t1"), ctx())
    assert result.feasible
    assert actions(result) == [
        P.KEEP_EE,
        P.KEEP_TOOL,
        P.MOVE_TO_WORKSPACE,
        P.EXECUTE_SUBGOAL,
    ]
    assert result.cost.ee_switches == 0
    assert result.cost.tool_switches == 0


def test_different_tool_returns_then_picks() -> None:
    result = build_transition(
        make_state(tool="t1"), make_candidate(tool="t2"), ctx()
    )
    assert result.feasible
    acts = actions(result)
    assert acts.index(P.RETURN_TOOL) < acts.index(P.PICK_TOOL)
    assert result.cost.tool_switches == 1


def test_ee_change_with_tool_returns_tool_before_detach() -> None:
    result = build_transition(
        make_state(ee="A", tool="t1"),
        make_candidate(ee="B", tool="t2"),
        ctx(),
    )
    assert result.feasible
    acts = actions(result)
    assert acts.index(P.RETURN_TOOL) < acts.index(P.DETACH_EE)
    assert acts.index(P.DETACH_EE) < acts.index(P.ATTACH_EE)
    assert acts.index(P.ATTACH_EE) < acts.index(P.PICK_TOOL)
    assert P.VERIFY_ATTACHMENT in acts
    assert result.cost.ee_switches == 1
    assert result.cost.tool_switches == 1
    assert result.next_held_tool == "t2"


def test_ee_change_with_same_tool_does_not_count_identity_change() -> None:
    result = build_transition(
        make_state(ee="A", tool="t1"),
        make_candidate(ee="B", tool="t1"),
        ctx(),
    )
    assert result.feasible
    assert result.cost.ee_switches == 1
    assert result.cost.tool_switches == 0


def test_ee_change_empty_hand() -> None:
    result = build_transition(make_state(ee="A"), make_candidate(ee="B"), ctx())
    assert result.feasible
    assert actions(result) == [
        P.MOVE_TO_EE_RACK,
        P.DETACH_EE,
        P.ATTACH_EE,
        P.VERIFY_ATTACHMENT,
        P.MOVE_TO_WORKSPACE,
        P.EXECUTE_SUBGOAL,
    ]


def test_initial_ee_attach_has_no_detach_or_exchange_cost() -> None:
    state = make_state(ee=None)
    result = build_transition(state, make_candidate(ee="B"), ctx(initial_ee=None))

    assert result.feasible
    assert actions(result) == [
        P.MOVE_TO_EE_RACK,
        P.ATTACH_EE,
        P.VERIFY_ATTACHMENT,
        P.MOVE_TO_WORKSPACE,
        P.EXECUTE_SUBGOAL,
    ]
    assert result.cost.ee_switches == 0


def test_initial_ee_attach_updates_explicit_rack_occupancy() -> None:
    state = make_state(ee=None)
    state = SearchState(
        completed_subgoals=state.completed_subgoals,
        current_ee=state.current_ee,
        held_tool=state.held_tool,
        group_ee_bindings=state.group_ee_bindings,
        symbolic_facts=state.symbolic_facts,
        scene_signature=state.scene_signature,
        rack_signature=(("SA", "A"), ("SB", "B")),
    )
    result = build_transition(state, make_candidate(ee="B"), ctx(initial_ee=None))

    assert result.feasible
    assert result.next_rack_signature == (("SA", "A"), ("SB", "empty"))


def test_object_held_ee_change_infeasible() -> None:
    result = build_transition(
        make_state(held_object="obj1"), make_candidate(ee="B"), ctx()
    )
    assert not result.feasible
    assert result.rejection is not None
    assert result.rejection.reason_code is ReasonCode.OBJECT_HELD_EE_SWITCH


def test_tool_none_with_held_tool_returns_tool_first() -> None:
    result = build_transition(make_state(tool="t1"), make_candidate(), ctx())
    assert result.feasible
    acts = actions(result)
    assert acts.index(P.RETURN_TOOL) < acts.index(P.MOVE_TO_WORKSPACE)
    assert P.PICK_TOOL not in acts
    assert result.cost.tool_switches == 1
    assert result.next_held_tool is None


def test_object_held_tool_acquisition_infeasible() -> None:
    result = build_transition(
        make_state(held_object="obj1"), make_candidate(tool="t1"), ctx()
    )
    assert not result.feasible
    assert result.rejection is not None
    assert result.rejection.reason_code is ReasonCode.OBJECT_HELD_TOOL_ACQUIRE


def test_explicit_pick_tool_executes_once_without_automatic_pick_transition() -> None:
    result = build_transition(
        make_state(tool=None),
        make_candidate(tool="t1", action_type="PICK_TOOL"),
        ctx(),
    )

    assert result.feasible
    assert actions(result) == [
        P.KEEP_EE,
        P.MOVE_TO_TOOL_RACK,
        P.EXECUTE_SUBGOAL,
    ]
    assert result.next_held_tool == "t1"
    assert result.cost.tool_switches == 1


def test_explicit_return_tool_executes_once_without_automatic_return_transition() -> None:
    result = build_transition(
        make_state(tool="t1"),
        make_candidate(tool="t1", action_type="RETURN_TOOL"),
        ctx(),
    )

    assert result.feasible
    assert actions(result) == [
        P.KEEP_EE,
        P.KEEP_TOOL,
        P.MOVE_TO_TOOL_RACK,
        P.EXECUTE_SUBGOAL,
    ]
    assert result.next_held_tool is None
    assert result.cost.tool_switches == 1


def test_explicit_return_rejects_a_tool_that_is_not_held() -> None:
    result = build_transition(
        make_state(tool=None),
        make_candidate(tool="t1", action_type="RETURN_TOOL"),
        ctx(),
    )

    assert not result.feasible
    assert result.rejection is not None
    assert result.rejection.reason_code is ReasonCode.TOOL_MISMATCH
