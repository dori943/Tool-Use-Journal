"""Transition rules for fixed tools and EE changes."""

from __future__ import annotations

from planning_b.candidate_provider import Candidate
from planning_b.diagnostics import ReasonCode
from planning_b.models import PlanningPolicy
from planning_b.state import SearchState
from planning_b.transitions import P, TransitionContext, build_transition

from conftest import base_catalog


def make_state(
    ee: str = "A",
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


def make_candidate(ee: str = "A", tool: str | None = None) -> Candidate:
    return Candidate(
        candidate_id=f"c-{ee}-{tool}",
        subgoal_id="S1",
        ee=ee,
        tool=tool,
    )


def ctx(initial_ee: str = "A") -> TransitionContext:
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
