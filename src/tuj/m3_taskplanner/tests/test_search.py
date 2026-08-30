"""Search behavior: interleaving without enumeration, lexicographic cost,
limits, and brute-force optimality on small random problems (test-only)."""

from __future__ import annotations

import random

import pytest

from tuj.m3_taskplanner.candidate_provider import StaticCandidateProvider, filter_candidates
from tuj.m3_taskplanner.conditions import (
    apply_effects,
    initial_facts,
    symbolic_precondition_unmet,
)
from tuj.m3_taskplanner.cost import CostVector
from tuj.m3_taskplanner.diagnostics import PlanStatus
from tuj.m3_taskplanner.feasibility import StaticFeasibilityChecker
from tuj.m3_taskplanner.models import TaskPlannerRequest, PlanningPolicy
from tuj.m3_taskplanner.planner import plan
from tuj.m3_taskplanner.state import SearchState, with_group_binding
from tuj.m3_taskplanner.transitions import (
    P,
    TransitionContext,
    build_terminal_transition,
    build_transition,
)
from tuj.m3_taskplanner.validation import build_predecessor_map, normalized_edges

from conftest import cond, make_request, prop, sg


# --------------------------------------------------------------------------- #
# CostVector semantics                                                        #
# --------------------------------------------------------------------------- #


def test_cost_vector_is_lexicographic_not_weighted() -> None:
    plan_a = CostVector(1, 5, 100, 0)
    plan_b = CostVector(2, 0, 0, 0)
    assert plan_a < plan_b  # Plan A must win despite huge later components.


def test_cost_vector_addition_and_nonnegativity() -> None:
    total = CostVector(1, 0, 2, 0) + CostVector(0, 1, 3, 4)
    assert total == CostVector(1, 1, 5, 4)
    with pytest.raises(ValueError):
        CostVector(ee_switches=-1)


# --------------------------------------------------------------------------- #
# Diamond DAG: interleavings are represented in the state space, not listed   #
# --------------------------------------------------------------------------- #


def _diamond_request() -> TaskPlannerRequest:
    subgoals = [
        sg("S1", targets=["obj1"], feasible=["A"]),
        sg("S2", targets=["obj2"], feasible=["B"]),
        sg("S3", targets=["obj3"], feasible=["A"]),
        sg("S4", targets=["obj4"], feasible=["A"]),
    ]
    proposals = {
        "S1": [prop("S1-c1", "S1", "A")],
        "S2": [prop("S2-c1", "S2", "B")],
        "S3": [prop("S3-c1", "S3", "A")],
        "S4": [prop("S4-c1", "S4", "A")],
    }
    return make_request(
        subgoals,
        edges=[("S1", "S3"), ("S2", "S3"), ("S3", "S4")],
        proposals=proposals,
        initial_ee="B",
    )


def test_diamond_picks_cheaper_interleaving_without_enumeration() -> None:
    result = plan(_diamond_request())
    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    # Starting on B, doing S2 first needs only one EE switch; S1-first needs 2.
    assert result.selected_plan.subgoal_order == ["S2", "S1", "S3", "S4"]
    assert result.selected_plan.cost_vector.ee_switches == 1
    # The search explored a polynomial-sized frontier, not 2 pre-listed orders
    # x candidate assignments; sanity-bound the expansion count.
    assert result.search_stats.states_expanded < 30


def test_search_limit_returns_limit_status_without_fabricating_a_plan() -> None:
    request = _diamond_request()
    request.planning_policy = PlanningPolicy(max_expansions=1)
    result = plan(request)
    assert result.status is PlanStatus.SEARCH_LIMIT_REACHED
    assert result.selected_plan is None


# --------------------------------------------------------------------------- #
# Lexicographic priority in an actual plan choice                             #
# --------------------------------------------------------------------------- #


def test_fixed_tool_continuity_preferred_over_extra_ee_switch() -> None:
    subgoals = [
        sg("S1", tool_id="t1", feasible=["A"], action="operate"),
        sg(
            "S2",
            targets=["obj1"],
            tool_id="t1",
            feasible=["A", "B"],
            action="operate",
        ),
    ]
    proposals = {
        "S1": [prop("S1-c1", "S1", "A", tool="t1")],
        "S2": [
            prop("S2-cx", "S2", "A", tool="t1"),
            prop("S2-cy", "S2", "B", tool="t1"),
        ],
    }
    result = plan(
        make_request(
            subgoals,
            edges=[("S1", "S2")],
            proposals=proposals,
            initial_ee="A",
        )
    )
    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    cv = result.selected_plan.cost_vector
    assert (cv.ee_switches, cv.tool_switches, cv.motion_cost) == (0, 2, 6)
    step_actions = [s.action for s in result.selected_plan.steps]
    assert P.KEEP_TOOL in step_actions
    assert P.TERMINAL_RETURN_TOOL in step_actions


# --------------------------------------------------------------------------- #
# Brute-force optimality oracle (test-only; production never enumerates)      #
# --------------------------------------------------------------------------- #


def _brute_force_best_cost(request: TaskPlannerRequest) -> CostVector | None:
    task_graph = request.task_graph
    subgoals = {s.subgoal_id: s for s in task_graph.subgoals}
    preds = build_predecessor_map(
        list(subgoals), normalized_edges(task_graph.order_constraints)
    )
    provider = StaticCandidateProvider(
        request.candidate_proposals or {}, request.resource_catalog
    )
    checker = StaticFeasibilityChecker(
        request.resource_catalog, request.planning_policy, {}
    )
    candidates = {}
    for sid, subgoal in subgoals.items():
        raw, _ = provider.candidates_for(subgoal)
        kept, _ = filter_candidates(subgoal, raw, checker, request.planning_policy)
        candidates[sid] = kept
    context = TransitionContext(
        catalog=request.resource_catalog,
        policy=request.planning_policy,
        initial_ee=task_graph.initial_state.current_ee,
    )
    initial = SearchState(
        completed_subgoals=frozenset(),
        current_ee=task_graph.initial_state.current_ee,
        held_tool=None,
        group_ee_bindings=(),
        symbolic_facts=initial_facts(task_graph.initial_state),
        scene_signature="brute",
    )
    best: list[CostVector | None] = [None]

    def rec(state: SearchState, cost: CostVector) -> None:
        if state.completed_subgoals == frozenset(subgoals):
            terminal = build_terminal_transition(state, context)
            if terminal.feasible:
                total = cost + terminal.cost
                if best[0] is None or total.as_tuple() < best[0].as_tuple():
                    best[0] = total
            return
        bindings = dict(state.group_ee_bindings)
        for sid in sorted(subgoals):
            if sid in state.completed_subgoals:
                continue
            if not preds.get(sid, frozenset()) <= state.completed_subgoals:
                continue
            subgoal = subgoals[sid]
            if symbolic_precondition_unmet(subgoal, state.symbolic_facts):
                continue
            for cand in candidates[sid]:
                if subgoal.group_id is not None and bindings.get(
                    subgoal.group_id
                ) not in (None, cand.ee):
                    continue
                transition = build_transition(state, cand, context)
                if not transition.feasible:
                    continue
                next_facts = apply_effects(
                    state.symbolic_facts, subgoal.destroy, subgoal.establish
                )
                next_group = state.group_ee_bindings
                if subgoal.group_id is not None and subgoal.group_id not in bindings:
                    next_group = with_group_binding(
                        next_group, subgoal.group_id, cand.ee
                    )
                rec(
                    SearchState(
                        completed_subgoals=state.completed_subgoals | {sid},
                        current_ee=cand.ee,
                        held_tool=transition.next_held_tool,
                        group_ee_bindings=next_group,
                        symbolic_facts=next_facts,
                        scene_signature="brute",
                    ),
                    cost + transition.cost,
                )

    rec(initial, CostVector())
    return best[0]


def _random_request(rng: random.Random) -> TaskPlannerRequest:
    n = rng.randint(3, 5)
    subgoals = []
    proposals = {}
    edges = []
    for i in range(1, n + 1):
        sid = f"S{i}"
        feasible = rng.choice([["A"], ["B"], ["A", "B"]])
        subgoals.append(
            sg(
                sid,
                targets=[f"obj{i}"],
                feasible=feasible,
                estab=[cond("done", sid)],
            )
        )
        proposals[sid] = [
            prop(f"{sid}-c{ee}", sid, ee)
            for ee in feasible
        ]
        for j in range(1, i):
            if rng.random() < 0.4:
                edges.append((f"S{j}", sid))
    return make_request(
        subgoals,
        edges=edges,
        proposals=proposals,
        initial_ee=rng.choice(["A", "B"]),
    )


@pytest.mark.parametrize("seed", range(15))
def test_dijkstra_matches_brute_force_on_small_random_problems(seed: int) -> None:
    request = _random_request(random.Random(seed))
    result = plan(request)
    expected = _brute_force_best_cost(request)
    assert expected is not None
    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    got = result.selected_plan.cost_vector
    assert (
        got.ee_switches,
        got.tool_switches,
        got.motion_cost,
        got.execution_cost,
    ) == expected.as_tuple()
