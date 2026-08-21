"""Dijkstra over order, EE, and fixed-tool transition state.

No topological orders are enumerated and no complete plans are materialized
up front. Alternative interleavings coexist as alternative paths and the
lexicographically cheapest one wins. Priority = (accumulated cost vector,
deterministic tie-breaker, insertion sequence).
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

from task_planner.candidate_provider import Candidate
from task_planner.conditions import (
    CheckerRegistry,
    apply_effects,
    symbolic_precondition_unmet,
)
from task_planner.constraints import PlannerAConstraintEngine
from task_planner.cost import CostVector
from task_planner.diagnostics import (
    ReasonCode,
    Rejection,
    fluent_str,
    make_rejection,
)
from task_planner.feasibility import FeasibilityStatus, GeometryCache
from task_planner.models import PlanningPolicy, Subgoal
from task_planner.scene import SceneState, SceneStateUpdater
from task_planner.state import SearchState, with_group_binding
from task_planner.transitions import (
    Primitive,
    TransitionContext,
    build_terminal_transition,
    build_transition,
    held_object_ids,
    transition_signature,
)


@dataclass(frozen=True, slots=True)
class ParentEdge:
    prev_state: SearchState
    subgoal_id: str | None  # None for the terminal edge
    candidate: Candidate | None
    primitives: tuple[Primitive, ...]
    edge_cost: CostVector


@dataclass(slots=True)
class SearchStats:
    states_popped: int = 0
    states_expanded: int = 0
    edges_generated: int = 0
    dominated_states_pruned: int = 0
    static_candidates_rejected: int = 0
    lazy_checks_executed: int = 0
    lazy_cache_hits: int = 0
    elapsed_ms: int = 0


@dataclass(slots=True)
class SearchProblem:
    subgoals: dict[str, Subgoal]
    candidates: dict[str, list[Candidate]]
    predecessors: dict[str, frozenset[str]]
    initial_state: SearchState
    policy: PlanningPolicy
    transition_context: TransitionContext
    geometry: GeometryCache
    scene_updater: SceneStateUpdater
    checker_registry: CheckerRegistry
    constraint_engine: PlannerAConstraintEngine
    banned_scene_candidates: frozenset[tuple[str, str]] = frozenset()
    banned_transitions: frozenset[tuple[str, ...]] = frozenset()


@dataclass(slots=True)
class SearchOutcome:
    goal_state: SearchState | None
    best_cost: dict[SearchState, CostVector]
    parent: dict[SearchState, ParentEdge]
    stats: SearchStats
    rejections: list[Rejection] = field(default_factory=list)
    limit_reached: bool = False
    frontier_exhausted: bool = False
    attempted_subgoals: set[str] = field(default_factory=set)
    unmet_preconditions: set[tuple[str, str]] = field(default_factory=set)
    constraint_blockers: set[tuple[str, str, str, str]] = field(
        default_factory=set
    )


def _is_goal(state: SearchState, problem: SearchProblem) -> bool:
    if state.completed_subgoals != frozenset(problem.subgoals):
        return False
    terminal = problem.policy.terminal
    if terminal.return_tool_at_end and state.held_tool is not None:
        return False
    if (
        terminal.require_empty_object_hand_at_end
        and held_object_ids(state, problem.transition_context)
    ):
        return False
    if (
        terminal.restore_initial_ee_at_end
        and state.current_ee != problem.transition_context.initial_ee
    ):
        return False
    return True


def _scene_ref(state: SearchState):
    """Reconstruction recipe handed to a motion-planner oracle."""
    from task_planner.motion_interface import SceneRef

    return SceneRef(
        signature=state.scene_signature,
        completed_subgoals=tuple(sorted(state.completed_subgoals)),
        facts=tuple(sorted(fluent_str(key) for key in state.symbolic_facts)),
    )


def _geometry_ok(
    problem: SearchProblem,
    state: SearchState,
    candidate: Candidate,
    rejections: dict[tuple[str, ...], Rejection],
    primitives: tuple[str, ...] = (),
) -> bool:
    """Geometric validation the *search* is allowed to run.

    Scope follows ``PlanningPolicy.in_search_geometry``. Under the default
    ``scene_independent`` only the EE exchange is queried: its answer does not
    depend on where movable objects are, so it needs no predicted-scene
    reconstruction and caches perfectly. Scene-dependent checks are deferred to
    post-search verification rather than answered badly here.
    """
    mode = problem.policy.in_search_geometry
    if mode == "none":
        return True
    held_objects = held_object_ids(state, problem.transition_context)
    held_object = next(iter(sorted(held_objects)), None)
    scene_ref = _scene_ref(state) if problem.geometry.is_motion_oracle else None
    checks = []
    if mode == "full":
        checks += [
            (
                "candidate",
                lambda: problem.geometry.check_candidate(
                    state.scene_signature, candidate, scene_ref=scene_ref
                ),
            ),
            (
                "transition",
                lambda: problem.geometry.check_transition(
                    state.scene_signature,
                    state.current_ee,
                    state.held_tool,
                    candidate,
                    scene_ref=scene_ref,
                    primitives=primitives,
                    held_object_id=held_object,
                ),
            ),
        ]
    # An EE exchange is a distinct physical action (dock / undock / re-couple)
    # that can fail on its own, so it gets its own query rather than riding
    # along inside the transition check.
    if state.current_ee != candidate.ee:
        checks.append(
            (
                "transition",
                lambda: problem.geometry.check_ee_exchange(
                    state.scene_signature,
                    state.current_ee,
                    candidate.ee,
                    held_tool=state.held_tool,
                    rack_occupancy=state.rack_signature,
                    scene_ref=scene_ref,
                ),
            )
        )
    for scope, run_check in checks:
        # Deliberately invoke these one at a time. A failed candidate pose
        # makes the transition irrelevant and must not trigger another lazy
        # checker call.
        result = run_check()
        if result.status is FeasibilityStatus.PASS:
            continue
        if (
            result.status is FeasibilityStatus.UNKNOWN
            and problem.policy.unknown_feasibility_policy != "reject"
        ):
            continue
        code = result.reason_code or (
            ReasonCode.GEOMETRY_UNKNOWN_REJECTED
            if result.status is FeasibilityStatus.UNKNOWN
            else ReasonCode.GEOMETRY_FAILED
        )
        key = (
            scope,
            state.scene_signature,
            state.current_ee if scope == "transition" else "",
            (state.held_tool or "") if scope == "transition" else "",
            candidate.candidate_id,
            code.value,
        )
        if key not in rejections:
            rejections[key] = make_rejection(
                "candidate" if scope == "candidate" else "transition",
                code,
                result.message or f"lazy geometric {scope} check failed",
                subgoal_id=candidate.subgoal_id,
                candidate_id=candidate.candidate_id,
            )
        return False
    return True


def _terminal_geometry_ok(
    problem: SearchProblem,
    state: SearchState,
    terminal,
    rejections: dict[tuple[str, ...], Rejection],
) -> bool:
    """Lazy geometric validation of the terminal cleanup edge.

    Returning the tool and re-docking the initial EE are real motions that can
    fail; a symbolically feasible terminal edge is not automatically realizable.

    Under ``scene_independent`` only the EE restore half is queried, because it
    is a rack exchange; the tool return depends on the scene and is deferred.
    """
    mode = problem.policy.in_search_geometry
    if mode == "none":
        return True
    if not problem.geometry.is_motion_oracle or not terminal.primitives:
        return True
    restore_ee = (
        terminal.next_ee if terminal.next_ee != state.current_ee else None
    )
    if mode == "scene_independent" and restore_ee is None:
        return True
    result = problem.geometry.check_terminal(
        state.scene_signature,
        state.current_ee,
        held_tool=state.held_tool,
        restore_ee=restore_ee,
        return_tool=state.held_tool,
        rack_occupancy=terminal.next_rack_signature,
        scene_ref=_scene_ref(state),
    )
    if result.status is FeasibilityStatus.PASS:
        return True
    if (
        result.status is FeasibilityStatus.UNKNOWN
        and problem.policy.unknown_feasibility_policy != "reject"
    ):
        return True
    code = result.reason_code or (
        ReasonCode.GEOMETRY_UNKNOWN_REJECTED
        if result.status is FeasibilityStatus.UNKNOWN
        else ReasonCode.GEOMETRY_FAILED
    )
    key = ("terminalGeometry", state.scene_signature, state.current_ee, code.value)
    rejections.setdefault(
        key,
        make_rejection(
            "terminal",
            code,
            result.message or "lazy geometric terminal check failed",
        ),
    )
    return False


def run_search(problem: SearchProblem) -> SearchOutcome:
    start_time = time.monotonic()
    stats = SearchStats()
    outcome = SearchOutcome(
        goal_state=None, best_cost={}, parent={}, stats=stats
    )
    dynamic_rejections: dict[tuple[str, ...], Rejection] = {}
    all_subgoal_ids = sorted(problem.subgoals)

    heap: list[
        tuple[tuple[int, ...], tuple[str, ...], int, SearchState]
    ] = []
    seq = 0
    initial = problem.initial_state
    outcome.best_cost[initial] = CostVector.zero()
    heapq.heappush(heap, (CostVector.zero().as_tuple(), ("", "", ""), seq, initial))

    while heap:
        if (
            problem.policy.timeout_seconds is not None
            and time.monotonic() - start_time > problem.policy.timeout_seconds
        ):
            outcome.limit_reached = True
            break

        cost_tuple, _tie, _seq, state = heapq.heappop(heap)
        stats.states_popped += 1
        best = outcome.best_cost.get(state)
        if best is None or cost_tuple > best.as_tuple():
            stats.dominated_states_pruned += 1
            continue
        cost = best

        if _is_goal(state, problem):
            outcome.goal_state = state
            break

        # max_expansions limits actual node expansions, not heap pops. Stale
        # queue entries therefore do not consume the user's search budget, and
        # an already-reached goal can still be accepted at the limit.
        if stats.states_expanded >= problem.policy.max_expansions:
            outcome.limit_reached = True
            break

        stats.states_expanded += 1
        bindings = dict(state.group_ee_bindings)

        # Terminal cleanup edge once every subgoal is complete.
        if state.completed_subgoals == frozenset(problem.subgoals):
            terminal = build_terminal_transition(
                state, problem.transition_context
            )
            if terminal.feasible and not _terminal_geometry_ok(
                problem, state, terminal, dynamic_rejections
            ):
                continue
            if terminal.feasible:
                next_state = SearchState(
                    completed_subgoals=state.completed_subgoals,
                    current_ee=terminal.next_ee,
                    held_tool=None if terminal.primitives else state.held_tool,
                    group_ee_bindings=state.group_ee_bindings,
                    symbolic_facts=state.symbolic_facts,
                    scene_signature=state.scene_signature,
                    rack_signature=terminal.next_rack_signature,
                )
                seq = _relax(
                    problem,
                    outcome,
                    heap,
                    stats,
                    state,
                    next_state,
                    cost,
                    terminal.cost,
                    ParentEdge(
                        prev_state=state,
                        subgoal_id=None,
                        candidate=None,
                        primitives=terminal.primitives,
                        edge_cost=terminal.cost,
                    ),
                    tie=("~terminal", "", ""),
                    seq=seq,
                )
            else:
                if terminal.rejection is not None:
                    key = ("terminal", state.scene_signature, "", "")
                    dynamic_rejections.setdefault(key, terminal.rejection)
            continue

        for sg_id in all_subgoal_ids:
            if sg_id in state.completed_subgoals:
                continue
            if not problem.predecessors.get(sg_id, frozenset()) <= (
                state.completed_subgoals
            ):
                continue
            blockers = problem.constraint_engine.blockers(
                sg_id, state.completed_subgoals, state.symbolic_facts
            )
            if blockers:
                for blocker in blockers:
                    outcome.constraint_blockers.add(
                        (
                            sg_id,
                            blocker.kind,
                            blocker.constraint_id,
                            blocker.message,
                        )
                    )
                continue
            subgoal = problem.subgoals[sg_id]
            unmet = symbolic_precondition_unmet(
                subgoal, state.symbolic_facts, problem.checker_registry
            )
            if unmet:
                for key in unmet:
                    outcome.unmet_preconditions.add((sg_id, fluent_str(key)))
                continue

            for candidate in problem.candidates.get(sg_id, []):
                # Group EE binding: first executed candidate binds the group;
                # later candidates must match.
                bound_ee = (
                    bindings.get(subgoal.group_id)
                    if subgoal.group_id is not None
                    else None
                )
                if bound_ee is not None and candidate.ee != bound_ee:
                    continue

                if (
                    state.scene_signature,
                    candidate.candidate_id,
                ) in problem.banned_scene_candidates:
                    key = (
                        "noGood",
                        state.scene_signature,
                        candidate.candidate_id,
                        "",
                    )
                    if key not in dynamic_rejections:
                        dynamic_rejections[key] = make_rejection(
                            "candidate",
                            ReasonCode.CANDIDATE_BANNED,
                            "banned for this scene by replanning no-good set",
                            subgoal_id=sg_id,
                            candidate_id=candidate.candidate_id,
                        )
                    continue

                sig = transition_signature(state, candidate)
                if (
                    sig in problem.banned_transitions
                    or (state.scene_signature,) + sig
                    in problem.banned_transitions
                ):
                    key = ("bannedTransition",) + sig
                    if key not in dynamic_rejections:
                        dynamic_rejections[key] = make_rejection(
                            "transition",
                            ReasonCode.TRANSITION_BANNED,
                            f"transition {sig!r} banned by replanning "
                            "no-good set",
                            subgoal_id=sg_id,
                            candidate_id=candidate.candidate_id,
                        )
                    continue

                transition = build_transition(
                    state, candidate, problem.transition_context, subgoal
                )
                if not transition.feasible:
                    if transition.rejection is not None:
                        key = (
                            "transitionInfeasible",
                            state.scene_signature,
                            candidate.candidate_id,
                            transition.rejection.reason_code.value,
                        )
                        dynamic_rejections.setdefault(key, transition.rejection)
                    continue

                # Lazy geometric validation only at actual expansion.
                if not _geometry_ok(
                    problem,
                    state,
                    candidate,
                    dynamic_rejections,
                    primitives=tuple(p.action for p in transition.primitives),
                ):
                    continue

                outcome.attempted_subgoals.add(sg_id)

                next_facts = apply_effects(
                    state.symbolic_facts, subgoal.destroy, subgoal.establish
                )
                next_bindings = state.group_ee_bindings
                if subgoal.group_id is not None and bound_ee is None:
                    next_bindings = with_group_binding(
                        next_bindings, subgoal.group_id, candidate.ee
                    )
                next_scene = problem.scene_updater.apply_subgoal_effects(
                    SceneState(
                        signature=state.scene_signature,
                        completed=tuple(sorted(state.completed_subgoals)),
                    ),
                    subgoal,
                    candidate,
                    next_facts,
                )
                next_state = SearchState(
                    completed_subgoals=state.completed_subgoals | {sg_id},
                    current_ee=candidate.ee,
                    held_tool=transition.next_held_tool,
                    group_ee_bindings=next_bindings,
                    symbolic_facts=next_facts,
                    scene_signature=next_scene.signature,
                    rack_signature=transition.next_rack_signature,
                )
                seq = _relax(
                    problem,
                    outcome,
                    heap,
                    stats,
                    state,
                    next_state,
                    cost,
                    transition.cost,
                    ParentEdge(
                        prev_state=state,
                        subgoal_id=sg_id,
                        candidate=candidate,
                        primitives=transition.primitives,
                        edge_cost=transition.cost,
                    ),
                    tie=(
                        sg_id,
                        candidate.candidate_id,
                        "|".join(p.action for p in transition.primitives),
                    ),
                    seq=seq,
                )
    else:
        outcome.frontier_exhausted = True

    if not heap and outcome.goal_state is None and not outcome.limit_reached:
        outcome.frontier_exhausted = True

    stats.lazy_checks_executed = problem.geometry.checks_executed
    stats.lazy_cache_hits = problem.geometry.cache_hits
    stats.elapsed_ms = int(round((time.monotonic() - start_time) * 1000))
    outcome.rejections = [
        dynamic_rejections[k] for k in sorted(dynamic_rejections)
    ]
    return outcome


def _relax(
    problem: SearchProblem,
    outcome: SearchOutcome,
    heap: list[tuple[tuple[int, ...], tuple[str, ...], int, SearchState]],
    stats: SearchStats,
    state: SearchState,
    next_state: SearchState,
    cost: CostVector,
    edge_cost: CostVector,
    edge: ParentEdge,
    tie: tuple[str, str, str],
    seq: int,
) -> int:
    stats.edges_generated += 1
    new_cost = cost + edge_cost
    known = outcome.best_cost.get(next_state)
    if known is not None and known.as_tuple() <= new_cost.as_tuple():
        stats.dominated_states_pruned += 1
        return seq
    outcome.best_cost[next_state] = new_cost
    outcome.parent[next_state] = edge
    seq += 1
    heapq.heappush(heap, (new_cost.as_tuple(), tie, seq, next_state))
    return seq


def reconstruct_edges(
    outcome: SearchOutcome, goal_state: SearchState
) -> list[ParentEdge]:
    """Edges from the initial state to the goal, in execution order."""
    edges: list[ParentEdge] = []
    state = goal_state
    while state in outcome.parent:
        edge = outcome.parent[state]
        edges.append(edge)
        state = edge.prev_state
    edges.reverse()
    return edges
