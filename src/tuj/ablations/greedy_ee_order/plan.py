"""Myopic EE-order policy: min immediate EE-switch among ready subgoals."""

from __future__ import annotations

import time
from collections import defaultdict

from tuj.m4_taskplanner.candidate_provider import (
    CatalogCandidateProvider,
    StaticCandidateProvider,
    filter_candidates,
)
from tuj.m4_taskplanner.conditions import (
    CheckerRegistry,
    apply_effects,
    initial_facts,
    symbolic_precondition_unmet,
)
from tuj.m4_taskplanner.constraints import TaskConstraintEngine
from tuj.m4_taskplanner.cost import CostVector
from tuj.m4_taskplanner.diagnostics import (
    Diagnostics,
    PlanStatus,
    ReasonCode,
    fluent_str,
    make_rejection,
)
from tuj.m4_taskplanner.feasibility import StaticFeasibilityChecker
from tuj.m4_taskplanner.models import TaskPlannerRequest
from tuj.m4_taskplanner.planner import _build_selected_plan
from tuj.m4_taskplanner.scene import SceneState, SymbolicSceneUpdater
from tuj.m4_taskplanner.search import ParentEdge, SearchOutcome, SearchStats
from tuj.m4_taskplanner.serialization import PlanningResult, SearchStatsModel
from tuj.m4_taskplanner.state import SearchState
from tuj.m4_taskplanner.suitability import PhysicsSuitabilityScorer
from tuj.m4_taskplanner.transitions import (
    TransitionContext,
    build_terminal_transition,
    build_transition,
)
from tuj.m4_taskplanner.validation import normalized_edges

METHOD = "greedy-ee-order"
VERSION = "greedy-min-ee-switch-v1"


def greedy_ee_order_plan(request: TaskPlannerRequest) -> PlanningResult:
    """Pick ready (subgoal, EE) pairs by immediate EE-switch cost, no lookahead.

    ``plan()`` / ``run_search()`` are never invoked. Transitions and
    ``_build_selected_plan`` preserve the M5 contract used by production.
    """
    started = time.monotonic()
    graph, catalog, policy = (
        request.task_graph,
        request.resource_catalog,
        request.planning_policy,
    )
    order = [sg.subgoal_id for sg in graph.subgoals]
    task = {
        **graph.task.model_dump(),
        "method": METHOD,
        "m4_invoked": False,
        "assignment_policy": VERSION,
        "scripted_grasp_greedy_extra": True,
    }
    rejections = []
    static_rejected = 0

    def finish(status, selected=None, blocked=None):
        task["assignment_time_ms"] = round((time.monotonic() - started) * 1000)
        result = PlanningResult(
            status=status,
            task=task,
            planner_version=VERSION,
            optimality_scope={
                "selection": (
                    "among dependency-ready subgoals, minimize immediate "
                    "EE-switch cost (no lookahead)"
                ),
                "search": "skipped",
            },
            selected_plan=selected,
            search_stats=SearchStatsModel(
                static_candidates_rejected=static_rejected,
                elapsed_ms=0,
            ),
            rejections=rejections,
            diagnostics=Diagnostics(blocked_subgoals=[blocked] if blocked else []),
        )
        result.model_extra["method"] = METHOD
        result.model_extra["m4_invoked"] = False
        result.model_extra["metrics"] = (
            {
                "N_EE": selected.cost_vector.ee_switches,
                "N_tool": selected.action_counts.n_tool_picks,
                "motion_cost": selected.cost_vector.motion_cost,
                "execution_cost": selected.cost_vector.execution_cost,
                "planning_time_ms": 0,
                "assignment_time_ms": task["assignment_time_ms"],
            }
            if selected is not None
            else {
                "N_EE": None,
                "N_tool": None,
                "motion_cost": None,
                "execution_cost": None,
                "planning_time_ms": 0,
                "assignment_time_ms": task["assignment_time_ms"],
            }
        )
        return result

    if len(order) != len(set(order)) or not order:
        rejections.append(
            make_rejection(
                "input",
                ReasonCode.DUPLICATE_SUBGOAL_ID,
                "subgoals must have unique IDs and be non-empty",
            )
        )
        return finish(PlanStatus.INVALID_INPUT)

    hard_edges = normalized_edges(graph.order_constraints)
    predecessors: dict[str, set[str]] = defaultdict(set)
    for before, after in hard_edges:
        if before not in order or after not in order:
            rejections.append(
                make_rejection(
                    "input",
                    ReasonCode.INVALID_TASK_CONTRACT,
                    f"order edge {before} -> {after} references unknown subgoal",
                    subgoal_id=after,
                )
            )
            return finish(PlanStatus.INFEASIBLE_NO_PLAN, blocked=after)
        predecessors[after].add(before)

    init = graph.initial_state
    updater = SymbolicSceneUpdater()
    initial_scene = updater.initial_scene(graph)
    initial_state = SearchState(
        completed_subgoals=frozenset(),
        current_ee=init.current_ee,
        held_tool=init.held_tool,
        group_ee_bindings=(),  # No group EE binding; greedy chooses per subgoal.
        symbolic_facts=initial_facts(init),
        scene_signature=initial_scene.signature,
        rack_signature=(
            tuple(sorted(init.rack_occupancy.items()))
            if init.rack_occupancy is not None
            else None
        ),
    )
    state = initial_state
    context = TransitionContext(
        catalog=catalog, policy=policy, initial_ee=init.current_ee
    )
    scorer = PhysicsSuitabilityScorer(catalog)
    static_checker = StaticFeasibilityChecker(catalog, policy, {})
    catalog_provider = CatalogCandidateProvider(catalog, policy)
    proposal_provider = (
        StaticCandidateProvider(request.candidate_proposals, catalog)
        if request.candidate_proposals is not None
        else None
    )
    checker = CheckerRegistry()
    constraints = TaskConstraintEngine(graph.constraints)
    edges = []
    cost = CostVector()
    subgoals = {sg.subgoal_id: sg for sg in graph.subgoals}
    remaining = set(order)
    candidate_cache: dict[str, list] = {}

    def candidates_for(sg):
        cached = candidate_cache.get(sg.subgoal_id)
        if cached is not None:
            return cached
        nonlocal static_rejected
        if (
            proposal_provider is not None
            and sg.subgoal_id in request.candidate_proposals
        ):
            raw, rejected = proposal_provider.candidates_for(sg)
        else:
            raw, rejected = catalog_provider.candidates_for(sg)
        rejections.extend(rejected)
        kept, rejected = filter_candidates(
            sg, raw, static_checker, policy, suitability_scorer=scorer
        )
        rejections.extend(rejected)
        static_rejected += sum(item.scope == "candidate" for item in rejected)
        candidate_cache[sg.subgoal_id] = kept
        return kept

    while remaining:
        ready = []
        for sg_id in remaining:
            if not predecessors[sg_id] <= state.completed_subgoals:
                continue
            sg = subgoals[sg_id]
            unmet = symbolic_precondition_unmet(sg, state.symbolic_facts, checker)
            blockers = constraints.blockers(
                sg_id, state.completed_subgoals, state.symbolic_facts
            )
            if unmet or blockers:
                continue
            ready.append(sg)

        if not ready:
            # Distinguish blocked-by-deps from empty remaining.
            blocked = sorted(remaining)[0]
            sg = subgoals[blocked]
            unmet = symbolic_precondition_unmet(sg, state.symbolic_facts, checker)
            blockers = constraints.blockers(
                blocked, state.completed_subgoals, state.symbolic_facts
            )
            message = (
                f"no ready subgoal under greedy EE-order; unmet="
                f"{[fluent_str(key) for key in unmet]} blockers="
                f"{[b.message for b in blockers]}"
            )
            rejections.append(
                make_rejection(
                    "subgoal",
                    ReasonCode.UNESTABLISHABLE_PRECONDITIONS,
                    message,
                    subgoal_id=blocked,
                )
            )
            return finish(PlanStatus.INFEASIBLE_NO_PLAN, blocked=blocked)

        best = None
        best_key = None
        for sg in ready:
            kept = candidates_for(sg)
            if not kept:
                continue
            for candidate in kept:
                transition = build_transition(state, candidate, context, sg)
                if not transition.feasible:
                    if transition.rejection is not None:
                        rejections.append(transition.rejection)
                    continue
                key = (
                    transition.cost.ee_switches,
                    transition.cost.motion_cost,
                    transition.cost.execution_cost,
                    sg.subgoal_id,
                    candidate.ee,
                    candidate.candidate_id,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = (sg, candidate, transition)

        if best is None:
            blocked = ready[0].subgoal_id
            rejections.append(
                make_rejection(
                    "subgoal",
                    ReasonCode.NO_CANDIDATES,
                    "no feasible (subgoal, EE) candidate among ready subgoals",
                    subgoal_id=blocked,
                )
            )
            return finish(PlanStatus.INFEASIBLE_NO_CANDIDATE, blocked=blocked)

        sg, candidate, transition = best
        facts = apply_effects(state.symbolic_facts, sg.destroy, sg.establish)
        scene = updater.apply_subgoal_effects(
            SceneState(
                signature=state.scene_signature,
                completed=tuple(sorted(state.completed_subgoals)),
            ),
            sg,
            candidate,
            facts,
        )
        edges.append(
            ParentEdge(
                state, sg.subgoal_id, candidate, transition.primitives, transition.cost
            )
        )
        state = SearchState(
            completed_subgoals=state.completed_subgoals | {sg.subgoal_id},
            current_ee=candidate.ee,
            held_tool=transition.next_held_tool,
            group_ee_bindings=(),
            symbolic_facts=facts,
            scene_signature=scene.signature,
            rack_signature=transition.next_rack_signature,
        )
        cost += transition.cost
        remaining.remove(sg.subgoal_id)

    terminal = build_terminal_transition(state, context)
    if not terminal.feasible:
        if terminal.rejection is not None:
            rejections.append(terminal.rejection)
        return finish(PlanStatus.INFEASIBLE_NO_PLAN)
    if terminal.primitives:
        edges.append(ParentEdge(state, None, None, terminal.primitives, terminal.cost))
        cost += terminal.cost
        state = SearchState(
            completed_subgoals=state.completed_subgoals,
            current_ee=terminal.next_ee,
            held_tool=None,
            group_ee_bindings=(),
            symbolic_facts=state.symbolic_facts,
            scene_signature=state.scene_signature,
            rack_signature=terminal.next_rack_signature,
        )
    outcome = SearchOutcome(state, {}, {}, {}, SearchStats())
    selected = _build_selected_plan(
        edges,
        subgoals,
        outcome,
        initial_state,
        cost,
        graph.constraints,
        hard_edges,
        catalog,
    )
    selected.model_extra["method"] = METHOD
    for assignment in selected.candidate_assignments:
        assignment.ee_selection_source = VERSION
    return finish(PlanStatus.SUCCESS, selected)
