"""Independent per-subgoal baseline with no joint EE/task search."""

from __future__ import annotations

import time

from tuj.m4_taskplanner.candidate_provider import (
    CatalogCandidateProvider,
    StaticCandidateProvider,
    _ranking_key,
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

METHOD = "without-planner"
VERSION = "independent-subgoal-assignment-v1"


def independent_plan(request: TaskPlannerRequest) -> PlanningResult:
    """Pick each subgoal's top feasible candidate in M2 order, without lookahead.

    The M4 plan()/run_search() functions are never invoked. M4's shared
    transition and output models merely preserve the existing M5 contract.
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
                "selection": "independent top-ranked feasible per subgoal",
                "search": "skipped",
            },
            selected_plan=selected,
            search_stats=SearchStatsModel(
                static_candidates_rejected=static_rejected,
                elapsed_ms=0,  # Global search was skipped, not measured at zero cost.
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
    positions = {sid: index for index, sid in enumerate(order)}
    hard_edges = normalized_edges(graph.order_constraints)
    for before, after in hard_edges:
        if (
            before not in positions
            or after not in positions
            or positions[before] >= positions[after]
        ):
            rejections.append(
                make_rejection(
                    "input",
                    ReasonCode.INVALID_TASK_CONTRACT,
                    f"M2 order violates {before} -> {after}; no reordering in {METHOD}",
                    subgoal_id=after,
                )
            )
            return finish(PlanStatus.INFEASIBLE_NO_PLAN, blocked=after)

    init = graph.initial_state
    updater = SymbolicSceneUpdater()
    initial_scene = updater.initial_scene(graph)
    initial_state = SearchState(
        completed_subgoals=frozenset(),
        current_ee=init.current_ee,
        held_tool=init.held_tool,
        group_ee_bindings=(),  # Deliberately no inter-subgoal EE binding.
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

    for sg in graph.subgoals:
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
        if not kept:
            rejections.append(
                make_rejection(
                    "subgoal",
                    ReasonCode.NO_CANDIDATES,
                    "no feasible candidate for independent assignment",
                    subgoal_id=sg.subgoal_id,
                )
            )
            return finish(PlanStatus.INFEASIBLE_NO_CANDIDATE, blocked=sg.subgoal_id)

        # This choice uses only this subgoal's M3/GK evidence, never state cost.
        candidate = min(kept, key=_ranking_key)
        unmet = symbolic_precondition_unmet(sg, state.symbolic_facts, checker)
        blockers = constraints.blockers(
            sg.subgoal_id, state.completed_subgoals, state.symbolic_facts
        )
        if unmet or blockers:
            message = f"M2 order has unmet preconditions {[fluent_str(key) for key in unmet]} / constraints {[b.message for b in blockers]}"
            rejections.append(
                make_rejection(
                    "subgoal",
                    ReasonCode.UNESTABLISHABLE_PRECONDITIONS,
                    message,
                    subgoal_id=sg.subgoal_id,
                )
            )
            return finish(PlanStatus.INFEASIBLE_NO_PLAN, blocked=sg.subgoal_id)
        transition = build_transition(state, candidate, context, sg)
        if not transition.feasible:
            if transition.rejection is not None:
                rejections.append(transition.rejection)
            return finish(PlanStatus.INFEASIBLE_NO_PLAN, blocked=sg.subgoal_id)
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
