"""Top-level orchestration: validate -> candidates -> Dijkstra -> result."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from tuj.m3_taskplanner.blocks import build_blocks
from tuj.m3_taskplanner.candidate_provider import (
    Candidate,
    CatalogCandidateProvider,
    StaticCandidateProvider,
    filter_candidates,
)
from tuj.m3_taskplanner.conditions import CheckerRegistry, initial_facts, true_facts
from tuj.m3_taskplanner.constraints import (
    TaskConstraintEngine,
    build_constraint_resolution_trace,
)
from tuj.m3_taskplanner.diagnostics import (
    ConstraintBlockerDetail,
    Diagnostics,
    PlanStatus,
    ReasonCode,
    Rejection,
    UnmetPrecondition,
    make_rejection,
)
from tuj.m3_taskplanner.feasibility import (
    AlwaysPassGeometryChecker,
    GeometricFeasibilityChecker,
    GeometryCache,
    StaticFeasibilityChecker,
)
from tuj.m3_taskplanner.models import ExecutionState, TaskPlannerRequest
from tuj.m3_taskplanner.scene import SceneStateUpdater, SymbolicSceneUpdater
from tuj.m3_taskplanner.search import (
    ParentEdge,
    SearchOutcome,
    SearchProblem,
    SearchStats,
    run_search,
)
from tuj.m3_taskplanner.serialization import (
    ActionCounts,
    CandidateAssignment,
    CostVectorModel,
    PlanStep,
    PlanningResult,
    SearchStatsModel,
    SelectedPlan,
)
from tuj.m3_taskplanner.state import SearchState
from tuj.m3_taskplanner.suitability import PhysicsSuitabilityScorer, SuitabilityScorer
from tuj.m3_taskplanner.transitions import P, TransitionContext
from tuj.m3_taskplanner.validation import normalized_edges, validate_request

if TYPE_CHECKING:
    from tuj.m3_taskplanner.replanning import NoGoodSet


def plan(
    request: TaskPlannerRequest,
    *,
    geometry_checker: GeometricFeasibilityChecker | None = None,
    scene_updater: SceneStateUpdater | None = None,
    checker_registry: CheckerRegistry | None = None,
    suitability_scorer: SuitabilityScorer | Literal["default"] | None = "default",
    no_goods: "NoGoodSet | None" = None,
    execution_state: ExecutionState | None = None,
) -> PlanningResult:
    task_graph = request.task_graph
    policy = request.planning_policy
    task = task_graph.task.model_dump()
    scene_updater = scene_updater or SymbolicSceneUpdater()
    checker_registry = checker_registry or CheckerRegistry()
    geometry = GeometryCache(checker=geometry_checker or AlwaysPassGeometryChecker())

    # ---- eager input validation (never searches on bad input) -------------
    outcome = validate_request(request)
    if outcome.status is not None:
        return PlanningResult(
            status=outcome.status,
            task=task,
            rejections=outcome.rejections,
            diagnostics=outcome.diagnostics,
        )

    subgoals = {sg.subgoal_id: sg for sg in task_graph.subgoals}
    constraint_engine = TaskConstraintEngine(task_graph.constraints)

    # ---- execution-state overrides (replanning) ---------------------------
    completed: frozenset[str] = frozenset()
    if execution_state is not None:
        execution_rejections = _validate_execution_state(
            execution_state,
            request,
            outcome.predecessors,
            outcome.group_feasible_ee,
        )
        if execution_rejections:
            return PlanningResult(
                status=PlanStatus.INVALID_INPUT,
                task=task,
                rejections=execution_rejections,
            )
        unknown = sorted(
            set(execution_state.completed_subgoals) - set(subgoals)
        )
        if unknown:
            return PlanningResult(
                status=PlanStatus.INVALID_INPUT,
                task=task,
                rejections=[
                    make_rejection(
                        "input",
                        ReasonCode.UNKNOWN_PREDECESSOR,
                        f"execution state lists unknown completed subgoals "
                        f"{unknown}",
                    )
                ],
            )
        completed = frozenset(execution_state.completed_subgoals)
        current_ee = execution_state.current_ee
        held_tool = execution_state.held_tool
        facts = true_facts(execution_state.facts)
        scene_signature = (
            execution_state.scene_signature
            or scene_updater.initial_scene(task_graph).signature
        )
        rack_occupancy = (
            execution_state.rack_occupancy
            if execution_state.rack_occupancy is not None
            else task_graph.initial_state.rack_occupancy
        )
        group_bindings = tuple(
            sorted(getattr(execution_state, "group_ee_bindings", {}).items())
        )
        if current_ee not in request.resource_catalog.end_effectors:
            return PlanningResult(
                status=PlanStatus.INVALID_INPUT,
                task=task,
                rejections=[
                    make_rejection(
                        "input",
                        ReasonCode.UNKNOWN_EE,
                        f"execution state current_ee {current_ee!r} not in "
                        "catalog",
                    )
                ],
            )
    else:
        init = task_graph.initial_state
        current_ee = init.current_ee
        held_tool = init.held_tool
        facts = initial_facts(init)
        scene_signature = scene_updater.initial_scene(task_graph).signature
        rack_occupancy = init.rack_occupancy
        group_bindings = ()

    rack_signature = (
        tuple(sorted(rack_occupancy.items()))
        if rack_occupancy is not None
        else None
    )

    initial_state = SearchState(
        completed_subgoals=completed,
        current_ee=current_ee,
        held_tool=held_tool,
        group_ee_bindings=group_bindings,
        symbolic_facts=facts,
        scene_signature=scene_signature,
        rack_signature=rack_signature,
    )

    # ---- candidate generation + eager filtering ---------------------------
    static_checker = StaticFeasibilityChecker(
        request.resource_catalog, policy, outcome.group_feasible_ee
    )
    active_suitability_scorer = (
        PhysicsSuitabilityScorer(request.resource_catalog)
        if isinstance(suitability_scorer, str)
        else suitability_scorer
    )
    catalog_provider = CatalogCandidateProvider(request.resource_catalog, policy)
    static_provider = (
        StaticCandidateProvider(request.candidate_proposals, request.resource_catalog)
        if request.candidate_proposals is not None
        else None
    )
    banned_global = (
        frozenset(no_goods.global_candidates) if no_goods is not None else frozenset()
    )

    candidates: dict[str, list[Candidate]] = {}
    all_rejections: list[Rejection] = []
    empty_subgoals: list[str] = []
    for sg_id in sorted(subgoals):
        if sg_id in completed:
            candidates[sg_id] = []
            continue
        subgoal = subgoals[sg_id]
        if (
            static_provider is not None
            and request.candidate_proposals is not None
            and sg_id in request.candidate_proposals
        ):
            raw, provider_rejections = static_provider.candidates_for(subgoal)
        else:
            raw, provider_rejections = catalog_provider.candidates_for(subgoal)
        all_rejections.extend(provider_rejections)
        kept, filter_rejections = filter_candidates(
            subgoal,
            raw,
            static_checker,
            policy,
            banned_global,
            active_suitability_scorer,
        )
        all_rejections.extend(filter_rejections)
        candidates[sg_id] = kept
        if not kept:
            empty_subgoals.append(sg_id)

    static_rejected = sum(1 for r in all_rejections if r.scope == "candidate")

    if empty_subgoals:
        return PlanningResult(
            status=PlanStatus.INFEASIBLE_NO_CANDIDATE,
            task=task,
            rejections=all_rejections
            + [
                make_rejection(
                    "subgoal",
                    ReasonCode.NO_CANDIDATES,
                    "no feasible candidate survived thresholding and static "
                    "checks",
                    subgoal_id=sg_id,
                )
                for sg_id in empty_subgoals
            ],
            diagnostics=Diagnostics(blocked_subgoals=empty_subgoals),
            search_stats=SearchStatsModel(
                static_candidates_rejected=static_rejected
            ),
        )

    # ---- unified Dijkstra search ------------------------------------------
    context = TransitionContext(
        catalog=request.resource_catalog,
        policy=policy,
        initial_ee=current_ee,
    )
    problem = SearchProblem(
        subgoals=subgoals,
        candidates=candidates,
        predecessors=outcome.predecessors,
        initial_state=initial_state,
        policy=policy,
        transition_context=context,
        geometry=geometry,
        scene_updater=scene_updater,
        checker_registry=checker_registry,
        constraint_engine=constraint_engine,
        banned_scene_candidates=(
            frozenset(no_goods.scene_candidates)
            if no_goods is not None
            else frozenset()
        ),
        banned_transitions=(
            frozenset(no_goods.transitions) if no_goods is not None else frozenset()
        ),
    )
    search_outcome = run_search(problem)
    all_rejections.extend(search_outcome.rejections)
    stats_model = _stats_model(search_outcome.stats, static_rejected)

    if search_outcome.goal_state is None:
        status = (
            PlanStatus.SEARCH_LIMIT_REACHED
            if search_outcome.limit_reached
            else PlanStatus.INFEASIBLE_NO_PLAN
        )
        blocked = sorted(
            set(subgoals) - completed - search_outcome.attempted_subgoals
        )
        diagnostics = Diagnostics(
            blocked_subgoals=blocked,
            unmet_preconditions=[
                UnmetPrecondition(subgoal_id=s, fluent=f)
                for s, f in sorted(search_outcome.unmet_preconditions)
            ],
            constraint_blockers=[
                ConstraintBlockerDetail(
                    subgoal_id=subgoal_id,
                    constraint_type=kind,
                    constraint_id=constraint_id,
                    message=message,
                )
                for subgoal_id, kind, constraint_id, message in sorted(
                    search_outcome.constraint_blockers
                )
            ],
            frontier_exhausted=search_outcome.frontier_exhausted,
        )
        if search_outcome.limit_reached:
            all_rejections.append(
                make_rejection(
                    "search",
                    ReasonCode.SEARCH_LIMIT,
                    "search expansion or timeout limit reached before a "
                    "complete plan was found",
                )
            )
        else:
            all_rejections.append(
                make_rejection(
                    "search",
                    ReasonCode.FRONTIER_EXHAUSTED,
                    "the feasible search frontier was exhausted without a "
                    "complete plan",
                )
            )
        return PlanningResult(
            status=status,
            task=task,
            rejections=all_rejections,
            diagnostics=diagnostics,
            search_stats=stats_model,
        )

    # ---- reconstruct the executable plan ----------------------------------
    from tuj.m3_taskplanner.search import reconstruct_edges

    edges = reconstruct_edges(search_outcome, search_outcome.goal_state)
    selected = _build_selected_plan(
        edges,
        subgoals,
        search_outcome,
        initial_state,
        search_outcome.best_cost[search_outcome.goal_state],
        task_graph.constraints,
        normalized_edges(task_graph.order_constraints),
    )
    return PlanningResult(
        status=PlanStatus.SUCCESS,
        task=task,
        selected_plan=selected,
        rejections=all_rejections,
        diagnostics=Diagnostics(),
        search_stats=stats_model,
    )


def _validate_execution_state(
    execution_state: ExecutionState,
    request: TaskPlannerRequest,
    predecessors: dict[str, frozenset[str]],
    group_feasible_ee: dict[str, frozenset[str]],
) -> list[Rejection]:
    """Validate a replanning root before it enters the search graph."""
    rejections: list[Rejection] = []
    subgoals = {sg.subgoal_id: sg for sg in request.task_graph.subgoals}
    completed = set(execution_state.completed_subgoals)
    catalog = request.resource_catalog

    def reject(code: ReasonCode, message: str, **extra) -> None:
        rejections.append(make_rejection("input", code, message, **extra))

    unknown_completed = sorted(completed - set(subgoals))
    if unknown_completed:
        reject(
            ReasonCode.UNKNOWN_PREDECESSOR,
            f"execution state lists unknown completed subgoals {unknown_completed}",
        )
    for subgoal_id in sorted(completed & set(subgoals)):
        missing = sorted(predecessors.get(subgoal_id, frozenset()) - completed)
        if missing:
            reject(
                ReasonCode.INVALID_COMPLETED_PREFIX,
                f"completed subgoal {subgoal_id!r} is missing predecessors {missing}",
                subgoal_id=subgoal_id,
            )

    ee_spec = catalog.end_effectors.get(execution_state.current_ee)
    if ee_spec is None:
        reject(
            ReasonCode.UNKNOWN_EE,
            f"execution state current_ee {execution_state.current_ee!r} not in catalog",
        )

    if execution_state.held_tool is not None:
        tool_spec = catalog.tools.get(execution_state.held_tool)
        if tool_spec is None:
            reject(
                ReasonCode.UNKNOWN_TOOL,
                f"execution state held_tool {execution_state.held_tool!r} not in catalog",
            )
        elif ee_spec is not None and (
            execution_state.current_ee not in tool_spec.compatible_ee
            or execution_state.held_tool not in ee_spec.compatible_tools
        ):
            reject(
                ReasonCode.EE_TOOL_INCOMPATIBLE,
                f"execution EE {execution_state.current_ee!r} and held tool "
                f"{execution_state.held_tool!r} are incompatible",
            )

    groups = {
        sg.group_id
        for sg in subgoals.values()
        if sg.group_id is not None
    }
    bindings = execution_state.group_ee_bindings
    for group_id, ee in sorted(bindings.items()):
        if group_id not in groups:
            reject(
                ReasonCode.UNKNOWN_GROUP_BINDING,
                f"execution state binds unknown group {group_id!r}",
                group_id=group_id,
            )
        elif ee not in group_feasible_ee.get(group_id, frozenset()):
            reject(
                ReasonCode.GROUP_EE_CONFLICT,
                f"execution binding {group_id!r}->{ee!r} is outside the group "
                f"feasible set {sorted(group_feasible_ee.get(group_id, frozenset()))}",
                group_id=group_id,
            )

    for group_id in sorted(groups):
        members = {
            sg.subgoal_id for sg in subgoals.values() if sg.group_id == group_id
        }
        if completed & members and members - completed and group_id not in bindings:
            reject(
                ReasonCode.INVALID_RESOURCE_STATE,
                f"partially completed group {group_id!r} has no persisted EE binding",
                group_id=group_id,
            )

    return rejections


def _stats_model(stats: SearchStats, static_rejected: int) -> SearchStatsModel:
    return SearchStatsModel(
        states_popped=stats.states_popped,
        states_expanded=stats.states_expanded,
        edges_generated=stats.edges_generated,
        dominated_states_pruned=stats.dominated_states_pruned,
        static_candidates_rejected=static_rejected,
        lazy_checks_executed=stats.lazy_checks_executed,
        lazy_cache_hits=stats.lazy_cache_hits,
        elapsed_ms=stats.elapsed_ms,
    )


def _build_selected_plan(
    edges: list[ParentEdge],
    subgoals: dict,
    search_outcome: SearchOutcome,
    initial_state: SearchState,
    total_cost,
    contract,
    hard_edges: list[tuple[str, str]],
) -> SelectedPlan:
    steps: list[PlanStep] = []
    subgoal_order: list[str] = []
    assignments: list[CandidateAssignment] = []
    counts = ActionCounts()
    step_index = 0

    for edge in edges:
        for prim in edge.primitives:
            params = prim.parameters_dict()
            if prim.action == P.EXECUTE_SUBGOAL:
                subgoal = subgoals[edge.subgoal_id]
                steps.append(
                    PlanStep(
                        step_index=step_index,
                        kind="subgoal",
                        action=prim.action,
                        parameters=params,
                        subgoal_id=edge.subgoal_id,
                        candidate_id=(
                            edge.candidate.candidate_id if edge.candidate else None
                        ),
                        preconditions=[
                            c.model_dump(mode="json", by_alias=True)
                            for c in subgoal.preconditions
                        ],
                        postconditions=[
                            c.model_dump(mode="json", by_alias=True)
                            for c in subgoal.postconditions
                        ],
                        cost_delta=prim.cost_delta.to_dict(skip_zero=True),
                        verification_required=prim.verification_required,
                    )
                )
            else:
                steps.append(
                    PlanStep(
                        step_index=step_index,
                        kind="transition",
                        action=prim.action,
                        parameters=params,
                        subgoal_id=edge.subgoal_id,
                        candidate_id=(
                            edge.candidate.candidate_id if edge.candidate else None
                        ),
                        preconditions=list(prim.preconditions),
                        postconditions=list(prim.effects),
                        cost_delta=prim.cost_delta.to_dict(skip_zero=True),
                        verification_required=prim.verification_required,
                    )
                )
            if prim.action == P.PICK_TOOL:
                counts.n_tool_picks += 1
            elif prim.action in (P.RETURN_TOOL, P.TERMINAL_RETURN_TOOL):
                counts.n_tool_returns += 1
            elif prim.action == P.DETACH_EE:
                counts.n_ee_detaches += 1
            elif prim.action == P.ATTACH_EE:
                counts.n_ee_attaches += 1
            elif prim.action == P.TERMINAL_RESTORE_EE:
                counts.n_ee_detaches += 1
                counts.n_ee_attaches += 1
            step_index += 1
        if edge.subgoal_id is not None:
            subgoal_order.append(edge.subgoal_id)
            if edge.candidate is not None:
                assignments.append(
                    CandidateAssignment(
                        subgoal_id=edge.subgoal_id,
                        candidate_id=edge.candidate.candidate_id,
                        group_id=subgoals[edge.subgoal_id].group_id,
                        ee=edge.candidate.ee,
                        tool=edge.candidate.tool,
                        action_type=edge.candidate.action_type,
                        target_ids=list(edge.candidate.target_ids),
                        goal_region_id=subgoals[edge.subgoal_id].goal_region_id,
                        description=subgoals[edge.subgoal_id].description,
                        grasp_id=edge.candidate.grasp_id,
                        grasp=edge.candidate.grasp,
                        action_parameters=dict(edge.candidate.metadata),
                        suitability_score=edge.candidate.suitability_score,
                        suitability=edge.candidate.metadata.get("suitability"),
                        ee_selection_source="task_planner",
                        ee_feasible_set_source=(
                            subgoals[edge.subgoal_id].feasible_ee_source
                        ),
                        tool_selection_source=(
                            subgoals[edge.subgoal_id].tool_selection_source
                        ),
                    )
                )

    goal = search_outcome.goal_state
    assert goal is not None
    step_dicts = [s.model_dump() for s in steps]
    ee_blocks, tool_blocks = build_blocks(
        step_dicts, initial_state.current_ee, initial_state.held_tool
    )
    terminal_state = {
        "current_ee": goal.current_ee,
        "held_tool": goal.held_tool,
        "completed_subgoals": sorted(goal.completed_subgoals),
        "facts": sorted(
            f"{name}({', '.join(args)})" for name, args in goal.symbolic_facts
        ),
        "scene_signature": goal.scene_signature,
        "rack_occupancy": (
            dict(goal.rack_signature)
            if goal.rack_signature is not None
            else None
        ),
    }
    return SelectedPlan(
        cost_vector=CostVectorModel.from_cost(total_cost),
        subgoal_order=subgoal_order,
        group_ee_assignments=dict(goal.group_ee_bindings),
        candidate_assignments=assignments,
        steps=steps,
        ee_blocks=ee_blocks,
        tool_blocks=tool_blocks,
        action_counts=counts,
        terminal_state=terminal_state,
        constraint_trace=build_constraint_resolution_trace(
            subgoal_order, contract, hard_edges
        ),
    )
