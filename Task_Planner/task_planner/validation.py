"""Input validation: edges/cycles, unestablishable, groups, schema checks.

The input's ``cycle_detected`` flag is honored but never trusted alone: cycles
are re-detected from the actual edges. ``n_topological_orders`` is diagnostic
only and is never used to generate or enumerate orders.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from task_planner.conditions import EffectConflictError, validate_subgoal_effects
from task_planner.diagnostics import (
    Diagnostics,
    PlanStatus,
    ReasonCode,
    Rejection,
    make_rejection,
)
from task_planner.models import OrderConstraints, TaskPlannerRequest


def normalized_edges(order_constraints: OrderConstraints) -> list[tuple[str, str]]:
    """Normalize edges into (predecessor, successor) pairs.

    Accepts ``[pred, succ]`` pairs or mappings with from/to, pred/succ,
    before/after, or source/target keys.
    """
    result: list[tuple[str, str]] = []
    for raw in order_constraints.edges:
        pair = _normalize_edge(raw)
        if pair is None:
            raise MalformedEdgeError(raw)
        result.append(pair)
    return result


class MalformedEdgeError(ValueError):
    def __init__(self, raw: Any) -> None:
        self.raw = raw
        super().__init__(f"unrecognized order-constraint edge: {raw!r}")


def _normalize_edge(raw: Any) -> tuple[str, str] | None:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        a, b = raw
        if isinstance(a, str) and isinstance(b, str):
            return (a, b)
        return None
    if isinstance(raw, dict):
        for pred_key, succ_key in (
            ("from", "to"),
            ("pred", "succ"),
            ("before", "after"),
            ("source", "target"),
        ):
            if pred_key in raw and succ_key in raw:
                a, b = raw[pred_key], raw[succ_key]
                if isinstance(a, str) and isinstance(b, str):
                    return (a, b)
    return None


def detect_cycle(node_ids: list[str], edges: list[tuple[str, str]]) -> bool:
    """Kahn's algorithm; True if the edge set contains a cycle."""
    indegree: dict[str, int] = {n: 0 for n in node_ids}
    successors: dict[str, list[str]] = {n: [] for n in node_ids}
    for pred, succ in edges:
        successors[pred].append(succ)
        indegree[succ] += 1
    queue = deque(sorted(n for n, d in indegree.items() if d == 0))
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for nxt in successors[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return visited != len(node_ids)


def build_predecessor_map(
    node_ids: list[str], edges: list[tuple[str, str]]
) -> dict[str, frozenset[str]]:
    preds: dict[str, set[str]] = {n: set() for n in node_ids}
    for pred, succ in edges:
        preds[succ].add(pred)
    return {n: frozenset(p) for n, p in preds.items()}


@dataclass
class ValidationOutcome:
    """Result of eager input validation, before any search runs."""

    status: PlanStatus | None = None  # None means "input is searchable"
    rejections: list[Rejection] = field(default_factory=list)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    predecessors: dict[str, frozenset[str]] = field(default_factory=dict)
    # group_id -> intersection of feasible_ee across the group's subgoals.
    group_feasible_ee: dict[str, frozenset[str]] = field(default_factory=dict)


def validate_request(request: TaskPlannerRequest) -> ValidationOutcome:
    outcome = ValidationOutcome()
    planner_a = request.planner_a
    subgoals = planner_a.subgoals
    catalog = request.resource_catalog

    def reject(scope, code: ReasonCode, message: str, **extra: Any) -> None:
        outcome.rejections.append(make_rejection(scope, code, message, **extra))

    # Duplicate subgoal ids.
    seen: set[str] = set()
    for sg in subgoals:
        if sg.subgoal_id in seen:
            reject(
                "input",
                ReasonCode.DUPLICATE_SUBGOAL_ID,
                f"duplicate subgoal_id {sg.subgoal_id!r}",
                subgoal_id=sg.subgoal_id,
            )
        seen.add(sg.subgoal_id)
    if any(r.reason_code == ReasonCode.DUPLICATE_SUBGOAL_ID for r in outcome.rejections):
        outcome.status = PlanStatus.INVALID_INPUT
        return outcome

    subgoal_ids = [sg.subgoal_id for sg in subgoals]
    id_set = set(subgoal_ids)

    # Planner A's optional-choice contract is validated before search so no
    # mutex/open/threat record can silently refer to a missing node/condition.
    outcome.rejections.extend(_validate_contract_references(request, id_set))
    if outcome.rejections:
        outcome.status = PlanStatus.INVALID_INPUT
        return outcome

    # Candidate ids are cache/no-good identities and therefore must be unique
    # across the request. Mapping keys and each proposal's declared subgoal id
    # must also refer to the same known subgoal.
    candidate_ids: set[str] = set()
    for mapped_sg, proposals in sorted(
        (request.candidate_proposals or {}).items()
    ):
        if mapped_sg not in id_set:
            reject(
                "input",
                ReasonCode.SUBGOAL_MISMATCH,
                f"candidate_proposals references unknown subgoal {mapped_sg!r}",
                subgoal_id=mapped_sg,
            )
        for proposal in proposals:
            if proposal.subgoal_id != mapped_sg or proposal.subgoal_id not in id_set:
                reject(
                    "input",
                    ReasonCode.SUBGOAL_MISMATCH,
                    f"proposal {proposal.candidate_id!r} declares "
                    f"subgoal_id={proposal.subgoal_id!r} but is mapped under {mapped_sg!r}",
                    subgoal_id=mapped_sg,
                    candidate_id=proposal.candidate_id,
                )
            if proposal.candidate_id in candidate_ids:
                reject(
                    "input",
                    ReasonCode.DUPLICATE_CANDIDATE_ID,
                    f"duplicate candidate_id {proposal.candidate_id!r}",
                    subgoal_id=proposal.subgoal_id,
                    candidate_id=proposal.candidate_id,
                )
            candidate_ids.add(proposal.candidate_id)
    if outcome.rejections:
        outcome.status = PlanStatus.INVALID_INPUT
        return outcome

    # Edge parsing.
    try:
        edges = normalized_edges(planner_a.order_constraints)
    except MalformedEdgeError as exc:
        outcome.status = PlanStatus.INVALID_INPUT
        reject("input", ReasonCode.MALFORMED_EDGE, str(exc))
        return outcome

    # Unknown endpoint ids.
    for pred, succ in edges:
        for node in (pred, succ):
            if node not in id_set:
                reject(
                    "input",
                    ReasonCode.UNKNOWN_PREDECESSOR,
                    f"order-constraint edge references unknown subgoal {node!r}",
                    subgoal_id=node,
                )
    if any(r.reason_code == ReasonCode.UNKNOWN_PREDECESSOR for r in outcome.rejections):
        outcome.status = PlanStatus.INVALID_INPUT
        return outcome

    # Cycle: re-detect from actual edges; also honor the input flag.
    has_cycle = detect_cycle(subgoal_ids, edges)
    if has_cycle or planner_a.order_constraints.cycle_detected:
        outcome.status = PlanStatus.INFEASIBLE_REDECOMPOSE
        reject(
            "input",
            ReasonCode.CYCLE_DETECTED,
            "precedence cycle detected"
            + ("" if has_cycle else " (declared by Planner A)"),
        )
        return outcome

    # Unestablishable preconditions -> ask Planner A to redecompose.
    redecompose_signals = planner_a.constraints.redecompose_signals
    if planner_a.order_constraints.unestablishable or redecompose_signals:
        outcome.status = PlanStatus.INFEASIBLE_REDECOMPOSE
        reject(
            "input",
            ReasonCode.UNESTABLISHABLE_PRECONDITIONS,
            "Planner A requested redecomposition: "
            f"{planner_a.order_constraints.unestablishable or redecompose_signals!r}",
        )
        return outcome

    # Per-subgoal schema checks.
    for sg in subgoals:
        if sg.unique_solution and len(sg.feasible_ee) != 1:
            reject(
                "subgoal",
                ReasonCode.UNIQUE_SOLUTION_VIOLATION,
                f"unique_solution=true but feasible_ee={sg.feasible_ee!r}",
                subgoal_id=sg.subgoal_id,
            )
        try:
            validate_subgoal_effects(sg)
        except EffectConflictError as exc:
            reject(
                "subgoal",
                ReasonCode.EFFECT_CONFLICT,
                str(exc),
                subgoal_id=sg.subgoal_id,
            )
    if outcome.rejections:
        outcome.status = PlanStatus.INVALID_INPUT
        return outcome

    # Initial EE must exist in the catalog.
    if planner_a.initial_state.current_ee not in catalog.end_effectors:
        outcome.status = PlanStatus.INVALID_INPUT
        reject(
            "input",
            ReasonCode.UNKNOWN_EE,
            f"initial current_ee {planner_a.initial_state.current_ee!r} not in "
            "resource catalog",
        )
        return outcome

    # Optional initial held-tool state must describe a coherent catalog entry.
    init = planner_a.initial_state
    if init.held_tool is not None:
        tool = catalog.tools.get(init.held_tool)
        if tool is None:
            outcome.status = PlanStatus.INVALID_INPUT
            reject(
                "input",
                ReasonCode.UNKNOWN_TOOL,
                f"initial held_tool {init.held_tool!r} not in catalog",
            )
            return outcome
        ee = catalog.end_effectors[init.current_ee]
        if (
            init.current_ee not in tool.compatible_ee
            or init.held_tool not in ee.compatible_tools
        ):
            outcome.status = PlanStatus.INVALID_INPUT
            reject(
                "input",
                ReasonCode.EE_TOOL_INCOMPATIBLE,
                f"initial EE {init.current_ee!r} and held tool "
                f"{init.held_tool!r} are incompatible",
            )
            return outcome
    # Empty feasible_ee makes a subgoal unassignable.
    for sg in subgoals:
        if not sg.feasible_ee:
            reject(
                "subgoal",
                ReasonCode.EMPTY_FEASIBLE_EE,
                "feasible_ee is empty; no EE can be assigned",
                subgoal_id=sg.subgoal_id,
            )
    if outcome.rejections:
        outcome.status = PlanStatus.INFEASIBLE_NO_CANDIDATE
        return outcome

    # Group-feasible EE = intersection over the group's subgoals.
    groups: dict[str, list[str]] = {}
    group_sets: dict[str, frozenset[str]] = {}
    for sg in subgoals:
        if sg.group_id is None:
            continue
        groups.setdefault(sg.group_id, []).append(sg.subgoal_id)
        fee = frozenset(sg.feasible_ee)
        if sg.group_id in group_sets:
            group_sets[sg.group_id] &= fee
        else:
            group_sets[sg.group_id] = fee
    empty_groups = sorted(g for g, s in group_sets.items() if not s)
    if empty_groups:
        outcome.status = PlanStatus.INFEASIBLE_NO_CANDIDATE
        outcome.diagnostics.groups_without_common_ee = empty_groups
        for g in empty_groups:
            reject(
                "group",
                ReasonCode.GROUP_NO_COMMON_EE,
                f"group {g!r} has no common feasible EE across subgoals "
                f"{sorted(groups[g])}",
                group_id=g,
            )
        return outcome

    outcome.predecessors = build_predecessor_map(subgoal_ids, edges)
    outcome.group_feasible_ee = group_sets
    return outcome


def _validate_contract_references(
    request: TaskPlannerRequest, id_set: set[str]
) -> list[Rejection]:
    """Validate every typed Planner-A contract reference against the DAG."""

    contract = request.planner_a.constraints
    condition_id_list = [
        condition.condition_id
        for subgoal in request.planner_a.subgoals
        for condition in (
            *subgoal.preconditions,
            *subgoal.establish,
            *subgoal.destroy,
        )
        if condition.condition_id is not None
    ]
    condition_ids = set(condition_id_list)
    rejections: list[Rejection] = []

    def reject(code: ReasonCode, message: str, **extra: Any) -> None:
        rejections.append(make_rejection("input", code, message, **extra))

    duplicate_condition_ids = sorted(
        condition_id
        for condition_id, count in Counter(condition_id_list).items()
        if count > 1
    )
    if duplicate_condition_ids:
        reject(
            ReasonCode.INVALID_PLANNER_A_CONTRACT,
            f"duplicate condition ids: {duplicate_condition_ids}",
        )

    def require_subgoal(node: str | None, clause: str) -> None:
        if node is not None and node not in id_set:
            reject(
                ReasonCode.UNKNOWN_CONSTRAINT_REFERENCE,
                f"{clause} references unknown subgoal {node!r}",
                subgoal_id=node,
            )

    def require_condition(condition_id: str | None, clause: str) -> None:
        if condition_id is not None and condition_id not in condition_ids:
            reject(
                ReasonCode.UNKNOWN_CONSTRAINT_REFERENCE,
                f"{clause} references unknown condition {condition_id!r}",
                condition_id=condition_id,
            )

    for index, item in enumerate(contract.mutex):
        clause = f"mutex[{index}]"
        if item.a == item.b:
            reject(
                ReasonCode.INVALID_PLANNER_A_CONTRACT,
                f"{clause} cannot relate a subgoal to itself",
                subgoal_id=item.a,
            )
        require_subgoal(item.a, clause)
        require_subgoal(item.b, clause)
        for producer in item.reestablished_by:
            require_subgoal(producer, clause)
        require_condition(item.condition, clause)

    for index, item in enumerate(contract.open_conditions):
        clause = f"open_conditions[{index}]"
        if item.subgoal in item.candidates:
            reject(
                ReasonCode.INVALID_PLANNER_A_CONTRACT,
                f"{clause} cannot use its consumer as a producer",
                subgoal_id=item.subgoal,
            )
        require_subgoal(item.subgoal, clause)
        for producer in item.candidates:
            require_subgoal(producer, clause)
        require_condition(item.condition, clause)

    for index, item in enumerate(contract.disjunctive_threats):
        clause = f"disjunctive_threats[{index}]"
        if item.link[0] == item.link[1] or item.threat in item.link:
            reject(
                ReasonCode.INVALID_PLANNER_A_CONTRACT,
                f"{clause} requires two distinct link endpoints and a distinct threat",
                subgoal_id=item.threat,
            )
        for node in (*item.link, item.threat):
            require_subgoal(node, clause)
        require_condition(item.condition, clause)

    for collection_name, collection in (
        ("deferred_conditions", contract.deferred_conditions),
        ("sg_observation_requests", contract.sg_observation_requests),
    ):
        for index, item in enumerate(collection):
            clause = f"{collection_name}[{index}]"
            require_subgoal(item.subgoal, clause)
            if item.depends_on == item.subgoal:
                reject(
                    ReasonCode.INVALID_PLANNER_A_CONTRACT,
                    f"{clause} cannot depend on its own subgoal",
                    subgoal_id=item.subgoal,
                )
            if item.depends_on not in (None, "motion"):
                require_subgoal(item.depends_on, clause)
            require_condition(item.condition, clause)

    return rejections
