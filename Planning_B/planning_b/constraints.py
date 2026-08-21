"""Planner-A contract enforcement without changing its immutable hard DAG.

The hard predecessor edges remain in ``validation.py``/``search.py``.  This
module handles only the choices Planner A intentionally leaves to Task Planner:
multi-producer open conditions, resource mutexes, and disjunctive threats.
Symbolic fluent truth is still maintained by ``conditions.py``; it is not used
to infer a new DAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from planning_b.conditions import FluentKey
from planning_b.models import PlannerAConstraints


@dataclass(frozen=True, slots=True)
class ConstraintBlocker:
    """Why a subgoal cannot be appended to the current search prefix."""

    kind: str
    constraint_id: str
    message: str


class PlannerAConstraintEngine:
    """Evaluate Planner-A choices against one concrete search prefix."""

    def __init__(
        self,
        contract: PlannerAConstraints,
    ) -> None:
        self.contract = contract

    def blockers(
        self,
        subgoal_id: str,
        completed: frozenset[str],
        facts: frozenset[FluentKey],
    ) -> tuple[ConstraintBlocker, ...]:
        """Return contract clauses that reject ``subgoal_id`` at this prefix."""

        blocked: list[ConstraintBlocker] = []

        for item in self.contract.open_conditions:
            if item.subgoal != subgoal_id:
                continue
            producers = sorted(completed.intersection(item.candidates))
            if not producers:
                blocked.append(
                    ConstraintBlocker(
                        "open_condition",
                        _open_id(item.subgoal, item.condition),
                        f"none of the producer choices {sorted(item.candidates)!r} "
                        "has completed",
                    )
                )

        for item in self.contract.mutex:
            if subgoal_id not in (item.a, item.b):
                continue
            other = item.b if subgoal_id == item.a else item.a
            if other not in completed:
                continue
            key = (item.predicate, tuple(item.args))
            if key not in facts:
                blocked.append(
                    ConstraintBlocker(
                        "mutex",
                        _mutex_id(item.a, item.b, item.condition, item.predicate),
                        f"{item.predicate}{tuple(item.args)!r} was consumed by "
                        f"{other!r} and has not been reestablished",
                    )
                )

        for item in self.contract.disjunctive_threats:
            producer, consumer = item.link
            if (
                subgoal_id == item.threat
                and producer in completed
                and consumer not in completed
            ):
                blocked.append(
                    ConstraintBlocker(
                        "disjunctive_threat",
                        _threat_id(producer, consumer, item.threat, item.condition),
                        f"threat {item.threat!r} cannot occur inside causal link "
                        f"{producer!r}->{consumer!r}; choose promotion or demotion",
                    )
                )

        # Duplicate mutex records can refer to the same pair/predicate. Keep a
        # stable single explanation per concrete clause.
        return tuple(
            sorted(set(blocked), key=lambda item: (item.kind, item.constraint_id))
        )


def build_constraint_resolution_trace(
    order: Sequence[str],
    contract: PlannerAConstraints,
    hard_edges: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
    """Explain how the selected total order satisfies every A contract clause."""

    position = {subgoal_id: index for index, subgoal_id in enumerate(order)}
    trace: list[dict[str, object]] = []

    for pred, succ in sorted(set(hard_edges)):
        if pred not in position or succ not in position:
            edge_status = "prefix_validated"
        else:
            edge_status = (
                "satisfied" if position[pred] < position[succ] else "violated"
            )
        trace.append(
            _trace_record(
                "hard_dag",
                f"edge:{pred}->{succ}",
                edge_status,
                f"{pred} before {succ}",
                [pred, succ],
                {"immutable": True},
            )
        )

    for item in contract.mutex:
        if item.a not in position or item.b not in position:
            first, second = item.a, item.b
            between: list[str] = []
            mutex_status = "prefix_validated"
        else:
            first, second = sorted(
                (item.a, item.b), key=lambda node: position[node]
            )
            between = [
                node
                for node in item.reestablished_by
                if position[first] < position.get(node, -1) < position[second]
            ]
            mutex_status = "satisfied" if between else "state_validated"
        trace.append(
            _trace_record(
                "mutex",
                _mutex_id(item.a, item.b, item.condition, item.predicate),
                mutex_status,
                f"{first} before {second}",
                [item.a, item.b, *between],
                {
                    "predicate": item.predicate,
                    "args": list(item.args),
                    "reestablished_between": between,
                },
            )
        )

    for item in contract.open_conditions:
        if item.subgoal not in position:
            selected = None
            open_status = "prefix_validated"
        else:
            target_position = position[item.subgoal]
            available = [
                node
                for node in item.candidates
                if position.get(node, target_position + 1) < target_position
            ]
            selected = (
                max(available, key=lambda node: position[node]) if available else None
            )
            open_status = "satisfied" if selected is not None else "violated"
        trace.append(
            _trace_record(
                "open_condition",
                _open_id(item.subgoal, item.condition),
                open_status,
                f"producer={selected}" if selected is not None else None,
                [item.subgoal, *item.candidates],
                {"selected_producer": selected, "candidates": list(item.candidates)},
            )
        )

    for item in contract.disjunctive_threats:
        producer, consumer = item.link
        if any(node not in position for node in (producer, consumer, item.threat)):
            option = None
            status = "prefix_validated"
        elif position[item.threat] < position[producer]:
            option = "promotion"
            status = "satisfied"
        elif position[consumer] < position[item.threat]:
            option = "demotion"
            status = "satisfied"
        else:
            option = None
            status = "violated"
        trace.append(
            _trace_record(
                "disjunctive_threat",
                _threat_id(producer, consumer, item.threat, item.condition),
                status,
                option,
                [producer, consumer, item.threat],
                {"causal_link": [producer, consumer], "threat": item.threat},
            )
        )

    for item in contract.deferred_conditions:
        trace.append(
            _trace_record(
                "deferred_condition",
                f"deferred:{item.subgoal}:{item.condition}",
                "runtime_verification",
                (
                    f"verify after {item.depends_on}"
                    if item.depends_on is not None
                    else "verify at execution"
                ),
                [node for node in (item.depends_on, item.subgoal) if node],
                {
                    "condition": item.condition,
                    "needs_observation": item.needs_observation,
                },
            )
        )

    for item in contract.sg_observation_requests:
        trace.append(
            _trace_record(
                "observation_request",
                f"observation:{item.subgoal}:{item.condition}",
                "scheduled",
                item.request or (
                    f"observe after {item.depends_on}"
                    if item.depends_on is not None
                    else "observe at execution"
                ),
                [node for node in (item.depends_on, item.subgoal) if node],
                {"condition": item.condition, "args": list(item.args)},
            )
        )

    return trace


def _trace_record(
    constraint_type: str,
    constraint_id: str,
    status: str,
    selected_option: str | None,
    involved_subgoals: list[str],
    details: dict[str, object],
) -> dict[str, object]:
    return {
        "constraint_type": constraint_type,
        "constraint_id": constraint_id,
        "status": status,
        "selected_option": selected_option,
        "involved_subgoals": involved_subgoals,
        "details": details,
    }


def _mutex_id(
    a: str, b: str, condition: str | None, predicate: str
) -> str:
    return f"mutex:{a}:{b}:{condition or predicate}"


def _open_id(subgoal: str, condition: str) -> str:
    return f"open:{subgoal}:{condition}"


def _threat_id(producer: str, consumer: str, threat: str, condition: str) -> str:
    return f"threat:{producer}:{consumer}:{threat}:{condition}"
