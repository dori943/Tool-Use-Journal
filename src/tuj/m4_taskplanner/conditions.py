"""Symbolic execution-state updates for the normalized M2 DAG.

``condition.check`` strings are never executed with ``eval()``. They are either
looked up in a :class:`CheckerRegistry` by name or carried as metadata.

This module does not derive, repair, or replace the upstream ordering contract.
Hard/optional ordering is handled by ``validation.py`` and ``constraints.py``;
the facts here only stop a DAG-valid prefix from using a condition that a
previous action has consumed.
"""

from __future__ import annotations

from typing import Callable, Iterable

from tuj.m4_taskplanner.models import Condition, InitialState, Subgoal

# Normalized fluent identity, e.g. ("holding", ("obj_bottle_0",)).
FluentKey = tuple[str, tuple[str, ...]]

# Checker callables receive the condition and the current symbolic facts and
# return True (satisfied) / False (unsatisfied).
CheckerFn = Callable[[Condition, frozenset[FluentKey]], bool]


class EffectConflictError(ValueError):
    """Raised when the same fluent appears in both establish and destroy."""

    def __init__(self, subgoal_id: str, fluents: list[FluentKey]) -> None:
        self.subgoal_id = subgoal_id
        self.fluents = fluents
        super().__init__(
            f"subgoal {subgoal_id!r} lists the same fluent(s) in both "
            f"establish and destroy: {sorted(fluents)}"
        )


def fluent_key(condition: Condition) -> FluentKey:
    return (condition.type, tuple(condition.args))


def effect_keys(conditions: Iterable[Condition]) -> frozenset[FluentKey]:
    return frozenset(fluent_key(c) for c in conditions)


def validate_subgoal_effects(subgoal: Subgoal) -> None:
    """Reject subgoals whose establish and destroy sets overlap."""
    overlap = effect_keys(subgoal.establish) & effect_keys(subgoal.destroy)
    if overlap:
        raise EffectConflictError(subgoal.subgoal_id, sorted(overlap))


def true_facts(conditions: Iterable[Condition]) -> frozenset[FluentKey]:
    """Normalize conditions whose current upstream judgement is true."""
    return frozenset(
        fluent_key(condition) for condition in conditions if condition.pass_ is True
    )


def initial_facts(initial_state: InitialState) -> frozenset[FluentKey]:
    return true_facts(initial_state.facts)


def apply_effects(
    facts: frozenset[FluentKey],
    destroy: Iterable[Condition],
    establish: Iterable[Condition],
) -> frozenset[FluentKey]:
    """STRIPS update: next = (facts - destroy) | establish."""
    return (facts - effect_keys(destroy)) | effect_keys(establish)


class CheckerRegistry:
    """Named condition checkers (no eval of condition.check strings)."""

    def __init__(self) -> None:
        self._checkers: dict[str, CheckerFn] = {}

    def register(self, name: str, fn: CheckerFn) -> None:
        self._checkers[name] = fn

    def has(self, name: str | None) -> bool:
        return name is not None and name in self._checkers

    def evaluate(
        self, name: str, condition: Condition, facts: frozenset[FluentKey]
    ) -> bool:
        return self._checkers[name](condition, facts)


def requires_runtime_verification(condition: Condition) -> bool:
    """Skip motion/deferred checks in symbolic state; verify them at runtime."""

    return condition.eval_by == "motion" or (
        condition.pass_ is None
        and (
            condition.depends_on not in (None, "motion")
            or condition.needs_observation
        )
    )


def subgoal_required_wrench(subgoal: Subgoal) -> float | None:
    """Largest numeric force/torque requirement declared by a subgoal."""

    required = subgoal.required_wrench
    for condition in (*subgoal.preconditions, *subgoal.postconditions):
        if condition.type in {"force", "torque", "required_wrench"} and isinstance(
            condition.limit, (int, float)
        ):
            required = max(required or 0.0, float(condition.limit))
    return required


def symbolic_precondition_unmet(
    subgoal: Subgoal,
    facts: frozenset[FluentKey],
    registry: CheckerRegistry | None = None,
) -> list[FluentKey]:
    """Return the fluent keys of unmet *symbolic* preconditions.

    Motion-evaluated conditions are skipped here; they are validated lazily by
    the geometric checker at node expansion. All preconditions are positive
    unless ``negated=True`` is set explicitly.
    """
    unmet: list[FluentKey] = []
    for cond in subgoal.preconditions:
        if requires_runtime_verification(cond):
            continue
        if registry is not None and registry.has(cond.check):
            if not registry.evaluate(cond.check, cond, facts):  # type: ignore[arg-type]
                unmet.append(fluent_key(cond))
            continue
        key = fluent_key(cond)
        holds = key in facts
        if cond.negated:
            if holds:
                unmet.append(key)
        elif not holds:
            unmet.append(key)
    return unmet
