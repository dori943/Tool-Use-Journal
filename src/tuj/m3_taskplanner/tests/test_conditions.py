"""Fluent semantics: pass field, STRIPS updates, negation, alias parsing."""

from __future__ import annotations

import pytest

from tuj.m3_taskplanner.conditions import (
    EffectConflictError,
    apply_effects,
    fluent_key,
    initial_facts,
    symbolic_precondition_unmet,
    validate_subgoal_effects,
)
from tuj.m3_taskplanner.models import Condition, InitialState

from conftest import cond, sg


def test_fluent_key_normalization() -> None:
    c = Condition(type="holding", args=["obj_bottle_0"])
    assert fluent_key(c) == ("holding", ("obj_bottle_0",))


def test_pass_alias_parses_from_json_key() -> None:
    c = Condition.model_validate({"type": "holding", "args": ["x"], "pass": False})
    assert c.pass_ is False
    # pass=false is NOT a negative precondition; the fluent is just not yet true.
    assert c.negated is False


def test_initial_facts_keep_only_pass_true() -> None:
    state = InitialState(
        current_ee="A",
        facts=[
            cond("clear", "region", pass_=True),
            cond("holding", "obj1", pass_=False),
            cond("unknown", "x"),
        ],
    )
    assert initial_facts(state) == frozenset({("clear", ("region",))})


def test_apply_effects_is_strips() -> None:
    facts = frozenset({("holding", ("obj1",)), ("clear", ("r",))})
    nxt = apply_effects(
        facts,
        destroy=[cond("holding", "obj1")],
        establish=[cond("in_region", "obj1", "r")],
    )
    assert nxt == frozenset({("clear", ("r",)), ("in_region", ("obj1", "r"))})


def test_effect_conflict_raises() -> None:
    subgoal = sg(
        "S1",
        estab=[cond("holding", "obj1")],
        destroy=[cond("holding", "obj1")],
    )
    with pytest.raises(EffectConflictError):
        validate_subgoal_effects(subgoal)


def test_precondition_pass_false_satisfied_after_establish() -> None:
    subgoal = sg("S2", pre=[cond("holding", "obj1", pass_=False)])
    assert symbolic_precondition_unmet(subgoal, frozenset()) == [
        ("holding", ("obj1",))
    ]
    facts_after = apply_effects(
        frozenset(), destroy=[], establish=[cond("holding", "obj1")]
    )
    assert symbolic_precondition_unmet(subgoal, facts_after) == []


def test_negated_precondition() -> None:
    subgoal = sg("S1", pre=[cond("blocked", "r", negated=True)])
    assert symbolic_precondition_unmet(subgoal, frozenset()) == []
    assert symbolic_precondition_unmet(
        subgoal, frozenset({("blocked", ("r",))})
    ) == [("blocked", ("r",))]


def test_motion_conditions_are_not_checked_symbolically() -> None:
    subgoal = sg("S1", pre=[cond("reachable", "obj1", eval_by="motion")])
    assert symbolic_precondition_unmet(subgoal, frozenset()) == []
