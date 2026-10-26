from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from tuj.m2_subgoal.core import (
    add_container_packing_sequence_pres,
    decompose,
    partial_order,
)


def _apply_policy():
    repository = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(str(repository / "scripts" / "run_m2.py"))
    return namespace["apply_task_packing_policy"]


def _payload():
    return {
        "m2_subgoals": [{"details": []}],
        "m2_partial_order": [],
        "m2_mutex": [],
        "m2_stats": {"n_edges": 0, "n_mutex": 0},
    }


def _relocate(subgoal_id, target):
    subgoal = {
        "subgoal_id": subgoal_id,
        "kind": "relocate",
        "target_ids": [target],
        "container_id": "box",
        "tool_candidate_ids": [],
    }
    subgoal["details"] = decompose(subgoal)
    return subgoal


def test_packing_sequence_creates_cross_group_causal_edges():
    rolling = _relocate("SG_roll", "rolling_pin")
    whisk = _relocate("SG_whisk", "whisk")

    logs = add_container_packing_sequence_pres(
        [whisk, rolling], ["rolling_pin", "whisk"]
    )
    edges, _ = partial_order(rolling["details"] + whisk["details"])

    assert len(logs) == 1
    assert {
        "from": "SG_roll_d3",
        "to": "SG_whisk_d1",
        "why": "causal_link: in(rolling_pin, box)",
    } in edges


def test_inline_packing_policy_rejects_unresolved_target_pairs(
    tmp_path, monkeypatch
):
    from tuj.m2_subgoal import core

    monkeypatch.setattr(
        core,
        "add_container_packing_sequence_pres",
        lambda _subgoals, _targets: [],
    )
    policy = tmp_path / "packing.json"
    policy.write_text(
        json.dumps({"target_order": ["milk", "cereal"]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="expected 1, got 0"):
        _apply_policy()(_payload(), policy)


def test_inline_packing_policy_records_only_fully_applied_policy(
    tmp_path, monkeypatch
):
    from tuj.m2_subgoal import core

    monkeypatch.setattr(
        core,
        "add_container_packing_sequence_pres",
        lambda _subgoals, _targets: ["milk -> cereal"],
    )
    monkeypatch.setattr(core, "partial_order", lambda _details: ([{"from": "a"}], []))
    policy = tmp_path / "packing.json"
    policy.write_text(
        json.dumps({"target_order": ["milk", "cereal"]}), encoding="utf-8"
    )
    payload = _payload()

    logs = _apply_policy()(payload, policy)

    assert logs == ["milk -> cereal"]
    assert payload["m2_packing_policy"]["target_order"] == ["milk", "cereal"]
    assert payload["m2_stats"]["n_edges"] == 1


def _tool_move(subgoal_id, target, kind="sweep_collect"):
    subgoal = {
        "subgoal_id": subgoal_id, "kind": kind,
        "target_ids": [target], "container_id": "box",
        "tool_candidate_ids": ["tool"], "selected_tool_id": "tool",
    }
    subgoal["details"] = decompose(subgoal)
    return subgoal


@pytest.mark.parametrize("kind", ["sweep_collect", "scoop_transfer"])
def test_packing_sequence_supports_tool_effects_and_keeps_actual_expression(kind):
    before = _tool_move("tool_move", "cereal", kind)
    after = _relocate("direct_move", "milk")
    logs = add_container_packing_sequence_pres([after, before], ["cereal", "milk"])
    edges, _ = partial_order(before["details"] + after["details"])
    assert len(logs) == 1
    assert {"from": "tool_move_d2", "to": "direct_move_d1",
            "why": "causal_link: in({cereal}, box)"} in edges
    assert after["details"][0]["pre"][-1]["expr"] == "in({cereal}, box)"


def test_packing_sequence_reorders_shared_tool_actions_without_reacquiring():
    from copy import deepcopy
    from tuj.m2_subgoal.regroup import _split_sweep, _cycle_check

    parent = _tool_move("sweep", "first")
    parent["target_ids"] = ["first", "second", "third"]
    children = _split_sweep(parent, [["first"], ["second"], ["third"]])
    order = ["third", "second", "first"]
    logs = add_container_packing_sequence_pres(children, order)
    details = [d for child in children for d in child["details"]]
    edges, _ = partial_order(details)
    assert len(logs) == 2
    assert not _cycle_check(details, edges)
    assert {"from": "sweep_s3_d2", "to": "sweep_s2_d2",
            "why": "causal_link: in({third}, box)"} in edges
    assert {"from": "sweep_s2_d2", "to": "sweep_s1_d2",
            "why": "causal_link: in({second}, box)"} in edges
    acquire = next(d for d in details if d["action_type"] == "acquire")
    assert not any(p.get("auto") == "container_packing_sequence" for p in acquire["pre"])
    assert sum(d["action_type"] == "acquire" for d in details) == 1
    # Reapplying policy neither duplicates conditions nor changes the tool flow.
    snapshot = deepcopy(children)
    add_container_packing_sequence_pres(children, order)
    assert children == snapshot


@pytest.mark.parametrize("bad_case, message", [
    ("missing", "MISSING_TARGET"),
    ("duplicate_target", "AMBIGUOUS_TARGET"),
    ("duplicate_order", "INVALID_TARGET_ORDER"),
    ("wrong_destination", "DESTINATION_MISMATCH"),
    ("wrong_effect_destination", "INVALID_EFFECT"),
    ("unresolved_tool", "UNRESOLVED_TOOL"),
    ("undeclared_tool", "UNRESOLVED_TOOL"),
    ("cycle", "CYCLE"),
])
def test_invalid_packing_plan_is_rejected_atomically(bad_case, message):
    from copy import deepcopy

    before = _tool_move("before", "cereal")
    after = _relocate("after", "milk")
    subgoals = [before, after]
    order = ["cereal", "milk"]
    if bad_case == "missing":
        order.append("missing")
    elif bad_case == "duplicate_target":
        subgoals.append(_relocate("again", "milk"))
    elif bad_case == "duplicate_order":
        order.append("milk")
    elif bad_case == "wrong_destination":
        after["container_id"] = "other_box"
        after["details"] = decompose(after)
    elif bad_case == "wrong_effect_destination":
        before["details"][1]["establish"][0]["expr"] = "in({cereal}, other_box)"
    elif bad_case == "unresolved_tool":
        before.pop("selected_tool_id")
    elif bad_case == "undeclared_tool":
        before["selected_tool_id"] = "unknown_tool"
    elif bad_case == "cycle":
        before["details"][0]["pre"].append({
            "id": "prior_order", "head": "in", "expr": "in(milk, box)",
            "eval_by": "m2", "auto": "instruction_order",
        })
    snapshot = deepcopy(subgoals)
    with pytest.raises(ValueError, match=f"PACKING_SEQUENCE_{message}"):
        add_container_packing_sequence_pres(subgoals, order)
    assert subgoals == snapshot
