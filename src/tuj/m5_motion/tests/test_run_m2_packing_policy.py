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
