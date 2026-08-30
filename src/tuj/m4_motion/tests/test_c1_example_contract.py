"""C1_1 example compatibility with the current M3 SelectedPlan schema."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tuj.m4_motion.examples.c1_1_gk_task_preview import _selected_resources
from tuj.m4_motion.examples.c1_1_openai_motion_run import (
    _selected_binding,
    _selected_bindings,
)


def _assignment(
    action_type: str,
    *,
    subgoal_id: str,
    target_ids: list[str] | None = None,
    goal_region_id: str | None = None,
    tool: str = "light_plate",
) -> SimpleNamespace:
    return SimpleNamespace(
        action_type=action_type,
        subgoal_id=subgoal_id,
        ee="2F",
        tool=tool,
        target_ids=target_ids or [],
        goal_region_id=goal_region_id,
    )


def test_c1_binding_reads_candidate_assignments() -> None:
    selected = SimpleNamespace(
        candidate_assignments=[
            _assignment("acquire", subgoal_id="SG1_d1"),
            _assignment(
                "tool_act",
                subgoal_id="SG1_d2",
                target_ids=["block_0", "block_1"],
                goal_region_id="collection_zone_visual",
            ),
        ]
    )

    sweep = _selected_binding(
        selected,
        action_type="tool_act",
        label="sweep",
    )

    assert sweep.subgoal_id == "SG1_d2"
    assert sweep.ee == "2F"
    assert sweep.tool == "light_plate"
    assert sweep.target_ids == ("block_0", "block_1")
    assert sweep.goal_region_id == "collection_zone_visual"


def test_c1_binding_rejects_missing_assignment() -> None:
    selected = SimpleNamespace(candidate_assignments=[])

    with pytest.raises(RuntimeError, match="at least one sweep assignment"):
        _selected_binding(
            selected,
            action_type="tool_act",
            label="sweep",
        )


def test_c1_binding_accepts_split_sweeps_from_gk_bundle() -> None:
    selected = SimpleNamespace(
        candidate_assignments=[
            _assignment("acquire", subgoal_id="SG1_s1_d1"),
            _assignment(
                "tool_act",
                subgoal_id="SG1_s1_d2",
                target_ids=["block_0", "block_1"],
                goal_region_id="collection_zone_visual",
            ),
            _assignment(
                "tool_act",
                subgoal_id="SG1_s2_d2",
                target_ids=["block_2"],
                goal_region_id="collection_zone_visual",
            ),
        ]
    )

    sweeps = _selected_bindings(
        selected,
        action_type="tool_act",
        label="sweep",
    )

    assert [item.subgoal_id for item in sweeps] == [
        "SG1_s1_d2",
        "SG1_s2_d2",
    ]
    assert {item.tool for item in sweeps} == {"light_plate"}


def test_preview_reads_candidate_assignments_without_heavy_plate_default() -> None:
    payload = {
        "status": "SUCCESS",
        "selected_plan": {
            "candidate_assignments": [
                {"ee": "2F", "tool": "light_plate"},
                {"ee": "2F", "tool": "light_plate"},
            ]
        },
    }

    assert _selected_resources(payload) == ("2F", "light_plate")
