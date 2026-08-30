"""C1_1 example compatibility with the current M4 SelectedPlan schema."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tuj.m5_motion.examples.c1_1_gk_task_preview import _selected_resources
from tuj.m5_motion.examples.c1_1_openai_motion_run import (
    _selected_sweep_binding,
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
        candidate_id=f"{subgoal_id}-candidate",
        action_type=action_type,
        subgoal_id=subgoal_id,
        mode=("sweep" if action_type == "tool_act" else None),
        ee="2F",
        tool=tool,
        target_ids=target_ids or [],
        goal_region_id=goal_region_id,
        grasp=None,
    )


def _step(assignment: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        kind="subgoal",
        action="EXECUTE_SUBGOAL",
        candidate_id=assignment.candidate_id,
        subgoal_id=assignment.subgoal_id,
        preconditions=[],
        postconditions=[],
    )


def test_c1_binding_reads_candidate_assignments() -> None:
    acquire = _assignment("acquire", subgoal_id="SG1_d1")
    sweep_assignment = _assignment(
        "tool_act",
        subgoal_id="SG1_d2",
        target_ids=["block_0", "block_1"],
        goal_region_id="collection_zone_visual",
    )
    selected = SimpleNamespace(
        candidate_assignments=[acquire, sweep_assignment],
        steps=[_step(sweep_assignment)],
    )

    sweep = _selected_sweep_binding(selected)

    assert sweep.subgoal_id == "SG1_d2"
    assert sweep.ee == "2F"
    assert sweep.tool == "light_plate"
    assert sweep.target_ids == ("block_0", "block_1")
    assert sweep.goal_region_id == "collection_zone_visual"


def test_c1_binding_rejects_missing_assignment() -> None:
    selected = SimpleNamespace(candidate_assignments=[], steps=[])

    with pytest.raises(RuntimeError, match="at least one sweep subgoal"):
        _selected_sweep_binding(selected)


def test_c1_binding_accepts_split_sweeps_from_gk_bundle() -> None:
    acquire = _assignment("acquire", subgoal_id="SG1_s1_d1")
    sweep_assignments = [
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
    selected = SimpleNamespace(
        candidate_assignments=[acquire, *sweep_assignments],
        steps=[_step(item) for item in sweep_assignments],
    )

    sweep = _selected_sweep_binding(selected)

    assert sweep.subgoal_id == "SG1_d2"
    assert sweep.target_ids == ("block_0", "block_1", "block_2")
    assert sweep.tool == "light_plate"


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
