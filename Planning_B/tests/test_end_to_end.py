"""End-to-end: the bottle & plate task from the example fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from planning_b.diagnostics import PlanStatus
from planning_b.cli import main as cli_main
from planning_b.models import (
    CandidateProposal,
    PlannerAOutput,
    PlanningBRequest,
    ResourceCatalog,
)
from planning_b.planner import plan
from planning_b.transitions import P

from conftest import EXAMPLES_DIR


def test_bottle_plate_end_to_end(example_request) -> None:
    result = plan(example_request)
    assert result.status is PlanStatus.SUCCESS
    selected = result.selected_plan
    assert selected is not None

    # Group EE assignments: bottle group on 3F, plate group on vac.
    assert selected.group_ee_assignments == {"G1": "3F", "G2": "vac"}
    for assignment in selected.candidate_assignments:
        expected_ee = "3F" if assignment.group_id == "G1" else "vac"
        assert assignment.ee == expected_ee
        assert assignment.tool is None

    # Precedence inside each group is respected.
    order = selected.subgoal_order
    assert set(order) == {"SG1", "SG2", "SG3", "SG4", "SG5", "SG6"}
    assert order.index("SG1") < order.index("SG2") < order.index("SG3")
    assert order.index("SG4") < order.index("SG5") < order.index("SG6")

    # Two EE switches (2F -> first group EE -> second group EE), no tools.
    cv = selected.cost_vector
    assert (cv.ee_switches, cv.tool_switches) == (2, 0)
    assert cv.motion_cost == 12  # 6x workspace moves + 2x EE-rack moves

    # Explicit EE exchange primitives appear in the step sequence.
    actions = [s.action for s in selected.steps]
    assert actions.count(P.DETACH_EE) == 2
    assert actions.count(P.ATTACH_EE) == 2
    assert selected.action_counts.n_ee_attaches == 2
    assert selected.action_counts.n_tool_picks == 0

    # Steps alternate transitions and subgoal executions; every subgoal step
    # is EXECUTE_SUBGOAL with its candidate recorded.
    subgoal_steps = [s for s in selected.steps if s.kind == "subgoal"]
    assert len(subgoal_steps) == 6
    assert all(s.action == P.EXECUTE_SUBGOAL for s in subgoal_steps)
    assert all(s.candidate_id is not None for s in subgoal_steps)

    # Holding facts are consumed by each place; goal facts contain in_region.
    facts = selected.terminal_state["facts"]
    assert "in_region(obj_bottle_0, region_box)" in facts
    assert "in_region(obj_plate_0, region_box)" in facts
    assert not any(f.startswith("holding(") for f in facts)

    # EE blocks: one block per contiguous EE segment (2F start, then the two
    # group EEs in the searched order).
    block_ees = [b["ee"] for b in selected.ee_blocks if b["subgoals"]]
    assert len(block_ees) == 2
    assert set(block_ees) == {"3F", "vac"}

    # No enumeration of the 20 possible topological orders: expansion count
    # stays small.
    assert result.search_stats.states_expanded < 40


def test_end_to_end_is_deterministic(example_request) -> None:
    first = plan(example_request).to_json()
    second = plan(example_request).to_json()
    # elapsed_ms may differ; normalize before comparing.
    a = json.loads(first)
    b = json.loads(second)
    a["search_stats"]["elapsed_ms"] = b["search_stats"]["elapsed_ms"] = 0
    assert a == b


def test_cli_plan_roundtrip(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    exit_code = cli_main(
        [
            "plan",
            "--planner-a",
            str(EXAMPLES_DIR / "bottle_plate_planner_a.json"),
            "--resources",
            str(EXAMPLES_DIR / "bottle_plate_resources.json"),
            "--candidates",
            str(EXAMPLES_DIR / "bottle_plate_candidates.json"),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "SUCCESS"
    assert payload["planner_version"] == "planning-b-state-space-v3"
    assert payload["selected_plan"]["group_ee_assignments"] == {
        "G1": "3F",
        "G2": "vac",
    }


def test_cli_failure_returns_nonzero(tmp_path: Path) -> None:
    planner_a = json.loads(
        (EXAMPLES_DIR / "bottle_plate_planner_a.json").read_text(encoding="utf-8")
    )
    planner_a["order_constraints"]["cycle_detected"] = True
    bad = tmp_path / "planner_a.json"
    bad.write_text(json.dumps(planner_a), encoding="utf-8")
    output = tmp_path / "result.json"
    exit_code = cli_main(
        [
            "plan",
            "--planner-a",
            str(bad),
            "--resources",
            str(EXAMPLES_DIR / "bottle_plate_resources.json"),
            "--candidates",
            str(EXAMPLES_DIR / "bottle_plate_candidates.json"),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 3
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "INFEASIBLE_REDECOMPOSE"


def test_heavy_crate_uses_planner_a_fixed_tools() -> None:
    planner_a = json.loads(
        (EXAMPLES_DIR / "heavy_crate_planner_a.json").read_text(encoding="utf-8")
    )
    resources = json.loads(
        (EXAMPLES_DIR / "heavy_crate_resources.json").read_text(encoding="utf-8")
    )
    proposals = json.loads(
        (EXAMPLES_DIR / "heavy_crate_candidates.json").read_text(encoding="utf-8")
    )
    request = PlanningBRequest(
        planner_a=PlannerAOutput.model_validate(planner_a),
        resource_catalog=ResourceCatalog.model_validate(resources),
        candidate_proposals={
            subgoal_id: [CandidateProposal.model_validate(item) for item in items]
            for subgoal_id, items in proposals.items()
        },
    )

    result = plan(request)
    assert result.status is PlanStatus.SUCCESS
    selected = result.selected_plan
    assert selected is not None
    assert selected.subgoal_order == ["SG_PULL", "SG_PUSH"]
    assert [assignment.tool for assignment in selected.candidate_assignments] == [
        "pull_hook",
        "push_bar",
    ]
    assert (
        selected.cost_vector.ee_switches,
        selected.cost_vector.tool_switches,
        selected.cost_vector.motion_cost,
        selected.cost_vector.execution_cost,
    ) == (0, 3, 8, 5)
    assert selected.action_counts.n_tool_picks == 2
    assert selected.action_counts.n_tool_returns == 2

    execution_steps = [step for step in selected.steps if step.kind == "subgoal"]
    assert [step.parameters["action_parameters"]["interaction_mode"] for step in execution_steps] == [
        "quasi_static_pull",
        "quasi_static_push",
    ]
    assert "in_region(obj_crate_0, loading_zone)" in selected.terminal_state["facts"]
