"""Shared builders for Task Planner tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pytest

from task_planner.models import (
    CandidateProposal,
    Condition,
    InitialState,
    OrderConstraints,
    PlannerAOutput,
    TaskPlannerRequest,
    PlanningPolicy,
    ResourceCatalog,
    Subgoal,
    TaskSpec,
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def cond(ctype: str, *args: str, **kwargs: Any) -> Condition:
    return Condition(type=ctype, args=list(args), **kwargs)


def sg(
    subgoal_id: str,
    *,
    group: str | None = None,
    targets: Iterable[str] = (),
    tool_id: str | None = None,
    feasible: Iterable[str] = ("A",),
    pre: Iterable[Condition] = (),
    post: Iterable[Condition] = (),
    estab: Iterable[Condition] = (),
    destroy: Iterable[Condition] = (),
    action: str = "acquire",
    unique: bool = False,
    tool_required: bool = False,
) -> Subgoal:
    return Subgoal(
        subgoal_id=subgoal_id,
        action_type=action,
        group_id=group,
        target_ids=list(targets),
        tool_id=tool_id,
        preconditions=list(pre),
        postconditions=list(post),
        establish=list(estab),
        destroy=list(destroy),
        feasible_ee=list(feasible),
        unique_solution=unique,
        tool_required=tool_required,
    )


def prop(
    candidate_id: str,
    subgoal_id: str,
    ee: str,
    *,
    tool: str | None = None,
    score: float | None = 0.9,
    source: str = "manual",
    **kwargs: Any,
) -> CandidateProposal:
    return CandidateProposal(
        candidate_id=candidate_id,
        subgoal_id=subgoal_id,
        ee=ee,
        tool=tool,
        suitability_score=score,
        source=source,  # type: ignore[arg-type]
        **kwargs,
    )


def base_catalog() -> ResourceCatalog:
    """Two EEs and two Planner-A-selected tools."""
    return ResourceCatalog.model_validate(
        {
            "end_effectors": {
                "A": {
                    "capabilities": ["grip", "tool_holding"],
                    "payload": 5.0,
                    "compatible_tools": ["t1", "t2"],
                    "home_slot": "SA",
                },
                "B": {
                    "capabilities": ["grip", "tool_holding", "suction"],
                    "payload": 5.0,
                    "compatible_tools": ["t1", "t2"],
                    "home_slot": "SB",
                },
            },
            "tools": {
                "t1": {
                    "mass": 0.5,
                    "required_capabilities": ["tool_holding"],
                    "compatible_ee": ["A", "B"],
                    "home_slot": "T1",
                },
                "t2": {
                    "mass": 0.4,
                    "required_capabilities": ["tool_holding"],
                    "compatible_ee": ["A", "B"],
                    "home_slot": "T2",
                },
            },
        }
    )


def make_request(
    subgoals: list[Subgoal],
    *,
    edges: Iterable[tuple[str, str]] = (),
    proposals: dict[str, list[CandidateProposal]] | None = None,
    catalog: ResourceCatalog | None = None,
    initial_ee: str = "A",
    policy: PlanningPolicy | None = None,
    facts: Iterable[Condition] = (),
    order_kwargs: dict[str, Any] | None = None,
    initial_kwargs: dict[str, Any] | None = None,
) -> TaskPlannerRequest:
    return TaskPlannerRequest(
        planner_a=PlannerAOutput(
            task=TaskSpec(instruction="test task"),
            initial_state=InitialState(
                current_ee=initial_ee,
                facts=list(facts),
                **(initial_kwargs or {}),
            ),
            subgoals=subgoals,
            order_constraints=OrderConstraints(
                edges=[list(e) for e in edges], **(order_kwargs or {})
            ),
        ),
        resource_catalog=catalog or base_catalog(),
        candidate_proposals=proposals,
        planning_policy=policy or PlanningPolicy(),
    )


@pytest.fixture()
def example_request() -> TaskPlannerRequest:
    planner_a = json.loads(
        (EXAMPLES_DIR / "bottle_plate_planner_a.json").read_text(encoding="utf-8")
    )
    resources = json.loads(
        (EXAMPLES_DIR / "bottle_plate_resources.json").read_text(encoding="utf-8")
    )
    candidates = json.loads(
        (EXAMPLES_DIR / "bottle_plate_candidates.json").read_text(encoding="utf-8")
    )
    return TaskPlannerRequest(
        planner_a=PlannerAOutput.model_validate(planner_a),
        resource_catalog=ResourceCatalog.model_validate(resources),
        candidate_proposals={
            sg_id: [CandidateProposal.model_validate(p) for p in items]
            for sg_id, items in candidates.items()
        },
    )
