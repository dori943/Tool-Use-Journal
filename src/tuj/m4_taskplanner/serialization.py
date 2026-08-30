"""JSON-serializable result models and I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tuj.m4_taskplanner.cost import CostVector
from tuj.m4_taskplanner.diagnostics import Diagnostics, PlanStatus, Rejection
from tuj.m4_taskplanner.models import GraspSpec

PLANNER_VERSION = "task-planner-state-space-v3"

OPTIMALITY_SCOPE = {
    "candidate_set": "after threshold, top-k, and hard pruning",
    "search": "lexicographically optimal among retained feasible candidates",
}


class CostVectorModel(BaseModel):
    ee_switches: int = 0
    tool_switches: int = 0
    motion_cost: int = 0
    execution_cost: int = 0

    @classmethod
    def from_cost(cls, cost: CostVector) -> "CostVectorModel":
        return cls(**cost.to_dict())


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    step_index: int
    kind: str  # "transition" | "subgoal"
    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    subgoal_id: str | None = None
    candidate_id: str | None = None
    preconditions: list[Any] = Field(default_factory=list)
    postconditions: list[Any] = Field(default_factory=list)
    cost_delta: dict[str, int] = Field(default_factory=dict)
    verification_required: bool = False


class CandidateAssignment(BaseModel):
    subgoal_id: str
    candidate_id: str
    group_id: str | None = None
    ee: str
    tool: str | None = None
    action_type: str | None = None
    mode: str | None = None
    target_ids: list[str] = Field(default_factory=list)
    goal_region_id: str | None = None
    description: str | None = None
    grasp_id: str | None = None
    grasp: GraspSpec | None = None
    action_parameters: dict[str, Any] = Field(default_factory=dict)
    source_binding: dict[str, Any] = Field(default_factory=dict)
    suitability_score: float | None = None
    suitability: dict[str, Any] | None = None
    ee_selection_source: str = "task_planner"
    ee_feasible_set_source: str = "request"
    tool_selection_source: str = "not_required"


class ConstraintResolution(BaseModel):
    """How one task-graph constraint was handled in the selected order."""

    constraint_type: str
    constraint_id: str
    status: str
    selected_option: str | None = None
    involved_subgoals: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ActionCounts(BaseModel):
    n_tool_picks: int = 0
    n_tool_returns: int = 0
    n_ee_detaches: int = 0
    n_ee_attaches: int = 0


class SelectedPlan(BaseModel):
    model_config = ConfigDict(extra="allow")

    cost_vector: CostVectorModel
    subgoal_order: list[str] = Field(default_factory=list)
    group_ee_assignments: dict[str, str] = Field(default_factory=dict)
    candidate_assignments: list[CandidateAssignment] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    ee_blocks: list[dict[str, Any]] = Field(default_factory=list)
    tool_blocks: list[dict[str, Any]] = Field(default_factory=list)
    action_counts: ActionCounts = Field(default_factory=ActionCounts)
    terminal_state: dict[str, Any] = Field(default_factory=dict)
    constraint_trace: list[ConstraintResolution] = Field(default_factory=list)


class SearchStatsModel(BaseModel):
    states_popped: int = 0
    states_expanded: int = 0
    edges_generated: int = 0
    dominated_states_pruned: int = 0
    static_candidates_rejected: int = 0
    lazy_checks_executed: int = 0
    lazy_cache_hits: int = 0
    elapsed_ms: int = 0


class ReplanTrace(BaseModel):
    model_config = ConfigDict(extra="allow")

    previous_subgoal_id: str | None = None
    previous_candidate_id: str | None = None
    failure_reason: str | None = None
    added_no_goods: list[str] = Field(default_factory=list)
    newly_selected_candidate_id: str | None = None
    new_cost_vector: CostVectorModel | None = None


class PlanningResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: PlanStatus
    task: dict[str, Any] = Field(default_factory=dict)
    planner_version: str = PLANNER_VERSION
    optimality_scope: dict[str, str] = Field(
        default_factory=lambda: dict(OPTIMALITY_SCOPE)
    )
    selected_plan: SelectedPlan | None = None
    search_stats: SearchStatsModel = Field(default_factory=SearchStatsModel)
    rejections: list[Rejection] = Field(default_factory=list)
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)
    replan_trace: ReplanTrace | None = None

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.model_dump(mode="json", by_alias=True),
            indent=indent,
            sort_keys=False,
        )


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_result(result: PlanningResult, path: str | Path) -> None:
    Path(path).write_text(result.to_json(), encoding="utf-8")
