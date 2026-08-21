"""Statuses, structured rejection reason codes, and diagnostics models."""

from __future__ import annotations

import enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PlanStatus(str, enum.Enum):
    SUCCESS = "SUCCESS"
    INVALID_INPUT = "INVALID_INPUT"
    INFEASIBLE_REDECOMPOSE = "INFEASIBLE_REDECOMPOSE"
    INFEASIBLE_NO_CANDIDATE = "INFEASIBLE_NO_CANDIDATE"
    INFEASIBLE_NO_PLAN = "INFEASIBLE_NO_PLAN"
    SEARCH_LIMIT_REACHED = "SEARCH_LIMIT_REACHED"


class ReasonCode(str, enum.Enum):
    # --- input validation ---------------------------------------------------
    CYCLE_DETECTED = "CYCLE_DETECTED"
    UNESTABLISHABLE_PRECONDITIONS = "UNESTABLISHABLE_PRECONDITIONS"
    UNKNOWN_PREDECESSOR = "UNKNOWN_PREDECESSOR"
    DUPLICATE_SUBGOAL_ID = "DUPLICATE_SUBGOAL_ID"
    UNIQUE_SOLUTION_VIOLATION = "UNIQUE_SOLUTION_VIOLATION"
    EFFECT_CONFLICT = "EFFECT_CONFLICT"
    MALFORMED_EDGE = "MALFORMED_EDGE"
    EMPTY_FEASIBLE_EE = "EMPTY_FEASIBLE_EE"
    GROUP_NO_COMMON_EE = "GROUP_NO_COMMON_EE"
    INVALID_RESOURCE_STATE = "INVALID_RESOURCE_STATE"
    INVALID_COMPLETED_PREFIX = "INVALID_COMPLETED_PREFIX"
    UNKNOWN_GROUP_BINDING = "UNKNOWN_GROUP_BINDING"
    DUPLICATE_CANDIDATE_ID = "DUPLICATE_CANDIDATE_ID"
    INVALID_PLANNER_A_CONTRACT = "INVALID_PLANNER_A_CONTRACT"
    UNKNOWN_CONSTRAINT_REFERENCE = "UNKNOWN_CONSTRAINT_REFERENCE"
    # --- inventory / static feasibility -------------------------------------
    UNKNOWN_EE = "UNKNOWN_EE"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    EE_NOT_FEASIBLE_FOR_SUBGOAL = "EE_NOT_FEASIBLE_FOR_SUBGOAL"
    GROUP_EE_CONFLICT = "GROUP_EE_CONFLICT"
    CAPABILITY_MISMATCH = "CAPABILITY_MISMATCH"
    EE_TOOL_INCOMPATIBLE = "EE_TOOL_INCOMPATIBLE"
    PAYLOAD_EXCEEDED = "PAYLOAD_EXCEEDED"
    WRENCH_INSUFFICIENT = "WRENCH_INSUFFICIENT"
    HOME_SLOT_MISSING = "HOME_SLOT_MISSING"
    TOOL_REQUIRED_MISSING = "TOOL_REQUIRED_MISSING"
    TOOL_MISMATCH = "TOOL_MISMATCH"
    SUBGOAL_MISMATCH = "SUBGOAL_MISMATCH"
    UNKNOWN_FEASIBILITY_REJECTED = "UNKNOWN_FEASIBILITY_REJECTED"
    UNKNOWN_SUITABILITY_REJECTED = "UNKNOWN_SUITABILITY_REJECTED"
    # --- candidate scoring / pruning ----------------------------------------
    MISSING_SCORE = "MISSING_SCORE"
    SCORE_BELOW_THRESHOLD = "SCORE_BELOW_THRESHOLD"
    TOP_K_PRUNED = "TOP_K_PRUNED"
    NO_CANDIDATES = "NO_CANDIDATES"
    # --- lazy geometric checks ----------------------------------------------
    IK_UNREACHABLE = "IK_UNREACHABLE"
    GEOMETRY_COLLISION = "GEOMETRY_COLLISION"
    GEOMETRY_FAILED = "GEOMETRY_FAILED"
    GEOMETRY_UNKNOWN_REJECTED = "GEOMETRY_UNKNOWN_REJECTED"
    # --- transitions ---------------------------------------------------------
    OBJECT_HELD_EE_SWITCH = "OBJECT_HELD_EE_SWITCH"
    OBJECT_HELD_TOOL_ACQUIRE = "OBJECT_HELD_TOOL_ACQUIRE"
    RACK_SLOT_UNAVAILABLE = "RACK_SLOT_UNAVAILABLE"
    TRANSITION_BANNED = "TRANSITION_BANNED"
    CANDIDATE_BANNED = "CANDIDATE_BANNED"
    # --- terminal -------------------------------------------------------------
    TERMINAL_OBJECT_HELD = "TERMINAL_OBJECT_HELD"
    # --- search ---------------------------------------------------------------
    SEARCH_LIMIT = "SEARCH_LIMIT"
    FRONTIER_EXHAUSTED = "FRONTIER_EXHAUSTED"


RejectionScope = Literal[
    "input", "subgoal", "group", "candidate", "transition", "terminal", "search"
]


class Rejection(BaseModel):
    model_config = ConfigDict(extra="allow")

    scope: RejectionScope
    reason_code: ReasonCode
    message: str = ""
    subgoal_id: str | None = None
    candidate_id: str | None = None
    group_id: str | None = None


class UnmetPrecondition(BaseModel):
    subgoal_id: str
    fluent: str


class ConstraintBlockerDetail(BaseModel):
    subgoal_id: str
    constraint_type: str
    constraint_id: str
    message: str


class Diagnostics(BaseModel):
    model_config = ConfigDict(extra="allow")

    blocked_subgoals: list[str] = Field(default_factory=list)
    groups_without_common_ee: list[str] = Field(default_factory=list)
    unmet_preconditions: list[UnmetPrecondition] = Field(default_factory=list)
    constraint_blockers: list[ConstraintBlockerDetail] = Field(default_factory=list)
    frontier_exhausted: bool = False
    notes: list[str] = Field(default_factory=list)


def fluent_str(key: tuple[str, tuple[str, ...]]) -> str:
    name, args = key
    return f"{name}({', '.join(args)})"


def make_rejection(
    scope: RejectionScope,
    reason_code: ReasonCode,
    message: str = "",
    *,
    subgoal_id: str | None = None,
    candidate_id: str | None = None,
    group_id: str | None = None,
    **extra: Any,
) -> Rejection:
    return Rejection(
        scope=scope,
        reason_code=reason_code,
        message=message,
        subgoal_id=subgoal_id,
        candidate_id=candidate_id,
        group_id=group_id,
        **extra,
    )
