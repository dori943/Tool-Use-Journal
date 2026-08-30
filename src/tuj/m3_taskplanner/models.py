"""Pydantic v2 models for the normalized GK + M1 task graph."""

from __future__ import annotations

import enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ExtensibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class _AliasedModel(_ExtensibleModel):
    model_config = ConfigDict(populate_by_name=True)


class Condition(_AliasedModel):
    """A symbolic (or motion-evaluated) condition.

    ``pass`` (aliased to ``pass_``) is the *current judgement* of the condition
    in the upstream M1 judgement. ``pass=False`` does NOT mean a negative
    precondition; it means the positive fluent does not hold yet and may be
    established later.
    Explicit negative preconditions use ``negated=True``.
    """

    condition_id: str | None = Field(default=None, alias="cond_id")
    type: str
    args: list[str] = Field(default_factory=list)
    value: Any = None
    limit: Any = None
    check: str | None = None
    pass_: bool | None = Field(default=None, alias="pass")
    eval_by: str | None = None
    # Upstream observability contract. A concrete subgoal id means
    # this condition can only be evaluated after that predecessor completes.
    depends_on: str | None = None
    needs_observation: bool = False
    nl: str | None = None
    negated: bool = False


class Subgoal(_AliasedModel):

    subgoal_id: str
    description: str | None = None
    action_type: str | None = None
    mode: str | None = None
    group_id: str | None = None
    source_kg_subgoal_id: str | None = None
    source_binding: dict[str, Any] = Field(default_factory=dict)
    condition_source: str | None = None
    target_ids: list[str] = Field(default_factory=list)
    goal_region_id: str | None = None
    tool_id: str | None = None
    preconditions: list[Condition] = Field(default_factory=list)
    postconditions: list[Condition] = Field(default_factory=list)
    establish: list[Condition] = Field(default_factory=list)
    destroy: list[Condition] = Field(default_factory=list)
    feasible_ee: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    required_wrench: float | None = Field(default=None, ge=0)
    unique_solution: bool = False
    tool_required: bool = False
    tool_selection_source: Literal["upstream_fixed", "not_required"] = (
        "not_required"
    )
    feasible_ee_source: Literal["gk", "m1", "request"] = "request"

    @model_validator(mode="after")
    def _normalize_tool_source(self) -> "Subgoal":
        extras = self.model_extra or {}
        legacy_candidates = extras.get("tool_candidates") or extras.get(
            "tool_candidate_ids"
        )
        if legacy_candidates:
            raise ValueError(
                "Task Planner does not accept tool candidates; provide the "
                "upstream-selected tool_id"
            )
        if self.tool_id is not None:
            if self.tool_selection_source == "not_required":
                self.tool_selection_source = "upstream_fixed"
        else:
            self.tool_selection_source = "not_required"
        return self


class OrderConstraints(_ExtensibleModel):

    edges: list[Any] = Field(default_factory=list)
    cycle_detected: bool = False
    cycles: list[list[str]] = Field(default_factory=list)
    unestablishable: list[Any] = Field(default_factory=list)
    # Diagnostic only. Task Planner never enumerates topological orders.
    n_topological_orders: int | None = None


class MutexConstraint(_ExtensibleModel):
    """Two consumers that require a resource condition between executions."""

    a: str
    b: str
    condition: str | None = None
    predicate: str
    args: list[str] = Field(default_factory=list)
    reestablished_by: list[str] = Field(default_factory=list)
    resolutions: list[str] = Field(default_factory=list)
    note: str | None = None


class OpenConditionConstraint(_ExtensibleModel):
    """A precondition with multiple possible producer subgoals."""

    subgoal: str
    condition: str
    candidates: list[str] = Field(min_length=1)
    note: str | None = None


class DisjunctiveThreatConstraint(_ExtensibleModel):
    """A threat that must be promoted before, or demoted after, a causal link."""

    key: tuple[str, str, str, str] | None = None
    link: tuple[str, str]
    threat: str
    condition: str
    options: list[str] = Field(default_factory=list)
    note: str | None = None


class DeferredConditionContract(_ExtensibleModel):
    """Condition whose truth can only be checked after another subgoal."""

    subgoal: str
    condition: str
    type: str
    args: list[str] = Field(default_factory=list)
    nl: str | None = None
    depends_on: str | None = None
    needs_observation: bool = False
    why: str | None = None


class ObservationRequest(DeferredConditionContract):
    """A deferred condition that requires a Scene Graph observation."""

    request: str | None = None


class RedecomposeSignal(_ExtensibleModel):
    """The upstream task decomposition is not a valid partial-order graph."""

    rule: int | str | None = None
    subgoal: str | None = None
    condition: str | None = None
    link: tuple[str, str] | None = None
    threat: str | None = None
    cycle: list[str] | None = None
    reason: str | None = None


class KGOrderAudit(_ExtensibleModel):
    """Diagnostic audit of the order originally supplied by the KG."""

    verdicts: list[dict[str, Any]] = Field(default_factory=list)
    kg_missing: list[Any] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


class TaskConstraints(_ExtensibleModel):
    """Typed constraints attached to the normalized GK + M1 task graph."""

    mutex: list[MutexConstraint] = Field(default_factory=list)
    open_conditions: list[OpenConditionConstraint] = Field(default_factory=list)
    disjunctive_threats: list[DisjunctiveThreatConstraint] = Field(
        default_factory=list
    )
    deferred_conditions: list[DeferredConditionContract] = Field(default_factory=list)
    sg_observation_requests: list[ObservationRequest] = Field(default_factory=list)
    redecompose_signals: list[RedecomposeSignal] = Field(default_factory=list)
    kg_order_audit: KGOrderAudit = Field(default_factory=KGOrderAudit)


class InitialState(_ExtensibleModel):

    current_ee: str
    # slot -> EE id; interpreted as *home slot assignment* per EE, not occupancy.
    rack: dict[str, str] = Field(default_factory=dict)
    facts: list[Condition] = Field(default_factory=list)
    # Optional extension fields (backward compatible defaults).
    held_tool: str | None = None
    # slot -> occupant EE id, or "empty" / "reserved" / "unknown".
    rack_occupancy: dict[str, str] | None = None
    scene_signature: str | None = None


class TaskSpec(_ExtensibleModel):

    instruction: str = ""


class Units(_ExtensibleModel):

    length: str = "mm"
    mass: str = "kg"


class TaskGraph(_ExtensibleModel):

    task: TaskSpec = Field(default_factory=TaskSpec)
    units: Units = Field(default_factory=Units)
    initial_state: InitialState
    subgoals: list[Subgoal] = Field(default_factory=list)
    order_constraints: OrderConstraints = Field(default_factory=OrderConstraints)
    constraints: TaskConstraints = Field(default_factory=TaskConstraints)
    log: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Resource catalog                                                            #
# --------------------------------------------------------------------------- #


class EndEffectorSpec(_ExtensibleModel):

    capabilities: list[str] = Field(default_factory=list)
    payload: float | None = Field(default=None, ge=0)
    compatible_tools: list[str] = Field(default_factory=list)
    home_slot: str | None = None


class ToolSpec(_ExtensibleModel):

    mass: float | None = Field(default=None, ge=0)
    capabilities: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    deliverable_wrench: float | None = Field(default=None, ge=0)
    compatible_ee: list[str] = Field(default_factory=list)
    home_slot: str | None = None


class ObjectSpec(_ExtensibleModel):
    """Object properties used for payload checks and scene pass-through."""

    mass_kg: float | None = Field(default=None, ge=0)
    bbox_mm: tuple[float, float, float] | None = None
    material: str | None = None

    @model_validator(mode="after")
    def _validate_bbox(self) -> "ObjectSpec":
        if self.bbox_mm is not None and any(value <= 0 for value in self.bbox_mm):
            raise ValueError("bbox_mm dimensions must all be positive")
        return self


class ResourceCatalog(_ExtensibleModel):

    end_effectors: dict[str, EndEffectorSpec] = Field(default_factory=dict)
    tools: dict[str, ToolSpec] = Field(default_factory=dict)
    objects: dict[str, ObjectSpec] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Candidate proposals                                                         #
# --------------------------------------------------------------------------- #

CandidateSource = Literal["vlm", "knowledge_graph", "deterministic_rule", "manual"]


class GraspSpec(_ExtensibleModel):
    """Externally generated grasp hypothesis passed through to motion planning.

    Task Planner treats ``pose`` and the optional approach/contact payloads as
    opaque data.  It only preserves grasp identity and ownership so that the
    motion planner can resolve the correct geometry without inspecting ad-hoc
    candidate metadata.
    """

    grasp_id: str = Field(min_length=1)
    owner_kind: Literal["object", "tool", "end_effector"]
    owner_id: str = Field(min_length=1)
    pose: Any = None
    approach_pose: Any = None
    score: float | None = Field(default=None, ge=0, le=1)
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateProposal(_ExtensibleModel):

    candidate_id: str
    subgoal_id: str
    ee: str
    tool: str | None = None
    grasp_id: str | None = None
    grasp: GraspSpec | None = None
    suitability_score: float | None = Field(default=None, ge=0, le=1)
    source: CandidateSource = "manual"
    required_capabilities: list[str] = Field(default_factory=list)
    nominal_execution_cost: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_grasp_identity(self) -> "CandidateProposal":
        if self.grasp is None:
            return self
        if self.grasp_id is None:
            self.grasp_id = self.grasp.grasp_id
        elif self.grasp_id != self.grasp.grasp_id:
            raise ValueError(
                "grasp_id must match grasp.grasp_id when both are supplied"
            )
        return self


# --------------------------------------------------------------------------- #
# Policy                                                                      #
# --------------------------------------------------------------------------- #


class MotionCosts(_ExtensibleModel):

    move_to_tool_rack: int = Field(default=2, ge=0)
    move_to_ee_rack: int = Field(default=3, ge=0)
    move_to_workspace: int = Field(default=1, ge=0)


class TerminalPolicy(_ExtensibleModel):

    return_tool_at_end: bool = True
    restore_initial_ee_at_end: bool = False
    require_empty_object_hand_at_end: bool = True


class PlanningPolicy(_ExtensibleModel):

    candidate_score_threshold: float = Field(default=0.6, ge=0, le=1)
    top_k_per_subgoal: int = Field(default=3, ge=0)
    preserve_ee_coverage: bool = True
    unknown_feasibility_policy: Literal["reject", "allow", "defer"] = "reject"
    # Which geometric checks the *search* is allowed to invoke.
    #
    # ``scene_independent`` (default) runs only checks whose answer does not
    # depend on where movable objects currently are -- in practice the EE rack
    # exchange. Those cache perfectly, never need a predicted-scene
    # reconstruction, and are the checks whose absence most distorts the plan
    # (EE switches dominate the lexicographic cost). Everything scene-dependent
    # is deferred to post-search verification, which sees one concrete ordering
    # and can simply walk it forward.
    #
    # ``full`` restores per-candidate and per-transition lazy checking inside
    # the search. ``none`` disables geometry in the search entirely.
    #
    # A check that is *not invoked* is not the same as a check that returned
    # UNKNOWN: skipped checks bypass ``unknown_feasibility_policy`` because the
    # planner is deferring the question, not failing to answer it.
    in_search_geometry: Literal["full", "scene_independent", "none"] = (
        "scene_independent"
    )
    unknown_suitability_policy: Literal["reject", "allow", "defer"] = "defer"
    motion_costs: MotionCosts = Field(default_factory=MotionCosts)
    terminal: TerminalPolicy = Field(default_factory=TerminalPolicy)
    max_expansions: int = Field(default=200_000, ge=0)
    timeout_seconds: float | None = Field(default=None, ge=0)


class TaskPlannerRequest(_ExtensibleModel):

    task_graph: TaskGraph
    resource_catalog: ResourceCatalog
    candidate_proposals: dict[str, list[CandidateProposal]] | None = None
    planning_policy: PlanningPolicy = Field(default_factory=PlanningPolicy)


# --------------------------------------------------------------------------- #
# Replanning inputs                                                           #
# --------------------------------------------------------------------------- #


class ExecutionState(_ExtensibleModel):
    """Actual world/robot state at replanning time."""

    completed_subgoals: list[str] = Field(default_factory=list)
    current_ee: str
    held_tool: str | None = None
    facts: list[Condition] = Field(default_factory=list)
    scene_signature: str | None = None
    rack_occupancy: dict[str, str] | None = None
    # EEs already committed per group by the executed prefix of the plan.
    group_ee_bindings: dict[str, str] = Field(default_factory=dict)


class FailureType(str, enum.Enum):
    CANDIDATE_INVALID = "CANDIDATE_INVALID"
    TRANSITION_INVALID = "TRANSITION_INVALID"
    RESOURCE_UNAVAILABLE = "RESOURCE_UNAVAILABLE"
    SCENE_CHANGED = "SCENE_CHANGED"
    ATTACHMENT_FAILED = "ATTACHMENT_FAILED"


class FailureFeedback(_ExtensibleModel):

    failure_type: FailureType
    subgoal_id: str | None = None
    candidate_id: str | None = None
    scene_signature: str | None = None
    # Optional explicit scope override; default derived from failure_type.
    scope: Literal["scene", "global", "transition"] | None = None
    # [current_ee, held_tool or "", candidate_id], optionally scene-prefixed.
    transition_signature: list[str] | None = Field(
        default=None, min_length=3, max_length=4
    )
    message: str | None = None
