"""Candidate generation (static proposals + catalog rules) and filtering.

Candidates are normalized ``(subgoal, EE, tool)`` assignments. A tool can be
fixed by the upstream GK + M2 contract or selected from the finite candidates
supplied by the GK/M2 adapter. Contact poses remain owned by motion planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from tuj.m4_taskplanner.diagnostics import ReasonCode, Rejection, make_rejection
from tuj.m4_taskplanner.models import (
    CandidateProposal,
    PlanningPolicy,
    ResourceCatalog,
    Subgoal,
)

if TYPE_CHECKING:
    from tuj.m4_taskplanner.feasibility import StaticFeasibilityChecker
    from tuj.m4_taskplanner.suitability import SuitabilityScorer


@dataclass(frozen=True, slots=True)
class Candidate:
    """Normalized ``(subgoal, EE, tool)`` tuple used by the search."""

    candidate_id: str
    subgoal_id: str
    ee: str
    tool: str | None
    grasp_id: str | None = field(compare=False, default=None)
    grasp: Any = field(compare=False, hash=False, default=None)
    target_ids: tuple[str, ...] = field(compare=False, default=())
    action_type: str | None = field(compare=False, default=None)
    source: str = field(compare=False, default="manual")
    suitability_score: float | None = field(compare=False, default=None)
    required_capabilities: frozenset[str] = field(compare=False, default=frozenset())
    nominal_execution_cost: int = field(compare=False, default=0)
    metadata: dict[str, Any] = field(compare=False, hash=False, default_factory=dict)


class StaticCandidateProvider:
    """Normalize externally supplied EE/tool proposals."""

    def __init__(
        self,
        proposals: dict[str, list[CandidateProposal]],
        catalog: ResourceCatalog,
    ) -> None:
        self._proposals = proposals
        self._catalog = catalog

    def candidates_for(
        self, subgoal: Subgoal
    ) -> tuple[list[Candidate], list[Rejection]]:
        rejections: list[Rejection] = []
        result: list[Candidate] = []
        for proposal in self._proposals.get(subgoal.subgoal_id, []):
            if proposal.subgoal_id != subgoal.subgoal_id:
                rejections.append(
                    make_rejection(
                        "candidate",
                        ReasonCode.SUBGOAL_MISMATCH,
                        f"proposal {proposal.candidate_id!r} declares "
                        f"subgoal_id={proposal.subgoal_id!r} but was listed "
                        f"under {subgoal.subgoal_id!r}",
                        subgoal_id=subgoal.subgoal_id,
                        candidate_id=proposal.candidate_id,
                    )
                )
                continue
            result.append(
                Candidate(
                    candidate_id=proposal.candidate_id,
                    subgoal_id=proposal.subgoal_id,
                    ee=proposal.ee,
                    tool=proposal.tool,
                    grasp_id=proposal.grasp_id,
                    grasp=proposal.grasp,
                    target_ids=tuple(subgoal.target_ids),
                    action_type=subgoal.action_type,
                    source=proposal.source,
                    suitability_score=proposal.suitability_score,
                    required_capabilities=frozenset(
                        proposal.required_capabilities
                    ),
                    nominal_execution_cost=proposal.nominal_execution_cost,
                    metadata=dict(proposal.metadata),
                )
            )
        result.sort(key=lambda candidate: candidate.candidate_id)
        return result, rejections


class CatalogCandidateProvider:
    """Generate one candidate per feasible EE for an upstream-fixed tool."""

    def __init__(self, catalog: ResourceCatalog, policy: PlanningPolicy) -> None:
        self._catalog = catalog
        self._policy = policy

    def candidates_for(
        self, subgoal: Subgoal
    ) -> tuple[list[Candidate], list[Rejection]]:
        rejections: list[Rejection] = []
        result: list[Candidate] = []
        tool = subgoal.tool_id

        if tool is not None and tool not in self._catalog.tools:
            rejections.append(
                make_rejection(
                    "subgoal",
                    ReasonCode.UNKNOWN_TOOL,
                    f"tool {tool!r} not in catalog",
                    subgoal_id=subgoal.subgoal_id,
                )
            )
            return [], rejections
        if tool is None and subgoal.tool_required:
            rejections.append(
                make_rejection(
                    "subgoal",
                    ReasonCode.TOOL_REQUIRED_MISSING,
                    "tool_required=true but upstream tool_id was not supplied",
                    subgoal_id=subgoal.subgoal_id,
                )
            )
            return [], rejections

        for ee in sorted(set(subgoal.feasible_ee)):
            if ee not in self._catalog.end_effectors:
                rejections.append(
                    make_rejection(
                        "candidate",
                        ReasonCode.UNKNOWN_EE,
                        f"feasible_ee entry {ee!r} not in catalog",
                        subgoal_id=subgoal.subgoal_id,
                    )
                )
                continue
            result.append(self._make(subgoal, ee, tool))
        return result, rejections

    @staticmethod
    def _make(subgoal: Subgoal, ee: str, tool: str | None) -> Candidate:
        return Candidate(
            candidate_id=f"{subgoal.subgoal_id}-auto-{ee}-{tool or 'notool'}",
            subgoal_id=subgoal.subgoal_id,
            ee=ee,
            tool=tool,
            target_ids=tuple(subgoal.target_ids),
            action_type=subgoal.action_type,
            source="deterministic_rule",
            suitability_score=None,
        )


def _ranking_key(candidate: Candidate) -> tuple[int, float, int, str]:
    """Rank complete physical assessments before incomplete ones."""

    details = candidate.metadata.get("suitability")
    unknown_count = 0
    if isinstance(details, Mapping):
        raw_count = details.get("unknown_component_count")
        if isinstance(raw_count, int):
            unknown_count = raw_count
        else:
            components = details.get("components")
            if isinstance(components, Mapping):
                unknown_count = sum(
                    isinstance(component, Mapping)
                    and component.get("status") == "UNKNOWN"
                    for component in components.values()
                )
    elif candidate.suitability_score is None:
        unknown_count = 1

    score = (
        candidate.suitability_score
        if candidate.suitability_score is not None
        else -1.0
    )
    return (int(unknown_count > 0), -score, unknown_count, candidate.candidate_id)


def filter_candidates(
    subgoal: Subgoal,
    candidates: Iterable[Candidate],
    static_checker: "StaticFeasibilityChecker",
    policy: PlanningPolicy,
    banned_candidate_ids: frozenset[str] = frozenset(),
    suitability_scorer: "SuitabilityScorer | None" = None,
) -> tuple[list[Candidate], list[Rejection]]:
    """Threshold -> static feasibility -> top-k (with EE coverage guard)."""
    rejections: list[Rejection] = []

    scored: list[Candidate] = []
    for cand in sorted(candidates, key=lambda c: c.candidate_id):
        if cand.candidate_id in banned_candidate_ids:
            rejections.append(
                make_rejection(
                    "candidate",
                    ReasonCode.CANDIDATE_BANNED,
                    "candidate globally banned by replanning no-good set",
                    subgoal_id=subgoal.subgoal_id,
                    candidate_id=cand.candidate_id,
                )
            )
            continue
        if suitability_scorer is not None:
            assessment = suitability_scorer.score(cand, subgoal)
            physical_rejection = assessment.rejection(cand, subgoal)
            if physical_rejection is not None:
                rejections.append(physical_rejection)
                continue
            metadata = dict(cand.metadata)
            suitability_details = assessment.to_dict()
            unknown_components = assessment.unknown_components
            if unknown_components:
                suitability_details["unknown_policy"] = (
                    policy.unknown_suitability_policy
                )
                if policy.unknown_suitability_policy == "reject":
                    rejections.append(
                        make_rejection(
                            "candidate",
                            ReasonCode.UNKNOWN_SUITABILITY_REJECTED,
                            "unknown physical suitability rejected by policy: "
                            + ", ".join(unknown_components),
                            subgoal_id=subgoal.subgoal_id,
                            candidate_id=cand.candidate_id,
                            suitability=suitability_details,
                        )
                    )
                    continue
                if policy.unknown_suitability_policy == "defer":
                    metadata["suitability_deferred"] = True
            if assessment.overall_score is not None:
                if cand.suitability_score is not None:
                    metadata["provided_suitability_score"] = cand.suitability_score
                    suitability_details["provided_suitability_score"] = (
                        cand.suitability_score
                    )
                cand = replace(cand, suitability_score=assessment.overall_score)
            metadata["suitability"] = suitability_details
            cand = replace(cand, metadata=metadata)
        if cand.suitability_score is None:
            if cand.source not in ("deterministic_rule", "manual"):
                rejections.append(
                    make_rejection(
                        "candidate",
                        ReasonCode.MISSING_SCORE,
                        f"source={cand.source!r} candidates must carry a "
                        "suitability_score",
                        subgoal_id=subgoal.subgoal_id,
                        candidate_id=cand.candidate_id,
                    )
                )
                continue
        elif cand.suitability_score < policy.candidate_score_threshold:
            rejections.append(
                make_rejection(
                    "candidate",
                    ReasonCode.SCORE_BELOW_THRESHOLD,
                    f"score {cand.suitability_score} < threshold "
                    f"{policy.candidate_score_threshold}",
                    subgoal_id=subgoal.subgoal_id,
                    candidate_id=cand.candidate_id,
                )
            )
            continue
        scored.append(cand)

    feasible: list[Candidate] = []
    for cand in scored:
        rejection = static_checker.check(cand, subgoal)
        if rejection is not None:
            rejections.append(rejection)
            continue
        feasible.append(cand)

    ranked = sorted(feasible, key=_ranking_key)
    kept = ranked[: policy.top_k_per_subgoal]
    if policy.preserve_ee_coverage:
        kept_ees = {c.ee for c in kept}
        for ee in sorted({c.ee for c in ranked} - kept_ees):
            kept.append(next(c for c in ranked if c.ee == ee))
    kept_ids = {c.candidate_id for c in kept}
    for cand in ranked:
        if cand.candidate_id not in kept_ids:
            rejections.append(
                make_rejection(
                    "candidate",
                    ReasonCode.TOP_K_PRUNED,
                    f"pruned by top_k_per_subgoal={policy.top_k_per_subgoal}",
                    subgoal_id=subgoal.subgoal_id,
                    candidate_id=cand.candidate_id,
                )
            )
    kept.sort(key=lambda c: c.candidate_id)
    return kept, rejections
