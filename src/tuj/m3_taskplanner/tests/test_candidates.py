"""Candidate generation, scoring threshold, top-k, static feasibility."""

from __future__ import annotations

import pytest

from tuj.m3_taskplanner.candidate_provider import (
    CatalogCandidateProvider,
    StaticCandidateProvider,
    filter_candidates,
)
from tuj.m3_taskplanner.diagnostics import PlanStatus, ReasonCode
from tuj.m3_taskplanner.feasibility import StaticFeasibilityChecker
from tuj.m3_taskplanner.models import PlanningPolicy, ResourceCatalog, Subgoal
from tuj.m3_taskplanner.planner import plan
from tuj.m3_taskplanner.suitability import (
    SuitabilityAssessment,
    SuitabilityComponent,
    SuitabilityStatus,
)

from conftest import base_catalog, make_request, prop, sg


def _checker(
    catalog: ResourceCatalog | None = None,
    policy: PlanningPolicy | None = None,
) -> StaticFeasibilityChecker:
    return StaticFeasibilityChecker(
        catalog or base_catalog(), policy or PlanningPolicy(), {}
    )


def _normalize(subgoal, proposals):
    provider = StaticCandidateProvider(
        {subgoal.subgoal_id: proposals}, base_catalog()
    )
    candidates, rejections = provider.candidates_for(subgoal)
    assert not rejections
    return candidates


def test_normalized_input_rejects_tool_candidates() -> None:
    with pytest.raises(ValueError, match="does not accept tool candidates"):
        Subgoal(
            subgoal_id="S1",
            feasible_ee=["A"],
            tool_candidates=["t1", "t2"],
        )


def test_score_below_threshold_removed() -> None:
    subgoal = sg("S1", targets=["obj1"])
    candidates = _normalize(
        subgoal,
        [
            prop("S1-c1", "S1", "A", score=0.5, source="vlm"),
            prop("S1-c2", "S1", "A", score=0.9, source="vlm"),
        ],
    )
    kept, rejections = filter_candidates(
        subgoal, candidates, _checker(), PlanningPolicy()
    )
    assert [c.candidate_id for c in kept] == ["S1-c2"]
    assert any(
        r.reason_code is ReasonCode.SCORE_BELOW_THRESHOLD
        and r.candidate_id == "S1-c1"
        for r in rejections
    )


def test_vlm_candidate_without_score_rejected() -> None:
    subgoal = sg("S1", targets=["obj1"])
    candidates = _normalize(
        subgoal,
        [prop("S1-c1", "S1", "A", score=None, source="vlm")],
    )
    kept, rejections = filter_candidates(
        subgoal, candidates, _checker(), PlanningPolicy()
    )
    assert kept == []
    assert rejections[0].reason_code is ReasonCode.MISSING_SCORE


def test_deterministic_rule_candidate_may_omit_score() -> None:
    subgoal = sg("S1", targets=["obj1"])
    candidates = _normalize(
        subgoal,
        [
            prop(
                "S1-c1",
                "S1",
                "A",
                score=None,
                source="deterministic_rule",
            )
        ],
    )
    kept, _ = filter_candidates(subgoal, candidates, _checker(), PlanningPolicy())
    assert [c.candidate_id for c in kept] == ["S1-c1"]


class _MixedCompletenessScorer:
    def score(self, candidate, _subgoal) -> SuitabilityAssessment:
        if candidate.ee == "A":
            return SuitabilityAssessment(
                0.803,
                {
                    "payload": SuitabilityComponent(
                        SuitabilityStatus.PASS, score=0.803
                    )
                },
            )
        return SuitabilityAssessment(
            None,
            {
                "payload": SuitabilityComponent.unknown(
                    "EE payload is missing"
                )
            },
        )


def test_unknown_suitability_is_not_ranked_as_perfect() -> None:
    subgoal = sg("S1", targets=["obj1"], feasible=["A", "B"])
    candidates = _normalize(
        subgoal,
        [
            prop(
                "S1-known",
                "S1",
                "A",
                score=None,
                source="deterministic_rule",
            ),
            prop(
                "S1-unknown",
                "S1",
                "B",
                score=None,
                source="deterministic_rule",
            ),
        ],
    )
    policy = PlanningPolicy(
        unknown_suitability_policy="allow",
        top_k_per_subgoal=1,
        preserve_ee_coverage=False,
    )
    kept, _ = filter_candidates(
        subgoal,
        candidates,
        _checker(policy=policy),
        policy,
        suitability_scorer=_MixedCompletenessScorer(),
    )
    assert [candidate.candidate_id for candidate in kept] == ["S1-known"]
    assert kept[0].suitability_score == 0.803


def test_unknown_suitability_obeys_reject_and_defer_policy() -> None:
    subgoal = sg("S1", targets=["obj1"], feasible=["B"])
    candidates = _normalize(
        subgoal,
        [
            prop(
                "S1-unknown",
                "S1",
                "B",
                score=None,
                source="deterministic_rule",
            )
        ],
    )

    reject_policy = PlanningPolicy(unknown_suitability_policy="reject")
    kept_reject, rejections = filter_candidates(
        subgoal,
        candidates,
        _checker(policy=reject_policy),
        reject_policy,
        suitability_scorer=_MixedCompletenessScorer(),
    )
    assert kept_reject == []
    assert rejections[0].reason_code is ReasonCode.UNKNOWN_SUITABILITY_REJECTED

    defer_policy = PlanningPolicy(unknown_suitability_policy="defer")
    kept_defer, _ = filter_candidates(
        subgoal,
        candidates,
        _checker(policy=defer_policy),
        defer_policy,
        suitability_scorer=_MixedCompletenessScorer(),
    )
    assert len(kept_defer) == 1
    assert kept_defer[0].metadata["suitability_deferred"] is True
    assert kept_defer[0].metadata["suitability"]["unknown_policy"] == "defer"


def test_top_k_prunes_but_preserves_ee_coverage() -> None:
    subgoal = sg("S1", targets=["obj1"], feasible=["A", "B"])
    candidates = _normalize(
        subgoal,
        [
            prop("S1-a1", "S1", "A", score=0.95),
            prop("S1-a2", "S1", "A", score=0.9),
            prop("S1-a3", "S1", "A", score=0.85),
            prop("S1-b1", "S1", "B", score=0.7),
        ],
    )
    policy = PlanningPolicy(top_k_per_subgoal=3, preserve_ee_coverage=True)
    kept, rejections = filter_candidates(subgoal, candidates, _checker(), policy)
    kept_ids = {c.candidate_id for c in kept}
    # B's best survives even though it is outside the global top-3.
    assert kept_ids == {"S1-a1", "S1-a2", "S1-a3", "S1-b1"}

    policy_off = PlanningPolicy(top_k_per_subgoal=3, preserve_ee_coverage=False)
    kept_off, rejections_off = filter_candidates(
        subgoal, candidates, _checker(), policy_off
    )
    assert {c.candidate_id for c in kept_off} == {"S1-a1", "S1-a2", "S1-a3"}
    assert any(
        r.reason_code is ReasonCode.TOP_K_PRUNED and r.candidate_id == "S1-b1"
        for r in rejections_off
    )


def test_unknown_resources_rejected() -> None:
    subgoal = sg("S1", targets=["obj1"], feasible=["ghost"])
    candidates = _normalize(subgoal, [prop("S1-c1", "S1", "ghost")])
    kept, rejections = filter_candidates(
        subgoal, candidates, _checker(), PlanningPolicy()
    )
    assert kept == []
    assert rejections[0].reason_code is ReasonCode.UNKNOWN_EE

    subgoal2 = sg("S2", targets=["obj1"], tool_id="ghost-tool")
    provider = StaticCandidateProvider(
        {"S2": [prop("S2-c1", "S2", "A", tool="ghost-tool")]},
        base_catalog(),
    )
    candidates2, _ = provider.candidates_for(subgoal2)
    kept2, rejections2 = filter_candidates(
        subgoal2, candidates2, _checker(), PlanningPolicy()
    )
    assert kept2 == []
    assert rejections2[0].reason_code is ReasonCode.UNKNOWN_TOOL


def test_tool_required_without_tool_id_or_proposals_is_incomplete_input() -> None:
    request = make_request(
        [sg("S1", targets=["obj1"], tool_required=True)],
        proposals=None,  # forces the catalog provider
    )
    result = plan(request)
    assert result.status is PlanStatus.INFEASIBLE_NO_CANDIDATE
    assert any(
        r.reason_code is ReasonCode.TOOL_REQUIRED_MISSING for r in result.rejections
    )


def test_catalog_provider_generates_one_candidate_per_ee_for_fixed_tool() -> None:
    provider = CatalogCandidateProvider(base_catalog(), PlanningPolicy())
    candidates, rejections = provider.candidates_for(
        sg("S1", tool_id="t1", feasible=["A", "B"])
    )
    assert not rejections
    assert [(c.ee, c.tool) for c in candidates] == [("A", "t1"), ("B", "t1")]
    assert all(c.source == "deterministic_rule" for c in candidates)


def test_catalog_provider_needs_no_contact_candidate_for_direct_work() -> None:
    provider = CatalogCandidateProvider(base_catalog(), PlanningPolicy())
    candidates, rejections = provider.candidates_for(
        sg("S1", targets=["obj1"], feasible=["A"])
    )
    assert not rejections
    assert [(candidate.ee, candidate.tool) for candidate in candidates] == [
        ("A", None)
    ]


def test_unknown_feasibility_policy_reject_vs_allow() -> None:
    catalog = base_catalog()
    catalog.tools["t1"].mass = None  # payload check becomes UNKNOWN
    subgoal = sg("S1", tool_id="t1", feasible=["A"])
    provider = CatalogCandidateProvider(catalog, PlanningPolicy())
    candidates, _ = provider.candidates_for(subgoal)

    kept_reject, rejections = filter_candidates(
        subgoal, candidates, _checker(catalog), PlanningPolicy()
    )
    assert kept_reject == []
    assert all(
        r.reason_code is ReasonCode.UNKNOWN_FEASIBILITY_REJECTED
        for r in rejections
    )

    allow_policy = PlanningPolicy(unknown_feasibility_policy="allow")
    kept_allow, _ = filter_candidates(
        subgoal, candidates, _checker(catalog, allow_policy), allow_policy
    )
    assert len(kept_allow) == 1


def test_payload_exceeded_rejected() -> None:
    catalog = base_catalog()
    catalog.tools["t1"].mass = 99.0
    subgoal = sg("S1", tool_id="t1", feasible=["A"])
    provider = CatalogCandidateProvider(catalog, PlanningPolicy())
    candidates, _ = provider.candidates_for(subgoal)
    kept, rejections = filter_candidates(
        subgoal, candidates, _checker(catalog), PlanningPolicy()
    )
    assert kept == []
    assert all(
        r.reason_code is ReasonCode.PAYLOAD_EXCEEDED for r in rejections
    )


def test_static_proposal_cannot_bypass_required_tool() -> None:
    required = sg("S1", targets=["obj1"], tool_required=True)
    candidates = _normalize(
        required,
        [prop("S1-c1", "S1", "A")],
    )
    kept, rejections = filter_candidates(
        required, candidates, _checker(), PlanningPolicy()
    )
    assert kept == []
    assert rejections[0].reason_code is ReasonCode.TOOL_REQUIRED_MISSING

    fixed = sg("S2", tool_id="t1", feasible=["A"], action="operate")
    wrong = _normalize(
        fixed,
        [prop("S2-c1", "S2", "A", tool="t2")],
    )
    kept2, rejections2 = filter_candidates(
        fixed, wrong, _checker(), PlanningPolicy()
    )
    assert kept2 == []
    assert rejections2[0].reason_code is ReasonCode.TOOL_MISMATCH

    no_tool = sg("S3", feasible=["A"], action="inspect")
    injected = _normalize(
        no_tool,
        [prop("S3-c1", "S3", "A", tool="t1")],
    )
    kept3, rejections3 = filter_candidates(
        no_tool, injected, _checker(), PlanningPolicy()
    )
    assert kept3 == []
    assert rejections3[0].reason_code is ReasonCode.TOOL_MISMATCH


def test_empty_compatibility_list_means_no_compatible_tools() -> None:
    catalog = base_catalog()
    catalog.end_effectors["A"].compatible_tools = []
    subgoal = sg("S1", tool_id="t1", feasible=["A"], action="operate")
    provider = CatalogCandidateProvider(catalog, PlanningPolicy())
    candidates, _ = provider.candidates_for(subgoal)
    kept, rejections = filter_candidates(
        subgoal, candidates, _checker(catalog), PlanningPolicy()
    )
    assert kept == []
    assert all(
        r.reason_code is ReasonCode.EE_TOOL_INCOMPATIBLE for r in rejections
    )
