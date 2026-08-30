"""Suitability checks retained at the task-planning layer."""

from __future__ import annotations

from tuj.m3_taskplanner.candidate_provider import Candidate
from tuj.m3_taskplanner.diagnostics import PlanStatus, ReasonCode
from tuj.m3_taskplanner.models import ResourceCatalog
from tuj.m3_taskplanner.planner import plan
from tuj.m3_taskplanner.suitability import (
    PhysicsSuitabilityScorer,
    SuitabilityAssessment,
    SuitabilityComponent,
    SuitabilityStatus,
)

from conftest import make_request, prop, sg


def catalog() -> ResourceCatalog:
    return ResourceCatalog.model_validate(
        {
            "end_effectors": {
                "A": {
                    "capabilities": ["carry", "tool_holding"],
                    "payload": 5.0,
                    "compatible_tools": ["t1"],
                    "home_slot": "SA",
                }
            },
            "tools": {
                "t1": {
                    "mass": 1.0,
                    "deliverable_wrench": 20.0,
                    "required_capabilities": ["tool_holding"],
                    "compatible_ee": ["A"],
                    "home_slot": "T1",
                }
            },
            "objects": {
                "light": {"mass_kg": 1.0},
                "heavy": {"mass_kg": 6.0},
            },
        }
    )


def candidate(*, tool: str | None = None, **metadata) -> Candidate:
    return Candidate(
        candidate_id="C",
        subgoal_id="S",
        ee="A",
        tool=tool,
        metadata=metadata,
    )


def test_payload_passes_and_fails_from_object_mass() -> None:
    scorer = PhysicsSuitabilityScorer(catalog())
    passed = scorer.score(candidate(), sg("S", targets=["light"], feasible=["A"]))
    failed = scorer.score(candidate(), sg("S", targets=["heavy"], feasible=["A"]))
    assert passed.components["payload"].status is SuitabilityStatus.PASS
    assert failed.failure_reason is ReasonCode.PAYLOAD_EXCEEDED


def test_supported_object_does_not_add_its_full_mass() -> None:
    scorer = PhysicsSuitabilityScorer(catalog())
    assessment = scorer.score(
        candidate(tool="t1", object_remains_supported=True),
        sg("S", targets=["heavy"], tool_id="t1", feasible=["A"]),
    )
    assert assessment.components["payload"].required == 1.0
    assert not assessment.failed


def test_wrench_capacity_is_checked_independently() -> None:
    scorer = PhysicsSuitabilityScorer(catalog())
    passed = scorer.score(
        candidate(tool="t1", object_remains_supported=True),
        sg("S", tool_id="t1", feasible=["A"], action="operate"),
    )
    assert passed.components["wrench"].status is SuitabilityStatus.NOT_APPLICABLE

    subgoal = sg("S", tool_id="t1", feasible=["A"], action="operate")
    subgoal.required_wrench = 25.0
    failed = scorer.score(candidate(tool="t1"), subgoal)
    assert failed.failure_reason is ReasonCode.WRENCH_INSUFFICIENT


def test_missing_payload_data_is_unknown_not_perfect() -> None:
    raw = catalog().model_dump()
    raw["end_effectors"]["A"].pop("payload")
    assessment = PhysicsSuitabilityScorer(
        ResourceCatalog.model_validate(raw)
    ).score(candidate(), sg("S", targets=["light"], feasible=["A"]))
    assert assessment.overall_score is None
    assert assessment.unknown_components == ("payload",)


def test_planner_serializes_payload_and_wrench_breakdown() -> None:
    subgoal = sg("S", targets=["light"], feasible=["A"])
    request = make_request(
        [subgoal],
        proposals={"S": [prop("C", "S", "A", score=0.95)]},
        catalog=catalog(),
    )
    result = plan(request)
    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    assignment = result.selected_plan.candidate_assignments[0]
    assert assignment.suitability is not None
    assert set(assignment.suitability["components"]) == {"payload", "wrench"}
    assert assignment.suitability["provided_suitability_score"] == 0.95


def test_planner_can_disable_or_inject_suitability_scorer() -> None:
    class RejectingScorer:
        def score(self, candidate, subgoal):
            return SuitabilityAssessment(
                0.0,
                {
                    "payload": SuitabilityComponent(
                        SuitabilityStatus.FAIL, score=0.0
                    )
                },
                ReasonCode.PAYLOAD_EXCEEDED,
                "injected rejection",
            )

    request = make_request(
        [sg("S", feasible=["A"])],
        proposals={"S": [prop("C", "S", "A")]},
        catalog=catalog(),
    )
    assert plan(request, suitability_scorer=None).status is PlanStatus.SUCCESS
    rejected = plan(request, suitability_scorer=RejectingScorer())
    assert rejected.status is PlanStatus.INFEASIBLE_NO_CANDIDATE
