"""The motion-planner contract must actually reach the search."""

from __future__ import annotations

import pytest

from conftest import make_request, prop, sg
from tuj.m4_taskplanner import plan
from tuj.m4_taskplanner.diagnostics import PlanStatus, ReasonCode
from tuj.m4_taskplanner.feasibility import CheckResult, FeasibilityStatus
from tuj.m4_taskplanner.models import PlanningPolicy
from tuj.m4_taskplanner.motion_interface import (
    CandidateQuery,
    EEExchangeQuery,
    MotionCostResult,
    SceneRef,
    TerminalQuery,
    TransitionQuery,
    UnknownMotionOracle,
    WorldSnapshot,
    as_scene_ref,
)


class RecordingOracle:
    """Passes everything, but records every query it was asked."""

    def __init__(self) -> None:
        self.candidates: list[CandidateQuery] = []
        self.transitions: list[TransitionQuery] = []
        self.exchanges: list[EEExchangeQuery] = []
        self.terminals: list[TerminalQuery] = []
        self.initialized: list[WorldSnapshot] = []

    def initialize(self, world: WorldSnapshot) -> None:
        self.initialized.append(world)

    def check_candidate(self, query: CandidateQuery) -> CheckResult:
        self.candidates.append(query)
        return CheckResult.ok()

    def check_transition(self, query: TransitionQuery) -> CheckResult:
        self.transitions.append(query)
        return CheckResult.ok()

    def check_ee_exchange(self, query: EEExchangeQuery) -> CheckResult:
        self.exchanges.append(query)
        return CheckResult.ok()

    def check_terminal(self, query: TerminalQuery) -> CheckResult:
        self.terminals.append(query)
        return CheckResult.ok()


class RackBlockedOracle(RecordingOracle):
    """Docking into ``blocked_ee`` is geometrically impossible."""

    def __init__(self, blocked_ee: str) -> None:
        super().__init__()
        self._blocked = blocked_ee

    def check_ee_exchange(self, query: EEExchangeQuery) -> CheckResult:
        self.exchanges.append(query)
        if query.to_ee == self._blocked:
            return CheckResult(
                FeasibilityStatus.FAIL,
                ReasonCode.RACK_SLOT_UNAVAILABLE,
                f"slot for {query.to_ee!r} is not reachable",
            )
        return CheckResult.ok()


def _two_ee_problem(**policy_kwargs):
    """S1 needs EE A, S2 needs EE B -> exactly one exchange."""
    subgoals = [
        sg("S1", targets=["obj1"], feasible=["A"]),
        sg("S2", targets=["obj2"], feasible=["B"]),
    ]
    proposals = {
        "S1": [prop("S1-A", "S1", "A")],
        "S2": [prop("S2-B", "S2", "B")],
    }
    return make_request(
        subgoals,
        edges=[("S1", "S2")],
        proposals=proposals,
        initial_ee="A",
        policy=PlanningPolicy(**policy_kwargs) if policy_kwargs else None,
    )


def test_oracle_receives_scene_history_not_just_a_signature() -> None:
    """Scene history reaches the oracle when the search runs full geometry."""
    oracle = RecordingOracle()
    result = plan(
        _two_ee_problem(in_search_geometry="full"), geometry_checker=oracle
    )

    assert result.status is PlanStatus.SUCCESS
    assert oracle.candidates, "candidate queries never reached the oracle"
    # The signature alone is not enough to rebuild a predicted scene; the
    # oracle must also receive the causal history behind it.
    later = [q for q in oracle.candidates if q.scene.completed_subgoals]
    assert later, "no query carried completed_subgoals"
    assert all(isinstance(q.scene, SceneRef) for q in oracle.candidates)


def test_ee_exchange_gets_its_own_query() -> None:
    oracle = RecordingOracle()
    result = plan(_two_ee_problem(), geometry_checker=oracle)

    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    assert result.selected_plan.cost_vector.ee_switches == 1
    assert any(
        q.from_ee == "A" and q.to_ee == "B" for q in oracle.exchanges
    ), "the A->B exchange was never geometrically validated"


def test_infeasible_ee_exchange_removes_the_edge() -> None:
    """A blocked dock must make the plan fail, not silently succeed."""
    oracle = RackBlockedOracle(blocked_ee="B")
    result = plan(_two_ee_problem(), geometry_checker=oracle)

    assert result.status is PlanStatus.INFEASIBLE_NO_PLAN
    assert any(
        r.reason_code is ReasonCode.RACK_SLOT_UNAVAILABLE for r in result.rejections
    )


def test_terminal_cleanup_is_geometrically_checked() -> None:
    """Restoring the initial EE at the end is a motion, so it must be checked."""
    oracle = RecordingOracle()
    request = make_request(
        [sg("S1", targets=["obj1"], feasible=["B"])],
        proposals={"S1": [prop("S1-B", "S1", "B")]},
        initial_ee="A",
        policy=PlanningPolicy.model_validate(
            {"terminal": {"restore_initial_ee_at_end": True}}
        ),
    )
    result = plan(request, geometry_checker=oracle)

    assert result.status is PlanStatus.SUCCESS
    assert oracle.terminals, "the terminal restore edge was never checked"
    assert oracle.terminals[0].restore_ee == "A"


def test_unknown_oracle_is_rejected_by_default_and_allowed_on_demand() -> None:
    rejecting = plan(
        _two_ee_problem(unknown_feasibility_policy="reject"),
        geometry_checker=UnknownMotionOracle(),
    )
    assert rejecting.status is not PlanStatus.SUCCESS

    allowing = plan(
        _two_ee_problem(unknown_feasibility_policy="allow"),
        geometry_checker=UnknownMotionOracle(),
    )
    assert allowing.status is PlanStatus.SUCCESS


def test_legacy_checker_still_works_without_the_new_hooks() -> None:
    """The old two-method protocol must keep working unchanged."""
    from tuj.m4_taskplanner.feasibility import AlwaysPassGeometryChecker

    checker = AlwaysPassGeometryChecker()
    result = plan(
        _two_ee_problem(in_search_geometry="full"), geometry_checker=checker
    )
    assert result.status is PlanStatus.SUCCESS
    assert checker.candidate_calls


def test_motion_cost_result_rejects_negative_cost() -> None:
    MotionCostResult(cost=0)
    with pytest.raises(ValueError):
        MotionCostResult(cost=-1)


def test_as_scene_ref_accepts_a_bare_signature() -> None:
    ref = as_scene_ref("sym-abc")
    assert ref.signature == "sym-abc"
    assert ref.completed_subgoals == ()
    assert as_scene_ref(ref) is ref
