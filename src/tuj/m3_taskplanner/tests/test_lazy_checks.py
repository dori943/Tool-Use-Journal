"""Lazy geometric checks: expansion-only calls, caching, scene sensitivity.

These exercise ``in_search_geometry="full"``, which is no longer the default.
The default (``scene_independent``) deliberately keeps scene-dependent checks
out of the search; see ``test_in_search_geometry_scope`` below for that mode.
"""

from __future__ import annotations

from tuj.m3_taskplanner.diagnostics import PlanStatus, ReasonCode
from tuj.m3_taskplanner.feasibility import (
    AlwaysPassGeometryChecker,
    CheckResult,
    FeasibilityStatus,
    TableGeometryChecker,
)
from tuj.m3_taskplanner.models import PlanningPolicy
from tuj.m3_taskplanner.planner import plan

from conftest import make_request, prop, sg

FULL = PlanningPolicy(in_search_geometry="full")


def _chain_request(initial_ee: str = "A", policy: PlanningPolicy | None = None):
    """S1 has a cheap (A) and an expensive (B) candidate; S2 needs A."""
    subgoals = [
        sg("S1", targets=["obj1"], feasible=["A", "B"]),
        sg("S2", targets=["obj2"], feasible=["A"]),
    ]
    proposals = {
        "S1": [
            prop("S1-cA", "S1", "A"),
            prop("S1-cB", "S1", "B"),
        ],
        "S2": [prop("S2-cA", "S2", "A")],
    }
    return make_request(
        subgoals,
        edges=[("S1", "S2")],
        proposals=proposals,
        initial_ee=initial_ee,
        policy=policy or FULL,
    )


def test_unexpanded_states_never_call_the_geometry_checker() -> None:
    checker = AlwaysPassGeometryChecker()
    result = plan(_chain_request(), geometry_checker=checker)
    assert result.status is PlanStatus.SUCCESS
    # From the initial state both S1 candidates are checked; S2 is only ever
    # checked from the cheap (A) successor. The expensive B successor is never
    # popped, so no transition from it is validated.
    checked_candidates = {cid for _, cid in checker.candidate_calls}
    assert checked_candidates == {"S1-cA", "S1-cB", "S2-cA"}
    assert len(checker.transition_calls) == 3
    froms = {(ee, cid) for _, ee, _, cid in checker.transition_calls}
    assert ("B", "S2-cA") not in froms  # S2 never attempted from the B branch


def test_same_state_candidate_check_result_is_cached() -> None:
    # Both S1 branches (A and B) are expanded before the goal state is popped,
    # so S2 is looked up twice in the same predicted symbolic scene: once
    # executed, once served from the (scene, candidate) cache.
    checker = AlwaysPassGeometryChecker()
    subgoals = [
        sg("S1", targets=["obj1"], feasible=["A", "B"]),
        sg("S2", targets=["obj2"], feasible=["A"]),
    ]
    proposals = {
        "S1": [
            prop("S1-cA", "S1", "A"),
            prop("S1-cB", "S1", "B"),
        ],
        "S2": [prop("S2-cA", "S2", "A")],
    }
    request = make_request(
        subgoals,
        edges=[("S1", "S2")],
        proposals=proposals,
        initial_ee="B",
        policy=FULL,
    )
    result = plan(request, geometry_checker=checker)
    assert result.status is PlanStatus.SUCCESS
    # S2's candidate check happens once per predicted scene; the second
    # arrival (from the other S1 branch, same symbolic scene) is a cache hit.
    s2_candidate_checks = [
        c for c in checker.candidate_calls if c[1] == "S2-cA"
    ]
    assert len(s2_candidate_checks) == 1
    assert result.search_stats.lazy_cache_hits >= 1


def test_different_scene_signature_triggers_a_new_check() -> None:
    # S1 and S2 are independent, so S1's candidate is validated both in the
    # initial scene and in the post-S2 scene.
    checker = AlwaysPassGeometryChecker()
    subgoals = [
        sg("S1", targets=["obj1"], feasible=["A"]),
        sg("S2", targets=["obj2"], feasible=["A"]),
    ]
    proposals = {
        "S1": [prop("S1-cA", "S1", "A")],
        "S2": [prop("S2-cA", "S2", "A")],
    }
    result = plan(
        make_request(subgoals, proposals=proposals, policy=FULL),
        geometry_checker=checker,
    )
    assert result.status is PlanStatus.SUCCESS
    s1_scenes = {scene for scene, cid in checker.candidate_calls if cid == "S1-cA"}
    assert len(s1_scenes) == 2


def test_geometry_failure_removes_edge_and_is_reported() -> None:
    failing = TableGeometryChecker(
        candidate_rules={
            ("*", "S1-cA"): CheckResult(
                FeasibilityStatus.FAIL,
                ReasonCode.IK_UNREACHABLE,
                "mock: unreachable",
            )
        }
    )
    result = plan(_chain_request(), geometry_checker=failing)
    # The A candidate is pruned lazily; the plan must route through B.
    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    s1_assignment = next(
        a
        for a in result.selected_plan.candidate_assignments
        if a.subgoal_id == "S1"
    )
    assert s1_assignment.candidate_id == "S1-cB"
    assert any(
        r.reason_code is ReasonCode.IK_UNREACHABLE and r.candidate_id == "S1-cA"
        for r in result.rejections
    )
    # Candidate failure short-circuits the transition checker for that edge.
    assert not any(call[-1] == "S1-cA" for call in failing.transition_calls)


def test_geometry_failure_of_only_candidate_is_infeasible_no_plan() -> None:
    failing = TableGeometryChecker(
        candidate_rules={
            ("*", "S2-cA"): CheckResult(
                FeasibilityStatus.FAIL,
                ReasonCode.GEOMETRY_COLLISION,
                "mock: collision",
            )
        }
    )
    result = plan(_chain_request(), geometry_checker=failing)
    assert result.status is PlanStatus.INFEASIBLE_NO_PLAN
    assert result.selected_plan is None
    assert any(
        r.reason_code is ReasonCode.GEOMETRY_COLLISION for r in result.rejections
    )
    assert result.diagnostics.frontier_exhausted


# --------------------------------------------------------------------------- #
# in_search_geometry scope                                                    #
# --------------------------------------------------------------------------- #


def test_default_mode_skips_scene_dependent_checks() -> None:
    """Candidate/transition geometry is deferred to post-search verification."""
    checker = AlwaysPassGeometryChecker()
    request = _chain_request(policy=PlanningPolicy())  # default
    result = plan(request, geometry_checker=checker)

    assert result.status is PlanStatus.SUCCESS
    assert checker.candidate_calls == []
    assert checker.transition_calls == []
    assert result.search_stats.lazy_checks_executed == 0


def test_default_mode_still_validates_the_ee_exchange() -> None:
    """The one check that survives: it is scene-independent and decisive."""

    class BlockedRack:
        """Docking EE 'A' is impossible; only the exchange hook is implemented."""

        def __init__(self) -> None:
            self.exchanges: list[tuple[str, str]] = []

        def check_candidate(self, query) -> CheckResult:  # pragma: no cover
            raise AssertionError("scene-dependent check must not run in this mode")

        def check_transition(self, query) -> CheckResult:  # pragma: no cover
            raise AssertionError("scene-dependent check must not run in this mode")

        def check_ee_exchange(self, query) -> CheckResult:
            self.exchanges.append((query.from_ee, query.to_ee))
            if query.to_ee == "A":
                return CheckResult(
                    FeasibilityStatus.FAIL,
                    ReasonCode.RACK_SLOT_UNAVAILABLE,
                    "mock: slot A unreachable",
                )
            return CheckResult.ok()

        def check_terminal(self, query) -> CheckResult:
            return CheckResult.ok()

    oracle = BlockedRack()
    # Start on B so reaching S2 (A-only) requires the blocked B->A exchange.
    result = plan(_chain_request(initial_ee="B", policy=PlanningPolicy()),
                  geometry_checker=oracle)

    assert oracle.exchanges, "the EE exchange was never validated"
    assert result.status is PlanStatus.INFEASIBLE_NO_PLAN
    assert any(
        r.reason_code is ReasonCode.RACK_SLOT_UNAVAILABLE for r in result.rejections
    )


def test_none_mode_runs_no_geometry_at_all() -> None:
    checker = AlwaysPassGeometryChecker()
    request = _chain_request(policy=PlanningPolicy(in_search_geometry="none"))
    result = plan(request, geometry_checker=checker)

    assert result.status is PlanStatus.SUCCESS
    assert checker.candidate_calls == []
    assert result.search_stats.lazy_checks_executed == 0
