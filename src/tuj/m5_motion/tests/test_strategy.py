"""Layered IK branch search backtracks before rejecting a strategy."""

from __future__ import annotations

from tuj.m5_motion import strategy as strategy_module
from tuj.m5_motion.kinematics import IKResult, IKSolutionSet
from tuj.m5_motion.schema import (
    KeyframePlanCandidate,
    KeyframePlannerType,
    KeyframeType,
    RelativeKeyframeSpec,
    StrategyGenerationProvenance,
    StrategyGeneratorKind,
)
from tuj.m5_motion.strategy import EdgePlanResult, FirstFeasibleBranchSelector


def _keyframe(identifier: str, kind: KeyframeType) -> RelativeKeyframeSpec:
    return RelativeKeyframeSpec(
        keyframe_id=identifier,
        keyframe_type=kind,
        frame_ref="object:obj1",
        anchor="center",
        approach_axis_xyz=(0.0, 0.0, 1.0),
        planner=KeyframePlannerType.JOINT,
    )


def _strategy() -> KeyframePlanCandidate:
    return KeyframePlanCandidate(
        strategy_id="top-down",
        keyframes=[
            _keyframe("pre", KeyframeType.PRE_GRASP),
            _keyframe("grasp", KeyframeType.GRASP),
        ],
        provenance=StrategyGenerationProvenance(
            generator_kind=StrategyGeneratorKind.TEMPLATE,
            generator_id="test-template",
            input_hash="input-hash",
        ),
    )


def _four_keyframe_strategy() -> KeyframePlanCandidate:
    return KeyframePlanCandidate(
        strategy_id="flatten:1",
        keyframes=[
            _keyframe("hover", KeyframeType.CUSTOM),
            _keyframe("engage", KeyframeType.CUSTOM),
            _keyframe("press", KeyframeType.CUSTOM),
            _keyframe("unload", KeyframeType.RETREAT),
        ],
        provenance=StrategyGenerationProvenance(
            generator_kind=StrategyGeneratorKind.TEMPLATE,
            generator_id="flatten-test-template",
            input_hash="flatten-input-hash",
        ),
    )


def _solution(q0: float, branch: str) -> IKResult:
    return IKResult(
        solved=True,
        qpos=(q0, 0.0),
        position_error_m=0.0,
        orientation_error_rad=0.0,
        detail="test",
        branch_id=branch,
    )


class _BacktrackingEdgePlanner:
    def plan(self, source, target, source_keyframe, target_keyframe):
        # The nearest pre-grasp branch is a dead end to every grasp branch.
        if source_keyframe is not None and source_keyframe.keyframe_id == "pre":
            if source[0] < 0.5:
                return EdgePlanResult(
                    valid=False,
                    failure_code="PATH_COLLISION",
                    detail="near branch hits rack",
                )
        return EdgePlanResult(valid=True, joint_path=(tuple(source), tuple(target)))


def test_selector_backtracks_to_another_pregrasp_branch() -> None:
    solution_sets = [
        IKSolutionSet(
            solutions=(
                _solution(0.1, "S+_E+_W+"),
                _solution(1.0, "S-_E-_W-"),
            )
        ),
        IKSolutionSet(
            solutions=(
                _solution(0.2, "S+_E+_W+"),
                _solution(1.1, "S-_E-_W-"),
            )
        ),
    ]

    result = FirstFeasibleBranchSelector().select(
        _strategy(),
        (0.0, 0.0),
        solution_sets,
        _BacktrackingEdgePlanner(),
    )

    assert result.solved
    assert result.connected is not None
    assert result.connected.nodes[0].solution.qpos == (1.0, 0.0)
    assert result.connected.nodes[1].solution.branch_id == "S-_E-_W-"
    assert any(edge.failure_code == "PATH_COLLISION" for edge in result.rejected_edges)


def test_selector_reports_a_keyframe_with_no_ik_branch() -> None:
    result = FirstFeasibleBranchSelector().select(
        _strategy(),
        (0.0, 0.0),
        [IKSolutionSet(), IKSolutionSet(solutions=(_solution(0.2, "B"),))],
        _BacktrackingEdgePlanner(),
    )
    assert not result.solved
    assert result.failure_code == "NO_IK_BRANCH"


def test_selector_evaluates_one_complete_chain_before_wall_timeout(
    monkeypatch,
) -> None:
    strategy = _four_keyframe_strategy()
    clock = [0.0]
    monkeypatch.setattr(strategy_module.time, "monotonic", lambda: clock[0])

    class AdvancingPlanner(_BacktrackingEdgePlanner):
        def plan(self, source, target, source_keyframe, target_keyframe):
            clock[0] = 1.0
            return super().plan(source, target, source_keyframe, target_keyframe)

    solution_sets = [
        IKSolutionSet(
            solutions=(
                _solution(1.0 + index / 10, "S-_E+_W+"),
                _solution(2.0 + index / 10, "S+_E-_W-"),
            )
        )
        for index in range(len(strategy.keyframes))
    ]

    result = FirstFeasibleBranchSelector(timeout_s=0.5).select(
        strategy,
        (0.0, 0.0),
        solution_sets,
        AdvancingPlanner(),
    )

    assert result.solved
    assert result.connected is not None
    assert result.connected.edge_evaluations == len(strategy.keyframes)
    assert [node.solution.branch_id for node in result.connected.nodes] == [
        "S-_E+_W+"
    ] * len(strategy.keyframes)


def test_selector_reapplies_wall_timeout_after_first_chain_budget(
    monkeypatch,
) -> None:
    clock_calls = 0

    def monotonic() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls == 1 else 1.0

    monkeypatch.setattr(strategy_module.time, "monotonic", monotonic)
    solution_sets = [
        IKSolutionSet(
            solutions=(
                _solution(0.1, "S+_E+_W+"),
                _solution(1.0, "S-_E-_W-"),
            )
        ),
        IKSolutionSet(
            solutions=(
                _solution(0.2, "S+_E+_W+"),
                _solution(1.1, "S-_E-_W-"),
            )
        ),
    ]

    result = FirstFeasibleBranchSelector(timeout_s=0.5).select(
        _strategy(),
        (0.0, 0.0),
        solution_sets,
        _BacktrackingEdgePlanner(),
    )

    assert not result.solved
    assert result.failure_code == "SEARCH_BUDGET_EXHAUSTED"
    assert result.detail == "evaluated 2 branch edges"


def test_selector_hard_edge_cap_still_preempts_minimum_chain() -> None:
    strategy = _four_keyframe_strategy()
    solution_sets = [
        IKSolutionSet(solutions=(_solution(1.0, "S-_E+_W+"),))
        for _ in strategy.keyframes
    ]

    result = FirstFeasibleBranchSelector(
        max_edge_evaluations=2,
        timeout_s=60.0,
    ).select(
        strategy,
        (0.0, 0.0),
        solution_sets,
        _BacktrackingEdgePlanner(),
    )

    assert not result.solved
    assert result.failure_code == "SEARCH_BUDGET_EXHAUSTED"
    assert result.detail == "evaluated 2 branch edges"
