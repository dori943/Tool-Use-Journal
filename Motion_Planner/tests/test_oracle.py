"""The oracle must satisfy the Task Planner contract, including its honesty rules."""

from __future__ import annotations

import pytest

from task_planner.feasibility import FeasibilityStatus
from task_planner.models import GraspSpec
from task_planner.motion_interface import (
    CandidateQuery,
    EEExchangeQuery,
    ResourceState,
    SceneRef,
    TerminalQuery,
    TransitionQuery,
    WorldSnapshot,
)

from motion_planner import MuJoCoMotionOracle, UR5eKinematics

SCENE = SceneRef(signature="sym-test", completed_subgoals=(), facts=())

# Comfortably inside the UR5e envelope with the demo base offset at (-0.45,0,0).
IN_REACH = [0.0, 0.2, 0.3]
OUT_OF_REACH = [3.0, 0.0, 0.5]


@pytest.fixture(scope="module")
def kinematics() -> UR5eKinematics:
    return UR5eKinematics()


def _oracle(kinematics, **world) -> MuJoCoMotionOracle:
    oracle = MuJoCoMotionOracle(kinematics)
    oracle.initialize(WorldSnapshot(scene=SCENE, **world))
    return oracle


def _candidate(pose, candidate_id="c1", targets=()) -> CandidateQuery:
    return CandidateQuery(
        scene=SCENE,
        subgoal_id="S1",
        candidate_id=candidate_id,
        ee="2f",
        tool=None,
        grasp_id="g1",
        grasp=GraspSpec(grasp_id="g1", owner_kind="object", owner_id="obj1", pose=pose),
        target_ids=tuple(targets),
    )


def test_reachable_grasp_pose_passes(kinematics) -> None:
    oracle = _oracle(kinematics)
    result = oracle.check_candidate(_candidate(IN_REACH))
    assert result.status is FeasibilityStatus.PASS


def test_unreachable_grasp_pose_fails_with_ik_reason(kinematics) -> None:
    oracle = _oracle(kinematics)
    result = oracle.check_candidate(_candidate(OUT_OF_REACH))
    assert result.status is FeasibilityStatus.FAIL
    assert result.reason_code is not None
    assert result.reason_code.value == "IK_UNREACHABLE"


def test_missing_pose_is_unknown_never_pass(kinematics) -> None:
    """The contract's central honesty rule: no data means UNKNOWN."""
    oracle = _oracle(kinematics)
    result = oracle.check_candidate(_candidate(None))
    assert result.status is FeasibilityStatus.UNKNOWN


def test_object_pose_from_world_model_is_used_when_grasp_has_none(kinematics) -> None:
    oracle = _oracle(kinematics, objects={"obj1": {"position": IN_REACH}})
    result = oracle.check_candidate(_candidate(None, targets=("obj1",)))
    assert result.status is FeasibilityStatus.PASS


def test_ee_exchange_checks_both_rack_slots(kinematics) -> None:
    oracle = _oracle(
        kinematics,
        rack={"2f": {"dock_pose": IN_REACH}, "vacuum": {"dock_pose": OUT_OF_REACH}},
    )
    query = EEExchangeQuery(scene=SCENE, from_ee="2f", to_ee="vacuum")
    result = oracle.check_ee_exchange(query)
    assert result.status is FeasibilityStatus.FAIL
    assert result.reason_code.value == "RACK_SLOT_UNAVAILABLE"

    reachable = EEExchangeQuery(scene=SCENE, from_ee="2f", to_ee="2f")
    assert oracle.check_ee_exchange(reachable).status is FeasibilityStatus.PASS


def test_ee_exchange_without_rack_data_is_unknown(kinematics) -> None:
    oracle = _oracle(kinematics)
    query = EEExchangeQuery(scene=SCENE, from_ee="2f", to_ee="vacuum")
    assert oracle.check_ee_exchange(query).status is FeasibilityStatus.UNKNOWN


def test_terminal_restore_reuses_the_exchange_check(kinematics) -> None:
    oracle = _oracle(
        kinematics,
        rack={"2f": {"dock_pose": IN_REACH}, "vacuum": {"dock_pose": OUT_OF_REACH}},
    )
    query = TerminalQuery(
        scene=SCENE,
        from_state=ResourceState(current_ee="2f"),
        restore_ee="vacuum",
    )
    assert oracle.check_terminal(query).status is FeasibilityStatus.FAIL


def test_unimplemented_transition_check_is_unknown_not_pass(kinematics) -> None:
    oracle = _oracle(kinematics)
    query = TransitionQuery(
        scene=SCENE,
        from_state=ResourceState(current_ee="2f"),
        candidate=_candidate(IN_REACH),
    )
    assert oracle.check_transition(query).status is FeasibilityStatus.UNKNOWN


def test_results_are_deterministic(kinematics) -> None:
    """Task Planner caches by query identity, so repeats must agree exactly."""
    oracle = _oracle(kinematics)
    query = _candidate(IN_REACH)
    statuses = {oracle.check_candidate(query).status for _ in range(5)}
    assert len(statuses) == 1


def test_ik_is_only_called_when_a_pose_exists(kinematics) -> None:
    oracle = _oracle(kinematics)
    oracle.check_candidate(_candidate(None))
    assert oracle.ik_calls == 0
    oracle.check_candidate(_candidate(IN_REACH))
    assert oracle.ik_calls == 1
