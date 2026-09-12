"""Stack placement succeeds on support, not on containment.

A sandwich cannot satisfy ``target_fully_inside_region`` at every layer: the
top slice of bread (113.1 x 113.3 mm) is placed onto the filling (turkey is
80.0 x 50.0 mm), and no ordering of the ingredients avoids that, because some
layer is always wider than the one beneath it.  These tests pin the predicate
that replaces containment for ``place_on`` and keep its negative cases.
"""
from __future__ import annotations

from tuj.m5_motion.contact_evaluation import StackPlacementEvaluator
from tuj.m5_motion.execution import GoalEvaluationStatus
from tuj.m5_motion.push_to_region import (
    target_fully_inside_region,
    target_resting_on_region,
)
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    GoalType,
    ModuleName,
    MotionGoal,
    MotionPlanRequest,
    MotionTask,
    RobotState,
    SceneRef,
    WorldSnapshot,
)

_BASE_SIZE_M = (.0632, .0613, .004)      # cheese_1
_WIDE_SIZE_M = (.080, .050, .00252)      # turkey_1, wider than the cheese


def _pose(position):
    return {"frame_id": "world", "position_m": list(position),
            "orientation_xyzw": [0., 0., 0., 1.]}


def _world(*, gap_m: float = 0., offset_x_m: float = 0.) -> WorldSnapshot:
    base_z = .93
    base_top = base_z + _BASE_SIZE_M[2] / 2
    return WorldSnapshot(
        scene=SceneRef(signature="stack-scene"),
        robot_state=RobotState(robot_id="robot", joint_names=["j1"],
                               joint_positions_rad=[0.]),
        objects={
            "base": {"pose": _pose((0., 0., base_z)),
                     "dimensions_m": list(_BASE_SIZE_M)},
            "slice": {"pose": _pose((offset_x_m, 0.,
                                     base_top + gap_m + _WIDE_SIZE_M[2] / 2)),
                      "dimensions_m": list(_WIDE_SIZE_M)},
        },
    )


def _request() -> MotionPlanRequest:
    return MotionPlanRequest(
        request_id="stack-request",
        provenance=ArtifactProvenance(
            artifact_id="stack-request-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.MOTION_PLANNER,
            invocation_id="stack-test",
        ),
        world=_world(),
        task=MotionTask(
            task_id="SG1_s4c",
            subgoal_id="SG1_s4c",
            action_type="place_on",
            ee="vac",
            target_ids=["slice"],
            goal=MotionGoal(goal_type=GoalType.POSE, target_region_id="base"),
        ),
    )


def test_wider_slice_rests_on_base_though_containment_fails():
    world = _world()
    assert not target_fully_inside_region(
        world, target_id="slice", region_id="base")
    assert target_resting_on_region(
        world, target_id="slice", region_id="base")


def test_floating_slice_is_not_resting():
    assert not target_resting_on_region(
        _world(gap_m=.05), target_id="slice", region_id="base")


def test_slice_pushed_off_the_base_is_not_supported():
    # The centre must stay over the base; half of the base is 31.6 mm.
    assert not target_resting_on_region(
        _world(offset_x_m=.06), target_id="slice", region_id="base")
    assert target_resting_on_region(
        _world(offset_x_m=.02), target_id="slice", region_id="base")


def test_evaluator_reports_support_for_a_wider_slice():
    evaluation = StackPlacementEvaluator().evaluate(_request(), None, _world())
    assert evaluation.status is GoalEvaluationStatus.SATISFIED
    assert evaluation.observed["resting_target_ids"] == ["slice"]
    assert evaluation.observed["predicate"] == "STACK_SUPPORT"


def test_evaluator_rejects_a_slice_that_fell_off():
    evaluation = StackPlacementEvaluator().evaluate(
        _request(), None, _world(offset_x_m=.06))
    assert evaluation.status is GoalEvaluationStatus.FAILED
    assert evaluation.observed["unsupported_target_ids"] == ["slice"]


_DISH_SIZE_M = (.1823, .1818, .0111)     # serving_plate
_DISH_RIM_TO_FLOOR_M = .0098             # measured from its collision points


def _dish_world(*, sink_m: float = 0.) -> WorldSnapshot:
    """A rimmed dish whose food surface is below its bbox top."""
    dish_z = .9255
    half = _DISH_SIZE_M[2] / 2
    floor_local = half - _DISH_RIM_TO_FLOOR_M
    # Rim vertices ring the outside; floor vertices sit lower and inboard.
    points = []
    for sign_x in (-1., 1.):
        for sign_y in (-1., 1.):
            points.append([sign_x * _DISH_SIZE_M[0] / 2,
                           sign_y * _DISH_SIZE_M[1] / 2, half])
    for offset_x in (-.02, 0., .02):
        for offset_y in (-.02, 0., .02):
            points.append([offset_x, offset_y, floor_local])
    bread = (.1131, .1133, .02292)
    return WorldSnapshot(
        scene=SceneRef(signature="dish-scene"),
        robot_state=RobotState(robot_id="robot", joint_names=["j1"],
                               joint_positions_rad=[0.]),
        objects={
            "dish": {"pose": _pose((0., 0., dish_z)),
                     "dimensions_m": list(_DISH_SIZE_M),
                     "collision_points_m": points},
            "bread": {"pose": _pose((0., 0.,
                                     dish_z + floor_local - sink_m
                                     + bread[2] / 2)),
                      "dimensions_m": list(bread)},
        },
    )


def test_bread_on_a_dish_floor_rests_despite_the_higher_rim():
    # The bbox top is the rim, 9.8 mm above the face the bread rests on.
    # Measuring against the rim would report the bread as sunk into the dish.
    assert target_resting_on_region(
        _dish_world(), target_id="bread", region_id="dish")


def test_bread_sunk_through_the_dish_floor_is_rejected():
    assert not target_resting_on_region(
        _dish_world(sink_m=.015), target_id="bread", region_id="dish")
