"""Stack placement succeeds on support, not on containment.

A sandwich cannot satisfy ``target_fully_inside_region`` at every layer: the
top slice of bread (113.1 x 113.3 mm) is placed onto the filling (turkey is
80.0 x 50.0 mm), and no ordering of the ingredients avoids that, because some
layer is always wider than the one beneath it.  These tests pin the predicate
that replaces containment for ``place_on`` and keep its negative cases.
"""
from __future__ import annotations

import pytest

from tuj.m5_motion.contact_evaluation import StackPlacementEvaluator, TaskAwareGoalEvaluator
from tuj.m5_motion.execution import GoalEvaluationStatus
from tuj.m5_motion.push_to_region import (
    target_fully_inside_region,
    target_resting_on_region,
)
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    ExecutionReport,
    ExecutionStatus,
    GoalType,
    ModuleName,
    MotionGoal,
    MotionPlanRequest,
    MotionTask,
    RobotState,
    SceneRef,
    SimulationMetrics,
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


def _tilted_mesh_world(gap_m=0.):
    import numpy as np
    from scipy.spatial.transform import Rotation
    world = _world()
    # An octahedron and its enclosing box have identical local dimensions.
    # Rotation makes the box extend below the actual contact surface.
    points = np.vstack((np.eye(3), -np.eye(3))) * .04
    rotation = Rotation.from_euler('y', 30., degrees=True)
    bottom = float(rotation.apply(points)[:, 2].min())
    top = world.objects['base']['pose']['position_m'][2] + _BASE_SIZE_M[2] / 2
    record = world.objects['slice']
    record.update(dimensions_m=[.08]*3, collision_points_m=points.tolist())
    record['pose'].update(position_m=[0., 0., top - bottom + gap_m],
                          orientation_xyzw=rotation.as_quat().tolist())
    return world


def test_tilted_irregular_mesh_uses_actual_contact_surface():
    import copy
    world = _tilted_mesh_world()
    before = copy.deepcopy(world.model_dump())
    assert target_resting_on_region(world, target_id='slice', region_id='base')
    assert world.model_dump() == before
    del world.objects['slice']['collision_points_m']
    assert not target_resting_on_region(world, target_id='slice', region_id='base')


def test_correct_mesh_contact_height_keeps_gap_and_penetration_limits():
    for gap in (.006, -.006, .02, -.02):
        assert not target_resting_on_region(_tilted_mesh_world(gap),
            target_id='slice', region_id='base')
    for gap in (.004, -.004):
        assert target_resting_on_region(_tilted_mesh_world(gap),
            target_id='slice', region_id='base')


def _released_report(world, *, attached=None):
    state = world.robot_state.model_copy(deep=True)
    state.attached_object_id = attached
    return ExecutionReport(
        report_id='stack-report', run_id='stack-run', plan_id='stack-plan',
        provenance=ArtifactProvenance(
            artifact_id='stack-report-artifact', artifact_type='ExecutionReport',
            produced_by=ModuleName.MOTION_PLANNER, invocation_id='stack-test',
        ),
        status=ExecutionStatus.SUCCESS, final_robot_state=state,
        metrics=SimulationMetrics(executed_duration_s=1., max_joint_tracking_error_rad=0.),
    )


@pytest.mark.parametrize('gap, offset, passed', [
    (0., 0., True), (.05, 0., False), (-.015, 0., False), (0., .06, False),
])
def test_task_aware_place_on_routes_to_existing_support_predicate(gap, offset, passed):
    request = _request()
    world = _world(gap_m=gap, offset_x_m=offset)
    result = TaskAwareGoalEvaluator().evaluate(request, _released_report(world), world)
    assert result.status is (GoalEvaluationStatus.SATISFIED if passed else GoalEvaluationStatus.FAILED)
    assert result.observed['predicate'] == 'STACK_SUPPORT'
    # Restore routing only; the original support tolerances remain authoritative.
    assert result.observed['maximum_gap_m'] == .005
    assert result.observed['maximum_penetration_m'] == .005


@pytest.mark.parametrize('action, expected', [
    ('place_on', GoalEvaluationStatus.SATISFIED),
    ('place', GoalEvaluationStatus.FAILED),
    ('release', GoalEvaluationStatus.FAILED),
])
def test_only_explicit_place_on_allows_a_wider_supported_layer(action, expected):
    request = _request()
    request.task.action_type = action
    world = _world()
    world.objects['slice']['dimensions_m'][:2] = [.16, .15]
    result = TaskAwareGoalEvaluator().evaluate(request, _released_report(world), world)
    assert result.status is expected
    assert (result.observed.get('predicate') == 'STACK_SUPPORT') is (action == 'place_on')


def test_place_on_requires_release_before_support_evaluation():
    request = _request()
    world = _world()
    result = TaskAwareGoalEvaluator().evaluate(
        request, _released_report(world, attached='slice'), world,
    )
    assert result.status is GoalEvaluationStatus.FAILED
    assert result.observed['attached_object_id'] == 'slice'
    assert 'still attached' in result.detail
