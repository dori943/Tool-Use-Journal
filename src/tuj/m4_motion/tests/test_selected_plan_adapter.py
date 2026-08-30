from __future__ import annotations

import math

import pytest

from tuj.m4_motion.schema import (
    GoalType,
    MotionConstraints,
    PlannerOptions,
    RobotState,
    SceneRef,
    WorldSnapshot,
)
from tuj.m4_motion.selected_plan_adapter import (
    SelectedPlanAdapterError,
    SelectedPlanMotionRequestAdapter,
)
from tuj.m3_taskplanner.models import GraspSpec
from tuj.m3_taskplanner.serialization import (
    CandidateAssignment,
    CostVectorModel,
    PlanStep,
    SelectedPlan,
)


def _world(signature: str, joints: list[float]) -> WorldSnapshot:
    return WorldSnapshot(
        scene=SceneRef(signature=signature),
        robot_state=RobotState(
            robot_id="ur5e",
            joint_names=["shoulder", "elbow"],
            joint_positions_rad=joints,
        ),
        objects={
            "part": {
                "frame_id": "world",
                "position_m": [0.4, 0.1, 0.2],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "bin": {},
        },
    )


def _selected_plan() -> SelectedPlan:
    place_parameters = {
        "target_pose": {
            "frame": "bin",
            "position_mm": [10.0, 20.0, 30.0],
            "yaw_deg": 90.0,
        },
        "approach_distance_mm": 40.0,
    }
    grasp = GraspSpec(
        grasp_id="grasp-1",
        owner_kind="object",
        owner_id="part",
        pose={
            "frame_id": "world",
            "position_m": [0.4, 0.1, 0.2],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    )
    return SelectedPlan(
        cost_vector=CostVectorModel(),
        subgoal_order=["sg-place", "sg-pick"],
        candidate_assignments=[
            CandidateAssignment(
                subgoal_id="sg-place",
                candidate_id="candidate-place",
                ee="2F",
                action_type="place",
                mode="in_region",
                source_binding={"?target": "part", "?region": "bin"},
                target_ids=["part"],
                goal_region_id="bin",
                action_parameters=place_parameters,
            ),
            CandidateAssignment(
                subgoal_id="sg-pick",
                candidate_id="candidate-pick",
                ee="3F",
                tool="finger-tip",
                action_type="acquire",
                target_ids=["part"],
                grasp_id="grasp-1",
                grasp=grasp,
            ),
        ],
        steps=[
            PlanStep(
                step_index=0,
                kind="transition",
                action="ATTACH_EE",
                subgoal_id="sg-place",
                candidate_id="candidate-place",
            ),
            PlanStep(
                step_index=1,
                kind="subgoal",
                action="EXECUTE_SUBGOAL",
                subgoal_id="sg-place",
                candidate_id="candidate-place",
                parameters={"action_parameters": place_parameters},
            ),
            PlanStep(
                step_index=2,
                kind="subgoal",
                action="EXECUTE_SUBGOAL",
                subgoal_id="sg-pick",
                candidate_id="candidate-pick",
            ),
        ],
    )


def test_converts_selected_order_to_grounded_motion_requests() -> None:
    selected = _selected_plan()
    constraints = MotionConstraints(collision_margin_m=0.01)
    options = PlannerOptions(random_seed=17)

    requests = SelectedPlanMotionRequestAdapter().convert(
        selected,
        worlds={
            "sg-place": _world("scene:place", [0.0, 0.1]),
            "sg-pick": _world("scene:pick", [0.2, 0.3]),
        },
        constraints=constraints,
        options=options,
        selected_plan_artifact_id="selected-plan:42",
    )

    assert [request.task.subgoal_id for request in requests] == [
        "sg-place",
        "sg-pick",
    ]
    assert requests[0].task.goal.goal_type is GoalType.POSE
    assert requests[0].task.action_type == "place"
    assert requests[0].task.goal.target_pose is not None
    assert requests[0].task.goal.target_pose.frame_id == "object:bin"
    assert requests[0].task.goal.target_pose.position_m == (0.01, 0.02, 0.03)
    assert requests[0].task.goal.target_pose.orientation_xyzw[2] == pytest.approx(
        math.sqrt(0.5)
    )
    assert requests[0].task.goal.approach_distance_m == pytest.approx(0.04)
    assert requests[0].task.metadata["mode"] == "in_region"
    assert requests[0].task.metadata["source_binding"] == {
        "?target": "part",
        "?region": "bin",
    }
    assert requests[0].task.metadata["task_planner_steps"][0]["action"] == "ATTACH_EE"

    assert requests[1].task.goal.goal_type is GoalType.POSE
    assert requests[1].task.action_type == "acquire"
    assert requests[1].task.grasp is not None
    assert requests[1].task.grasp.grasp_id == "grasp-1"
    assert requests[1].task.allowed_touch_objects == ["part"]
    assert requests[1].world.robot_state.joint_positions_rad == [0.2, 0.3]

    assert requests[0].constraints == constraints
    assert requests[1].constraints == constraints
    assert requests[0].options == options
    assert requests[1].options == options
    assert requests[0].provenance.input_artifact_ids == ["selected-plan:42"]
    assert requests[0].request_id != requests[1].request_id


def test_rejects_one_stale_world_for_multiple_subgoals() -> None:
    with pytest.raises(SelectedPlanAdapterError, match="distinct WorldSnapshot"):
        SelectedPlanMotionRequestAdapter().convert(
            _selected_plan(),
            worlds=_world("scene:one", [0.0, 0.0]),
            constraints=MotionConstraints(),
        )


def test_acquire_requires_structured_grasp() -> None:
    selected = _selected_plan()
    selected.candidate_assignments[1].grasp = None
    selected.candidate_assignments[1].grasp_id = None

    with pytest.raises(SelectedPlanAdapterError, match="requires a structured grasp"):
        SelectedPlanMotionRequestAdapter().convert(
            selected,
            worlds={
                "sg-place": _world("scene:place", [0.0, 0.1]),
                "sg-pick": _world("scene:pick", [0.2, 0.3]),
            },
            constraints=MotionConstraints(),
        )


def test_joint_target_uses_snapshot_robot_dof() -> None:
    selected = _selected_plan()
    assignment = selected.candidate_assignments[0]
    assignment.action_type = "move_joint"
    assignment.action_parameters = {"target_joint_positions_rad": [0.5, -0.5]}
    execution = next(
        step
        for step in selected.steps
        if step.subgoal_id == "sg-place" and step.kind == "subgoal"
    )
    execution.parameters = {}

    requests = SelectedPlanMotionRequestAdapter().convert(
        selected,
        worlds={
            "sg-place": _world("scene:place", [0.0, 0.1]),
            "sg-pick": _world("scene:pick", [0.2, 0.3]),
        },
        constraints=MotionConstraints(),
    )

    assert requests[0].task.goal.goal_type is GoalType.JOINT
    assert requests[0].task.goal.target_joint_positions_rad == [0.5, -0.5]
