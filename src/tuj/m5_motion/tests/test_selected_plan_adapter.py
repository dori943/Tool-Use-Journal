from __future__ import annotations

import math

import pytest

from tuj.m5_motion.schema import (
    GoalType,
    MotionConstraints,
    PlannerOptions,
    RobotState,
    SceneRef,
    WorldSnapshot,
)
from tuj.m5_motion.selected_plan_adapter import (
    SelectedPlanAdapterError,
    SelectedPlanMotionRequestAdapter,
)
from tuj.m4_taskplanner.models import GraspSpec
from tuj.m4_taskplanner.serialization import (
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
                ee_capabilities=["multi_finger_contact"],
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


def test_acquire_without_structured_grasp_is_grounded_for_m5_planning() -> None:
    selected = _selected_plan()
    selected.candidate_assignments[1].grasp = None
    selected.candidate_assignments[1].grasp_id = None

    requests = SelectedPlanMotionRequestAdapter().convert(
        selected,
        worlds={
            "sg-place": _world("scene:place", [0.0, 0.1]),
            "sg-pick": _world("scene:pick", [0.2, 0.3]),
        },
        constraints=MotionConstraints(),
    )

    assert requests[1].task.grasp is None
    assert requests[1].task.goal.target_object_id == "part"
    assert requests[1].task.goal.target_pose is not None


def test_promotes_physical_grasp_metadata_for_two_finger_acquire() -> None:
    selected = _selected_plan()
    assignment = selected.candidate_assignments[1]
    assignment.ee = "2F"
    assignment.action_parameters = {
        "grasp_execution_mode": "CONTACT_FRICTION",
        "grasp_profile": {"minimum_lift_m": 0.07},
    }

    requests = SelectedPlanMotionRequestAdapter(
        acquire_task_metadata={"grasp_execution_mode": "KINEMATIC"}
    ).convert(
        selected,
        worlds={
            "sg-place": _world("scene:place", [0.0, 0.1]),
            "sg-pick": _world("scene:pick", [0.2, 0.3]),
        },
        constraints=MotionConstraints(),
    )

    metadata = requests[1].task.metadata
    assert metadata["grasp_execution_mode"] == "CONTACT_FRICTION"
    assert metadata["grasp_profile"] == {"minimum_lift_m": 0.07}
    assert metadata["operation"] == "ACQUIRE"
    assert metadata["attach_target"] is True


def test_promotes_explicit_pick_support_collision_policy() -> None:
    selected = _selected_plan()
    selected.candidate_assignments[1].action_parameters = {
        "support_collision_selectors": ["fixture-top"],
        "support_contact_tolerance_m": 0.002,
        "support_penetration_tolerance_m": 0.0005,
    }

    request = SelectedPlanMotionRequestAdapter().convert(
        selected,
        worlds={
            "sg-place": _world("scene:place", [0.0, 0.1]),
            "sg-pick": _world("scene:pick", [0.2, 0.3]),
        },
        constraints=MotionConstraints(),
    )[1]

    assert request.task.metadata["support_collision_selectors"] == [
        "fixture-top"
    ]
    assert request.task.metadata["support_contact_tolerance_m"] == pytest.approx(
        0.002
    )
    assert request.task.metadata[
        "support_penetration_tolerance_m"
    ] == pytest.approx(0.0005)


def test_infers_pick_support_from_grounded_geometry_without_name_policy() -> None:
    selected = _selected_plan()
    pick_world = _world("scene:pick", [0.2, 0.3])
    pick_world.objects["part"] = {
        "pose": {
            "frame_id": "world",
            "position_m": [0.4, 0.1, 0.05],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "dimensions_m": [0.2, 0.1, 0.1],
        "anchors": {"center": [0.0, 0.0, 0.0]},
    }
    pick_world.obstacles = [
        {
            "obstacle_id": "fixture-top",
            "aabb_min_m": [0.0, -0.5, -0.1],
            "aabb_max_m": [1.0, 0.5, 0.0],
            "collision_enabled": True,
        }
    ]

    request = SelectedPlanMotionRequestAdapter().convert(
        selected,
        worlds={
            "sg-place": _world("scene:place", [0.0, 0.1]),
            "sg-pick": pick_world,
        },
        constraints=MotionConstraints(collision_margin_m=0.005),
    )[1]

    metadata = request.task.metadata
    assert metadata["support_collision_selectors"] == ["fixture-top"]
    assert metadata["support_collision_policy"] == "AUTO_INITIAL_SUPPORT_V1"
    assert metadata["support_collision_detection_source"] == (
        "world.obstacles.aabb"
    )
    assert metadata["support_initial_clearance_m"] == pytest.approx(0.0)
    assert metadata["support_horizontal_overlap_ratio"] == pytest.approx(1.0)
    assert metadata["support_min_horizontal_overlap_ratio"] == pytest.approx(
        0.5
    )


def test_does_not_infer_support_from_a_sliver_aabb_overlap() -> None:
    selected = _selected_plan()
    pick_world = _world("scene:pick", [0.2, 0.3])
    pick_world.objects["part"] = {
        "pose": {
            "frame_id": "world",
            "position_m": [0.4, 0.1, 0.05],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "dimensions_m": [0.2, 0.1, 0.1],
    }
    pick_world.obstacles = [
        {
            "obstacle_id": "neighbor-edge",
            "aabb_min_m": [0.49, 0.05, -0.1],
            "aabb_max_m": [0.7, 0.15, 0.0],
            "collision_enabled": True,
        }
    ]

    request = SelectedPlanMotionRequestAdapter().convert(
        selected,
        worlds={
            "sg-place": _world("scene:place", [0.0, 0.1]),
            "sg-pick": pick_world,
        },
        constraints=MotionConstraints(collision_margin_m=0.005),
    )[1]

    assert "support_collision_selectors" not in request.task.metadata


def test_inference_skips_sliver_and_selects_valid_support_candidate() -> None:
    selected = _selected_plan()
    pick_world = _world("scene:pick", [0.2, 0.3])
    pick_world.objects["part"] = {
        "pose": {
            "frame_id": "world",
            "position_m": [0.4, 0.1, 0.05],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "dimensions_m": [0.2, 0.1, 0.1],
    }
    pick_world.obstacles = [
        {
            "obstacle_id": "neighbor-edge",
            "aabb_min_m": [0.49, 0.05, -0.1],
            "aabb_max_m": [0.7, 0.15, 0.0],
            "collision_enabled": True,
        },
        {
            "obstacle_id": "real-support",
            "aabb_min_m": [0.0, -0.5, -0.1],
            "aabb_max_m": [1.0, 0.5, -0.0005],
            "collision_enabled": True,
        },
    ]

    request = SelectedPlanMotionRequestAdapter().convert(
        selected,
        worlds={
            "sg-place": _world("scene:place", [0.0, 0.1]),
            "sg-pick": pick_world,
        },
        constraints=MotionConstraints(collision_margin_m=0.005),
    )[1]

    # Primary remains real-support; the coplanar sliver neighbor is also
    # listed so a seam-like mesh contact cannot reject LIFT.
    assert request.task.metadata["support_collision_selectors"] == [
        "neighbor-edge",
        "real-support",
    ]
    assert request.task.metadata["support_initial_clearance_m"] == pytest.approx(
        0.0
    )
    assert request.task.metadata["support_horizontal_overlap_ratio"] == pytest.approx(
        1.0
    )


def test_infers_all_coplanar_tiled_supports_for_pick_acm() -> None:
    selected = _selected_plan()
    pick_world = _world("scene:pick", [0.2, 0.3])
    # Object straddles the seam between two coplanar tiles.
    pick_world.objects["part"] = {
        "pose": {
            "frame_id": "world",
            "position_m": [0.5, 0.0, 0.05],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "dimensions_m": [0.2, 0.2, 0.1],
    }
    pick_world.obstacles = [
        {
            "obstacle_id": "counter_tile_a",
            "aabb_min_m": [0.0, -0.5, -0.1],
            "aabb_max_m": [1.0, 0.0, 0.0],
            "collision_enabled": True,
        },
        {
            "obstacle_id": "counter_tile_b",
            "aabb_min_m": [0.0, 0.0, -0.1],
            "aabb_max_m": [1.0, 0.5, 0.0],
            "collision_enabled": True,
        },
        {
            "obstacle_id": "lower_shelf",
            "aabb_min_m": [0.0, -0.5, -0.5],
            "aabb_max_m": [1.0, 0.5, -0.2],
            "collision_enabled": True,
        },
    ]

    request = SelectedPlanMotionRequestAdapter().convert(
        selected,
        worlds={
            "sg-place": _world("scene:place", [0.0, 0.1]),
            "sg-pick": pick_world,
        },
        constraints=MotionConstraints(collision_margin_m=0.005),
    )[1]

    assert request.task.metadata["support_collision_selectors"] == [
        "counter_tile_a",
        "counter_tile_b",
    ]
    assert "lower_shelf" not in request.task.metadata["support_collision_selectors"]
    assert request.task.metadata["support_collision_policy"] == "AUTO_INITIAL_SUPPORT_V1"


def test_normalizes_contact_friction_mode_before_support_inference() -> None:
    selected = _selected_plan()
    selected.candidate_assignments[1].action_parameters = {
        "grasp_execution_mode": " contact-friction ",
    }
    pick_world = _world("scene:pick", [0.2, 0.3])
    pick_world.objects["part"] = {
        "pose": {
            "frame_id": "world",
            "position_m": [0.4, 0.1, 0.05],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "dimensions_m": [0.2, 0.1, 0.1],
    }
    pick_world.obstacles = [
        {
            "obstacle_id": "fixture-top",
            "aabb_min_m": [0.0, -0.5, -0.1],
            "aabb_max_m": [1.0, 0.5, 0.0],
            "collision_enabled": True,
        }
    ]

    request = SelectedPlanMotionRequestAdapter().convert(
        selected,
        worlds={
            "sg-place": _world("scene:place", [0.0, 0.1]),
            "sg-pick": pick_world,
        },
        constraints=MotionConstraints(),
    )[1]

    assert request.task.metadata["grasp_execution_mode"] == "CONTACT_FRICTION"
    assert "support_collision_selectors" not in request.task.metadata


def test_acquire_execution_metadata_changes_request_identity() -> None:
    selected = _selected_plan()
    selected.candidate_assignments[1].ee = "2F"
    worlds = {
        "sg-place": _world("scene:place", [0.0, 0.1]),
        "sg-pick": _world("scene:pick", [0.2, 0.3]),
    }
    kinematic = SelectedPlanMotionRequestAdapter(
        acquire_task_metadata={"grasp_execution_mode": "KINEMATIC"}
    ).convert(selected, worlds=worlds, constraints=MotionConstraints())[1]
    physical = SelectedPlanMotionRequestAdapter(
        acquire_task_metadata={
            "grasp_execution_mode": "CONTACT_FRICTION",
            "grasp_profile": {"minimum_lift_m": 0.07},
        }
    ).convert(selected, worlds=worlds, constraints=MotionConstraints())[1]

    assert kinematic.request_id != physical.request_id
    assert kinematic.provenance.artifact_id != physical.provenance.artifact_id


def test_adapter_preserves_object_independent_clearance_requirements() -> None:
    selected = _selected_plan()
    requirements = {
        "required_contact_clearance_m": 0.02,
        "uses_underside_contact": False,
    }
    selected.candidate_assignments[1].action_parameters = {
        "grasp_clearance_requirements": requirements,
    }

    requests = SelectedPlanMotionRequestAdapter().convert(
        selected,
        worlds={
            "sg-place": _world("scene:place", [0.0, 0.1]),
            "sg-pick": _world("scene:pick", [0.2, 0.3]),
        },
        constraints=MotionConstraints(),
    )

    assert requests[1].task.metadata["grasp_clearance_requirements"] == requirements


def test_contact_intent_allows_grounded_targets_without_scene_name_policy() -> None:
    selected = _selected_plan()
    assignment = selected.candidate_assignments[1]
    assignment.action_type = "tool_act"
    assignment.mode = "surface_contact"
    assignment.action_parameters = {
        "contact": {"primitive": "surface_contact", "maintain_contact": True}
    }

    requests = SelectedPlanMotionRequestAdapter().convert(
        selected,
        worlds={
            "sg-place": _world("scene:place", [0.0, 0.1]),
            "sg-pick": _world("scene:pick", [0.2, 0.3]),
        },
        constraints=MotionConstraints(collision_margin_m=0.013),
    )

    request = requests[1]
    assert request.task.allowed_touch_objects == ["part"]
    assert request.task.contact is not None
    assert request.task.contact.primitive == "surface_contact"
    assert request.constraints.collision_margin_m == pytest.approx(0.013)
    assert "motion_policy" not in request.task.metadata


def test_auto_grasp_mode_uses_selected_ee_capabilities() -> None:
    requests = SelectedPlanMotionRequestAdapter(
        acquire_task_metadata={"grasp_execution_mode": "AUTO"}
    ).convert(
        _selected_plan(),
        worlds={
            "sg-place": _world("scene:place", [0.0, 0.1]),
            "sg-pick": _world("scene:pick", [0.2, 0.3]),
        },
        constraints=MotionConstraints(),
    )

    assert requests[1].task.ee == "3F"
    assert requests[1].task.metadata["grasp_execution_mode"] == "KINEMATIC"
    assert requests[1].task.metadata["ee_capabilities"] == ["multi_finger_contact"]


def test_auto_grasp_mode_rejects_missing_ee_capabilities() -> None:
    selected = _selected_plan()
    selected.candidate_assignments[1].ee_capabilities = []

    with pytest.raises(SelectedPlanAdapterError, match="no EE capabilities"):
        SelectedPlanMotionRequestAdapter(
            acquire_task_metadata={"grasp_execution_mode": "AUTO"}
        ).convert(
            selected,
            worlds={
                "sg-place": _world("scene:place", [0.0, 0.1]),
                "sg-pick": _world("scene:pick", [0.2, 0.3]),
            },
            constraints=MotionConstraints(),
        )


@pytest.mark.parametrize("dimensions", [[0.18, 0.18, 0.01], [0.05, 0.05, 0.05], [0.03, 0.04, 0.2]])
def test_bbox_does_not_select_a_grasp_template_or_aperture(dimensions) -> None:
    selected = _selected_plan()
    selected.candidate_assignments[1].ee = "2F"
    pick_world = _world("scene:pick", [0.2, 0.3])
    pick_world.objects["part"].update(
        {
            "dimensions_m": dimensions,
            "anchors": {"center": [0.0, 0.0, 0.0]},
        }
    )

    requests = SelectedPlanMotionRequestAdapter(
        acquire_task_metadata={"grasp_execution_mode": "CONTACT_FRICTION"}
    ).convert(
        selected,
        worlds={
            "sg-place": _world("scene:place", [0.0, 0.1]),
            "sg-pick": pick_world,
        },
        constraints=MotionConstraints(),
    )

    metadata = requests[1].task.metadata
    assert "grasp_geometry_provider" not in metadata
    assert "grasp_preshape_aperture_m" not in metadata


def test_auto_contact_friction_detects_explicit_opposed_contact() -> None:
    selected = _selected_plan()
    selected.candidate_assignments[1].ee = "2F"
    selected.candidate_assignments[1].ee_capabilities = [
        "contact_friction",
        "opposed_finger_contact",
    ]
    pick_world = _world("scene:pick", [0.2, 0.3])
    pick_world.objects["part"].update(
        {
            "pose": {
                "position_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "anchors": {"center": [0.0, 0.0, 0.0]},
            "physical_metadata": {
                "grasp_feature": {
                    "provider": "TWO_FINGER_OPPOSED_CONTACT",
                    "contact_center_local_m": [0.0, -0.04, -0.02],
                    "closing_axis_local_xyz": [1.0, 0.0, 0.0],
                    "approach_axis_local_xyz": [0.0, 0.0, 1.0],
                    "contact_span_m": 0.015,
                }
            },
        }
    )

    requests = SelectedPlanMotionRequestAdapter(
        acquire_task_metadata={"grasp_execution_mode": "CONTACT_FRICTION"}
    ).convert(
        selected,
        worlds={
            "sg-place": _world("scene:place", [0.0, 0.1]),
            "sg-pick": pick_world,
        },
        constraints=MotionConstraints(),
    )

    metadata = requests[1].task.metadata
    assert metadata["grasp_geometry_provider"] == (
        "TWO_FINGER_OPPOSED_CONTACT"
    )
    assert metadata["grasp_preshape_aperture_m"] == pytest.approx(0.035)


@pytest.mark.parametrize(
    ("target_id", "message"),
    [
        ("?tool", "unresolved targets"),
        ("missing-object", "absent from the WorldSnapshot"),
    ],
)
def test_rejects_unresolved_or_absent_targets(
    target_id: str,
    message: str,
) -> None:
    selected = _selected_plan()
    selected.candidate_assignments[1].target_ids = [target_id]
    selected.candidate_assignments[1].grasp = None
    selected.candidate_assignments[1].grasp_id = None

    with pytest.raises(SelectedPlanAdapterError, match=message):
        SelectedPlanMotionRequestAdapter().convert(
            selected,
            worlds={
                "sg-place": _world("scene:place", [0.0, 0.1]),
                "sg-pick": _world("scene:pick", [0.2, 0.3]),
            },
            constraints=MotionConstraints(),
        )


def test_rejects_grasp_owned_by_a_different_resource() -> None:
    selected = _selected_plan()
    selected.candidate_assignments[1].grasp = GraspSpec(
        grasp_id="wrong-owner",
        owner_kind="object",
        owner_id="bin",
    )

    with pytest.raises(SelectedPlanAdapterError, match="does not match targets"):
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
