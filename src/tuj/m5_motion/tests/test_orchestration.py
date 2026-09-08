from __future__ import annotations

import pytest

from tuj.m5_motion.orchestration import (
    MotionPlanStore,
    SelectedPlanMotionOrchestrator,
)
from tuj.m5_motion.physical_grasp import releases_contact_friction
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    GoalType,
    InterpolationType,
    JointDynamicLimit,
    ModuleName,
    MotionConstraints,
    MotionPlan,
    RobotState,
    SceneRef,
    SegmentType,
    TrajectorySegment,
    TrajectoryWaypoint,
    WorldSnapshot,
)
from tuj.m5_motion.selected_plan_adapter import SelectedPlanMotionRequestAdapter
from tuj.m4_taskplanner.models import GraspSpec
from tuj.m4_taskplanner.serialization import (
    CandidateAssignment,
    CostVectorModel,
    PlanStep,
    SelectedPlan,
)


def _selected() -> SelectedPlan:
    grasp = GraspSpec(
        grasp_id="grasp-1",
        owner_kind="object",
        owner_id="part",
        pose={
            "frame_id": "world",
            "position_m": [0.4, 0.0, 0.2],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    )
    return SelectedPlan(
        cost_vector=CostVectorModel(),
        subgoal_order=["pick-part"],
        candidate_assignments=[
            CandidateAssignment(
                subgoal_id="pick-part",
                candidate_id="candidate-1",
                ee="3F",
                action_type="acquire",
                target_ids=["part"],
                grasp_id="grasp-1",
                grasp=grasp,
            )
        ],
        steps=[
            PlanStep(
                step_index=0,
                kind="transition",
                action="DETACH_EE",
                parameters={"ee": "2F"},
                subgoal_id="pick-part",
                candidate_id="candidate-1",
            ),
            PlanStep(
                step_index=1,
                kind="transition",
                action="ATTACH_EE",
                parameters={"ee": "3F"},
                subgoal_id="pick-part",
                candidate_id="candidate-1",
            ),
            PlanStep(
                step_index=2,
                kind="subgoal",
                action="EXECUTE_SUBGOAL",
                subgoal_id="pick-part",
                candidate_id="candidate-1",
            ),
        ],
    )


def _world() -> WorldSnapshot:
    return WorldSnapshot(
        scene=SceneRef(signature="scene:initial"),
        robot_state=RobotState(
            robot_id="robot",
            joint_names=["j1"],
            joint_positions_rad=[0.0],
        ),
        objects={
            "part": {
                "pose": {
                    "frame_id": "world",
                    "position_m": [0.4, 0.0, 0.2],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        },
        metadata={"physical_active_ee": "2F"},
    )


def _fake_planner(request):
    start = request.world.robot_state.joint_positions_rad[0]
    end = start + 0.1
    attached = request.world.robot_state.attached_object_id
    if request.task.action_type.casefold() in {"pick", "acquire"}:
        attached = request.task.goal.target_object_id
    return MotionPlan(
        plan_id=f"plan:{request.request_id}",
        request_id=request.request_id,
        provenance=ArtifactProvenance(
            artifact_id=f"plan-artifact:{request.request_id}",
            artifact_type="MotionPlan",
            produced_by=ModuleName.MOTION_PLANNER,
            invocation_id=f"fake:{request.request_id}",
            input_artifact_ids=[request.provenance.artifact_id],
        ),
        scene_signature=request.world.scene.signature,
        robot_id="robot",
        joint_names=["j1"],
        duration_s=1.0,
        segments=[
            TrajectorySegment(
                segment_id=f"segment:{request.request_id}",
                segment_type=SegmentType.CUSTOM,
                start_time_s=0.0,
                end_time_s=1.0,
                interpolation=InterpolationType.LINEAR,
                waypoints=[
                    TrajectoryWaypoint(
                        time_from_start_s=0.0,
                        joint_positions_rad=[start],
                    ),
                    TrajectoryWaypoint(
                        time_from_start_s=1.0,
                        joint_positions_rad=[end],
                    ),
                ],
                collision_checked=True,
            )
        ],
        expected_final_state=RobotState(
            robot_id="robot",
            joint_names=["j1"],
            joint_positions_rad=[end],
            joint_velocities_rad_s=[0.0],
            attached_object_id=attached,
        ),
    )


def test_live_executor_state_replaces_prediction_before_function():
    observed = []
    def execute(request, plan):
        world = request.world.model_copy(deep=True)
        world.robot_state.joint_positions_rad = [2.0 + len(observed)]
        world.metadata["physical_active_ee"] = (
            "3F" if request.task.action_type == "EE_EXCHANGE" else "2F")
        world.scene.signature = f"live:{len(observed)}"
        observed.append(request)
        return world
    def handler(request):
        if request.task.action_type != "acquire":
            return None
        assert request.world.robot_state.joint_positions_rad == [3.0]
        world = request.world.model_copy(deep=True)
        world.robot_state.joint_positions_rad = [4.0]
        world.robot_state.attached_object_id = "part"
        return world
    result = SelectedPlanMotionOrchestrator(_fake_planner,
        request_handler=handler, plan_executor=execute).plan(_selected(),
        initial_world=_world(), constraints=MotionConstraints(joint_limits={
            "j1": JointDynamicLimit(max_velocity_rad_s=1., max_acceleration_rad_s2=2.)}))
    assert len(result.plans) == 2
    assert result.final_world.robot_state.joint_positions_rad == [4.0]
    assert result.final_world.robot_state.attached_object_id == "part"
    assert result.final_world.scene.completed_subgoals == ["pick-part"]


def test_orchestrator_plans_transition_then_subgoal_and_persists(tmp_path) -> None:
    constraints = MotionConstraints(
        joint_limits={
            "j1": JointDynamicLimit(
                max_velocity_rad_s=1.0,
                max_acceleration_rad_s2=2.0,
            )
        }
    )
    orchestrator = SelectedPlanMotionOrchestrator(
        _fake_planner,
        store=MotionPlanStore(tmp_path),
    )

    result = orchestrator.plan(
        _selected(),
        initial_world=_world(),
        constraints=constraints,
    )

    assert [request.task.goal.goal_type for request in result.requests] == [
        GoalType.POSE,
        GoalType.POSE,
        GoalType.POSE,
    ]
    assert [request.task.action_type for request in result.requests] == [
        "EE_EXCHANGE_ENTRY",
        "EE_EXCHANGE",
        "acquire",
    ]
    entry = result.requests[0]
    assert entry.task.ee == "2F"
    assert entry.task.metadata["entry_ee"] == "2F"
    assert entry.task.metadata["next_ee"] == "3F"
    assert result.requests[1].world.robot_state.joint_positions_rad == [0.1]
    assert result.requests[1].world.metadata["physical_active_ee"] == "2F"
    assert result.requests[2].world.robot_state.joint_positions_rad == [0.2]
    assert result.requests[2].world.metadata["physical_active_ee"] == "3F"
    assert result.requests[2].world.scene.signature.startswith("predicted:")
    assert result.final_world.robot_state.joint_positions_rad == pytest.approx([0.3])
    assert result.final_world.robot_state.attached_object_id == "part"
    assert result.final_world.scene.completed_subgoals == ["pick-part"]
    assert len(result.request_paths) == 3
    assert all(path.is_file() for path in result.request_paths)
    assert len(result.plan_paths) == 3
    assert all(path.is_file() for path in result.plan_paths)
    assert result.manifest_path is not None
    assert result.manifest_path.is_file()
    restored = MotionPlanStore(tmp_path).load_manifest()
    assert restored.requests == result.requests
    assert restored.plans == result.plans
    assert restored.final_world == result.final_world


def test_orchestrator_carries_generic_contact_friction_state_through_place() -> None:
    subgoals = ["acquire-part", "transport-part", "place-part"]
    actions = ["acquire", "transport", "place"]
    selected = SelectedPlan(
        cost_vector=CostVectorModel(),
        subgoal_order=subgoals,
        candidate_assignments=[
            CandidateAssignment(
                subgoal_id=subgoal_id,
                candidate_id=f"candidate-{index}",
                ee="2F",
                action_type=action,
                target_ids=["part"],
            )
            for index, (subgoal_id, action) in enumerate(zip(subgoals, actions))
        ],
        steps=[
            PlanStep(
                step_index=index,
                kind="subgoal",
                action="EXECUTE_SUBGOAL",
                subgoal_id=subgoal_id,
                candidate_id=f"candidate-{index}",
            )
            for index, subgoal_id in enumerate(subgoals)
        ],
    )
    transform = {
        "object_id": "part",
        "free_joint_name": "part_free",
        "reference_kind": "body",
        "reference_name": "right_hand",
        "position_in_reference_m": [0.0, 0.0, 0.1],
        "orientation_in_reference_xyzw": [0.0, 0.0, 0.0, 1.0],
    }

    def planner(request):
        start = request.world.robot_state.joint_positions_rad[0]
        end = start + 0.1
        final_state = request.world.robot_state.model_copy(deep=True)
        final_state.joint_positions_rad = [end]
        final_state.joint_velocities_rad_s = [0.0]
        metadata = {}
        if request.task.action_type == "acquire":
            final_state.held_tool_id = "part"
            metadata = {
                "grasp_execution_mode": "CONTACT_FRICTION",
                "planned_contact_friction_transform": transform,
            }
        elif request.task.action_type == "place":
            assert releases_contact_friction(request)
            final_state.held_tool_id = None
        return MotionPlan(
            plan_id=f"plan:{request.request_id}",
            request_id=request.request_id,
            provenance=ArtifactProvenance(
                artifact_id=f"plan-artifact:{request.request_id}",
                artifact_type="MotionPlan",
                produced_by=ModuleName.MOTION_PLANNER,
                invocation_id=f"fake:{request.request_id}",
                input_artifact_ids=[request.provenance.artifact_id],
            ),
            scene_signature=request.world.scene.signature,
            robot_id="robot",
            joint_names=["j1"],
            duration_s=1.0,
            segments=[
                TrajectorySegment(
                    segment_id=f"segment:{request.request_id}",
                    segment_type=SegmentType.CUSTOM,
                    start_time_s=0.0,
                    end_time_s=1.0,
                    interpolation=InterpolationType.LINEAR,
                    waypoints=[
                        TrajectoryWaypoint(
                            time_from_start_s=0.0,
                            joint_positions_rad=[start],
                        ),
                        TrajectoryWaypoint(
                            time_from_start_s=1.0,
                            joint_positions_rad=[end],
                        ),
                    ],
                    collision_checked=True,
                )
            ],
            expected_final_state=final_state,
            metadata=metadata,
        )

    result = SelectedPlanMotionOrchestrator(
        planner,
        adapter=SelectedPlanMotionRequestAdapter(
            acquire_task_metadata={
                "grasp_execution_mode": "CONTACT_FRICTION",
            }
        ),
    ).plan(
        selected,
        initial_world=_world(),
        constraints=MotionConstraints(
            joint_limits={
                "j1": JointDynamicLimit(
                    max_velocity_rad_s=1.0,
                    max_acceleration_rad_s2=2.0,
                )
            }
        ),
    )

    acquire, transport, place = result.requests
    assert acquire.task.metadata["operation"] == "ACQUIRE"
    assert transport.world.robot_state.held_tool_id == "part"
    assert transport.world.metadata["contact_friction_held_objects"] == {
        "part": transform
    }
    assert place.world.robot_state.held_tool_id == "part"
    assert releases_contact_friction(place)
    assert result.final_world.robot_state.held_tool_id is None
    assert result.final_world.metadata["contact_friction_held_objects"] == {}


def test_orchestrator_plans_initial_attach_from_empty_mount(tmp_path) -> None:
    selected = _selected().model_copy(deep=True)
    selected.steps = [
        PlanStep(
            step_index=0,
            kind="transition",
            action="ATTACH_EE",
            parameters={"ee": "3F"},
            subgoal_id="pick-part",
            candidate_id="candidate-1",
        ),
        PlanStep(
            step_index=1,
            kind="subgoal",
            action="EXECUTE_SUBGOAL",
            subgoal_id="pick-part",
            candidate_id="candidate-1",
        ),
    ]
    initial_world = _world().model_copy(deep=True)
    initial_world.metadata["physical_active_ee"] = None
    initial_world.metadata["declared_active_ee"] = None
    constraints = MotionConstraints(
        joint_limits={
            "j1": JointDynamicLimit(
                max_velocity_rad_s=1.0,
                max_acceleration_rad_s2=2.0,
            )
        }
    )

    result = SelectedPlanMotionOrchestrator(
        _fake_planner,
        store=MotionPlanStore(tmp_path),
    ).plan(selected, initial_world=initial_world, constraints=constraints)

    assert [request.task.action_type for request in result.requests] == [
        "EE_ATTACH",
        "acquire",
    ]
    attach = result.requests[0]
    assert attach.task.metadata["from_ee"] is None
    assert attach.task.metadata["to_ee"] == "3F"
    assert attach.task.target_ids == ["3F"]
    assert result.requests[1].world.metadata["physical_active_ee"] == "3F"
