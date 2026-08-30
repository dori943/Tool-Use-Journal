from __future__ import annotations

from tuj.m4_motion.orchestration import (
    MotionPlanStore,
    SelectedPlanMotionOrchestrator,
)
from tuj.m4_motion.schema import (
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
from tuj.m3_taskplanner.models import GraspSpec
from tuj.m3_taskplanner.serialization import (
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
    ]
    assert [request.task.action_type for request in result.requests] == [
        "EE_EXCHANGE",
        "acquire",
    ]
    assert result.requests[1].world.robot_state.joint_positions_rad == [0.1]
    assert result.requests[1].world.scene.signature.startswith("predicted:")
    assert result.final_world.robot_state.joint_positions_rad == [0.2]
    assert result.final_world.robot_state.attached_object_id == "part"
    assert result.final_world.scene.completed_subgoals == ["pick-part"]
    assert len(result.request_paths) == 2
    assert all(path.is_file() for path in result.request_paths)
    assert len(result.plan_paths) == 2
    assert all(path.is_file() for path in result.plan_paths)
    assert result.manifest_path is not None
    assert result.manifest_path.is_file()
    restored = MotionPlanStore(tmp_path).load_manifest()
    assert restored.requests == result.requests
    assert restored.plans == result.plans
    assert restored.final_world == result.final_world


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
