from __future__ import annotations

import json

import tuj.m4_motion.generic_runner as generic_runner
from tuj.m3_taskplanner.serialization import (
    CandidateAssignment,
    CostVectorModel,
    PlanStep,
    SelectedPlan,
)

from tuj.m4_motion.generic_runner import (
    default_constraints,
    load_selected_plan,
    load_world,
    main,
    validate_selected_plan,
)
from tuj.m4_motion.schema import RobotState, SceneRef, WorldSnapshot
from tuj.m4_motion.schema import (
    ArtifactProvenance,
    InterpolationType,
    ModuleName,
    MotionPlan,
    SegmentType,
    TrajectorySegment,
    TrajectoryWaypoint,
)


def _selected() -> SelectedPlan:
    return SelectedPlan(
        cost_vector=CostVectorModel(),
        subgoal_order=["move-widget"],
        candidate_assignments=[
            CandidateAssignment(
                subgoal_id="move-widget",
                candidate_id="candidate-widget",
                ee="3F",
                action_type="MOVE",
                target_ids=["widget"],
            )
        ],
        steps=[
            PlanStep(
                step_index=0,
                kind="transition",
                action="ATTACH_EE",
                parameters={"ee": "3F"},
                subgoal_id="move-widget",
                candidate_id="candidate-widget",
            ),
            PlanStep(
                step_index=1,
                kind="subgoal",
                action="EXECUTE_SUBGOAL",
                subgoal_id="move-widget",
                candidate_id="candidate-widget",
            ),
        ],
    )


def _world() -> WorldSnapshot:
    return WorldSnapshot(
        scene=SceneRef(signature="scene-other-task"),
        robot_state=RobotState(
            robot_id="ur5e",
            joint_names=["j1", "j2"],
            joint_positions_rad=[0.0, 0.0],
        ),
        objects={
            "widget": {
                "pose": {
                    "frame_id": "world",
                    "position_m": [0.4, 0.0, 0.2],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        },
        metadata={
            "environment_name": "C2_1_ObjectSorting",
            "physical_active_ee": None,
        },
    )


def test_load_selected_plan_accepts_planning_result_and_bare_plan(tmp_path) -> None:
    selected = _selected()
    envelope_path = tmp_path / "result.json"
    envelope_path.write_text(
        json.dumps(
            {
                "status": "SUCCESS",
                "selected_plan": selected.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    bare_path = tmp_path / "selected.json"
    bare_path.write_text(selected.model_dump_json(), encoding="utf-8")

    loaded_envelope, envelope = load_selected_plan(envelope_path)
    loaded_bare, bare = load_selected_plan(bare_path)

    assert loaded_envelope == selected
    assert envelope["status"] == "SUCCESS"
    assert loaded_bare == selected
    assert "selected_plan" not in bare


def test_generic_validation_accepts_non_c1_task() -> None:
    selected = _selected()
    world = _world()

    report = validate_selected_plan(
        selected,
        world,
        default_constraints(world),
        options={},
    )

    assert report["status"] == "VALID"
    assert report["subgoal_order"] == ["move-widget"]
    assert report["resources"] == [
        {
            "subgoal_id": "move-widget",
            "ee": "3F",
            "tool": None,
            "action_type": "MOVE",
            "target_ids": ["widget"],
        }
    ]
    assert report["environment_name"] == "C2_1_ObjectSorting"


def test_generic_cli_validates_explicit_task_and_world_files(tmp_path) -> None:
    task_path = tmp_path / "another_task.json"
    task_path.write_text(
        json.dumps(
            {
                "status": "SUCCESS",
                "selected_plan": _selected().model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    world_path = tmp_path / "world.json"
    world_path.write_text(_world().model_dump_json(), encoding="utf-8")
    output_dir = tmp_path / "motion"

    exit_code = main(
        [
            "--task-planner",
            str(task_path),
            "--initial-world",
            str(world_path),
            "--output-dir",
            str(output_dir),
            "--validate-input-only",
        ]
    )

    assert exit_code == 0
    report = json.loads(
        (output_dir / "m5_input_validation.json").read_text(encoding="utf-8")
    )
    assert report["subgoal_order"] == ["move-widget"]
    assert (output_dir / "initial_world.json").is_file()


def test_load_world_accepts_direct_and_wrapped_snapshot(tmp_path) -> None:
    world = _world()
    direct = tmp_path / "direct.json"
    wrapped = tmp_path / "wrapped.json"
    direct.write_text(world.model_dump_json(), encoding="utf-8")
    wrapped.write_text(
        json.dumps({"world": world.model_dump(mode="json")}),
        encoding="utf-8",
    )

    assert load_world(direct) == world
    assert load_world(wrapped) == world


def test_generic_cli_plans_with_request_backend_and_writes_manifest(
    tmp_path, monkeypatch
) -> None:
    task_path = tmp_path / "another_task.json"
    task_path.write_text(
        json.dumps(
            {
                "status": "SUCCESS",
                "selected_plan": _selected().model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    world_path = tmp_path / "world.json"
    world_path.write_text(_world().model_dump_json(), encoding="utf-8")
    output_dir = tmp_path / "motion"

    class _Pool:
        def __init__(self, repository, *, seed):
            del repository, seed

        def __call__(self, request):
            start = request.world.robot_state.joint_positions_rad
            end = [start[0] + 0.1, start[1]]
            return MotionPlan(
                plan_id=f"plan:{request.request_id}",
                request_id=request.request_id,
                provenance=ArtifactProvenance(
                    artifact_id=f"plan-artifact:{request.request_id}",
                    artifact_type="MotionPlan",
                    produced_by=ModuleName.MOTION_PLANNER,
                    invocation_id="fake-generic-runner",
                ),
                scene_signature=request.world.scene.signature,
                robot_id=request.world.robot_state.robot_id,
                joint_names=list(request.world.robot_state.joint_names),
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
                                joint_positions_rad=list(start),
                            ),
                            TrajectoryWaypoint(
                                time_from_start_s=1.0,
                                joint_positions_rad=end,
                            ),
                        ],
                        collision_checked=True,
                    )
                ],
                expected_final_state=RobotState(
                    robot_id=request.world.robot_state.robot_id,
                    joint_names=list(request.world.robot_state.joint_names),
                    joint_positions_rad=end,
                    joint_velocities_rad_s=[0.0, 0.0],
                ),
            )

        def close(self):
            return None

    monkeypatch.setattr(generic_runner, "ToolUseJournalPlannerPool", _Pool)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")

    exit_code = main(
        [
            "--task-planner",
            str(task_path),
            "--initial-world",
            str(world_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    summary = json.loads(
        (output_dir / "m5_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "SUCCESS"
    assert summary["motion_plan_count"] == 2
    assert (output_dir / "motion-plan-manifest.json").is_file()
