from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import tuj.m5_motion.generic_runner as generic_runner
from tuj.m4_taskplanner.serialization import (
    CandidateAssignment,
    CostVectorModel,
    PlanStep,
    SelectedPlan,
)

from tuj.m5_motion.generic_runner import (
    default_constraints,
    load_selected_plan,
    load_world,
    main,
    validate_selected_plan,
)
from tuj.m5_motion.schema import RobotState, SceneRef, WorldSnapshot
from tuj.m5_motion.schema import (
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


def _m4_tool_contract_selected() -> SelectedPlan:
    assignments = [
        CandidateAssignment(
            subgoal_id="pick-tool",
            candidate_id="pick-light-plate",
            ee="2F",
            tool="light_plate",
            action_type="PICK_TOOL",
            target_ids=["light_plate"],
        ),
        CandidateAssignment(
            subgoal_id="sweep-blocks",
            candidate_id="sweep-with-light-plate",
            ee="2F",
            tool="light_plate",
            action_type="tool_act",
            mode="sweep",
            target_ids=["block_0", "block_1"],
            goal_region_id="collection_zone_visual",
        ),
        CandidateAssignment(
            subgoal_id="return-tool",
            candidate_id="return-light-plate",
            ee="2F",
            tool="light_plate",
            action_type="RETURN_TOOL",
            target_ids=["light_plate"],
            goal_region_id="tool_rest",
        ),
    ]
    return SelectedPlan(
        cost_vector=CostVectorModel(),
        subgoal_order=[item.subgoal_id for item in assignments],
        candidate_assignments=assignments,
        steps=[
            PlanStep(
                step_index=index,
                kind="subgoal",
                action="EXECUTE_SUBGOAL",
                subgoal_id=item.subgoal_id,
                candidate_id=item.candidate_id,
            )
            for index, item in enumerate(assignments)
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


class _FakePlannerPool:
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


def test_generic_validation_accepts_current_m4_tool_resource_contract() -> None:
    world = _world()
    world.objects.update(
        {
            "light_plate": {
                "pose": {
                    "frame_id": "world",
                    "position_m": [0.0, -0.25, 0.08],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "dimensions_m": [0.24, 0.24, 0.02],
            },
            "block_0": {
                "pose": {
                    "frame_id": "world",
                    "position_m": [0.20, 0.0, 0.02],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "dimensions_m": [0.02, 0.02, 0.02],
            },
            "block_1": {
                "pose": {
                    "frame_id": "world",
                    "position_m": [0.24, 0.0, 0.02],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "dimensions_m": [0.02, 0.02, 0.02],
            },
            "collection_zone_visual": {
                "pose": {
                    "frame_id": "world",
                    "position_m": [0.0, 0.0, 0.005],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "dimensions_m": [0.20, 0.20, 0.01],
            },
        }
    )

    report = validate_selected_plan(
        _m4_tool_contract_selected(),
        world,
        default_constraints(world),
        options={},
    )

    assert report["status"] == "VALID"
    assert [item["action_type"] for item in report["resources"]] == [
        "PICK_TOOL",
        "tool_act",
        "RETURN_TOOL",
    ]
    assert report["resources"][0]["target_ids"] == ["light_plate"]
    assert report["resources"][2]["target_ids"] == ["light_plate"]


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

    monkeypatch.setattr(
        generic_runner,
        "ToolUseJournalPlannerPool",
        _FakePlannerPool,
    )
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


def test_generic_cli_video_runs_controller_simulation_and_writes_summary(
    tmp_path, monkeypatch
) -> None:
    task_path = tmp_path / "video_task.json"
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
    video_path = tmp_path / "run.mp4"
    observed: dict[str, object] = {}

    def fake_execute(planning, **kwargs):
        observed["planning"] = planning
        observed.update(kwargs)
        return SimpleNamespace(
            successful=True,
            status=SimpleNamespace(value="SUCCESS"),
            runs=(object(), object()),
            reports=(object(), object()),
            manifest_path=output_dir / "simulation" / "simulation-manifest.json",
            detail="ok",
            failed_index=None,
        )

    monkeypatch.setattr(
        generic_runner,
        "ToolUseJournalPlannerPool",
        _FakePlannerPool,
    )
    monkeypatch.setattr(
        generic_runner,
        "execute_planning_result",
        fake_execute,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")

    exit_code = main(
        [
            "--task-planner",
            str(task_path),
            "--initial-world",
            str(world_path),
            "--output-dir",
            str(output_dir),
            "--video",
            str(video_path),
            "--no-scripted-grasps",
        ]
    )

    assert exit_code == 0
    assert observed["mode"] == "controller"
    assert observed["show_viewer"] is False
    assert observed["video"] == video_path.resolve()
    summary = json.loads(
        (output_dir / "m5_summary.json").read_text(encoding="utf-8")
    )
    assert summary["planning_status"] == "SUCCESS"
    assert summary["simulation_status"] == "SUCCESS"
    assert summary["simulation_mode"] == "controller"
    assert summary["video"] == str(video_path.resolve())


@pytest.mark.parametrize("mode_args", [
    ["--simulate", "controller"],
    ["--video", "test.mp4"],
    ["--simulate", "controller", "--scripted-grasps"],
])
def test_controller_defaults_to_scripted_before_constructing_planner(
    tmp_path, monkeypatch, mode_args
):
    from tuj.m5_motion.scripted_grasps import cli

    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps({"status": "SUCCESS",
        "selected_plan": _selected().model_dump(mode="json")}), encoding="utf-8")
    world_path = tmp_path / "world.json"
    world_path.write_text(_world().model_dump_json(), encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    observed = []

    def forbidden(*args, **kwargs):
        pytest.fail("batch LLM planner must not run before scripted acquisition")

    def execute(args, *rest):
        observed.append(args.scripted_grasps)
        return 0

    monkeypatch.setattr(generic_runner, "ToolUseJournalPlannerPool", forbidden)
    monkeypatch.setattr(cli, "execute_selected_plan_live", execute)
    assert main(["--task-planner", str(task_path), "--initial-world", str(world_path),
        "--output-dir", str(tmp_path / "out"), *mode_args]) == 0
    assert observed == [True]


def test_generic_video_recorder_follows_runtime_environment_swap(
    tmp_path, monkeypatch
) -> None:
    import cv2

    frames: list[np.ndarray] = []

    class _Writer:
        released = False

        def isOpened(self):
            return True

        def write(self, frame):
            frames.append(frame.copy())

        def release(self):
            self.released = True

    writer = _Writer()
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args, **kwargs: writer)
    monkeypatch.setattr(cv2, "VideoWriter_fourcc", lambda *args: 0)

    def environment(value: int, simulation_time: float):
        return SimpleNamespace(
            sim=SimpleNamespace(
                data=SimpleNamespace(time=simulation_time),
                render=lambda **kwargs: np.full((8, 8, 3), value, dtype=np.uint8),
            )
        )

    class _Runtime:
        def __init__(self):
            self.env = environment(1, 0.0)
            self.callback = None

        def set_render_callback(self, callback):
            self.callback = callback

    runtime = _Runtime()
    recorder = generic_runner.GenericSimulationVideoRecorder(
        runtime,
        tmp_path / "run.mp4",
        camera="agentview",
        width=8,
        height=8,
        fps=20.0,
    )
    assert runtime.callback is not None
    runtime.env.sim.data.time = 0.05
    runtime.callback(runtime.env)
    runtime.env = environment(7, 0.10)
    runtime.callback(runtime.env)
    recorder.close()

    assert len(frames) == 3
    assert int(frames[-1][0, 0, 0]) == 7
    assert runtime.callback is None
    assert writer.released is True


def test_runtime_render_callback_uses_current_environment() -> None:
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime

    first = SimpleNamespace(render=lambda: None)
    second = SimpleNamespace(render=lambda: None)
    runtime = object.__new__(ToolUseJournalEERuntime)
    runtime._closed = False
    runtime._env = first
    runtime._render_callback = None
    observed: list[object] = []

    runtime.set_render_callback(observed.append)
    runtime.render()
    runtime._env = second
    runtime.render()

    assert observed == [first, second]
