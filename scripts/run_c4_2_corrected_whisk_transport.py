"""Run the corrected C4-2 whisk grasp and attached-object transport.

This is an evidence run for the C4-2 IK_SEARCH_EXHAUSTED diagnosis.  It keeps
the existing object-specific grasp function, attaches the whisk, derives an
EEF pre-place anchor from the measured attachment transform, and executes a
new collision-checked M5 MotionPlan.  Release belongs to the following PLACE
subgoal and is intentionally not performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
sys.path.insert(0, str(REPOSITORY / "src"))

# C4-2 packing policy retained from the previously feasible strategy.  These
# numbers describe this box / long-object layout and are not generic planner
# constants.
WHISK_DIAGONAL_AXIS_BOX = (-0.09493973907627477, 0.22721515344418652, 0.9692057160321863)
WHISK_DIAGONAL_ROLL_RAD = 1.5656615696165832
WHISK_CENTER_CLEARANCE_ABOVE_BOX_M = 0.07075


def _write_json(path: Path, value) -> None:
    if hasattr(value, "model_dump_json"):
        payload = value.model_dump_json(indent=2)
    else:
        payload = json.dumps(value, ensure_ascii=False, indent=2)
    path.write_text(payload + "\n", encoding="utf-8")


class FrozenProvider:
    def __init__(self, artifact):
        self.artifact = artifact

    def generate(self, request):
        if self.artifact.scene_signature != request.world.scene.signature:
            raise ValueError("corrected keyframes belong to another scene")
        if self.artifact.subgoal_id != request.task.subgoal_id:
            raise ValueError("corrected keyframes belong to another subgoal")
        return self.artifact


def _corrected_anchor(world, attachment) -> tuple[list[float], list[float]]:
    """Return box-local EEF anchor and desired world object center.

    Given the measured position p_EO of the object origin in the EEF frame,
    choose the C4-2 pre-place object pose and solve

        p_WE = p_WO - R_WE p_EO.

    This is the translation part of T_WE = T_WO @ inverse(T_EO).
    """
    import numpy as np
    from tuj.m5_motion.geometry import quaternion_matrix_xyzw, tool_rotation_from_axis

    box = world.objects["packing_box"]
    box_pose = box["pose"]
    box_position = np.asarray(box_pose["position_m"], dtype=float)
    box_rotation = quaternion_matrix_xyzw(box_pose["orientation_xyzw"])
    top_local_z = float(box["anchors"]["top_center"][2])
    desired_object_center_box = np.asarray(
        (0.0, 0.0, top_local_z + WHISK_CENTER_CLEARANCE_ABOVE_BOX_M), dtype=float
    )
    desired_object_center_world = box_position + box_rotation @ desired_object_center_box

    axis_world = box_rotation @ np.asarray(WHISK_DIAGONAL_AXIS_BOX, dtype=float)
    # The symbolic keyframe aligns the tool's -z axis with the outward axis.
    eef_rotation_world = tool_rotation_from_axis(-axis_world, WHISK_DIAGONAL_ROLL_RAD)
    object_position_in_eef = np.asarray(attachment.position_in_reference_m, dtype=float)
    eef_position_world = desired_object_center_world - eef_rotation_world @ object_position_in_eef
    eef_position_box = box_rotation.T @ (eef_position_world - box_position)
    return eef_position_box.tolist(), desired_object_center_world.tolist()


def _corrected_artifact(source, request):
    from tuj.m5_motion.schema import KeyframeType

    source_candidate = source.candidates[0]
    keyframes = []
    for index, source_keyframe in enumerate(source_candidate.keyframes[:2], start=1):
        keyframe_type = KeyframeType.TRANSFER if index == 1 else KeyframeType.PRE_PLACE
        keyframes.append(
            source_keyframe.model_copy(
                update={
                    "keyframe_id": f"{request.task.subgoal_id}:attachment_aware_transport:{index}",
                    "keyframe_type": keyframe_type,
                    "frame_ref": "object:packing_box",
                    "anchor": "held_transport_goal",
                    "approach_axis_xyz": WHISK_DIAGONAL_AXIS_BOX,
                    "tool_axis_to_align": "-z",
                    "offset_along_approach_m": 0.0,
                    "roll_rad": WHISK_DIAGONAL_ROLL_RAD,
                    "events_after": [],
                    "metadata": {
                        **source_keyframe.metadata,
                        "correction": "attachment-aware EEF target",
                        "stage_owner": "TRANSPORT",
                    },
                }
            )
        )

    digest = hashlib.sha256(
        f"{request.request_id}|attachment-aware-whisk-transport-v1".encode("utf-8")
    ).hexdigest()[:24]
    candidate = source_candidate.model_copy(
        update={
            "strategy_id": f"{request.task.subgoal_id}:attachment_aware_transport",
            "keyframes": keyframes,
            "rationale": (
                "Transport the attached whisk to a collision-checked pre-place pose; "
                "the following PLACE subgoal owns descent, detach, open, and retreat."
            ),
            "metadata": {
                **source_candidate.metadata,
                "correction": "derive TCP target from the measured attachment transform",
                "task_specific_profile": "C4-2 whisk diagonal pre-place",
            },
        }
    )
    provenance = source.provenance.model_copy(
        update={
            "artifact_id": f"keyframe-plan-artifact:{digest}",
            "invocation_id": f"corrected-keyframes:{request.request_id}",
            "input_artifact_ids": [request.provenance.artifact_id],
            "metadata": {
                **source.provenance.metadata,
                "derived_from_artifact_id": source.artifact_id,
                "correction": "attachment-aware target plus transport-stage boundary",
            },
        }
    )
    return source.model_copy(
        update={
            "artifact_id": f"keyframe-plan:{digest}",
            "provenance": provenance,
            "scene_signature": request.world.scene.signature,
            "subgoal_id": request.task.subgoal_id,
            "candidates": [candidate],
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    from tuj.m5_motion.execution import SimulationArtifactStore
    from tuj.m5_motion.generic_runner import (
        GenericSimulationVideoRecorder,
        default_constraints,
        load_keyframe_artifact,
        load_selected_plan,
    )
    from tuj.m5_motion.object_function_grasp import (
        execute_object_function_grasp,
        make_function_runtime,
        prepare_catalog_environment,
        snapshot,
    )
    from tuj.m5_motion.orchestration import SelectedPlanPlanningResult
    from tuj.m5_motion.schema import PlannerOptions
    from tuj.m5_motion.selected_plan_adapter import SelectedPlanMotionRequestAdapter
    from tuj.m5_motion.tool_use_journal import (
        ToolUseJournalCollisionModelCompiler,
        settle_tool_use_journal_free_objects,
    )
    from tuj.m5_motion.tool_use_journal_execution import ToolUseJournalExecutionAdapter
    from tuj.m5_motion.tool_use_journal_planning import ToolUseJournalMotionRequestPlanner

    selected, _ = load_selected_plan(REPOSITORY / "output/c4_2/m4-object-function-vac.json")
    source_artifact_path = (
        WORKSPACE
        / "reports/c4_2_before_fresh_m1_20260905/m5_scripted_clearance/keyframes"
        / "d6e01149ffc70f26da7b85f354b45b8a1ca3d2eeab8f6c96f23f820c2bb78872.json"
    )
    source_artifact = load_keyframe_artifact(source_artifact_path)

    runtime = recorder = None
    summary = {"status": "FAILED", "stage": "INITIALIZE"}
    try:
        runtime = make_function_runtime(
            REPOSITORY,
            "C4_2_DiagonalFitPacking",
            active_ee="2F",
            seed=0,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=False,
        )
        settle_tool_use_journal_free_objects(runtime.env, duration_s=3.0)
        initial_world = snapshot(runtime)
        _write_json(args.output_dir / "initial_world.json", initial_world)
        recorder = GenericSimulationVideoRecorder(
            runtime,
            args.video,
            camera=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )

        summary["stage"] = "FUNCTION_GRASP"
        grasp = execute_object_function_grasp(
            runtime,
            object_id="whisk",
            output_dir=args.output_dir / "function_grasp",
            repository=REPOSITORY,
            on_control_step=runtime.render,
            seed=0,
        )
        attached_world = snapshot(runtime, previous=initial_world)
        _write_json(args.output_dir / "attached_world_before_correction.json", attached_world)

        anchor_box, desired_object_center_world = _corrected_anchor(
            attached_world, runtime.attachment
        )
        corrected_world = attached_world.model_copy(deep=True)
        corrected_world.objects["packing_box"].setdefault("anchors", {})[
            "held_transport_goal"
        ] = anchor_box
        _write_json(args.output_dir / "corrected_world.json", corrected_world)

        adapter = SelectedPlanMotionRequestAdapter(
            acquire_task_metadata={"grasp_execution_mode": "KINEMATIC"}
        )
        requests = adapter.convert(
            selected,
            worlds=lambda _subgoal, _index: corrected_world,
            constraints=default_constraints(corrected_world),
            options=PlannerOptions(
                allowed_planning_time_s=12.0,
                max_attempts=8,
                random_seed=0,
            ),
        )
        request = next(
            item for item in requests if item.task.subgoal_id == "SG1_s5_d2"
        )
        _write_json(args.output_dir / "corrected_motion_request.json", request)

        artifact = _corrected_artifact(source_artifact, request)
        _write_json(args.output_dir / "corrected_keyframes.json", artifact)

        summary["stage"] = "MOTION_PLANNING"
        planner = ToolUseJournalMotionRequestPlanner.from_environment(
            runtime.env,
            REPOSITORY,
            seed=0,
            provider=FrozenProvider(artifact),
            environment_preparer=lambda env, ee: prepare_catalog_environment(
                env, ee, REPOSITORY
            ),
        )
        planning = planner(request)
        plan = planning.plan
        _write_json(args.output_dir / "corrected_motion_plan.json", plan)
        compilation_summary = {
            "solved": planning.compilation.solved,
            "selected_strategy_id": (
                planning.compilation.connected.strategy_id
                if planning.compilation.connected is not None
                else None
            ),
            "attempts": [
                {
                    "strategy_id": attempt.strategy_id,
                    "failure_code": attempt.failure_code,
                    "detail": attempt.detail,
                    "ik_solution_counts": [
                        len(item.ik_solutions.solutions)
                        for item in attempt.resolved_keyframes
                    ],
                }
                for attempt in planning.compilation.attempts
            ],
        }
        _write_json(args.output_dir / "compilation_summary.json", compilation_summary)

        summary["stage"] = "MOTION_EXECUTION"
        compiler = ToolUseJournalCollisionModelCompiler.from_repository(
            runtime.env,
            REPOSITORY,
            seed=0,
            environment_preparer=lambda env, ee: prepare_catalog_environment(
                env, ee, REPOSITORY
            ),
        )
        execution_adapter = ToolUseJournalExecutionAdapter(
            runtime,
            compiler=compiler,
            controller=True,
            realtime_factor=0.0,
            render=True,
            random_seed=0,
        )
        execution = execution_adapter.execute(
            SelectedPlanPlanningResult((request,), (plan,), request.world),
            store=SimulationArtifactStore(args.output_dir / "simulation"),
        )
        recorder.hold_final_frame(2.0)

        report = execution.reports[-1] if execution.reports else None
        final_world = snapshot(runtime, previous=corrected_world)
        _write_json(args.output_dir / "final_world.json", final_world)
        summary = {
            "status": (
                "SUCCESS"
                if report is not None and report.status.value == "SUCCESS"
                else "FAILED"
            ),
            "stage": "COMPLETE",
            "function_grasp": grasp.function,
            "attachment_mode": runtime.attachment.mode,
            "attached_object_id": runtime.attached_object_id,
            "corrected_anchor_box_m": anchor_box,
            "desired_whisk_center_world_m": desired_object_center_world,
            "plan_id": plan.plan_id,
            "plan_duration_s": plan.duration_s,
            "segment_count": len(plan.segments),
            "waypoint_count": sum(len(segment.waypoints) for segment in plan.segments),
            "execution_status": report.status.value if report else None,
            "sequence_status": execution.status.value,
            "sequence_detail": execution.detail,
            "goal_evaluations": [item.as_dict() for item in execution.goal_evaluations],
            "metrics": report.metrics.model_dump(mode="json") if report else None,
            "video": str(args.video.resolve()),
            "release_performed": False,
            "release_owner": "following PLACE subgoal SG1_s5_d3",
        }
        return 0 if summary["status"] == "SUCCESS" else 2
    except Exception as error:
        import traceback

        summary.update(detail=str(error))
        (args.output_dir / "error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        return 2
    finally:
        _write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        try:
            if recorder is not None:
                recorder.close()
        finally:
            if runtime is not None:
                runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
