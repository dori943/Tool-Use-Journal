"""Incremental M4 execution: catalog acquisition, then ordinary M5 planning."""
from __future__ import annotations

import json
from pathlib import Path

from tuj.m5_motion.object_function_grasp import (
    catalog_library, execute_object_function_grasp, make_function_runtime, snapshot,
    prepare_catalog_environment,
)
from tuj.m5_motion.orchestration import (
    MotionPlanStore, SelectedPlanMotionOrchestrator, SelectedPlanPlanningResult,
)
from tuj.m5_motion.selected_plan_adapter import SelectedPlanMotionRequestAdapter
from tuj.m5_motion.task_semantics import is_acquire_task, is_acquire_action, is_release_task, task_operation


def _write_planning_failure(output_dir, request, error):
    """Persist the compiler evidence that the exception string intentionally abbreviates."""

    compilation = getattr(error, "compilation", None)
    attempts = []
    for attempt in getattr(compilation, "attempts", ()) or ():
        selection = getattr(attempt, "selection", None)
        attempts.append({
            "strategy_id": attempt.strategy_id,
            "failure_code": attempt.failure_code,
            "detail": attempt.detail,
            "resolved_keyframes": [
                {
                    "keyframe_id": item.keyframe_id,
                    "position_m": list(item.pose.position_m),
                    "orientation_xyzw": list(item.pose.orientation_xyzw),
                    "ik_solution_count": len(item.ik_solutions.solutions),
                }
                for item in attempt.resolved_keyframes
            ],
            "ik_diagnostics": [
                {
                    "keyframe_id": item.keyframe_id,
                    "raw_ik_count": item.raw_ik_count,
                    "valid_ik_count": item.valid_ik_count,
                    "attempted_seeds": item.attempted_seeds,
                    "solver_detail": item.solver_detail,
                    "validity_detail": item.validity_detail,
                }
                for item in attempt.ik_diagnostics
            ],
            "rejected_edges": [
                {
                    "from": item.source_keyframe_id,
                    "to": item.target_keyframe_id,
                    "from_branch": item.source_branch_id,
                    "to_branch": item.target_branch_id,
                    "failure_code": item.failure_code,
                    "detail": item.detail,
                }
                for item in (getattr(selection, "rejected_edges", ()) or ())
            ],
        })
    directory = Path(output_dir) / "planning_failures"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{request.task.subgoal_id}.json"
    path.write_text(
        json.dumps(
            {
                "subgoal_id": request.task.subgoal_id,
                "action_type": request.task.action_type,
                "detail": str(error),
                "attempts": attempts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


class ReleaseSettlingProvider:
    """Keep the arm at PLACE while release commands take physical effect."""
    def __init__(self, provider):
        self.provider = provider

    def generate(self, request):
        artifact = self.provider.generate(request)
        if not is_release_task(request.task):
            return artifact
        from tuj.m5_motion.schema import KeyframeEventType
        artifact = artifact.model_copy(deep=True)
        for candidate in artifact.candidates:
            for keyframe in candidate.keyframes:
                if KeyframeEventType.DETACH_OBJECT not in keyframe.events_after:
                    continue
                metadata = dict(keyframe.metadata)
                metadata.setdefault("hold_duration_after_s", 1.6)
                metadata.setdefault("allow_release_contact", True)
                # Provider-specific values win. These defaults also keep VLM
                # release events beyond the player's tracking-settle gate.
                offsets = dict(metadata.get("event_time_offsets_s", {}))
                for event in keyframe.events_after:
                    offsets.setdefault(event.value, .1)
                metadata["event_time_offsets_s"] = offsets
                tracking = dict(metadata.get("tracking_settle", {}))
                tracking.setdefault("joint_tolerance_rad", .01)
                tracking.setdefault("max_wait_s", 3.)
                tracking.setdefault("required_consecutive_ticks", 5)
                metadata["tracking_settle"] = tracking
                keyframe.metadata = metadata
        return artifact


def validate_function_assignments(selected, repository):
    catalog = catalog_library(repository)

    for assignment in selected.candidate_assignments:
        if not is_acquire_action(assignment.action_type):
            continue
        if len(assignment.target_ids) != 1:
            raise ValueError("object-function acquisition requires exactly one target")
        recipe = catalog.get_recipe(assignment.target_ids[0])
        if assignment.ee != recipe.ee_id:
            raise ValueError(
                f"{recipe.object_id}: M4 selects {assignment.ee}, "
                f"existing function requires {recipe.ee_id}"
            )


def run_object_function_sequence(*, selected, repository, world, constraints,
                                 options, output_dir, args):
    from tuj.m5_motion.generic_runner import GenericSimulationVideoRecorder, _safe_slug
    from tuj.m5_motion.execution import SimulationArtifactStore
    from tuj.m5_motion.tool_use_journal import settle_tool_use_journal_free_objects
    from tuj.m5_motion.tool_use_journal_execution import ToolUseJournalExecutionAdapter
    from tuj.m5_motion.tool_use_journal_planning import ToolUseJournalMotionRequestPlanner

    output_dir = Path(output_dir)
    records = []
    show_viewer = not args.headless and not args.video
    render = show_viewer or bool(args.video)
    realtime_factor = args.realtime_factor if args.realtime_factor is not None else (1. if show_viewer else 0.)
    runtime = recorder = None
    summary = {"grasp_provider": "object-function", "status": "FAILED", "steps": records}
    try:
        validate_function_assignments(selected, repository)
        runtime = make_function_runtime(
            repository, world.metadata["environment_name"],
            active_ee=world.metadata.get("physical_active_ee"), seed=args.seed,
            ignore_done=True, use_camera_obs=False, has_renderer=show_viewer,
            # Existing functions perform their own M1 capture even without video.
            has_offscreen_renderer=True,
        )
        if (
            "geometry_observation_artifact" in world.metadata
            or "geometry_evidence" in world.metadata
        ):
            from tuj.m5_motion.tool_use_journal import apply_world_snapshot_state

            apply_world_snapshot_state(runtime.env, world)
        from tuj.m5_motion.generic_runner import _validate_runtime_start
        _validate_runtime_start(runtime, world)
        settle_tool_use_journal_free_objects(runtime.env, duration_s=args.settle_seconds)
        initial = snapshot(runtime, previous=world)
        (output_dir / "function_initial_world.json").write_text(initial.model_dump_json(indent=2), encoding="utf-8")
        if args.video:
            recorder = GenericSimulationVideoRecorder(runtime, args.video,
                camera=args.camera, width=args.width, height=args.height, fps=args.video_fps)

        def function_handler(request):
            if not is_acquire_task(request.task):
                return None
            step_dir = output_dir / "functions" / f"{len(records):03d}-{_safe_slug(request.task.subgoal_id)}"
            step_dir.parent.mkdir(parents=True, exist_ok=True)
            target = request.task.goal.target_object_id
            if not target:
                raise ValueError("acquire request has no target_object_id")
            import time
            wall_start, sim_start = time.monotonic(), float(runtime.env.sim.data.time)
            def control_step():
                if render: runtime.render()
                if realtime_factor > 0:
                    delay = (float(runtime.env.sim.data.time) - sim_start) / realtime_factor - (time.monotonic() - wall_start)
                    if delay > 0: time.sleep(min(delay, .05))
            execute_object_function_grasp(runtime, object_id=target,
                resource_kind="tool" if task_operation(request.task) == "PICK_TOOL" or request.task.tool == target else "object",
                output_dir=step_dir, repository=repository, on_control_step=control_step, seed=args.seed)
            (step_dir / "request.json").write_text(request.model_dump_json(indent=2), encoding="utf-8")
            records.append({"subgoal_id": request.task.subgoal_id, "kind": "object_function",
                            "status": "SUCCESS", "artifact": str(step_dir)})
            return snapshot(runtime, previous=request.world)

        def planner(request):
            # Compile from the corrected live model and current attachment state.
            from tuj.m5_motion.vlm_provider import OpenAIKeyframeProvider
            bound = ToolUseJournalMotionRequestPlanner.from_environment(
                runtime.env, repository, seed=args.seed,
                provider=ReleaseSettlingProvider(
                    OpenAIKeyframeProvider()
                ),
                ee_attach_registry_root=args.ee_attach_registry,
                ee_attach_trajectory_paths=tuple(args.ee_attach_trajectory or ()),
                ee_return_trajectory_paths=tuple(args.ee_return_trajectory or ()),
                ee_attach_policy=args.ee_attach_policy,
                ee_attach_start_tolerance_rad=args.ee_attach_start_tolerance_rad,
                environment_preparer=lambda env, ee: prepare_catalog_environment(env, ee, repository),
            )
            try:
                return bound(request)
            except Exception as error:
                if getattr(error, "compilation", None) is not None:
                    _write_planning_failure(output_dir, request, error)
                raise

        def execute_plan(request, plan):
            from tuj.m5_motion.tool_use_journal import ToolUseJournalCollisionModelCompiler
            compiler = ToolUseJournalCollisionModelCompiler.from_repository(
                runtime.env, repository, seed=args.seed,
                environment_preparer=lambda env, ee: prepare_catalog_environment(env, ee, repository))
            adapter = ToolUseJournalExecutionAdapter(
                runtime, compiler=compiler, controller=True,
                realtime_factor=realtime_factor, render=render,
            )
            directory = output_dir / "simulation" / f"{len(records):03d}-{_safe_slug(request.task.subgoal_id)}"
            execution = adapter.execute(
                SelectedPlanPlanningResult((request,), (plan,), request.world),
                store=SimulationArtifactStore(directory),
            )
            (directory / "final_world.json").write_text(
                execution.final_world.model_dump_json(indent=2), encoding="utf-8"
            )
            records.append({"subgoal_id": request.task.subgoal_id, "kind": "motion_plan",
                            "status": execution.status.value, "artifact": str(directory)})
            if not execution.successful:
                raise RuntimeError(execution.detail)
            return execution.final_world

        result = SelectedPlanMotionOrchestrator(
            planner, store=MotionPlanStore(output_dir),
            adapter=SelectedPlanMotionRequestAdapter(acquire_task_metadata={"grasp_execution_mode": "KINEMATIC"}),
            request_handler=function_handler, plan_executor=execute_plan,
        ).plan(selected, initial_world=initial, constraints=constraints, options=options)
        (output_dir / "final_world.json").write_text(result.final_world.model_dump_json(indent=2), encoding="utf-8")
        summary.update(status="SUCCESS", simulation_successful=True,
                       attached_object_id=runtime.attached_object_id)
        if recorder:
            recorder.hold_final_frame(args.video_hold_seconds)
        return 0
    except Exception as error:
        import traceback
        summary.update(detail=str(error), simulation_successful=False)
        (output_dir / "function_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 2
    finally:
        (output_dir / "m5_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        try:
            if recorder: recorder.close()
        finally:
            if runtime: runtime.close()
