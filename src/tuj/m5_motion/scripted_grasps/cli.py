"""CLI lifecycle for opt-in live scripted acquisition."""
import json
import time

from .live import ScriptedGraspSession, snapshot


def execution_initial_world(runtime, preview, *, externally_supplied):
    """File snapshots must reproduce; a live capture uses the actual robot.

    The preview environment may settle slightly differently with rendering.
    Never reset joints to that preview or weaken the file-snapshot check.
    """
    from tuj.m5_motion.generic_runner import _validate_runtime_start
    if externally_supplied:
        _validate_runtime_start(runtime, preview)
    return snapshot(runtime, preview)


def execute_selected_plan_live(args, selected, initial_world, constraints, options,
                               repository, output, artifact_id, planner_options):
    from tuj.m5_motion.generic_runner import (
        GenericSimulationVideoRecorder, _runtime_active_ee, _runtime_environment_name,
        validate_selected_plan,
    )
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime

    runtime = recorder = session = None
    summary = {"mode": "SCRIPTED_GRASP_LIVE", "status": "FAILED",
               "task_goal_status": "NOT_EVALUATED"}
    try:
        runtime = ToolUseJournalEERuntime.from_repository_for_controller(repository,
            _runtime_environment_name(initial_world), active_ee=_runtime_active_ee(initial_world),
            seed=args.seed, scripted_grasps=True, ignore_done=True,
            has_renderer=not args.headless and args.video is None,
            has_offscreen_renderer=args.video is not None, use_camera_obs=False,
            render_camera=args.camera)
        preview = initial_world
        initial_world = execution_initial_world(runtime, preview,
            externally_supplied=args.initial_world is not None)
        report = validate_selected_plan(selected, initial_world, constraints, options)
        (output / "preview_world.json").write_text(preview.model_dump_json(indent=2), encoding="utf-8")
        (output / "initial_world.json").write_text(initial_world.model_dump_json(indent=2), encoding="utf-8")
        report.update(world_source="LIVE_EXECUTION_RUNTIME",
                      initial_world=str((output / "initial_world.json").resolve()))
        (output / "m5_input_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        runtime.scripted_render = not args.headless or args.video is not None
        runtime.scripted_realtime_factor = args.realtime_factor if args.realtime_factor is not None else (1. if not args.headless and args.video is None else 0.)
        if args.video is not None:
            camera = args.camera
            if camera not in runtime.env.sim.model.camera_names:
                camera = runtime.env.render_camera
            recorder = GenericSimulationVideoRecorder(runtime, args.video.resolve(), camera=camera,
                width=args.width, height=args.height, fps=args.video_fps)
        session = ScriptedGraspSession(runtime, repository, output / "live", **planner_options)
        session.world = snapshot(runtime, initial_world)
        manifest = session.execute_selected_plan(selected, constraints=constraints,
            options=options, selected_plan_artifact_id=artifact_id)
        summary.update(status="SUCCESS", manifest=str(manifest.resolve()),
            scripted_grasp_count=sum(r["route"] == "SCRIPTED_GRASP" for r in session.records),
            motion_plan_count=sum(r["route"] == "M5_MOTION_PLAN" for r in session.records))
        if recorder:
            recorder.hold_final_frame(args.video_hold_seconds)
        elif runtime.scripted_render:
            end = time.monotonic() + args.hold_seconds
            while time.monotonic() < end:
                runtime.render()
                time.sleep(.02)
    except Exception as error:
        summary["detail"] = f"{type(error).__name__}: {error}"
        if session:
            session.failure = summary["detail"]
            summary["manifest"] = str(session.save_manifest().resolve())
    finally:
        if recorder:
            recorder.close()
        if runtime:
            runtime.close()
        path = output / "m5_summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "SUCCESS" else 2
