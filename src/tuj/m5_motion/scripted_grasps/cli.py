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


def _planning_rejections(error):
    """Per-strategy rejected edges carried on a motion-planning failure."""
    compilation = getattr(error, "compilation", None)
    attempts = getattr(compilation, "attempts", None)
    if not attempts:
        return None
    report = []
    for attempt in attempts:
        selection = getattr(attempt, "selection", None)
        edges = list(getattr(selection, "rejected_edges", ()) or ())
        counts = {}
        for edge in edges:
            counts[edge.failure_code] = counts.get(edge.failure_code, 0) + 1
        report.append({
            "strategy_id": attempt.strategy_id,
            "failure_code": (attempt.failure_code
                             or getattr(selection, "failure_code", None) or "?"),
            "detail": attempt.detail or getattr(selection, "detail", ""),
            "rejected_edge_counts": counts,
            "rejected_edges": [
                {"from": edge.source_keyframe_id, "to": edge.target_keyframe_id,
                 "from_branch": edge.source_branch_id,
                 "to_branch": edge.target_branch_id,
                 "failure_code": edge.failure_code, "detail": edge.detail}
                for edge in edges],
        })
    return report


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
        # The offscreen context is sized from the environment's camera options
        # (256x256 by default). Recording asks sim.render() for the video size,
        # and reading a 960x540 image out of a 256x256 buffer yields noise and
        # a vertically flipped picture -- so declare the recording size the way
        # the generic and live-execution runners already do.
        recording_options = (
            {"camera_names": args.camera, "camera_heights": args.height,
             "camera_widths": args.width}
            if args.video is not None
            else {}
        )
        runtime = ToolUseJournalEERuntime.from_repository_for_controller(repository,
            _runtime_environment_name(initial_world), active_ee=_runtime_active_ee(initial_world),
            seed=args.seed, scripted_grasps=True, ignore_done=True,
            has_renderer=not args.headless and args.video is None,
            has_offscreen_renderer=args.video is not None, use_camera_obs=False,
            render_camera=args.camera, **recording_options)
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
            if _runtime_environment_name(initial_world) == 'C4_2_DiagonalFitPacking' and camera == 'robot0_robotview':
                # The kitchen's stock robot view points below the work area.
                # Use the existing scene-observation framing for the recording.
                import importlib.util
                spec = importlib.util.spec_from_file_location('scripted_video_camera', repository / 'scripts/run_m1.py')
                capture = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(capture)
                cid = runtime.env.sim.model.camera_name2id(camera)
                runtime.env.sim.model.cam_fovy[cid] = 60.
                capture.fit_camera_to_points(runtime.env, cid,
                    capture.object_bound_points(runtime.env, capture.task_spec('c4_2')))
                runtime.env.sim.forward()
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
        # A planning failure carries its per-strategy rejected edges on the
        # exception. This path catches the exception instead of letting it
        # reach the integrated runner's dump, so the reasons were lost and a
        # failure could only be read as "no strategy produced a path".
        rejections = _planning_rejections(error)
        if rejections is not None:
            path = output / "m5_failure.json"
            path.write_text(json.dumps(rejections, ensure_ascii=False, indent=2),
                            encoding="utf-8")
            summary["failure_detail"] = str(path.resolve())
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
