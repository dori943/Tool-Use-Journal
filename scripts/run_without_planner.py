"""Separate M2-order, independent EE/tool baseline for scripts/run.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tuj.ablations.without_planner import METHOD, independent_plan
from tuj.m4_taskplanner.gk_adapter import build_request_from_gk
from tuj.m4_taskplanner.models import InitialState
from tuj.m4_taskplanner.serialization import dump_result


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_task_instruction(task, m2_path):
    """Reject M2 artifacts generated for an obsolete task instruction."""
    from task_registry import instruction

    recorded = _read(m2_path).get("task")
    current = instruction(task)
    if recorded != current:
        raise ValueError(
            f"M2 task instruction mismatch for {task}: "
            f"stored={recorded!r}, current={current!r}. "
            "Regenerate M2 from the current task registry before planning."
        )


def run(args, pipeline):
    if args.grounding_mode != "full":
        raise ValueError(
            "without-planner and without_grounding are separate experimental conditions"
        )
    if args.skip_m4:
        raise ValueError("--skip-m4 would skip the independent assignment as well")
    if any(token.split("=", 1)[0] == "--task-planner" for token in args.m5_args):
        raise ValueError("M5 must consume this run's independent-assignment m4.json")
    partial_execution = any(
        token.split("=", 1)[0] in ("--stop-after-subgoal", "--stop-after-pick")
        for token in args.m5_args
    )
    start = pipeline.STAGES.index(args.start_from) if args.start_from else 0
    stop = pipeline.STAGES.index(args.stop_after) if args.stop_after else 3
    if start > stop:
        raise ValueError("--start-from must not follow --stop-after")

    task = args.task
    out = (
        args.output_dir.resolve()
        if args.output_dir
        else pipeline.ROOT / "output" / METHOD / task / f"seed_{args.seed}"
    )
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "ablation_manifest.json"
    if manifest_path.exists():
        manifest = _read(manifest_path)
        if (manifest.get("method"), manifest.get("task"), manifest.get("seed")) != (
            METHOD,
            task,
            args.seed,
        ):
            raise ValueError(
                "output directory belongs to a different ablation/task/seed"
            )
        if not args.start_from:
            raise ValueError(
                "existing run requires --start-from or a fresh output directory"
            )
    else:
        if not args.start_from and any(out.iterdir()):
            raise ValueError(
                "new without-planner run requires an empty output directory"
            )
        if (out / "m4.json").exists():
            raise ValueError(
                "cannot overwrite an existing plan without a matching ablation manifest"
            )
        manifest = {
            "method": METHOD,
            "task": task,
            "seed": args.seed,
            "m4_invoked": False,
            "stages": {},
            "m4_sha256": None,
        }

    result_path = out / "without_planner_result.json"
    report = (
        _read(result_path)
        if result_path.exists()
        else {
            "method": METHOD,
            "task": task,
            "seed": args.seed,
            "m4_invoked": False,
            "plan_status": None,
            "m5_status": None,
            "success": None,
            "evaluation_scope": "partial" if partial_execution else "full",
            "metrics": None,
        }
    )
    report["evaluation_scope"] = "partial" if partial_execution else "full"
    _write(manifest_path, manifest)
    _write(result_path, report)
    print(f"[run] method={METHOD} task={task} seed={args.seed} out={out}")
    print("[without-planner] M4 EE Swap-Aware Planner invocation: SKIPPED")
    stage = "m1"
    summary_path = out / "m5" / "m5_summary.json"
    previous_summary_mtime_ns = None
    try:
        manifest.pop("failure", None)
        if start <= 0:
            manifest["m4_sha256"] = None
            report["plan_status"] = report["m5_status"] = report["success"] = report[
                "metrics"
            ] = None
            report.pop("m5_failure_detail", None)
            pipeline.stage_m1(task, out, args)
            manifest["stages"][stage] = "completed"
            _write(manifest_path, manifest)
            _write(result_path, report)
        if stop == 0:
            return
        stage = "m2"
        if start <= 1:
            manifest["m4_sha256"] = None
            report["plan_status"] = report["m5_status"] = report["success"] = report[
                "metrics"
            ] = None
            report.pop("m5_failure_detail", None)
            pipeline.stage_m2(task, out, args)
            manifest["stages"][stage] = "completed"
            _write(manifest_path, manifest)
            _write(result_path, report)
        _validate_task_instruction(task, out / "m2.json")
        if stop == 1:
            return

        stage = "independent_assignment"
        if start <= 2:
            manifest["stages"]["m4"] = "skipped"
            manifest["m4_sha256"] = None
            report["m5_status"] = report["success"] = None
            report.pop("m5_failure_detail", None)
            gk_paths = pipeline.stage_gk(task, out)
            bundle = pipeline.build_gk_bundle(out, gk_paths)
            initial = (
                InitialState.model_validate(_read(args.initial_state))
                if args.initial_state
                else None
            )
            request = build_request_from_gk(
                _read(bundle),
                _read(out / "m2.json"),
                m1_payload=_read(out / "m1.json"),
                robot_spec_payload=_read(args.robot_spec),
                initial_state=initial,
            )
            environment = args.m5_environment or pipeline.TASK_ENV.get(task)
            if "--no-scripted-grasps" not in args.m5_args and environment:
                from tuj.m5_motion.scripted_grasps.task_constraints import (
                    constrain_task_request,
                )

                request, changes = constrain_task_request(request, environment)
            else:
                changes = []
            result = independent_plan(request)
            result.task["execution_compatibility"] = {
                "source": "scripted_grasp_registry" if changes else "none",
                "environment": environment,
                "changes": changes,
            }
            _write(
                out / "m4_request.json", request.model_dump(mode="json", by_alias=True)
            )
            dump_result(result, out / "m4.json")
            manifest["m4_sha256"] = _hash(out / "m4.json")
            manifest["stages"][stage] = result.status.value
            report["plan_status"] = result.status.value
            report["metrics"] = result.model_extra["metrics"]
            _write(manifest_path, manifest)
            _write(result_path, report)
            print(
                f"[without-planner] independent assignment: {result.status.value}; M4 search calls=0"
            )
            if result.selected_plan is None:
                report["success"] = False
                _write(result_path, report)
                raise RuntimeError(
                    f"independent assignment failed: {result.status.value}; see m4.json rejections"
                )
        else:
            if (
                not manifest["m4_sha256"]
                or _hash(out / "m4.json") != manifest["m4_sha256"]
            ):
                raise ValueError(
                    "M5 resume requires this run's unchanged independent plan"
                )
            plan = _read(out / "m4.json")
            if plan.get("method") != METHOD or plan.get("m4_invoked") is not False:
                raise ValueError("M5 plan was not produced by without-planner")
        if stop == 2 or args.skip_m5:
            return

        stage = "m5"
        report["m5_status"] = report["success"] = None
        report.pop("m5_failure_detail", None)
        manifest["stages"][stage] = "running"
        _write(result_path, report)
        _write(manifest_path, manifest)
        previous_summary_mtime_ns = (
            summary_path.stat().st_mtime_ns if summary_path.is_file() else None
        )
        pipeline.stage_m5(task, out, args)
        if (
            not summary_path.is_file()
            or summary_path.stat().st_mtime_ns == previous_summary_mtime_ns
        ):
            raise RuntimeError("M5 returned without a new m5_summary.json")
        summary = _read(summary_path)
        report["m5_status"] = summary.get("status")
        report["success"] = (
            None
            if args.m5_validate_only or partial_execution
            else summary.get("status") == "SUCCESS"
        )
        manifest["stages"][stage] = report["m5_status"]
        _write(result_path, report)
        _write(manifest_path, manifest)
    except BaseException as error:
        manifest["stages"][stage] = "failed"
        manifest["failure"] = {
            "stage": stage,
            "type": type(error).__name__,
            "causal_attribution": "not_established",
        }
        if stage == "m5":
            fresh_summary = (
                summary_path.is_file()
                and summary_path.stat().st_mtime_ns != previous_summary_mtime_ns
            )
            summary = _read(summary_path) if fresh_summary else {}
            report["m5_status"] = summary.get("status", "FAILED")
            report["m5_failure_detail"] = summary.get("detail", str(error))
            report["success"] = (
                None if args.m5_validate_only or partial_execution else False
            )
        _write(result_path, report)
        _write(manifest_path, manifest)
        raise
