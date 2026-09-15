"""Separate greedy EE-order experiment runner (production C3-2 untouched)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from tuj.ablations.greedy_ee_order import METHOD, greedy_ee_order_plan
from tuj.ablations.timing_summary import (
    build_timing_summary,
    merge_stage_seconds,
    merge_stage_status,
    print_timing_summary,
)
from tuj.m4_taskplanner.gk_adapter import build_request_from_gk
from tuj.m4_taskplanner.models import InitialState
from tuj.m4_taskplanner.serialization import dump_result
from tuj.m5_motion.scripted_grasps.registry import GREEDY_EXTRA_ENTRIES

STAGE_ORDER = ("m1", "m2", "greedy_assignment", "m5")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, pipeline):
    if getattr(args, "grounding_mode", "full") != "full":
        raise ValueError(
            "greedy-ee-order requires grounding_mode=full"
        )
    if getattr(args, "planner_mode", "full") not in {"full", "greedy-ee-order"}:
        raise ValueError(
            "greedy-ee-order cannot be combined with without-planner"
        )
    if args.skip_m4:
        raise ValueError("--skip-m4 would skip the greedy assignment as well")
    if any(token.split("=", 1)[0] == "--task-planner" for token in args.m5_args):
        raise ValueError("M5 must consume this run's greedy-assignment m4.json")
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
                "output directory belongs to a different experiment/task/seed"
            )
        if not args.start_from:
            raise ValueError(
                "existing run requires --start-from or a fresh output directory"
            )
    else:
        if not args.start_from and any(out.iterdir()):
            raise ValueError(
                "new greedy-ee-order run requires an empty output directory"
            )
        if (out / "m4.json").exists():
            raise ValueError(
                "cannot overwrite an existing plan without a matching experiment manifest"
            )
        manifest = {
            "method": METHOD,
            "task": task,
            "seed": args.seed,
            "m4_invoked": False,
            "scripted_grasp_greedy_extra": True,
            "stages": {},
            "m4_sha256": None,
        }

    result_path = out / "greedy_ee_order_result.json"
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
    print("[greedy-ee-order] M4 global search: SKIPPED; myopic EE-switch policy")
    stage = "m1"
    summary_path = out / "m5" / "m5_summary.json"
    previous_summary_mtime_ns = None
    run_started = time.monotonic()
    previous_timing = (
        _read(out / "timing_summary.json")
        if (out / "timing_summary.json").is_file()
        else None
    )
    stage_seconds: dict[str, float | None] = {name: None for name in STAGE_ORDER}
    stage_status: dict[str, str] = {name: "not_run" for name in STAGE_ORDER}
    llm_usage: dict | None = None
    assignment_time_ms: float | None = None

    def persist_timing(*, incomplete: bool = False) -> dict:
        usage = llm_usage
        if usage is None and (out / "m2.json").is_file():
            usage = _read(out / "m2.json").get("m2_stats", {}).get("llm_usage")
        phys = None
        retrieval = out / "m0_retrieval.json"
        if retrieval.is_file():
            phys = (_read(retrieval).get("token_summary") or {}).get("physical_total")
        assign_ms = assignment_time_ms
        if assign_ms is None and (out / "m4.json").is_file():
            assign_ms = _read(out / "m4.json").get("task", {}).get("assignment_time_ms")
        merged_seconds = merge_stage_seconds(previous_timing, stage_seconds)
        merged_status = merge_stage_status(previous_timing, stage_status)
        summary = build_timing_summary(
            stage_order=STAGE_ORDER,
            stage_seconds=merged_seconds,
            stage_status=merged_status,
            llm_usage=usage,
            wall_seconds=None,
            combined_stage_keys=("m2", "greedy_assignment"),
            combined_label="m2_through_assignment_seconds",
            extras={
                "assignment_time_ms": assign_ms,
                "m1_physical_token_summary": phys,
                "session_wall_seconds": round(time.monotonic() - run_started, 2),
            },
        )
        measured = summary.get("measured_stage_sum_seconds")
        summary["total_seconds"] = measured
        if incomplete:
            summary["incomplete"] = True
        _write(out / "timing_summary.json", summary)
        manifest["timing_summary"] = summary
        report["timing_summary"] = summary
        _write(manifest_path, manifest)
        _write(result_path, report)
        print_timing_summary(
            summary,
            stage_order=STAGE_ORDER,
            combined_label="m2_through_assignment_seconds",
        )
        print(f"[timing] -> {out / 'timing_summary.json'}")
        return summary

    try:
        manifest.pop("failure", None)
        if start <= 0:
            manifest["m4_sha256"] = None
            report["plan_status"] = report["m5_status"] = report["success"] = report[
                "metrics"
            ] = None
            report.pop("m5_failure_detail", None)
            t0 = time.monotonic()
            pipeline.stage_m1(task, out, args)
            stage_seconds["m1"] = time.monotonic() - t0
            stage_status["m1"] = "completed"
            manifest["stages"][stage] = "completed"
            _write(manifest_path, manifest)
            _write(result_path, report)
        else:
            stage_status["m1"] = "resumed"
        if stop == 0:
            persist_timing()
            return
        stage = "m2"
        if start <= 1:
            manifest["m4_sha256"] = None
            report["plan_status"] = report["m5_status"] = report["success"] = report[
                "metrics"
            ] = None
            report.pop("m5_failure_detail", None)
            t0 = time.monotonic()
            pipeline.stage_m2(task, out, args)
            stage_seconds["m2"] = time.monotonic() - t0
            llm_usage = _read(out / "m2.json").get("m2_stats", {}).get("llm_usage")
            stage_status["m2"] = "completed"
            manifest["stages"][stage] = "completed"
            _write(manifest_path, manifest)
            _write(result_path, report)
        else:
            if (out / "m2.json").is_file():
                llm_usage = _read(out / "m2.json").get("m2_stats", {}).get("llm_usage")
            stage_status["m2"] = "resumed"
        if stop == 1:
            persist_timing()
            return

        stage = "greedy_assignment"
        if start <= 2:
            manifest["stages"]["m4"] = "skipped"
            manifest["m4_sha256"] = None
            report["m5_status"] = report["success"] = None
            report.pop("m5_failure_detail", None)
            t0 = time.monotonic()
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

                request, changes = constrain_task_request(
                    request,
                    environment,
                    extra_entries=GREEDY_EXTRA_ENTRIES,
                )
            else:
                changes = []
            result = greedy_ee_order_plan(request)
            result.task["execution_compatibility"] = {
                "source": (
                    "scripted_grasp_registry+greedy_extra" if changes else "none"
                ),
                "environment": environment,
                "greedy_extra_ees": sorted(
                    {
                        (e.object_id, e.ee)
                        for e in GREEDY_EXTRA_ENTRIES
                        if e.environment == environment
                    }
                ),
                "changes": changes,
            }
            result.task["scripted_grasp_greedy_extra"] = True
            _write(
                out / "m4_request.json", request.model_dump(mode="json", by_alias=True)
            )
            dump_result(result, out / "m4.json")
            stage_seconds["greedy_assignment"] = time.monotonic() - t0
            assignment_time_ms = result.task.get("assignment_time_ms")
            stage_status["greedy_assignment"] = result.status.value
            manifest["m4_sha256"] = _hash(out / "m4.json")
            manifest["stages"][stage] = result.status.value
            report["plan_status"] = result.status.value
            report["metrics"] = result.model_extra["metrics"]
            _write(manifest_path, manifest)
            _write(result_path, report)
            print(
                f"[greedy-ee-order] assignment: {result.status.value}; "
                f"M4 search calls=0"
            )
            if result.selected_plan is None:
                report["success"] = False
                _write(result_path, report)
                raise RuntimeError(
                    f"greedy assignment failed: {result.status.value}; see m4.json"
                )
        else:
            if (
                not manifest["m4_sha256"]
                or _hash(out / "m4.json") != manifest["m4_sha256"]
            ):
                raise ValueError(
                    "M5 resume requires this run's unchanged greedy plan"
                )
            plan = _read(out / "m4.json")
            if plan.get("method") != METHOD or plan.get("m4_invoked") is not False:
                raise ValueError("M5 plan was not produced by greedy-ee-order")
            if not plan.get("task", {}).get("scripted_grasp_greedy_extra"):
                raise ValueError(
                    "greedy plan missing scripted_grasp_greedy_extra for M5 resolve"
                )
            assignment_time_ms = plan.get("task", {}).get("assignment_time_ms")
            stage_status["greedy_assignment"] = "resumed"
        if stop == 2 or args.skip_m5:
            persist_timing()
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
        t0 = time.monotonic()
        pipeline.stage_m5(task, out, args)
        stage_seconds["m5"] = time.monotonic() - t0
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
        stage_status["m5"] = str(report["m5_status"])
        manifest["stages"][stage] = report["m5_status"]
        _write(result_path, report)
        _write(manifest_path, manifest)
        persist_timing()
    except BaseException as error:
        manifest["stages"][stage] = "failed"
        if stage in stage_status:
            stage_status[stage] = "failed"
        elif stage == "greedy_assignment":
            stage_status["greedy_assignment"] = "failed"
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
        try:
            persist_timing(incomplete=True)
        except Exception:  # noqa: BLE001 - never mask the original failure
            pass
        raise
