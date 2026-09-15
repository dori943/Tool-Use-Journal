"""Separate orchestration selected by run.py --grounding-mode without_grounding."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

from assemble_gk import assemble

from task_registry import instruction as task_instruction
from tuj.ablations.timing_summary import (
    build_timing_summary as _build_timing_summary,
    merge_stage_seconds,
    merge_stage_status,
    print_timing_summary as _print_timing_summary,
)
from tuj.ablations.without_grounding import (
    MODE,
    POLICY_VERSION,
    VisualSemanticRough,
    build_m2,
    build_m4_request,
    decision_robot,
    decision_scene,
)
from tuj.m4_taskplanner.planner import plan
from tuj.m4_taskplanner.serialization import dump_result

STAGE_ORDER = ("m1", "m2", "m4", "m5")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_timing_summary(
    *,
    stage_seconds: dict[str, float | None],
    stage_status: dict[str, str],
    llm_usage: dict | None,
    wall_seconds: float | None,
) -> dict:
    """Assemble the ablation timing/token report written to timing_summary.json."""

    summary = _build_timing_summary(
        stage_order=STAGE_ORDER,
        stage_seconds=stage_seconds,
        stage_status=stage_status,
        llm_usage=llm_usage,
        wall_seconds=None,
        combined_stage_keys=("m2", "m4"),
        combined_label="m2_through_m4_seconds",
        extras={
            "session_wall_seconds": (
                None if wall_seconds is None else round(float(wall_seconds), 2)
            )
        },
    )
    measured = summary.get("measured_stage_sum_seconds")
    summary["total_seconds"] = measured if measured is not None else (
        None if wall_seconds is None else round(float(wall_seconds), 2)
    )
    return summary


def print_timing_summary(summary: dict) -> None:
    _print_timing_summary(
        summary,
        stage_order=STAGE_ORDER,
        combined_label="m2_through_m4_seconds",
    )


@contextmanager
def isolated_motion_cache(directory):
    key = "MOTION_PLANNER_KEYFRAME_CACHE"
    previous = os.environ.get(key)
    os.environ[key] = str(directory)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def verify_artifact(out, manifest, name):
    expected = manifest.get("artifacts", {}).get(name)
    if expected is None or not (out / name).is_file() or digest(out / name) != expected:
        raise ValueError(f"Missing, changed or foreign ablation artifact: {name}")


def prepare_directory(out, identity, resume):
    manifest_path = out / "ablation_manifest.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest.get("identity") != identity:
            raise ValueError(
                "Ablation configuration differs; use a new output directory"
            )
        if not resume:
            raise ValueError(
                "Existing run: use --start-from or choose a new output directory"
            )
        return manifest
    if resume or (out.exists() and any(out.iterdir())):
        raise ValueError(
            "Resume requires an ablation manifest; new runs require an empty output directory"
        )
    out.mkdir(parents=True, exist_ok=True)
    return {
        "identity": identity,
        "artifacts": {},
        "stages": {},
        "scope": "explicit physical/geometric information removed from M2/M4 only",
        "scripted_registry_prefilter": False,
        "m5_execution_information": "same common runner; live state and collision geometry retained",
    }


def run(args, pipeline):
    start = pipeline.STAGES.index(args.start_from) if args.start_from else 0
    stop = pipeline.STAGES.index(args.stop_after) if args.stop_after else 3
    if start > stop:
        raise ValueError("--start-from must not follow --stop-after")
    if args.initial_state:
        raise ValueError(
            "v1 uses robot-spec resource state; arbitrary initial-state facts are not permitted"
        )
    if args.m5_physical:
        raise ValueError(
            "Use the common generic M5 runner; legacy --m5-physical is not supported by this mode"
        )
    # Own plan/output/seed/provider routing so --m5-args cannot substitute full artifacts.
    forbidden = {
        "--task-planner",
        "--output-dir",
        "--seed",
        "--provider",
        "--model",
        "--repository",
    }
    if any(token.split("=", 1)[0] in forbidden for token in args.m5_args):
        raise ValueError(
            "Set plan/output/seed/provider/model on run.py, not through --m5-args"
        )
    out = (
        args.output_dir.resolve()
        if args.output_dir
        else (pipeline.ROOT / "output" / MODE / args.task / f"seed_{args.seed}")
    )
    robot = decision_robot(read(Path(args.robot_spec)))
    identity = {
        "mode": MODE,
        "policy_version": POLICY_VERSION,
        "task": args.task,
        "seed": args.seed,
        "model": args.model,
        "provider": os.environ["TUJ_LLM_PROVIDER"],
        "robot_sha256": hashlib.sha256(
            json.dumps(robot, sort_keys=True).encode()
        ).hexdigest(),
    }
    manifest = prepare_directory(out, identity, resume=bool(args.start_from))
    if "failure" in manifest:
        manifest.setdefault("failure_history", []).append(manifest.pop("failure"))
    manifest_path = out / "ablation_manifest.json"
    write(manifest_path, manifest)
    stage = "m1"
    run_started = time.monotonic()
    previous_timing = (
        read(out / "timing_summary.json")
        if (out / "timing_summary.json").is_file()
        else None
    )
    stage_seconds: dict[str, float | None] = {
        "m1": None,
        "m2": None,
        "m4": None,
        "m5": None,
    }
    stage_status: dict[str, str] = {
        "m1": "not_run",
        "m2": "not_run",
        "m4": "not_run",
        "m5": "not_run",
    }
    llm_usage: dict | None = None

    def checkpoint(name):
        manifest["artifacts"][name] = digest(out / name)
        write(manifest_path, manifest)

    def persist_timing(*, incomplete: bool = False) -> dict:
        usage = llm_usage
        if usage is None and (out / "m2.json").is_file():
            usage = read(out / "m2.json").get("m2_stats", {}).get("llm_usage")
        summary = build_timing_summary(
            stage_seconds=merge_stage_seconds(previous_timing, stage_seconds),
            stage_status=merge_stage_status(previous_timing, stage_status),
            llm_usage=usage,
            wall_seconds=time.monotonic() - run_started,
        )
        if incomplete:
            summary["incomplete"] = True
        write(out / "timing_summary.json", summary)
        manifest["timing_summary"] = summary
        write(manifest_path, manifest)
        print_timing_summary(summary)
        print(f"[timing] -> {out / 'timing_summary.json'}")
        return summary

    try:
        if start == 0:
            # Rerunning only M1 must also invalidate downstream resume checkpoints.
            manifest["artifacts"].clear()
            manifest["stages"].clear()
            write(manifest_path, manifest)
            execution = out / "execution"
            execution.mkdir(exist_ok=True)
            t0 = time.monotonic()
            pipeline.stage_m1(args.task, execution, args)
            source_frame = args.scene_frame or (
                Path(args.m1_json).resolve().parent / "frame.png"
                if args.m1_json
                else execution / "frame.png"
            )
            if not source_frame.is_file():
                raise ValueError("Matching frame.png is missing; provide --scene-frame")
            shutil.copyfile(source_frame, out / "frame.png")
            write(out / "m1.json", decision_scene(read(execution / "m1.json")))
            write(out / "robot_decision.json", robot)
            stage_seconds["m1"] = time.monotonic() - t0
            for name in (
                "execution/m1.json",
                "frame.png",
                "m1.json",
                "robot_decision.json",
            ):
                checkpoint(name)
            manifest["stages"][stage] = "prepared"
            stage_status["m1"] = "prepared"
            write(manifest_path, manifest)
        else:
            for name in (
                "execution/m1.json",
                "frame.png",
                "m1.json",
                "robot_decision.json",
            ):
                verify_artifact(out, manifest, name)
            if (
                args.m1_json
                and digest(Path(args.m1_json))
                != manifest["artifacts"]["execution/m1.json"]
            ):
                raise ValueError("M1 source differs from the resumed run")
            if (
                args.scene_frame
                and digest(args.scene_frame) != manifest["artifacts"]["frame.png"]
            ):
                raise ValueError("Scene frame differs from the resumed run")
            stage_status["m1"] = "resumed"
        if stop == 0:
            persist_timing()
            return
        stage = "m2"
        if start <= 1:
            # Later checkpoints become invalid whenever an upstream stage reruns.
            for name in ("m2.json", "m4.json", "m4_request.json", "gk_bundle.json"):
                manifest["artifacts"].pop(name, None)
            for name in ("m2", "m4", "m5"):
                manifest["stages"].pop(name, None)
            write(manifest_path, manifest)
            t0 = time.monotonic()
            rough = VisualSemanticRough(args.model, out / "frame.png", robot)
            m2 = build_m2(task_instruction(args.task), read(out / "m1.json"), rough)
            write(out / "m2.json", m2)
            stage_seconds["m2"] = time.monotonic() - t0
            llm_usage = (m2.get("m2_stats") or {}).get("llm_usage")
            checkpoint("m2.json")
            manifest["stages"][stage] = "completed"
            stage_status["m2"] = "completed"
            write(manifest_path, manifest)
        else:
            verify_artifact(out, manifest, "m2.json")
            llm_usage = read(out / "m2.json").get("m2_stats", {}).get("llm_usage")
            stage_status["m2"] = "resumed"
        if stop == 1:
            persist_timing()
            return
        stage = "m4"
        if start <= 2 and not args.skip_m4:
            manifest["artifacts"].pop("m4.json", None)
            write(manifest_path, manifest)
            t0 = time.monotonic()
            request, bundle = build_m4_request(
                read(out / "m1.json"), read(out / "m2.json"), robot, assemble
            )
            write(out / "gk_bundle.json", bundle)
            write(
                out / "m4_request.json", request.model_dump(mode="json", by_alias=True)
            )
            result = plan(request)
            result.task["ablation"] = {"mode": MODE, "policy_version": POLICY_VERSION}
            result.task["execution_compatibility"] = {
                "source": "visual_semantic_unverified",
                "scripted_registry_prefilter": False,
            }
            if result.selected_plan:
                for assignment in result.selected_plan.candidate_assignments:
                    assignment.ee_feasible_set_source = "visual_semantic_unverified"
                    assignment.tool_selection_source = (
                        "visual_semantic" if assignment.tool else "not_required"
                    )
            dump_result(result, out / "m4.json")
            stage_seconds["m4"] = time.monotonic() - t0
            for name in ("m4_request.json", "gk_bundle.json", "m4.json"):
                checkpoint(name)
            manifest["stages"][stage] = result.status.value
            stage_status["m4"] = result.status.value
            write(manifest_path, manifest)
            if not result.selected_plan:
                raise RuntimeError(
                    f"M4 found no plan: {result.status.value}; see m4.json rejections"
                )
        else:
            verify_artifact(out, manifest, "m4.json")
            stage_status["m4"] = "resumed"
        if stop == 2 or args.skip_m5:
            persist_timing()
            return
        stage = "m5"
        # Isolate each M5 attempt, including caches and old summaries on resume.
        attempt = len(manifest.setdefault("m5_attempts", [])) + 1
        attempt_out = out / f"motion_attempt_{attempt:03d}"
        attempt_out.mkdir()
        shutil.copyfile(out / "m4.json", attempt_out / "m4.json")
        manifest["m5_attempts"].append(
            {
                "directory": str(attempt_out),
                "m4_sha256": digest(out / "m4.json"),
                "m5_args": args.m5_args,
                "simulate": args.m5_simulate,
                "validate_only": args.m5_validate_only,
                "environment": args.m5_environment or pipeline.TASK_ENV.get(args.task),
            }
        )
        write(manifest_path, manifest)
        t0 = time.monotonic()
        with isolated_motion_cache(attempt_out / "keyframe_cache"):
            pipeline.stage_m5(args.task, attempt_out, copy.deepcopy(args))
        stage_seconds["m5"] = time.monotonic() - t0
        summary = attempt_out / "m5" / "m5_summary.json"
        manifest["stages"][stage] = (
            read(summary).get("status", "returned") if summary.exists() else "returned"
        )
        stage_status["m5"] = manifest["stages"][stage]
        write(manifest_path, manifest)
        persist_timing()
    except BaseException as exc:
        # Record the failing boundary, not an unsupported causal attribution.
        manifest["stages"][stage] = "failed"
        stage_status[stage] = "failed"
        manifest["failure"] = {
            "stage": stage,
            "type": type(exc).__name__,
            "causal_attribution": "not_established",
        }
        write(manifest_path, manifest)
        try:
            persist_timing(incomplete=True)
        except Exception:  # noqa: BLE001 - never mask the original failure
            pass
        raise
