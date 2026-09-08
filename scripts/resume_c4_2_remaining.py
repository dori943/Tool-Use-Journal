"""Execute a C4-2 M4 slice, optionally resuming a physical checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))


class _UnexpectedFallback:
    def generate(self, request):
        raise RuntimeError(
            f"no local keyframe provider for {request.task.subgoal_id}"
        )


def _safe_slug(value: str) -> str:
    """Return a cross-platform artifact directory component."""
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    )


def _write(path: Path, value) -> None:
    payload = (
        value.model_dump(mode="json")
        if hasattr(value, "model_dump")
        else value
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _save_state(path: Path, runtime) -> None:
    from tuj.m5_motion.tool_use_journal import _raw_model_data

    path.parent.mkdir(parents=True, exist_ok=True)
    _model, data = _raw_model_data(runtime.env)
    np.savez_compressed(
        path,
        qpos=np.asarray(data.qpos, dtype=float),
        qvel=np.asarray(data.qvel, dtype=float),
        ctrl=np.asarray(data.ctrl, dtype=float),
        time=float(data.time),
    )


def _restore_state(path: Path, runtime) -> None:
    from tuj.m5_motion.tool_use_journal import _raw_model_data

    saved = np.load(path)
    model, data = _raw_model_data(runtime.env)
    for name in ("qpos", "qvel", "ctrl"):
        destination = getattr(data, name)
        source = saved[name]
        if destination.shape != source.shape:
            raise ValueError(
                f"saved {name} shape {source.shape} != runtime {destination.shape}"
            )
        destination[:] = source
    data.time = float(saved["time"])
    mujoco.mj_forward(model, data)


def _slice_selected(selected, start: str, through: str | None):
    start_index = selected.subgoal_order.index(start)
    end_index = (
        selected.subgoal_order.index(through) + 1
        if through is not None
        else len(selected.subgoal_order)
    )
    if end_index <= start_index:
        raise ValueError("--through must not precede --start")
    order = selected.subgoal_order[start_index:end_index]
    order_set = set(order)
    steps = [step for step in selected.steps if step.subgoal_id in order_set]
    for index, step in enumerate(steps):
        step.step_index = index
    return selected.model_copy(
        deep=True,
        update={
            "subgoal_order": order,
            "candidate_assignments": [
                item
                for item in selected.candidate_assignments
                if item.subgoal_id in order_set
            ],
            "steps": steps,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "resume a versioned M5 checkpoint including EE, gripper, "
            "attachment transform, and completed packing state"
        ),
    )
    parser.add_argument(
        "--task-planner",
        type=Path,
        default=REPOSITORY / "output/c4_2/m4.json",
    )
    parser.add_argument(
        "--initial-ee",
        default="2F",
        help="mounted EE represented by a resumed state; use bare for a fresh run",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.0,
        help="settle restored free objects before taking the initial snapshot",
    )
    parser.add_argument(
        "--attached-object",
        help="restore this object as a kinematic gripper attachment after --state",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preplanned-plan",
        type=Path,
        help="replay one previously validated MotionPlan for a one-step slice",
    )
    parser.add_argument(
        "--kinematic-playback",
        action="store_true",
        help="apply validated trajectory waypoints directly instead of controller tracking",
    )
    parser.add_argument("--start", default="SG1_s4_d1")
    parser.add_argument("--through")
    parser.add_argument("--planning-time-s", type=float, default=12.0)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--rrt-max-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--collision-margin-m",
        type=float,
        default=0.005,
        help=(
            "minimum collision clearance used while planning; lower this only "
            "for a verified, already non-colliding recovery state"
        ),
    )
    parser.add_argument(
        "--min-jacobian-singular-value", type=float, default=1e-4
    )
    parser.add_argument(
        "--max-jacobian-condition-number", type=float, default=1e4
    )
    args = parser.parse_args()
    if args.state is not None and args.checkpoint is not None:
        raise ValueError("--state and --checkpoint are mutually exclusive")
    if args.checkpoint is not None and args.attached_object:
        raise ValueError(
            "--attached-object is legacy --state metadata; a checkpoint "
            "already contains its exact attachment"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)

    from tuj.m5_motion.execution import SimulationArtifactStore
    from tuj.m5_motion.generic_runner import default_constraints, load_selected_plan
    from tuj.m5_motion.object_function_grasp import (
        execute_object_function_grasp,
        make_function_runtime,
        prepare_catalog_environment,
        snapshot,
    )
    from tuj.m5_motion.object_function_runner import (
        ReleaseSettlingProvider,
        _write_planning_failure,
        validate_function_assignments,
    )
    from tuj.m5_motion.orchestration import (
        MotionPlanStore,
        SelectedPlanMotionOrchestrator,
        SelectedPlanPlanningResult,
    )
    from tuj.m5_motion.packing import PackingKeyframeProvider
    from tuj.m5_motion.push_to_region import target_fully_inside_region
    from tuj.m5_motion.runtime_checkpoint import (
        restore_runtime_checkpoint,
        save_runtime_checkpoint,
    )
    from tuj.m5_motion.schema import MotionPlan, PlannerOptions
    from tuj.m5_motion.selected_plan_adapter import SelectedPlanMotionRequestAdapter
    from tuj.m5_motion.task_semantics import is_acquire_task, task_operation
    from tuj.m5_motion.tool_use_journal import ToolUseJournalCollisionModelCompiler
    from tuj.m5_motion.tool_use_journal_execution import ToolUseJournalExecutionAdapter
    from tuj.m5_motion.tool_use_journal_planning import ToolUseJournalMotionRequestPlanner

    selected, _ = load_selected_plan(args.task_planner)
    selected = _slice_selected(selected, args.start, args.through)
    if args.preplanned_plan is not None and len(selected.subgoal_order) != 1:
        raise ValueError("--preplanned-plan requires a one-subgoal slice")
    preplanned_plan = (
        MotionPlan.model_validate(
            json.loads(args.preplanned_plan.read_text(encoding="utf-8"))
        )
        if args.preplanned_plan is not None
        else None
    )
    validate_function_assignments(selected, REPOSITORY)
    _write(args.output_dir / "selected_plan.json", selected)

    checkpoint_active_ee = None
    if args.checkpoint is not None:
        checkpoint_header = json.loads(
            args.checkpoint.read_text(encoding="utf-8")
        )
        checkpoint_active_ee = checkpoint_header.get("active_ee")
    requested_initial_ee = (
        checkpoint_active_ee
        if args.checkpoint is not None
        else (
            None
            if args.initial_ee.strip().lower() in {"bare", "none", "null"}
            else args.initial_ee
        )
    )
    runtime = make_function_runtime(
        REPOSITORY,
        "C4_2_DiagonalFitPacking",
        active_ee=requested_initial_ee,
        seed=args.seed,
        ignore_done=True,
        use_camera_obs=False,
        has_offscreen_renderer=True,
    )
    records = []
    summary = {"status": "FAILED", "steps": records}
    packed_targets: set[str] = set()

    def save_checkpoint(path: Path) -> None:
        save_runtime_checkpoint(
            path,
            runtime,
            progress={
                "completed_subgoals": [
                    record["subgoal_id"]
                    for record in records
                    if record.get("status") == "SUCCESS"
                ],
                "packed_targets": sorted(packed_targets),
            },
        )

    try:
        restored_progress = {}
        if args.checkpoint is not None:
            restored_progress = dict(
                restore_runtime_checkpoint(args.checkpoint, runtime).progress
            )
        elif args.state is not None:
            _restore_state(args.state, runtime)
            if args.attached_object:
                runtime.command_gripper(engaged=True, suction=False)
                runtime.attach_object(
                    args.attached_object,
                    attachment_mode="KINEMATIC",
                    max_attach_distance_m=0.04,
                )
                runtime.capture_gripper_hold()
            if args.settle_seconds > 0.0:
                from tuj.m5_motion.tool_use_journal import (
                    settle_tool_use_journal_free_objects,
                )

                settle_tool_use_journal_free_objects(
                    runtime.env, duration_s=args.settle_seconds
                )
        else:
            from tuj.m5_motion.tool_use_journal import (
                settle_tool_use_journal_free_objects,
            )

            settle_tool_use_journal_free_objects(runtime.env, duration_s=5.0)
        initial = snapshot(runtime)
        _write(args.output_dir / "initial_world.json", initial)
        _save_state(args.output_dir / "initial_state.npz", runtime)
        packing_policy = json.loads(
            (REPOSITORY / "configs/c4_2_packing_policy.json").read_text(
                encoding="utf-8"
            )
        )
        aliases = json.loads(
            (REPOSITORY / "output/c4_2/id_aliases.json").read_text(
                encoding="utf-8"
            )
        )
        packing_targets = {
            aliases[target]: target for target in packing_policy["target_order"]
        }
        packed_targets = {
            canonical
            for canonical in packing_targets
            if target_fully_inside_region(
                initial,
                target_id=canonical,
                region_id="packing_box",
                include_vertical=True,
            )
        }
        packed_targets.update(
            canonical
            for canonical in restored_progress.get("packed_targets", [])
            if canonical in packing_targets
        )
        save_checkpoint(args.output_dir / "initial.checkpoint.json")

        def function_handler(request):
            if not is_acquire_task(request.task):
                return None
            index = len(records)
            step_dir = args.output_dir / "functions" / (
                f"{index:03d}-{_safe_slug(request.task.subgoal_id)}"
            )
            target = request.task.goal.target_object_id
            if not target:
                raise ValueError("acquire request has no target_object_id")
            def control_step():
                return None

            execute_object_function_grasp(
                runtime,
                object_id=target,
                resource_kind=(
                    "tool"
                    if task_operation(request.task) == "PICK_TOOL"
                    or request.task.tool == target
                    else "object"
                ),
                output_dir=step_dir,
                repository=REPOSITORY,
                on_control_step=control_step,
                seed=args.seed,
            )
            _write(step_dir / "request.json", request)
            records.append(
                {
                    "subgoal_id": request.task.subgoal_id,
                    "kind": "object_function",
                    "status": "SUCCESS",
                    "artifact": str(step_dir),
                }
            )
            _save_state(
                args.output_dir / "checkpoints" / f"{index:03d}-after.npz",
                runtime,
            )
            save_checkpoint(
                args.output_dir
                / "checkpoints"
                / f"{index:03d}-after.checkpoint.json"
            )
            return snapshot(runtime, previous=request.world)

        def planner(request):
            if preplanned_plan is not None:
                first = np.asarray(
                    preplanned_plan.segments[0].waypoints[0].joint_positions_rad,
                    dtype=float,
                )
                current = np.asarray(
                    request.world.robot_state.joint_positions_rad, dtype=float
                )
                if first.shape != current.shape or not np.allclose(
                    first, current, atol=1e-6, rtol=0.0
                ):
                    raise ValueError(
                        "preplanned plan does not start at the restored checkpoint"
                    )
                return preplanned_plan.model_copy(
                    deep=True,
                    update={
                        "request_id": request.request_id,
                        "scene_signature": request.world.scene.signature,
                    },
                )
            bound = ToolUseJournalMotionRequestPlanner.from_environment(
                runtime.env,
                REPOSITORY,
                seed=args.seed,
                provider=ReleaseSettlingProvider(
                    PackingKeyframeProvider(_UnexpectedFallback())
                ),
                ee_attach_trajectory_paths=(
                    REPOSITORY
                    / "configs/precomputed_ee_paths/C1_1_LegoSweep/bare_to_2F.json",
                    REPOSITORY
                    / "configs/precomputed_ee_paths/C1_1_LegoSweep/bare_to_3F.json",
                    REPOSITORY
                    / "configs/precomputed_ee_paths/C1_1_LegoSweep/bare_to_vac.json",
                ),
                ee_return_trajectory_paths=(
                    REPOSITORY
                    / "configs/precomputed_ee_paths/C1_1_LegoSweep/2F_to_bare.json",
                    REPOSITORY
                    / "configs/precomputed_ee_paths/C1_1_LegoSweep/3F_to_bare.json",
                    REPOSITORY
                    / "configs/precomputed_ee_paths/C1_1_LegoSweep/vac_to_bare.json",
                ),
                environment_preparer=lambda env, ee: prepare_catalog_environment(
                    env, ee, REPOSITORY
                ),
            )
            try:
                return bound(request)
            except Exception as error:
                if getattr(error, "compilation", None) is not None:
                    _write_planning_failure(args.output_dir, request, error)
                raise

        def execute_plan(request, plan):
            index = len(records)
            compiler = ToolUseJournalCollisionModelCompiler.from_repository(
                runtime.env,
                REPOSITORY,
                seed=args.seed,
                environment_preparer=lambda env, ee: prepare_catalog_environment(
                    env, ee, REPOSITORY
                ),
            )
            execution = ToolUseJournalExecutionAdapter(
                runtime,
                compiler=compiler,
                # Direct waypoint replay is safe for already collision-checked
                # transport, but release tasks need live physics so gravity and
                # contact settling actually advance during the hold window.
                controller=(
                    not args.kinematic_playback
                    or task_operation(request.task) in {"PLACE", "PLACE_ON"}
                ),
                realtime_factor=0.0,
                render=False,
            ).execute(
                SelectedPlanPlanningResult((request,), (plan,), request.world),
                store=SimulationArtifactStore(
                    args.output_dir
                    / "simulation"
                    / f"{index:03d}-{_safe_slug(request.task.subgoal_id)}"
                ),
            )
            records.append(
                {
                    "subgoal_id": request.task.subgoal_id,
                    "kind": "motion_plan",
                    "status": execution.status.value,
                    "artifact": str(execution.manifest_path),
                }
            )
            _save_state(
                args.output_dir / "checkpoints" / f"{index:03d}-after.npz",
                runtime,
            )
            if not execution.successful:
                raise RuntimeError(execution.detail)
            if task_operation(request.task) == "PLACE":
                target = request.task.goal.target_object_id
                if target in packing_targets:
                    required = packed_targets | {target}
                    outside = sorted(
                        object_id
                        for object_id in required
                        if not target_fully_inside_region(
                            execution.final_world,
                            target_id=object_id,
                            region_id="packing_box",
                            include_vertical=True,
                        )
                    )
                    records[-1]["packing_invariant"] = {
                        "required_inside": sorted(required),
                        "outside": outside,
                    }
                    if outside:
                        raise RuntimeError(
                            "previously packed object left packing_box: "
                            + ", ".join(outside)
                        )
                    packed_targets.add(target)
            save_checkpoint(
                args.output_dir
                / "checkpoints"
                / f"{index:03d}-after.checkpoint.json"
            )
            return execution.final_world

        constraints = default_constraints(initial).model_copy(
            update={
                "collision_margin_m": args.collision_margin_m,
                "min_jacobian_singular_value": (
                    args.min_jacobian_singular_value
                ),
                "max_jacobian_condition_number": (
                    args.max_jacobian_condition_number
                ),
            }
        )
        result = SelectedPlanMotionOrchestrator(
            planner,
            store=MotionPlanStore(args.output_dir),
            adapter=SelectedPlanMotionRequestAdapter(
                acquire_task_metadata={"grasp_execution_mode": "KINEMATIC"}
            ),
            request_handler=function_handler,
            plan_executor=execute_plan,
        ).plan(
            selected,
            initial_world=initial,
            constraints=constraints,
            options=PlannerOptions(
                allowed_planning_time_s=args.planning_time_s,
                max_attempts=args.max_attempts,
                rrt_max_iterations=args.rrt_max_iterations,
                random_seed=args.seed,
            ),
        )
        _write(args.output_dir / "final_world.json", result.final_world)
        _save_state(args.output_dir / "final_state.npz", runtime)
        save_checkpoint(args.output_dir / "final.checkpoint.json")
        summary.update(
            status="SUCCESS",
            final_active_ee=runtime.active_ee,
            attached_object_id=runtime.attached_object_id,
        )
        return 0
    except Exception as error:
        import traceback

        summary["detail"] = str(error)
        (args.output_dir / "error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        _save_state(args.output_dir / "failed_state.npz", runtime)
        save_checkpoint(args.output_dir / "failed.checkpoint.json")
        return 2
    finally:
        _write(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
