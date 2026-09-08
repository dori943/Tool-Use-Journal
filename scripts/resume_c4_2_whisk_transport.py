"""Resume the saved successful whisk grasp and verify generic diagonal transport."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))


class _UnexpectedFallback:
    def generate(self, request):
        raise RuntimeError(
            f"packing geometry did not bind {request.task.subgoal_id}"
        )


def _write(path: Path, value) -> None:
    if hasattr(value, "model_dump_json"):
        text = value.model_dump_json(indent=2)
    else:
        payload = asdict(value) if is_dataclass(value) else value
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    path.write_text(text + "\n", encoding="utf-8")


def _save_state(path: Path, runtime) -> None:
    from tuj.m5_motion.tool_use_journal import _raw_model_data

    _model, data = _raw_model_data(runtime.env)
    np.savez_compressed(
        path,
        qpos=np.asarray(data.qpos, dtype=float),
        qvel=np.asarray(data.qvel, dtype=float),
        ctrl=np.asarray(data.ctrl, dtype=float),
        time=float(data.time),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--place-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    from tuj.m5_motion.execution import SimulationArtifactStore
    from tuj.m5_motion.generic_runner import (
        default_constraints,
        load_selected_plan,
    )
    from tuj.m5_motion.object_function_grasp import (
        make_function_runtime,
        prepare_catalog_environment,
        snapshot,
    )
    from tuj.m5_motion.object_function_runner import _write_planning_failure
    from tuj.m5_motion.orchestration import SelectedPlanPlanningResult
    from tuj.m5_motion.packing import PackingKeyframeProvider
    from tuj.m5_motion.schema import PlannerOptions
    from tuj.m5_motion.selected_plan_adapter import SelectedPlanMotionRequestAdapter
    from tuj.m5_motion.tool_use_journal import (
        ToolUseJournalCollisionModelCompiler,
        _raw_model_data,
    )
    from tuj.m5_motion.tool_use_journal_execution import ToolUseJournalExecutionAdapter
    from tuj.m5_motion.tool_use_journal_planning import ToolUseJournalMotionRequestPlanner

    runtime = make_function_runtime(
        REPOSITORY,
        "C4_2_DiagonalFitPacking",
        active_ee="2F",
        seed=0,
        ignore_done=True,
        use_camera_obs=False,
        has_offscreen_renderer=False,
    )
    summary = {"status": "FAILED", "stage": "RESTORE"}
    try:
        state = np.load(args.state)
        model, data = _raw_model_data(runtime.env)
        for name in ("qpos", "qvel", "ctrl"):
            destination = getattr(data, name)
            source = state[name]
            if destination.shape != source.shape:
                raise ValueError(
                    f"saved {name} shape {source.shape} != runtime {destination.shape}"
                )
            destination[:] = source
        data.time = float(state["time"])
        mujoco.mj_forward(model, data)
        runtime.command_gripper(engaged=True, suction=False)
        runtime.attach_object(
            "whisk",
            attachment_mode="KINEMATIC",
            # Saved post-transport poses can have a small collision-mesh gap
            # while retaining the exact rigid transform used during playback.
            max_attach_distance_m=0.04,
        )
        runtime.capture_gripper_hold()

        world = snapshot(runtime)
        _write(args.output_dir / "restored_world.json", world)
        selected, _ = load_selected_plan(REPOSITORY / "output/c4_2/m4.json")
        options = PlannerOptions(
            allowed_planning_time_s=12.0,
            max_attempts=8,
            random_seed=0,
        )
        requests = SelectedPlanMotionRequestAdapter(
            acquire_task_metadata={"grasp_execution_mode": "KINEMATIC"}
        ).convert(
            selected,
            worlds=lambda _subgoal, _index: world,
            constraints=default_constraints(world),
            options=options,
        )
        transport_request = next(
            item for item in requests if item.task.subgoal_id == "SG1_s5_d2"
        )
        _write(args.output_dir / "transport_request.json", transport_request)

        summary["stage"] = "PLAN"
        planner = ToolUseJournalMotionRequestPlanner.from_environment(
            runtime.env,
            REPOSITORY,
            seed=0,
            provider=PackingKeyframeProvider(_UnexpectedFallback()),
            environment_preparer=lambda env, ee: prepare_catalog_environment(
                env, ee, REPOSITORY
            ),
        )
        compiler = ToolUseJournalCollisionModelCompiler.from_repository(
            runtime.env,
            REPOSITORY,
            seed=0,
            environment_preparer=lambda env, ee: prepare_catalog_environment(
                env, ee, REPOSITORY
            ),
        )
        if args.place_only:
            transport_world = world
            transport_status = "RESTORED"
        else:
            transport_planning = planner(transport_request)
            _write(
                args.output_dir / "transport_keyframes.json",
                transport_planning.keyframe_artifact,
            )
            _write(
                args.output_dir / "transport_motion_plan.json",
                transport_planning.plan,
            )

            summary["stage"] = "EXECUTE"
            transport_execution = ToolUseJournalExecutionAdapter(
                runtime,
                compiler=compiler,
                controller=True,
                realtime_factor=0.0,
                render=False,
            ).execute(
                SelectedPlanPlanningResult(
                    (transport_request,),
                    (transport_planning.plan,),
                    transport_request.world,
                ),
                store=SimulationArtifactStore(
                    args.output_dir / "transport_simulation"
                ),
            )
            _write(
                args.output_dir / "transport_execution.json",
                transport_execution,
            )
            _save_state(args.output_dir / "transport_final_state.npz", runtime)
            if not transport_execution.successful:
                summary.update(
                    status="FAILED",
                    stage="TRANSPORT",
                    execution_status=transport_execution.status.value,
                    detail=transport_execution.detail,
                )
                return 2
            transport_world = transport_execution.final_world
            transport_status = transport_execution.status.value

        summary["stage"] = "PLACE_PLAN"
        place_request = next(
            item for item in requests if item.task.subgoal_id == "SG1_s5_d3"
        )
        place_request.world = transport_world.model_copy(deep=True)
        _write(args.output_dir / "place_request.json", place_request)
        try:
            place_planning = planner(place_request)
        except Exception as error:
            if getattr(error, "compilation", None) is not None:
                _write_planning_failure(args.output_dir, place_request, error)
            raise
        _write(
            args.output_dir / "place_keyframes.json",
            place_planning.keyframe_artifact,
        )
        _write(
            args.output_dir / "place_motion_plan.json", place_planning.plan
        )

        summary["stage"] = "PLACE_EXECUTE"
        place_execution = ToolUseJournalExecutionAdapter(
            runtime,
            compiler=compiler,
            controller=True,
            realtime_factor=0.0,
            render=False,
        ).execute(
            SelectedPlanPlanningResult(
                (place_request,), (place_planning.plan,), place_request.world
            ),
            store=SimulationArtifactStore(args.output_dir / "place_simulation"),
        )
        _write(args.output_dir / "place_execution.json", place_execution)
        _write(args.output_dir / "final_world.json", place_execution.final_world)
        _save_state(args.output_dir / "place_final_state.npz", runtime)
        summary.update(
            status="SUCCESS" if place_execution.successful else "FAILED",
            stage="COMPLETE",
            transport_status=transport_status,
            place_status=place_execution.status.value,
            detail=place_execution.detail,
        )
        return 0 if place_execution.successful else 2
    except Exception as error:
        import traceback

        summary["detail"] = str(error)
        (args.output_dir / "error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        return 2
    finally:
        _write(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
