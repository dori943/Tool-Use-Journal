"""Generic M4 SelectedPlan to M5 MotionPlan command-line workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tuj.m4_taskplanner.serialization import SelectedPlan

from tuj.m5_motion.orchestration import (
    MotionPlanStore,
    SelectedPlanMotionOrchestrator,
    SelectedPlanPlanningResult,
)
from tuj.m5_motion.execution import SimulationArtifactStore
from tuj.m5_motion.schema import (
    JointDynamicLimit,
    MotionConstraints,
    PlannerOptions,
    WorldSnapshot,
)
from tuj.m5_motion.precomputed_ee_attach import EEAttachPolicy
from tuj.m5_motion.selected_plan_adapter import (
    ConstraintSource,
    OptionSource,
    SelectedPlanAdapterError,
    SelectedPlanMotionRequestAdapter,
)


REPOSITORY = Path(__file__).resolve().parents[3]


class GenericMotionRunnerError(ValueError):
    """The generic runner input contract is incomplete or inconsistent."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise GenericMotionRunnerError(f"input file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise GenericMotionRunnerError(
            f"invalid JSON in {path}: line {error.lineno}, column {error.colno}"
        ) from error


def load_selected_plan(path: Path) -> tuple[SelectedPlan, dict[str, Any]]:
    """Accept a complete PlanningResult envelope or a bare SelectedPlan."""

    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise GenericMotionRunnerError("Task Planner input must be a JSON object")
    status = payload.get("status")
    if status is not None and str(status).upper() != "SUCCESS":
        raise GenericMotionRunnerError(
            f"Task Planner result status is {status!r}, expected 'SUCCESS'"
        )
    raw_selected = payload.get("selected_plan", payload)
    if not isinstance(raw_selected, Mapping):
        raise GenericMotionRunnerError("Task Planner result has no selected_plan")
    try:
        selected = SelectedPlan.model_validate(raw_selected)
    except Exception as error:  # noqa: BLE001 - normalize Pydantic errors for CLI
        raise GenericMotionRunnerError(
            f"invalid Task Planner selected_plan: {error}"
        ) from error
    return selected, dict(payload)


def load_world(path: Path) -> WorldSnapshot:
    payload = _read_json(path)
    if isinstance(payload, Mapping) and "world" in payload:
        payload = payload["world"]
    try:
        return WorldSnapshot.model_validate(payload)
    except Exception as error:  # noqa: BLE001
        raise GenericMotionRunnerError(f"invalid initial WorldSnapshot: {error}") from error


def default_constraints(world: WorldSnapshot) -> MotionConstraints:
    """Create conservative limits when a task has no explicit constraint file."""

    return MotionConstraints(
        joint_limits={
            name: JointDynamicLimit(
                max_velocity_rad_s=1.0,
                max_acceleration_rad_s2=2.0,
            )
            for name in world.robot_state.joint_names
        }
    )


def _load_source(
    path: Path | None,
    selected: SelectedPlan,
    model: type[MotionConstraints] | type[PlannerOptions],
    fallback: MotionConstraints | PlannerOptions,
    label: str,
) -> ConstraintSource | OptionSource:
    if path is None:
        return fallback
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise GenericMotionRunnerError(f"{label} JSON must be an object")
    subgoals = set(selected.subgoal_order)
    if subgoals and subgoals <= set(payload):
        try:
            return {
                subgoal_id: model.model_validate(payload[subgoal_id])
                for subgoal_id in selected.subgoal_order
            }
        except Exception as error:  # noqa: BLE001
            raise GenericMotionRunnerError(
                f"invalid per-subgoal {label}: {error}"
            ) from error
    try:
        return model.model_validate(payload)
    except Exception as error:  # noqa: BLE001
        raise GenericMotionRunnerError(f"invalid {label}: {error}") from error


def _source_for_validation(
    source: ConstraintSource | OptionSource,
    selected: SelectedPlan,
) -> ConstraintSource | OptionSource:
    if isinstance(source, Mapping):
        return source
    return {
        subgoal_id: source.model_copy(deep=True)
        for subgoal_id in selected.subgoal_order
    }


def validate_selected_plan(
    selected: SelectedPlan,
    world: WorldSnapshot,
    constraints: ConstraintSource,
    options: OptionSource,
) -> dict[str, Any]:
    """Validate every grounded subgoal without OpenAI or trajectory planning."""

    worlds = {
        subgoal_id: world.model_copy(deep=True)
        for subgoal_id in selected.subgoal_order
    }
    requests = SelectedPlanMotionRequestAdapter().convert(
        selected,
        worlds=worlds,
        constraints=_source_for_validation(constraints, selected),
        options=_source_for_validation(options, selected),
    )
    transitions = [step.action for step in selected.steps if step.kind == "transition"]
    resources = [
        {
            "subgoal_id": assignment.subgoal_id,
            "ee": assignment.ee,
            "tool": assignment.tool,
            "action_type": assignment.action_type,
            "target_ids": list(assignment.target_ids),
        }
        for assignment in selected.candidate_assignments
    ]
    return {
        "status": "VALID",
        "subgoal_count": len(selected.subgoal_order),
        "request_count": len(requests),
        "subgoal_order": list(selected.subgoal_order),
        "resources": resources,
        "transition_actions": transitions,
        "environment_name": world.metadata.get("environment_name"),
        "initial_active_ee": world.metadata.get("physical_active_ee"),
    }


def _parse_initial_ee(value: str) -> str | None:
    normalized = value.strip()
    if normalized.casefold() in {"", "none", "null", "bare", "bare-flange"}:
        return None
    return normalized


def capture_initial_world(
    repository: Path,
    environment_name: str,
    *,
    initial_ee: str | None,
    seed: int,
) -> WorldSnapshot:
    from tuj.m5_motion.tool_use_journal import (
        ToolUseJournalEnvironmentAdapter,
        make_tool_use_journal_env,
    )

    env = make_tool_use_journal_env(
        repository,
        environment_name,
        active_ee=initial_ee,
        seed=seed,
    )
    try:
        env.reset()  # type: ignore[attr-defined]
        adapter = ToolUseJournalEnvironmentAdapter(env)
        adapter.require_physical_ee(initial_ee)
        world = adapter.world_snapshot()
        world.metadata["environment_name"] = environment_name
        world.metadata["physical_active_ee"] = initial_ee
        world.metadata["declared_active_ee"] = initial_ee
        return world
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


class ToolUseJournalPlannerPool:
    """Lazily bind one physical planner to each requested environment/EE state."""

    def __init__(
        self,
        repository: Path,
        *,
        seed: int,
        ee_attach_registry_root: Path | None = None,
        ee_attach_trajectory_paths: Sequence[Path] = (),
        ee_return_trajectory_paths: Sequence[Path] = (),
        ee_attach_policy: EEAttachPolicy | str = EEAttachPolicy.PRECOMPUTED_REQUIRED,
        ee_attach_start_tolerance_rad: float = 0.01,
    ) -> None:
        self.repository = repository
        self.seed = seed
        self.ee_attach_registry_root = ee_attach_registry_root
        self.ee_attach_trajectory_paths = tuple(ee_attach_trajectory_paths)
        self.ee_return_trajectory_paths = tuple(ee_return_trajectory_paths)
        self.ee_attach_policy = EEAttachPolicy(ee_attach_policy)
        self.ee_attach_start_tolerance_rad = ee_attach_start_tolerance_rad
        self._planners: dict[tuple[str, str | None], Any] = {}
        self._environments: list[Any] = []

    def __call__(self, request: Any) -> Any:
        from tuj.m5_motion.tool_use_journal import make_tool_use_journal_env
        from tuj.m5_motion.tool_use_journal_planning import (
            ToolUseJournalMotionRequestPlanner,
        )

        raw_environment = request.world.metadata.get("environment_name")
        if not isinstance(raw_environment, str) or not raw_environment:
            raise GenericMotionRunnerError(
                "WorldSnapshot.metadata.environment_name is required for planning"
            )
        raw_active_ee = request.world.metadata.get("physical_active_ee")
        active_ee = raw_active_ee if isinstance(raw_active_ee, str) else None
        key = (raw_environment, active_ee)
        planner = self._planners.get(key)
        if planner is None:
            env = make_tool_use_journal_env(
                self.repository,
                raw_environment,
                active_ee=active_ee,
                seed=self.seed,
            )
            env.reset()
            self._environments.append(env)
            planner = ToolUseJournalMotionRequestPlanner.from_environment(
                env,
                self.repository,
                seed=self.seed,
                ee_attach_registry_root=self.ee_attach_registry_root,
                ee_attach_trajectory_paths=self.ee_attach_trajectory_paths,
                ee_return_trajectory_paths=self.ee_return_trajectory_paths,
                ee_attach_policy=self.ee_attach_policy,
                ee_attach_start_tolerance_rad=(
                    self.ee_attach_start_tolerance_rad
                ),
            )
            self._planners[key] = planner
        return planner(request)

    def close(self) -> None:
        for env in reversed(self._environments):
            close = getattr(env, "close", None)
            if callable(close):
                close()
        self._environments.clear()
        self._planners.clear()


class GenericSimulationVideoRecorder:
    """Record the runtime's current EE environment at simulated-time cadence."""

    def __init__(
        self,
        runtime: Any,
        path: Path,
        *,
        camera: str,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("video dimensions must be positive")
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("video FPS must be finite and positive")
        import cv2

        self.runtime = runtime
        self.path = path.resolve()
        self.camera = camera
        self.width = width
        self.height = height
        self.fps = fps
        self._cv2 = cv2
        self._last_simulation_time_s: float | None = None
        self._capture_credit = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not self._writer.isOpened():
            self._writer.release()
            raise RuntimeError(f"could not open video writer for {self.path}")
        try:
            runtime.set_render_callback(self.capture)
            self._write_frame(runtime.env)
        except Exception:
            runtime.set_render_callback(None)
            self._writer.release()
            raise

    @staticmethod
    def _simulation_time(env: Any) -> float:
        raw_data = getattr(getattr(env, "sim", None), "data", None)
        raw_time = getattr(raw_data, "time", None)
        if raw_time is None and raw_data is not None:
            raw_time = getattr(getattr(raw_data, "_data", None), "time", None)
        return float(raw_time or 0.0)

    def _write_frame(self, env: Any) -> None:
        rgb = env.sim.render(
            camera_name=self.camera,
            width=self.width,
            height=self.height,
        )[::-1]
        self._writer.write(
            self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR)
        )

    def capture(self, env: Any) -> None:
        """Capture at the requested FPS even when runtime swaps EE models."""

        current = self._simulation_time(env)
        previous = self._last_simulation_time_s
        self._last_simulation_time_s = current
        if previous is None or current < previous:
            self._capture_credit = 0.0
            self._write_frame(env)
            return
        self._capture_credit += (current - previous) * self.fps
        frame_count = int(self._capture_credit + 1e-12)
        if frame_count <= 0:
            return
        self._capture_credit -= frame_count
        for _ in range(frame_count):
            self._write_frame(env)

    def hold_final_frame(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        for _ in range(round(seconds * self.fps)):
            self._write_frame(self.runtime.env)

    def close(self) -> None:
        self.runtime.set_render_callback(None)
        self._writer.release()


def _runtime_active_ee(world: WorldSnapshot) -> str | None:
    raw = world.metadata.get(
        "physical_active_ee",
        world.metadata.get("declared_active_ee"),
    )
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise GenericMotionRunnerError(
            "initial world active EE metadata must be a string or null"
        )
    return _parse_initial_ee(raw)


def _runtime_environment_name(world: WorldSnapshot) -> str:
    raw = world.metadata.get("environment_name")
    if not isinstance(raw, str) or not raw:
        raise GenericMotionRunnerError(
            "WorldSnapshot.metadata.environment_name is required for simulation"
        )
    return raw


def _validate_runtime_start(
    runtime: Any,
    world: WorldSnapshot,
    *,
    joint_tolerance_rad: float = 1e-4,
) -> None:
    """Fail closed when a file snapshot cannot be reproduced by env reset."""

    from tuj.m5_motion.tool_use_journal import ToolUseJournalEnvironmentAdapter

    if runtime.active_ee != _runtime_active_ee(world):
        raise GenericMotionRunnerError(
            "simulation runtime active EE differs from the initial WorldSnapshot"
        )
    observed = ToolUseJournalEnvironmentAdapter(runtime.env).world_snapshot()
    expected_state = world.robot_state
    observed_state = observed.robot_state
    if (
        expected_state.attached_object_id is not None
        or expected_state.held_tool_id is not None
    ):
        raise GenericMotionRunnerError(
            "an initial WorldSnapshot with a held or attached object cannot be "
            "reproduced by a fresh environment reset; start from an empty mount "
            "or add an explicit acquisition subgoal"
        )
    if expected_state.robot_id != observed_state.robot_id:
        raise GenericMotionRunnerError(
            "simulation runtime robot_id differs from the initial WorldSnapshot"
        )
    observed_positions = dict(
        zip(
            observed_state.joint_names,
            observed_state.joint_positions_rad,
            strict=True,
        )
    )
    missing = [
        name for name in expected_state.joint_names if name not in observed_positions
    ]
    if missing:
        raise GenericMotionRunnerError(
            f"simulation runtime is missing initial joints {missing}"
        )
    error = max(
        (
            abs(float(expected) - float(observed_positions[name]))
            for name, expected in zip(
                expected_state.joint_names,
                expected_state.joint_positions_rad,
                strict=True,
            )
        ),
        default=0.0,
    )
    if error > joint_tolerance_rad:
        raise GenericMotionRunnerError(
            "initial WorldSnapshot cannot be reproduced by the selected "
            f"environment reset (max joint error {error:.6f} rad). Use "
            "--environment to capture and execute one deterministic reset."
        )


def execute_planning_result(
    planning: SelectedPlanPlanningResult,
    *,
    repository: Path,
    initial_world: WorldSnapshot,
    output_dir: Path,
    mode: str,
    seed: int,
    show_viewer: bool,
    realtime_factor: float,
    hold_seconds: float,
    video: Path | None,
    camera: str,
    width: int,
    height: int,
    video_fps: float,
    video_hold_seconds: float,
) -> Any:
    """Replay a planned sequence in one state-preserving Tool-Use-Journal runtime."""

    from tuj.m5_motion.tool_use_journal_execution import (
        ToolUseJournalExecutionAdapter,
    )
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime

    environment_name = _runtime_environment_name(initial_world)
    active_ee = _runtime_active_ee(initial_world)
    record_video = video is not None
    suite_options = {
        "ignore_done": True,
        "use_camera_obs": False,
        "has_renderer": bool(show_viewer and not record_video),
        "has_offscreen_renderer": record_video,
        "render_camera": camera,
    }
    if record_video:
        suite_options.update(
            {
                "camera_names": camera,
                "camera_heights": height,
                "camera_widths": width,
            }
        )
    if mode == "controller":
        runtime = ToolUseJournalEERuntime.from_repository_for_controller(
            repository,
            environment_name,
            active_ee=active_ee,
            seed=seed,
            **suite_options,
        )
    elif mode == "kinematic":
        runtime = ToolUseJournalEERuntime.from_repository(
            repository,
            environment_name,
            active_ee=active_ee,
            seed=seed,
            **suite_options,
        )
    else:
        raise GenericMotionRunnerError(f"unsupported simulation mode {mode!r}")

    recorder: GenericSimulationVideoRecorder | None = None
    try:
        _validate_runtime_start(runtime, initial_world)
        if video is not None:
            recorder = GenericSimulationVideoRecorder(
                runtime,
                video,
                camera=camera,
                width=width,
                height=height,
                fps=video_fps,
            )
        adapter = ToolUseJournalExecutionAdapter.from_repository(
            runtime,
            repository,
            seed=seed,
            controller=mode == "controller",
            realtime_factor=realtime_factor,
            render=show_viewer or record_video,
        )
        execution = adapter.execute(
            planning,
            store=SimulationArtifactStore(output_dir / "simulation"),
        )
        if recorder is not None:
            recorder.hold_final_frame(video_hold_seconds)
        elif show_viewer and hold_seconds > 0.0:
            deadline = time.monotonic() + hold_seconds
            while time.monotonic() < deadline:
                runtime.render()
                time.sleep(0.02)
        return execution
    finally:
        try:
            if recorder is not None:
                recorder.close()
        finally:
            runtime.close()


def _safe_slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return normalized or "task"


def _parser(repository: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan any M4 SelectedPlan with the generic M5 orchestration path. "
            "Use run_m5_c1_1_motion_planner.py for the C1_1-specific physical demo."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task-planner", type=Path, required=True)
    parser.add_argument(
        "--initial-world",
        type=Path,
        help="WorldSnapshot JSON; omit to capture from --environment",
    )
    parser.add_argument(
        "--environment",
        help="Tool-Use-Journal environment name used when capturing a world",
    )
    parser.add_argument(
        "--initial-ee",
        default="none",
        help="mounted EE for environment capture; use none/null/bare for no EE",
    )
    parser.add_argument("--constraints", type=Path)
    parser.add_argument("--options", type=Path)
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ee-attach-policy",
        choices=tuple(policy.value for policy in EEAttachPolicy),
        default=EEAttachPolicy.PRECOMPUTED_REQUIRED.value,
        help=(
            "Require validated EE attach/return trajectories, or explicitly "
            "allow development-only dynamic planning fallback"
        ),
    )
    parser.add_argument(
        "--ee-attach-registry",
        type=Path,
        help=(
            "Root containing <environment>/bare_to_<EE>.json and "
            "<EE>_to_bare.json trajectories"
        ),
    )
    parser.add_argument(
        "--ee-attach-trajectory",
        type=Path,
        action="append",
        default=[],
        help="Explicit EEAttachTrajectoryTemplate override; may be repeated",
    )
    parser.add_argument(
        "--ee-return-trajectory",
        type=Path,
        action="append",
        default=[],
        help="Explicit EEReturnTrajectoryTemplate override; may be repeated",
    )
    parser.add_argument(
        "--ee-attach-start-tolerance-rad",
        type=float,
        default=0.01,
        help=(
            "Maximum per-joint error from stored bare-home or EE exchange-entry"
        ),
    )
    parser.add_argument("--validate-input-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--simulate",
        choices=("kinematic", "controller"),
        help=(
            "replay all generated MotionPlans in MuJoCo; --video implies "
            "controller when this option is omitted"
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run simulation without opening the live MuJoCo viewer",
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        help="simulation playback speed; defaults to 1 for viewer and 0 otherwise",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=3.0,
        help="keep the live viewer open after execution",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="record offscreen simulation to MP4 (implies --simulate controller)",
    )
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--video-hold-seconds", type=float, default=3.0)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    repository: Path = REPOSITORY,
) -> int:
    parser = _parser(repository)
    args = parser.parse_args(argv)
    if args.realtime_factor is not None and (
        not math.isfinite(args.realtime_factor) or args.realtime_factor < 0.0
    ):
        parser.error("--realtime-factor must be finite and non-negative")
    if (
        not math.isfinite(args.ee_attach_start_tolerance_rad)
        or args.ee_attach_start_tolerance_rad < 0.0
    ):
        parser.error(
            "--ee-attach-start-tolerance-rad must be finite and non-negative"
        )
    if not math.isfinite(args.hold_seconds) or args.hold_seconds < 0.0:
        parser.error("--hold-seconds must be finite and non-negative")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if not math.isfinite(args.video_fps) or args.video_fps <= 0.0:
        parser.error("--video-fps must be finite and positive")
    if (
        not math.isfinite(args.video_hold_seconds)
        or args.video_hold_seconds < 0.0
    ):
        parser.error("--video-hold-seconds must be finite and non-negative")
    if args.video is not None and args.validate_input_only:
        parser.error("--video cannot be combined with --validate-input-only")
    if args.video is not None and args.dry_run:
        parser.error("--video cannot be combined with --dry-run")
    simulation_mode = args.simulate or (
        "controller" if args.video is not None else None
    )
    if simulation_mode is not None and args.validate_input_only:
        parser.error("--simulate cannot be combined with --validate-input-only")
    if simulation_mode is not None and args.dry_run:
        parser.error("--simulate cannot be combined with --dry-run")
    if args.headless and simulation_mode is None:
        parser.error("--headless requires --simulate")
    task_planner = args.task_planner.expanduser().resolve()
    repository_path = args.repository.expanduser().resolve()
    if not repository_path.is_dir():
        parser.error(f"repository not found: {repository_path}")

    try:
        selected, envelope = load_selected_plan(task_planner)
        if args.initial_world is not None:
            world = load_world(args.initial_world.expanduser().resolve())
            world_environment = world.metadata.get("environment_name")
            if args.environment and world_environment not in {None, args.environment}:
                raise GenericMotionRunnerError(
                    "--environment does not match initial world metadata: "
                    f"{args.environment!r} != {world_environment!r}"
                )
            if args.environment and world_environment is None:
                world.metadata["environment_name"] = args.environment
        else:
            if not args.environment:
                raise GenericMotionRunnerError(
                    "provide --initial-world or --environment"
                )
            world = capture_initial_world(
                repository_path,
                args.environment,
                initial_ee=_parse_initial_ee(args.initial_ee),
                seed=args.seed,
            )

        constraints = _load_source(
            args.constraints.expanduser().resolve() if args.constraints else None,
            selected,
            MotionConstraints,
            default_constraints(world),
            "MotionConstraints",
        )
        options = _load_source(
            args.options.expanduser().resolve() if args.options else None,
            selected,
            PlannerOptions,
            PlannerOptions(random_seed=args.seed),
            "PlannerOptions",
        )
        report = validate_selected_plan(selected, world, constraints, options)
    except (GenericMotionRunnerError, SelectedPlanAdapterError) as error:
        parser.error(str(error))

    slug_source = task_planner.parent.name or task_planner.stem
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else repository_path / "artifacts" / _safe_slug(slug_source)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_world_path = output_dir / "initial_world.json"
    initial_world_path.write_text(world.model_dump_json(indent=2), encoding="utf-8")
    report["initial_world"] = str(initial_world_path)
    validation_path = output_dir / "m5_input_validation.json"
    validation_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[M5] validation: {validation_path}")
    if args.validate_input_only or args.dry_run:
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error(
            "OPENAI_API_KEY is required for generic keyframe generation; "
            "use --validate-input-only to check inputs without it"
        )

    selected_hash = hashlib.sha256(
        json.dumps(
            selected.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    artifact_id = str(
        envelope.get("artifact_id")
        or f"task-planner:selected-plan:{selected_hash[:24]}"
    )
    planner_pool_options: dict[str, Any] = {"seed": args.seed}
    if args.ee_attach_registry is not None:
        planner_pool_options["ee_attach_registry_root"] = (
            args.ee_attach_registry.expanduser().resolve()
        )
    if args.ee_attach_trajectory:
        planner_pool_options["ee_attach_trajectory_paths"] = tuple(
            path.expanduser().resolve() for path in args.ee_attach_trajectory
        )
    if args.ee_return_trajectory:
        planner_pool_options["ee_return_trajectory_paths"] = tuple(
            path.expanduser().resolve() for path in args.ee_return_trajectory
        )
    if args.ee_attach_policy != EEAttachPolicy.PRECOMPUTED_REQUIRED.value:
        planner_pool_options["ee_attach_policy"] = args.ee_attach_policy
    if not math.isclose(args.ee_attach_start_tolerance_rad, 0.01, abs_tol=0.0):
        planner_pool_options["ee_attach_start_tolerance_rad"] = (
            args.ee_attach_start_tolerance_rad
        )
    planners = ToolUseJournalPlannerPool(repository_path, **planner_pool_options)
    try:
        result = SelectedPlanMotionOrchestrator(
            planners,
            store=MotionPlanStore(output_dir),
        ).plan(
            selected,
            initial_world=world,
            constraints=constraints,
            options=options,
            selected_plan_artifact_id=artifact_id,
        )
    finally:
        planners.close()

    summary = {
        **report,
        "status": "SUCCESS",
        "planning_status": "SUCCESS",
        "planned_request_count": len(result.requests),
        "motion_plan_count": len(result.plans),
        "manifest": str(result.manifest_path) if result.manifest_path else None,
        "final_scene_signature": result.final_world.scene.signature,
    }
    exit_code = 0
    if simulation_mode is not None:
        show_viewer = not args.headless and args.video is None
        realtime_factor = (
            args.realtime_factor
            if args.realtime_factor is not None
            else (1.0 if show_viewer else 0.0)
        )
        video_path = (
            args.video.expanduser().resolve() if args.video is not None else None
        )
        try:
            execution = execute_planning_result(
                result,
                repository=repository_path,
                initial_world=world,
                output_dir=output_dir,
                mode=simulation_mode,
                seed=args.seed,
                show_viewer=show_viewer,
                realtime_factor=realtime_factor,
                hold_seconds=args.hold_seconds,
                video=video_path,
                camera=args.camera,
                width=args.width,
                height=args.height,
                video_fps=args.video_fps,
                video_hold_seconds=args.video_hold_seconds,
            )
        except GenericMotionRunnerError as error:
            summary.update(
                {
                    "status": "SIMULATION_SETUP_FAILED",
                    "simulation_status": "SIMULATION_SETUP_FAILED",
                    "simulation_successful": False,
                    "simulation_mode": simulation_mode,
                    "simulation_detail": str(error),
                    "video": str(video_path) if video_path is not None else None,
                }
            )
            exit_code = 2
        else:
            summary.update(
                {
                    "status": (
                        "SUCCESS"
                        if execution.successful
                        else execution.status.value
                    ),
                    "simulation_status": execution.status.value,
                    "simulation_successful": execution.successful,
                    "simulation_mode": simulation_mode,
                    "simulation_run_count": len(execution.runs),
                    "simulation_report_count": len(execution.reports),
                    "simulation_manifest": (
                        str(execution.manifest_path)
                        if execution.manifest_path is not None
                        else None
                    ),
                    "video": str(video_path) if video_path is not None else None,
                }
            )
            if not execution.successful:
                summary["simulation_detail"] = execution.detail
                summary["simulation_failed_index"] = execution.failed_index
                exit_code = 2
    summary_path = output_dir / "m5_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[M5] output: {output_dir}")
    return exit_code


__all__ = [
    "GenericMotionRunnerError",
    "GenericSimulationVideoRecorder",
    "ToolUseJournalPlannerPool",
    "capture_initial_world",
    "default_constraints",
    "execute_planning_result",
    "load_selected_plan",
    "load_world",
    "main",
    "validate_selected_plan",
]
