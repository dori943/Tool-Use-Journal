"""Generic M4 SelectedPlan to M5 MotionPlan command-line workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tuj.m3_taskplanner.serialization import SelectedPlan

from tuj.m4_motion.orchestration import (
    MotionPlanStore,
    SelectedPlanMotionOrchestrator,
)
from tuj.m4_motion.schema import (
    JointDynamicLimit,
    MotionConstraints,
    PlannerOptions,
    WorldSnapshot,
)
from tuj.m4_motion.selected_plan_adapter import (
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
    from tuj.m4_motion.tool_use_journal import (
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

    def __init__(self, repository: Path, *, seed: int) -> None:
        self.repository = repository
        self.seed = seed
        self._planners: dict[tuple[str, str | None], Any] = {}
        self._environments: list[Any] = []

    def __call__(self, request: Any) -> Any:
        from tuj.m4_motion.tool_use_journal import make_tool_use_journal_env
        from tuj.m4_motion.tool_use_journal_planning import (
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
    parser.add_argument("--validate-input-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    repository: Path = REPOSITORY,
) -> int:
    parser = _parser(repository)
    args = parser.parse_args(argv)
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
    planners = ToolUseJournalPlannerPool(repository_path, seed=args.seed)
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
        "planned_request_count": len(result.requests),
        "motion_plan_count": len(result.plans),
        "manifest": str(result.manifest_path) if result.manifest_path else None,
        "final_scene_signature": result.final_world.scene.signature,
    }
    summary_path = output_dir / "m5_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[M5] output: {output_dir}")
    return 0


__all__ = [
    "GenericMotionRunnerError",
    "ToolUseJournalPlannerPool",
    "capture_initial_world",
    "default_constraints",
    "load_selected_plan",
    "load_world",
    "main",
    "validate_selected_plan",
]
