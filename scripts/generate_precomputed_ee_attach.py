#!/usr/bin/env python
"""Generate, controller-verify, replay, and save one bare -> EE trajectory.

This is a development/commissioning command.  A template is published only
after the dynamically generated MotionPlan executes successfully and the saved
template then succeeds repeatedly from a fresh canonical bare workcell.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    REPOSITORY / "src",
    REPOSITORY.parent / "dain-m3" / "src",
    REPOSITORY.parent / "tuj-m3" / "src",
)


def _install_source_roots() -> None:
    for root in reversed(SOURCE_ROOTS):
        if root.is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))


def _request(
    world: Any,
    *,
    target_ee: str,
    planning_seed: int,
    planning_time_s: float,
    rrt_max_iterations: int,
) -> Any:
    from tuj.m5_motion.schema import (
        ArtifactProvenance,
        GoalType,
        JointDynamicLimit,
        ModuleName,
        MotionConstraints,
        MotionGoal,
        MotionPlanRequest,
        MotionTask,
        PlannerOptions,
        Pose,
    )

    joint_names = list(world.robot_state.joint_names)
    return MotionPlanRequest(
        request_id=f"commission:bare-to-{target_ee}:seed-{planning_seed}",
        provenance=ArtifactProvenance(
            artifact_id=(
                f"commission-request-artifact:bare-to-{target_ee}:"
                f"seed-{planning_seed}"
            ),
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="precomputed-ee-attach-commissioning",
        ),
        world=world,
        task=MotionTask(
            task_id=f"commission:attach:{target_ee}",
            subgoal_id=f"commission:attach:{target_ee}",
            action_type="EE_ATTACH",
            ee=target_ee,
            target_ids=[target_ee],
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_pose=Pose.model_validate(
                    world.rack[target_ee]["dock_pose"]
                ),
                target_object_id=target_ee,
            ),
            metadata={"from_ee": None, "to_ee": target_ee},
        ),
        constraints=MotionConstraints(
            velocity_scaling=0.5,
            acceleration_scaling=0.5,
            jerk_scaling=0.5,
            max_joint_path_step_rad=0.02,
            joint_limits={
                name: JointDynamicLimit(
                    max_velocity_rad_s=1.0,
                    max_acceleration_rad_s2=2.0,
                )
                for name in joint_names
            },
        ),
        options=PlannerOptions(
            random_seed=planning_seed,
            allowed_planning_time_s=planning_time_s,
            max_attempts=12,
            rrt_max_iterations=rrt_max_iterations,
        ),
    )


def _execute_once(runtime: Any, planner: Any, request: Any, plan: Any) -> Any:
    from tuj.m5_motion.orchestration import SelectedPlanPlanningResult
    from tuj.m5_motion.tool_use_journal_execution import (
        ToolUseJournalExecutionAdapter,
    )

    execution = ToolUseJournalExecutionAdapter(
        runtime,
        compiler=planner.collision_context_factory.compiler,
        controller=True,
        realtime_factor=0.0,
        render=False,
        terminate_on_collision=True,
        random_seed=request.options.random_seed,
    ).execute(
        SelectedPlanPlanningResult(
            requests=(request,),
            plans=(plan,),
            final_world=request.world,
        )
    )
    if not execution.successful:
        raise RuntimeError(
            f"controller validation failed: {execution.status.value}: "
            f"{execution.detail}"
        )
    if runtime.active_ee != request.task.ee:
        raise RuntimeError(
            f"controller ended with active EE {runtime.active_ee!r}, "
            f"expected {request.task.ee!r}"
        )
    transitions = runtime.transitions
    if not transitions or transitions[-1].to_ee != request.task.ee:
        raise RuntimeError("runtime did not record the expected TOOL_LOCK transition")
    return execution


def _make_runtime(repository: Path, environment: str, seed: int) -> Any:
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime

    return ToolUseJournalEERuntime.from_repository_for_controller(
        repository,
        environment,
        active_ee=None,
        seed=seed,
        ignore_done=True,
        use_camera_obs=False,
        has_offscreen_renderer=False,
    )


def _make_planner(
    runtime: Any,
    repository: Path,
    *,
    seed: int,
    policy: str,
    trajectory_path: Path | None = None,
    empty_registry: Path | None = None,
) -> Any:
    from tuj.m5_motion.ee_exchange import (
        EEExchangeKeyframeProvider,
        RoutedKeyframeStrategyProvider,
    )
    from tuj.m5_motion.schema import KeyframePlannerType
    from tuj.m5_motion.tool_use_journal_planning import (
        ToolUseJournalMotionRequestPlanner,
    )

    class CommissioningEEExchangeProvider:
        """Use collision-validated RRT retreat when Cartesian IK is brittle."""

        def __init__(self) -> None:
            self._base = EEExchangeKeyframeProvider()

        def generate(self, request: Any) -> Any:
            artifact = self._base.generate(request)
            candidates = []
            for candidate in artifact.candidates:
                keyframes = list(candidate.keyframes)
                keyframes[-1] = keyframes[-1].model_copy(
                    update={"planner": KeyframePlannerType.SAMPLING_BASED}
                )
                candidates.append(
                    candidate.model_copy(update={"keyframes": keyframes})
                )
            return artifact.model_copy(update={"candidates": candidates})

    provider = RoutedKeyframeStrategyProvider(
        EEExchangeKeyframeProvider(),
        ee_exchange_provider=CommissioningEEExchangeProvider(),
    )
    return ToolUseJournalMotionRequestPlanner.from_environment(
        runtime.env,
        repository,
        provider=provider,
        seed=seed,
        ee_attach_registry_root=(empty_registry or repository),
        ee_attach_trajectory_paths=(
            (trajectory_path,) if trajectory_path is not None else ()
        ),
        ee_attach_policy=policy,
        ignore_done=True,
        use_camera_obs=False,
        has_offscreen_renderer=False,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_ee", choices=("2F", "3F", "vac"))
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--environment", default="C1_1_LegoSweep")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generation-attempts", type=int, default=8)
    parser.add_argument("--planning-time", type=float, default=12.0)
    parser.add_argument("--rrt-max-iterations", type=int, default=5000)
    parser.add_argument("--replay-count", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--replay-existing",
        action="store_true",
        help="Skip dynamic generation and repeatedly validate the existing output",
    )
    args = parser.parse_args()
    if args.generation_attempts <= 0:
        parser.error("--generation-attempts must be positive")
    if args.replay_count <= 0:
        parser.error("--replay-count must be positive")
    if not math.isfinite(args.planning_time) or args.planning_time <= 0:
        parser.error("--planning-time must be finite and positive")
    if args.rrt_max_iterations <= 0:
        parser.error("--rrt-max-iterations must be positive")
    return args


def _planning_failure_detail(error: Exception) -> str:
    compilation = getattr(error, "compilation", None)
    attempts = getattr(compilation, "attempts", ())
    summaries: list[str] = []
    for attempt in attempts:
        selection = getattr(attempt, "selection", None)
        rejected = getattr(selection, "rejected_edges", ())
        grouped = collections.Counter(
            (
                item.source_keyframe_id,
                item.target_keyframe_id,
                item.failure_code,
            )
            for item in rejected
        )
        summaries.extend(
            f"{source}->{target}:{code}={count}"
            for (source, target, code), count in grouped.most_common()
        )
    suffix = f"; rejected[{', '.join(summaries)}]" if summaries else ""
    return f"{error}{suffix}"


def _replay_template(
    path: Path,
    *,
    repository: Path,
    environment: str,
    target_ee: str,
    seed: int,
    replay_count: int,
    planning_time_s: float,
    rrt_max_iterations: int,
) -> list[dict[str, Any]]:
    from tuj.m5_motion.tool_use_journal import ToolUseJournalEnvironmentAdapter

    results: list[dict[str, Any]] = []
    for replay_index in range(replay_count):
        runtime = _make_runtime(repository, environment, seed)
        try:
            world = ToolUseJournalEnvironmentAdapter(runtime.env).world_snapshot()
            request = _request(
                world,
                target_ee=target_ee,
                planning_seed=seed,
                planning_time_s=planning_time_s,
                rrt_max_iterations=rrt_max_iterations,
            )
            request.request_id = (
                f"commission:replay:{replay_index}:bare-to-{target_ee}"
            )
            planner = _make_planner(
                runtime,
                repository,
                seed=seed,
                policy="precomputed-required",
                trajectory_path=path,
            )
            plan = planner(request)
            execution = _execute_once(runtime, planner, request, plan)
            results.append(
                {
                    "replay_index": replay_index,
                    "status": execution.status.value,
                    "trajectory_id": plan.metadata["trajectory_id"],
                    "dynamic_planner_invoked": plan.metadata[
                        "dynamic_planner_invoked"
                    ],
                }
            )
        finally:
            runtime.close()
    return results


def main() -> int:
    _install_source_roots()
    args = _parse_args()
    repository = args.repository.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else repository
        / "configs"
        / "precomputed_ee_paths"
        / args.environment
        / f"bare_to_{args.target_ee}.json"
    )
    if args.replay_existing and not output.is_file():
        raise SystemExit(f"trajectory to replay does not exist: {output}")
    if output.exists() and not args.overwrite and not args.replay_existing:
        raise SystemExit(f"refusing to overwrite existing trajectory: {output}")

    if args.replay_existing:
        replay_results = _replay_template(
            output,
            repository=repository,
            environment=args.environment,
            target_ee=args.target_ee,
            seed=args.seed,
            replay_count=args.replay_count,
            planning_time_s=args.planning_time,
            rrt_max_iterations=args.rrt_max_iterations,
        )
        print(
            json.dumps(
                {
                    "status": "REPLAY_VALIDATED",
                    "environment": args.environment,
                    "target_ee": args.target_ee,
                    "output": str(output),
                    "replays": replay_results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    from tuj.m5_motion.pipeline import MotionPlanningPipelineError
    from tuj.m5_motion.precomputed_ee_attach import save_ee_attach_template
    from tuj.m5_motion.tool_use_journal import ToolUseJournalEnvironmentAdapter

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ee-attach-commission-") as temporary:
        temporary_root = Path(temporary)
        candidate = temporary_root / f"bare_to_{args.target_ee}.json"
        generated = False
        for offset in range(args.generation_attempts):
            planning_seed = args.seed + offset
            runtime = _make_runtime(repository, args.environment, args.seed)
            try:
                world = ToolUseJournalEnvironmentAdapter(
                    runtime.env
                ).world_snapshot()
                dynamic_planner = _make_planner(
                    runtime,
                    repository,
                    seed=args.seed,
                    policy="precomputed-or-plan",
                    empty_registry=temporary_root / "empty-registry",
                )
                request = _request(
                    world.model_copy(deep=True),
                    target_ee=args.target_ee,
                    planning_seed=planning_seed,
                    planning_time_s=args.planning_time,
                    rrt_max_iterations=args.rrt_max_iterations,
                )
                try:
                    result = dynamic_planner(request)
                except MotionPlanningPipelineError as error:
                    failures.append(
                        f"seed {planning_seed}: {_planning_failure_detail(error)}"
                    )
                    continue
                plan = result.plan
                try:
                    _execute_once(runtime, dynamic_planner, request, plan)
                except RuntimeError as error:
                    failures.append(f"seed {planning_seed}: {error}")
                    continue
                save_ee_attach_template(
                    candidate,
                    request,
                    plan,
                    trajectory_id=f"ur5e-bare-to-{args.target_ee}-v1",
                )
                generated = True
                break
            finally:
                runtime.close()
        if not generated:
            raise RuntimeError(
                "no generation attempt passed controller validation:\n"
                + "\n".join(failures)
            )

        replay_results = _replay_template(
            candidate,
            repository=repository,
            environment=args.environment,
            target_ee=args.target_ee,
            seed=args.seed,
            replay_count=args.replay_count,
            planning_time_s=args.planning_time,
            rrt_max_iterations=args.rrt_max_iterations,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        candidate.replace(output)
    print(
        json.dumps(
            {
                "status": "VALIDATED_AND_SAVED",
                "environment": args.environment,
                "target_ee": args.target_ee,
                "output": str(output),
                "replays": replay_results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
