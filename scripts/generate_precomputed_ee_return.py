#!/usr/bin/env python
"""Derive, controller-verify, replay, and save one EE -> bare trajectory.

The candidate is the time-reversal of the already commissioned bare -> EE
trajectory, with velocities, collision contexts, and unlock events rebuilt for
the return direction.  It is published only after return-only controller replay
and complete source-EE -> alternate-EE composed exchanges succeed.
"""

from __future__ import annotations

import argparse
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
TARGETS = ("2F", "3F", "vac")


def _install_source_roots() -> None:
    for root in reversed(SOURCE_ROOTS):
        if root.is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))


def _make_runtime(
    repository: Path, environment: str, source_ee: str | None, seed: int
) -> Any:
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime

    return ToolUseJournalEERuntime.from_repository_for_controller(
        repository,
        environment,
        active_ee=source_ee,
        seed=seed,
        ignore_done=True,
        use_camera_obs=False,
        has_offscreen_renderer=False,
    )


def _ensure_attach_start_eef_metadata(
    repository: Path,
    environment: str,
    environment_root: Path,
    seed: int,
) -> None:
    """Migrate legacy 1.0 files with the canonical bare-home EEF pose."""

    from tuj.m5_motion.precomputed_ee_attach import EEAttachTrajectoryTemplate
    from tuj.m5_motion.tool_use_journal import ToolUseJournalEnvironmentAdapter

    paths = [environment_root / f"bare_to_{target}.json" for target in TARGETS]
    templates = [
        EEAttachTrajectoryTemplate.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        for path in paths
    ]
    if all(template.start_eef_pose is not None for template in templates):
        return
    runtime = _make_runtime(repository, environment, None, seed)
    try:
        world = ToolUseJournalEnvironmentAdapter(runtime.env).world_snapshot()
        pose = world.robot_state.eef_pose
        if pose is None:
            raise RuntimeError("canonical bare runtime has no EEF pose")
        for path, template in zip(paths, templates):
            maximum_error = max(
                abs(left - right)
                for left, right in zip(
                    world.robot_state.joint_positions_rad,
                    template.start_joint_positions_rad,
                )
            )
            if maximum_error > 1e-8:
                raise RuntimeError(
                    f"{path.name} start state differs from canonical bare home"
                )
            template.start_eef_pose = pose.model_copy(deep=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(template.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(path)
    finally:
        runtime.close()


def _set_exchange_entry(runtime: Any, joint_names: list[str], positions: list[float]) -> None:
    import mujoco

    from tuj.m5_motion.tool_use_journal import ToolUseJournalEnvironmentAdapter

    adapter = ToolUseJournalEnvironmentAdapter(runtime.env)
    if list(adapter.robot.robot_model.joints) != joint_names:
        raise RuntimeError("runtime joint names differ from the attach trajectory")
    for name, position in zip(joint_names, positions):
        joint_id = mujoco.mj_name2id(
            adapter.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise RuntimeError(f"runtime joint {name!r} is absent")
        adapter.data.qpos[adapter.model.jnt_qposadr[joint_id]] = float(position)
        adapter.data.qvel[adapter.model.jnt_dofadr[joint_id]] = 0.0
    adapter.data.time = 0.0
    mujoco.mj_forward(adapter.model, adapter.data)
    runtime.synchronize_attached_object()


def _request(
    world: Any,
    *,
    source_ee: str,
    target_ee: str,
    seed: int,
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

    return MotionPlanRequest(
        request_id=f"commission:return:{source_ee}-to-{target_ee}:seed-{seed}",
        provenance=ArtifactProvenance(
            artifact_id=(
                f"commission-return-request-artifact:{source_ee}-to-{target_ee}:"
                f"seed-{seed}"
            ),
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="precomputed-ee-return-commissioning",
        ),
        world=world,
        task=MotionTask(
            task_id=f"commission:return:{source_ee}-to-{target_ee}",
            subgoal_id=f"commission:return:{source_ee}-to-{target_ee}",
            action_type="EE_EXCHANGE",
            ee=target_ee,
            target_ids=[source_ee, target_ee],
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_pose=Pose.model_validate(world.rack[target_ee]["dock_pose"]),
                target_object_id=target_ee,
            ),
            metadata={"from_ee": source_ee, "to_ee": target_ee},
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
                for name in world.robot_state.joint_names
            },
        ),
        options=PlannerOptions(random_seed=seed),
    )


def _make_planner(
    runtime: Any,
    repository: Path,
    registry_root: Path,
    *,
    seed: int,
    return_path: Path | None = None,
) -> Any:
    from tuj.m5_motion.tool_use_journal_planning import (
        ToolUseJournalMotionRequestPlanner,
    )

    return ToolUseJournalMotionRequestPlanner.from_environment(
        runtime.env,
        repository,
        seed=seed,
        ee_attach_registry_root=registry_root,
        ee_return_trajectory_paths=(
            (return_path,) if return_path is not None else ()
        ),
        ee_attach_policy="precomputed-required",
        ignore_done=True,
        use_camera_obs=False,
        has_offscreen_renderer=False,
    )


def _return_plan_and_context(
    runtime: Any,
    planner: Any,
    request: Any,
    *,
    template: Any | None = None,
) -> tuple[Any, Any]:
    contexts, registry = (
        planner.collision_context_factory.prepare_precomputed_ee_exchange(request)
    )
    plan = planner.precomputed_ee_exchange_planner.plan_return_only(
        request,
        collision_contexts=contexts,
        collision_checker=registry,
        template=template,
    )
    return plan, registry


def _execute_return_only(runtime: Any, request: Any, plan: Any, registry: Any) -> Any:
    from tuj.m5_motion.schema import (
        ArtifactProvenance,
        ExecutionStatus,
        ModuleName,
        SimulationConfig,
        SimulationRun,
    )
    from tuj.m5_motion.tool_use_journal_runtime import (
        ToolUseJournalControllerTrajectoryPlayer,
    )

    run = SimulationRun(
        run_id=f"{request.request_id}:return-only-run",
        provenance=ArtifactProvenance(
            artifact_id=f"{request.request_id}:return-only-run-artifact",
            artifact_type="SimulationRun",
            produced_by=ModuleName.SIMULATOR,
            invocation_id="precomputed-ee-return-controller-validation",
            input_artifact_ids=[plan.provenance.artifact_id],
        ),
        plan=plan,
        config=SimulationConfig(
            physics_timestep_s=float(runtime.env.model_timestep),
            control_timestep_s=float(runtime.env.control_timestep),
            realtime_factor=0.0,
            max_duration_s=plan.duration_s + 5.0,
            terminate_on_collision=True,
            render=False,
            random_seed=request.options.random_seed,
        ),
    )
    report = ToolUseJournalControllerTrajectoryPlayer(
        runtime,
        collision_probe=registry,
    ).execute(run)
    if report.status is not ExecutionStatus.SUCCESS:
        raise RuntimeError(
            f"return controller validation failed: {report.status.value}: "
            f"{report.failure}"
        )
    if runtime.active_ee is not None:
        raise RuntimeError(
            f"return controller ended with active EE {runtime.active_ee!r}"
        )
    if not runtime.transitions or runtime.transitions[-1].to_ee is not None:
        raise RuntimeError("runtime did not record the expected TOOL_UNLOCK transition")
    return report


def _execute_exchange(runtime: Any, planner: Any, request: Any, plan: Any) -> Any:
    from tuj.m5_motion.orchestration import SelectedPlanPlanningResult
    from tuj.m5_motion.tool_use_journal_execution import ToolUseJournalExecutionAdapter

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
            f"composed exchange failed: {execution.status.value}: {execution.detail}"
        )
    if runtime.active_ee != request.task.ee:
        raise RuntimeError(
            f"composed exchange ended with {runtime.active_ee!r}, "
            f"expected {request.task.ee!r}"
        )
    return execution


def _fresh_request(
    *,
    repository: Path,
    registry_root: Path,
    environment: str,
    source_ee: str,
    target_ee: str,
    attach: Any,
    seed: int,
    return_path: Path,
) -> tuple[Any, Any, Any]:
    from tuj.m5_motion.tool_use_journal import ToolUseJournalEnvironmentAdapter

    runtime = _make_runtime(repository, environment, source_ee, seed)
    _set_exchange_entry(
        runtime,
        list(attach.joint_names),
        list(attach.segments[-1].waypoints[-1].joint_positions_rad),
    )
    world = ToolUseJournalEnvironmentAdapter(runtime.env).world_snapshot()
    request = _request(
        world,
        source_ee=source_ee,
        target_ee=target_ee,
        seed=seed,
    )
    planner = _make_planner(
        runtime,
        repository,
        registry_root,
        seed=seed,
        return_path=return_path,
    )
    return runtime, planner, request


def _validate_saved(
    path: Path,
    *,
    repository: Path,
    registry_root: Path,
    environment: str,
    source_ee: str,
    attach: Any,
    seed: int,
    replay_count: int,
) -> dict[str, Any]:
    alternate_targets = [target for target in TARGETS if target != source_ee]
    return_replays: list[dict[str, Any]] = []
    for replay_index in range(replay_count):
        target = alternate_targets[replay_index % len(alternate_targets)]
        runtime, planner, request = _fresh_request(
            repository=repository,
            registry_root=registry_root,
            environment=environment,
            source_ee=source_ee,
            target_ee=target,
            attach=attach,
            seed=seed,
            return_path=path,
        )
        try:
            plan, registry = _return_plan_and_context(
                runtime, planner, request
            )
            report = _execute_return_only(runtime, request, plan, registry)
            return_replays.append(
                {
                    "replay_index": replay_index,
                    "status": report.status.value,
                    "dynamic_planner_invoked": plan.metadata[
                        "dynamic_planner_invoked"
                    ],
                }
            )
        finally:
            runtime.close()

    exchanges: list[dict[str, Any]] = []
    for target in alternate_targets:
        runtime, planner, request = _fresh_request(
            repository=repository,
            registry_root=registry_root,
            environment=environment,
            source_ee=source_ee,
            target_ee=target,
            attach=attach,
            seed=seed,
            return_path=path,
        )
        try:
            plan = planner(request)
            execution = _execute_exchange(runtime, planner, request, plan)
            exchanges.append(
                {
                    "transition": f"{source_ee}->{target}",
                    "status": execution.status.value,
                    "dynamic_planner_invoked": plan.metadata[
                        "dynamic_planner_invoked"
                    ],
                }
            )
        finally:
            runtime.close()
    return {"return_replays": return_replays, "composed_exchanges": exchanges}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_ee", choices=TARGETS)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--environment", default="C1_1_LegoSweep")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-count", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replay-existing", action="store_true")
    args = parser.parse_args()
    if args.replay_count <= 0:
        parser.error("--replay-count must be positive")
    return args


def main() -> int:
    _install_source_roots()
    args = _parse_args()
    repository = args.repository.expanduser().resolve()
    registry_root = repository / "configs" / "precomputed_ee_paths"
    environment_root = registry_root / args.environment
    attach_path = environment_root / f"bare_to_{args.source_ee}.json"
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else environment_root / f"{args.source_ee}_to_bare.json"
    )
    if not attach_path.is_file():
        raise SystemExit(f"validated attach trajectory does not exist: {attach_path}")
    if args.replay_existing and not output.is_file():
        raise SystemExit(f"return trajectory to replay does not exist: {output}")
    if output.exists() and not args.overwrite and not args.replay_existing:
        raise SystemExit(f"refusing to overwrite existing trajectory: {output}")

    from tuj.m5_motion.precomputed_ee_attach import EEAttachTrajectoryTemplate
    from tuj.m5_motion.precomputed_ee_exchange import (
        derive_return_template_from_attach,
        save_ee_return_template,
    )
    from tuj.m5_motion.tool_use_journal import ToolUseJournalEnvironmentAdapter

    _ensure_attach_start_eef_metadata(
        repository,
        args.environment,
        environment_root,
        args.seed,
    )

    attach = EEAttachTrajectoryTemplate.model_validate_json(
        attach_path.read_text(encoding="utf-8")
    )
    if args.replay_existing:
        result = _validate_saved(
            output,
            repository=repository,
            registry_root=registry_root,
            environment=args.environment,
            source_ee=args.source_ee,
            attach=attach,
            seed=args.seed,
            replay_count=args.replay_count,
        )
        print(
            json.dumps(
                {
                    "status": "REPLAY_VALIDATED",
                    "environment": args.environment,
                    "source_ee": args.source_ee,
                    "output": str(output),
                    **result,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    with tempfile.TemporaryDirectory(prefix="ee-return-commission-") as temporary:
        candidate_path = Path(temporary) / f"{args.source_ee}_to_bare.json"
        target = next(item for item in TARGETS if item != args.source_ee)
        runtime = _make_runtime(
            repository, args.environment, args.source_ee, args.seed
        )
        try:
            _set_exchange_entry(
                runtime,
                list(attach.joint_names),
                list(attach.segments[-1].waypoints[-1].joint_positions_rad),
            )
            world = ToolUseJournalEnvironmentAdapter(runtime.env).world_snapshot()
            request = _request(
                world,
                source_ee=args.source_ee,
                target_ee=target,
                seed=args.seed,
            )
            planner = _make_planner(
                runtime,
                repository,
                registry_root,
                seed=args.seed,
            )
            contexts, registry = (
                planner.collision_context_factory.prepare_precomputed_ee_exchange(
                    request
                )
            )
            candidate = derive_return_template_from_attach(
                attach,
                world,
                contexts,
                trajectory_id=f"ur5e-{args.source_ee}-to-bare-v1",
            )
            plan = planner.precomputed_ee_exchange_planner.plan_return_only(
                request,
                collision_contexts=contexts,
                collision_checker=registry,
                template=candidate,
            )
            report = _execute_return_only(runtime, request, plan, registry)
            candidate.metadata["requires_controller_validation"] = False
            candidate.metadata["commissioning_controller_status"] = report.status.value
            save_ee_return_template(candidate_path, candidate)
        finally:
            runtime.close()

        validation = _validate_saved(
            candidate_path,
            repository=repository,
            registry_root=registry_root,
            environment=args.environment,
            source_ee=args.source_ee,
            attach=attach,
            seed=args.seed,
            replay_count=args.replay_count,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.replace(output)

    print(
        json.dumps(
            {
                "status": "VALIDATED_AND_SAVED",
                "environment": args.environment,
                "source_ee": args.source_ee,
                "output": str(output),
                **validation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
