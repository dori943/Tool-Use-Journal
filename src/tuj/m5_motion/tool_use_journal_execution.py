"""Production execution bindings for Tool-Use-Journal workcells."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from tuj.m5_motion.execution import SelectedPlanSimulationOrchestrator
from tuj.m5_motion.mujoco_collision import MuJoCoCollisionModelRegistry
from tuj.m5_motion.orchestration import SelectedPlanPlanningResult
from tuj.m5_motion.schema import (
    CollisionContext,
    ExecutionReport,
    MotionPlan,
    MotionPlanRequest,
    SimulationConfig,
    WorldSnapshot,
)
from tuj.m5_motion.tool_use_journal import (
    ToolUseJournalCollisionModelCompiler,
    ToolUseJournalEnvironmentAdapter,
)
from tuj.m5_motion.tool_use_journal_planning import (
    attached_object_transform_from_state,
)
from tuj.m5_motion.tool_use_journal_runtime import (
    ToolUseJournalControllerTrajectoryPlayer,
    ToolUseJournalEERuntime,
    ToolUseJournalKinematicTrajectoryPlayer,
)


CollisionProbeSource = (
    Mapping[str, MuJoCoCollisionModelRegistry]
    | Callable[[MotionPlanRequest, MotionPlan, int], MuJoCoCollisionModelRegistry]
)


class ToolUseJournalExecutionAdapter:
    """Bind one live runtime to per-plan players, collision models, and snapshots.

    When an already-built collision registry is not supplied, the adapter
    rebuilds it from the exact event-scoped contexts embedded in each plan.  It
    never falls back to an unscoped workcell collision model.
    """

    def __init__(
        self,
        runtime: ToolUseJournalEERuntime,
        *,
        compiler: ToolUseJournalCollisionModelCompiler | None = None,
        collision_probes: CollisionProbeSource | None = None,
        controller: bool = True,
        realtime_factor: float = 0.0,
        render: bool = False,
        max_duration_padding_s: float = 5.0,
        terminate_on_collision: bool = True,
        random_seed: int = 0,
    ) -> None:
        if compiler is None and collision_probes is None:
            raise ValueError(
                "a collision compiler or explicit per-plan collision probes are required"
            )
        if realtime_factor < 0:
            raise ValueError("realtime_factor must be non-negative")
        if max_duration_padding_s <= 0:
            raise ValueError("max_duration_padding_s must be positive")
        self.runtime = runtime
        self.compiler = compiler
        self._collision_probes = collision_probes
        self.controller = controller
        self.realtime_factor = realtime_factor
        self.render = render
        self.max_duration_padding_s = max_duration_padding_s
        self.terminate_on_collision = terminate_on_collision
        self.random_seed = random_seed
        self._compiled_probes: dict[str, MuJoCoCollisionModelRegistry] = {}

    @classmethod
    def from_repository(
        cls,
        runtime: ToolUseJournalEERuntime,
        repository_root: str | Path,
        *,
        seed: int = 0,
        controller: bool = True,
        **kwargs: Any,
    ) -> "ToolUseJournalExecutionAdapter":
        compiler = ToolUseJournalCollisionModelCompiler.from_repository(
            runtime.env,
            repository_root,
            seed=seed,
        )
        return cls(
            runtime,
            compiler=compiler,
            controller=controller,
            random_seed=seed,
            **kwargs,
        )

    @staticmethod
    def _contexts(plan: MotionPlan) -> dict[str, CollisionContext]:
        contexts: dict[str, CollisionContext] = {}
        for segment in plan.segments:
            for context in (
                segment.collision_context_before,
                segment.collision_context_after,
            ):
                if context is None:
                    continue
                previous = contexts.get(context.context_id)
                if previous is not None and previous != context:
                    raise ValueError(
                        f"plan reuses collision context {context.context_id!r} "
                        "with different definitions"
                    )
                contexts[context.context_id] = context
        if not contexts:
            raise ValueError("MotionPlan has no event-scoped collision contexts")
        return contexts

    def collision_probe(
        self, request: MotionPlanRequest, plan: MotionPlan, index: int
    ) -> MuJoCoCollisionModelRegistry:
        if callable(self._collision_probes):
            probe = self._collision_probes(request, plan, index)
            if not isinstance(probe, MuJoCoCollisionModelRegistry):
                raise TypeError("collision probe provider returned the wrong type")
            return probe
        if self._collision_probes is not None:
            try:
                return self._collision_probes[plan.plan_id]
            except KeyError as error:
                raise KeyError(
                    f"no collision probe registered for plan {plan.plan_id!r}"
                ) from error
        cached = self._compiled_probes.get(plan.plan_id)
        if cached is not None:
            return cached
        if self.compiler is None:
            raise RuntimeError("collision compiler is unavailable")
        contexts = self._contexts(plan)
        initial = plan.segments[0].collision_context_before
        probe = self.compiler.build_collision_registry(
            contexts,
            collision_margin_m=request.constraints.collision_margin_m,
            allowed_collision_pairs=request.constraints.allowed_collision_pairs,
            default_active_ee=(initial.active_ee if initial is not None else None),
        )
        self._compiled_probes[plan.plan_id] = probe
        return probe

    def player(
        self, request: MotionPlanRequest, plan: MotionPlan, index: int
    ) -> ToolUseJournalKinematicTrajectoryPlayer:
        probe = self.collision_probe(request, plan, index)
        player_type = (
            ToolUseJournalControllerTrajectoryPlayer
            if self.controller
            else ToolUseJournalKinematicTrajectoryPlayer
        )
        return player_type(self.runtime, collision_probe=probe)

    def config(
        self, request: MotionPlanRequest, plan: MotionPlan, index: int
    ) -> SimulationConfig:
        del request, index
        env = self.runtime.env
        return SimulationConfig(
            physics_timestep_s=float(env.model_timestep),
            control_timestep_s=float(env.control_timestep),
            realtime_factor=self.realtime_factor,
            max_duration_s=float(plan.duration_s) + self.max_duration_padding_s,
            terminate_on_collision=self.terminate_on_collision,
            render=self.render,
            random_seed=self.random_seed,
        )

    def world_snapshot(
        self, request: MotionPlanRequest, report: ExecutionReport
    ) -> WorldSnapshot:
        completed = list(request.world.scene.completed_subgoals)
        transform = (
            attached_object_transform_from_state(self.runtime.attachment)
            if self.runtime.attachment is not None
            else None
        )
        world = ToolUseJournalEnvironmentAdapter(self.runtime.env).world_snapshot(
            completed_subgoals=completed,
            facts=request.world.scene.facts,
            attached_object_id=self.runtime.attached_object_id,
            attached_object_transform=transform,
        )
        if report.final_robot_state is not None:
            world.robot_state = report.final_robot_state.model_copy(deep=True)
        return world

    def orchestrator(self, **kwargs: Any) -> SelectedPlanSimulationOrchestrator:
        if "goal_evaluator" not in kwargs:
            from tuj.m5_motion.contact_evaluation import TaskAwareGoalEvaluator

            kwargs["goal_evaluator"] = TaskAwareGoalEvaluator()
        return SelectedPlanSimulationOrchestrator(
            self.player,
            config=self.config,
            world_snapshot_provider=self.world_snapshot,
            **kwargs,
        )

    def execute(
        self,
        planning: SelectedPlanPlanningResult,
        **orchestrator_kwargs: Any,
    ):
        return self.orchestrator(**orchestrator_kwargs).execute(planning)


__all__ = ["CollisionProbeSource", "ToolUseJournalExecutionAdapter"]
