"""Production execution bindings for Tool-Use-Journal workcells."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from tuj.m5_motion.execution import SelectedPlanSimulationOrchestrator
from tuj.m5_motion.mujoco_collision import MuJoCoCollisionModelRegistry
from tuj.m5_motion.orchestration import SelectedPlanPlanningResult
from tuj.m5_motion.physical_grasp import (
    PhysicalGraspControllerTrajectoryPlayer,
    PhysicalGraspMonitor,
    uses_contact_friction,
)
from tuj.m5_motion.profiles import PhysicalGraspProfile
from tuj.m5_motion.schema import (
    AttachedObjectTransform,
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


_CONTACT_FRICTION_HELD_METADATA_KEY = "contact_friction_held_objects"


def _canonical_attached_object_transform(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the planning contract while discarding diagnostic-only fields."""

    payload = {
        name: value[name]
        for name in AttachedObjectTransform.model_fields
        if name in value
    }
    return AttachedObjectTransform.model_validate(payload).model_dump(mode="json")


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
        if self.controller and uses_contact_friction(request):
            capabilities = {
                str(value).strip().lower()
                for value in request.task.metadata.get("ee_capabilities", [])
                if isinstance(value, str)
            }
            if "opposed_finger_contact" not in capabilities:
                raise ValueError(
                    "CONTACT_FRICTION requires opposed_finger_contact capability"
                )
            target = (
                request.task.goal.target_object_id
                or request.task.tool
                or next(iter(request.task.target_ids), None)
            )
            if target is None:
                raise ValueError("CONTACT_FRICTION PICK has no target object")
            raw_profile = request.task.metadata.get("grasp_profile")
            profile = PhysicalGraspProfile.from_mapping(
                raw_profile if isinstance(raw_profile, Mapping) else None
            )
            monitor = PhysicalGraspMonitor.from_runtime(
                self.runtime, str(target), profile
            )
            raw_preshape = request.task.metadata.get(
                "grasp_preshape_aperture_m"
            )
            preshape = (
                float(raw_preshape)
                if isinstance(raw_preshape, (int, float))
                else None
            )
            raw_preshape_tolerance = request.task.metadata.get(
                "grasp_preshape_tolerance_m"
            )
            preshape_tolerance = (
                float(raw_preshape_tolerance)
                if isinstance(raw_preshape_tolerance, (int, float))
                else None
            )
            return PhysicalGraspControllerTrajectoryPlayer(
                self.runtime,
                collision_probe=probe,
                monitor=monitor,
                preshape_aperture_m=preshape,
                preshape_tolerance_m=preshape_tolerance,
            )
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
        held: dict[str, dict[str, Any]] = {}
        held_object_id = world.robot_state.held_tool_id
        previous_holds = request.world.metadata.get(
            _CONTACT_FRICTION_HELD_METADATA_KEY, {}
        )
        if held_object_id is not None and isinstance(previous_holds, Mapping):
            previous_transform = previous_holds.get(held_object_id)
            if isinstance(previous_transform, Mapping):
                held[held_object_id] = _canonical_attached_object_transform(
                    previous_transform
                )
        retention = getattr(
            self.runtime, "_contact_friction_retention", None
        )
        if (
            held_object_id is not None
            and retention is not None
            and getattr(retention, "object_id", None) == held_object_id
        ):
            held[held_object_id] = retention.transform().model_dump(mode="json")
        transform = report.metadata.get("physical_grasp_transform")
        if (
            report.metadata.get("physical_grasp_execution_succeeded") is True
            and isinstance(transform, Mapping)
        ):
            canonical = _canonical_attached_object_transform(transform)
            object_id = str(canonical.get("object_id") or request.task.tool)
            if world.robot_state.held_tool_id == object_id:
                held[object_id] = canonical
        world.metadata[_CONTACT_FRICTION_HELD_METADATA_KEY] = held
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
