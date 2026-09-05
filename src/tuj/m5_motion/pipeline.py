"""End-to-end keyframe proposal -> validated MotionPlan orchestration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Mapping, Protocol

from tuj.m5_motion.compiler import (
    FirstFeasibleStrategyCompiler,
    StrategyCompilationResult,
)
from tuj.m5_motion.kinematics import UR5eKinematics
from tuj.m5_motion.plan_builder import FinalSegmentValidator, MotionPlanBuilder
from tuj.m5_motion.path_planning import (
    CartesianEdgePlanner,
    PlannerDispatchEdgePlanner,
    RRTConnectEdgePlanner,
)
from tuj.m5_motion.safety import KinematicSafetyValidator
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    CollisionContext,
    KeyframePlanArtifact,
    ModuleName,
    MotionPlan,
    MotionPlanRequest,
)
from tuj.m5_motion.strategy import (
    EdgePlanner,
    FirstFeasibleBranchSelector,
    InterpolatingEdgePlanner,
    StateValidator,
)


class KeyframeStrategyProvider(Protocol):
    def generate(self, request: MotionPlanRequest) -> KeyframePlanArtifact: ...


@dataclass(frozen=True, slots=True)
class CollisionPlanningSetup:
    """Request-scoped collision state prepared after keyframe generation.

    A factory may deterministically annotate generated keyframes with context
    transitions (for example GRASP -> attached object) before robot validation.
    The returned validator and final checker must interpret the exact same
    ``collision_contexts`` mapping.
    """

    keyframe_artifact: KeyframePlanArtifact
    state_validator: StateValidator
    collision_contexts: Mapping[str, CollisionContext]
    initial_collision_context_id: str
    final_segment_validator: FinalSegmentValidator


class CollisionContextFactory(Protocol):
    def prepare(
        self,
        request: MotionPlanRequest,
        artifact: KeyframePlanArtifact,
    ) -> CollisionPlanningSetup: ...


class MotionPlanningPipelineError(RuntimeError):
    """No schema-valid, connected MotionPlan could be finalized."""

    def __init__(
        self,
        message: str,
        *,
        compilation: StrategyCompilationResult | None = None,
    ) -> None:
        super().__init__(message)
        self.compilation = compilation


@dataclass(frozen=True, slots=True)
class MotionPlanningResult:
    keyframe_artifact: KeyframePlanArtifact
    compilation: StrategyCompilationResult
    plan: MotionPlan


class MotionPlanningPipeline:
    """Connect a proposal provider to deterministic robot-model validation.

    The provider may be stochastic.  Everything after the frozen
    ``KeyframePlanArtifact`` is deterministic for fixed seeds and scene models.
    """

    def __init__(
        self,
        provider: KeyframeStrategyProvider,
        kinematics: UR5eKinematics,
        *,
        plan_builder: MotionPlanBuilder | None = None,
    ) -> None:
        self._provider = provider
        self._kinematics = kinematics
        forward_pose = getattr(kinematics, "forward_pose_world", None)
        self._builder = plan_builder or MotionPlanBuilder(
            forward_pose=forward_pose if callable(forward_pose) else None
        )

    @staticmethod
    def _validate_artifact(
        request: MotionPlanRequest, artifact: KeyframePlanArtifact
    ) -> None:
        if artifact.scene_signature != request.world.scene.signature:
            raise MotionPlanningPipelineError(
                "keyframe artifact scene_signature does not match the request"
            )
        if artifact.subgoal_id != request.task.subgoal_id:
            raise MotionPlanningPipelineError(
                "keyframe artifact subgoal_id does not match the request"
            )

    @staticmethod
    def _plan_identity(
        request: MotionPlanRequest, artifact: KeyframePlanArtifact
    ) -> tuple[str, ArtifactProvenance]:
        digest = hashlib.sha256(
            (
                f"{request.request_id}|{request.world.scene.signature}|"
                f"{artifact.artifact_id}"
            ).encode("utf-8")
        ).hexdigest()[:24]
        plan_id = f"motion-plan:{digest}"
        return plan_id, ArtifactProvenance(
            artifact_id=f"motion-plan-artifact:{digest}",
            artifact_type="MotionPlan",
            produced_by=ModuleName.MOTION_PLANNER,
            invocation_id=f"motion-planning:{request.request_id}",
            input_artifact_ids=[
                request.provenance.artifact_id,
                artifact.provenance.artifact_id,
            ],
            metadata={"keyframe_artifact_id": artifact.artifact_id},
        )

    def plan(
        self,
        request: MotionPlanRequest,
        *,
        state_validator: StateValidator | None = None,
        collision_contexts: Mapping[str, CollisionContext] | None = None,
        initial_collision_context_id: str | None = None,
        final_segment_validator: FinalSegmentValidator | None = None,
        collision_context_factory: CollisionContextFactory | None = None,
        edge_planner: EdgePlanner | None = None,
    ) -> MotionPlanningResult:
        """Generate candidates, choose a connected branch, and finalize timing."""

        update_reference = getattr(self._kinematics, "set_reference_qpos", None)
        if callable(update_reference):
            update_reference(request.world.robot_state.joint_positions_rad)
        artifact = self._provider.generate(request)
        self._validate_artifact(request, artifact)
        explicit_collision_arguments = (
            state_validator,
            collision_contexts,
            initial_collision_context_id,
            final_segment_validator,
        )
        if collision_context_factory is not None:
            if any(value is not None for value in explicit_collision_arguments):
                raise ValueError(
                    "collision_context_factory cannot be combined with explicit "
                    "collision planning arguments"
                )
            setup = collision_context_factory.prepare(request, artifact)
            artifact = setup.keyframe_artifact
            state_validator = setup.state_validator
            collision_contexts = setup.collision_contexts
            initial_collision_context_id = setup.initial_collision_context_id
            final_segment_validator = setup.final_segment_validator
            self._validate_artifact(request, artifact)
        else:
            missing = [
                name
                for name, value in (
                    ("state_validator", state_validator),
                    ("collision_contexts", collision_contexts),
                    (
                        "initial_collision_context_id",
                        initial_collision_context_id,
                    ),
                    ("final_segment_validator", final_segment_validator),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "missing collision planning arguments: " + ", ".join(missing)
                )
        # Narrow the optional public API after the mutually exclusive setup
        # paths above.  These guards also make malformed third-party factories
        # fail at the pipeline boundary rather than midway through planning.
        if state_validator is None or final_segment_validator is None:
            raise ValueError("collision setup has no validators")
        if collision_contexts is None or initial_collision_context_id is None:
            raise ValueError("collision setup has no contexts")
        if initial_collision_context_id not in collision_contexts:
            raise ValueError("initial collision context is not registered")
        effective_state_validator: StateValidator = state_validator
        if hasattr(self._kinematics, "jacobian_singular_values"):
            effective_state_validator = KinematicSafetyValidator(
                state_validator,
                self._kinematics,
                min_singular_value=(
                    request.constraints.min_jacobian_singular_value
                ),
                max_condition_number=(
                    request.constraints.max_jacobian_condition_number
                ),
            )
        max_strategy_keyframes = max(
            len(candidate.keyframes) for candidate in artifact.candidates
        )
        # Branch selection owns the whole layered keyframe graph, while an RRT
        # timeout applies to one edge.  A fixed 10 s / 256-edge selector budget
        # incorrectly rejects longer task-geometry strategies even when every
        # pose has valid IK and individual edges remain inside their request
        # budgets.  Scale the outer graph budget with the frozen strategy size,
        # but retain finite caps so malformed candidate sets still fail closed.
        branch_selector = FirstFeasibleBranchSelector(
            max_edge_evaluations=max(
                256,
                min(
                    4096,
                    max_strategy_keyframes
                    * request.options.max_attempts
                    * 8,
                ),
            ),
            timeout_s=max(
                10.0,
                min(
                    300.0,
                    request.options.allowed_planning_time_s
                    * min(max_strategy_keyframes, 12),
                ),
            ),
        )
        compiler = FirstFeasibleStrategyCompiler(
            self._kinematics,
            position_tolerance_m=request.constraints.position_tolerance_m,
            orientation_tolerance_rad=request.constraints.orientation_tolerance_rad,
            branch_selector=branch_selector,
        )
        joint_planner = InterpolatingEdgePlanner(
            state_validator=effective_state_validator,
            max_joint_step_rad=request.constraints.max_joint_path_step_rad,
            wrap_joints=False,
        )
        joint_limits = getattr(self._kinematics, "joint_limits_rad", None)
        if joint_limits is None:
            joint_limits = tuple(
                (-2.0 * 3.141592653589793, 2.0 * 3.141592653589793)
                for _ in request.world.robot_state.joint_names
            )
        selected_edge_planner = edge_planner or PlannerDispatchEdgePlanner(
            joint=joint_planner,
            cartesian=CartesianEdgePlanner(
                self._kinematics,
                request.world,
                effective_state_validator,
                translation_step_m=(
                    request.options.cartesian_translation_step_m
                ),
                rotation_step_rad=request.options.cartesian_rotation_step_rad,
                max_joint_step_rad=(
                    request.constraints.max_joint_path_step_rad
                ),
                wrap_joints=False,
            ),
            sampling_based=RRTConnectEdgePlanner(
                effective_state_validator,
                joint_limits,
                random_seed=request.options.random_seed,
                max_iterations=request.options.rrt_max_iterations,
                timeout_s=request.options.allowed_planning_time_s,
                extension_step_rad=request.options.rrt_extension_step_rad,
                validation_step_rad=(
                    request.constraints.max_joint_path_step_rad
                ),
                goal_bias=request.options.rrt_goal_bias,
                wrap_joints=False,
            ),
        )
        compilation = compiler.compile(
            request.world,
            artifact.candidates,
            start_joint_config=request.world.robot_state.joint_positions_rad,
            state_validator=effective_state_validator,
            edge_planner=selected_edge_planner,
        )
        if compilation.connected is None:
            failures = "; ".join(
                (
                    f"{attempt.strategy_id}="
                    f"{attempt.failure_code or 'UNKNOWN'}"
                    + (f" ({attempt.detail})" if attempt.detail else "")
                )
                for attempt in compilation.attempts
            )
            raise MotionPlanningPipelineError(
                f"no generated keyframe strategy produced a connected path: {failures}",
                compilation=compilation,
            )

        plan_id, provenance = self._plan_identity(request, artifact)
        try:
            def final_validator(
                waypoints: tuple,
                context: CollisionContext,
            ) -> bool:
                if not final_segment_validator(waypoints, context):
                    source_owner = getattr(
                        final_segment_validator, "__self__", None
                    )
                    setattr(
                        final_validator,
                        "last_path_collision_check",
                        getattr(
                            source_owner,
                            "last_path_collision_check",
                            None,
                        ),
                    )
                    return False
                if not isinstance(
                    effective_state_validator, KinematicSafetyValidator
                ):
                    return True
                for waypoint_index, waypoint in enumerate(waypoints):
                    safety_report = effective_state_validator.check(
                        waypoint.joint_positions_rad,
                        context=context,
                    )
                    if not safety_report.valid:
                        safety_report = replace(
                            safety_report,
                            detail=(
                                f"waypoint {waypoint_index}: "
                                f"{safety_report.detail}"
                            ),
                        )
                        setattr(
                            final_validator,
                            "last_path_collision_check",
                            safety_report,
                        )
                        return False
                return True

            plan = self._builder.build(
                request,
                compilation.connected,
                plan_id=plan_id,
                provenance=provenance,
                collision_contexts=collision_contexts,
                initial_collision_context_id=initial_collision_context_id,
                final_segment_validator=final_validator,
                joint_position_limits_rad=joint_limits,
            )
        except Exception as error:
            raise MotionPlanningPipelineError(
                f"connected path could not be finalized ({type(error).__name__})",
                compilation=compilation,
            ) from error
        return MotionPlanningResult(
            keyframe_artifact=artifact,
            compilation=compilation,
            plan=plan,
        )


__all__ = [
    "CollisionContextFactory",
    "CollisionPlanningSetup",
    "KeyframeStrategyProvider",
    "MotionPlanningPipeline",
    "MotionPlanningPipelineError",
    "MotionPlanningResult",
]
