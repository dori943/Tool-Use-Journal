"""End-to-end keyframe proposal -> validated MotionPlan orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Protocol

from tuj.m5_motion.attachment_retarget import retarget_resolved_pose
from tuj.m5_motion.compiler import (
    FirstFeasibleStrategyCompiler,
    StrategyAttempt,
    StrategyCompilationResult,
)
from tuj.m5_motion.geometry import RelativePoseResolver
from tuj.m5_motion.kinematics import UR5eKinematics
from tuj.m5_motion.plan_builder import (
    FinalSegmentValidator,
    MotionPlanBuildError,
    MotionPlanBuilder,
)
from tuj.m5_motion.path_planning import (
    CartesianEdgePlanner,
    PlannerDispatchEdgePlanner,
    RRTConnectEdgePlanner,
)
from tuj.m5_motion.phase_contract import (
    KeyframePhaseContractError,
    validate_keyframe_phase_contract,
)
from tuj.m5_motion.safety import KinematicSafetyValidator
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    CollisionContext,
    KeyframeEventType,
    KeyframePlanArtifact,
    ModuleName,
    MotionPlan,
    MotionPlanRequest,
    Pose,
    RelativeKeyframeSpec,
)
from tuj.m5_motion.strategy import (
    EdgePlanner,
    FirstFeasibleBranchSelector,
    InterpolatingEdgePlanner,
    StateValidator,
)
from tuj.m5_motion.trajectory_processing import TrajectoryProcessingError


class KeyframeStrategyProvider(Protocol):
    def generate(self, request: MotionPlanRequest) -> KeyframePlanArtifact: ...


EdgeContextMaterializer = Callable[[RelativeKeyframeSpec, Pose], None]


def _debug_name(request: MotionPlanRequest) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", request.task.subgoal_id).strip("._")
    digest = hashlib.sha256(request.request_id.encode("utf-8")).hexdigest()[:12]
    return f"{label or 'request'}-{digest}"


def _resolved_debug_keyframes(
    request: MotionPlanRequest,
    artifact: KeyframePlanArtifact,
    *,
    retarget: bool,
) -> list[dict[str, object]]:
    resolver = RelativePoseResolver(request.world)
    strategies: list[dict[str, object]] = []
    for strategy in artifact.candidates:
        keyframes: list[dict[str, object]] = []
        for index, keyframe in enumerate(strategy.keyframes):
            entry: dict[str, object] = {
                "index": index,
                "keyframe_id": keyframe.keyframe_id,
                "phase": keyframe.keyframe_type.value,
                "frame_ref": keyframe.frame_ref,
                "anchor": keyframe.anchor,
                "raw_offset_m": keyframe.offset_along_approach_m,
                "approach_axis_xyz": list(keyframe.approach_axis_xyz),
                "tool_axis_to_align": keyframe.tool_axis_to_align,
                "roll_rad": keyframe.roll_rad,
                "planner": keyframe.planner.value,
                "metadata": keyframe.metadata,
            }
            try:
                pose = resolver.resolve(keyframe)
                entry["resolver_world_pose"] = pose.model_dump(mode="json")
                entry["raw_position_world_m"] = list(pose.position_m)
                if retarget:
                    final_pose = retarget_resolved_pose(request.world, keyframe, pose)
                    entry["final_tcp_world_pose"] = final_pose.model_dump(mode="json")
            except Exception as error:  # diagnostics must not affect planning
                entry["resolution_error"] = f"{type(error).__name__}: {error}"
            keyframes.append(entry)
        strategies.append({"strategy_id": strategy.strategy_id, "keyframes": keyframes})
    return strategies


def _write_debug_json(path: Path, payload: object) -> None:
    """Best-effort atomic diagnostics; failures never change planning behavior."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        pass


class DebugKeyframeStrategyProvider:
    """Persist the generated artifact before physical/collision binders mutate it."""

    def __init__(self, provider: KeyframeStrategyProvider, directory: str | Path) -> None:
        self._provider = provider
        self._directory = Path(directory)
        self.supports_collision_feedback = bool(
            getattr(provider, "supports_collision_feedback", False)
        )

    def generate(self, request: MotionPlanRequest) -> KeyframePlanArtifact:
        artifact = self._provider.generate(request)
        _write_debug_json(
            self._directory / f"{_debug_name(request)}-generated.json",
            {
                "stage": "VLM_GENERATED",
                "request_id": request.request_id,
                "subgoal_id": request.task.subgoal_id,
                "artifact": artifact.model_dump(mode="json"),
                "strategies": _resolved_debug_keyframes(request, artifact, retarget=False),
                "raw_position_note": (
                    "VLM emits frame/anchor/offset, not absolute XYZ; "
                    "raw_position_world_m is the first resolver result."
                ),
            },
        )
        return artifact


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
    edge_context_materializer: EdgeContextMaterializer | None = None


@dataclass(slots=True)
class _EdgeContextMaterializingPlanner:
    """Refresh branch-dependent collision state before an outgoing edge."""

    delegate: EdgePlanner
    kinematics: UR5eKinematics
    materialize: EdgeContextMaterializer

    def plan(
        self,
        source: tuple[float, ...],
        target: tuple[float, ...],
        source_keyframe: RelativeKeyframeSpec | None,
        target_keyframe: RelativeKeyframeSpec,
    ):
        if (
            source_keyframe is not None
            and KeyframeEventType.ATTACH_OBJECT in source_keyframe.events_after
        ):
            position, orientation = self.kinematics.forward_pose_world(source)
            self.materialize(
                source_keyframe,
                Pose(
                    frame_id="world",
                    position_m=position,
                    orientation_xyzw=orientation,
                ),
            )
        return self.delegate.plan(
            source,
            target,
            source_keyframe,
            target_keyframe,
        )


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


_COLLISION_REPAIR_FEEDBACK_KEY = "collision_repair_feedback"
_COLLISION_REPAIR_CONTRACT = "COLLISION_REPAIR_V1"
_MAX_COLLISION_REPAIR_BATCHES = 2
_MAX_COLLISION_STRATEGIES_PER_BATCH = 4
_MAX_COLLISION_HISTORY_STRATEGIES = 8
_COLLISION_OBSERVATION = re.compile(
    r"COLLISION_MARGIN_VIOLATION:\s*"
    r"(?P<geometry_a>.+?)\s*<->\s*(?P<geometry_b>.+?)\s*"
    r"clearance\s+(?P<clearance>[-+0-9.eE]+)\s*m\s*"
    r"is below required\s+(?P<required>[-+0-9.eE]+)\s*m"
)


def _safe_collision_label(value: object, *, limit: int = 160) -> str:
    """Keep validator identifiers as inert labels in the repair payload."""

    rendered = str(value)[:limit]
    if re.fullmatch(r"[A-Za-z0-9_.:+/\\-]+", rendered):
        return rendered
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]
    return f"label_{digest}"


def _collision_repair_attempt(request: MotionPlanRequest) -> int:
    raw = request.task.metadata.get(_COLLISION_REPAIR_FEEDBACK_KEY)
    if not isinstance(raw, Mapping):
        return 0
    try:
        attempt = int(raw.get("repair_attempt", 0))
    except (TypeError, ValueError):
        return 0
    return max(0, attempt)


def _provider_supports_collision_feedback(provider: object) -> bool:
    """Follow the small provider-decorator chain used by the production runner."""

    pending = [provider]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if bool(getattr(current, "supports_collision_feedback", False)):
            return True
        for attribute in ("_provider", "_default", "provider", "default_provider"):
            nested = getattr(current, attribute, None)
            if nested is not None:
                pending.append(nested)
    return False


def _is_collision_terminal(attempt: StrategyAttempt) -> bool:
    return bool(
        attempt.failure_code == "COLLISION_FILTERED_ALL"
        or (
            attempt.failure_code == "FINAL_VALIDATION_FAILED"
            and "COLLISION" in attempt.detail.upper()
        )
    )


def _structured_collision_observations(
    attempt: StrategyAttempt,
) -> list[dict[str, object]]:
    """Extract only bounded numeric observations owned by the validator."""

    if not _is_collision_terminal(attempt):
        return []
    details = [attempt.detail]
    details.extend(
        diagnostic.validity_detail
        for diagnostic in attempt.ik_diagnostics[:12]
    )
    observations: list[dict[str, object]] = []
    observed: set[tuple[str, str, float, float]] = set()
    for detail in details:
        for match in _COLLISION_OBSERVATION.finditer(detail):
            try:
                clearance = float(match.group("clearance"))
                required = float(match.group("required"))
            except ValueError:
                continue
            if (
                not math.isfinite(clearance)
                or not math.isfinite(required)
                or required < 0.0
            ):
                continue
            item = (
                _safe_collision_label(match.group("geometry_a")),
                _safe_collision_label(match.group("geometry_b")),
                clearance,
                required,
            )
            if item in observed:
                continue
            observed.add(item)
            observations.append(
                {
                    "geometry_a": item[0],
                    "geometry_b": item[1],
                    "measured_clearance_m": item[2],
                    "required_clearance_m": item[3],
                }
            )
            if len(observations) >= 8:
                return observations
    return observations


def _collision_repair_eligible(compilation: StrategyCompilationResult) -> bool:
    """Retry when at least one terminal candidate has structured collision data."""

    return any(
        _structured_collision_observations(attempt)
        for attempt in compilation.attempts
    )


def _bounded_prior_collision_strategy(
    source: Mapping[object, object],
    *,
    current_repair_attempt: int,
) -> dict[str, object] | None:
    """Revalidate one earlier internal record before cumulative reuse."""

    try:
        source_attempt = int(source.get("source_repair_attempt", -1))
    except (TypeError, ValueError):
        return None
    if source_attempt < 0 or source_attempt >= current_repair_attempt:
        return None
    diagnostics: list[dict[str, object]] = []
    raw_diagnostics = source.get("ik_diagnostics", [])
    if isinstance(raw_diagnostics, list):
        for diagnostic in raw_diagnostics[:12]:
            if not isinstance(diagnostic, Mapping):
                continue
            try:
                raw_count = int(diagnostic.get("raw_ik_branch_count", -1))
                valid_count = int(diagnostic.get("valid_ik_branch_count", -1))
            except (TypeError, ValueError):
                continue
            if raw_count < 0 or valid_count < 0 or valid_count > raw_count:
                continue
            diagnostics.append(
                {
                    "keyframe_id": _safe_collision_label(
                        diagnostic.get("keyframe_id", "unknown")
                    ),
                    "raw_ik_branch_count": raw_count,
                    "valid_ik_branch_count": valid_count,
                }
            )
    observations: list[dict[str, object]] = []
    raw_observations = source.get("collision_observations", [])
    if isinstance(raw_observations, list):
        for observation in raw_observations[:8]:
            if not isinstance(observation, Mapping):
                continue
            try:
                clearance = float(observation.get("measured_clearance_m"))
                required = float(observation.get("required_clearance_m"))
            except (TypeError, ValueError):
                continue
            if (
                not math.isfinite(clearance)
                or not math.isfinite(required)
                or required < 0.0
            ):
                continue
            observations.append(
                {
                    "geometry_a": _safe_collision_label(
                        observation.get("geometry_a", "unknown")
                    ),
                    "geometry_b": _safe_collision_label(
                        observation.get("geometry_b", "unknown")
                    ),
                    "measured_clearance_m": clearance,
                    "required_clearance_m": required,
                }
            )
    if not observations:
        return None
    return {
        "source_repair_attempt": source_attempt,
        "strategy_id": _safe_collision_label(
            source.get("strategy_id", "unknown")
        ),
        "failure_code": _safe_collision_label(
            source.get("failure_code", "unknown")
        ),
        "ik_diagnostics": diagnostics,
        "collision_observations": observations,
    }


def _collision_repair_feedback(
    request: MotionPlanRequest,
    compilation: StrategyCompilationResult,
) -> dict[str, object]:
    """Build bounded, validator-owned data for one new proposal batch.

    Free-form diagnostic text is deliberately not returned to the model. Only
    parsed collision measurements and fixed-shape IK counts cross the boundary.
    """

    source_repair_attempt = _collision_repair_attempt(request)
    strategies: list[dict[str, object]] = []
    previous_feedback = request.task.metadata.get(_COLLISION_REPAIR_FEEDBACK_KEY)
    if (
        isinstance(previous_feedback, Mapping)
        and previous_feedback.get("contract_version")
        == _COLLISION_REPAIR_CONTRACT
    ):
        previous_strategies = previous_feedback.get("failed_strategies")
        if isinstance(previous_strategies, list):
            for source in previous_strategies[
                :_MAX_COLLISION_STRATEGIES_PER_BATCH
            ]:
                if not isinstance(source, Mapping):
                    continue
                prior = _bounded_prior_collision_strategy(
                    source,
                    current_repair_attempt=source_repair_attempt,
                )
                if prior is not None:
                    strategies.append(prior)
    collision_attempts = [
        attempt
        for attempt in compilation.attempts
        if _structured_collision_observations(attempt)
    ][:_MAX_COLLISION_STRATEGIES_PER_BATCH]
    for attempt in collision_attempts:
        diagnostics: list[dict[str, object]] = []
        for diagnostic in attempt.ik_diagnostics[:12]:
            if diagnostic.valid_ik_count >= diagnostic.raw_ik_count:
                continue
            diagnostics.append(
                {
                    "keyframe_id": _safe_collision_label(diagnostic.keyframe_id),
                    "raw_ik_branch_count": int(diagnostic.raw_ik_count),
                    "valid_ik_branch_count": int(diagnostic.valid_ik_count),
                }
            )
        strategies.append(
            {
                "source_repair_attempt": source_repair_attempt,
                "strategy_id": _safe_collision_label(attempt.strategy_id),
                "failure_code": str(attempt.failure_code),
                "ik_diagnostics": diagnostics,
                "collision_observations": _structured_collision_observations(
                    attempt
                ),
            }
        )
    return {
        "contract_version": _COLLISION_REPAIR_CONTRACT,
        "repair_attempt": _collision_repair_attempt(request) + 1,
        "maximum_repair_attempts": _MAX_COLLISION_REPAIR_BATCHES,
        "required_collision_margin_m": float(
            request.constraints.collision_margin_m
        ),
        "failed_strategies": strategies[:_MAX_COLLISION_HISTORY_STRATEGIES],
    }


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
        debug_dir: str | Path | None = None,
    ) -> None:
        self._provider = provider
        self._kinematics = kinematics
        forward_pose = getattr(kinematics, "forward_pose_world", None)
        self._builder = plan_builder or MotionPlanBuilder(
            forward_pose=forward_pose if callable(forward_pose) else None
        )
        self._debug_dir = Path(debug_dir) if debug_dir is not None else None

    def _dump_planning_debug(
        self,
        request: MotionPlanRequest,
        binder_artifact: KeyframePlanArtifact,
        final_artifact: KeyframePlanArtifact,
        compilation: StrategyCompilationResult,
    ) -> None:
        if self._debug_dir is None:
            return
        final_by_id = {
            keyframe.keyframe_id: keyframe
            for strategy in final_artifact.candidates
            for keyframe in strategy.keyframes
        }
        final_location_by_id = {
            keyframe.keyframe_id: (index, keyframe.keyframe_type.value)
            for strategy in final_artifact.candidates
            for index, keyframe in enumerate(strategy.keyframes)
        }
        attempts: list[dict[str, object]] = []
        for attempt in compilation.attempts:
            failed_keyframe_id = (
                attempt.ik_diagnostics[-1].keyframe_id
                if attempt.failure_code and attempt.ik_diagnostics
                else None
            )
            attempts.append(
                {
                    "strategy_id": attempt.strategy_id,
                    "failure_code": attempt.failure_code,
                    "failure_detail": attempt.detail,
                    "failed_keyframe_id": failed_keyframe_id,
                    "failed_keyframe_index": (
                        final_location_by_id[failed_keyframe_id][0]
                        if failed_keyframe_id in final_location_by_id
                        else None
                    ),
                    "failed_keyframe_phase": (
                        final_location_by_id[failed_keyframe_id][1]
                        if failed_keyframe_id in final_location_by_id
                        else None
                    ),
                    "failed_keyframe": (
                        final_by_id[failed_keyframe_id].model_dump(mode="json")
                        if failed_keyframe_id in final_by_id
                        else None
                    ),
                    "resolved_final_tcp_keyframes": [
                        {
                            "keyframe_id": resolved.keyframe_id,
                            "final_tcp_world_pose": resolved.pose.model_dump(mode="json"),
                        }
                        for resolved in attempt.resolved_keyframes
                    ],
                    "ik_diagnostics": [
                        {
                            "keyframe_id": diagnostic.keyframe_id,
                            "raw_ik_count": diagnostic.raw_ik_count,
                            "valid_ik_count": diagnostic.valid_ik_count,
                            "attempted_seeds": diagnostic.attempted_seeds,
                            "solver_failure_code": diagnostic.solver_failure_code,
                            "solver_detail": diagnostic.solver_detail,
                            "validity_detail": diagnostic.validity_detail,
                        }
                        for diagnostic in attempt.ik_diagnostics
                    ],
                }
            )
        _write_debug_json(
            self._debug_dir / f"{_debug_name(request)}-planning.json",
            {
                "stage": "BINDER_AND_COMPILER",
                "request_id": request.request_id,
                "subgoal_id": request.task.subgoal_id,
                "binder_artifact_id": binder_artifact.artifact_id,
                "binder_strategies": _resolved_debug_keyframes(
                    request, binder_artifact, retarget=True
                ),
                "final_artifact_id": final_artifact.artifact_id,
                "final_strategies": _resolved_debug_keyframes(
                    request, final_artifact, retarget=True
                ),
                "compilation_attempts": attempts,
            },
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
        try:
            validate_keyframe_phase_contract(request, artifact)
        except KeyframePhaseContractError as error:
            raise MotionPlanningPipelineError(
                f"keyframe artifact violates the task phase contract: {error}"
            ) from error

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
        final_plan_validator: Callable[[MotionPlanRequest, MotionPlan], None] | None = None,
        collision_context_factory: CollisionContextFactory | None = None,
        edge_planner: EdgePlanner | None = None,
    ) -> MotionPlanningResult:
        """Generate candidates, choose a connected branch, and finalize timing."""

        update_reference = getattr(self._kinematics, "set_reference_qpos", None)
        if callable(update_reference):
            update_reference(request.world.robot_state.joint_positions_rad)
        artifact = self._provider.generate(request)
        self._validate_artifact(request, artifact)
        binder_artifact = artifact
        edge_context_materializer: EdgeContextMaterializer | None = None
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
            edge_context_materializer = setup.edge_context_materializer
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
        if edge_context_materializer is not None:
            selected_edge_planner = _EdgeContextMaterializingPlanner(
                selected_edge_planner,
                self._kinematics,
                edge_context_materializer,
            )
        plan_id, provenance = self._plan_identity(request, artifact)

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

        remaining_candidates = list(artifact.candidates)
        attempts: list[StrategyAttempt] = []
        last_build_error: Exception | None = None
        while remaining_candidates:
            current = compiler.compile(
                request.world,
                remaining_candidates,
                start_joint_config=request.world.robot_state.joint_positions_rad,
                state_validator=effective_state_validator,
                edge_planner=selected_edge_planner,
                request=request,
            )
            attempts.extend(current.attempts)
            compilation = StrategyCompilationResult(
                connected=current.connected,
                attempts=tuple(attempts),
            )
            if current.connected is None:
                break
            try:
                setattr(final_validator, "last_path_collision_check", None)
                plan = self._builder.build(
                    request,
                    current.connected,
                    plan_id=plan_id,
                    provenance=provenance,
                    collision_contexts=collision_contexts,
                    initial_collision_context_id=initial_collision_context_id,
                    final_segment_validator=final_validator,
                    joint_position_limits_rad=joint_limits,
                )
                if final_plan_validator is not None:
                    final_plan_validator(request, plan)
            except (
                MotionPlanBuildError,
                TrajectoryProcessingError,
                ValueError,
            ) as error:
                last_build_error = error
                connected_attempt = attempts.pop()
                attempts.append(
                    StrategyAttempt(
                        strategy_id=connected_attempt.strategy_id,
                        resolved_keyframes=connected_attempt.resolved_keyframes,
                        selection=connected_attempt.selection,
                        failure_code="FINAL_VALIDATION_FAILED",
                        detail=f"{type(error).__name__}: {error}",
                        ik_diagnostics=connected_attempt.ik_diagnostics,
                    )
                )
                attempted_ids = {
                    attempt.strategy_id for attempt in current.attempts
                }
                remaining_candidates = [
                    candidate
                    for candidate in remaining_candidates
                    if candidate.strategy_id not in attempted_ids
                ]
                continue
            compilation = StrategyCompilationResult(
                connected=current.connected,
                attempts=tuple(attempts),
            )
            self._dump_planning_debug(
                request, binder_artifact, artifact, compilation
            )
            return MotionPlanningResult(
                keyframe_artifact=artifact,
                compilation=compilation,
                plan=plan,
            )

        compilation = StrategyCompilationResult(
            connected=None,
            attempts=tuple(attempts),
        )
        self._dump_planning_debug(
            request, binder_artifact, artifact, compilation
        )
        failures = "; ".join(
            (
                f"{attempt.strategy_id}="
                f"{attempt.failure_code or 'UNKNOWN'}"
                + (f" ({attempt.detail})" if attempt.detail else "")
            )
            for attempt in compilation.attempts
        )
        message = "no generated keyframe strategy produced a finalizable path"
        error = MotionPlanningPipelineError(
            f"{message}: {failures}",
            compilation=compilation,
        )
        repair_attempt = _collision_repair_attempt(request)
        if (
            repair_attempt < _MAX_COLLISION_REPAIR_BATCHES
            and _provider_supports_collision_feedback(self._provider)
            and _collision_repair_eligible(compilation)
        ):
            retry_request = request.model_copy(deep=True)
            retry_request.task.metadata = {
                **retry_request.task.metadata,
                _COLLISION_REPAIR_FEEDBACK_KEY: _collision_repair_feedback(
                    request,
                    compilation,
                ),
            }
            retry_arguments = {
                "edge_planner": edge_planner,
                "final_plan_validator": final_plan_validator,
            }
            if collision_context_factory is not None:
                retry_arguments["collision_context_factory"] = (
                    collision_context_factory
                )
            else:
                retry_arguments.update(
                    {
                        "state_validator": state_validator,
                        "collision_contexts": collision_contexts,
                        "initial_collision_context_id": (
                            initial_collision_context_id
                        ),
                        "final_segment_validator": final_segment_validator,
                    }
                )
            try:
                repaired = self.plan(retry_request, **retry_arguments)
            except MotionPlanningPipelineError as repair_error:
                repair_compilation = repair_error.compilation
                combined_attempts = compilation.attempts + (
                    repair_compilation.attempts
                    if repair_compilation is not None
                    else ()
                )
                raise MotionPlanningPipelineError(
                    "collision-feedback repair exhausted after "
                    f"{_MAX_COLLISION_REPAIR_BATCHES} additional batch: "
                    f"{repair_error}",
                    compilation=StrategyCompilationResult(
                        connected=None,
                        attempts=combined_attempts,
                    ),
                ) from repair_error
            return MotionPlanningResult(
                keyframe_artifact=repaired.keyframe_artifact,
                compilation=StrategyCompilationResult(
                    connected=repaired.compilation.connected,
                    attempts=(
                        compilation.attempts
                        + repaired.compilation.attempts
                    ),
                ),
                plan=repaired.plan,
            )
        if last_build_error is not None:
            raise error from last_build_error
        raise error


__all__ = [
    "CollisionContextFactory",
    "CollisionPlanningSetup",
    "KeyframeStrategyProvider",
    "MotionPlanningPipeline",
    "MotionPlanningPipelineError",
    "MotionPlanningResult",
]
