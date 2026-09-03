"""Validated precomputed EE return and return-plus-attach composition.

An ``EE_EXCHANGE`` is represented as two independently commissioned fixed
workcell trajectories::

    current EE -> bare canonical home -> requested EE

This avoids storing every pairwise EE-to-EE combination.  Both artifacts are
validated against the current workcell and dynamic scene before they are bound
into one request-scoped ``MotionPlan``.  No IK, RRT, or Cartesian planner is
invoked on a cache hit.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from tuj.m5_motion.precomputed_ee_attach import (
    CollisionChecker,
    EEAttachPathFailureCode,
    EEAttachTrajectoryEventTemplate,
    EEAttachTrajectorySegmentTemplate,
    EEAttachTrajectoryTemplate,
    PrecomputedEEAttachPlanner,
    PrecomputedEEAttachRegistry,
    PrecomputedEEPathError,
    _TemplateModel,
    _digest,
    _robot_model,
    compute_rack_signature,
    compute_workcell_signature,
    normalize_ee_id,
)
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    CollisionContext,
    EventType,
    ModuleName,
    MotionPlan,
    MotionPlanRequest,
    MotionTask,
    Pose,
    RobotState,
    SegmentType,
    TrajectoryEvent,
    TrajectoryProcessingStep,
    TrajectorySegment,
    TrajectoryWaypoint,
    WorldSnapshot,
)
from tuj.m5_motion.task_semantics import task_operation


EE_RETURN_TEMPLATE_SCHEMA_VERSION = "1.0"


def is_ee_exchange_request(request: MotionPlanRequest) -> bool:
    """Whether a request is a physical attached-EE to different-EE exchange."""

    if task_operation(request.task) != "EE_EXCHANGE":
        return False
    raw_source = request.task.metadata.get("from_ee")
    raw_target = request.task.metadata.get("to_ee") or request.task.ee
    try:
        source = normalize_ee_id(raw_source)
        target = normalize_ee_id(raw_target)
    except ValueError:
        return False
    physical = request.world.metadata.get("physical_active_ee")
    try:
        active = normalize_ee_id(physical)
    except ValueError:
        return False
    return source == active and source != target


class EEReturnTrajectoryTemplate(_TemplateModel):
    """Request-independent, controller-verified attached EE -> bare path."""

    schema_version: Literal["1.0"] = EE_RETURN_TEMPLATE_SCHEMA_VERSION
    trajectory_id: str = Field(min_length=1)
    environment_name: str = Field(min_length=1)
    robot_model: str = Field(min_length=1)
    source_active_ee: str = Field(min_length=1)
    target_active_ee: None = None
    joint_names: list[str] = Field(min_length=1)
    start_joint_positions_rad: list[float] = Field(min_length=1)
    start_eef_pose: Pose | None = None
    workcell_signature: str = Field(min_length=1)
    rack_signature: str = Field(min_length=1)
    collision_model_versions: dict[str, str] = Field(min_length=1)
    segments: list[EEAttachTrajectorySegmentTemplate] = Field(min_length=1)
    events: list[EEAttachTrajectoryEventTemplate] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_active_ee", mode="before")
    @classmethod
    def _normalize_source(cls, value: object) -> str:
        return normalize_ee_id(value)

    @model_validator(mode="after")
    def _validate_trajectory(self) -> "EEReturnTrajectoryTemplate":
        dof = len(self.joint_names)
        if len(self.start_joint_positions_rad) != dof:
            raise ValueError("start joint positions must match joint_names")
        if not all(math.isfinite(value) for value in self.start_joint_positions_rad):
            raise ValueError("start joint positions must be finite")
        previous: EEAttachTrajectorySegmentTemplate | None = None
        for segment in self.segments:
            if segment.collision_context_before not in self.collision_model_versions:
                raise ValueError("segment collision_context_before has no model version")
            if segment.collision_context_after not in self.collision_model_versions:
                raise ValueError("segment collision_context_after has no model version")
            if any(len(item.joint_positions_rad) != dof for item in segment.waypoints):
                raise ValueError("all waypoints must match template joint_names")
            if previous is None:
                if not math.isclose(
                    segment.waypoints[0].time_from_start_s, 0.0, abs_tol=1e-9
                ):
                    raise ValueError("the first segment must start at t=0")
                if any(
                    abs(left - right) > 1e-9
                    for left, right in zip(
                        segment.waypoints[0].joint_positions_rad,
                        self.start_joint_positions_rad,
                    )
                ):
                    raise ValueError("first waypoint must match start joint positions")
            else:
                if not math.isclose(
                    previous.waypoints[-1].time_from_start_s,
                    segment.waypoints[0].time_from_start_s,
                    abs_tol=1e-9,
                ):
                    raise ValueError("template segments must have a continuous timeline")
                if any(
                    abs(left - right) > 1e-9
                    for left, right in zip(
                        previous.waypoints[-1].joint_positions_rad,
                        segment.waypoints[0].joint_positions_rad,
                    )
                ):
                    raise ValueError("template segment joint paths must be continuous")
            previous = segment
        duration = self.segments[-1].waypoints[-1].time_from_start_s
        if any(event.time_from_start_s > duration for event in self.events):
            raise ValueError("template events must occur within its duration")
        return self


class PrecomputedEEReturnRegistry:
    """Load environment-scoped EE -> bare templates."""

    def __init__(
        self,
        root: str | Path,
        *,
        trajectory_paths: Sequence[str | Path] = (),
    ) -> None:
        self.root = Path(root)
        self._overrides: dict[tuple[str, str], Path] = {}
        for raw_path in trajectory_paths:
            path = Path(raw_path)
            template = self._load_file(path)
            key = (template.environment_name, template.source_active_ee)
            if key in self._overrides:
                raise ValueError(f"duplicate EE return trajectory override for {key}")
            self._overrides[key] = path

    @staticmethod
    def _load_file(path: Path) -> EEReturnTrajectoryTemplate:
        try:
            return EEReturnTrajectoryTemplate.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except FileNotFoundError as error:
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_NOT_FOUND,
                f"trajectory file not found: {path}",
            ) from error
        except Exception as error:  # noqa: BLE001
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                f"invalid EE return trajectory template {path}: {error}",
            ) from error

    def load(
        self, environment_name: str, source_ee: object
    ) -> EEReturnTrajectoryTemplate:
        try:
            source = normalize_ee_id(source_ee)
        except ValueError as error:
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_NOT_FOUND,
                str(error),
            ) from error
        path = self._overrides.get((environment_name, source))
        if path is None:
            path = self.root / environment_name / f"{source}_to_bare.json"
        template = self._load_file(path)
        if (
            template.environment_name != environment_name
            or template.source_active_ee != source
        ):
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                "EE return metadata does not match the registry lookup",
                trajectory_id=template.trajectory_id,
            )
        return template


def _reverse_waypoint(
    waypoint: TrajectoryWaypoint, duration_s: float
) -> TrajectoryWaypoint:
    return waypoint.model_copy(
        deep=True,
        update={
            "time_from_start_s": duration_s - waypoint.time_from_start_s,
            "joint_velocities_rad_s": (
                [-value for value in waypoint.joint_velocities_rad_s]
                if waypoint.joint_velocities_rad_s is not None
                else None
            ),
            # q_rev(t)=q(T-t): acceleration keeps its sign under time reversal.
            "joint_accelerations_rad_s2": (
                list(waypoint.joint_accelerations_rad_s2)
                if waypoint.joint_accelerations_rad_s2 is not None
                else None
            ),
        },
    )


def derive_return_template_from_attach(
    attach: EEAttachTrajectoryTemplate,
    world: WorldSnapshot,
    collision_contexts: Mapping[str, CollisionContext],
    *,
    trajectory_id: str | None = None,
) -> EEReturnTrajectoryTemplate:
    """Create a return commissioning candidate by reversing an attach path.

    The candidate is not considered validated until a caller executes it through
    the controller and replays the saved artifact.  Contexts and events are
    rebuilt for the reverse physical state transition rather than copied.
    """

    source = attach.target_active_ee
    physical = world.metadata.get("physical_active_ee")
    try:
        physical = normalize_ee_id(physical)
    except ValueError as error:
        raise ValueError("return commissioning world has no active rack EE") from error
    if physical != source:
        raise ValueError("return commissioning world active EE differs from attach")
    final_attach = attach.segments[-1].waypoints[-1]
    if world.robot_state.joint_names != attach.joint_names:
        raise ValueError("return commissioning joint names differ from attach")
    if max(
        abs(left - right)
        for left, right in zip(
            world.robot_state.joint_positions_rad,
            final_attach.joint_positions_rad,
        )
    ) > 1e-8:
        raise ValueError("return commissioning world is not at exchange-entry")

    attached_contact = f"ee-attached-dock-contact:{source}"
    bare_contact = f"bare-flange-dock-contact:{source}"
    required_context_ids = {"bare-flange", bare_contact, attached_contact}
    missing = required_context_ids - set(collision_contexts)
    if missing:
        raise ValueError(f"return collision contexts are missing {sorted(missing)}")
    context_subset = {
        context_id: collision_contexts[context_id]
        for context_id in sorted(required_context_ids)
    }
    duration = attach.segments[-1].waypoints[-1].time_from_start_s
    lock_events = [
        event for event in attach.events if event.event_type is EventType.TOOL_LOCK
    ]
    verify_events = [
        event
        for event in attach.events
        if event.event_type is EventType.VERIFY_TOOL_LOCK
    ]
    if len(lock_events) != 1 or len(verify_events) != 1:
        raise ValueError("attach candidate lacks exactly one lock/verify pair")
    release_time = duration - lock_events[0].time_from_start_s

    segments: list[EEAttachTrajectorySegmentTemplate] = []
    reversed_source = list(reversed(attach.segments))
    for index, original in enumerate(reversed_source):
        waypoints = [
            _reverse_waypoint(item, duration)
            for item in reversed(original.waypoints)
        ]
        segments.append(
            EEAttachTrajectorySegmentTemplate(
                segment_id=f"{source}-return:{index}:{original.segment_id}",
                segment_type=(
                    SegmentType.EE_UNDOCK if index == 0 else SegmentType.RETREAT
                ),
                collision_context_before=(
                    attached_contact
                    if index == 0
                    else bare_contact if index == 1 else "bare-flange"
                ),
                collision_context_after=(
                    bare_contact if index == 0 else "bare-flange"
                ),
                interpolation=original.interpolation,
                waypoints=waypoints,
                metadata={
                    "source": "reverse-attach-commissioning-candidate",
                    "source_segment_id": original.segment_id,
                },
            )
        )

    fingerprint = _digest(
        {
            "joint_names": attach.joint_names,
            "segments": [segment.model_dump(mode="json") for segment in segments],
            "source_attach": attach.trajectory_id,
        }
    )
    return EEReturnTrajectoryTemplate(
        trajectory_id=trajectory_id or f"ur5e-{source}-to-bare-{fingerprint[:12]}",
        environment_name=attach.environment_name,
        robot_model=attach.robot_model,
        source_active_ee=source,
        joint_names=list(attach.joint_names),
        start_joint_positions_rad=list(final_attach.joint_positions_rad),
        start_eef_pose=(
            world.robot_state.eef_pose.model_copy(deep=True)
            if world.robot_state.eef_pose is not None
            else None
        ),
        workcell_signature=compute_workcell_signature(world, context_subset),
        rack_signature=compute_rack_signature(world),
        collision_model_versions={
            context_id: context.collision_model_version
            for context_id, context in context_subset.items()
        },
        segments=segments,
        events=[
            EEAttachTrajectoryEventTemplate(
                time_from_start_s=release_time,
                event_type=EventType.TOOL_UNLOCK,
                target_id=source,
            ),
            EEAttachTrajectoryEventTemplate(
                time_from_start_s=release_time,
                event_type=EventType.VERIFY_TOOL_RELEASE,
                target_id=source,
            ),
        ],
        metadata={
            "derived_from_attach_trajectory_id": attach.trajectory_id,
            "source_plan_fingerprint": fingerprint,
            "requires_controller_validation": True,
        },
    )


class PrecomputedEEExchangePlanner:
    """Validate return and attach templates, then compose one MotionPlan."""

    def __init__(
        self,
        return_registry: PrecomputedEEReturnRegistry,
        attach_planner: PrecomputedEEAttachPlanner,
        *,
        log: Any = print,
    ) -> None:
        self.return_registry = return_registry
        self.attach_planner = attach_planner
        self._log = log

    @staticmethod
    def _failure(
        code: EEAttachPathFailureCode,
        detail: str,
        template: EEReturnTrajectoryTemplate | EEAttachTrajectoryTemplate | None = None,
    ) -> PrecomputedEEPathError:
        return PrecomputedEEPathError(
            code,
            detail,
            trajectory_id=(template.trajectory_id if template is not None else None),
        )

    def load(
        self, request: MotionPlanRequest
    ) -> tuple[EEReturnTrajectoryTemplate, EEAttachTrajectoryTemplate]:
        environment = request.world.metadata.get("environment_name")
        if not isinstance(environment, str) or not environment:
            raise self._failure(
                EEAttachPathFailureCode.WORKCELL_SIGNATURE_MISMATCH,
                "request has no environment_name",
            )
        source = request.task.metadata.get("from_ee")
        target = request.task.metadata.get("to_ee") or request.task.ee
        return (
            self.return_registry.load(environment, source),
            self.attach_planner.registry.load(environment, target),
        )

    def _validate_return_state(
        self,
        request: MotionPlanRequest,
        template: EEReturnTrajectoryTemplate,
    ) -> None:
        if not is_ee_exchange_request(request):
            raise self._failure(
                EEAttachPathFailureCode.START_STATE_MISMATCH,
                "request is not an attached-EE EE_EXCHANGE",
                template,
            )
        state = request.world.robot_state
        if state.attached_object_id is not None or state.held_tool_id is not None:
            raise self._failure(
                EEAttachPathFailureCode.START_STATE_MISMATCH,
                "EE exchange requires no attached object and no held tool",
                template,
            )
        source = normalize_ee_id(request.task.metadata.get("from_ee"))
        if template.source_active_ee != source:
            raise self._failure(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                "stored return source differs from the request",
                template,
            )
        if state.joint_names != template.joint_names:
            raise self._failure(
                EEAttachPathFailureCode.START_STATE_MISMATCH,
                "joint name or order differs from the stored return trajectory",
                template,
            )
        maximum_error = max(
            abs(current - stored)
            for current, stored in zip(
                state.joint_positions_rad,
                template.start_joint_positions_rad,
            )
        )
        if maximum_error > self.attach_planner.start_tolerance_rad:
            raise self._failure(
                EEAttachPathFailureCode.START_STATE_MISMATCH,
                f"maximum return start joint error {maximum_error:.6f} rad exceeds "
                f"{self.attach_planner.start_tolerance_rad:.6f} rad; move to the "
                "stored exchange-entry state first",
                template,
            )

    def _validate_return_events(
        self,
        template: EEReturnTrajectoryTemplate,
        contexts: Mapping[str, CollisionContext],
    ) -> None:
        unlock = [
            event for event in template.events if event.event_type is EventType.TOOL_UNLOCK
        ]
        verify = [
            event
            for event in template.events
            if event.event_type is EventType.VERIFY_TOOL_RELEASE
        ]
        if len(unlock) != 1 or len(verify) != 1:
            raise self._failure(
                EEAttachPathFailureCode.RELEASE_EVENT_MISSING,
                "return trajectory requires exactly one TOOL_UNLOCK and "
                "VERIFY_TOOL_RELEASE",
                template,
            )
        try:
            targets = {
                normalize_ee_id(unlock[0].target_id),
                normalize_ee_id(verify[0].target_id),
            }
        except ValueError as error:
            raise self._failure(
                EEAttachPathFailureCode.RELEASE_EVENT_MISSING,
                str(error),
                template,
            ) from error
        if targets != {template.source_active_ee} or (
            verify[0].time_from_start_s < unlock[0].time_from_start_s
        ):
            raise self._failure(
                EEAttachPathFailureCode.RELEASE_EVENT_MISSING,
                "unlock/verify target or ordering is invalid",
                template,
            )
        initial = contexts.get(template.segments[0].collision_context_before)
        final = contexts.get(template.segments[-1].collision_context_after)
        if initial is None or initial.active_ee != template.source_active_ee:
            raise self._failure(
                EEAttachPathFailureCode.FINAL_EE_STATE_INVALID,
                "return trajectory does not start with the source EE attached",
                template,
            )
        if final is None or final.active_ee is not None:
            raise self._failure(
                EEAttachPathFailureCode.FINAL_EE_STATE_INVALID,
                "return trajectory does not finish with a bare flange",
                template,
            )

    @staticmethod
    def _subset_contexts(
        versions: Mapping[str, str],
        contexts: Mapping[str, CollisionContext],
    ) -> dict[str, CollisionContext]:
        try:
            return {context_id: contexts[context_id] for context_id in versions}
        except KeyError as error:
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                f"required collision context is unavailable: {error.args[0]}",
            ) from error

    def _validate_return_workcell(
        self,
        request: MotionPlanRequest,
        template: EEReturnTrajectoryTemplate,
        contexts: Mapping[str, CollisionContext],
    ) -> dict[str, CollisionContext]:
        selected = self._subset_contexts(template.collision_model_versions, contexts)
        if template.robot_model != _robot_model(request.world):
            raise self._failure(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                "robot model differs from the stored return trajectory",
                template,
            )
        if template.rack_signature != compute_rack_signature(request.world):
            raise self._failure(
                EEAttachPathFailureCode.WORKCELL_SIGNATURE_MISMATCH,
                "rack/dock signature differs from the stored return trajectory",
                template,
            )
        versions = {
            context_id: context.collision_model_version
            for context_id, context in selected.items()
        }
        if versions != template.collision_model_versions:
            raise self._failure(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                "return collision model versions differ from the stored trajectory",
                template,
            )
        if template.start_eef_pose is None:
            raise self._failure(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                "stored return trajectory has no mounted-EE canonical EEF pose",
                template,
            )
        # The controller may finish the entry move within the configured joint
        # tolerance rather than at bit-identical q.  The workcell hash is a
        # static compatibility check, so evaluate it at the artifact's
        # canonical entry state instead of accidentally hashing tracking error.
        canonical_world = request.world.model_copy(deep=True)
        canonical_world.robot_state.joint_positions_rad = list(
            template.start_joint_positions_rad
        )
        canonical_world.robot_state.joint_velocities_rad_s = [
            0.0
        ] * len(template.joint_names)
        canonical_world.robot_state.eef_pose = template.start_eef_pose.model_copy(
            deep=True
        )
        if compute_workcell_signature(canonical_world, selected) != template.workcell_signature:
            raise self._failure(
                EEAttachPathFailureCode.WORKCELL_SIGNATURE_MISMATCH,
                "static workcell signature differs from the stored return trajectory",
                template,
            )
        return selected

    def _seam_request(
        self,
        request: MotionPlanRequest,
        return_template: EEReturnTrajectoryTemplate,
        attach_template: EEAttachTrajectoryTemplate,
    ) -> MotionPlanRequest:
        return_end = return_template.segments[-1].waypoints[-1]
        attach_start = attach_template.segments[0].waypoints[0]
        maximum_error = max(
            abs(left - right)
            for left, right in zip(
                return_end.joint_positions_rad,
                attach_start.joint_positions_rad,
            )
        )
        if maximum_error > 1e-9:
            raise self._failure(
                EEAttachPathFailureCode.TRANSITION_SEAM_MISMATCH,
                f"return/attach seam joint error is {maximum_error:.9f} rad",
                return_template,
            )
        canonical_start_eef = attach_template.start_eef_pose or attach_start.eef_pose
        if canonical_start_eef is None:
            raise self._failure(
                EEAttachPathFailureCode.TRANSITION_SEAM_MISMATCH,
                "attach template has no canonical bare-home EEF pose",
                attach_template,
            )
        target = normalize_ee_id(request.task.metadata.get("to_ee") or request.task.ee)
        seam_world = request.world.model_copy(deep=True)
        seam_world.robot_state.joint_positions_rad = list(
            attach_start.joint_positions_rad
        )
        seam_world.robot_state.joint_velocities_rad_s = [
            0.0
        ] * len(attach_template.joint_names)
        seam_world.robot_state.eef_pose = canonical_start_eef.model_copy(deep=True)
        seam_world.metadata["physical_active_ee"] = None
        seam_world.metadata["declared_active_ee"] = None
        return request.model_copy(
            deep=True,
            update={
                "request_id": f"{request.request_id}:attach-component",
                "world": seam_world,
                "task": MotionTask(
                    task_id=f"{request.task.task_id}:attach-component",
                    subgoal_id=f"{request.task.subgoal_id}:attach-component",
                    action_type="EE_ATTACH",
                    ee=target,
                    target_ids=[target],
                    goal=request.task.goal.model_copy(deep=True),
                    allowed_touch_objects=list(request.task.allowed_touch_objects),
                    metadata={"from_ee": None, "to_ee": target},
                ),
            },
        )

    def _bind_return_segments(
        self,
        request: MotionPlanRequest,
        template: EEReturnTrajectoryTemplate,
        contexts: Mapping[str, CollisionContext],
        clearances: Mapping[str, float | None],
    ) -> list[TrajectorySegment]:
        return [
            TrajectorySegment(
                segment_id=(
                    f"{request.task.subgoal_id}:precomputed:return:{index}:"
                    f"{segment.segment_id}"
                ),
                segment_type=segment.segment_type,
                start_time_s=segment.waypoints[0].time_from_start_s,
                end_time_s=segment.waypoints[-1].time_from_start_s,
                interpolation=segment.interpolation,
                waypoints=[item.model_copy(deep=True) for item in segment.waypoints],
                collision_checked=True,
                min_clearance_m=clearances[segment.segment_id],
                collision_context_before=contexts[segment.collision_context_before],
                collision_context_after=contexts[segment.collision_context_after],
                processing_steps=[
                    TrajectoryProcessingStep.RAW_PATH,
                    TrajectoryProcessingStep.TIME_PARAMETERIZATION,
                    TrajectoryProcessingStep.FINAL_COLLISION_CHECK,
                    TrajectoryProcessingStep.DYNAMICS_CHECK,
                ],
                metadata={
                    **segment.metadata,
                    "source": "precomputed",
                    "trajectory_id": template.trajectory_id,
                    "source_segment_id": segment.segment_id,
                },
            )
            for index, segment in enumerate(template.segments)
        ]

    def plan_return_only(
        self,
        request: MotionPlanRequest,
        *,
        collision_contexts: Mapping[str, CollisionContext],
        collision_checker: CollisionChecker,
        template: EEReturnTrajectoryTemplate | None = None,
    ) -> MotionPlan:
        """Bind a return-only plan for commissioning and artifact replay."""

        selected = template
        if selected is None:
            environment = request.world.metadata.get("environment_name")
            selected = self.return_registry.load(
                str(environment), request.task.metadata.get("from_ee")
            )
        self._validate_return_state(request, selected)
        return_contexts = self._validate_return_workcell(
            request, selected, collision_contexts
        )
        self._validate_return_events(selected, return_contexts)
        self.attach_planner._validate_dynamics(request, selected)  # noqa: SLF001
        clearances = self.attach_planner._validate_collisions(  # noqa: SLF001
            request,
            selected,
            collision_contexts,
            collision_checker,
        )
        segments = self._bind_return_segments(
            request, selected, collision_contexts, clearances
        )
        source = selected.source_active_ee
        events = [
            TrajectoryEvent(
                event_id=(
                    f"{request.task.subgoal_id}:precomputed:return:event:{index}:"
                    f"{event.event_type.value}"
                ),
                time_from_start_s=event.time_from_start_s,
                event_type=event.event_type,
                target_id=source,
                command=event.command,
                parameters=dict(event.parameters),
            )
            for index, event in enumerate(selected.events)
        ]
        final_waypoint = segments[-1].waypoints[-1]
        digest = hashlib.sha256(
            (
                f"{request.request_id}|{request.world.scene.signature}|"
                f"{selected.trajectory_id}|return-only"
            ).encode("utf-8")
        ).hexdigest()[:24]
        return MotionPlan(
            plan_id=f"motion-plan:{digest}",
            request_id=request.request_id,
            provenance=ArtifactProvenance(
                artifact_id=f"motion-plan-artifact:{digest}",
                artifact_type="MotionPlan",
                produced_by=ModuleName.MOTION_PLANNER,
                invocation_id=f"precomputed-ee-return:{request.request_id}",
                input_artifact_ids=[request.provenance.artifact_id],
                metadata={
                    "source": "precomputed",
                    "trajectory_id": selected.trajectory_id,
                },
            ),
            scene_signature=request.world.scene.signature,
            robot_id=request.world.robot_state.robot_id,
            joint_names=list(selected.joint_names),
            duration_s=segments[-1].end_time_s,
            segments=segments,
            events=events,
            expected_final_state=RobotState(
                robot_id=request.world.robot_state.robot_id,
                joint_names=list(selected.joint_names),
                joint_positions_rad=list(final_waypoint.joint_positions_rad),
                joint_velocities_rad_s=(
                    list(final_waypoint.joint_velocities_rad_s)
                    if final_waypoint.joint_velocities_rad_s is not None
                    else [0.0] * len(selected.joint_names)
                ),
                eef_pose=(
                    final_waypoint.eef_pose.model_copy(deep=True)
                    if final_waypoint.eef_pose is not None
                    else None
                ),
                gripper=(
                    request.world.robot_state.gripper.model_copy(deep=True)
                    if request.world.robot_state.gripper is not None
                    else None
                ),
                attached_object_id=None,
                held_tool_id=None,
            ),
            metadata={
                "source": "precomputed",
                "trajectory_id": selected.trajectory_id,
                "selection_policy": "PRECOMPUTED_EE_RETURN",
                "source_active_ee": source,
                "target_active_ee": None,
                "start_state_validation": "PASS",
                "workcell_signature_validation": "PASS",
                "collision_validation": "PASS",
                "dynamic_planner_invoked": False,
            },
        )

    def plan(
        self,
        request: MotionPlanRequest,
        *,
        collision_contexts: Mapping[str, CollisionContext],
        collision_checker: CollisionChecker,
        templates: tuple[EEReturnTrajectoryTemplate, EEAttachTrajectoryTemplate]
        | None = None,
    ) -> MotionPlan:
        return_template, attach_template = templates or self.load(request)
        source = normalize_ee_id(request.task.metadata.get("from_ee"))
        target = normalize_ee_id(request.task.metadata.get("to_ee") or request.task.ee)
        if source == target:
            raise self._failure(
                EEAttachPathFailureCode.START_STATE_MISMATCH,
                "EE exchange source and target must differ",
            )
        self._validate_return_state(request, return_template)
        return_contexts = self._validate_return_workcell(
            request, return_template, collision_contexts
        )
        self._validate_return_events(return_template, return_contexts)
        self.attach_planner._validate_dynamics(request, return_template)  # noqa: SLF001
        return_clearances = self.attach_planner._validate_collisions(  # noqa: SLF001
            request,
            return_template,
            collision_contexts,
            collision_checker,
        )

        seam_request = self._seam_request(request, return_template, attach_template)
        attach_contexts = self._subset_contexts(
            attach_template.collision_model_versions, collision_contexts
        )
        attach_plan = self.attach_planner.plan(
            seam_request,
            collision_contexts=attach_contexts,
            collision_checker=collision_checker,
            template=attach_template,
        )

        return_segments = self._bind_return_segments(
            request, return_template, collision_contexts, return_clearances
        )
        return_duration = return_segments[-1].end_time_s
        attach_segments: list[TrajectorySegment] = []
        for segment in attach_plan.segments:
            shifted_waypoints = [
                waypoint.model_copy(
                    deep=True,
                    update={
                        "time_from_start_s": (
                            waypoint.time_from_start_s + return_duration
                        )
                    },
                )
                for waypoint in segment.waypoints
            ]
            attach_segments.append(
                segment.model_copy(
                    deep=True,
                    update={
                        "segment_id": f"{request.task.subgoal_id}:exchange:{segment.segment_id}",
                        "start_time_s": segment.start_time_s + return_duration,
                        "end_time_s": segment.end_time_s + return_duration,
                        "waypoints": shifted_waypoints,
                    },
                )
            )

        events = [
            TrajectoryEvent(
                event_id=(
                    f"{request.task.subgoal_id}:precomputed:return:event:{index}:"
                    f"{event.event_type.value}"
                ),
                time_from_start_s=event.time_from_start_s,
                event_type=event.event_type,
                target_id=source,
                command=event.command,
                parameters=dict(event.parameters),
            )
            for index, event in enumerate(return_template.events)
        ]
        events.extend(
            event.model_copy(
                deep=True,
                update={
                    "event_id": f"{request.task.subgoal_id}:exchange:{event.event_id}",
                    "time_from_start_s": event.time_from_start_s + return_duration,
                    "target_id": target,
                },
            )
            for event in attach_plan.events
        )
        digest = hashlib.sha256(
            (
                f"{request.request_id}|{request.world.scene.signature}|"
                f"{return_template.trajectory_id}|{attach_template.trajectory_id}"
            ).encode("utf-8")
        ).hexdigest()[:24]
        final_state = attach_plan.expected_final_state.model_copy(deep=True)
        plan = MotionPlan(
            plan_id=f"motion-plan:{digest}",
            request_id=request.request_id,
            provenance=ArtifactProvenance(
                artifact_id=f"motion-plan-artifact:{digest}",
                artifact_type="MotionPlan",
                produced_by=ModuleName.MOTION_PLANNER,
                invocation_id=f"precomputed-ee-exchange:{request.request_id}",
                input_artifact_ids=[request.provenance.artifact_id],
                metadata={
                    "source": "precomputed",
                    "return_trajectory_id": return_template.trajectory_id,
                    "attach_trajectory_id": attach_template.trajectory_id,
                },
            ),
            scene_signature=request.world.scene.signature,
            robot_id=request.world.robot_state.robot_id,
            joint_names=list(attach_template.joint_names),
            duration_s=attach_segments[-1].end_time_s,
            segments=[*return_segments, *attach_segments],
            events=events,
            expected_final_state=RobotState(
                robot_id=final_state.robot_id,
                joint_names=list(final_state.joint_names),
                joint_positions_rad=list(final_state.joint_positions_rad),
                joint_velocities_rad_s=list(final_state.joint_velocities_rad_s),
                eef_pose=(
                    final_state.eef_pose.model_copy(deep=True)
                    if final_state.eef_pose is not None
                    else None
                ),
                gripper=(
                    request.world.robot_state.gripper.model_copy(deep=True)
                    if request.world.robot_state.gripper is not None
                    else None
                ),
                attached_object_id=None,
                held_tool_id=None,
            ),
            metadata={
                "source": "precomputed",
                "trajectory_id": (
                    f"{return_template.trajectory_id}+{attach_template.trajectory_id}"
                ),
                "return_trajectory_id": return_template.trajectory_id,
                "attach_trajectory_id": attach_template.trajectory_id,
                "selection_policy": "PRECOMPUTED_EE_EXCHANGE",
                "source_active_ee": source,
                "target_active_ee": target,
                "start_state_validation": "PASS",
                "workcell_signature_validation": "PASS",
                "transition_seam_validation": "PASS",
                "collision_validation": "PASS",
                "dynamic_planner_invoked": False,
            },
        )
        self._log(
            f"[M5][EE_PATH] hit: {return_template.trajectory_id} + "
            f"{attach_template.trajectory_id}"
        )
        self._log("[M5][EE_PATH] return_start_state=PASS")
        self._log("[M5][EE_PATH] transition_seam=PASS")
        self._log("[M5][EE_PATH] collision_validation=PASS")
        self._log("[M5][EE_PATH] source=precomputed")
        return plan


def save_ee_return_template(
    path: str | Path, template: EEReturnTrajectoryTemplate
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(template.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(destination)


__all__ = [
    "EE_RETURN_TEMPLATE_SCHEMA_VERSION",
    "EEReturnTrajectoryTemplate",
    "PrecomputedEEExchangePlanner",
    "PrecomputedEEReturnRegistry",
    "derive_return_template_from_attach",
    "is_ee_exchange_request",
    "save_ee_return_template",
]
