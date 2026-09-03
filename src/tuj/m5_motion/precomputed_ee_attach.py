"""Validated playback of precomputed initial bare-flange EE trajectories.

Initial rack attachment is a fixed workcell operation.  This module stores the
successful, time-parameterized joint trajectory separately from a request-bound
``MotionPlan`` and binds it to the current request only after fail-closed state,
workcell, collision, and dynamics validation.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tuj.m5_motion.schema import (
    ArtifactProvenance,
    CollisionContext,
    EventType,
    InterpolationType,
    ModuleName,
    MotionPlan,
    MotionPlanRequest,
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


EE_ATTACH_TEMPLATE_SCHEMA_VERSION = "1.0"
SUPPORTED_EE_IDS = ("2F", "3F", "vac")


def normalize_ee_id(value: object) -> str:
    """Normalize the supported rack EE spellings to their runtime ids."""

    normalized = str(value or "").strip().casefold()
    aliases = {
        "2f": "2F",
        "3f": "3F",
        "vac": "vac",
        "vacuum": "vac",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported target EE {value!r}") from error


def is_initial_ee_attach(request: MotionPlanRequest) -> bool:
    raw_from_ee = request.task.metadata.get("from_ee")
    physical_active_ee = request.world.metadata.get("physical_active_ee")
    return (
        task_operation(request.task) == "EE_ATTACH"
        and raw_from_ee in {None, ""}
        and physical_active_ee is None
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _robot_model(world: WorldSnapshot) -> str:
    declared = world.metadata.get("robot_model")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    robot_id = world.robot_state.robot_id.casefold()
    if "ur5e" in robot_id:
        return "UR5e"
    return world.robot_state.robot_id


def compute_rack_signature(world: WorldSnapshot) -> str:
    """Hash only static rack and dock geometry, excluding scene objects."""

    return _digest(world.rack)


def compute_workcell_signature(
    world: WorldSnapshot,
    collision_contexts: Mapping[str, CollisionContext],
) -> str:
    """Build the static compatibility hash used before trajectory playback.

    Dynamic objects are deliberately absent.  They are validated against the
    dense trajectory by the current request's collision registry instead.
    """

    model_versions = {
        context_id: context.collision_model_version
        for context_id, context in sorted(collision_contexts.items())
    }
    eef_pose = world.robot_state.eef_pose
    canonical_start_eef_pose = (
        {
            "frame_id": eef_pose.frame_id,
            "position_m": [round(value, 8) for value in eef_pose.position_m],
            "orientation_xyzw": [
                round(value, 8) for value in eef_pose.orientation_xyzw
            ],
        }
        if eef_pose is not None
        else None
    )
    payload = {
        "environment_name": world.metadata.get("environment_name"),
        "robot_model": _robot_model(world),
        "robot_id": world.robot_state.robot_id,
        "joint_names": world.robot_state.joint_names,
        # At the canonical joint state the world-frame EEF pose is a compact,
        # dynamic-object-independent check that the UR5e base pose is unchanged.
        "canonical_start_eef_pose": canonical_start_eef_pose,
        "rack": world.rack,
        "source_revision": world.metadata.get("source_revision"),
        "rack_collision_policy": world.metadata.get("rack_collision_policy"),
        "collision_model_versions": model_versions,
    }
    return _digest(payload)


class _TemplateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EEAttachTrajectorySegmentTemplate(_TemplateModel):
    segment_id: str = Field(min_length=1)
    segment_type: SegmentType
    collision_context_before: str = Field(min_length=1)
    collision_context_after: str = Field(min_length=1)
    interpolation: InterpolationType = InterpolationType.QUINTIC
    waypoints: list[TrajectoryWaypoint] = Field(min_length=2)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_waypoint_timeline(self) -> "EEAttachTrajectorySegmentTemplate":
        times = [item.time_from_start_s for item in self.waypoints]
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("segment waypoint times must be strictly increasing")
        return self


class EEAttachTrajectoryEventTemplate(_TemplateModel):
    time_from_start_s: float = Field(ge=0)
    event_type: EventType
    target_id: str | None = None
    command: float | bool | str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class EEAttachTrajectoryTemplate(_TemplateModel):
    """Request-independent, executed joint trajectory for bare -> EE attach."""

    schema_version: Literal["1.0"] = EE_ATTACH_TEMPLATE_SCHEMA_VERSION
    trajectory_id: str = Field(min_length=1)
    environment_name: str = Field(min_length=1)
    robot_model: str = Field(min_length=1)
    source_active_ee: None = None
    target_active_ee: str = Field(min_length=1)
    joint_names: list[str] = Field(min_length=1)
    start_joint_positions_rad: list[float] = Field(min_length=1)
    start_eef_pose: Pose | None = None
    workcell_signature: str = Field(min_length=1)
    rack_signature: str = Field(min_length=1)
    collision_model_versions: dict[str, str] = Field(min_length=1)
    segments: list[EEAttachTrajectorySegmentTemplate] = Field(min_length=1)
    events: list[EEAttachTrajectoryEventTemplate] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_active_ee", mode="before")
    @classmethod
    def _normalize_target(cls, value: object) -> str:
        return normalize_ee_id(value)

    @model_validator(mode="after")
    def _validate_trajectory(self) -> "EEAttachTrajectoryTemplate":
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

    @classmethod
    def from_motion_plan(
        cls,
        request: MotionPlanRequest,
        plan: MotionPlan,
        *,
        collision_contexts: Mapping[str, CollisionContext] | None = None,
        trajectory_id: str | None = None,
    ) -> "EEAttachTrajectoryTemplate":
        """Strip request identity from a successfully executed attach plan."""

        if not is_initial_ee_attach(request):
            raise ValueError("only an initial bare -> EE attach can be exported")
        if plan.request_id != request.request_id:
            raise ValueError("MotionPlan does not belong to the request")
        target = normalize_ee_id(
            request.task.metadata.get("to_ee") or request.task.ee
        )
        segments: list[EEAttachTrajectorySegmentTemplate] = []
        context_map: dict[str, CollisionContext] = dict(collision_contexts or {})
        for segment in plan.segments:
            before = segment.collision_context_before
            after = segment.collision_context_after or before
            if before is None or after is None:
                raise ValueError("exported attach segments require collision contexts")
            context_map[before.context_id] = before
            context_map[after.context_id] = after
            segments.append(
                EEAttachTrajectorySegmentTemplate(
                    segment_id=segment.segment_id,
                    segment_type=segment.segment_type,
                    collision_context_before=before.context_id,
                    collision_context_after=after.context_id,
                    interpolation=segment.interpolation,
                    waypoints=[item.model_copy(deep=True) for item in segment.waypoints],
                    metadata=dict(segment.metadata),
                )
            )
        model_versions = {
            context_id: context.collision_model_version
            for context_id, context in sorted(context_map.items())
        }
        fingerprint = _digest(
            {
                "joint_names": plan.joint_names,
                "segments": [item.model_dump(mode="json") for item in segments],
                "events": [item.model_dump(mode="json") for item in plan.events],
            }
        )
        return cls(
            trajectory_id=trajectory_id or (
                f"ur5e-bare-to-{target}-{fingerprint[:12]}"
            ),
            environment_name=str(request.world.metadata.get("environment_name") or ""),
            robot_model=_robot_model(request.world),
            target_active_ee=target,
            joint_names=list(plan.joint_names),
            start_joint_positions_rad=list(
                request.world.robot_state.joint_positions_rad
            ),
            start_eef_pose=(
                request.world.robot_state.eef_pose.model_copy(deep=True)
                if request.world.robot_state.eef_pose is not None
                else None
            ),
            workcell_signature=compute_workcell_signature(
                request.world, context_map
            ),
            rack_signature=compute_rack_signature(request.world),
            collision_model_versions=model_versions,
            segments=segments,
            events=[
                EEAttachTrajectoryEventTemplate(
                    time_from_start_s=event.time_from_start_s,
                    event_type=event.event_type,
                    target_id=event.target_id,
                    command=event.command,
                    parameters=dict(event.parameters),
                )
                for event in plan.events
            ],
            metadata={"source_plan_fingerprint": fingerprint},
        )


class EEAttachPathFailureCode(str, enum.Enum):
    PRECOMPUTED_EE_PATH_NOT_FOUND = "PRECOMPUTED_EE_PATH_NOT_FOUND"
    PRECOMPUTED_EE_PATH_STALE = "PRECOMPUTED_EE_PATH_STALE"
    WORKCELL_SIGNATURE_MISMATCH = "WORKCELL_SIGNATURE_MISMATCH"
    START_STATE_MISMATCH = "START_STATE_MISMATCH"
    PRECOMPUTED_PATH_COLLISION = "PRECOMPUTED_PATH_COLLISION"
    PRECOMPUTED_PATH_DYNAMICS_INVALID = "PRECOMPUTED_PATH_DYNAMICS_INVALID"
    LOCK_EVENT_MISSING = "LOCK_EVENT_MISSING"
    RELEASE_EVENT_MISSING = "RELEASE_EVENT_MISSING"
    TRANSITION_SEAM_MISMATCH = "TRANSITION_SEAM_MISMATCH"
    FINAL_EE_STATE_INVALID = "FINAL_EE_STATE_INVALID"


class PrecomputedEEPathError(RuntimeError):
    def __init__(
        self,
        failure_code: EEAttachPathFailureCode,
        detail: str,
        *,
        trajectory_id: str | None = None,
    ) -> None:
        self.failure_code = failure_code
        self.detail = detail
        self.trajectory_id = trajectory_id
        super().__init__(f"{failure_code.value}: {detail}")


class EEAttachPolicy(str, enum.Enum):
    PRECOMPUTED_REQUIRED = "precomputed-required"
    PRECOMPUTED_OR_PLAN = "precomputed-or-plan"


class PrecomputedEEAttachRegistry:
    """Load environment-scoped bare -> EE templates with optional overrides."""

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
            key = (template.environment_name, template.target_active_ee)
            if key in self._overrides:
                raise ValueError(f"duplicate EE attach trajectory override for {key}")
            self._overrides[key] = path

    @staticmethod
    def _load_file(path: Path) -> EEAttachTrajectoryTemplate:
        try:
            return EEAttachTrajectoryTemplate.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except FileNotFoundError as error:
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_NOT_FOUND,
                f"trajectory file not found: {path}",
            ) from error
        except Exception as error:  # noqa: BLE001 - normalize artifact failures
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                f"invalid trajectory template {path}: {error}",
            ) from error

    def load(self, environment_name: str, target_ee: object) -> EEAttachTrajectoryTemplate:
        try:
            target = normalize_ee_id(target_ee)
        except ValueError as error:
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_NOT_FOUND,
                str(error),
            ) from error
        path = self._overrides.get((environment_name, target))
        if path is None:
            path = self.root / environment_name / f"bare_to_{target}.json"
        template = self._load_file(path)
        if (
            template.environment_name != environment_name
            or template.target_active_ee != target
        ):
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                "trajectory metadata does not match the registry lookup",
                trajectory_id=template.trajectory_id,
            )
        return template


class CollisionChecker(Protocol):
    def check(
        self,
        joint_config: Sequence[float],
        keyframe: object | None = None,
        *,
        context: CollisionContext | None = None,
        context_id: str | None = None,
    ) -> Any: ...


def _dense_joint_path(
    waypoints: Sequence[TrajectoryWaypoint],
    max_joint_step_rad: float,
) -> list[list[float]]:
    dense: list[list[float]] = []
    for left, right in zip(waypoints, waypoints[1:]):
        start = left.joint_positions_rad
        finish = right.joint_positions_rad
        maximum = max(abs(b - a) for a, b in zip(start, finish))
        steps = max(1, math.ceil(maximum / max_joint_step_rad))
        if not dense:
            dense.append(list(start))
        dense.extend(
            [a + (b - a) * index / steps for a, b in zip(start, finish)]
            for index in range(1, steps + 1)
        )
    return dense


LogSink = Callable[[str], None]


class PrecomputedEEAttachPlanner:
    """Validate and bind a complete template without invoking IK or RRT."""

    def __init__(
        self,
        registry: PrecomputedEEAttachRegistry,
        *,
        start_tolerance_rad: float = 0.01,
        joint_position_limits_rad: Sequence[tuple[float, float]] | None = None,
        log: LogSink = print,
    ) -> None:
        if not math.isfinite(start_tolerance_rad) or start_tolerance_rad < 0:
            raise ValueError("start_tolerance_rad must be finite and non-negative")
        self.registry = registry
        self.start_tolerance_rad = start_tolerance_rad
        self.joint_position_limits_rad = (
            tuple(joint_position_limits_rad)
            if joint_position_limits_rad is not None
            else None
        )
        self._log = log

    @staticmethod
    def _failure(
        code: EEAttachPathFailureCode,
        detail: str,
        template: EEAttachTrajectoryTemplate | None = None,
    ) -> PrecomputedEEPathError:
        return PrecomputedEEPathError(
            code,
            detail,
            trajectory_id=(template.trajectory_id if template is not None else None),
        )

    def _validate_request_state(
        self,
        request: MotionPlanRequest,
        template: EEAttachTrajectoryTemplate,
    ) -> None:
        state = request.world.robot_state
        if not is_initial_ee_attach(request):
            raise self._failure(
                EEAttachPathFailureCode.START_STATE_MISMATCH,
                "request is not a bare-flange initial EE_ATTACH",
                template,
            )
        if state.attached_object_id is not None or state.held_tool_id is not None:
            raise self._failure(
                EEAttachPathFailureCode.START_STATE_MISMATCH,
                "initial attach requires no attached object and no held tool",
                template,
            )
        if state.joint_names != template.joint_names:
            raise self._failure(
                EEAttachPathFailureCode.START_STATE_MISMATCH,
                "joint name or order differs from the stored trajectory",
                template,
            )
        maximum_error = max(
            abs(current - stored)
            for current, stored in zip(
                state.joint_positions_rad,
                template.start_joint_positions_rad,
            )
        )
        if maximum_error > self.start_tolerance_rad:
            raise self._failure(
                EEAttachPathFailureCode.START_STATE_MISMATCH,
                f"maximum start joint error {maximum_error:.6f} rad exceeds "
                f"{self.start_tolerance_rad:.6f} rad",
                template,
            )

    def _validate_workcell(
        self,
        request: MotionPlanRequest,
        template: EEAttachTrajectoryTemplate,
        collision_contexts: Mapping[str, CollisionContext],
    ) -> None:
        if template.robot_model != _robot_model(request.world):
            raise self._failure(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                "robot model differs from the stored trajectory",
                template,
            )
        if template.rack_signature != compute_rack_signature(request.world):
            raise self._failure(
                EEAttachPathFailureCode.WORKCELL_SIGNATURE_MISMATCH,
                "rack/dock signature differs from the stored trajectory",
                template,
            )
        current_versions = {
            context_id: context.collision_model_version
            for context_id, context in collision_contexts.items()
        }
        if template.collision_model_versions != current_versions:
            raise self._failure(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                "collision model versions differ from the stored trajectory",
                template,
            )
        current_signature = compute_workcell_signature(
            request.world, collision_contexts
        )
        if template.workcell_signature != current_signature:
            raise self._failure(
                EEAttachPathFailureCode.WORKCELL_SIGNATURE_MISMATCH,
                "static workcell signature differs from the stored trajectory",
                template,
            )

    def _validate_events_and_final_state(
        self,
        template: EEAttachTrajectoryTemplate,
        collision_contexts: Mapping[str, CollisionContext],
    ) -> None:
        target = template.target_active_ee
        lock_events = [
            event for event in template.events if event.event_type is EventType.TOOL_LOCK
        ]
        verify_events = [
            event
            for event in template.events
            if event.event_type is EventType.VERIFY_TOOL_LOCK
        ]
        if len(lock_events) != 1 or len(verify_events) != 1:
            raise self._failure(
                EEAttachPathFailureCode.LOCK_EVENT_MISSING,
                "trajectory requires exactly one TOOL_LOCK and VERIFY_TOOL_LOCK",
                template,
            )
        try:
            targets = {
                normalize_ee_id(lock_events[0].target_id),
                normalize_ee_id(verify_events[0].target_id),
            }
        except ValueError as error:
            raise self._failure(
                EEAttachPathFailureCode.LOCK_EVENT_MISSING,
                str(error),
                template,
            ) from error
        if targets != {target} or (
            verify_events[0].time_from_start_s < lock_events[0].time_from_start_s
        ):
            raise self._failure(
                EEAttachPathFailureCode.LOCK_EVENT_MISSING,
                "lock/verify target or ordering is invalid",
                template,
            )
        final_context_id = template.segments[-1].collision_context_after
        final_context = collision_contexts.get(final_context_id)
        if final_context is None or final_context.active_ee != target:
            raise self._failure(
                EEAttachPathFailureCode.FINAL_EE_STATE_INVALID,
                "final segment does not leave the target EE physically attached",
                template,
            )

    def _validate_dynamics(
        self,
        request: MotionPlanRequest,
        template: EEAttachTrajectoryTemplate,
    ) -> None:
        limits = request.constraints.joint_limits
        if set(limits) != set(template.joint_names):
            raise self._failure(
                EEAttachPathFailureCode.PRECOMPUTED_PATH_DYNAMICS_INVALID,
                "request joint dynamics do not cover the stored trajectory",
                template,
            )
        position_limits = self.joint_position_limits_rad
        for segment in template.segments:
            for waypoint in segment.waypoints:
                if (
                    waypoint.joint_velocities_rad_s is None
                    or waypoint.joint_accelerations_rad_s2 is None
                ):
                    raise self._failure(
                        EEAttachPathFailureCode.PRECOMPUTED_PATH_DYNAMICS_INVALID,
                        "stored waypoints require velocity and acceleration commands",
                        template,
                    )
                for index, name in enumerate(template.joint_names):
                    position = waypoint.joint_positions_rad[index]
                    if position_limits is not None:
                        lower, upper = position_limits[index]
                        if position < lower - 1e-9 or position > upper + 1e-9:
                            raise self._failure(
                                EEAttachPathFailureCode.PRECOMPUTED_PATH_DYNAMICS_INVALID,
                                f"joint {name} position exceeds its limit",
                                template,
                            )
                    limit = limits[name]
                    if abs(waypoint.joint_velocities_rad_s[index]) > (
                        limit.max_velocity_rad_s
                        * request.constraints.velocity_scaling
                        + 1e-9
                    ):
                        raise self._failure(
                            EEAttachPathFailureCode.PRECOMPUTED_PATH_DYNAMICS_INVALID,
                            f"joint {name} velocity exceeds the request limit",
                            template,
                        )
                    if abs(waypoint.joint_accelerations_rad_s2[index]) > (
                        limit.max_acceleration_rad_s2
                        * request.constraints.acceleration_scaling
                        + 1e-9
                    ):
                        raise self._failure(
                            EEAttachPathFailureCode.PRECOMPUTED_PATH_DYNAMICS_INVALID,
                            f"joint {name} acceleration exceeds the request limit",
                            template,
                        )
            for left, right in zip(segment.waypoints, segment.waypoints[1:]):
                dt = right.time_from_start_s - left.time_from_start_s
                for index, name in enumerate(template.joint_names):
                    limit = limits[name].max_jerk_rad_s3
                    if limit is None:
                        continue
                    jerk = abs(
                        right.joint_accelerations_rad_s2[index]
                        - left.joint_accelerations_rad_s2[index]
                    ) / dt
                    if jerk > limit * request.constraints.jerk_scaling + 1e-9:
                        raise self._failure(
                            EEAttachPathFailureCode.PRECOMPUTED_PATH_DYNAMICS_INVALID,
                            f"joint {name} jerk exceeds the request limit",
                            template,
                        )

    def _validate_collisions(
        self,
        request: MotionPlanRequest,
        template: EEAttachTrajectoryTemplate,
        collision_contexts: Mapping[str, CollisionContext],
        collision_checker: CollisionChecker,
    ) -> dict[str, float | None]:
        clearances: dict[str, float | None] = {}
        for segment in template.segments:
            context = collision_contexts.get(segment.collision_context_before)
            after = collision_contexts.get(segment.collision_context_after)
            if context is None or after is None:
                raise self._failure(
                    EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                    f"unknown collision context in segment {segment.segment_id!r}",
                    template,
                )
            minimum: float | None = None
            for state_index, joint_config in enumerate(
                _dense_joint_path(
                    segment.waypoints,
                    request.constraints.max_joint_path_step_rad,
                )
            ):
                result = collision_checker.check(joint_config, context=context)
                valid = bool(getattr(result, "valid", result))
                clearance = getattr(result, "min_clearance_m", None)
                if clearance is not None:
                    minimum = (
                        float(clearance)
                        if minimum is None
                        else min(minimum, float(clearance))
                    )
                if not valid:
                    failure_code = getattr(result, "failure_code", "COLLISION")
                    detail = getattr(result, "detail", "stored path is invalid")
                    raise self._failure(
                        EEAttachPathFailureCode.PRECOMPUTED_PATH_COLLISION,
                        f"segment {segment.segment_id!r}, interpolated state "
                        f"{state_index}: {failure_code}: {detail}",
                        template,
                    )
            clearances[segment.segment_id] = minimum
        return clearances

    def load(self, request: MotionPlanRequest) -> EEAttachTrajectoryTemplate:
        """Resolve the request artifact before compiling collision variants."""

        environment_name = request.world.metadata.get("environment_name")
        if not isinstance(environment_name, str) or not environment_name:
            raise self._failure(
                EEAttachPathFailureCode.WORKCELL_SIGNATURE_MISMATCH,
                "request has no environment_name",
            )
        raw_target = request.task.metadata.get("to_ee") or request.task.ee
        return self.registry.load(environment_name, raw_target)

    def plan(
        self,
        request: MotionPlanRequest,
        *,
        collision_contexts: Mapping[str, CollisionContext],
        collision_checker: CollisionChecker,
        template: EEAttachTrajectoryTemplate | None = None,
    ) -> MotionPlan:
        raw_target = request.task.metadata.get("to_ee") or request.task.ee
        template = template or self.load(request)
        requested_target = normalize_ee_id(raw_target)
        if template.target_active_ee != requested_target:
            raise self._failure(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                "stored target EE differs from the request",
                template,
        )
        self._validate_request_state(request, template)
        self._validate_workcell(request, template, collision_contexts)
        self._validate_events_and_final_state(template, collision_contexts)
        self._validate_dynamics(request, template)
        clearances = self._validate_collisions(
            request,
            template,
            collision_contexts,
            collision_checker,
        )
        self._log(f"[M5][EE_PATH] hit: {template.trajectory_id}")
        self._log("[M5][EE_PATH] start_state=PASS")
        self._log("[M5][EE_PATH] workcell_signature=PASS")
        self._log("[M5][EE_PATH] collision_validation=PASS")

        digest = _digest(
            {
                "request_id": request.request_id,
                "scene_signature": request.world.scene.signature,
                "trajectory_id": template.trajectory_id,
            }
        )[:24]
        segments = [
            TrajectorySegment(
                segment_id=(
                    f"{request.task.subgoal_id}:precomputed:{index}:"
                    f"{segment.segment_id}"
                ),
                segment_type=segment.segment_type,
                start_time_s=segment.waypoints[0].time_from_start_s,
                end_time_s=segment.waypoints[-1].time_from_start_s,
                interpolation=segment.interpolation,
                waypoints=[item.model_copy(deep=True) for item in segment.waypoints],
                collision_checked=True,
                min_clearance_m=clearances[segment.segment_id],
                collision_context_before=collision_contexts[
                    segment.collision_context_before
                ],
                collision_context_after=collision_contexts[
                    segment.collision_context_after
                ],
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
        events = [
            TrajectoryEvent(
                event_id=(
                    f"{request.task.subgoal_id}:precomputed:event:{index}:"
                    f"{event.event_type.value}"
                ),
                time_from_start_s=event.time_from_start_s,
                event_type=event.event_type,
                target_id=requested_target,
                command=event.command,
                parameters=dict(event.parameters),
            )
            for index, event in enumerate(template.events)
        ]
        final_waypoint = segments[-1].waypoints[-1]
        initial_state = request.world.robot_state
        plan = MotionPlan(
            plan_id=f"motion-plan:{digest}",
            request_id=request.request_id,
            provenance=ArtifactProvenance(
                artifact_id=f"motion-plan-artifact:{digest}",
                artifact_type="MotionPlan",
                produced_by=ModuleName.MOTION_PLANNER,
                invocation_id=f"precomputed-ee-attach:{request.request_id}",
                input_artifact_ids=[request.provenance.artifact_id],
                metadata={
                    "source": "precomputed",
                    "trajectory_id": template.trajectory_id,
                },
            ),
            scene_signature=request.world.scene.signature,
            robot_id=initial_state.robot_id,
            joint_names=list(template.joint_names),
            duration_s=segments[-1].end_time_s,
            segments=segments,
            events=events,
            expected_final_state=RobotState(
                robot_id=initial_state.robot_id,
                joint_names=list(template.joint_names),
                joint_positions_rad=list(final_waypoint.joint_positions_rad),
                joint_velocities_rad_s=(
                    list(final_waypoint.joint_velocities_rad_s)
                    if final_waypoint.joint_velocities_rad_s is not None
                    else [0.0] * len(template.joint_names)
                ),
                eef_pose=(
                    final_waypoint.eef_pose.model_copy(deep=True)
                    if final_waypoint.eef_pose is not None
                    else None
                ),
                gripper=(
                    initial_state.gripper.model_copy(deep=True)
                    if initial_state.gripper is not None
                    else None
                ),
                attached_object_id=None,
                held_tool_id=None,
            ),
            metadata={
                "source": "precomputed",
                "trajectory_id": template.trajectory_id,
                "selection_policy": "PRECOMPUTED_EE_ATTACH",
                "target_active_ee": requested_target,
                "start_state_validation": "PASS",
                "workcell_signature_validation": "PASS",
                "collision_validation": "PASS",
                "dynamic_planner_invoked": False,
            },
        )
        self._log("[M5][EE_PATH] source=precomputed")
        return plan


def save_ee_attach_template(
    path: str | Path,
    request: MotionPlanRequest,
    plan: MotionPlan,
    *,
    collision_contexts: Mapping[str, CollisionContext] | None = None,
    trajectory_id: str | None = None,
) -> EEAttachTrajectoryTemplate:
    """Export a verified plan atomically after controller validation succeeds."""

    template = EEAttachTrajectoryTemplate.from_motion_plan(
        request,
        plan,
        collision_contexts=collision_contexts,
        trajectory_id=trajectory_id,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(template.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return template


__all__ = [
    "EEAttachPathFailureCode",
    "EEAttachPolicy",
    "EEAttachTrajectoryEventTemplate",
    "EEAttachTrajectorySegmentTemplate",
    "EEAttachTrajectoryTemplate",
    "PrecomputedEEAttachPlanner",
    "PrecomputedEEAttachRegistry",
    "PrecomputedEEPathError",
    "SUPPORTED_EE_IDS",
    "compute_rack_signature",
    "compute_workcell_signature",
    "is_initial_ee_attach",
    "normalize_ee_id",
    "save_ee_attach_template",
]
