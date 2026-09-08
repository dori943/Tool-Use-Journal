"""Reusable contact-friction grasp planning, control, and validation.

This module deliberately contains no task id or object-shape assumptions.  A
geometry provider remains responsible for proposing useful grasp poses; this
layer changes the execution contract from a synthetic attachment to persistent
gripper contact and requires physical evidence before a tool is marked held.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import mujoco
import numpy as np

from tuj.m5_motion.profiles import PhysicalGraspProfile
from tuj.m5_motion.schema import (
    ExecutionReport,
    ExecutionStatus,
    KeyframeEventType,
    KeyframePlanArtifact,
    KeyframePlanCandidate,
    KeyframeType,
    MotionPlanRequest,
    SimulationRun,
)
from tuj.m5_motion.task_semantics import is_acquire_task, is_release_task
from tuj.m5_motion.tool_use_journal_runtime import (
    ToolUseJournalControllerTrajectoryPlayer,
    ToolUseJournalEERuntime,
)


class GraspExecutionMode(str, Enum):
    KINEMATIC = "KINEMATIC"
    CONTACT_FRICTION = "CONTACT_FRICTION"


def normalized_grasp_execution_mode(value: object) -> GraspExecutionMode:
    normalized = str(getattr(value, "value", value) or "KINEMATIC")
    normalized = normalized.strip().replace("-", "_").upper()
    return GraspExecutionMode(normalized)


def uses_contact_friction(request: MotionPlanRequest) -> bool:
    if not is_acquire_task(request.task):
        return False
    try:
        mode = normalized_grasp_execution_mode(
            request.task.metadata.get("grasp_execution_mode")
        )
    except ValueError:
        return False
    return mode is GraspExecutionMode.CONTACT_FRICTION


def releases_contact_friction(request: MotionPlanRequest) -> bool:
    if not is_release_task(request.task):
        return False
    target = (
        request.task.goal.target_object_id
        or request.task.tool
        or next(iter(request.task.target_ids), None)
    )
    held = request.world.robot_state.held_tool_id
    transforms = request.world.metadata.get("contact_friction_held_objects")
    return (
        target is not None
        and held == target
        and isinstance(transforms, Mapping)
        and target in transforms
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _matrix_quaternion_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            )
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                (
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                )
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                )
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                )
            )
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def with_contact_friction_grasp(
    raw: KeyframePlanArtifact,
    *,
    profile: PhysicalGraspProfile,
) -> KeyframePlanArtifact:
    """Replace synthetic attachment events with a persistent physical close."""

    bound = raw.model_copy(deep=True)
    valid_candidates: list[KeyframePlanCandidate] = []
    rejected_candidates: list[dict[str, str]] = []
    for candidate in bound.candidates:
        if candidate.metadata.get("support_collision_risk") is True:
            reasons = candidate.metadata.get("support_collision_reasons", [])
            detail = (
                ",".join(str(reason) for reason in reasons)
                if isinstance(reasons, Sequence)
                and not isinstance(reasons, (str, bytes))
                else str(reasons)
            )
            rejected_candidates.append(
                {
                    "strategy_id": candidate.strategy_id,
                    "reason": (
                        "SUPPORT_COLLISION_RISK"
                        + (f":{detail}" if detail else "")
                    ),
                }
            )
            continue
        grasp = next(
            (
                keyframe
                for keyframe in candidate.keyframes
                if keyframe.keyframe_type is KeyframeType.GRASP
            ),
            None,
        )
        lift = next(
            (
                keyframe
                for keyframe in reversed(candidate.keyframes)
                if keyframe.keyframe_type is KeyframeType.LIFT
            ),
            None,
        )
        if grasp is None or lift is None:
            rejected_candidates.append(
                {
                    "strategy_id": candidate.strategy_id,
                    "reason": (
                        "MISSING_GRASP_KEYFRAME"
                        if grasp is None
                        else "MISSING_LIFT_KEYFRAME"
                    ),
                }
            )
            continue

        grasp.offset_along_approach_m += profile.clearance_reserve_m
        grasp.events_after = [
            event
            for event in grasp.events_after
            if event is not KeyframeEventType.ATTACH_OBJECT
        ]
        if KeyframeEventType.GRIPPER_CLOSE not in grasp.events_after:
            grasp.events_after.insert(0, KeyframeEventType.GRIPPER_CLOSE)
        raw_parameters = grasp.metadata.get("event_parameters", {})
        event_parameters = (
            dict(raw_parameters) if isinstance(raw_parameters, Mapping) else {}
        )
        event_parameters.pop(KeyframeEventType.ATTACH_OBJECT.value, None)
        event_parameters[KeyframeEventType.GRIPPER_CLOSE.value] = {
            "command": profile.force.close_command
        }
        grasp.metadata = {
            **grasp.metadata,
            "grasp_execution_mode": GraspExecutionMode.CONTACT_FRICTION.value,
            "hold_duration_after_s": profile.grasp_hold_duration_s,
            "event_time_offsets_s": {
                KeyframeEventType.GRIPPER_CLOSE.value: 0.0
            },
            "event_parameters": event_parameters,
        }
        lift.metadata = {
            **lift.metadata,
            "hold_duration_after_s": profile.lift_hold_duration_s,
            "physical_retention_hold": True,
        }
        valid_candidates.append(candidate)

    if not valid_candidates:
        details = ", ".join(
            f"{item['strategy_id']}={item['reason']}"
            for item in rejected_candidates
        )
        raise ValueError(
            "contact-friction PICK has no feasible GRASP/LIFT candidate: "
            + details
        )
    bound.candidates = valid_candidates
    digest = hashlib.sha256(
        _canonical_json(
            {
                "source_artifact_id": raw.artifact_id,
                "profile": profile,
                "accepted_strategy_ids": [
                    candidate.strategy_id for candidate in valid_candidates
                ],
                "rejected_candidates": rejected_candidates,
            }
        ).encode("utf-8")
    ).hexdigest()[:16]
    bound.artifact_id = f"{raw.artifact_id}:contact-friction:{digest}"
    bound.provenance = raw.provenance.model_copy(
        update={
            "artifact_id": (
                f"{raw.provenance.artifact_id}:contact-friction:{digest}"
            ),
            "metadata": {
                **raw.provenance.metadata,
                "grasp_execution_mode": (
                    GraspExecutionMode.CONTACT_FRICTION.value
                ),
                "rejected_candidates": rejected_candidates,
            },
        }
    )
    return bound


def with_contact_friction_release(
    raw: KeyframePlanArtifact,
) -> KeyframePlanArtifact:
    """Open the gripper instead of detaching a non-existent synthetic weld."""

    bound = raw.model_copy(deep=True)
    changed = False
    for candidate in bound.candidates:
        for keyframe in candidate.keyframes:
            if KeyframeEventType.DETACH_OBJECT not in keyframe.events_after:
                continue
            replacement_events: list[KeyframeEventType] = []
            for event in keyframe.events_after:
                replacement = (
                    KeyframeEventType.GRIPPER_OPEN
                    if event is KeyframeEventType.DETACH_OBJECT
                    else event
                )
                if replacement not in replacement_events:
                    replacement_events.append(replacement)
            keyframe.events_after = replacement_events
            raw_parameters = keyframe.metadata.get("event_parameters", {})
            event_parameters = (
                dict(raw_parameters)
                if isinstance(raw_parameters, Mapping)
                else {}
            )
            event_parameters.pop(KeyframeEventType.DETACH_OBJECT.value, None)
            event_parameters[KeyframeEventType.GRIPPER_OPEN.value] = {
                "command": -1.0
            }
            keyframe.metadata = {
                **keyframe.metadata,
                "grasp_execution_mode": (
                    GraspExecutionMode.CONTACT_FRICTION.value
                ),
                "event_parameters": event_parameters,
            }
            changed = True
    if not changed:
        raise ValueError(
            "contact-friction release requires a DETACH_OBJECT keyframe"
        )
    digest = hashlib.sha256(
        _canonical_json({"source_artifact_id": raw.artifact_id}).encode("utf-8")
    ).hexdigest()[:16]
    bound.artifact_id = f"{raw.artifact_id}:contact-friction-release:{digest}"
    bound.provenance = raw.provenance.model_copy(
        update={
            "artifact_id": (
                f"{raw.provenance.artifact_id}:contact-friction-release:{digest}"
            ),
            "metadata": {
                **raw.provenance.metadata,
                "grasp_execution_mode": (
                    GraspExecutionMode.CONTACT_FRICTION.value
                ),
            },
        }
    )
    return bound


class KeyframeProvider(Protocol):
    def generate(self, request: MotionPlanRequest) -> KeyframePlanArtifact: ...


class RetargetedAcquireKeyframeProvider:
    """Reuse one relative acquire artifact against the current grounded tool.

    Only symbolic object-frame references and event targets are changed.  The
    current request still supplies the scene, collision models, IK, and final
    validation, so a frozen proposal cannot bypass deterministic planning.
    """

    def __init__(self, artifact: KeyframePlanArtifact) -> None:
        self._artifact = artifact.model_copy(deep=True)

    def generate(self, request: MotionPlanRequest) -> KeyframePlanArtifact:
        if not is_acquire_task(request.task):
            raise ValueError(
                "retargeted acquire keyframes can only serve a PICK request"
            )
        target = (
            request.task.goal.target_object_id
            or request.task.tool
            or next(iter(request.task.target_ids), None)
        )
        if target is None:
            raise ValueError("PICK request has no grounded target object")

        source_frames = {
            keyframe.frame_ref
            for candidate in self._artifact.candidates
            for keyframe in candidate.keyframes
            if keyframe.frame_ref.startswith("object:")
        }
        if len(source_frames) != 1:
            raise ValueError(
                "reusable PICK artifact must reference exactly one object frame"
            )
        source_frame = next(iter(source_frames))
        target_frame = f"object:{target}"
        bound = self._artifact.model_copy(deep=True)
        for candidate in bound.candidates:
            for keyframe in candidate.keyframes:
                if keyframe.frame_ref == source_frame:
                    keyframe.frame_ref = target_frame
                if (
                    keyframe.events_after
                    or keyframe.keyframe_type
                    in {KeyframeType.GRASP, KeyframeType.LIFT}
                ):
                    keyframe.metadata = {
                        **keyframe.metadata,
                        "event_target_id": str(target),
                    }

        digest = hashlib.sha256(
            _canonical_json(
                {
                    "source_artifact_id": self._artifact.artifact_id,
                    "request_id": request.request_id,
                    "scene_signature": request.world.scene.signature,
                    "target": target,
                }
            ).encode("utf-8")
        ).hexdigest()[:16]
        bound.artifact_id = f"keyframe-plan:retargeted-acquire:{digest}"
        bound.scene_signature = request.world.scene.signature
        bound.subgoal_id = request.task.subgoal_id
        bound.provenance = self._artifact.provenance.model_copy(
            update={
                "artifact_id": (
                    f"keyframe-plan-artifact:retargeted-acquire:{digest}"
                ),
                "invocation_id": f"retargeted-acquire:{request.request_id}",
                "input_artifact_ids": [
                    self._artifact.provenance.artifact_id,
                    request.provenance.artifact_id,
                ],
                "metadata": {
                    **self._artifact.provenance.metadata,
                    "binding": "RETARGETED_RELATIVE_ACQUIRE_V1",
                    "source_artifact_id": self._artifact.artifact_id,
                    "source_object_frame": source_frame,
                    "target_object_frame": target_frame,
                },
            }
        )
        return bound


class ContactFrictionKeyframeProvider:
    """Decorate only requests explicitly selecting physical friction grasping."""

    def __init__(self, provider: KeyframeProvider) -> None:
        self._provider = provider

    def generate(self, request: MotionPlanRequest) -> KeyframePlanArtifact:
        artifact = self._provider.generate(request)
        if releases_contact_friction(request):
            return with_contact_friction_release(artifact)
        if not uses_contact_friction(request):
            return artifact
        from tuj.m5_motion.grasp_geometry import bind_grasp_geometry

        artifact = bind_grasp_geometry(artifact, request)
        raw_profile = request.task.metadata.get("grasp_profile")
        profile = PhysicalGraspProfile.from_mapping(
            raw_profile if isinstance(raw_profile, Mapping) else None
        )
        return with_contact_friction_grasp(artifact, profile=profile)


@dataclass(frozen=True, slots=True)
class RuntimeGraspParameters:
    maximum_grip_force_n: float
    sliding_friction: float
    retention_target_normal_force_n: float
    retention_model: str = "WEIGHT"
    retention_control_mode: str = "FORCE_TARGET"


def _runtime_grasp_parameters(
    runtime: ToolUseJournalEERuntime,
    object_id: str,
    profile: PhysicalGraspProfile,
) -> RuntimeGraspParameters:
    maximum_force = profile.force.fallback_maximum_grip_force_n
    sliding_friction = profile.force.fallback_sliding_friction
    mass_kg: float | None = None
    object_minimum_retention_force_n = 0.0
    retention_control_mode = "FORCE_TARGET"

    ee_pool = getattr(runtime.env, "robot_spec", {}).get("ee_pool", [])
    selected_ee = next(
        (
            record
            for record in ee_pool
            if isinstance(record, Mapping)
            and str(record.get("ee_id")) == str(runtime.active_ee)
        ),
        None,
    )
    if isinstance(selected_ee, Mapping):
        raw_force = selected_ee.get("grip_force_n")
        if isinstance(raw_force, (int, float)) and float(raw_force) > 0.0:
            maximum_force = float(raw_force)
        friction = selected_ee.get("fingerpad_friction")
        if (
            isinstance(friction, (list, tuple))
            and len(friction) == 3
            and all(isinstance(value, (int, float)) for value in friction)
        ):
            runtime.set_finger_gripper_contact_friction(
                tuple(float(value) for value in friction)
            )

    metadata_provider = getattr(runtime.env, "get_tool_physical_metadata", None)
    if callable(metadata_provider):
        raw_metadata = metadata_provider(object_id)
        if isinstance(raw_metadata, Mapping):
            raw_mass = raw_metadata.get("mass_kg")
            if isinstance(raw_mass, (int, float)) and float(raw_mass) > 0.0:
                mass_kg = float(raw_mass)
            raw_minimum_retention_force = raw_metadata.get(
                "minimum_retention_force_n"
            )
            if (
                isinstance(raw_minimum_retention_force, (int, float))
                and float(raw_minimum_retention_force) > 0.0
            ):
                object_minimum_retention_force_n = float(
                    raw_minimum_retention_force
                )
            raw_retention_control_mode = str(
                raw_metadata.get("retention_control_mode", "FORCE_TARGET")
            ).upper()
            if raw_retention_control_mode not in {
                "FORCE_TARGET",
                "POSITION_HOLD",
            }:
                raise ValueError(
                    "retention_control_mode must be FORCE_TARGET or POSITION_HOLD"
                )
            retention_control_mode = raw_retention_control_mode
            friction = raw_metadata.get("friction")
            if (
                isinstance(friction, (list, tuple))
                and friction
                and isinstance(friction[0], (int, float))
                and float(friction[0]) > 0.0
            ):
                sliding_friction = float(friction[0])

    target_force = profile.force.fallback_retention_force_n
    retention_model = "FALLBACK"
    if mass_kg is not None:
        weight_limited_force = (
            profile.force.grip_force_safety_factor
            * mass_kg
            * 9.81
            / sliding_friction
        )
        retention_floor = max(
            profile.force.fallback_retention_force_n,
            object_minimum_retention_force_n,
        )
        target_force = max(
            retention_floor,
            weight_limited_force,
        )
        retention_model = (
            "WEIGHT"
            if weight_limited_force >= retention_floor
            else "WEIGHT_WITH_RETENTION_FLOOR"
        )
    # The force floor is explicit input, even if mass is unavailable. Bbox
    # dimensions do not establish a contact moment arm or a torque model.
    if object_minimum_retention_force_n > target_force:
        target_force = object_minimum_retention_force_n
        retention_model = "EXPLICIT_RETENTION_FLOOR"
    target_force = min(maximum_force, target_force)
    return RuntimeGraspParameters(
        maximum_grip_force_n=maximum_force,
        sliding_friction=sliding_friction,
        retention_target_normal_force_n=target_force,
        retention_model=retention_model,
        retention_control_mode=retention_control_mode,
    )


@dataclass(slots=True)
class PhysicalGraspMonitor:
    """Observe bilateral 2F contact, lift, and final retention in MuJoCo."""

    runtime: ToolUseJournalEERuntime
    object_id: str
    profile: PhysicalGraspProfile
    runtime_parameters: RuntimeGraspParameters
    initial_position_m: tuple[float, float, float]
    samples: list[dict[str, Any]] = field(default_factory=list)
    _bilateral_ticks: int = 0
    _formation_time_s: float | None = None
    _last_valid_contact_time_s: float | None = None
    _lift_hold_started_time_s: float | None = None
    _active_actuator_kp: float | None = None
    _retention_control_active: bool = False

    @classmethod
    def from_runtime(
        cls,
        runtime: ToolUseJournalEERuntime,
        object_id: str,
        profile: PhysicalGraspProfile,
    ) -> "PhysicalGraspMonitor":
        position, _ = _object_pose(runtime, object_id)
        parameters = _runtime_grasp_parameters(runtime, object_id, profile)
        return cls(
            runtime=runtime,
            object_id=object_id,
            profile=profile,
            runtime_parameters=parameters,
            initial_position_m=tuple(float(value) for value in position),
        )

    @staticmethod
    def _bilateral(groups: set[str]) -> bool:
        return "left_finger" in groups and "right_finger" in groups

    def sample(self, simulation_time_s: float) -> None:
        if not self.runtime.grasp_engaged:
            return
        position, _ = _object_pose(self.runtime, self.object_id)
        _, _, reference_position, reference_rotation = self.runtime._grasp_reference(
            self.runtime.env
        )
        contact = self.runtime.object_contact_metrics(self.object_id)
        groups = set(contact.contact_groups)
        bilateral = self._bilateral(groups)
        friction_utilization = (
            float(contact.tangential_force_n)
            / (
                self.runtime_parameters.sliding_friction
                * float(contact.normal_force_n)
            )
            if contact.normal_force_n > 1e-9
            else math.inf
        )
        valid_contact = (
            bilateral
            and contact.normal_force_n
            >= self.profile.validation.minimum_normal_force_n
            and friction_utilization
            <= self.profile.validation.max_friction_utilization
        )
        lift_m = float(position[2] - self.initial_position_m[2])

        if valid_contact:
            self._bilateral_ticks += 1
            self._last_valid_contact_time_s = float(simulation_time_s)
        elif self._formation_time_s is None:
            self._bilateral_ticks = 0
        if (
            self._formation_time_s is None
            and self._bilateral_ticks
            >= self.profile.validation.required_contact_ticks
        ):
            self._formation_time_s = float(simulation_time_s)

        if (
            not self._retention_control_active
            and self._bilateral_ticks
            >= self.profile.validation.contact_freeze_ticks
        ):
            if self.runtime_parameters.retention_control_mode == "FORCE_TARGET":
                self._active_actuator_kp = self.profile.force.closure_actuator_kp
                self.runtime.set_finger_gripper_force_target(
                    total_force_n=(
                        self.runtime_parameters.retention_target_normal_force_n
                    ),
                    actuator_kp=self._active_actuator_kp,
                    max_grip_force_n=(
                        self.runtime_parameters.maximum_grip_force_n
                    ),
                )
            self.runtime.hold_gripper_position()
            self._retention_control_active = True
        elif (
            self._active_actuator_kp is not None
            and valid_contact
            and contact.normal_force_n > 1e-9
        ):
            target = self.runtime_parameters.retention_target_normal_force_n
            normalized_error = float(
                np.clip((target - contact.normal_force_n) / target, -1.0, 1.0)
            )
            next_kp = float(
                np.clip(
                    self._active_actuator_kp
                    * (
                        1.0
                        + self.profile.force.force_feedback_gain
                        * normalized_error
                    ),
                    self.profile.force.closure_actuator_kp,
                    self.profile.force.maximum_actuator_kp,
                )
            )
            if not math.isclose(
                next_kp, self._active_actuator_kp, rel_tol=1e-4, abs_tol=1e-4
            ):
                self.runtime.set_finger_gripper_force_target(
                    total_force_n=target,
                    actuator_kp=next_kp,
                    max_grip_force_n=(
                        self.runtime_parameters.maximum_grip_force_n
                    ),
                )
                self._active_actuator_kp = next_kp

        retained = self._retained_at(float(simulation_time_s))
        if retained and lift_m >= self.profile.validation.minimum_lift_m:
            if self._lift_hold_started_time_s is None:
                self._lift_hold_started_time_s = float(simulation_time_s)
        else:
            self._lift_hold_started_time_s = None
        self.samples.append(
            {
                "simulation_time_s": float(simulation_time_s),
                "object_position_m": [float(value) for value in position],
                "grasp_reference_position_m": [
                    float(value) for value in reference_position
                ],
                "grasp_reference_rotation": [
                    [float(value) for value in row]
                    for row in np.asarray(reference_rotation, dtype=float)
                ],
                "lift_m": lift_m,
                "contact_count": int(contact.contact_count),
                "normal_force_n": float(contact.normal_force_n),
                "tangential_force_n": float(contact.tangential_force_n),
                "contact_groups": sorted(groups),
                "bilateral_contact": bilateral,
                "valid_contact": valid_contact,
                "friction_utilization": friction_utilization,
                "retained": retained,
                "actuator_kp": self._active_actuator_kp,
            }
        )

    def contact_follow_translation_m(self) -> np.ndarray | None:
        """Damp tick-to-tick free-object motion relative to the grasp frame."""

        if self._formation_time_s is None or len(self.samples) < 2:
            return None
        latest, previous = self.samples[-1], self.samples[-2]
        if not bool(latest["valid_contact"]) or not bool(previous["valid_contact"]):
            return None

        def local_offset(sample: Mapping[str, Any]) -> np.ndarray:
            object_position = np.asarray(sample["object_position_m"], dtype=float)
            reference_position = np.asarray(
                sample["grasp_reference_position_m"], dtype=float
            )
            reference_rotation = np.asarray(
                sample["grasp_reference_rotation"], dtype=float
            )
            return reference_rotation.T @ (object_position - reference_position)

        latest_rotation = np.asarray(
            latest["grasp_reference_rotation"], dtype=float
        )
        relative_tick_delta_world = latest_rotation @ (
            local_offset(latest) - local_offset(previous)
        )
        correction = (
            -self.profile.stabilization.contact_follow_gain
            * relative_tick_delta_world
        )
        correction_norm = float(np.linalg.norm(correction))
        if correction_norm > self.profile.stabilization.maximum_translation_m:
            correction *= (
                self.profile.stabilization.maximum_translation_m
                / correction_norm
            )
        correction_norm = float(np.linalg.norm(correction))
        if (
            correction_norm
            > self.profile.stabilization.maximum_translation_per_tick_m
        ):
            correction *= (
                self.profile.stabilization.maximum_translation_per_tick_m
                / correction_norm
            )
        return correction

    def _retained_at(self, simulation_time_s: float) -> bool:
        if self._formation_time_s is None or self._last_valid_contact_time_s is None:
            return False
        return (
            simulation_time_s - self._last_valid_contact_time_s
            <= self.profile.validation.contact_loss_grace_s
        )

    def relative_transform(self) -> dict[str, Any]:
        position, rotation = _object_pose(self.runtime, self.object_id)
        reference_kind, reference_name, reference_position, reference_rotation = (
            self.runtime._grasp_reference(self.runtime.env)
        )
        reference_rotation = np.asarray(reference_rotation, dtype=float)
        translation = reference_rotation.T @ (
            position - np.asarray(reference_position, dtype=float)
        )
        relative_rotation = reference_rotation.T @ rotation
        _, _, free_joint_name = self.runtime._object_free_joint(
            self.runtime.env, self.object_id
        )
        return {
            "object_id": self.object_id,
            "free_joint_name": str(free_joint_name),
            "reference_kind": str(reference_kind),
            "reference_name": str(reference_name),
            "position_in_reference_m": [float(value) for value in translation],
            "orientation_in_reference_xyzw": list(
                _matrix_quaternion_xyzw(relative_rotation)
            ),
            # Keep the rotation matrix as diagnostic evidence; the fields above
            # form the planning contract consumed by AttachedObjectTransform.
            "rotation_matrix": [
                [float(value) for value in row] for row in relative_rotation
            ],
        }

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {
                "status": "FAILED",
                "reason": "NO_PHYSICAL_GRASP_SAMPLES",
                "object_id": self.object_id,
                "contact_formation": {"status": "FAILED"},
                "grasp_retention": {"status": "NOT_EVALUATED"},
            }
        final = self.samples[-1]
        final_time = float(final["simulation_time_s"])
        formed = self._formation_time_s is not None
        retained = self._retained_at(final_time)
        max_lift = max(float(sample["lift_m"]) for sample in self.samples)
        final_lift = float(final["lift_m"])
        hold_duration = (
            final_time - self._lift_hold_started_time_s
            if self._lift_hold_started_time_s is not None
            else 0.0
        )
        lifted = max_lift >= self.profile.validation.minimum_lift_m
        hold_satisfied = (
            hold_duration >= self.profile.validation.final_hold_duration_s
        )
        succeeded = formed and retained and lifted and hold_satisfied
        return {
            "status": "SUCCESS" if succeeded else "FAILED",
            "object_id": self.object_id,
            "contact_formation": {
                "status": "SUCCESS" if formed else "FAILED",
                "formed_at_simulation_time_s": self._formation_time_s,
                "required_contact_ticks": (
                    self.profile.validation.required_contact_ticks
                ),
            },
            "grasp_retention": {
                "status": "SUCCESS" if retained and hold_satisfied else "FAILED",
                "retained_at_end": retained,
                "final_hold_duration_s": hold_duration,
                "required_final_hold_duration_s": (
                    self.profile.validation.final_hold_duration_s
                ),
            },
            "sample_count": len(self.samples),
            "final_contact_count": int(final["contact_count"]),
            "final_contact_groups": list(final["contact_groups"]),
            "final_bilateral_contact": bool(final["bilateral_contact"]),
            "final_lift_m": final_lift,
            "max_lift_m": max_lift,
            "minimum_lift_m": self.profile.validation.minimum_lift_m,
            "final_normal_force_n": float(final["normal_force_n"]),
            "retention_target_normal_force_n": (
                self.runtime_parameters.retention_target_normal_force_n
            ),
            "retention_force_model": self.runtime_parameters.retention_model,
            "retention_control_mode": (
                self.runtime_parameters.retention_control_mode
            ),
            "relative_grasp_transform": self.relative_transform(),
        }


def _object_pose(
    runtime: ToolUseJournalEERuntime, object_id: str
) -> tuple[np.ndarray, np.ndarray]:
    env = runtime.env
    try:
        body_id = int(env.obj_body_id[object_id])
        data = env.sim.data._data
        position = np.asarray(data.xpos[body_id], dtype=float).copy()
        rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3).copy()
    except (AttributeError, KeyError, TypeError, ValueError, IndexError) as error:
        raise ValueError(f"physical grasp object {object_id!r} is absent") from error
    return position, rotation


class PhysicalGraspControllerTrajectoryPlayer(
    ToolUseJournalControllerTrajectoryPlayer
):
    """Controller player that turns live contact evidence into PICK state."""

    def __init__(
        self,
        runtime: ToolUseJournalEERuntime,
        *,
        monitor: PhysicalGraspMonitor,
        preshape_aperture_m: float | None = None,
        preshape_tolerance_m: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(runtime, **kwargs)
        self.monitor = monitor
        self.preshape_aperture_m = preshape_aperture_m
        self.preshape_tolerance_m = preshape_tolerance_m

    def _advance_controller(self, action: np.ndarray) -> float:
        simulation_time_s = super()._advance_controller(
            self._contact_follow_action(action)
        )
        self.monitor.sample(simulation_time_s)
        return simulation_time_s

    def _contact_follow_action(self, action: np.ndarray) -> np.ndarray:
        translation = self.monitor.contact_follow_translation_m()
        if translation is None or not np.any(np.abs(translation) > 1e-12):
            return action
        env = self.runtime.env
        model = env.sim.model._model
        data = env.sim.data._data
        robot = env.robots[0]
        joint_names = tuple(str(name) for name in robot.robot_model.joints)
        arm_start, arm_end = (
            robot.composite_controller._action_split_indexes["right"]
        )
        reference_kind, reference_name, _, _ = self.runtime._grasp_reference(env)
        jacobian_position = np.zeros((3, model.nv), dtype=float)
        jacobian_rotation = np.zeros((3, model.nv), dtype=float)
        if reference_kind == "site":
            reference_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, reference_name
            )
            mujoco.mj_jacSite(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                reference_id,
            )
        elif reference_kind == "body":
            reference_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, reference_name
            )
            mujoco.mj_jacBody(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                reference_id,
            )
        else:
            return action
        dof_ids = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            dof_ids.append(int(model.jnt_dofadr[joint_id]))
        jacobian = jacobian_position[:, dof_ids]
        damping = 1e-4
        joint_delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping * np.eye(3),
            translation,
        )
        limit = self.monitor.profile.stabilization.maximum_joint_step_rad
        joint_delta = np.clip(joint_delta, -limit, limit)
        corrected = np.asarray(action, dtype=float).copy()
        corrected[arm_start:arm_end] += joint_delta
        return corrected

    def execute(
        self, run: SimulationRun, *, report_id: str | None = None
    ) -> ExecutionReport:
        if self.preshape_aperture_m is not None:
            self.preshape_finger_gripper_to_aperture(
                target_aperture_m=self.preshape_aperture_m,
                tolerance_m=(
                    self.preshape_tolerance_m
                    if self.preshape_tolerance_m is not None
                    else self.monitor.profile.preshape.tolerance_m
                ),
                final_settle_ticks=self.monitor.profile.preshape.settle_ticks,
            )
        # The ordinary player verifies logical held-tool state against the
        # runtime.  For contact friction that state can only be committed after
        # the physical samples have been evaluated, so execute against the
        # pre-validation runtime state and restore the logical result below.
        runtime_plan = run.plan.model_copy(
            update={
                "expected_final_state": run.plan.expected_final_state.model_copy(
                    update={
                        "held_tool_id": self.runtime.held_tool_id,
                        "attached_object_id": self.runtime.attached_object_id,
                    }
                )
            }
        )
        report = super().execute(
            run.model_copy(update={"plan": runtime_plan}), report_id=report_id
        )
        validation = self.monitor.summary()
        succeeded = (
            validation["status"] == "SUCCESS"
            and report.status is ExecutionStatus.SUCCESS
        )
        if succeeded:
            self.runtime.mark_contact_friction_object_as_tool(
                self.monitor.object_id
            )
        final_state = report.final_robot_state
        if final_state is not None:
            final_state = final_state.model_copy(
                update={
                    "held_tool_id": self.runtime.held_tool_id,
                    "attached_object_id": None,
                }
            )
        return report.model_copy(
            update={
                "final_robot_state": final_state,
                "metadata": {
                    **report.metadata,
                    "grasp_execution_mode": (
                        GraspExecutionMode.CONTACT_FRICTION.value
                    ),
                    "grasp_retention_validation": validation,
                    "physical_grasp_transform": validation.get(
                        "relative_grasp_transform"
                    ),
                    "physical_grasp_execution_succeeded": succeeded,
                    "trajectory_execution_status": (
                        report.status.value
                        if isinstance(report.status, ExecutionStatus)
                        else str(report.status)
                    ),
                },
            }
        )


__all__ = [
    "ContactFrictionKeyframeProvider",
    "GraspExecutionMode",
    "PhysicalGraspControllerTrajectoryPlayer",
    "PhysicalGraspMonitor",
    "RetargetedAcquireKeyframeProvider",
    "RuntimeGraspParameters",
    "normalized_grasp_execution_mode",
    "releases_contact_friction",
    "uses_contact_friction",
    "with_contact_friction_grasp",
    "with_contact_friction_release",
]
