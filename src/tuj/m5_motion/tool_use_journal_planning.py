"""Production planning bindings for Tool-Use-Journal workcells.

This module keeps collision semantics deterministic.  A VLM proposes symbolic
keyframes, then :class:`ToolUseJournalCollisionContextFactory` assigns the
physical scene active for every incoming segment and every event boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from glob import has_magic
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from tuj.m5_motion.attachment_retarget import (
    ATTACHED_OBJECT_POSE_SUBJECT,
    POSE_SUBJECT_KEY,
    POSE_SUBJECT_OBJECT_ID_KEY,
    attachment_transform,
    object_pose_for_end_effector_pose,
)
from tuj.m5_motion.ee_exchange import RoutedKeyframeStrategyProvider
from tuj.m5_motion.ee_exchange_entry import (
    EEExchangeEntryPlanner,
    is_ee_exchange_entry_request,
)
from tuj.m5_motion.geometry import RelativePoseResolver
from tuj.m5_motion.kinematics import UR5eKinematics
from tuj.m5_motion.pipeline import (
    CollisionPlanningSetup,
    KeyframeStrategyProvider,
    MotionPlanningPipeline,
    MotionPlanningResult,
)
from tuj.m5_motion.precomputed_ee_attach import (
    EEAttachPathFailureCode,
    EEAttachPolicy,
    PrecomputedEEAttachPlanner,
    PrecomputedEEAttachRegistry,
    PrecomputedEEPathError,
    is_initial_ee_attach,
    normalize_ee_id,
)
from tuj.m5_motion.precomputed_ee_exchange import (
    PrecomputedEEExchangePlanner,
    PrecomputedEEReturnRegistry,
    is_ee_exchange_request,
)
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    AttachedObjectTransform,
    CollisionContext,
    FreeObjectPose,
    GoalType,
    KeyframeEventType,
    KeyframePlanArtifact,
    KeyframePlanCandidate,
    KeyframeType,
    ModuleName,
    MotionPlanRequest,
    Pose,
    WorldSnapshot,
)
from tuj.m5_motion.task_semantics import (
    is_acquire_task,
    is_ee_exchange_task,
    is_release_task,
)
from tuj.m5_motion.tool_use_journal import (
    ToolUseJournalCollisionModelCompiler,
    ToolUseJournalCompatibilityError,
    ToolUseJournalEnvironmentAdapter,
)
from tuj.m5_motion.vlm_provider import OpenAIKeyframeProvider


_ATTACHMENT_METADATA_KEY = "attached_object_transforms"
_CONTACT_FRICTION_HELD_METADATA_KEY = "contact_friction_held_objects"
_BOUNDED_COLLISION_ALLOWANCES_KEY = "bounded_collision_allowances"
_POST_SEGMENT_VALIDATION_CONTEXT_KEY = "post_segment_validation_context_id"
_MAX_SUPPORT_PENETRATION_TOLERANCE_M = 0.001
_DEFAULT_SUPPORT_MIN_HORIZONTAL_OVERLAP_RATIO = 0.5


class ToolUseJournalCollisionBindingError(RuntimeError):
    """A request cannot be bound to one unambiguous physical collision scene."""


class MotionRequestPlanner(Protocol):
    def __call__(self, request: MotionPlanRequest) -> Any: ...


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _short_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:16]


def _quaternion_matrix_xyzw(values: Sequence[float]) -> np.ndarray:
    if len(values) != 4:
        raise ToolUseJournalCollisionBindingError(
            "orientation quaternion must have four values"
        )
    x, y, z, w = (float(value) for value in values)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ToolUseJournalCollisionBindingError(
            "orientation quaternion must be finite and non-zero"
        )
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=float,
    )


def _matrix_quaternion_xyzw(matrix: np.ndarray) -> tuple[float, float, float, float]:
    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = np.asarray(
            (
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            )
        )
    else:
        diagonal = int(np.argmax(np.diag(rotation)))
        if diagonal == 0:
            scale = math.sqrt(
                max(
                    1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2],
                    0.0,
                )
            ) * 2.0
            values = np.asarray(
                (
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                )
            )
        elif diagonal == 1:
            scale = math.sqrt(
                max(
                    1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2],
                    0.0,
                )
            ) * 2.0
            values = np.asarray(
                (
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                )
            )
        else:
            scale = math.sqrt(
                max(
                    1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1],
                    0.0,
                )
            ) * 2.0
            values = np.asarray(
                (
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                )
            )
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ToolUseJournalCollisionBindingError(
            "rotation matrix could not be converted to a quaternion"
        )
    values /= norm
    if values[3] < 0.0:
        values *= -1.0
    return tuple(float(value) for value in values)


def attached_object_transform_from_state(state: object) -> AttachedObjectTransform:
    """Convert a runtime ``AttachedObjectState`` without importing the runtime.

    The duck-typed boundary avoids an import cycle because the runtime already
    depends on the Tool-Use-Journal environment adapter.
    """

    try:
        rotation = np.asarray(
            getattr(state, "rotation_in_reference"), dtype=float
        ).reshape(3, 3)
        return AttachedObjectTransform(
            object_id=str(getattr(state, "object_id")),
            free_joint_name=str(getattr(state, "free_joint_name")),
            reference_kind=str(getattr(state, "reference_kind")),
            reference_name=str(getattr(state, "reference_name")),
            position_in_reference_m=tuple(
                float(value)
                for value in getattr(state, "position_in_reference_m")
            ),
            orientation_in_reference_xyzw=_matrix_quaternion_xyzw(rotation),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ToolUseJournalCollisionBindingError(
            "runtime attachment state is incomplete"
        ) from error


def _object_pose(world: WorldSnapshot, object_id: str) -> Pose:
    record = world.objects.get(object_id)
    if not isinstance(record, Mapping):
        raise ToolUseJournalCollisionBindingError(
            f"world has no object record for {object_id!r}"
        )
    raw_pose = record.get("pose")
    if isinstance(raw_pose, Mapping):
        raw_pose = {**raw_pose, "frame_id": raw_pose.get("frame_id", "world")}
    try:
        pose = Pose.model_validate(raw_pose)
    except (TypeError, ValueError) as error:
        raise ToolUseJournalCollisionBindingError(
            f"object {object_id!r} has no valid world pose"
        ) from error
    if pose.frame_id != "world":
        raise ToolUseJournalCollisionBindingError(
            f"object {object_id!r} pose is not expressed in world"
        )
    return pose


def _free_joint_name(world: WorldSnapshot, object_id: str) -> str:
    record = world.objects.get(object_id)
    value = record.get("free_joint_name") if isinstance(record, Mapping) else None
    if not isinstance(value, str) or not value:
        raise ToolUseJournalCollisionBindingError(
            f"object {object_id!r} has no MuJoCo free_joint_name"
        )
    return value


def _free_object_poses(
    world: WorldSnapshot,
    *,
    exclude: Sequence[str] = (),
    overrides: Mapping[str, Pose] | None = None,
) -> list[FreeObjectPose]:
    excluded = set(exclude)
    replacements = dict(overrides or {})
    result: list[FreeObjectPose] = []
    for object_id, record in sorted(world.objects.items()):
        if object_id in excluded or not isinstance(record, Mapping):
            continue
        free_joint_name = record.get("free_joint_name")
        if not isinstance(free_joint_name, str) or not free_joint_name:
            continue
        pose = replacements.get(object_id) or _object_pose(world, object_id)
        result.append(
            FreeObjectPose(
                object_id=object_id,
                free_joint_name=free_joint_name,
                pose=pose,
            )
        )
    for object_id, pose in sorted(replacements.items()):
        if object_id in excluded or any(
            item.object_id == object_id for item in result
        ):
            continue
        result.append(
            FreeObjectPose(
                object_id=object_id,
                free_joint_name=_free_joint_name(world, object_id),
                pose=pose,
            )
        )
    return result


def _attachment_transforms(world: WorldSnapshot) -> dict[str, AttachedObjectTransform]:
    raw = world.metadata.get(_ATTACHMENT_METADATA_KEY, {})
    if raw is None:
        raw = {}
    if isinstance(raw, Mapping):
        values = list(raw.values())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
    else:
        raise ToolUseJournalCollisionBindingError(
            f"world metadata {_ATTACHMENT_METADATA_KEY!r} must be a mapping or list"
        )
    result: dict[str, AttachedObjectTransform] = {}
    for value in values:
        try:
            transform = AttachedObjectTransform.model_validate(value)
        except (TypeError, ValueError) as error:
            raise ToolUseJournalCollisionBindingError(
                "world contains an invalid attached-object transform"
            ) from error
        if transform.object_id in result:
            raise ToolUseJournalCollisionBindingError(
                f"duplicate attachment transform for {transform.object_id!r}"
            )
        result[transform.object_id] = transform
    expected = world.robot_state.attached_object_id
    if expected is None and result:
        raise ToolUseJournalCollisionBindingError(
            "world has attachment transforms but robot_state has no attached object"
        )
    if expected is not None and set(result) != {expected}:
        raise ToolUseJournalCollisionBindingError(
            f"attached object {expected!r} requires exactly one matching transform"
        )
    return result


def _relative_attachment(
    request: MotionPlanRequest,
    keyframe: Any,
    *,
    object_id: str,
    reference_kind: str,
    reference_name: str,
) -> AttachedObjectTransform:
    reference_pose = RelativePoseResolver(request.world).resolve(keyframe)
    object_pose = _object_pose(request.world, object_id)
    reference_rotation = _quaternion_matrix_xyzw(
        reference_pose.orientation_xyzw
    )
    object_rotation = _quaternion_matrix_xyzw(object_pose.orientation_xyzw)
    relative_position = reference_rotation.T @ (
        np.asarray(object_pose.position_m, dtype=float)
        - np.asarray(reference_pose.position_m, dtype=float)
    )
    relative_rotation = reference_rotation.T @ object_rotation
    return AttachedObjectTransform(
        object_id=object_id,
        free_joint_name=_free_joint_name(request.world, object_id),
        reference_kind=reference_kind,
        reference_name=reference_name,
        position_in_reference_m=tuple(
            float(value) for value in relative_position
        ),
        orientation_in_reference_xyzw=_matrix_quaternion_xyzw(
            relative_rotation
        ),
    )


def _stamp_bound_artifact(
    request: MotionPlanRequest,
    source: KeyframePlanArtifact,
    bound: KeyframePlanArtifact,
) -> KeyframePlanArtifact:
    digest = _short_digest(
        {
            "request_id": request.request_id,
            "source_artifact_id": source.artifact_id,
            "candidates": [
                candidate.model_dump(mode="json") for candidate in bound.candidates
            ],
        }
    )
    bound.artifact_id = f"keyframe-plan-bound:{digest}"
    bound.provenance = ArtifactProvenance(
        artifact_id=f"keyframe-plan-bound-artifact:{digest}",
        artifact_type="CollisionBoundKeyframePlanArtifact",
        produced_by=ModuleName.MOTION_PLANNER,
        invocation_id=f"collision-context-binding:{request.request_id}",
        input_artifact_ids=[source.provenance.artifact_id],
        attempt=source.provenance.attempt,
        metadata={
            "source_keyframe_artifact_id": source.artifact_id,
            "binding": "tool-use-journal-collision-context-v1",
        },
    )
    return bound


class ToolUseJournalCollisionContextFactory:
    """Bind one generated artifact to request-scoped MuJoCo scene models."""

    def __init__(
        self,
        compiler: ToolUseJournalCollisionModelCompiler,
        *,
        attachment_reference_name: str,
        attachment_reference_kind: str = "body",
    ) -> None:
        if attachment_reference_kind not in {"body", "site"}:
            raise ValueError("attachment_reference_kind must be 'body' or 'site'")
        if not attachment_reference_name:
            raise ValueError("attachment_reference_name is required")
        self.compiler = compiler
        self.attachment_reference_name = attachment_reference_name
        self.attachment_reference_kind = attachment_reference_kind

    def _validate_environment(self, request: MotionPlanRequest) -> None:
        environment_name = request.world.metadata.get("environment_name")
        if environment_name != self.compiler.environment_name:
            raise ToolUseJournalCollisionBindingError(
                f"request environment {environment_name!r} does not match "
                f"collision compiler {self.compiler.environment_name!r}"
            )

    @staticmethod
    def _active_ee(request: MotionPlanRequest) -> str:
        active_ee = request.world.metadata.get("physical_active_ee")
        if not isinstance(active_ee, str) or not active_ee:
            raise ToolUseJournalCollisionBindingError(
                "request world has no physical_active_ee"
            )
        return active_ee

    def _base_context(
        self,
        request: MotionPlanRequest,
        *,
        active_ee: str,
    ) -> CollisionContext:
        transforms = _attachment_transforms(request.world)
        attached_id = request.world.robot_state.attached_object_id
        held = request.world.robot_state.held_tool_id
        physical = request.world.metadata.get("contact_friction_held_objects", {})
        if attached_id is None and held in physical:
            transform = AttachedObjectTransform.model_validate(physical[held])
            if transform.object_id != held:
                raise ToolUseJournalCollisionBindingError("contact grasp transform targets a different object")
            return CollisionContext(
                context_id=f"contact-held:{held}:initial",
                scene_state_id=f"{request.world.scene.signature}:contact-held:{held}",
                active_ee=active_ee,
                attached_object_ids=[held],
                attached_object_transforms=[transform],
                free_object_poses=_free_object_poses(request.world, exclude=(held,)),
                touch_links=[active_ee],
                collision_model_version=self.compiler.model_version_for(active_ee),
                metadata={"attachment_proxy": "CONTACT_FRICTION"},
            )
        if attached_id is None:
            held_tool_id = request.world.robot_state.held_tool_id
            raw_contact_transforms = request.world.metadata.get(
                _CONTACT_FRICTION_HELD_METADATA_KEY, {}
            )
            if held_tool_id is not None and isinstance(
                raw_contact_transforms, Mapping
            ) and held_tool_id in raw_contact_transforms:
                try:
                    transform = AttachedObjectTransform.model_validate(
                        raw_contact_transforms[held_tool_id]
                    )
                except (TypeError, ValueError) as error:
                    raise ToolUseJournalCollisionBindingError(
                        "world contains an invalid contact-friction transform"
                    ) from error
                context_id = f"contact-friction-held:{held_tool_id}:initial"
                return CollisionContext(
                    context_id=context_id,
                    scene_state_id=(
                        f"{request.world.scene.signature}:ee:{active_ee}:"
                        f"contact-held:{held_tool_id}"
                    ),
                    active_ee=active_ee,
                    attached_object_ids=[held_tool_id],
                    attached_object_transforms=[transform],
                    free_object_poses=_free_object_poses(
                        request.world, exclude=(held_tool_id,)
                    ),
                    touch_links=[active_ee],
                    collision_model_version=self.compiler.model_version_for(
                        active_ee
                    ),
                    metadata={"attachment_proxy": "CONTACT_FRICTION"},
                )
            context_id = f"ee-attached:{active_ee}"
            return CollisionContext(
                context_id=context_id,
                scene_state_id=f"{request.world.scene.signature}:ee:{active_ee}",
                active_ee=active_ee,
                free_object_poses=_free_object_poses(request.world),
                collision_model_version=self.compiler.model_version_for(active_ee),
            )
        transform = transforms[attached_id]
        context_id = f"object-attached:{attached_id}:initial"
        return CollisionContext(
            context_id=context_id,
            scene_state_id=(
                f"{request.world.scene.signature}:ee:{active_ee}:"
                f"attached:{attached_id}"
            ),
            active_ee=active_ee,
            attached_object_ids=[attached_id],
            attached_object_transforms=[transform],
            free_object_poses=_free_object_poses(
                request.world, exclude=(attached_id,)
            ),
            touch_links=[active_ee],
            collision_model_version=self.compiler.model_version_for(active_ee),
        )

    @staticmethod
    def _verify_context_references(
        artifact: KeyframePlanArtifact,
        contexts: Mapping[str, CollisionContext],
    ) -> None:
        referenced: set[str] = set()
        for candidate in artifact.candidates:
            for keyframe in candidate.keyframes:
                if keyframe.collision_context_id is not None:
                    referenced.add(keyframe.collision_context_id)
                if keyframe.collision_context_after_events_id is not None:
                    referenced.add(keyframe.collision_context_after_events_id)
        missing = referenced - set(contexts)
        if missing:
            raise ToolUseJournalCollisionBindingError(
                f"keyframes reference unregistered collision contexts {sorted(missing)}"
            )

    @staticmethod
    def _event_keyframe(
        candidate: KeyframePlanCandidate,
        event: KeyframeEventType,
        expected_type: KeyframeType,
    ) -> Any:
        matches = [
            keyframe
            for keyframe in candidate.keyframes
            if event in keyframe.events_after
        ]
        if len(matches) != 1 or matches[0].keyframe_type is not expected_type:
            raise ToolUseJournalCollisionBindingError(
                f"strategy {candidate.strategy_id!r} requires exactly one "
                f"{expected_type.value} keyframe with {event.value}"
            )
        return matches[0]

    @staticmethod
    def _contact_pairs(
        left: str,
        selectors: Sequence[str],
    ) -> list[tuple[str, str]]:
        return sorted({tuple(sorted((left, item))) for item in selectors if item})

    @staticmethod
    def _region_occupant_ids(
        request: MotionPlanRequest,
        target: str,
    ) -> list[str]:
        """Ids of scene objects already resting inside the destination region.

        Multi-object kitting packs several objects into one region and the task
        permits them to rest against or on top of one another, so the held
        object is allowed to contact whatever is already there.  Membership uses
        the region's world AABB: an object whose centre lies within the region
        footprint and at or above its floor counts as an occupant.  A
        single-object region has no occupants, so this returns an empty list and
        leaves those tasks unchanged.
        """
        region_id = request.task.goal.target_region_id
        if not region_id:
            return []
        objects = request.world.objects
        region = objects.get(region_id)
        if (
            not isinstance(region, dict)
            or "pose" not in region
            or "dimensions_m" not in region
        ):
            return []
        from scipy.spatial.transform import Rotation

        r_pose = region["pose"]
        r_pos = np.asarray(r_pose["position_m"], dtype=float)
        r_rot = Rotation.from_quat(r_pose["orientation_xyzw"]).as_matrix()
        r_center_local = np.asarray(
            region.get("anchors", {}).get("center", [0.0, 0.0, 0.0]), dtype=float
        )
        r_center = r_pos + r_rot @ r_center_local
        r_half = np.abs(r_rot) @ (np.asarray(region["dimensions_m"], dtype=float) / 2.0)
        region_bottom = float(r_center[2] - r_half[2])
        occupants: list[str] = []
        for other_id, other in objects.items():
            if other_id in {target, region_id}:
                continue
            if (
                not isinstance(other, dict)
                or "pose" not in other
                or "dimensions_m" not in other
            ):
                continue
            pose = other["pose"]
            if pose.get("frame_id", "world") != "world":
                continue
            center = np.asarray(pose["position_m"], dtype=float)
            anchor = other.get("anchors", {}).get("center")
            if anchor is not None:
                center = center + Rotation.from_quat(
                    pose["orientation_xyzw"]
                ).as_matrix() @ np.asarray(anchor, dtype=float)
            inside = bool(np.all(np.abs(center[:2] - r_center[:2]) <= r_half[:2]))
            if inside and center[2] >= region_bottom - 1e-6:
                occupants.append(other_id)
        return occupants

    @staticmethod
    def _metadata_selectors(request: MotionPlanRequest, key: str) -> list[str]:
        raw = request.task.metadata.get(key, ())
        if isinstance(raw, str):
            return [raw] if raw else []
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if not all(isinstance(item, str) and item for item in raw):
                raise ToolUseJournalCollisionBindingError(
                    f"task metadata {key!r} must contain non-empty strings"
                )
            return list(raw)
        raise ToolUseJournalCollisionBindingError(
            f"task metadata {key!r} must be a string or list of strings"
        )

    @staticmethod
    def _metadata_nonnegative_distance(
        request: MotionPlanRequest,
        key: str,
        *,
        default: float,
    ) -> float:
        raw = request.task.metadata.get(key, default)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) < 0.0
        ):
            raise ToolUseJournalCollisionBindingError(
                f"task metadata {key!r} must be finite and non-negative"
            )
        return float(raw)

    @staticmethod
    def _metadata_unit_interval(
        request: MotionPlanRequest,
        key: str,
        *,
        default: float,
    ) -> float:
        raw = request.task.metadata.get(key, default)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or not 0.0 <= float(raw) <= 1.0
        ):
            raise ToolUseJournalCollisionBindingError(
                f"task metadata {key!r} must be finite and within [0, 1]"
            )
        return float(raw)

    def _pick_support_policy(
        self,
        request: MotionPlanRequest,
        target: str,
    ) -> tuple[list[str], dict[str, Any], float]:
        """Resolve a narrowly scoped initial support contact for any PICK."""

        metadata = request.task.metadata
        minimum_overlap_ratio = self._metadata_unit_interval(
            request,
            "support_min_horizontal_overlap_ratio",
            default=_DEFAULT_SUPPORT_MIN_HORIZONTAL_OVERLAP_RATIO,
        )
        if "support_collision_selectors" in metadata:
            selectors = self._metadata_selectors(
                request, "support_collision_selectors"
            )
            evidence: dict[str, Any] = {
                "policy": str(
                    metadata.get(
                        "support_collision_policy", "EXPLICIT_TASK_METADATA"
                    )
                ),
                "detection_source": str(
                    metadata.get(
                        "support_collision_detection_source",
                        "task.metadata.support_collision_selectors",
                    )
                ),
            }
            for key in (
                "support_initial_clearance_m",
                "support_horizontal_overlap_ratio",
                "support_min_horizontal_overlap_ratio",
            ):
                if key in metadata:
                    evidence[key] = metadata[key]
        else:
            tolerance_m = self._metadata_nonnegative_distance(
                request,
                "support_contact_tolerance_m",
                default=request.constraints.collision_margin_m,
            )
            from tuj.m5_motion.grasp_geometry import (
                support_clearance_context_from_world,
            )

            support = support_clearance_context_from_world(
                request.world.objects.get(target),
                request.world,
                target,
                tolerance_m=tolerance_m,
                minimum_horizontal_overlap_ratio=minimum_overlap_ratio,
            )
            if (
                support is None
                or support.support_id == target
                or abs(support.under_clearance_m) > tolerance_m + 1e-9
                or (
                    support.source
                    in {"world.obstacles.aabb", "world.objects.obb"}
                    and support.horizontal_overlap_ratio + 1e-9
                    < minimum_overlap_ratio
                )
            ):
                return [], {}, 0.0
            selectors = [support.support_id]
            evidence = {
                "policy": "AUTO_INITIAL_SUPPORT_V1",
                "detection_source": support.source,
                "support_initial_clearance_m": support.under_clearance_m,
                "support_horizontal_overlap_ratio": (
                    support.horizontal_overlap_ratio
                ),
                "support_min_horizontal_overlap_ratio": (
                    minimum_overlap_ratio
                ),
            }

        if not selectors:
            return [], evidence, 0.0
        if any(has_magic(selector) for selector in selectors):
            raise ToolUseJournalCollisionBindingError(
                "support_collision_selectors must use exact selectors, not patterns"
            )
        if target in selectors:
            raise ToolUseJournalCollisionBindingError(
                "support_collision_selectors cannot contain the PICK target"
            )
        if evidence.get("policy") == "AUTO_INITIAL_SUPPORT_V1":
            raw_overlap = evidence.get("support_horizontal_overlap_ratio")
            if (
                isinstance(raw_overlap, bool)
                or not isinstance(raw_overlap, (int, float))
                or not math.isfinite(float(raw_overlap))
                or float(raw_overlap) + 1e-9 < minimum_overlap_ratio
            ):
                raise ToolUseJournalCollisionBindingError(
                    "automatically inferred PICK support has insufficient "
                    "horizontal overlap"
                )
        maximum_penetration_m = min(
            _MAX_SUPPORT_PENETRATION_TOLERANCE_M,
            request.constraints.collision_margin_m,
        )
        raw_initial_clearance = evidence.get("support_initial_clearance_m")
        if (
            isinstance(raw_initial_clearance, (int, float))
            and not isinstance(raw_initial_clearance, bool)
            and math.isfinite(float(raw_initial_clearance))
        ):
            initial_penetration_m = max(0.0, -float(raw_initial_clearance))
            if initial_penetration_m > maximum_penetration_m + 1e-9:
                raise ToolUseJournalCollisionBindingError(
                    "initial support penetration exceeds the 1 mm hard limit"
                )
        penetration_tolerance_m = self._metadata_nonnegative_distance(
            request,
            "support_penetration_tolerance_m",
            default=maximum_penetration_m,
        )
        if penetration_tolerance_m > maximum_penetration_m + 1e-9:
            raise ToolUseJournalCollisionBindingError(
                "support_penetration_tolerance_m cannot exceed the smaller of "
                "1 mm and collision_margin_m"
            )
        evidence["maximum_penetration_m"] = penetration_tolerance_m
        return selectors, evidence, penetration_tolerance_m

    def _bind_generic_touch_policy(
        self,
        request: MotionPlanRequest,
        base: CollisionContext,
    ) -> CollisionContext:
        if not base.attached_object_ids or not request.task.allowed_touch_objects:
            return base
        pairs = {
            *base.allowed_collision_pairs,
            *(
                tuple(sorted((attached, selector)))
                for attached in base.attached_object_ids
                for selector in request.task.allowed_touch_objects
                if selector
            ),
        }
        return base.model_copy(update={"allowed_collision_pairs": sorted(pairs)})

    def _bind_default(
        self,
        request: MotionPlanRequest,
        artifact: KeyframePlanArtifact,
        base: CollisionContext,
    ) -> tuple[KeyframePlanArtifact, dict[str, CollisionContext]]:
        source = artifact
        bound = artifact.model_copy(deep=True)
        contexts = {base.context_id: base}
        from .flatten_contact import is_flatten_contact
        contact_id = None
        if is_flatten_contact(request.task) and len(request.task.target_ids) == 1:
            target = request.task.target_ids[0]
            config = request.world.objects[target].get("flattening_configuration", {})
            if config.get("allow_active_ee_contact") is True:
                contact_id = base.context_id + ":material-contact"
                contexts[contact_id] = base.model_copy(update={
                    "context_id": contact_id,
                    "allowed_collision_pairs": sorted({
                        *base.allowed_collision_pairs,
                        tuple(sorted((base.active_ee, target))),
                    }),
                })
        for candidate in bound.candidates:
            current_id = base.context_id
            for keyframe in candidate.keyframes:
                selected = keyframe.collision_context_id or current_id
                if contact_id is not None:
                    # Incoming engagement and withdrawal edges include intentional
                    # contact. Hover and subsequent unrelated edges retain the base.
                    selected = (contact_id if keyframe.metadata.get("flatten_stage")
                                in {"engage", "press", "unload"} else base.context_id)
                if selected not in contexts:
                    raise ToolUseJournalCollisionBindingError(
                        f"strategy {candidate.strategy_id!r} supplies unknown "
                        f"collision context {selected!r}"
                    )
                keyframe.collision_context_id = selected
                after_id = keyframe.collision_context_after_events_id
                if after_id is not None:
                    if after_id not in contexts:
                        raise ToolUseJournalCollisionBindingError(
                            f"strategy {candidate.strategy_id!r} supplies unknown "
                            f"post-event collision context {after_id!r}"
                        )
                    current_id = after_id
                else:
                    current_id = selected
        return _stamp_bound_artifact(request, source, bound), contexts

    def _bind_pick(
        self,
        request: MotionPlanRequest,
        artifact: KeyframePlanArtifact,
        base: CollisionContext,
        active_ee: str,
    ) -> tuple[KeyframePlanArtifact, dict[str, CollisionContext]]:
        if request.world.robot_state.attached_object_id is not None:
            raise ToolUseJournalCollisionBindingError(
                "PICK cannot start while another object is attached"
            )
        target = request.task.goal.target_object_id
        if not target:
            raise ToolUseJournalCollisionBindingError("PICK has no target object")
        _free_joint_name(request.world, target)
        from tuj.m5_motion.support_distance import refine_support_request

        request = refine_support_request(request, self.compiler)
        from tuj.m5_motion.physical_grasp import uses_contact_friction

        if uses_contact_friction(request):
            return self._bind_contact_friction_pick(
                request,
                artifact,
                base,
                active_ee,
                target=target,
            )
        source = artifact
        bound = artifact.model_copy(deep=True)
        contexts: dict[str, CollisionContext] = {base.context_id: base}
        touch_selectors = [target, *request.task.allowed_touch_objects]
        (
            support_selectors,
            support_evidence,
            support_penetration_tolerance_m,
        ) = self._pick_support_policy(request, target)
        for candidate in bound.candidates:
            grasp = self._event_keyframe(
                candidate,
                KeyframeEventType.ATTACH_OBJECT,
                KeyframeType.GRASP,
            )
            grasp_index = candidate.keyframes.index(grasp)
            release_index = grasp_index + 1
            if support_selectors:
                if release_index >= len(candidate.keyframes):
                    raise ToolUseJournalCollisionBindingError(
                        f"strategy {candidate.strategy_id!r} declares PICK support "
                        "contact but has no post-grasp separation keyframe"
                    )
                release_keyframe = candidate.keyframes[release_index]
                if release_keyframe.keyframe_type not in {
                    KeyframeType.LIFT,
                    KeyframeType.RETREAT,
                }:
                    raise ToolUseJournalCollisionBindingError(
                        f"strategy {candidate.strategy_id!r} must use LIFT or "
                        "RETREAT immediately after a supported GRASP"
                    )
            token = _short_digest((candidate.strategy_id, grasp.keyframe_id))
            contact_id = f"grasp-contact:{target}:{token}"
            attached_id = f"object-attached:{target}:{token}"
            release_id = f"object-attached-release:{target}:{token}"
            # 0911: 파지 중 문맥은 [EE, 대상] 만 면제하고 받침에는 5mm 마진을
            # 그대로 요구했다. 얇은 물체를 집을 때 이 요구는 원리상 만족이
            # 불가능하다 -- 대상 윗면에 EE 를 대면 받침은 대상 두께만큼 아래에
            # 있기 때문이다. c2_2 의 turkey_3(두께 2.5mm)은 바로 아래 turkey_2
            # 와의 여유가 -0.001847 / -0.000082 / +0.000633 m 로 나와 생성된
            # 전략이 전부 COLLISION_FILTERED_ALL 로 기각됐고, 접시 위 단독
            # 슬라이스도 같은 벽에 부딪힌다. 파지 후 문맥은 이미 [대상, 받침]
            # 을 1mm 관통까지 허용하므로 같은 장치를 파지 중에도 건다.
            # 무제한 면제가 아니라 EE 가 받침을 뚫고 들어가는 것은 계속 막는다.
            contact_metadata: dict[str, Any] = {}
            if support_selectors:
                contact_metadata[_BOUNDED_COLLISION_ALLOWANCES_KEY] = [
                    {
                        "selectors": [active_ee, selector],
                        "minimum_distance_m": -support_penetration_tolerance_m,
                    }
                    for selector in support_selectors
                ]
            contact = base.model_copy(
                update={
                    "context_id": contact_id,
                    "allowed_collision_pairs": self._contact_pairs(
                        active_ee, [*touch_selectors, *support_selectors]
                    ),
                    "metadata": contact_metadata,
                }
            )
            transform = _relative_attachment(
                request,
                grasp,
                object_id=target,
                reference_kind=self.attachment_reference_kind,
                reference_name=self.attachment_reference_name,
            )
            attached = CollisionContext(
                context_id=attached_id,
                scene_state_id=f"{base.scene_state_id}:attached:{target}:{token}",
                active_ee=active_ee,
                attached_object_ids=[target],
                attached_object_transforms=[transform],
                free_object_poses=_free_object_poses(
                    request.world, exclude=(target,)
                ),
                touch_links=[active_ee],
                collision_model_version=self.compiler.model_version_for(active_ee),
            )
            contexts[contact_id] = contact
            contexts[attached_id] = attached
            if support_selectors:
                # 0912: 이 문맥은 [대상, 받침] 만 면제하고 [EE, 받침] 에는 5mm
                # 마진을 그대로 요구했다. 얇은 물체를 받침에서 떼는 동안 EE 는
                # 물체 두께만큼만 받침에서 떨어져 있으므로 원리상 만족이
                # 불가능하다. c2_2 의 turkey_1(두께 2.5mm)을 흡착으로 들어올릴
                # 때 흡착컵과 접시 여유가 0.000439 / 0.000737 / 0.001557 /
                # 0.001993 m 로 나와 생성된 전략이 전부 기각됐다. 파지 중 문맥이
                # 0911 에 같은 이유로 이미 하는 처리를 파지 후에도 건다. 무제한
                # 면제가 아니라 EE 가 받침을 뚫는 것은 계속 막는다.
                support_allowances = [
                    {
                        "selectors": [selector_a, selector],
                        "minimum_distance_m": -support_penetration_tolerance_m,
                    }
                    for selector_a in (target, active_ee)
                    for selector in support_selectors
                ]
                release = attached.model_copy(
                    update={
                        "context_id": release_id,
                        "allowed_collision_pairs": [
                            *self._contact_pairs(target, support_selectors),
                            *self._contact_pairs(active_ee, support_selectors),
                        ],
                        "metadata": {
                            "support_separation": {
                                **support_evidence,
                                "target_selector": target,
                                "support_selectors": list(support_selectors),
                            },
                            _BOUNDED_COLLISION_ALLOWANCES_KEY: support_allowances,
                            _POST_SEGMENT_VALIDATION_CONTEXT_KEY: attached_id,
                        },
                    }
                )
                contexts[release_id] = release
            else:
                release_id = attached_id
            for keyframe_index, keyframe in enumerate(candidate.keyframes):
                keyframe.collision_context_after_events_id = None
                if keyframe_index < grasp_index:
                    keyframe.collision_context_id = base.context_id
                elif keyframe_index == grasp_index:
                    keyframe.collision_context_id = contact_id
                    keyframe.collision_context_after_events_id = release_id
                elif support_selectors and keyframe_index == release_index:
                    keyframe.collision_context_id = release_id
                    keyframe.collision_context_after_events_id = attached_id
                    keyframe.metadata = {
                        **keyframe.metadata,
                        "validate_endpoint_after_context": True,
                    }
                else:
                    keyframe.collision_context_id = attached_id
        return _stamp_bound_artifact(request, source, bound), contexts

    def _bind_contact_friction_pick(
        self,
        request: MotionPlanRequest,
        artifact: KeyframePlanArtifact,
        base: CollisionContext,
        active_ee: str,
        *,
        target: str,
    ) -> tuple[KeyframePlanArtifact, dict[str, CollisionContext]]:
        """Keep a PICK target physically free while allowing finger contact.

        A contact-friction grasp has no ATTACH_OBJECT transition.  The target
        remains a MuJoCo free body throughout planning and execution, so the
        gripper controller and contact friction are solely responsible for
        lifting it.
        """

        source = artifact
        bound = artifact.model_copy(deep=True)
        contexts: dict[str, CollisionContext] = {base.context_id: base}
        touch_selectors = [target, *request.task.allowed_touch_objects]
        for candidate in bound.candidates:
            grasp = self._event_keyframe(
                candidate,
                KeyframeEventType.GRIPPER_CLOSE,
                KeyframeType.GRASP,
            )
            if any(
                KeyframeEventType.ATTACH_OBJECT in keyframe.events_after
                for keyframe in candidate.keyframes
            ):
                raise ToolUseJournalCollisionBindingError(
                    f"contact-friction strategy {candidate.strategy_id!r} "
                    "must not contain ATTACH_OBJECT"
                )
            planning_transform = _relative_attachment(
                request,
                grasp,
                object_id=target,
                reference_kind=self.attachment_reference_kind,
                reference_name=self.attachment_reference_name,
            )
            grasp.metadata = {
                **grasp.metadata,
                "planned_contact_friction_transform": (
                    planning_transform.model_dump(mode="json")
                ),
            }
            token = _short_digest((candidate.strategy_id, grasp.keyframe_id))
            contact_id = f"physical-grasp-contact:{target}:{token}"
            contact = base.model_copy(
                update={
                    "context_id": contact_id,
                    "allowed_collision_pairs": self._contact_pairs(
                        active_ee, touch_selectors
                    ),
                }
            )
            contexts[contact_id] = contact
            grasp_index = candidate.keyframes.index(grasp)
            for keyframe_index, keyframe in enumerate(candidate.keyframes):
                keyframe.collision_context_after_events_id = None
                keyframe.collision_context_id = (
                    contact_id
                    if keyframe_index >= grasp_index
                    else base.context_id
                )
        return _stamp_bound_artifact(request, source, bound), contexts

    def _bind_place(
        self,
        request: MotionPlanRequest,
        artifact: KeyframePlanArtifact,
        base: CollisionContext,
        active_ee: str,
    ) -> tuple[KeyframePlanArtifact, dict[str, CollisionContext]]:
        target = request.task.goal.target_object_id
        if not target:
            raise ToolUseJournalCollisionBindingError("PLACE has no target object")
        from tuj.m5_motion.physical_grasp import releases_contact_friction

        physical_release = releases_contact_friction(request)
        if (
            request.world.robot_state.attached_object_id != target
            and not physical_release
        ):
            raise ToolUseJournalCollisionBindingError(
                f"PLACE target {target!r} is not the currently attached object"
            )
        target_pose = request.task.goal.target_pose
        if target_pose is None or target_pose.frame_id != "world":
            raise ToolUseJournalCollisionBindingError(
                "PLACE requires a world-frame target pose"
            )
        source = artifact
        bound = artifact.model_copy(deep=True)
        contexts: dict[str, CollisionContext] = {base.context_id: base}
        contact_selectors = list(request.task.allowed_touch_objects)
        if request.task.goal.target_region_id:
            contact_selectors.append(request.task.goal.target_region_id)
            # Objects already packed into the destination region may be
            # contacted (or rested upon) by the held object -- the task packs
            # everything into one region and permits overlap -- so admit each
            # occupant as an allowed collision pair for the PLACE contact.
            for occupant in self._region_occupant_ids(request, target):
                if occupant not in contact_selectors:
                    contact_selectors.append(occupant)
        for candidate in bound.candidates:
            place = self._event_keyframe(
                candidate,
                (
                    KeyframeEventType.GRIPPER_OPEN
                    if physical_release
                    else KeyframeEventType.DETACH_OBJECT
                ),
                KeyframeType.PLACE,
            )
            token = _short_digest((candidate.strategy_id, place.keyframe_id))
            resolved_place = RelativePoseResolver(request.world).resolve(place)
            if (
                str(place.metadata.get(POSE_SUBJECT_KEY, "")).upper()
                == ATTACHED_OBJECT_POSE_SUBJECT
                and place.metadata.get(POSE_SUBJECT_OBJECT_ID_KEY) == target
            ):
                # The symbolic PLACE keyframe describes the held object's
                # desired pose.  Freeze the newly detached collision body at
                # that candidate-specific pose, not at the stale task goal
                # pose captured before keyframe generation.
                detached_target_pose = resolved_place
            else:
                # Untagged generated keyframes describe the grasp reference,
                # not the object. Keep geometry continuous across DETACH.
                detached_target_pose = object_pose_for_end_effector_pose(
                    resolved_place, attachment_transform(request.world, target)
                )
            contact_id = f"place-contact:{target}:{token}"
            detached_id = f"object-detached:{target}:{token}"
            contact = base.model_copy(
                update={
                    "context_id": contact_id,
                    "allowed_collision_pairs": self._contact_pairs(
                        target, contact_selectors
                    ),
                }
            )
            detached = CollisionContext(
                context_id=detached_id,
                scene_state_id=f"{base.scene_state_id}:detached:{target}:{token}",
                active_ee=active_ee,
                free_object_poses=_free_object_poses(
                    request.world,
                    overrides={target: detached_target_pose},
                ),
                collision_model_version=self.compiler.model_version_for(active_ee),
            )
            contexts[contact_id] = contact
            contexts[detached_id] = detached
            # Fingers still touch the now-free object while opening. Allow
            # only that target contact through the first withdrawal edge;
            # subsequent motion uses the ordinary detached context.
            release_id = detached_id
            if place.metadata.get("allow_release_contact") is True:
                release_id = f"object-release-contact:{target}:{token}"
                contexts[release_id] = detached.model_copy(update={
                    "context_id": release_id,
                    "allowed_collision_pairs": self._contact_pairs(active_ee, [target]),
                })
            # Approach keyframes (TRANSFER/PRE_PLACE) carry the held object over
            # an increasingly crowded region.  The task packs everything into one
            # region and permits overlap, so let the held object contact the
            # region and its current occupants throughout the descent -- not only
            # at the final PLACE pose -- otherwise the approach into a dense tray
            # is collision-filtered before it can reach the drop point.  For a
            # region with no occupants this context only adds the region/touch
            # pairs, matching the prior place-near-region behaviour.
            current_id = contact_id
            withdrawal_pending = False
            for keyframe in candidate.keyframes:
                keyframe.collision_context_after_events_id = None
                if keyframe is place:
                    keyframe.collision_context_id = contact_id
                    keyframe.collision_context_after_events_id = release_id
                    current_id = release_id
                    withdrawal_pending = release_id != detached_id
                else:
                    keyframe.collision_context_id = current_id
                    if withdrawal_pending:
                        current_id = detached_id
                        withdrawal_pending = False
        from tuj.m5_motion.release_separation import bind_release_separation

        bind_release_separation(self.compiler, request, bound, contexts, target)
        # Withdrawal from a crowded region.  After the held object is released,
        # the empty gripper retreats past objects already packed into the same
        # region.  The descent already tolerates the HELD object contacting
        # those occupants (the task packs everything into one region and permits
        # overlap); extend the identical tolerance to the withdrawing GRIPPER so
        # a post-release retreat that grazes a neighbour -- e.g. an open 2F
        # finger passing 2.5 mm from an already-placed spoon while lifting away
        # -- is not collision-filtered.  Scoped to the release/retreat contexts
        # of this place only (every keyframe after the PLACE); approach,
        # transport and free-space margins are unchanged, and a single-object
        # region has no occupants so those tasks are untouched.
        occupants = self._region_occupant_ids(request, target)
        occupant_pairs = self._contact_pairs(active_ee, occupants) if occupants else []
        if occupant_pairs:
            for candidate in bound.candidates:
                place_kf = next(
                    (
                        keyframe
                        for keyframe in candidate.keyframes
                        if keyframe.keyframe_type is KeyframeType.PLACE
                    ),
                    None,
                )
                if place_kf is None:
                    continue
                start = candidate.keyframes.index(place_kf) + 1
                retreat_context_ids = {
                    context_id
                    for keyframe in candidate.keyframes[start:]
                    for context_id in (
                        keyframe.collision_context_id,
                        keyframe.collision_context_after_events_id,
                    )
                    if context_id in contexts
                }
                for context_id in retreat_context_ids:
                    ctx = contexts[context_id]
                    merged = list(ctx.allowed_collision_pairs)
                    for pair in occupant_pairs:
                        if pair not in merged:
                            merged.append(pair)
                    contexts[context_id] = ctx.model_copy(
                        update={"allowed_collision_pairs": merged}
                    )
        return _stamp_bound_artifact(request, source, bound), contexts

    def _bind_ee_exchange(
        self,
        request: MotionPlanRequest,
        artifact: KeyframePlanArtifact,
    ) -> tuple[
        KeyframePlanArtifact,
        dict[str, CollisionContext],
        str,
        str | None,
    ]:
        if (
            request.world.robot_state.attached_object_id is not None
            or request.world.robot_state.held_tool_id is not None
        ):
            raise ToolUseJournalCollisionBindingError(
                "EE_EXCHANGE requires an empty end effector"
            )
        raw_from_ee = request.task.metadata.get("from_ee")
        from_ee = str(raw_from_ee) if raw_from_ee else None
        to_ee = str(request.task.metadata.get("to_ee") or "")
        raw_active_ee = request.world.metadata.get("physical_active_ee")
        active_ee = raw_active_ee if isinstance(raw_active_ee, str) else None
        if not to_ee or active_ee != from_ee:
            raise ToolUseJournalCollisionBindingError(
                "EE transition from_ee/to_ee does not match the current workcell"
            )
        contexts = self.compiler.build_ee_exchange_contexts(
            from_ee=from_ee,
            to_ee=to_ee,
        )
        free_poses = _free_object_poses(request.world)
        contexts = {
            context_id: context.model_copy(
                update={"free_object_poses": free_poses}
            )
            for context_id, context in contexts.items()
        }
        self._verify_context_references(artifact, contexts)
        initial_id = f"ee-attached:{from_ee}" if from_ee else "bare-flange"
        return artifact, contexts, initial_id, from_ee

    def prepare_precomputed_ee_attach(
        self,
        request: MotionPlanRequest,
    ) -> tuple[Mapping[str, CollisionContext], object]:
        """Build current-scene collision models without generating keyframes."""

        self._validate_environment(request)
        if not is_initial_ee_attach(request):
            raise ToolUseJournalCollisionBindingError(
                "precomputed EE attach setup requires an initial bare EE_ATTACH"
            )
        if (
            request.world.robot_state.attached_object_id is not None
            or request.world.robot_state.held_tool_id is not None
        ):
            raise ToolUseJournalCollisionBindingError(
                "EE_ATTACH requires an empty bare flange"
            )
        try:
            to_ee = normalize_ee_id(
                request.task.metadata.get("to_ee") or request.task.ee
            )
        except ValueError as error:
            raise ToolUseJournalCollisionBindingError(str(error)) from error
        contexts = self.compiler.build_ee_exchange_contexts(
            from_ee=None,
            to_ee=to_ee,
        )
        free_poses = _free_object_poses(request.world)
        contexts = {
            context_id: context.model_copy(
                update={"free_object_poses": free_poses}
            )
            for context_id, context in contexts.items()
        }
        try:
            registry = self.compiler.build_collision_registry(
                contexts,
                collision_margin_m=request.constraints.collision_margin_m,
                allowed_collision_pairs=(
                    request.constraints.allowed_collision_pairs
                ),
                default_active_ee=None,
            )
        except ToolUseJournalCompatibilityError:
            raise
        except Exception as error:  # noqa: BLE001
            raise ToolUseJournalCollisionBindingError(
                "failed to build precomputed EE attach collision registry"
            ) from error
        return contexts, registry

    def prepare_precomputed_ee_exchange(
        self,
        request: MotionPlanRequest,
    ) -> tuple[Mapping[str, CollisionContext], object]:
        """Build the current-scene collision variants for a composed exchange."""

        self._validate_environment(request)
        if not is_ee_exchange_request(request):
            raise ToolUseJournalCollisionBindingError(
                "precomputed EE exchange setup requires an attached-EE EE_EXCHANGE"
            )
        if (
            request.world.robot_state.attached_object_id is not None
            or request.world.robot_state.held_tool_id is not None
        ):
            raise ToolUseJournalCollisionBindingError(
                "EE_EXCHANGE requires an empty end effector"
            )
        try:
            from_ee = normalize_ee_id(request.task.metadata.get("from_ee"))
            to_ee = normalize_ee_id(
                request.task.metadata.get("to_ee") or request.task.ee
            )
        except ValueError as error:
            raise ToolUseJournalCollisionBindingError(str(error)) from error
        contexts = self.compiler.build_ee_exchange_contexts(
            from_ee=from_ee,
            to_ee=to_ee,
        )
        free_poses = _free_object_poses(request.world)
        contexts = {
            context_id: context.model_copy(
                update={"free_object_poses": free_poses}
            )
            for context_id, context in contexts.items()
        }
        try:
            registry = self.compiler.build_collision_registry(
                contexts,
                collision_margin_m=request.constraints.collision_margin_m,
                allowed_collision_pairs=(
                    request.constraints.allowed_collision_pairs
                ),
                default_active_ee=from_ee,
            )
        except ToolUseJournalCompatibilityError:
            raise
        except Exception as error:  # noqa: BLE001
            raise ToolUseJournalCollisionBindingError(
                "failed to build precomputed EE exchange collision registry"
            ) from error
        return contexts, registry

    def prepare_ee_exchange_entry(
        self,
        request: MotionPlanRequest,
    ) -> tuple[Mapping[str, CollisionContext], object]:
        """Build attached-EE collision models for the entry positioning leg."""

        self._validate_environment(request)
        if not is_ee_exchange_entry_request(request):
            raise ToolUseJournalCollisionBindingError(
                "exchange-entry setup requires EE_EXCHANGE_ENTRY"
            )
        if (
            request.world.robot_state.attached_object_id is not None
            or request.world.robot_state.held_tool_id is not None
        ):
            raise ToolUseJournalCollisionBindingError(
                "EE_EXCHANGE_ENTRY requires an empty end effector"
            )
        try:
            source = normalize_ee_id(
                request.task.metadata.get("entry_ee")
                or request.task.metadata.get("from_ee")
                or request.task.ee
            )
            target = normalize_ee_id(request.task.metadata.get("next_ee"))
            physical = normalize_ee_id(
                request.world.metadata.get("physical_active_ee")
            )
        except ValueError as error:
            raise ToolUseJournalCollisionBindingError(str(error)) from error
        if source != physical or source == target:
            raise ToolUseJournalCollisionBindingError(
                "exchange-entry source/target does not match the mounted EE transition"
            )
        contexts = self.compiler.build_ee_exchange_contexts(
            from_ee=source,
            to_ee=target,
        )
        free_poses = _free_object_poses(request.world)
        contexts = {
            context_id: context.model_copy(
                update={"free_object_poses": free_poses}
            )
            for context_id, context in contexts.items()
        }
        try:
            registry = self.compiler.build_collision_registry(
                contexts,
                collision_margin_m=request.constraints.collision_margin_m,
                allowed_collision_pairs=(
                    request.constraints.allowed_collision_pairs
                ),
                default_active_ee=source,
            )
        except ToolUseJournalCompatibilityError:
            raise
        except Exception as error:  # noqa: BLE001
            raise ToolUseJournalCollisionBindingError(
                "failed to build EE exchange-entry collision registry"
            ) from error
        return contexts, registry

    def prepare(
        self,
        request: MotionPlanRequest,
        artifact: KeyframePlanArtifact,
    ) -> CollisionPlanningSetup:
        self._validate_environment(request)
        if is_ee_exchange_task(request.task):
            bound, contexts, initial_id, default_ee = self._bind_ee_exchange(
                request, artifact
            )
        else:
            active_ee = self._active_ee(request)
            if request.task.ee != active_ee:
                raise ToolUseJournalCollisionBindingError(
                    f"task requests EE {request.task.ee!r}, but world has "
                    f"{active_ee!r} physically active"
                )
            base = self._base_context(request, active_ee=active_ee)
            if is_acquire_task(request.task):
                bound, contexts = self._bind_pick(
                    request, artifact, base, active_ee
                )
            elif is_release_task(request.task):
                bound, contexts = self._bind_place(
                    request, artifact, base, active_ee
                )
            else:
                base = self._bind_generic_touch_policy(request, base)
                bound, contexts = self._bind_default(request, artifact, base)
            initial_id = base.context_id
            default_ee = active_ee
        self._verify_context_references(bound, contexts)
        try:
            registry = self.compiler.build_collision_registry(
                contexts,
                collision_margin_m=request.constraints.collision_margin_m,
                allowed_collision_pairs=(
                    request.constraints.allowed_collision_pairs
                ),
                default_active_ee=default_ee,
            )
        except ToolUseJournalCompatibilityError:
            raise
        except Exception as error:  # noqa: BLE001 - normalize plugin/compiler errors
            raise ToolUseJournalCollisionBindingError(
                "failed to build the request collision registry"
            ) from error
        return CollisionPlanningSetup(
            keyframe_artifact=bound,
            state_validator=registry,
            collision_contexts=contexts,
            initial_collision_context_id=initial_id,
            final_segment_validator=registry.final_segment_validator,
        )


class ToolUseJournalMotionRequestPlanner:
    """Callable ``plan_one_request`` bound to one workcell environment."""

    def __init__(
        self,
        pipeline: MotionPlanningPipeline,
        collision_context_factory: ToolUseJournalCollisionContextFactory,
        *,
        precomputed_ee_attach_planner: PrecomputedEEAttachPlanner | None = None,
        precomputed_ee_exchange_planner: PrecomputedEEExchangePlanner | None = None,
        ee_exchange_entry_planner: EEExchangeEntryPlanner | None = None,
        ee_attach_policy: EEAttachPolicy | str = EEAttachPolicy.PRECOMPUTED_REQUIRED,
        log: Any = print,
    ) -> None:
        self.pipeline = pipeline
        self.collision_context_factory = collision_context_factory
        self.precomputed_ee_attach_planner = precomputed_ee_attach_planner
        self.precomputed_ee_exchange_planner = precomputed_ee_exchange_planner
        self.ee_exchange_entry_planner = ee_exchange_entry_planner
        self.ee_attach_policy = EEAttachPolicy(ee_attach_policy)
        self._log = log

    @property
    def environment_name(self) -> str:
        return self.collision_context_factory.compiler.environment_name

    @classmethod
    def from_environment(
        cls,
        env: object,
        repository_root: str | Path,
        *,
        provider: KeyframeStrategyProvider | None = None,
        seed: int = 0,
        ee_attach_registry_root: str | Path | None = None,
        ee_attach_trajectory_paths: Sequence[str | Path] = (),
        ee_return_trajectory_paths: Sequence[str | Path] = (),
        ee_attach_policy: EEAttachPolicy | str = EEAttachPolicy.PRECOMPUTED_REQUIRED,
        ee_attach_start_tolerance_rad: float = 0.01,
        log: Any = print,
        **suite_make_kwargs: Any,
    ) -> "ToolUseJournalMotionRequestPlanner":
        adapter = ToolUseJournalEnvironmentAdapter(env)
        adapter.require_physical_ee()
        compiler = ToolUseJournalCollisionModelCompiler.from_repository(
            env,
            repository_root,
            seed=seed,
            **suite_make_kwargs,
        )
        selected_provider = provider or OpenAIKeyframeProvider()
        routed_provider = (
            selected_provider
            if isinstance(selected_provider, RoutedKeyframeStrategyProvider)
            else RoutedKeyframeStrategyProvider(selected_provider)
        )
        # The routed provider still owns geometry generation.  This decorator
        # changes only explicitly selected physical PICK requests from a
        # synthetic ATTACH_OBJECT event to persistent contact friction.
        from tuj.m5_motion.physical_grasp import ContactFrictionKeyframeProvider

        execution_provider = ContactFrictionKeyframeProvider(routed_provider)
        kinematics = adapter.make_kinematics()
        if getattr(env, "scripted_grasp_profile", None) is not None:
            # Continue from the physical grasp pose. Tight IK tolerances leave
            # room for controller error inside M5's unchanged goal tolerance.
            from .scripted_grasps.ik_continuity import ContinuousIK
            kinematics = ContinuousIK(
                kinematics, adapter.data.qpos[adapter.robot._ref_joint_pos_indexes]
            )
        pipeline = MotionPlanningPipeline(execution_provider, kinematics)
        # Portable (cross-environment) EE-path validation compares a template's
        # stored canonical EEF pose against current-model forward kinematics.
        # Every portable template records that pose in the bare-flange frame
        # (it is commissioned with the no-gripper kinematics), so validation
        # must evaluate the flange too.  ``kinematics`` above targets the
        # mounted EE's grip site when one is mounted (needed for IK), which for
        # a return/exchange template would offset the check by the whole
        # gripper mount (~0.14 m) and reject a geometrically identical rack.
        # Build a dedicated EE-independent flange FK for validation only; it is
        # never used for IK or planning.
        from tuj.m5_motion.kinematics import UR5eKinematics

        try:
            portable_validation_kinematics: Any = UR5eKinematics.from_robosuite_env(
                env
            )
        except Exception:  # noqa: BLE001 - fall back to the planning kinematics
            portable_validation_kinematics = kinematics
        factory = ToolUseJournalCollisionContextFactory(
            compiler,
            attachment_reference_name=(
                str(
                    next(iter(adapter.robot.gripper.values())).important_sites[
                        "grip_site"
                    ]
                )
                if isinstance(adapter.robot.gripper, Mapping)
                and adapter.robot.gripper
                and "grip_site"
                in next(iter(adapter.robot.gripper.values())).important_sites
                else adapter.hand_body
            ),
            attachment_reference_kind=(
                "site"
                if isinstance(adapter.robot.gripper, Mapping)
                and adapter.robot.gripper
                and "grip_site"
                in next(iter(adapter.robot.gripper.values())).important_sites
                else "body"
            ),
        )
        registry_root = (
            Path(ee_attach_registry_root)
            if ee_attach_registry_root is not None
            else Path(repository_root) / "configs" / "precomputed_ee_paths"
        )
        precomputed = PrecomputedEEAttachPlanner(
            PrecomputedEEAttachRegistry(
                registry_root,
                trajectory_paths=ee_attach_trajectory_paths,
            ),
            forward_kinematics=portable_validation_kinematics,
            start_tolerance_rad=ee_attach_start_tolerance_rad,
            joint_position_limits_rad=getattr(
                kinematics, "joint_limits_rad", None
            ),
            log=log,
        )
        return_registry = PrecomputedEEReturnRegistry(
            registry_root,
            trajectory_paths=ee_return_trajectory_paths,
        )
        precomputed_exchange = PrecomputedEEExchangePlanner(
            return_registry,
            precomputed,
            log=log,
        )
        entry_planner = EEExchangeEntryPlanner(
            return_registry,
            joint_position_limits_rad=getattr(kinematics, "joint_limits_rad"),
            log=log,
        )
        return cls(
            pipeline,
            factory,
            precomputed_ee_attach_planner=precomputed,
            precomputed_ee_exchange_planner=precomputed_exchange,
            ee_exchange_entry_planner=entry_planner,
            ee_attach_policy=ee_attach_policy,
            log=log,
        )

    def __call__(self, request: MotionPlanRequest, *, final_plan_validator=None) -> Any:
        if is_ee_exchange_entry_request(request):
            source = str(
                request.task.metadata.get("entry_ee")
                or request.task.metadata.get("from_ee")
                or request.task.ee
            )
            if self.ee_exchange_entry_planner is None:
                raise PrecomputedEEPathError(
                    EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_NOT_FOUND,
                    "no EE exchange-entry planner is configured",
                )
            try:
                template = self.ee_exchange_entry_planner.load(request)
                contexts, collision_registry = (
                    self.collision_context_factory.prepare_ee_exchange_entry(
                        request
                    )
                )
                return self.ee_exchange_entry_planner.plan(
                    request,
                    collision_contexts=contexts,
                    collision_checker=collision_registry,
                    template=template,
                )
            except PrecomputedEEPathError as error:
                self._log(
                    f"[M5][EE_ENTRY] miss: {source} "
                    f"code={error.failure_code.value}"
                )
                raise
        if is_initial_ee_attach(request):
            target = str(request.task.metadata.get("to_ee") or request.task.ee)
            try:
                if self.precomputed_ee_attach_planner is None:
                    raise PrecomputedEEPathError(
                        EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_NOT_FOUND,
                        "no precomputed EE attach registry is configured",
                    )
                template = self.precomputed_ee_attach_planner.load(request)
                contexts, collision_registry = (
                    self.collision_context_factory.prepare_precomputed_ee_attach(
                        request
                    )
                )
                return self.precomputed_ee_attach_planner.plan(
                    request,
                    collision_contexts=contexts,
                    collision_checker=collision_registry,
                    template=template,
                )
            except PrecomputedEEPathError as error:
                self._log(
                    f"[M5][EE_PATH] miss: bare->{target} "
                    f"code={error.failure_code.value}"
                )
                if self.ee_attach_policy is EEAttachPolicy.PRECOMPUTED_REQUIRED:
                    raise
                self._log("[M5][EE_PATH] fallback=dynamic-planner")
        elif is_ee_exchange_request(request):
            source = str(request.task.metadata.get("from_ee") or "")
            target = str(request.task.metadata.get("to_ee") or request.task.ee)
            try:
                if self.precomputed_ee_exchange_planner is None:
                    raise PrecomputedEEPathError(
                        EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_NOT_FOUND,
                        "no precomputed EE exchange registry is configured",
                    )
                templates = self.precomputed_ee_exchange_planner.load(request)
                contexts, collision_registry = (
                    self.collision_context_factory.prepare_precomputed_ee_exchange(
                        request
                    )
                )
                return self.precomputed_ee_exchange_planner.plan(
                    request,
                    collision_contexts=contexts,
                    collision_checker=collision_registry,
                    templates=templates,
                )
            except PrecomputedEEPathError as error:
                self._log(
                    f"[M5][EE_PATH] miss: {source}->{target} "
                    f"code={error.failure_code.value}"
                )
                if self.ee_attach_policy is EEAttachPolicy.PRECOMPUTED_REQUIRED:
                    raise
                self._log("[M5][EE_PATH] fallback=dynamic-planner")
        self._ground_held_region_goal(request)
        return self.pipeline.plan(
            request,
            collision_context_factory=self.collision_context_factory,
            **(
                {"final_plan_validator": final_plan_validator}
                if final_plan_validator is not None
                else {}
            ),
        )

    def _ground_held_region_goal(self, request: MotionPlanRequest) -> None:
        """Give a held TRANSPORT/MOVE or region PLACE an object-space destination.

        The scripted live runtime does this through its grasp retention; the
        generic ``run.py`` pipeline reaches the keyframe generator without it,
        so the model used to aim the gripper at a bare region anchor and the
        carried object clipped the table (transport) or sank through the tray
        floor (place).  Grounding here derives the current object pose from
        ``robot_state.eef_pose`` and the recorded grasp transform, picks a free
        spot inside the region, and publishes ``held_transport_goal`` /
        ``held_place_goal`` (place also rewrites the stale M4 fallback
        ``goal.target_pose`` so the released body is frozen where it lands).
        """

        from tuj.m5_motion.scripted_grasps.transport import (
            HELD_PLACE_GOAL_ANCHOR,
            HELD_TRANSPORT_GOAL_ANCHOR,
            ground_held_region_goal,
        )

        try:
            ground_held_region_goal(request)
        except ValueError as error:
            self._log(f"[M5][REGION_GOAL] grounding skipped: {error}")
            return
        for key in (HELD_TRANSPORT_GOAL_ANCHOR, HELD_PLACE_GOAL_ANCHOR):
            goal = request.task.metadata.get(key)
            if isinstance(goal, Mapping):
                self._log(
                    f"[M5][REGION_GOAL] {key} object={goal.get('object_id')} "
                    f"region={request.task.goal.target_region_id} "
                    f"source={goal.get('source')}"
                )


class WorkcellMotionRequestRouter:
    """Route MotionPlanRequest objects to the compiler for their environment."""

    def __init__(self, planners: Mapping[str, MotionRequestPlanner]) -> None:
        if not planners:
            raise ValueError("at least one workcell planner is required")
        self._planners = dict(planners)

    def __call__(self, request: MotionPlanRequest) -> Any:
        environment_name = request.world.metadata.get("environment_name")
        planner = self._planners.get(str(environment_name))
        if planner is None:
            raise ToolUseJournalCollisionBindingError(
                f"no motion planner is registered for environment "
                f"{environment_name!r}"
            )
        return planner(request)


__all__ = [
    "ToolUseJournalCollisionBindingError",
    "ToolUseJournalCollisionContextFactory",
    "ToolUseJournalMotionRequestPlanner",
    "WorkcellMotionRequestRouter",
    "attached_object_transform_from_state",
]
