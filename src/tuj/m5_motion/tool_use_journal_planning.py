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
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from tuj.m5_motion.ee_exchange import RoutedKeyframeStrategyProvider
from tuj.m5_motion.ee_exchange_entry import (
    EEExchangeEntryPlanner,
    is_ee_exchange_entry_request,
)
from tuj.m5_motion.geometry import RelativePoseResolver
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
        if attached_id is None:
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
        for candidate in bound.candidates:
            current_id = base.context_id
            for keyframe in candidate.keyframes:
                selected = keyframe.collision_context_id or current_id
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
        if request.task.metadata.get("grasp_execution_mode") == "CONTACT_FRICTION":
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
        for candidate in bound.candidates:
            grasp = self._event_keyframe(
                candidate,
                KeyframeEventType.ATTACH_OBJECT,
                KeyframeType.GRASP,
            )
            token = _short_digest((candidate.strategy_id, grasp.keyframe_id))
            contact_id = f"grasp-contact:{target}:{token}"
            attached_id = f"object-attached:{target}:{token}"
            release_id = f"object-attached-release:{target}:{token}"
            contact = base.model_copy(
                update={
                    "context_id": contact_id,
                    "allowed_collision_pairs": self._contact_pairs(
                        active_ee, touch_selectors
                    ),
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
            support_selectors = self._metadata_selectors(
                request, "support_collision_selectors"
            )
            if support_selectors:
                release = attached.model_copy(
                    update={
                        "context_id": release_id,
                        "allowed_collision_pairs": self._contact_pairs(
                            target, support_selectors
                        ),
                    }
                )
                contexts[release_id] = release
            else:
                release_id = attached_id
            current_id = base.context_id
            first_post_grasp = True
            for keyframe in candidate.keyframes:
                keyframe.collision_context_after_events_id = None
                if keyframe is grasp:
                    keyframe.collision_context_id = contact_id
                    keyframe.collision_context_after_events_id = release_id
                    current_id = release_id
                else:
                    keyframe.collision_context_id = current_id
                    if current_id == release_id and first_post_grasp:
                        first_post_grasp = False
                        current_id = attached_id
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
        if request.world.robot_state.attached_object_id != target:
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
        for candidate in bound.candidates:
            place = self._event_keyframe(
                candidate,
                KeyframeEventType.DETACH_OBJECT,
                KeyframeType.PLACE,
            )
            token = _short_digest((candidate.strategy_id, place.keyframe_id))
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
                    overrides={target: target_pose},
                ),
                collision_model_version=self.compiler.model_version_for(active_ee),
            )
            contexts[contact_id] = contact
            contexts[detached_id] = detached
            current_id = base.context_id
            for keyframe in candidate.keyframes:
                keyframe.collision_context_after_events_id = None
                if keyframe is place:
                    keyframe.collision_context_id = contact_id
                    keyframe.collision_context_after_events_id = detached_id
                    current_id = detached_id
                else:
                    keyframe.collision_context_id = current_id
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
        kinematics = adapter.make_kinematics()
        pipeline = MotionPlanningPipeline(routed_provider, kinematics)
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

    def __call__(self, request: MotionPlanRequest) -> Any:
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
        return self.pipeline.plan(
            request,
            collision_context_factory=self.collision_context_factory,
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
