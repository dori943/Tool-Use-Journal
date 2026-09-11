"""Convert attached-object pose intent into an end-effector pose target."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tuj.m5_motion.geometry import matrix_quaternion_xyzw, quaternion_matrix_xyzw
from tuj.m5_motion.schema import (
    AttachedObjectTransform,
    MotionPlanRequest,
    Pose,
    RelativeKeyframeSpec,
    WorldSnapshot,
)
from tuj.m5_motion.task_semantics import is_acquire_task, is_release_task, task_operation


POSE_SUBJECT_KEY = "pose_subject"
POSE_SUBJECT_OBJECT_ID_KEY = "pose_subject_object_id"
ATTACHED_OBJECT_POSE_SUBJECT = "ATTACHED_OBJECT"


class AttachmentRetargetError(ValueError):
    """The requested attached-object pose cannot be grounded to the robot."""


def _target_object_id(request: MotionPlanRequest) -> str | None:
    state = request.world.robot_state
    held = {state.attached_object_id, state.held_tool_id}
    for candidate in (
        request.task.tool,
        request.task.goal.target_object_id,
        *request.task.target_ids,
    ):
        if candidate is not None and candidate in held:
            return candidate
    return (
        request.task.goal.target_object_id
        or request.task.tool
        or next(iter(request.task.target_ids), None)
    )


def held_pose_subject(request: MotionPlanRequest) -> str | None:
    """Return the task target when its pose is controlled through the EE."""

    if is_acquire_task(request.task):
        return None
    target = _target_object_id(request)
    if target is None:
        return None
    state = request.world.robot_state
    if target in {state.attached_object_id, state.held_tool_id}:
        return target
    return None


def keyframe_describes_held_object(
    request: MotionPlanRequest,
    keyframe_type: object,
) -> bool:
    """Mark only phases where a held object's pose, rather than EE pose, is intended."""

    if held_pose_subject(request) is None:
        return False
    if is_release_task(request.task):
        # A release task controls the object through PLACE. RETREAT is an EE-only
        # motion after the object has been released.
        return str(getattr(keyframe_type, "value", keyframe_type)).upper() in {
            "TRANSFER",
            "PRE_PLACE",
            "PLACE",
        }
    if request.task.contact is not None:
        return True
    return task_operation(request.task) in {"TRANSPORT", "MOVE"}


def _transform_payload(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    fields = AttachedObjectTransform.model_fields
    return {key: value[key] for key in fields if key in value}


def _transforms_from_metadata(
    world: WorldSnapshot,
    metadata_key: str,
) -> dict[str, AttachedObjectTransform]:
    raw = world.metadata.get(metadata_key, {})
    if isinstance(raw, Mapping):
        values = list(raw.values())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
    else:
        raise AttachmentRetargetError(
            f"world metadata {metadata_key!r} must be a mapping or list"
        )
    result: dict[str, AttachedObjectTransform] = {}
    for value in values:
        try:
            transform = AttachedObjectTransform.model_validate(
                _transform_payload(value)
            )
        except (TypeError, ValueError) as error:
            raise AttachmentRetargetError(
                f"world metadata {metadata_key!r} contains an invalid transform"
            ) from error
        if transform.object_id in result:
            raise AttachmentRetargetError(
                f"duplicate attachment transform for {transform.object_id!r}"
            )
        result[transform.object_id] = transform
    return result


def attachment_transform(
    world: WorldSnapshot,
    object_id: str,
) -> AttachedObjectTransform:
    """Load the exact runtime-captured transform for a held object."""

    state = world.robot_state
    if state.attached_object_id == object_id:
        metadata_key = "attached_object_transforms"
    elif state.held_tool_id == object_id:
        metadata_key = "contact_friction_held_objects"
    else:
        raise AttachmentRetargetError(
            f"object {object_id!r} is not attached or contact-friction held"
        )
    transform = _transforms_from_metadata(world, metadata_key).get(object_id)
    if transform is None:
        raise AttachmentRetargetError(
            f"held object {object_id!r} has no matching runtime transform"
        )
    return transform


def end_effector_pose_for_object_pose(
    object_pose: Pose,
    transform: AttachedObjectTransform,
) -> Pose:
    """Apply ``T_world_ref = T_world_object * inverse(T_ref_object)``."""

    if object_pose.frame_id != "world":
        raise AttachmentRetargetError("desired object pose must use the world frame")
    object_rotation = quaternion_matrix_xyzw(object_pose.orientation_xyzw)
    relative_rotation = quaternion_matrix_xyzw(
        transform.orientation_in_reference_xyzw
    )
    reference_rotation = object_rotation @ relative_rotation.T
    object_position = np.asarray(object_pose.position_m, dtype=float)
    relative_position = np.asarray(transform.position_in_reference_m, dtype=float)
    reference_position = object_position - reference_rotation @ relative_position
    return Pose(
        frame_id="world",
        position_m=tuple(float(value) for value in reference_position),
        orientation_xyzw=matrix_quaternion_xyzw(reference_rotation),
    )


def object_pose_for_end_effector_pose(
    reference_pose: Pose,
    transform: AttachedObjectTransform,
) -> Pose:
    """Apply ``T_world_object = T_world_ref * T_ref_object`` at release."""
    if reference_pose.frame_id != "world":
        raise AttachmentRetargetError("release reference pose must use the world frame")
    reference_rotation = quaternion_matrix_xyzw(reference_pose.orientation_xyzw)
    object_position = np.asarray(reference_pose.position_m) + reference_rotation @ np.asarray(
        transform.position_in_reference_m
    )
    object_rotation = reference_rotation @ quaternion_matrix_xyzw(
        transform.orientation_in_reference_xyzw
    )
    return Pose(
        frame_id="world",
        position_m=tuple(float(value) for value in object_position),
        orientation_xyzw=matrix_quaternion_xyzw(object_rotation),
    )


def retarget_resolved_pose(
    world: WorldSnapshot,
    keyframe: RelativeKeyframeSpec,
    resolved_pose: Pose,
) -> Pose:
    """Retarget an explicitly tagged attached-object pose; otherwise pass through."""

    subject = str(keyframe.metadata.get(POSE_SUBJECT_KEY, "")).upper()
    if subject != ATTACHED_OBJECT_POSE_SUBJECT:
        return resolved_pose
    object_id = keyframe.metadata.get(POSE_SUBJECT_OBJECT_ID_KEY)
    if not isinstance(object_id, str) or not object_id:
        raise AttachmentRetargetError(
            f"keyframe {keyframe.keyframe_id!r} has no pose-subject object id"
        )
    return end_effector_pose_for_object_pose(
        resolved_pose,
        attachment_transform(world, object_id),
    )


__all__ = [
    "ATTACHED_OBJECT_POSE_SUBJECT",
    "AttachmentRetargetError",
    "POSE_SUBJECT_KEY",
    "POSE_SUBJECT_OBJECT_ID_KEY",
    "attachment_transform",
    "end_effector_pose_for_object_pose",
    "held_pose_subject",
    "keyframe_describes_held_object",
    "retarget_resolved_pose",
]
