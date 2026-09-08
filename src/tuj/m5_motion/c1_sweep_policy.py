"""Deterministic collision policy for a vacuum-held plate in C1_1 sweeps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
import re
from typing import Any

import numpy as np

from tuj.m5_motion.attachment_retarget import (
    ATTACHED_OBJECT_POSE_SUBJECT,
    POSE_SUBJECT_KEY,
    POSE_SUBJECT_OBJECT_ID_KEY,
)
from tuj.m5_motion.geometry import (
    RelativePoseResolver,
    quaternion_matrix_xyzw,
    tool_rotation_from_axis,
)
from tuj.m5_motion.schema import KeyframePlanArtifact, MotionPlanRequest


C1_SWEEP_POLICY_ID = "C1_PLATE_SWEEP_V1"
C1_SWEEP_COLLISION_MARGIN_M = 0.002
C1_PLATE_TABLE_CLEARANCE_M = 0.003
C1_LADLE_LATERAL_CLEARANCE_M = 0.011


def _is_c1_plate_sweep(request: MotionPlanRequest) -> bool:
    contact = request.task.contact
    primitive = str(getattr(contact, "primitive", "")).strip().lower()
    return (
        request.world.metadata.get("environment_name") == "C1_1_LegoSweep"
        and request.task.tool == "plate"
        and primitive == "sweep"
    )


def policy_metadata() -> dict[str, object]:
    return {
        "policy_id": C1_SWEEP_POLICY_ID,
        "collision_margin_m": C1_SWEEP_COLLISION_MARGIN_M,
        "plate_table_clearance_m": C1_PLATE_TABLE_CLEARANCE_M,
        "ladle_lateral_clearance_m": C1_LADLE_LATERAL_CLEARANCE_M,
        "allow_target_block_contact": True,
        "allow_plate_table_collision_globally": False,
    }


def apply_request_policy(request: MotionPlanRequest) -> bool:
    """Apply idempotent request-level policy before generation and collision setup."""

    if not _is_c1_plate_sweep(request):
        return False
    allowed = list(request.task.allowed_touch_objects)
    for target in request.task.target_ids:
        if target not in allowed:
            allowed.append(target)
    request.task.allowed_touch_objects = allowed
    request.constraints = request.constraints.model_copy(
        update={
            "collision_margin_m": min(
                request.constraints.collision_margin_m,
                C1_SWEEP_COLLISION_MARGIN_M,
            )
        }
    )
    request.task.metadata = {
        **request.task.metadata,
        "motion_policy": policy_metadata(),
    }
    return True


def _mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    return dump(mode="python") if callable(dump) else None


def _xyz(value: Any) -> np.ndarray | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        result = np.asarray(value[:3], dtype=float)
        return result if np.all(np.isfinite(result)) else None
    return None


def _record_geometry(record: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    item = _mapping(record)
    if item is None:
        return None
    pose = _mapping(item.get("pose")) or item
    position = _xyz(pose.get("position_m"))
    quaternion = pose.get("orientation_xyzw", (0.0, 0.0, 0.0, 1.0))
    dimensions = _xyz(item.get("dimensions_m"))
    if position is None or dimensions is None:
        return None
    rotation = quaternion_matrix_xyzw(quaternion)
    anchors = _mapping(item.get("anchors")) or {}
    center_local = _xyz(anchors.get("center"))
    if center_local is None:
        center_local = np.zeros(3)
    return position, rotation, center_local, dimensions


def _table_top_m(request: MotionPlanRequest) -> float | None:
    for obstacle in request.world.obstacles:
        item = _mapping(obstacle)
        if item is None:
            continue
        identifier = item.get("obstacle_id") or item.get("object_id") or item.get("name")
        if identifier != "table_collision":
            continue
        maximum = _xyz(item.get("aabb_max_m"))
        if maximum is not None:
            return float(maximum[2])
    return None


def _aabb_half_extents(rotation: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    return np.abs(rotation) @ (dimensions * 0.5)


def _separate_from_ladle(
    position: np.ndarray,
    rotation: np.ndarray,
    plate_geometry: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ladle_geometry: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if ladle_geometry is None:
        return position, np.zeros(2)
    _, _, plate_center_local, plate_dimensions = plate_geometry
    ladle_position, ladle_rotation, ladle_center_local, ladle_dimensions = ladle_geometry
    plate_center = position + rotation @ plate_center_local
    ladle_center = ladle_position + ladle_rotation @ ladle_center_local
    plate_half = _aabb_half_extents(rotation, plate_dimensions)
    ladle_half = _aabb_half_extents(ladle_rotation, ladle_dimensions)
    vertical_gap = abs(float(plate_center[2] - ladle_center[2])) - float(
        plate_half[2] + ladle_half[2]
    )
    if vertical_gap >= C1_LADLE_LATERAL_CLEARANCE_M:
        return position, np.zeros(2)
    delta = plate_center[:2] - ladle_center[:2]
    gaps = np.abs(delta) - (plate_half[:2] + ladle_half[:2])
    if np.any(gaps >= C1_LADLE_LATERAL_CLEARANCE_M):
        return position, np.zeros(2)
    required = C1_LADLE_LATERAL_CLEARANCE_M - gaps
    axis = int(np.argmin(required))
    direction = 1.0 if delta[axis] >= 0.0 else -1.0
    shift = np.zeros(2)
    shift[axis] = direction * (float(required[axis]) + 1.0e-6)
    corrected = position.copy()
    corrected[:2] += shift
    return corrected, shift


def _encode_world_pose(
    request: MotionPlanRequest,
    *,
    strategy_id: str,
    keyframe: Any,
    position: np.ndarray,
    rotation: np.ndarray,
    table_raise_m: float,
    ladle_shift_xy_m: np.ndarray,
) -> Any:
    token = hashlib.sha256(
        f"{request.request_id}:{strategy_id}:{keyframe.keyframe_id}".encode("utf-8")
    ).hexdigest()[:16]
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", keyframe.keyframe_id).strip("_") or "keyframe"
    frame_id = f"c1_sweep_policy_{slug}_{token}"
    request.world.objects[frame_id] = {
        "pose": {
            "frame_id": "world",
            "position_m": position.tolist(),
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "anchors": {"center": [0.0, 0.0, 0.0]},
        "reference_frame_kind": "TASK_GEOMETRY",
        "collision_enabled": False,
        "policy_id": C1_SWEEP_POLICY_ID,
    }
    z_axis = rotation[:, 2]
    base = tool_rotation_from_axis(z_axis, 0.0)
    x_axis = rotation[:, 0]
    roll = math.atan2(float(x_axis @ base[:, 1]), float(x_axis @ base[:, 0]))
    metadata = {
        **keyframe.metadata,
        POSE_SUBJECT_KEY: ATTACHED_OBJECT_POSE_SUBJECT,
        POSE_SUBJECT_OBJECT_ID_KEY: "plate",
        "c1_sweep_policy": {
            "policy_id": C1_SWEEP_POLICY_ID,
            "source_frame_ref": keyframe.frame_ref,
            "source_anchor": keyframe.anchor,
            "table_raise_m": table_raise_m,
            "ladle_shift_xy_m": ladle_shift_xy_m.tolist(),
            "plate_table_clearance_m": C1_PLATE_TABLE_CLEARANCE_M,
            "ladle_lateral_clearance_m": C1_LADLE_LATERAL_CLEARANCE_M,
        },
    }
    return keyframe.model_copy(
        update={
            "frame_ref": f"object:{frame_id}",
            "anchor": "center",
            "approach_axis_xyz": tuple(float(value) for value in z_axis),
            "tool_axis_to_align": "+z",
            "offset_along_approach_m": 0.0,
            "roll_rad": roll,
            "metadata": metadata,
        }
    )


class C1SweepPolicyProvider:
    """Decorate any strategy provider with C1 plate clearance enforcement."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def generate(self, request: MotionPlanRequest) -> KeyframePlanArtifact:
        if not apply_request_policy(request):
            return self.provider.generate(request)
        artifact = self.provider.generate(request)
        plate_geometry = _record_geometry(request.world.objects.get("plate"))
        if plate_geometry is None:
            raise ValueError("C1 plate sweep requires plate pose and dimensions")
        ladle_geometry = _record_geometry(request.world.objects.get("ladle"))
        table_top = _table_top_m(request)
        resolver = RelativePoseResolver(request.world)
        candidates = []
        corrected_count = 0
        for candidate in artifact.candidates:
            keyframes = []
            for keyframe in candidate.keyframes:
                resolved = resolver.resolve(keyframe)
                position = np.asarray(resolved.position_m, dtype=float)
                rotation = quaternion_matrix_xyzw(resolved.orientation_xyzw)
                _, _, plate_center_local, plate_dimensions = plate_geometry
                plate_center = position + rotation @ plate_center_local
                plate_half = _aabb_half_extents(rotation, plate_dimensions)
                table_raise = 0.0
                if table_top is not None:
                    minimum_center_z = (
                        table_top + C1_PLATE_TABLE_CLEARANCE_M + plate_half[2]
                    )
                    table_raise = max(0.0, float(minimum_center_z - plate_center[2]))
                    position[2] += table_raise
                position, ladle_shift = _separate_from_ladle(
                    position,
                    rotation,
                    plate_geometry,
                    ladle_geometry,
                )
                keyframes.append(
                    _encode_world_pose(
                        request,
                        strategy_id=candidate.strategy_id,
                        keyframe=keyframe,
                        position=position,
                        rotation=rotation,
                        table_raise_m=table_raise,
                        ladle_shift_xy_m=ladle_shift,
                    )
                )
                corrected_count += 1
            candidates.append(
                candidate.model_copy(
                    update={
                        "keyframes": keyframes,
                        "metadata": {
                            **candidate.metadata,
                            "c1_sweep_policy": C1_SWEEP_POLICY_ID,
                        },
                    }
                )
            )
        provenance = artifact.provenance.model_copy(
            update={
                "metadata": {
                    **artifact.provenance.metadata,
                    "c1_sweep_policy": C1_SWEEP_POLICY_ID,
                    "corrected_keyframe_count": corrected_count,
                }
            }
        )
        digest = hashlib.sha256(
            repr([candidate.model_dump(mode="json") for candidate in candidates]).encode("utf-8")
        ).hexdigest()[:24]
        return artifact.model_copy(
            update={
                "artifact_id": f"keyframe-plan:c1-sweep:{digest}",
                "candidates": candidates,
                "provenance": provenance,
            }
        )


__all__ = [
    "C1_LADLE_LATERAL_CLEARANCE_M",
    "C1_PLATE_TABLE_CLEARANCE_M",
    "C1_SWEEP_COLLISION_MARGIN_M",
    "C1_SWEEP_POLICY_ID",
    "C1SweepPolicyProvider",
    "apply_request_policy",
    "policy_metadata",
]
