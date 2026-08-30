"""Deterministic resolution of symbolic, scene-relative keyframes.

The strategy generator chooses only a frame, named anchor, unit approach axis,
offset, and roll.  This module owns every metric coordinate and quaternion
calculation so a VLM never fabricates world-frame poses.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tuj.m4_motion.schema import Pose, RelativeKeyframeSpec, WorldSnapshot


class GeometryResolutionError(ValueError):
    pass


def _quaternion_matrix_xyzw(values: Sequence[float]) -> np.ndarray:
    if len(values) != 4:
        raise GeometryResolutionError("quaternion must have four values")
    x, y, z, w = (float(value) for value in values)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise GeometryResolutionError("quaternion must be finite and non-zero")
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
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            (m[2, 1] - m[1, 2]) / scale,
            (m[0, 2] - m[2, 0]) / scale,
            (m[1, 0] - m[0, 1]) / scale,
            0.25 * scale,
        )
    else:
        diagonal = int(np.argmax(np.diag(m)))
        if diagonal == 0:
            scale = math.sqrt(max(1 + m[0, 0] - m[1, 1] - m[2, 2], 0.0)) * 2
            values = (
                0.25 * scale,
                (m[0, 1] + m[1, 0]) / scale,
                (m[0, 2] + m[2, 0]) / scale,
                (m[2, 1] - m[1, 2]) / scale,
            )
        elif diagonal == 1:
            scale = math.sqrt(max(1 + m[1, 1] - m[0, 0] - m[2, 2], 0.0)) * 2
            values = (
                (m[0, 1] + m[1, 0]) / scale,
                0.25 * scale,
                (m[1, 2] + m[2, 1]) / scale,
                (m[0, 2] - m[2, 0]) / scale,
            )
        else:
            scale = math.sqrt(max(1 + m[2, 2] - m[0, 0] - m[1, 1], 0.0)) * 2
            values = (
                (m[0, 2] + m[2, 0]) / scale,
                (m[1, 2] + m[2, 1]) / scale,
                0.25 * scale,
                (m[1, 0] - m[0, 1]) / scale,
            )
    quaternion = np.asarray(values, dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def _as_xyz(value: Any) -> np.ndarray | None:
    if isinstance(value, Mapping):
        for key in ("position_m", "position", "pos", "xyz", "translation"):
            if key in value:
                return _as_xyz(value[key])
        if {"x", "y", "z"} <= set(value):
            return np.asarray((value["x"], value["y"], value["z"]), dtype=float)
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) >= 3:
            return np.asarray(value[:3], dtype=float)
    return None


def _as_quaternion(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("orientation_xyzw", "orientation", "quaternion_xyzw"):
        orientation = value.get(key)
        if isinstance(orientation, Sequence) and len(orientation) == 4:
            matrix = _quaternion_matrix_xyzw(orientation)
            return _matrix_quaternion_xyzw(matrix)
    return None


def _pose_from_record(record: Any, *, rack: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(record, Mapping):
        preferred = ("dock_pose", "pose") if rack else ("pose",)
        for key in preferred:
            if key in record:
                nested = record[key]
                position = _as_xyz(nested)
                if position is not None:
                    quaternion = _as_quaternion(nested) or (0.0, 0.0, 0.0, 1.0)
                    return position, _quaternion_matrix_xyzw(quaternion)
        position = _as_xyz(record)
        if position is not None:
            quaternion = _as_quaternion(record) or (0.0, 0.0, 0.0, 1.0)
            return position, _quaternion_matrix_xyzw(quaternion)
    position = _as_xyz(record)
    if position is None:
        raise GeometryResolutionError("scene record has no usable pose")
    return position, np.eye(3)


def _dimensions(record: Mapping[str, Any]) -> np.ndarray | None:
    for key in ("dimensions_m", "dimensions", "size"):
        value = record.get(key)
        if isinstance(value, Sequence) and len(value) >= 3:
            result = np.asarray(value[:3], dtype=float)
            if np.all(np.isfinite(result)) and np.all(result >= 0.0):
                return result
    return None


def _anchor_local(record: Any, anchor: str) -> np.ndarray:
    normalized = anchor.strip().lower()
    if normalized in {"origin", "center", "dock", "dock_center"}:
        return np.zeros(3)
    if isinstance(record, Mapping):
        anchors = record.get("anchors")
        if isinstance(anchors, Mapping) and anchor in anchors:
            position = _as_xyz(anchors[anchor])
            if position is None:
                raise GeometryResolutionError(f"anchor {anchor!r} has no position")
            return position
        dimensions = _dimensions(record)
        if dimensions is not None:
            if normalized in {"top", "top_center"}:
                return np.asarray((0.0, 0.0, dimensions[2] * 0.5))
            if normalized in {"bottom", "bottom_center"}:
                return np.asarray((0.0, 0.0, -dimensions[2] * 0.5))
    raise GeometryResolutionError(f"unknown anchor {anchor!r}")


def _tool_rotation_from_axis(axis_world: np.ndarray, roll_rad: float) -> np.ndarray:
    z_axis = axis_world / np.linalg.norm(axis_world)
    reference = np.asarray((0.0, 0.0, 1.0))
    if abs(float(np.dot(reference, z_axis))) > 0.95:
        reference = np.asarray((1.0, 0.0, 0.0))
    x_axis = reference - float(np.dot(reference, z_axis)) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    base = np.column_stack((x_axis, y_axis, z_axis))
    cosine, sine = math.cos(roll_rad), math.sin(roll_rad)
    roll = np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
    return base @ roll


def quaternion_matrix_xyzw(values: Sequence[float]) -> np.ndarray:
    """Public quaternion-to-matrix utility for geometry strategy providers."""

    return _quaternion_matrix_xyzw(values)


def matrix_quaternion_xyzw(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Public matrix-to-quaternion utility for geometry strategy providers."""

    return _matrix_quaternion_xyzw(matrix)


def tool_rotation_from_axis(axis_world: np.ndarray, roll_rad: float) -> np.ndarray:
    """Public axis/roll convention shared by keyframe generators and resolver."""

    return _tool_rotation_from_axis(axis_world, roll_rad)


class RelativePoseResolver:
    """Resolve object/rack-relative keyframes against one immutable snapshot."""

    def __init__(self, world: WorldSnapshot) -> None:
        self._world = world

    def _record(self, frame_ref: str) -> tuple[Any, bool]:
        if frame_ref == "world":
            return {"pose": {"position_m": (0.0, 0.0, 0.0), "orientation_xyzw": (0.0, 0.0, 0.0, 1.0)}}, False
        prefix, separator, identifier = frame_ref.partition(":")
        if not separator or not identifier:
            raise GeometryResolutionError(
                "frame_ref must be 'world', 'object:<id>', or 'rack:<id>'"
            )
        if prefix == "object":
            record = self._world.objects.get(identifier)
            if record is None:
                raise GeometryResolutionError(f"unknown object frame {frame_ref!r}")
            return record, False
        if prefix == "rack":
            record = self._world.rack.get(identifier)
            if record is None:
                raise GeometryResolutionError(f"unknown rack frame {frame_ref!r}")
            return record, True
        raise GeometryResolutionError(f"unsupported frame namespace {prefix!r}")

    def resolve(self, keyframe: RelativeKeyframeSpec) -> Pose:
        record, is_rack = self._record(keyframe.frame_ref)
        frame_position, frame_rotation = _pose_from_record(record, rack=is_rack)
        anchor_world = frame_position + frame_rotation @ _anchor_local(
            record, keyframe.anchor
        )
        axis_local = np.asarray(keyframe.approach_axis_xyz, dtype=float)
        axis_world = frame_rotation @ axis_local
        axis_world /= np.linalg.norm(axis_world)
        position = anchor_world + axis_world * keyframe.offset_along_approach_m
        aligned_axis = (
            axis_world
            if keyframe.tool_axis_to_align == "+z"
            else -axis_world
        )
        rotation = _tool_rotation_from_axis(aligned_axis, keyframe.roll_rad)
        return Pose(
            frame_id="world",
            position_m=tuple(float(value) for value in position),
            orientation_xyzw=_matrix_quaternion_xyzw(rotation),
        )
