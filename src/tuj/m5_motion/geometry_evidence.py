"""Normalize M1 bbox observations and bind them to an M5 world snapshot.

M1 produces coarse, camera-derived axis-aligned bounds.  M5 owns the exact
MuJoCo collision geometry.  This module keeps both sources explicit, checks
that they describe the same scene, and converts the observed box corners into
the object's local frame so the planning envelope follows a moved object.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from itertools import product
from typing import Any

import numpy as np

from tuj.m5_motion.geometry import _pose_from_record, quaternion_matrix_xyzw
from tuj.m5_motion.scene_context import spatial_record
from tuj.m5_motion.schema import WorldSnapshot


GEOMETRY_EVIDENCE_SCHEMA = "M1_M5_GEOMETRY_EVIDENCE_V2"


class GeometryEvidenceError(ValueError):
    """M1 geometry cannot be interpreted in the M5 world frame."""


def _xyz(value: Any, *, positive: bool = False) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise GeometryEvidenceError("expected a three-element numeric vector")
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise GeometryEvidenceError("expected a finite three-element numeric vector")
    if positive and not np.all(result > 0.0):
        raise GeometryEvidenceError("bbox dimensions must be positive")
    return result


def _frame_transform_m(
    m1: Mapping[str, Any],
) -> tuple[float, np.ndarray, np.ndarray, str]:
    metadata = m1.get("geometry_metadata")
    if not isinstance(metadata, Mapping):
        raise GeometryEvidenceError(
            "M1 geometry_metadata is required; implicit coordinate frames are unsafe"
        )
    try:
        meters_per_unit = float(metadata["meters_per_unit"])
    except (KeyError, TypeError, ValueError) as error:
        raise GeometryEvidenceError(
            "M1 geometry_metadata.meters_per_unit must be explicit"
        ) from error
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise GeometryEvidenceError("meters_per_unit must be finite and positive")

    frame = metadata.get("coordinate_frame")
    if not isinstance(frame, Mapping):
        raise GeometryEvidenceError("M1 coordinate_frame is required")
    frame_id = str(frame.get("frame_id", "")).strip()
    if not frame_id:
        raise GeometryEvidenceError("M1 coordinate_frame.frame_id is required")
    transform = frame.get("transform_to_world")
    if not isinstance(transform, Mapping):
        raise GeometryEvidenceError(
            "M1 coordinate_frame.transform_to_world is required"
        )
    translation = _xyz(transform.get("translation_m"))
    orientation = _xyz_quaternion(transform.get("orientation_xyzw"))
    rotation = quaternion_matrix_xyzw(orientation)
    return meters_per_unit, rotation, translation, frame_id


def _xyz_quaternion(value: Any) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise GeometryEvidenceError("expected a four-element quaternion")
    result = np.asarray(value, dtype=float)
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise GeometryEvidenceError("expected a finite four-element quaternion")
    norm = float(np.linalg.norm(result))
    if norm <= 0.0:
        raise GeometryEvidenceError("coordinate-frame quaternion must be non-zero")
    return result / norm


def _canonical_id(node: Mapping[str, Any], aliases: Mapping[str, str]) -> str:
    raw_id = str(node.get("id", ""))
    explicit = aliases.get(raw_id)
    if explicit:
        return str(explicit)
    canonical = node.get("canonical_id")
    if isinstance(canonical, str) and canonical.strip():
        return canonical.strip()
    raise GeometryEvidenceError(
        f"M1 node {raw_id!r} has no canonical_id or explicit alias"
    )


def _bounds_from_record(record: Mapping[str, Any], *, rack: bool = False):
    bounds = spatial_record(record, rack=rack).get("world_bounds", {})
    if not isinstance(bounds, Mapping) or bounds.get("status") == "UNKNOWN":
        return None
    try:
        return _xyz(bounds.get("aabb_min_m")), _xyz(bounds.get("aabb_max_m"))
    except GeometryEvidenceError:
        return None


def _corners(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    selector = np.asarray(list(product((0, 1), repeat=3)), dtype=int)
    return np.where(selector == 0, lower, upper)


def _local_observation_corners(
    record: Mapping[str, Any], lower: np.ndarray, upper: np.ndarray, *, rack: bool
) -> list[list[float]]:
    position, rotation = _pose_from_record(record, rack=rack)
    local = (_corners(lower, upper) - position) @ rotation
    return local.tolist()


def integrate_m1_geometry(
    world: WorldSnapshot,
    m1: Mapping[str, Any],
    *,
    aliases: Mapping[str, str] | None = None,
    required_object_ids: Iterable[str] = (),
    separation_tolerance_m: float,
) -> tuple[WorldSnapshot, dict[str, Any]]:
    """Attach normalized M1 evidence and return a scene-alignment report.

    Camera bounds may cover only the visible portion of an object, so size and
    center differences are diagnostics.  A required object's observed and
    simulator envelopes being disjoint is unsafe and makes ``planning_safe``
    false.  Final collision validation remains owned by MuJoCo geometry.
    """

    if not isinstance(m1, Mapping):
        raise GeometryEvidenceError("M1 geometry JSON must be an object")
    nodes = m1.get("nodes")
    if not isinstance(nodes, list):
        raise GeometryEvidenceError("M1 geometry JSON must contain a nodes list")
    if not np.isfinite(separation_tolerance_m) or separation_tolerance_m < 0.0:
        raise GeometryEvidenceError("geometry separation tolerance must be non-negative")

    meters_per_unit, frame_rotation, translation, source_frame_id = (
        _frame_transform_m(m1)
    )
    aliases = aliases or {}
    required = {str(value) for value in required_object_ids if value}
    result = world.model_copy(deep=True)
    records: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    unsafe_ids: list[str] = []

    for raw_node in nodes:
        if not isinstance(raw_node, Mapping):
            records.append({"status": "INVALID_NODE", "detail": "node is not an object"})
            continue
        raw_id = str(raw_node.get("id", ""))
        try:
            object_id = _canonical_id(raw_node, aliases)
            geometry = raw_node.get("geometry")
            if not isinstance(geometry, Mapping):
                raise GeometryEvidenceError(
                    f"M1 node {raw_id!r} has no explicit geometry record"
                )
            source_center = _xyz(geometry.get("center")) * meters_per_unit
            source_dimensions = (
                _xyz(geometry.get("aabb_size"), positive=True) * meters_per_unit
            )
            source_lower = source_center - source_dimensions / 2.0
            source_upper = source_center + source_dimensions / 2.0
            observed_corners = (
                _corners(source_lower, source_upper) @ frame_rotation.T
                + translation
            )
            observed_lower = observed_corners.min(axis=0)
            observed_upper = observed_corners.max(axis=0)
            dimensions = observed_upper - observed_lower
        except GeometryEvidenceError as error:
            records.append(
                {"m1_id": raw_id, "status": "INVALID_NODE", "detail": str(error)}
            )
            aliased = aliases.get(raw_id)
            if aliased in required:
                unsafe_ids.append(str(aliased))
            continue
        observed_ids.add(object_id)

        rack = False
        target = result.objects.get(object_id)
        target_group = "objects"
        if not isinstance(target, Mapping):
            target = result.rack.get(object_id)
            target_group = "rack"
            rack = True
        if not isinstance(target, Mapping):
            records.append(
                {"m1_id": raw_id, "object_id": object_id, "status": "UNMATCHED",
                 "observed_aabb_min_m": observed_lower.tolist(),
                 "observed_aabb_max_m": observed_upper.tolist()}
            )
            if object_id in required:
                unsafe_ids.append(object_id)
            continue

        simulator_bounds = _bounds_from_record(target, rack=rack)
        if simulator_bounds is None:
            records.append(
                {"m1_id": raw_id, "object_id": object_id,
                 "status": "SIMULATOR_BOUNDS_UNKNOWN"}
            )
            if object_id in required:
                unsafe_ids.append(object_id)
            continue
        simulator_lower, simulator_upper = simulator_bounds
        axis_separation = np.maximum(
            np.maximum(simulator_lower - observed_upper, observed_lower - simulator_upper),
            0.0,
        )
        maximum_separation = float(np.max(axis_separation))
        status = (
            "ALIGNED"
            if maximum_separation <= separation_tolerance_m
            else "DISJOINT"
        )
        if status == "DISJOINT" and object_id in required:
            unsafe_ids.append(object_id)

        local_corners = _local_observation_corners(
            target, observed_lower, observed_upper, rack=rack
        )
        evidence = {
            "schema": GEOMETRY_EVIDENCE_SCHEMA,
            "source": "M1_CAMERA_BBOX",
            "m1_id": raw_id,
            "source_frame_id": source_frame_id,
            "initial_world_aabb_min_m": observed_lower.tolist(),
            "initial_world_aabb_max_m": observed_upper.tolist(),
            "local_bbox_corners_m": local_corners,
            "alignment_status": status,
        }
        updated = dict(target)
        updated["observation_geometry"] = evidence
        if target_group == "objects":
            result.objects[object_id] = updated
        else:
            result.rack[object_id] = updated

        simulator_dimensions = simulator_upper - simulator_lower
        records.append(
            {
                "m1_id": raw_id,
                "object_id": object_id,
                "target_group": target_group,
                "status": status,
                "observed_aabb_min_m": observed_lower.tolist(),
                "observed_aabb_max_m": observed_upper.tolist(),
                "simulator_aabb_min_m": simulator_lower.tolist(),
                "simulator_aabb_max_m": simulator_upper.tolist(),
                "axis_separation_m": axis_separation.tolist(),
                "maximum_separation_m": maximum_separation,
                "observed_dimensions_m": dimensions.tolist(),
                "simulator_aabb_dimensions_m": simulator_dimensions.tolist(),
            }
        )

    missing_required = sorted(required - observed_ids)
    unsafe_ids.extend(missing_required)
    unsafe_ids = sorted(set(unsafe_ids))
    report = {
        "schema": GEOMETRY_EVIDENCE_SCHEMA,
        "planning_safe": not unsafe_ids,
        "source_frame_id": source_frame_id,
        "meters_per_unit": meters_per_unit,
        "translation_to_world_m": translation.tolist(),
        "orientation_to_world_xyzw": _xyz_quaternion(
            m1["geometry_metadata"]["coordinate_frame"]["transform_to_world"][
                "orientation_xyzw"
            ]
        ).tolist(),
        "separation_tolerance_m": float(separation_tolerance_m),
        "required_object_ids": sorted(required),
        "unsafe_required_object_ids": unsafe_ids,
        "missing_required_observations": missing_required,
        "matched_count": sum(item.get("status") in {"ALIGNED", "DISJOINT"} for item in records),
        "records": records,
    }
    result.metadata["geometry_evidence"] = {
        "schema": GEOMETRY_EVIDENCE_SCHEMA,
        "planning_safe": report["planning_safe"],
        "source_frame_id": source_frame_id,
        "meters_per_unit": meters_per_unit,
        "translation_to_world_m": translation.tolist(),
        "orientation_to_world_xyzw": report["orientation_to_world_xyzw"],
        "unsafe_required_object_ids": unsafe_ids,
        "collision_authority": "MUJOCO_COLLISION_GEOMETRY",
        "proposal_envelope": "UNION_OF_MUJOCO_AND_ALIGNED_M1_BOUNDS",
    }
    return result, report


def carry_observation_geometry(
    previous: WorldSnapshot, current: WorldSnapshot
) -> WorldSnapshot:
    """Carry object-local M1 evidence into a later runtime snapshot."""

    result = current.model_copy(deep=True)
    for group_name in ("objects", "rack"):
        previous_group = getattr(previous, group_name)
        current_group = getattr(result, group_name)
        for object_id, current_record in current_group.items():
            old_record = previous_group.get(object_id)
            if not isinstance(old_record, Mapping) or not isinstance(current_record, Mapping):
                continue
            evidence = old_record.get("observation_geometry")
            if isinstance(evidence, Mapping):
                updated = dict(current_record)
                updated["observation_geometry"] = deepcopy(dict(evidence))
                current_group[object_id] = updated
    metadata = previous.metadata.get("geometry_evidence")
    if isinstance(metadata, Mapping):
        result.metadata["geometry_evidence"] = deepcopy(dict(metadata))
    return result


__all__ = [
    "GEOMETRY_EVIDENCE_SCHEMA",
    "GeometryEvidenceError",
    "carry_observation_geometry",
    "integrate_m1_geometry",
]
