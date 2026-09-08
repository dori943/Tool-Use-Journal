"""Metric scene evidence for language-model waypoint proposals, not a validator."""

from collections.abc import Mapping
from itertools import product

import numpy as np

from tuj.m5_motion.geometry import _anchor_local, _pose_from_record


def spatial_record(record, *, rack=False):
    """Keep semantic facts, replace dense local meshes with world-space bounds."""
    if not isinstance(record, Mapping):
        return record
    result = {key: value for key, value in record.items() if key != "collision_points_m"}
    try:
        position, rotation = _pose_from_record(record, rack=rack)
        if not np.all(np.isfinite(position)):
            raise ValueError("non-finite position")
        result["origin_world_m"] = position.tolist()
        result["local_axes_world"] = rotation.T.tolist()
        anchors = record.get("anchors", {})
        result["anchors_world_m"] = {}
        for name in sorted(set(anchors) | {"origin", "center"}):
            # Match the compiler's anchor semantics, including its reserved
            # center/origin anchors. Bounds still use the geometric center.
            point = _anchor_local(record, name)
            if point.shape == (3,) and np.all(np.isfinite(point)):
                result["anchors_world_m"][name] = (position + rotation @ point).tolist()
        raw_points = record.get("collision_points_m")
        if raw_points is not None:
            points = np.asarray(raw_points, dtype=float)
            source = "collision_points"
        else:
            dimensions = np.asarray(record.get("dimensions_m"), dtype=float)
            if dimensions.shape != (3,) or not np.all(dimensions >= 0):
                raise ValueError("missing dimensions")
            center = np.asarray(anchors.get("center", [0, 0, 0]), dtype=float)
            points = center + np.asarray(list(product((-0.5, 0.5), repeat=3))) * dimensions
            source = "dimensions_envelope"
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.all(np.isfinite(points)):
            raise ValueError("invalid points")
        observation = record.get("observation_geometry")
        if isinstance(observation, Mapping) and observation.get("alignment_status") == "ALIGNED":
            observed = np.asarray(observation.get("local_bbox_corners_m"), dtype=float)
            if (
                observed.ndim == 2
                and observed.shape[1] == 3
                and len(observed)
                and np.all(np.isfinite(observed))
            ):
                points = np.concatenate((points, observed), axis=0)
                source += "+m1_observation_union"
        world_points = points @ rotation.T + position
        result["world_bounds"] = {
            "aabb_min_m": world_points.min(axis=0).tolist(),
            "aabb_max_m": world_points.max(axis=0).tolist(),
            "source": source,
            "meaning": "outer envelope; container interiors are described by packing_metadata",
        }
    except (TypeError, ValueError):
        result["world_bounds"] = {"status": "UNKNOWN"}
    return result
