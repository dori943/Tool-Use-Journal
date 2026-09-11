"""Object-independent contact binding and horizontal-support clearance screening."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from tuj.m5_motion.geometry import (
    RelativePoseResolver,
    quaternion_matrix_xyzw,
    tool_rotation_from_axis,
)
from tuj.m5_motion.schema import (
    KeyframePlanArtifact,
    KeyframePlanCandidate,
    KeyframeType,
    MotionPlanRequest,
    RelativeKeyframeSpec,
    WorldSnapshot,
)
from tuj.m5_motion.task_semantics import is_acquire_task, is_release_task, task_operation


TWO_FINGER_OPPOSED_CONTACT = "TWO_FINGER_OPPOSED_CONTACT"
SUCTION_SURFACE_CONTACT = "SUCTION_SURFACE_CONTACT"
MULTI_FINGER_TABLETOP_ENCLOSURE = "MULTI_FINGER_TABLETOP_ENCLOSURE"
_SURFACE_ANCHOR_NAMES = frozenset(
    {"top", "top_center", "bottom", "bottom_center"}
)
_SUCTION_SURFACE_EPS_M = 1e-4
# Lateral sample radius for approach-facing TARGET collision points around the
# grasp axis. Matches the mounted vacuum cup outer radius so the binder scores
# the surface the sealing face can reach.
_SUCTION_CONTACT_SAMPLE_RADIUS_M = 0.03
# When the cup disk is empty, search this far laterally for a cup-centered
# TARGET patch (never copy a neighboring height onto the hollow center XY).
# Bound is local: about one cup diameter beyond the sealing radius.
_SUCTION_CONTACT_PATCH_SEARCH_RADIUS_M = 0.08
# Minimum TARGET collision points inside a relocated cup disk for a reliable
# surface patch (sparse mesh vertices still qualify at the rim).
_SUCTION_CONTACT_MIN_PATCH_POINTS = 3
_SUCTION_PATCH_ANCHOR_PREFIX = "suction_surface_patch"
_TABLETOP_ENCLOSURE_APPROACH_LOCAL = (0.0, 0.0, 1.0)
_TABLETOP_ENCLOSURE_PRE_GRASP_OFFSET_M = 0.12
# Prefer the clearance above; shorter values are only tried when the preferred
# TCP is full-pose IK unreachable (outer UR5e workspace). Keep the floor above
# support-collision standoffs observed offline (~0.04 m filtered).
_TABLETOP_ENCLOSURE_PRE_GRASP_REACH_FALLBACKS_M = (0.10, 0.08, 0.06)
_TABLETOP_ENCLOSURE_LIFT_OFFSET_M = 0.18
_TABLETOP_ENCLOSURE_LIFT_REACH_FALLBACKS_M = (0.15, 0.12, 0.10, 0.08, 0.06)
# Deterministic shorten step implied by the PRE ladder spacing (0.10→0.08).
# Catalog scripted reach fallback reuses this spacing instead of absolute lists.
_TABLETOP_ENCLOSURE_REACH_FALLBACK_STEP_M = (
    _TABLETOP_ENCLOSURE_PRE_GRASP_REACH_FALLBACKS_M[0]
    - _TABLETOP_ENCLOSURE_PRE_GRASP_REACH_FALLBACKS_M[1]
)
_TABLETOP_ENCLOSURE_REACH_FALLBACK_FLOOR_M = min(
    _TABLETOP_ENCLOSURE_PRE_GRASP_REACH_FALLBACKS_M
)
# Conservative downward reach of mounted 3F finger collision geoms below the
# grip TCP (distal + tip boxes). Shared by tabletop acquire GRASP raise and
# multi-finger region PLACE raise. Offline HOLDING poses measure tip≈22 mm and
# distal≈21–28 mm; 30 mm keeps a small pad without per-request MuJoCo probes.
_TABLETOP_ENCLOSURE_FINGER_BELOW_TCP_M = 0.030
_TABLETOP_ENCLOSURE_ANCHORS = frozenset(
    {"center", "top", "top_center"}
)


def multi_finger_finger_below_tcp_m() -> float:
    """Downward finger extent below grip TCP used for support clearance."""

    return float(_TABLETOP_ENCLOSURE_FINGER_BELOW_TCP_M)


def multi_finger_tabletop_standoff_candidates(
    preferred_offset_m: float,
    *,
    fallbacks_m: Sequence[float],
) -> tuple[float, ...]:
    """Return preferred approach standoff first, then shorter reach fallbacks."""

    preferred = float(preferred_offset_m)
    if not math.isfinite(preferred) or preferred < 0.0:
        raise ValueError("preferred_offset_m must be a finite non-negative length")
    candidates = [preferred]
    for alt in fallbacks_m:
        value = float(alt)
        if value < preferred - 1e-9:
            candidates.append(value)
    return tuple(candidates)


def catalog_reach_fallback_step_m() -> float:
    """Shorten step reused from tabletop PRE ladder spacing (no new magic)."""

    return float(_TABLETOP_ENCLOSURE_REACH_FALLBACK_STEP_M)


def catalog_pre_grasp_reach_minimum_m() -> float:
    """PRE floor reused from the shortest tabletop PRE reach fallback."""

    return float(_TABLETOP_ENCLOSURE_REACH_FALLBACK_FLOOR_M)


def reach_standoff_schedule_m(
    preferred_offset_m: float,
    *,
    step_m: float,
    minimum_m: float,
) -> tuple[float, ...]:
    """Preferred standoff first, then ``preferred - n*step`` while ``>= minimum``.

    Always includes the original preferred distance (even if it is already below
    ``minimum_m``) so callers try the recipe pose before any shortening.
    """

    preferred = float(preferred_offset_m)
    step = float(step_m)
    minimum = float(minimum_m)
    if not math.isfinite(preferred) or preferred < 0.0:
        raise ValueError("preferred_offset_m must be a finite non-negative length")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("step_m must be a finite positive length")
    if not math.isfinite(minimum) or minimum < 0.0:
        raise ValueError("minimum_m must be a finite non-negative length")
    # Round to nm-scale so binary float drift does not leak into poses/tests.
    def _clean(value: float) -> float:
        return float(round(value, 9))

    candidates = [_clean(preferred)]
    if preferred < minimum - 1e-9:
        return tuple(candidates)
    n = 1
    while n <= 1000:
        distance = preferred - n * step
        if distance < minimum - 1e-9:
            break
        candidates.append(_clean(distance))
        n += 1
    else:
        raise ValueError("reach standoff schedule did not terminate")
    return tuple(candidates)


def catalog_pre_grasp_reach_standoff_candidates(
    preferred_offset_m: float,
) -> tuple[float, ...]:
    """Recipe PRE distance, then deterministic shorter approach standoffs."""

    return reach_standoff_schedule_m(
        preferred_offset_m,
        step_m=catalog_reach_fallback_step_m(),
        minimum_m=catalog_pre_grasp_reach_minimum_m(),
    )


def catalog_lift_reach_standoff_candidates(
    preferred_offset_m: float,
    *,
    minimum_lift_m: float,
) -> tuple[float, ...]:
    """Recipe LIFT height, then shorter climbs down to ``minimum_lift_m``."""

    return reach_standoff_schedule_m(
        preferred_offset_m,
        step_m=catalog_reach_fallback_step_m(),
        minimum_m=float(minimum_lift_m),
    )


def multi_finger_tabletop_pre_grasp_standoff_candidates(
    preferred_offset_m: float,
) -> tuple[float, ...]:
    """Preferred PRE clearance, then shorter reach fallbacks."""

    return multi_finger_tabletop_standoff_candidates(
        preferred_offset_m,
        fallbacks_m=_TABLETOP_ENCLOSURE_PRE_GRASP_REACH_FALLBACKS_M,
    )


def multi_finger_tabletop_lift_standoff_candidates(
    preferred_offset_m: float,
) -> tuple[float, ...]:
    """Preferred LIFT clearance, then shorter reach fallbacks.

    Same outer-workspace cliff as PRE: at larger base-relative radii a
    top-down LIFT at the binder's preferred 0.18 m can be IK-empty while a
    shorter clear-of-support lift remains reachable.
    """

    return multi_finger_tabletop_standoff_candidates(
        preferred_offset_m,
        fallbacks_m=_TABLETOP_ENCLOSURE_LIFT_REACH_FALLBACKS_M,
    )


def multi_finger_tabletop_grasp_offset_above_support_m(
    *,
    request: MotionPlanRequest,
    tool_id: str,
    anchor: str,
    support: SupportClearanceContext,
) -> float:
    """Raise enclosure GRASP TCP so 3F fingertips clear the tabletop support.

    Thick objects whose contact anchor already sits high enough keep offset 0.
    Thin objects (spoons, utensils) need a positive approach offset so distal
    finger collision geoms do not penetrate the support under the object.
    """

    record = request.world.objects.get(tool_id)
    if not isinstance(record, Mapping):
        return 0.0
    pose = record.get("pose")
    if not isinstance(pose, Mapping):
        return 0.0
    position = _finite_vector(pose.get("position_m"), 3)
    orientation = _finite_vector(pose.get("orientation_xyzw"), 4)
    if position is None or orientation is None:
        return 0.0
    anchor_local = _anchor_local_position(record, anchor)
    if anchor_local is None:
        anchor_local = np.zeros(3, dtype=float)
    rotation = quaternion_matrix_xyzw(orientation)
    anchor_world = position + rotation @ anchor_local
    approach_world = rotation @ np.asarray(
        _TABLETOP_ENCLOSURE_APPROACH_LOCAL, dtype=float
    )
    norm = float(np.linalg.norm(approach_world))
    if norm < 1e-9:
        return 0.0
    approach_world = approach_world / norm
    z_component = float(approach_world[2])
    if z_component <= 0.25:
        return 0.0
    margin = float(request.constraints.collision_margin_m)
    min_tcp_z = (
        float(support.surface_z_m)
        + multi_finger_finger_below_tcp_m()
        + max(0.0, margin)
    )
    needed = (min_tcp_z - float(anchor_world[2])) / z_component
    if not math.isfinite(needed):
        return 0.0
    return max(0.0, needed)


class GraspGeometryBinder(Protocol):
    def bind(
        self, artifact: KeyframePlanArtifact, request: MotionPlanRequest
    ) -> KeyframePlanArtifact: ...


@dataclass(frozen=True, slots=True)
class OpposedContactSpec:
    """Object-frame contact geometry for a parallel-jaw opposed grasp."""

    contact_center_local_m: tuple[float, float, float]
    closing_axis_local_xyz: tuple[float, float, float]
    approach_axis_local_xyz: tuple[float, float, float]
    contact_span_m: float
    preshape_aperture_m: float | None = None
    preshape_tolerance_m: float | None = None
    grip_site_to_contact_center_m: float = 0.0
    approach_distance_m: float = 0.08
    lift_distance_m: float = 0.12

    def __post_init__(self) -> None:
        for name in (
            "contact_center_local_m",
            "closing_axis_local_xyz",
            "approach_axis_local_xyz",
        ):
            values = getattr(self, name)
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain three finite values")
        closing = np.asarray(self.closing_axis_local_xyz, dtype=float)
        approach = np.asarray(self.approach_axis_local_xyz, dtype=float)
        if not math.isclose(float(np.linalg.norm(closing)), 1.0, abs_tol=1e-6):
            raise ValueError("closing_axis_local_xyz must be a unit vector")
        if not math.isclose(float(np.linalg.norm(approach)), 1.0, abs_tol=1e-6):
            raise ValueError("approach_axis_local_xyz must be a unit vector")
        if abs(float(np.dot(closing, approach))) > 1e-6:
            raise ValueError("closing and approach axes must be orthogonal")
        for name in (
            "contact_span_m",
            "approach_distance_m",
            "lift_distance_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.preshape_aperture_m is not None and (
            not math.isfinite(self.preshape_aperture_m)
            or self.preshape_aperture_m <= self.contact_span_m
        ):
            raise ValueError(
                "preshape_aperture_m must exceed the opposed contact span"
            )
        if self.preshape_tolerance_m is not None and (
            not math.isfinite(self.preshape_tolerance_m)
            or self.preshape_tolerance_m <= 0.0
        ):
            raise ValueError("preshape_tolerance_m must be finite and positive")
        if (
            not math.isfinite(self.grip_site_to_contact_center_m)
            or self.grip_site_to_contact_center_m < 0.0
        ):
            raise ValueError(
                "grip_site_to_contact_center_m must be finite and non-negative"
            )


def opposed_contact_spec(record: object) -> OpposedContactSpec | None:
    """Read explicit object-frame opposed-contact geometry from a snapshot."""

    if not isinstance(record, Mapping):
        return None
    physical = record.get("physical_metadata")
    if not isinstance(physical, Mapping):
        return None
    feature = physical.get("grasp_feature")
    if not isinstance(feature, Mapping):
        return None
    provider = str(feature.get("provider", "")).upper()
    if provider != TWO_FINGER_OPPOSED_CONTACT:
        return None

    def vector(name: str) -> tuple[float, float, float]:
        raw = feature.get(name)
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != 3
        ):
            raise ValueError(f"opposed contact {name} must have three values")
        return tuple(float(value) for value in raw)

    raw_span = feature.get("contact_span_m")
    if raw_span is None:
        raise ValueError("opposed contact requires contact_span_m")
    return OpposedContactSpec(
        contact_center_local_m=vector("contact_center_local_m"),
        closing_axis_local_xyz=vector("closing_axis_local_xyz"),
        approach_axis_local_xyz=vector("approach_axis_local_xyz"),
        contact_span_m=float(raw_span),
        preshape_aperture_m=(
            float(feature["preshape_aperture_m"])
            if feature.get("preshape_aperture_m") is not None
            else None
        ),
        preshape_tolerance_m=(
            float(feature["preshape_tolerance_m"])
            if feature.get("preshape_tolerance_m") is not None
            else None
        ),
        grip_site_to_contact_center_m=float(
            feature.get("grip_site_to_contact_center_m", 0.0)
        ),
        approach_distance_m=float(feature.get("approach_distance_m", 0.08)),
        lift_distance_m=float(feature.get("lift_distance_m", 0.12)),
    )


class TwoFingerOpposedContactBinder:
    """Bind a symbolic PICK to explicit object-frame opposed contacts."""

    def bind(
        self, artifact: KeyframePlanArtifact, request: MotionPlanRequest
    ) -> KeyframePlanArtifact:
        tool_id = (
            request.task.goal.target_object_id
            or request.task.tool
            or next(iter(request.task.target_ids), None)
        )
        if tool_id is None:
            raise ValueError("opposed-contact grasp has no target object")
        record = request.world.objects.get(tool_id)
        spec = opposed_contact_spec(record)
        if spec is None or not isinstance(record, Mapping):
            raise ValueError(f"{tool_id!r} has no opposed-contact grasp feature")
        raw_anchors = record.get("anchors", {})
        if not isinstance(raw_anchors, Mapping):
            raise ValueError(f"{tool_id!r} anchors must be a mapping")

        approach_local = np.asarray(spec.approach_axis_local_xyz, dtype=float)
        contact_center = np.asarray(spec.contact_center_local_m, dtype=float)
        grip_site_anchor = (
            contact_center
            - approach_local * spec.grip_site_to_contact_center_m
        )
        anchor_name = "2f_opposed_contact"
        anchors = dict(raw_anchors)
        anchors[anchor_name] = [float(value) for value in grip_site_anchor]
        request.world.objects[tool_id] = {**dict(record), "anchors": anchors}

        pose = record.get("pose")
        if not isinstance(pose, Mapping):
            raise ValueError(f"{tool_id!r} has no world pose")
        orientation = pose.get("orientation_xyzw")
        if not isinstance(orientation, Sequence) or len(orientation) != 4:
            raise ValueError(f"{tool_id!r} has no world orientation")
        object_rotation = quaternion_matrix_xyzw(orientation)
        approach_world = object_rotation @ approach_local
        aligned_tool_axis = -approach_world
        closing_world = object_rotation @ np.asarray(
            spec.closing_axis_local_xyz, dtype=float
        )
        closing_world -= aligned_tool_axis * float(
            np.dot(closing_world, aligned_tool_axis)
        )
        closing_world /= np.linalg.norm(closing_world)
        zero_roll = tool_rotation_from_axis(aligned_tool_axis, 0.0)
        roll_rad = math.atan2(
            float(np.dot(closing_world, zero_roll[:, 1])),
            float(np.dot(closing_world, zero_roll[:, 0])),
        )

        candidates: list[KeyframePlanCandidate] = []
        for candidate in artifact.candidates:
            keyframes: list[RelativeKeyframeSpec] = []
            for keyframe in candidate.keyframes:
                updates: dict[str, object] = {}
                if keyframe.frame_ref == f"object:{tool_id}":
                    updates.update(
                        anchor=anchor_name,
                        approach_axis_xyz=spec.approach_axis_local_xyz,
                        tool_axis_to_align="-z",
                        roll_rad=roll_rad,
                    )
                    if keyframe.keyframe_type is KeyframeType.PRE_GRASP:
                        updates["offset_along_approach_m"] = max(
                            request.task.goal.approach_distance_m or 0.0,
                            spec.approach_distance_m,
                        )
                    elif keyframe.keyframe_type is KeyframeType.GRASP:
                        updates["offset_along_approach_m"] = 0.0
                        feature = record["physical_metadata"]["grasp_feature"]
                        updates["metadata"] = {
                            **keyframe.metadata,
                            **{
                                name: feature[name]
                                for name in (
                                    "uses_underside_contact",
                                    "required_under_clearance_m",
                                    "required_grasp_clearance_m",
                                    "required_contact_clearance_m",
                                )
                                if name in feature
                            },
                            "contact_center_local_m": list(
                                spec.contact_center_local_m
                            ),
                            "contact_geometry_source": "EXPLICIT_OPPOSED_CONTACT",
                        }
                    elif keyframe.keyframe_type in {
                        KeyframeType.LIFT,
                        KeyframeType.RETREAT,
                    }:
                        updates["offset_along_approach_m"] = spec.lift_distance_m
                keyframes.append(keyframe.model_copy(update=updates))
            candidates.append(
                candidate.model_copy(
                    update={
                        "strategy_id": (
                            f"{candidate.strategy_id}:2f-opposed-contact"
                        ),
                        "keyframes": keyframes,
                        "rationale": (
                            f"{candidate.rationale} The metric grasp is bound to "
                            "an object-frame opposed-contact feature."
                        ),
                        "metadata": {
                            **candidate.metadata,
                            "geometry_binder": TWO_FINGER_OPPOSED_CONTACT,
                            "contact_anchor": anchor_name,
                            "contact_span_m": spec.contact_span_m,
                            "roll_rad": roll_rad,
                        },
                    }
                )
            )

        digest = hashlib.sha256(
            json.dumps(
                {
                    "source_artifact_id": artifact.artifact_id,
                    "tool_id": tool_id,
                    "spec": {
                        name: getattr(spec, name)
                        for name in spec.__dataclass_fields__
                    },
                    "roll_rad": roll_rad,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        bound = artifact.model_copy(
            update={
                "artifact_id": (
                    f"{artifact.artifact_id}:2f-opposed-contact:{digest}"
                ),
                "provenance": artifact.provenance.model_copy(
                    update={
                        "artifact_id": (
                            f"{artifact.provenance.artifact_id}:"
                            f"2f-opposed-contact:{digest}"
                        ),
                        "metadata": {
                            **artifact.provenance.metadata,
                            "geometry_binder": TWO_FINGER_OPPOSED_CONTACT,
                            "source_keyframe_artifact_id": artifact.artifact_id,
                        },
                    }
                ),
                "candidates": candidates,
            }
        )
        return annotate_support_clearance(bound, request)


@dataclass(frozen=True, slots=True)
class SupportClearanceContext:
    """Vertical clearance facts between any object and a horizontal support.

    The context is deliberately shape- and class-agnostic.  A plate resting on
    a table is only one instance of the relation; boxes, handles, and stacked
    tools use the same measurements.
    """

    support_id: str
    surface_z_m: float
    object_bottom_z_m: float
    object_top_z_m: float
    under_clearance_m: float
    object_top_clearance_m: float
    vertical_extent_m: float
    horizontal_overlap_ratio: float
    source: str


def _finite_vector(value: object, length: int) -> np.ndarray | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        return None
    try:
        vector = np.asarray(tuple(float(item) for item in value), dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _object_dimensions(record: Mapping[str, object]) -> tuple[float, float, float] | None:
    raw = record.get("dimensions_m")
    vector = _finite_vector(raw, 3)
    if vector is None or np.any(vector <= 0.0):
        return None
    return tuple(float(value) for value in vector)


def _mapping_get(value: object, key: str) -> object:
    return value.get(key) if isinstance(value, Mapping) else None


def _finite_points(value: object) -> "np.ndarray | None":
    """An (N, 3) array of finite points, or None when the record has none."""
    if value is None:
        return None
    try:
        points = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] != 3:
        return None
    if not np.all(np.isfinite(points)):
        return None
    return points


def _object_world_bounds(
    record: Mapping[str, object],
    dimensions: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a world AABB computed from an object's oriented metric bbox."""

    resolved_dimensions = dimensions or _object_dimensions(record)
    pose = record.get("pose")
    if resolved_dimensions is None or not isinstance(pose, Mapping):
        return None
    position = _finite_vector(pose.get("position_m"), 3)
    orientation = _finite_vector(pose.get("orientation_xyzw"), 4)
    if position is None or orientation is None:
        return None
    anchors = record.get("anchors", {})
    if not isinstance(anchors, Mapping):
        return None
    center = _finite_vector(anchors.get("center", (0.0, 0.0, 0.0)), 3)
    if center is None:
        return None
    try:
        rotation = quaternion_matrix_xyzw(orientation)
    except ValueError:
        return None
    size = _finite_vector(resolved_dimensions, 3)
    if size is None or np.any(size <= 0.0):
        return None
    half = size * 0.5
    corners_local = np.asarray(
        [
            center + np.asarray((sx * half[0], sy * half[1], sz * half[2]))
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    )
    corners_world = position + corners_local @ rotation.T
    return np.min(corners_world, axis=0), np.max(corners_world, axis=0)


def _horizontal_overlap_ratio(
    object_minimum: np.ndarray,
    object_maximum: np.ndarray,
    support_minimum: np.ndarray,
    support_maximum: np.ndarray,
) -> float:
    overlap = np.maximum(
        np.minimum(object_maximum[:2], support_maximum[:2])
        - np.maximum(object_minimum[:2], support_minimum[:2]),
        0.0,
    )
    object_size = np.maximum(object_maximum[:2] - object_minimum[:2], 0.0)
    object_area = float(np.prod(object_size))
    if object_area <= 1e-12:
        return 0.0
    return min(1.0, float(np.prod(overlap)) / object_area)


def _make_support_context(
    *,
    support_id: str,
    surface_z_m: float,
    object_bottom_z_m: float,
    object_top_z_m: float,
    horizontal_overlap_ratio: float,
    source: str,
) -> SupportClearanceContext | None:
    values = (
        surface_z_m,
        object_bottom_z_m,
        object_top_z_m,
        horizontal_overlap_ratio,
    )
    if not all(math.isfinite(value) for value in values):
        return None
    if object_top_z_m < object_bottom_z_m or not 0.0 <= horizontal_overlap_ratio <= 1.0:
        return None
    return SupportClearanceContext(
        support_id=support_id,
        surface_z_m=surface_z_m,
        object_bottom_z_m=object_bottom_z_m,
        object_top_z_m=object_top_z_m,
        under_clearance_m=object_bottom_z_m - surface_z_m,
        object_top_clearance_m=object_top_z_m - surface_z_m,
        vertical_extent_m=object_top_z_m - object_bottom_z_m,
        horizontal_overlap_ratio=horizontal_overlap_ratio,
        source=source,
    )


def _explicit_support_clearance(
    record: Mapping[str, object],
    world: WorldSnapshot,
    object_id: str,
    bounds: tuple[np.ndarray, np.ndarray] | None,
) -> SupportClearanceContext | None:
    """Read an explicit support relation supplied by an upstream module."""

    candidates: list[tuple[object, str]] = []
    for key in ("support_surface", "support"):
        if key in record:
            candidates.append((record[key], f"object.{key}"))
    relations = world.metadata.get("support_relations")
    if isinstance(relations, Mapping) and object_id in relations:
        candidates.append((relations[object_id], "world.support_relations"))
    elif isinstance(relations, Sequence) and not isinstance(relations, (str, bytes)):
        for relation in relations:
            if isinstance(relation, Mapping) and str(
                relation.get("object_id", "")
            ) == object_id:
                candidates.append((relation, "world.support_relations"))

    for raw, source in candidates:
        if not isinstance(raw, Mapping):
            continue
        support_id = raw.get(
            "support_id", raw.get("surface_id", raw.get("obstacle_id"))
        )
        surface_z = raw.get(
            "surface_z_m", raw.get("support_top_z_m", raw.get("top_z_m"))
        )
        bottom_z = raw.get("object_bottom_z_m")
        top_z = raw.get("object_top_z_m")
        if support_id is None or surface_z is None:
            continue
        try:
            surface_z_m = float(surface_z)
            object_bottom_z_m = (
                float(bottom_z)
                if bottom_z is not None
                else float(bounds[0][2])
                if bounds is not None
                else math.nan
            )
            object_top_z_m = (
                float(top_z)
                if top_z is not None
                else float(bounds[1][2])
                if bounds is not None
                else math.nan
            )
        except (TypeError, ValueError):
            continue
        overlap_ratio = raw.get("horizontal_overlap_ratio", 1.0)
        try:
            overlap_ratio_m = float(overlap_ratio)
        except (TypeError, ValueError):
            continue
        context = _make_support_context(
            support_id=str(support_id),
            surface_z_m=surface_z_m,
            object_bottom_z_m=object_bottom_z_m,
            object_top_z_m=object_top_z_m,
            horizontal_overlap_ratio=overlap_ratio_m,
            source=source,
        )
        if context is not None:
            return context
    return None


def _support_rank_key(
    item: SupportClearanceContext,
) -> tuple[float, float, float]:
    return (
        abs(item.under_clearance_m),
        -item.horizontal_overlap_ratio,
        -item.surface_z_m,
    )


def _inferred_support_clearance_candidates(
    record: Mapping[str, object],
    world: WorldSnapshot,
    object_id: str,
    dimensions: tuple[float, float, float] | None,
    *,
    tolerance_m: float,
    minimum_horizontal_overlap_ratio: float,
) -> list[SupportClearanceContext]:
    """All obstacle/object AABBs that vertically support the object footprint."""

    bounds = _object_world_bounds(record, dimensions)
    if bounds is None:
        return []
    object_minimum, object_maximum = bounds
    # The vertical gap decides whether the object is resting on the support,
    # and the collision check that later rejects the pick measures the object's
    # actual collision geometry.  Measure the same geometry here when the
    # record carries it: a metric bbox is a loose hull, and for an irregular
    # mesh its bottom sinks below the surface by more than this tolerance
    # (bread 8.6 mm, apple 6.0 mm) even though the mesh rests on it within
    # 0.1 mm -- so the contact went unrecognised and the lift was refused for
    # touching the very table the object was sitting on.  Horizontal overlap
    # stays on the bbox: it is a footprint measure, not a contact one.
    object_bottom_z = float(object_minimum[2])
    contact_points = _finite_points(record.get("collision_points_m"))
    if contact_points is not None:
        pose_position = _finite_vector(
            _mapping_get(record.get("pose"), "position_m"), 3
        )
        pose_orientation = _finite_vector(
            _mapping_get(record.get("pose"), "orientation_xyzw"), 4
        )
        if pose_position is not None and pose_orientation is not None:
            try:
                point_rotation = quaternion_matrix_xyzw(pose_orientation)
            except ValueError:
                point_rotation = None
            if point_rotation is not None:
                object_bottom_z = float(
                    (contact_points @ point_rotation.T + pose_position)[:, 2].min()
                )

    inferred: list[SupportClearanceContext] = []

    def consider(
        support_id: str,
        support_minimum: np.ndarray,
        support_maximum: np.ndarray,
        source: str,
    ) -> None:
        if np.any(support_maximum < support_minimum):
            return
        overlap_ratio = _horizontal_overlap_ratio(
            object_minimum,
            object_maximum,
            support_minimum,
            support_maximum,
        )
        gap_m = object_bottom_z - float(support_maximum[2])
        if (
            overlap_ratio <= 0.0
            or overlap_ratio + 1e-9 < minimum_horizontal_overlap_ratio
            or abs(gap_m) > tolerance_m
        ):
            return
        context = _make_support_context(
            support_id=support_id,
            surface_z_m=float(support_maximum[2]),
            object_bottom_z_m=object_bottom_z,
            object_top_z_m=float(object_maximum[2]),
            horizontal_overlap_ratio=overlap_ratio,
            source=source,
        )
        if context is not None:
            inferred.append(context)

    for obstacle in world.obstacles:
        if not isinstance(obstacle, Mapping):
            continue
        if (
            obstacle.get("collision_enabled") is False
            or obstacle.get("collision_enabled_in_source") is False
        ):
            continue
        minimum = _finite_vector(obstacle.get("aabb_min_m"), 3)
        maximum = _finite_vector(obstacle.get("aabb_max_m"), 3)
        if minimum is None or maximum is None:
            continue
        consider(
            str(obstacle.get("obstacle_id", "unknown-support")),
            minimum,
            maximum,
            "world.obstacles.aabb",
        )

    for candidate_id, candidate_record in world.objects.items():
        if candidate_id == object_id or not isinstance(candidate_record, Mapping):
            continue
        if (
            candidate_record.get("collision_enabled") is False
            or candidate_record.get("collision_enabled_in_source") is False
        ):
            continue
        candidate_bounds = _object_world_bounds(candidate_record)
        if candidate_bounds is None:
            continue
        consider(
            str(candidate_id),
            candidate_bounds[0],
            candidate_bounds[1],
            "world.objects.obb",
        )
    return inferred


def support_clearance_contexts_from_world(
    record: object,
    world: WorldSnapshot,
    object_id: str,
    dimensions: tuple[float, float, float] | None = None,
    *,
    tolerance_m: float = 0.01,
    minimum_horizontal_overlap_ratio: float = 0.0,
) -> tuple[SupportClearanceContext, ...]:
    """Resolve every coplanar resting support under a pick target.

    A primary support must still clear ``minimum_horizontal_overlap_ratio`` so
    sliver AABB neighbors alone do not invent a PICK support.  Once that
    primary exists, every other vertically matching surface that shares the
    same height band and has positive footprint overlap is included too --
    tiled counters often contact the mesh on a secondary tile that the
    primary-only ACM would still reject.
    """

    if tolerance_m < 0.0 or not math.isfinite(tolerance_m):
        raise ValueError("tolerance_m must be finite and non-negative")
    if (
        not math.isfinite(minimum_horizontal_overlap_ratio)
        or not 0.0 <= minimum_horizontal_overlap_ratio <= 1.0
    ):
        raise ValueError(
            "minimum_horizontal_overlap_ratio must be finite and within [0, 1]"
        )
    if not isinstance(record, Mapping):
        return ()
    bounds = _object_world_bounds(record, dimensions)
    explicit = _explicit_support_clearance(record, world, object_id, bounds)
    if explicit is not None:
        return (explicit,)
    primaries = _inferred_support_clearance_candidates(
        record,
        world,
        object_id,
        dimensions,
        tolerance_m=tolerance_m,
        minimum_horizontal_overlap_ratio=minimum_horizontal_overlap_ratio,
    )
    if not primaries:
        return ()
    primary = min(primaries, key=_support_rank_key)
    neighbors = _inferred_support_clearance_candidates(
        record,
        world,
        object_id,
        dimensions,
        tolerance_m=tolerance_m,
        minimum_horizontal_overlap_ratio=0.0,
    )
    selected = [
        item
        for item in neighbors
        if abs(item.surface_z_m - primary.surface_z_m) <= tolerance_m + 1e-9
    ]
    by_id = {item.support_id: item for item in selected}
    by_id[primary.support_id] = primary
    return tuple(
        sorted(
            by_id.values(),
            key=lambda item: (
                0 if item.support_id == primary.support_id else 1,
                _support_rank_key(item),
                item.support_id,
            ),
        )
    )


def support_clearance_context_from_world(
    record: object,
    world: WorldSnapshot,
    object_id: str,
    dimensions: tuple[float, float, float] | None = None,
    *,
    tolerance_m: float = 0.01,
    minimum_horizontal_overlap_ratio: float = 0.0,
) -> SupportClearanceContext | None:
    """Resolve the best support clearance for any metric-bbox object.

    Explicit M1-M4 support relations take precedence.  Otherwise the relation
    is inferred from the object's oriented bbox and overlapping obstacle or
    object bboxes. These are coarse horizontal-support measurements, not local
    contact geometry or proof of graspability. ``object_top_clearance_m`` is
    the bbox top height, never the height of an arbitrary grasp edge.

    Callers that bind PICK ACM pairs should prefer
    :func:`support_clearance_contexts_from_world` so tiled coplanar supports
    are all allowed during the separation edge.
    """

    contexts = support_clearance_contexts_from_world(
        record,
        world,
        object_id,
        dimensions,
        tolerance_m=tolerance_m,
        minimum_horizontal_overlap_ratio=minimum_horizontal_overlap_ratio,
    )
    if not contexts:
        return None
    # contexts[0] is the primary (>= minimum overlap); do not re-rank by
    # clearance alone or a coplanar sliver neighbor can displace it.
    return contexts[0]


def support_clearance_context(
    record: object,
    request: MotionPlanRequest,
    object_id: str,
    dimensions: tuple[float, float, float] | None = None,
    *,
    tolerance_m: float = 0.01,
    minimum_horizontal_overlap_ratio: float = 0.0,
) -> SupportClearanceContext | None:
    """Resolve support clearance from a complete motion request."""

    return support_clearance_context_from_world(
        record,
        request.world,
        object_id,
        dimensions,
        tolerance_m=tolerance_m,
        minimum_horizontal_overlap_ratio=minimum_horizontal_overlap_ratio,
    )


def _support_metadata(
    support: SupportClearanceContext,
    *,
    collision_margin_m: float,
) -> dict[str, object]:
    return {
        "support_aware": True,
        "support_id": support.support_id,
        "support_surface_z_m": support.surface_z_m,
        "under_clearance_m": support.under_clearance_m,
        "object_top_clearance_m": support.object_top_clearance_m,
        "object_vertical_extent_m": support.vertical_extent_m,
        "support_horizontal_overlap_ratio": support.horizontal_overlap_ratio,
        "support_detection_source": support.source,
        "support_blocks_under_grasp": (
            support.under_clearance_m <= collision_margin_m
        ),
    }


def _approach_mode(axis_world: np.ndarray) -> str:
    vertical = float(axis_world[2])
    if vertical > 0.25:
        return "FROM_ABOVE"
    if vertical < -0.25:
        return "FROM_BELOW"
    return "LATERAL"


def _candidate_support_measurements(
    keyframe: RelativeKeyframeSpec,
    candidate_metadata: Mapping[str, Any],
    request: MotionPlanRequest,
    record: object,
    support: SupportClearanceContext | None,
) -> dict[str, Any]:
    """Screen a supplied pose; never infer a contact from an object class."""

    requirements = request.task.metadata.get("grasp_clearance_requirements", {})
    sources = (
        keyframe.metadata,
        candidate_metadata,
        requirements if isinstance(requirements, Mapping) else {},
    )

    def supplied(name: str) -> Any:
        return next((source[name] for source in sources if name in source), None)

    blocked: list[str] = []
    unknown: list[str] = []
    measurements: dict[str, Any] = {
        "requires_gripper_envelope_check": True,
        "requires_physical_contact_validation": True,
        "support_check_scope": "HORIZONTAL_SUPPORT_BBOX_SCREEN",
    }
    if supplied("required_edge_height_m") is not None:
        blocked.append("AMBIGUOUS_REQUIREMENT:required_edge_height_m")
    for name in (
        "required_under_clearance_m",
        "required_grasp_clearance_m",
        "required_contact_clearance_m",
    ):
        raw = supplied(name)
        if raw is None:
            continue
        try:
            value = float(raw)
            if isinstance(raw, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(name)
            measurements[name] = value
        except (TypeError, ValueError):
            blocked.append(f"INVALID_CLEARANCE_REQUIREMENT:{name}")
    underside = supplied("uses_underside_contact")
    if underside is not None and not isinstance(underside, bool):
        blocked.append("INVALID_CLEARANCE_REQUIREMENT:uses_underside_contact")
    measurements["uses_underside_contact"] = underside is True

    if support is None:
        unknown.append("SUPPORT_RELATION_UNAVAILABLE")
    else:
        margin = float(request.constraints.collision_margin_m)
        try:
            grasp_pose = RelativePoseResolver(request.world).resolve(keyframe)
            rotation = quaternion_matrix_xyzw(grasp_pose.orientation_xyzw)
            # Resolve from the resulting pose so object, world, and rack frames
            # all use the same convention.
            axis_world = rotation[:, 2] * (
                1.0 if keyframe.tool_axis_to_align == "+z" else -1.0
            )
            mode = _approach_mode(axis_world)
            clearance = float(grasp_pose.position_m[2]) - support.surface_z_m
            measurements["support_approach_mode"] = mode
            measurements["grasp_point_clearance_m"] = clearance
            if clearance < max(
                margin, measurements.get("required_grasp_clearance_m", 0.0)
            ):
                blocked.append("INSUFFICIENT_GRASP_POINT_CLEARANCE")
            needs_under = (
                underside is True
                or "required_under_clearance_m" in measurements
                or mode == "FROM_BELOW"
            )
            if needs_under and support.under_clearance_m <= max(
                margin, measurements.get("required_under_clearance_m", 0.0)
            ):
                blocked.append("INSUFFICIENT_UNDER_CLEARANCE")
        except (TypeError, ValueError):
            blocked.append("UNRESOLVED_GRASP_POSE")

        # An explicitly supplied contact center is expressed in the target
        # object's local frame, independently of the grip-site pose.
        raw_contact = supplied("contact_center_local_m")
        if raw_contact is not None:
            contact = _finite_vector(raw_contact, 3)
            pose = record.get("pose") if isinstance(record, Mapping) else None
            position = (
                _finite_vector(pose.get("position_m"), 3)
                if isinstance(pose, Mapping) else None
            )
            orientation = (
                _finite_vector(pose.get("orientation_xyzw"), 4)
                if isinstance(pose, Mapping) else None
            )
            try:
                if contact is None or position is None or orientation is None:
                    raise ValueError("invalid contact geometry")
                contact_world = position + quaternion_matrix_xyzw(
                    orientation
                ) @ contact
                measurements["contact_point_clearance_m"] = (
                    float(contact_world[2]) - support.surface_z_m
                )
                if measurements["contact_point_clearance_m"] < margin:
                    blocked.append("CONTACT_POINT_BELOW_CLEARANCE_MARGIN")
            except (TypeError, ValueError):
                blocked.append("INVALID_CONTACT_POINT_GEOMETRY")
        required_contact = measurements.get("required_contact_clearance_m")
        if required_contact is not None:
            contact_clearance = measurements.get("contact_point_clearance_m")
            if contact_clearance is None:
                unknown.append("CONTACT_POINT_UNAVAILABLE")
            elif contact_clearance < max(margin, required_contact):
                blocked.append("INSUFFICIENT_CONTACT_POINT_CLEARANCE")

    measurements["support_collision_reasons"] = list(dict.fromkeys(blocked))
    measurements["support_unknown_reasons"] = unknown
    measurements["support_clearance_status"] = (
        "BLOCKED" if blocked else "UNKNOWN" if unknown else "SCREENED"
    )
    measurements["support_collision_risk"] = (
        True if blocked else None if unknown else False
    )
    return measurements


def annotate_support_clearance(
    artifact: KeyframePlanArtifact,
    request: MotionPlanRequest,
    *,
    tolerance_m: float = 0.01,
) -> KeyframePlanArtifact:
    """Annotate supplied candidates without changing their count or geometry.

    SCREENED means only that the available coarse clearance checks passed.
    Missing support/contact data stays UNKNOWN, not safe. Full gripper/path
    collision checks and physical contact validation remain mandatory.
    """

    object_id = (
        request.task.goal.target_object_id
        or request.task.tool
        or next(iter(request.task.target_ids), None)
    )
    record = request.world.objects.get(object_id) if object_id is not None else None
    support = support_clearance_context(
        record, request, str(object_id), tolerance_m=tolerance_m
    )
    common = (
        _support_metadata(
            support, collision_margin_m=float(request.constraints.collision_margin_m)
        )
        if support is not None
        else {"support_aware": False}
    )
    candidates: list[KeyframePlanCandidate] = []
    for candidate in artifact.candidates:
        keyframes: list[RelativeKeyframeSpec] = []
        per_grasp: dict[str, dict[str, Any]] = {}
        for keyframe in candidate.keyframes:
            if keyframe.keyframe_type is not KeyframeType.GRASP:
                keyframes.append(keyframe)
                continue
            measurements = _candidate_support_measurements(
                keyframe, candidate.metadata, request, record, support
            )
            per_grasp[keyframe.keyframe_id] = measurements
            keyframes.append(keyframe.model_copy(update={
                "metadata": {**keyframe.metadata, **common, **measurements}
            }))
        blocked = list(dict.fromkeys(
            reason for measurement in per_grasp.values()
            for reason in measurement["support_collision_reasons"]
        ))
        unknown = list(dict.fromkeys(
            reason for measurement in per_grasp.values()
            for reason in measurement["support_unknown_reasons"]
        ))
        if not per_grasp:
            unknown.append("NO_GRASP_KEYFRAME")
        candidate_measurements = (
            dict(next(iter(per_grasp.values()))) if len(per_grasp) == 1 else {}
        )
        candidate_measurements.update({
            "grasp_support_measurements": per_grasp,
            "support_collision_reasons": blocked,
            "support_unknown_reasons": unknown,
            "support_clearance_status": (
                "BLOCKED" if blocked else "UNKNOWN" if unknown else "SCREENED"
            ),
            "support_collision_risk": (
                True if blocked else None if unknown else False
            ),
            "requires_gripper_envelope_check": True,
            "requires_physical_contact_validation": True,
        })
        candidates.append(candidate.model_copy(update={
            "keyframes": keyframes,
            "metadata": {**candidate.metadata, **common, **candidate_measurements},
        }))
    digest = hashlib.sha256(json.dumps({
        "source_artifact_id": artifact.artifact_id,
        "scene_signature": request.world.scene.signature,
        "measurements": [candidate.metadata for candidate in candidates],
    }, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return artifact.model_copy(update={
        "artifact_id": f"{artifact.artifact_id}:support-clearance:{digest}",
        "candidates": candidates,
        "provenance": artifact.provenance.model_copy(update={
            "artifact_id": (
                f"{artifact.provenance.artifact_id}:support-clearance:{digest}"
            ),
            "metadata": {
                **artifact.provenance.metadata,
                "support_clearance_version": "GENERIC_BBOX_SCREEN_V1",
                "source_keyframe_artifact_id": artifact.artifact_id,
            },
        }),
    })


def _request_uses_suction(request: MotionPlanRequest) -> bool:
    capabilities = {
        str(value).strip().lower()
        for value in request.task.metadata.get("ee_capabilities", [])
        if isinstance(value, str)
    }
    if "suction" in capabilities:
        return True
    ee = str(request.task.ee or "").strip().lower()
    return ee in {"vac", "vacuum"}


def _object_center_local(record: Mapping[str, object]) -> np.ndarray:
    anchors = record.get("anchors", {})
    if isinstance(anchors, Mapping):
        center = _finite_vector(anchors.get("center"), 3)
        if center is not None:
            return center
    return np.zeros(3, dtype=float)


def _approach_facing_surface_offset_from_center_m(
    record: Mapping[str, object],
    approach_local: np.ndarray,
) -> float | None:
    """Signed distance from object center to the AABB face along approach.

    ``approach_local`` is the outward (free-space) unit axis in the object
    frame, matching RelativeKeyframeSpec / VLM convention.  The support
    function of an axis-aligned box recovers ``top_center`` when approach is
    +Z and ``bottom_center`` when approach is -Z.
    """

    dimensions = _object_dimensions(record)
    if dimensions is None:
        return None
    approach = np.asarray(approach_local, dtype=float)
    norm = float(np.linalg.norm(approach))
    if not math.isfinite(norm) or norm < 1e-12:
        return None
    unit = approach / norm
    half = np.asarray(dimensions, dtype=float) * 0.5
    return float(np.dot(half, np.abs(unit)))


@dataclass(frozen=True, slots=True)
class _SuctionContactSurface:
    """Approach-facing TARGET contact for a suction GRASP TCP."""

    surface_offset_from_center_m: float
    source: str
    grasp_local_m: np.ndarray
    relocated_laterally: bool


def _cup_disk_surface_offset_from_center_m(
    points: np.ndarray,
    center: np.ndarray,
    unit: np.ndarray,
    grasp_local: np.ndarray,
    sample_radius_m: float,
    surface_rank: int = 1,
) -> tuple[float, int] | None:
    """Ranked approach projection and count of TARGET points under a cup disk."""

    relative = points - np.asarray(grasp_local, dtype=float)
    lateral = relative - np.outer(relative @ unit, unit)
    lateral_dist = np.linalg.norm(lateral, axis=1)
    in_cup = points[lateral_dist <= float(sample_radius_m) + 1e-12]
    if in_cup.size == 0:
        return None
    along = (in_cup - center) @ unit
    rank = min(max(int(surface_rank), 1), int(in_cup.shape[0]))
    return float(np.partition(along, -rank)[-rank]), int(in_cup.shape[0])


def _approach_facing_collision_surface_offset_from_center_m(
    record: Mapping[str, object],
    approach_local: np.ndarray,
    grasp_local: np.ndarray,
    *,
    sample_radius_m: float = _SUCTION_CONTACT_SAMPLE_RADIUS_M,
    patch_search_radius_m: float = _SUCTION_CONTACT_PATCH_SEARCH_RADIUS_M,
    min_patch_points: int = _SUCTION_CONTACT_MIN_PATCH_POINTS,
    surface_rank: int = 1,
) -> _SuctionContactSurface | None:
    """Approach-facing TARGET surface from body-local collision points.

    ``collision_points_m`` on world object records are unique vertices of the
    object's collision geoms in the body frame.  Priority:

    1. Points inside the cup disk (``sample_radius_m``) under the current
       grasp axis -- first TARGET surface the sealing face meets.
    2. Else a bounded nearby lateral search for a cup-sized TARGET patch:
       relocate the grasp axis onto that patch, then take the approach-facing
       max under the relocated cup (never copy a neighbor height onto an
       empty hollow center).
    3. Else AABB face fallback at the original grasp XY.

    ``None`` only when even the AABB face cannot be recovered.  No fixed
    immersion is applied.
    """

    aabb_offset = _approach_facing_surface_offset_from_center_m(
        record, approach_local
    )
    approach = np.asarray(approach_local, dtype=float)
    norm = float(np.linalg.norm(approach))
    if not math.isfinite(norm) or norm < 1e-12:
        return None
    unit = approach / norm
    center = _object_center_local(record)
    origin = np.asarray(grasp_local, dtype=float)
    points = _finite_points(record.get("collision_points_m"))
    if (
        points is None
        or not math.isfinite(float(sample_radius_m))
        or sample_radius_m < 0.0
        or not math.isfinite(float(patch_search_radius_m))
        or patch_search_radius_m < float(sample_radius_m)
        or int(min_patch_points) < 1
    ):
        if aabb_offset is None:
            return None
        return _SuctionContactSurface(
            float(aabb_offset), "aabb_fallback", origin, False
        )

    direct = _cup_disk_surface_offset_from_center_m(
        points, center, unit, origin, sample_radius_m, surface_rank
    )
    if direct is not None:
        surface_offset, _count = direct
        return _SuctionContactSurface(
            surface_offset, "collision_points", origin, False
        )

    relative = points - origin
    lateral = relative - np.outer(relative @ unit, unit)
    lateral_dist = np.linalg.norm(lateral, axis=1)
    seeds = points[
        (lateral_dist > float(sample_radius_m) + 1e-12)
        & (lateral_dist <= float(patch_search_radius_m) + 1e-12)
    ]
    best: tuple[float, float, np.ndarray, float] | None = None
    for seed in seeds:
        lat_vec = (seed - origin) - float(np.dot(seed - origin, unit)) * unit
        shift = float(np.linalg.norm(lat_vec))
        if shift <= 1e-12:
            continue
        candidate = origin + lat_vec
        scored = _cup_disk_surface_offset_from_center_m(
            points, center, unit, candidate, sample_radius_m, surface_rank
        )
        if scored is None:
            continue
        surface_offset, count = scored
        if count < int(min_patch_points):
            continue
        # Prefer higher approach-facing contact (support clearance), then
        # smaller lateral shift from the original grasp axis.
        rank = (surface_offset, -shift)
        if best is None or rank > (best[0], best[1]):
            best = (surface_offset, -shift, candidate, surface_offset)

    if best is not None:
        surface_offset, _neg_shift, candidate, _surf = best
        return _SuctionContactSurface(
            float(surface_offset),
            "nearby_surface_patch",
            np.asarray(candidate, dtype=float),
            True,
        )

    if aabb_offset is None:
        return None
    return _SuctionContactSurface(
        float(aabb_offset), "aabb_fallback", origin, False
    )



def _anchor_local_position(
    record: Mapping[str, object], anchor: str
) -> np.ndarray | None:
    """Object-frame anchor position using the same rules as pose resolution."""

    normalized = anchor.strip().lower()
    anchors = record.get("anchors", {})
    if not isinstance(anchors, Mapping):
        anchors = {}
    if normalized in {"origin", "dock", "dock_center"}:
        return np.zeros(3, dtype=float)
    if anchor in anchors:
        return _finite_vector(anchors.get(anchor), 3)
    if normalized == "center":
        return _object_center_local(record)
    dimensions = _object_dimensions(record)
    if dimensions is None:
        return None
    center = _object_center_local(record)
    half_z = float(dimensions[2]) * 0.5
    if normalized in {"top", "top_center"}:
        return center + np.asarray((0.0, 0.0, half_z))
    if normalized in {"bottom", "bottom_center"}:
        return center + np.asarray((0.0, 0.0, -half_z))
    return None


@dataclass(frozen=True, slots=True)
class SuctionSurfaceContactBinder:
    """Bind vacuum GRASP TCP to the approach-facing TARGET collision surface.

    Symbolic anchors such as ``center`` remain valid.  Only object-acquire
    GRASP keyframes for suction EEs are rewritten.  When the world record
    carries body-local ``collision_points_m``, the contact plane is the
    highest TARGET collision point under the cup footprint around the grasp
    axis.  Empty cup disks relocate laterally onto a nearby cup-covered
    TARGET patch before falling back to the AABB face.  Matching PRE_GRASP
    and LIFT keyframes follow the same lateral anchor.  Clear free-space
    standoffs beyond the AABB face are preserved; there is no fixed
    immersion past the recovered TARGET surface.
    """

    def bind(
        self, artifact: KeyframePlanArtifact, request: MotionPlanRequest
    ) -> KeyframePlanArtifact:
        if not self._should_bind(request):
            return artifact
        tool_id = (
            request.task.goal.target_object_id
            or request.task.tool
            or next(iter(request.task.target_ids), None)
        )
        if tool_id is None:
            return artifact
        record = request.world.objects.get(tool_id)
        if not isinstance(record, Mapping):
            return artifact

        frame_ref = f"object:{tool_id}"
        candidates: list[KeyframePlanCandidate] = []
        changed = False
        for candidate in artifact.candidates:
            bound_frames, record = self._bind_candidate(
                candidate,
                record=record,
                request=request,
                tool_id=str(tool_id),
                frame_ref=frame_ref,
            )
            if bound_frames is None:
                candidates.append(candidate)
                continue
            changed = True
            candidates.append(
                candidate.model_copy(
                    update={
                        "keyframes": bound_frames,
                        "rationale": (
                            f"{candidate.rationale} Vacuum GRASP TCP is bound "
                            "to the approach-facing suction contact surface."
                        ),
                        "metadata": {
                            **candidate.metadata,
                            "geometry_binder": SUCTION_SURFACE_CONTACT,
                        },
                    }
                )
            )
        if not changed:
            return artifact
        digest = hashlib.sha256(
            json.dumps(
                {
                    "source_artifact_id": artifact.artifact_id,
                    "tool_id": tool_id,
                    "binder": SUCTION_SURFACE_CONTACT,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        return artifact.model_copy(
            update={
                "artifact_id": (
                    f"{artifact.artifact_id}:suction-surface:{digest}"
                ),
                "provenance": artifact.provenance.model_copy(
                    update={
                        "artifact_id": (
                            f"{artifact.provenance.artifact_id}:"
                            f"suction-surface:{digest}"
                        ),
                        "metadata": {
                            **artifact.provenance.metadata,
                            "geometry_binder": SUCTION_SURFACE_CONTACT,
                            "source_keyframe_artifact_id": artifact.artifact_id,
                        },
                    }
                ),
                "candidates": candidates,
            }
        )

    @staticmethod
    def _should_bind(request: MotionPlanRequest) -> bool:
        if not is_acquire_task(request.task):
            return False
        if task_operation(request.task) == "PICK_TOOL":
            return False
        return _request_uses_suction(request)

    def _bind_candidate(
        self,
        candidate: KeyframePlanCandidate,
        *,
        record: Mapping[str, object],
        request: MotionPlanRequest,
        tool_id: str,
        frame_ref: str,
    ) -> tuple[list[RelativeKeyframeSpec] | None, Mapping[str, object]]:
        grasp = next(
            (
                keyframe
                for keyframe in candidate.keyframes
                if keyframe.keyframe_type is KeyframeType.GRASP
                and keyframe.frame_ref == frame_ref
            ),
            None,
        )
        if grasp is None:
            return None, record
        bound_grasp, record = self._bind_grasp_keyframe(
            grasp,
            record=record,
            request=request,
            tool_id=tool_id,
            frame_ref=frame_ref,
            strategy_id=candidate.strategy_id,
        )
        if bound_grasp is grasp:
            return None, record
        patch_anchor = bound_grasp.anchor
        relocated = bool(
            (bound_grasp.metadata or {}).get("suction_lateral_relocation_m")
        )
        keyframes: list[RelativeKeyframeSpec] = []
        for keyframe in candidate.keyframes:
            if keyframe is grasp:
                keyframes.append(bound_grasp)
                continue
            if (
                relocated
                and keyframe.frame_ref == frame_ref
                and keyframe.keyframe_type
                in {KeyframeType.PRE_GRASP, KeyframeType.LIFT}
                and keyframe.anchor == grasp.anchor
                and keyframe.approach_axis_xyz == grasp.approach_axis_xyz
            ):
                keyframes.append(
                    keyframe.model_copy(
                        update={
                            "anchor": patch_anchor,
                            "metadata": {
                                **keyframe.metadata,
                                "suction_follow_grasp_patch": True,
                                "suction_source_anchor": grasp.anchor,
                            },
                        }
                    )
                )
                continue
            keyframes.append(keyframe)
        return keyframes, record

    def _inject_patch_anchor(
        self,
        *,
        record: Mapping[str, object],
        request: MotionPlanRequest,
        tool_id: str,
        strategy_id: str,
        contact_local: np.ndarray,
    ) -> tuple[str, Mapping[str, object]]:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "strategy_id": strategy_id,
                    "contact_local_m": [
                        round(float(v), 6) for v in contact_local.tolist()
                    ],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        anchor_name = f"{_SUCTION_PATCH_ANCHOR_PREFIX}_{digest}"
        raw_anchors = record.get("anchors", {})
        if not isinstance(raw_anchors, Mapping):
            raw_anchors = {}
        anchors = dict(raw_anchors)
        anchors[anchor_name] = [float(v) for v in np.asarray(contact_local, dtype=float)]
        updated = {**dict(record), "anchors": anchors}
        request.world.objects[tool_id] = updated
        return anchor_name, updated

    def _bind_grasp_keyframe(
        self,
        keyframe: RelativeKeyframeSpec,
        *,
        record: Mapping[str, object],
        request: MotionPlanRequest,
        tool_id: str,
        frame_ref: str,
        strategy_id: str,
    ) -> tuple[RelativeKeyframeSpec, Mapping[str, object]]:
        approach = _finite_vector(keyframe.approach_axis_xyz, 3)
        if approach is None:
            self._log_bind(
                frame_ref,
                keyframe,
                applied=False,
                reason="invalid_approach_axis",
            )
            return keyframe, record
        norm = float(np.linalg.norm(approach))
        if not math.isfinite(norm) or norm < 1e-12:
            self._log_bind(
                frame_ref,
                keyframe,
                applied=False,
                reason="degenerate_approach_axis",
            )
            return keyframe, record
        unit = approach / norm
        center = _object_center_local(record)
        anchor_local = _anchor_local_position(record, keyframe.anchor)
        if anchor_local is None:
            self._log_bind(
                frame_ref,
                keyframe,
                applied=False,
                reason="unresolved_anchor",
                approach_unit=unit,
            )
            return keyframe, record
        aabb_offset = _approach_facing_surface_offset_from_center_m(record, unit)
        surface = _approach_facing_collision_surface_offset_from_center_m(
            record,
            unit,
            anchor_local,
        )
        if surface is None:
            self._log_bind(
                frame_ref,
                keyframe,
                applied=False,
                reason="missing_object_dimensions",
                approach_unit=unit,
                surface_offset_m=aabb_offset,
            )
            return keyframe, record
        surface_offset = float(surface.surface_offset_from_center_m)
        surface_source = surface.source
        offset_before = float(keyframe.offset_along_approach_m)
        anchor_along = float(np.dot(anchor_local - center, unit))
        current_along = anchor_along + offset_before
        lateral_shift = surface.grasp_local_m - np.asarray(anchor_local, dtype=float)
        lateral_shift = lateral_shift - float(np.dot(lateral_shift, unit)) * unit
        shift_m = float(np.linalg.norm(lateral_shift))
        relocated = bool(surface.relocated_laterally and shift_m > _SUCTION_SURFACE_EPS_M)

        # Clear free-space standoff beyond the AABB face: do not pull inward,
        # and do not laterally relocate an intentional hover pose.
        if (
            aabb_offset is not None
            and current_along > float(aabb_offset) + _SUCTION_SURFACE_EPS_M
        ):
            self._log_bind(
                frame_ref,
                keyframe,
                applied=False,
                reason="already_beyond_aabb_standoff",
                approach_unit=unit,
                surface_offset_m=surface_offset,
                offset_before=offset_before,
                offset_after=offset_before,
            )
            return keyframe, record

        if not relocated:
            # Already on the TARGET contact plane (neither buried nor above it).
            if abs(current_along - surface_offset) <= _SUCTION_SURFACE_EPS_M:
                self._log_bind(
                    frame_ref,
                    keyframe,
                    applied=False,
                    reason="already_on_contact_plane",
                    approach_unit=unit,
                    surface_offset_m=surface_offset,
                    offset_before=offset_before,
                    offset_after=offset_before,
                )
                return keyframe, record
            if (
                keyframe.anchor.strip().lower() in _SURFACE_ANCHOR_NAMES
                and offset_before >= -_SUCTION_SURFACE_EPS_M
                and abs(current_along - surface_offset) <= _SUCTION_SURFACE_EPS_M
            ):
                self._log_bind(
                    frame_ref,
                    keyframe,
                    applied=False,
                    reason="named_surface_anchor",
                    approach_unit=unit,
                    surface_offset_m=surface_offset,
                    offset_before=offset_before,
                    offset_after=offset_before,
                )
                return keyframe, record
            corrected_offset = surface_offset - anchor_along
            bound = keyframe.model_copy(
                update={
                    "offset_along_approach_m": corrected_offset,
                    "metadata": {
                        **keyframe.metadata,
                        "contact_geometry_source": SUCTION_SURFACE_CONTACT,
                        "suction_surface_source": surface_source,
                        "suction_surface_offset_from_center_m": surface_offset,
                        "suction_aabb_offset_from_center_m": (
                            None if aabb_offset is None else float(aabb_offset)
                        ),
                        "suction_surface_correction_m": (
                            corrected_offset - offset_before
                        ),
                    },
                }
            )
            self._log_bind(
                frame_ref,
                bound,
                applied=True,
                reason=f"bound_to_target_{surface_source}_surface",
                approach_unit=unit,
                surface_offset_m=surface_offset,
                offset_before=offset_before,
                offset_after=corrected_offset,
            )
            return bound, record

        # Lateral patch: place a contact anchor on the TARGET surface under the
        # relocated cup, then keep GRASP offset 0 on that anchor.
        contact_local = (
            np.asarray(center, dtype=float)
            + surface_offset * unit
            + lateral_shift
        )
        patch_anchor, record = self._inject_patch_anchor(
            record=record,
            request=request,
            tool_id=tool_id,
            strategy_id=strategy_id,
            contact_local=contact_local,
        )
        bound = keyframe.model_copy(
            update={
                "anchor": patch_anchor,
                "offset_along_approach_m": 0.0,
                "metadata": {
                    **keyframe.metadata,
                    "contact_geometry_source": SUCTION_SURFACE_CONTACT,
                    "suction_surface_source": surface_source,
                    "suction_surface_offset_from_center_m": surface_offset,
                    "suction_aabb_offset_from_center_m": (
                        None if aabb_offset is None else float(aabb_offset)
                    ),
                    "suction_surface_correction_m": 0.0 - offset_before,
                    "suction_source_anchor": keyframe.anchor,
                    "suction_lateral_relocation_m": shift_m,
                    "suction_patch_anchor": patch_anchor,
                },
            }
        )
        self._log_bind(
            frame_ref,
            bound,
            applied=True,
            reason="bound_to_nearby_surface_patch",
            approach_unit=unit,
            surface_offset_m=surface_offset,
            offset_before=offset_before,
            offset_after=0.0,
        )
        return bound, record


    @staticmethod
    def _log_bind(
        frame_ref: str,
        keyframe: RelativeKeyframeSpec,
        *,
        applied: bool,
        reason: str,
        approach_unit: np.ndarray | None = None,
        surface_offset_m: float | None = None,
        offset_before: float | None = None,
        offset_after: float | None = None,
    ) -> None:
        target = frame_ref.split(":", 1)[-1] if ":" in frame_ref else frame_ref
        axis = keyframe.approach_axis_xyz
        if approach_unit is not None:
            axis = (
                float(approach_unit[0]),
                float(approach_unit[1]),
                float(approach_unit[2]),
            )
        before = (
            offset_before
            if offset_before is not None
            else float(keyframe.offset_along_approach_m)
        )
        after = (
            offset_after
            if offset_after is not None
            else float(keyframe.offset_along_approach_m)
        )
        surface = (
            f"{surface_offset_m:.6f}"
            if surface_offset_m is not None
            else "n/a"
        )
        print(
            "[M5][SUCTION_BIND] "
            f"target={target} "
            f"anchor={keyframe.anchor!r} "
            f"approach_axis=({axis[0]:.4f},{axis[1]:.4f},{axis[2]:.4f}) "
            f"offset_before={before:.6f} "
            f"offset_after={after:.6f} "
            f"surface_extent_m={surface} "
            f"applied={applied} "
            f"reason={reason}"
        )


def bind_suction_surface_contact(
    artifact: KeyframePlanArtifact, request: MotionPlanRequest
) -> KeyframePlanArtifact:
    """Apply vacuum GRASP surface binding when the request is in scope."""

    return SuctionSurfaceContactBinder().bind(artifact, request)


_ACQUIRE_CONTACT_KEYFRAME_TYPES = frozenset(
    {
        KeyframeType.PRE_GRASP,
        KeyframeType.GRASP,
        KeyframeType.LIFT,
    }
)
ACQUIRE_WORLD_CONTACT_FRAME = "ACQUIRE_WORLD_CONTACT_FRAME"


def _acquire_target_object_id(request: MotionPlanRequest) -> str | None:
    tool_id = (
        request.task.goal.target_object_id
        or request.task.tool
        or next(iter(request.task.target_ids), None)
    )
    if tool_id is None:
        return None
    return str(tool_id)


@dataclass(frozen=True, slots=True)
class AcquireWorldContactFrameBinder:
    """Rewrite object-acquire contact phases off ``frame_ref=world``.

    ``RelativePoseResolver`` treats ``world`` as the origin, so PRE/GRASP/LIFT
    emitted on that frame with a small +Z offset land near ``(0,0,offset)`` —
    typically ~3.7 m from a kitchen UR5e base and outside the reach envelope.
    Object acquires must describe those phases in ``object:<target>`` so the
    approach offset stays relative to the grasp target. No-op for PICK_TOOL,
    non-acquire tasks, and keyframes already on the target object frame.
    """

    def bind(
        self, artifact: KeyframePlanArtifact, request: MotionPlanRequest
    ) -> KeyframePlanArtifact:
        if not is_acquire_task(request.task):
            return artifact
        if task_operation(request.task) == "PICK_TOOL":
            return artifact
        tool_id = _acquire_target_object_id(request)
        if tool_id is None or tool_id not in request.world.objects:
            return artifact
        target_frame = f"object:{tool_id}"
        candidates: list[KeyframePlanCandidate] = []
        changed = False
        for candidate in artifact.candidates:
            keyframes: list[RelativeKeyframeSpec] = []
            candidate_changed = False
            for keyframe in candidate.keyframes:
                if (
                    keyframe.keyframe_type in _ACQUIRE_CONTACT_KEYFRAME_TYPES
                    and keyframe.frame_ref == "world"
                ):
                    keyframe = keyframe.model_copy(
                        update={
                            "frame_ref": target_frame,
                            "metadata": {
                                **keyframe.metadata,
                                "contact_geometry_source": (
                                    ACQUIRE_WORLD_CONTACT_FRAME
                                ),
                                "acquire_world_frame_rewritten_from": "world",
                            },
                        }
                    )
                    candidate_changed = True
                    changed = True
                keyframes.append(keyframe)
            if not candidate_changed:
                candidates.append(candidate)
                continue
            candidates.append(
                candidate.model_copy(
                    update={
                        "keyframes": keyframes,
                        "rationale": (
                            f"{candidate.rationale} Acquire contact phases on "
                            "frame_ref=world were rewritten onto the target "
                            "object frame."
                        ),
                        "metadata": {
                            **candidate.metadata,
                            "geometry_binder": ACQUIRE_WORLD_CONTACT_FRAME,
                        },
                    }
                )
            )
        if not changed:
            return artifact
        digest = hashlib.sha256(
            json.dumps(
                {
                    "source_artifact_id": artifact.artifact_id,
                    "binder": ACQUIRE_WORLD_CONTACT_FRAME,
                    "tool_id": tool_id,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        return artifact.model_copy(
            update={
                "artifact_id": f"{artifact.artifact_id}:acq-world:{digest}",
                "provenance": artifact.provenance.model_copy(
                    update={
                        "artifact_id": (
                            f"{artifact.provenance.artifact_id}:"
                            f"acq-world:{digest}"
                        ),
                        "metadata": {
                            **artifact.provenance.metadata,
                            "geometry_binder": ACQUIRE_WORLD_CONTACT_FRAME,
                            "source_keyframe_artifact_id": artifact.artifact_id,
                        },
                    }
                ),
                "candidates": candidates,
            }
        )


def bind_acquire_world_contact_frames(
    artifact: KeyframePlanArtifact, request: MotionPlanRequest
) -> KeyframePlanArtifact:
    """Rewrite acquire PRE/GRASP/LIFT off world origin when in scope."""

    return AcquireWorldContactFrameBinder().bind(artifact, request)


def _request_uses_multi_finger(request: MotionPlanRequest) -> bool:
    capabilities = {
        str(value).strip().lower()
        for value in request.task.metadata.get("ee_capabilities", [])
        if isinstance(value, str)
    }
    if "multi_finger_contact" in capabilities:
        return True
    return str(request.task.ee or "").strip() in {"3F", "3f"}


def _support_blocks_under_grasp(
    request: MotionPlanRequest, object_id: str
) -> SupportClearanceContext | None:
    record = request.world.objects.get(object_id)
    support = support_clearance_context(record, request, object_id)
    if support is None:
        return None
    margin = float(request.constraints.collision_margin_m)
    if support.under_clearance_m > margin + 1e-9:
        return None
    return support


@dataclass(frozen=True, slots=True)
class MultiFingerTabletopEnclosureBinder:
    """Force top-down enclosure for multi-finger acquires on blocked supports.

    Lateral / inverted / underside approaches for short objects on a tabletop
    bury the forearm or palm in the support (observed as large negative
    island↔forearm clearance).  Canonical top-down ``approach=+Z`` with
    ``tool_axis_to_align=-z`` keeps the hand above the support while fingers
    enclose the object.  Roll is preserved for candidate diversity.  Thin
    objects also raise GRASP TCP above the support so distal finger geoms
    clear the tabletop.
    """

    def bind(
        self, artifact: KeyframePlanArtifact, request: MotionPlanRequest
    ) -> KeyframePlanArtifact:
        if not self._should_bind(request):
            return artifact
        tool_id = (
            request.task.goal.target_object_id
            or request.task.tool
            or next(iter(request.task.target_ids), None)
        )
        if tool_id is None:
            return artifact
        support = _support_blocks_under_grasp(request, str(tool_id))
        if support is None:
            return artifact
        frame_ref = f"object:{tool_id}"
        candidates: list[KeyframePlanCandidate] = []
        changed = False
        for candidate in artifact.candidates:
            keyframes: list[RelativeKeyframeSpec] = []
            candidate_changed = False
            for keyframe in candidate.keyframes:
                bound = self._bind_keyframe(
                    keyframe,
                    frame_ref=frame_ref,
                    request=request,
                    tool_id=str(tool_id),
                    support=support,
                )
                if bound is not keyframe:
                    candidate_changed = True
                    changed = True
                keyframes.append(bound)
            if not candidate_changed:
                candidates.append(candidate)
                continue
            candidates.append(
                candidate.model_copy(
                    update={
                        "keyframes": keyframes,
                        "rationale": (
                            f"{candidate.rationale} Multi-finger tabletop "
                            "enclosure uses a support-clear top-down approach."
                        ),
                        "metadata": {
                            **candidate.metadata,
                            "geometry_binder": MULTI_FINGER_TABLETOP_ENCLOSURE,
                            "tabletop_support_id": support.support_id,
                        },
                    }
                )
            )
        if not changed:
            return artifact
        digest = hashlib.sha256(
            json.dumps(
                {
                    "source_artifact_id": artifact.artifact_id,
                    "tool_id": tool_id,
                    "binder": MULTI_FINGER_TABLETOP_ENCLOSURE,
                    "support_id": support.support_id,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        return artifact.model_copy(
            update={
                "artifact_id": (
                    f"{artifact.artifact_id}:mf-tabletop:{digest}"
                ),
                "provenance": artifact.provenance.model_copy(
                    update={
                        "artifact_id": (
                            f"{artifact.provenance.artifact_id}:"
                            f"mf-tabletop:{digest}"
                        ),
                        "metadata": {
                            **artifact.provenance.metadata,
                            "geometry_binder": MULTI_FINGER_TABLETOP_ENCLOSURE,
                            "source_keyframe_artifact_id": artifact.artifact_id,
                        },
                    }
                ),
                "candidates": candidates,
            }
        )

    @staticmethod
    def _should_bind(request: MotionPlanRequest) -> bool:
        if not is_acquire_task(request.task):
            return False
        if task_operation(request.task) == "PICK_TOOL":
            return False
        if _request_uses_suction(request):
            return False
        provider = str(
            request.task.metadata.get("grasp_geometry_provider") or ""
        ).upper()
        if provider == TWO_FINGER_OPPOSED_CONTACT:
            return False
        return _request_uses_multi_finger(request)

    def _bind_keyframe(
        self,
        keyframe: RelativeKeyframeSpec,
        *,
        frame_ref: str,
        request: MotionPlanRequest,
        tool_id: str,
        support: SupportClearanceContext,
    ) -> RelativeKeyframeSpec:
        if keyframe.frame_ref != frame_ref:
            return keyframe
        if keyframe.keyframe_type not in {
            KeyframeType.PRE_GRASP,
            KeyframeType.GRASP,
            KeyframeType.LIFT,
        }:
            return keyframe
        approach = _TABLETOP_ENCLOSURE_APPROACH_LOCAL
        already_top_down = (
            math.isclose(keyframe.approach_axis_xyz[0], 0.0, abs_tol=1e-3)
            and math.isclose(keyframe.approach_axis_xyz[1], 0.0, abs_tol=1e-3)
            and keyframe.approach_axis_xyz[2] > 0.25
            and keyframe.tool_axis_to_align == "-z"
        )
        anchor = keyframe.anchor
        if anchor.strip().lower() not in _TABLETOP_ENCLOSURE_ANCHORS:
            anchor = "center"
        if keyframe.keyframe_type is KeyframeType.PRE_GRASP:
            offset = (
                max(
                    float(keyframe.offset_along_approach_m),
                    _TABLETOP_ENCLOSURE_PRE_GRASP_OFFSET_M,
                )
                if already_top_down
                else _TABLETOP_ENCLOSURE_PRE_GRASP_OFFSET_M
            )
        elif keyframe.keyframe_type is KeyframeType.GRASP:
            offset = multi_finger_tabletop_grasp_offset_above_support_m(
                request=request,
                tool_id=tool_id,
                anchor=anchor,
                support=support,
            )
        else:
            offset = (
                max(
                    float(keyframe.offset_along_approach_m),
                    _TABLETOP_ENCLOSURE_LIFT_OFFSET_M,
                )
                if already_top_down
                else _TABLETOP_ENCLOSURE_LIFT_OFFSET_M
            )
        if (
            already_top_down
            and anchor == keyframe.anchor
            and math.isclose(
                float(keyframe.offset_along_approach_m), offset, abs_tol=1e-9
            )
            and keyframe.tool_axis_to_align == "-z"
        ):
            if keyframe.metadata.get("contact_geometry_source") == (
                MULTI_FINGER_TABLETOP_ENCLOSURE
            ):
                return keyframe
            return keyframe.model_copy(
                update={
                    "metadata": {
                        **keyframe.metadata,
                        "contact_geometry_source": MULTI_FINGER_TABLETOP_ENCLOSURE,
                    }
                }
            )
        metadata = {
            **keyframe.metadata,
            "contact_geometry_source": MULTI_FINGER_TABLETOP_ENCLOSURE,
            "tabletop_enclosure_rewritten": True,
            "tabletop_enclosure_prior_approach_xyz": list(
                keyframe.approach_axis_xyz
            ),
            "tabletop_enclosure_prior_tool_axis": (
                keyframe.tool_axis_to_align
            ),
        }
        if (
            keyframe.keyframe_type is KeyframeType.GRASP
            and offset > 1e-9
        ):
            metadata["tabletop_enclosure_grasp_clearance_above_support_m"] = (
                offset
            )
        return keyframe.model_copy(
            update={
                "anchor": anchor,
                "approach_axis_xyz": approach,
                "tool_axis_to_align": "-z",
                "offset_along_approach_m": offset,
                "metadata": metadata,
            }
        )


def bind_multi_finger_tabletop_enclosure(
    artifact: KeyframePlanArtifact, request: MotionPlanRequest
) -> KeyframePlanArtifact:
    """Apply multi-finger tabletop enclosure binding when in scope."""

    return MultiFingerTabletopEnclosureBinder().bind(artifact, request)


def _region_support_surface_z_m(
    request: MotionPlanRequest, region_id: str
) -> float | None:
    """Best-effort interior floor height for a place region (world z)."""

    record = request.world.objects.get(region_id)
    if not isinstance(record, Mapping):
        return None
    pose = record.get("pose")
    if not isinstance(pose, Mapping):
        return None
    position = _finite_vector(pose.get("position_m"), 3)
    orientation = _finite_vector(pose.get("orientation_xyzw"), 4)
    if position is None or orientation is None:
        return None
    rotation = quaternion_matrix_xyzw(orientation)
    dims = _finite_vector(record.get("dimensions_m"), 3)
    anchors = record.get("anchors", {})
    if not isinstance(anchors, Mapping):
        anchors = {}
    bottom_local = _finite_vector(anchors.get("bottom"), 3)
    if bottom_local is None and dims is not None:
        bottom_local = np.array([0.0, 0.0, -0.5 * float(dims[2])], dtype=float)
    if bottom_local is None:
        return None
    floor_z = float((position + rotation @ bottom_local)[2])
    # Raise to the highest near-floor collision vertex when available so thin
    # interior floor convex pieces are covered, not only the symbolic bottom.
    points = record.get("collision_points_m")
    if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
        try:
            cloud = np.asarray(points, dtype=float)
        except (TypeError, ValueError):
            cloud = None
        if cloud is not None and cloud.ndim == 2 and cloud.shape[1] >= 3:
            world = (rotation @ cloud[:, :3].T).T + position
            center_z = float(position[2])
            lower = world[world[:, 2] <= center_z + 1e-9, 2]
            if lower.size:
                floor_z = max(floor_z, float(lower.max()))
    return floor_z


def _region_top_surface_z_m(
    request: MotionPlanRequest, region_id: str
) -> float | None:
    """Symbolic top / rim height for a place region (world z)."""

    record = request.world.objects.get(region_id)
    if not isinstance(record, Mapping):
        return None
    pose = record.get("pose")
    if not isinstance(pose, Mapping):
        return None
    position = _finite_vector(pose.get("position_m"), 3)
    orientation = _finite_vector(pose.get("orientation_xyzw"), 4)
    if position is None or orientation is None:
        return None
    rotation = quaternion_matrix_xyzw(orientation)
    dims = _finite_vector(record.get("dimensions_m"), 3)
    anchors = record.get("anchors", {})
    if not isinstance(anchors, Mapping):
        anchors = {}
    top_local = _finite_vector(anchors.get("top"), 3)
    if top_local is None:
        top_local = _finite_vector(anchors.get("top_center"), 3)
    if top_local is None and dims is not None:
        top_local = np.array([0.0, 0.0, 0.5 * float(dims[2])], dtype=float)
    if top_local is None:
        return None
    return float((position + rotation @ top_local)[2])


def _region_finger_clearance_surface_z_m(
    request: MotionPlanRequest, region_id: str
) -> float | None:
    """Height fingers must clear for multi-finger region PLACE (world z).

    Interior floor alone is not enough for rimmed containers: 3F fingertips hang
    below the grip TCP and clip upper rim convex pieces (near the region top)
    even after the TCP clears the floor by ``finger_below + margin``. Use the
    higher of floor and symbolic top so flat supports stay unchanged while
    trays clear their lip.
    """

    floor_z = _region_support_surface_z_m(request, region_id)
    top_z = _region_top_surface_z_m(request, region_id)
    if floor_z is None:
        return top_z
    if top_z is None:
        return floor_z
    return max(float(floor_z), float(top_z))


def multi_finger_place_raise_above_support_m(
    request: MotionPlanRequest,
    keyframe: RelativeKeyframeSpec,
) -> float:
    """Extra PLACE/PRE_PLACE approach offset so 3F fingertips clear the region.

    Release goals often seat the held object just above the region floor. The
    grip TCP then sits only ~finger length above that floor, so finger collision
    geoms penetrate interior floor meshes and rim lips. Raise the held-object
    keyframe along its approach until the retargeted TCP clears
    ``finger_clearance_surface + finger_below + margin`` (shared finger_below
    with tabletop acquire).
    """

    if not is_release_task(request.task):
        return 0.0
    if not _request_uses_multi_finger(request):
        return 0.0
    if keyframe.keyframe_type not in {KeyframeType.PLACE, KeyframeType.PRE_PLACE}:
        return 0.0
    region_id = request.task.goal.target_region_id
    if not isinstance(region_id, str) or not region_id:
        return 0.0
    clearance_z = _region_finger_clearance_surface_z_m(request, region_id)
    if clearance_z is None:
        return 0.0
    margin = float(request.constraints.collision_margin_m)
    min_tcp_z = (
        float(clearance_z) + multi_finger_finger_below_tcp_m() + max(0.0, margin)
    )
    try:
        from tuj.m5_motion.attachment_retarget import (
            ATTACHED_OBJECT_POSE_SUBJECT,
            POSE_SUBJECT_KEY,
            POSE_SUBJECT_OBJECT_ID_KEY,
            retarget_resolved_pose,
        )

        subject = str(keyframe.metadata.get(POSE_SUBJECT_KEY, "")).upper()
        if subject != ATTACHED_OBJECT_POSE_SUBJECT:
            return 0.0
        resolved = RelativePoseResolver(request.world).resolve(keyframe)
        eef_pose = retarget_resolved_pose(request.world, keyframe, resolved)
    except Exception:
        return 0.0
    deficit = min_tcp_z - float(eef_pose.position_m[2])
    if deficit <= 1e-9:
        return 0.0
    approach = np.asarray(keyframe.approach_axis_xyz, dtype=float)
    packing = keyframe.metadata.get("packing_orientation_xyzw")
    if isinstance(packing, Sequence) and not isinstance(packing, (str, bytes)):
        try:
            rotation = quaternion_matrix_xyzw(packing)
            approach_world = rotation @ approach
        except (TypeError, ValueError):
            approach_world = approach
    else:
        # Fall back to object pose orientation when packing pin is absent.
        object_id = (
            keyframe.metadata.get(POSE_SUBJECT_OBJECT_ID_KEY)
            or request.task.goal.target_object_id
            or next(iter(request.task.target_ids), None)
        )
        record = (
            request.world.objects.get(str(object_id))
            if object_id is not None
            else None
        )
        approach_world = approach
        if isinstance(record, Mapping):
            pose = record.get("pose")
            if isinstance(pose, Mapping):
                orientation = _finite_vector(pose.get("orientation_xyzw"), 4)
                if orientation is not None:
                    approach_world = quaternion_matrix_xyzw(orientation) @ approach
    z_component = float(np.asarray(approach_world, dtype=float)[2])
    if z_component <= 0.25:
        return max(0.0, deficit)
    needed = deficit / z_component
    if not math.isfinite(needed):
        return 0.0
    return max(0.0, needed)


MULTI_FINGER_PLACE_SUPPORT_CLEARANCE = "MULTI_FINGER_PLACE_SUPPORT_CLEARANCE"
MULTI_FINGER_PLACE_RETREAT_CLEARANCE = "MULTI_FINGER_PLACE_RETREAT_CLEARANCE"
# Extra post-detach retreat lengths tried when the preferred TCP clearance is
# still collision-invalid for the empty EE (finger vs released object).
_MULTI_FINGER_PLACE_RETREAT_EXTENSIONS_M = (0.0, 0.02, 0.04, 0.06, 0.08)


def multi_finger_place_post_detach_retreat_clearance_m(
    request: MotionPlanRequest,
) -> float:
    """Minimum empty-EE retreat along approach after PLACE DETACH."""

    return float(
        multi_finger_finger_below_tcp_m()
        + max(0.0, float(request.constraints.collision_margin_m))
    )


def multi_finger_place_retreat_standoff_candidates(
    preferred_offset_m: float,
) -> tuple[float, ...]:
    """Preferred post-detach retreat, then longer bounded extensions."""

    preferred = float(preferred_offset_m)
    if not math.isfinite(preferred) or preferred < 0.0:
        raise ValueError("preferred_offset_m must be a finite non-negative length")
    return tuple(
        preferred + float(extra) for extra in _MULTI_FINGER_PLACE_RETREAT_EXTENSIONS_M
    )


def _region_pose_rotation(
    request: MotionPlanRequest, region_id: str
) -> tuple[np.ndarray, np.ndarray] | None:
    record = request.world.objects.get(region_id)
    if not isinstance(record, Mapping):
        return None
    pose = record.get("pose")
    if not isinstance(pose, Mapping):
        return None
    position = _finite_vector(pose.get("position_m"), 3)
    orientation = _finite_vector(pose.get("orientation_xyzw"), 4)
    if position is None or orientation is None:
        return None
    return position, quaternion_matrix_xyzw(orientation)


def _publish_region_anchor(
    request: MotionPlanRequest,
    *,
    region_id: str,
    anchor: str,
    local_position_m: Sequence[float],
) -> None:
    record = request.world.objects.get(region_id)
    if not isinstance(record, Mapping):
        raise ValueError(f"place retreat requires region {region_id!r}")
    raw_anchors = record.get("anchors", {})
    anchors = dict(raw_anchors) if isinstance(raw_anchors, Mapping) else {}
    anchors[anchor] = [float(value) for value in local_position_m]
    request.world.objects[region_id] = {**dict(record), "anchors": anchors}


def multi_finger_place_retreat_from_place_tcp(
    request: MotionPlanRequest,
    *,
    place_keyframe: RelativeKeyframeSpec,
    retreat_keyframe: RelativeKeyframeSpec,
    anchor_name: str,
) -> RelativeKeyframeSpec | None:
    """Rewrite post-DETACH RETREAT as PLACE_TCP + approach * clearance.

    Region PLACE RETREAT is an empty-EE motion. Using the object-origin anchor
    with a small fixed offset lands near the PLACE TCP (grasp height ≈ 5 cm),
    so fingertips stay in the just-released body. Anchor the retreat at the
    PLACE TCP in the region frame and offset along the PLACE approach by
    ``finger_below + collision_margin``. Does not mutate the PLACE keyframe.
    """

    if not is_release_task(request.task):
        return None
    if not _request_uses_multi_finger(request):
        return None
    region_id = request.task.goal.target_region_id
    if not isinstance(region_id, str) or not region_id:
        return None
    if retreat_keyframe.keyframe_type is not KeyframeType.RETREAT:
        return None
    if place_keyframe.keyframe_type is not KeyframeType.PLACE:
        return None
    region_pose = _region_pose_rotation(request, region_id)
    if region_pose is None:
        return None
    region_position, region_rotation = region_pose
    try:
        from tuj.m5_motion.attachment_retarget import retarget_resolved_pose

        place_object = RelativePoseResolver(request.world).resolve(place_keyframe)
        place_tcp = retarget_resolved_pose(
            request.world, place_keyframe, place_object
        )
    except Exception:
        return None
    clearance = multi_finger_place_post_detach_retreat_clearance_m(request)
    approach_local = np.asarray(place_keyframe.approach_axis_xyz, dtype=float)
    norm = float(np.linalg.norm(approach_local))
    if norm <= 1e-12:
        return None
    approach_local = approach_local / norm
    approach_world = region_rotation @ approach_local
    approach_norm = float(np.linalg.norm(approach_world))
    if approach_norm <= 1e-12:
        return None
    approach_world = approach_world / approach_norm
    place_tcp_position = np.asarray(place_tcp.position_m, dtype=float)
    tcp_local = region_rotation.T @ (place_tcp_position - region_position)
    _publish_region_anchor(
        request,
        region_id=region_id,
        anchor=anchor_name,
        local_position_m=tcp_local,
    )
    metadata = {
        key: value
        for key, value in retreat_keyframe.metadata.items()
        if key not in {"pose_subject", "pose_subject_object_id"}
    }
    metadata.update(
        {
            "contact_geometry_source": MULTI_FINGER_PLACE_RETREAT_CLEARANCE,
            "place_retreat_from_place_tcp": True,
            "place_retreat_clearance_m": clearance,
            "place_retreat_place_keyframe_id": place_keyframe.keyframe_id,
            "packing_orientation_xyzw": list(place_tcp.orientation_xyzw),
        }
    )
    return retreat_keyframe.model_copy(
        update={
            "frame_ref": f"object:{region_id}",
            "anchor": anchor_name,
            "approach_axis_xyz": tuple(float(value) for value in approach_local),
            "tool_axis_to_align": place_keyframe.tool_axis_to_align,
            "offset_along_approach_m": clearance,
            "roll_rad": float(place_keyframe.roll_rad),
            "metadata": metadata,
        }
    )


@dataclass(frozen=True, slots=True)
class MultiFingerPlaceSupportClearanceBinder:
    """Raise multi-finger PLACE/PRE_PLACE and clear empty-EE RETREAT after DETACH."""

    def bind(
        self, artifact: KeyframePlanArtifact, request: MotionPlanRequest
    ) -> KeyframePlanArtifact:
        if not is_release_task(request.task):
            return artifact
        if not _request_uses_multi_finger(request):
            return artifact
        if not request.task.goal.target_region_id:
            return artifact
        candidates: list[KeyframePlanCandidate] = []
        changed = False
        for candidate in artifact.candidates:
            keyframes: list[RelativeKeyframeSpec] = []
            candidate_changed = False
            for keyframe in candidate.keyframes:
                bound = self._bind_release_height(keyframe, request=request)
                if bound is not keyframe:
                    candidate_changed = True
                    changed = True
                keyframes.append(bound)
            rewritten = self._bind_post_detach_retreat(
                keyframes, request=request, strategy_id=candidate.strategy_id
            )
            if rewritten is not keyframes:
                keyframes = list(rewritten)
                candidate_changed = True
                changed = True
            if not candidate_changed:
                candidates.append(candidate)
                continue
            candidates.append(
                candidate.model_copy(
                    update={
                        "keyframes": keyframes,
                        "rationale": (
                            f"{candidate.rationale} Multi-finger place raises "
                            "release height so fingertips clear the region floor "
                            "and retreats the empty EE from PLACE TCP."
                        ),
                        "metadata": {
                            **candidate.metadata,
                            "geometry_binder": MULTI_FINGER_PLACE_SUPPORT_CLEARANCE,
                        },
                    }
                )
            )
        if not changed:
            return artifact
        digest = hashlib.sha256(
            json.dumps(
                {
                    "source_artifact_id": artifact.artifact_id,
                    "binder": MULTI_FINGER_PLACE_SUPPORT_CLEARANCE,
                    "region_id": request.task.goal.target_region_id,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        return artifact.model_copy(
            update={
                "artifact_id": f"{artifact.artifact_id}:mf-place:{digest}",
                "provenance": artifact.provenance.model_copy(
                    update={
                        "artifact_id": (
                            f"{artifact.provenance.artifact_id}:"
                            f"mf-place:{digest}"
                        ),
                        "metadata": {
                            **artifact.provenance.metadata,
                            "geometry_binder": MULTI_FINGER_PLACE_SUPPORT_CLEARANCE,
                            "source_keyframe_artifact_id": artifact.artifact_id,
                        },
                    }
                ),
                "candidates": candidates,
            }
        )

    def _bind_release_height(
        self,
        keyframe: RelativeKeyframeSpec,
        *,
        request: MotionPlanRequest,
    ) -> RelativeKeyframeSpec:
        if keyframe.keyframe_type not in {
            KeyframeType.PLACE,
            KeyframeType.PRE_PLACE,
        }:
            return keyframe
        raise_m = multi_finger_place_raise_above_support_m(request, keyframe)
        if raise_m <= 1e-9:
            return keyframe
        new_offset = float(keyframe.offset_along_approach_m) + raise_m
        return keyframe.model_copy(
            update={
                "offset_along_approach_m": new_offset,
                "metadata": {
                    **keyframe.metadata,
                    "contact_geometry_source": MULTI_FINGER_PLACE_SUPPORT_CLEARANCE,
                    "tabletop_enclosure_place_raise_m": raise_m,
                },
            }
        )

    def _bind_post_detach_retreat(
        self,
        keyframes: Sequence[RelativeKeyframeSpec],
        *,
        request: MotionPlanRequest,
        strategy_id: str,
    ) -> Sequence[RelativeKeyframeSpec]:
        place_index = next(
            (
                index
                for index, keyframe in enumerate(keyframes)
                if keyframe.keyframe_type is KeyframeType.PLACE
            ),
            None,
        )
        if place_index is None:
            return keyframes
        retreat_index = next(
            (
                index
                for index in range(place_index + 1, len(keyframes))
                if keyframes[index].keyframe_type is KeyframeType.RETREAT
            ),
            None,
        )
        if retreat_index is None:
            return keyframes
        place = keyframes[place_index]
        retreat = keyframes[retreat_index]
        anchor = f"mf_place_retreat_{strategy_id}_{retreat.keyframe_id}"
        # Anchor names are used as MuJoCo-style identifiers; keep them compact.
        anchor = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in anchor)[
            :80
        ]
        rewritten = multi_finger_place_retreat_from_place_tcp(
            request,
            place_keyframe=place,
            retreat_keyframe=retreat,
            anchor_name=anchor,
        )
        if rewritten is None or rewritten is retreat:
            return keyframes
        updated = list(keyframes)
        updated[retreat_index] = rewritten
        return updated


def bind_multi_finger_place_support_clearance(
    artifact: KeyframePlanArtifact, request: MotionPlanRequest
) -> KeyframePlanArtifact:
    """Apply multi-finger place floor clearance when in scope."""

    return MultiFingerPlaceSupportClearanceBinder().bind(artifact, request)


def bind_grasp_geometry(
    artifact: KeyframePlanArtifact, request: MotionPlanRequest
) -> KeyframePlanArtifact:
    provider = request.task.metadata.get("grasp_geometry_provider")
    if provider in {None, "", "NONE"}:
        return annotate_support_clearance(artifact, request)
    if str(provider).upper() == TWO_FINGER_OPPOSED_CONTACT:
        return TwoFingerOpposedContactBinder().bind(artifact, request)
    raise ValueError(
        f"unsupported grasp geometry provider: {provider!r}; "
        "supply object-independent contact geometry or use the source candidates"
    )


__all__ = [
    "ACQUIRE_WORLD_CONTACT_FRAME",
    "AcquireWorldContactFrameBinder",
    "GraspGeometryBinder",
    "MULTI_FINGER_PLACE_RETREAT_CLEARANCE",
    "MULTI_FINGER_PLACE_SUPPORT_CLEARANCE",
    "MULTI_FINGER_TABLETOP_ENCLOSURE",
    "MultiFingerPlaceSupportClearanceBinder",
    "MultiFingerTabletopEnclosureBinder",
    "OpposedContactSpec",
    "SUCTION_SURFACE_CONTACT",
    "SupportClearanceContext",
    "SuctionSurfaceContactBinder",
    "TWO_FINGER_OPPOSED_CONTACT",
    "TwoFingerOpposedContactBinder",
    "annotate_support_clearance",
    "bind_acquire_world_contact_frames",
    "bind_grasp_geometry",
    "bind_multi_finger_place_support_clearance",
    "bind_multi_finger_tabletop_enclosure",
    "bind_suction_surface_contact",
    "multi_finger_finger_below_tcp_m",
    "multi_finger_place_post_detach_retreat_clearance_m",
    "multi_finger_place_raise_above_support_m",
    "multi_finger_place_retreat_from_place_tcp",
    "multi_finger_place_retreat_standoff_candidates",
    "multi_finger_tabletop_grasp_offset_above_support_m",
    "catalog_lift_reach_standoff_candidates",
    "catalog_pre_grasp_reach_minimum_m",
    "catalog_pre_grasp_reach_standoff_candidates",
    "catalog_reach_fallback_step_m",
    "multi_finger_tabletop_lift_standoff_candidates",
    "multi_finger_tabletop_pre_grasp_standoff_candidates",
    "multi_finger_tabletop_standoff_candidates",
    "reach_standoff_schedule_m",
    "opposed_contact_spec",
    "support_clearance_context",
    "support_clearance_context_from_world",
    "support_clearance_contexts_from_world",
]
