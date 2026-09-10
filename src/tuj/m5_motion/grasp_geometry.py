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


TWO_FINGER_OPPOSED_CONTACT = "TWO_FINGER_OPPOSED_CONTACT"


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


def support_clearance_context_from_world(
    record: object,
    world: WorldSnapshot,
    object_id: str,
    dimensions: tuple[float, float, float] | None = None,
    *,
    tolerance_m: float = 0.01,
    minimum_horizontal_overlap_ratio: float = 0.0,
) -> SupportClearanceContext | None:
    """Resolve support clearance for any metric-bbox object.

    Explicit M1-M4 support relations take precedence.  Otherwise the relation
    is inferred from the object's oriented bbox and overlapping obstacle or
    object bboxes. These are coarse horizontal-support measurements, not local
    contact geometry or proof of graspability. ``object_top_clearance_m`` is
    the bbox top height, never the height of an arbitrary grasp edge.
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
        return None
    bounds = _object_world_bounds(record, dimensions)
    explicit = _explicit_support_clearance(record, world, object_id, bounds)
    if explicit is not None:
        return explicit
    if bounds is None:
        return None
    object_minimum, object_maximum = bounds

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
        gap_m = float(object_minimum[2] - support_maximum[2])
        if (
            overlap_ratio <= 0.0
            or overlap_ratio + 1e-9 < minimum_horizontal_overlap_ratio
            or abs(gap_m) > tolerance_m
        ):
            return
        context = _make_support_context(
            support_id=support_id,
            surface_z_m=float(support_maximum[2]),
            object_bottom_z_m=float(object_minimum[2]),
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

    if not inferred:
        return None
    return min(
        inferred,
        key=lambda item: (
            abs(item.under_clearance_m),
            -item.horizontal_overlap_ratio,
            -item.surface_z_m,
        ),
    )


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
    "GraspGeometryBinder",
    "OpposedContactSpec",
    "SupportClearanceContext",
    "TWO_FINGER_OPPOSED_CONTACT",
    "TwoFingerOpposedContactBinder",
    "annotate_support_clearance",
    "bind_grasp_geometry",
    "opposed_contact_spec",
    "support_clearance_context",
    "support_clearance_context_from_world",
]
