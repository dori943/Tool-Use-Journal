"""Geometry-derived keyframes for placing an attached object in a container.

The policy is independent of task, environment, object, and region names. It
activates only when the live request identifies both an attached target object
and a three-dimensional target region with enough clearance for that object.
Unsupported requests are delegated unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
import hashlib
import math
from typing import Any

import numpy as np

from tuj.m5_motion.attachment_retarget import (
    ATTACHED_OBJECT_POSE_SUBJECT,
    POSE_SUBJECT_KEY,
    POSE_SUBJECT_OBJECT_ID_KEY,
    held_pose_subject,
    retarget_resolved_pose,
)
from tuj.m5_motion.geometry import (
    RelativePoseResolver,
    matrix_quaternion_xyzw,
    quaternion_matrix_xyzw,
    tool_rotation_from_axis,
)
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    KeyframeEventType,
    KeyframePlanArtifact,
    KeyframePlanCandidate,
    KeyframePlannerType,
    KeyframeType,
    ModuleName,
    MotionPlanRequest,
    RelativeKeyframeSpec,
    StrategyGenerationProvenance,
    StrategyGeneratorKind,
)
from tuj.m5_motion.task_semantics import task_operation


PACKING_PROFILE_METADATA_KEY = "packing_profile"

# Coordinates are normalized to available horizontal clearance, not fixed in
# metres. The first candidates favour the near half of a region; later ones
# widen the deterministic search.
_NORMALIZED_XY_CANDIDATES = (
    (-1.00, -0.40),
    (-0.73, 0.00),
    (-0.47, 0.57),
    (-0.87, -0.57),
    (-0.60, 0.00),
    (-0.33, 0.57),
    (-1.00, 0.85),
    (-0.70, 0.85),
    (0.00, 0.00),
    (0.00, -0.85),
    (0.00, 0.85),
    (0.45, -0.45),
    (0.45, 0.45),
    (0.85, -0.85),
    (0.85, 0.00),
    (0.85, 0.85),
)


@dataclass(frozen=True)
class PackingProfile:
    """Tunable policy values; distances are expressed in metres."""

    inset_margin_m: float = 0.002
    minimum_container_height_m: float = 0.05
    minimum_container_to_object_height_ratio: float = 0.5
    hover_clearance_m: float = 0.025
    hover_clearance_object_height_ratio: float = 0.18
    insertion_depth_m: float = 0.012
    insertion_depth_object_height_ratio: float = 0.20
    retreat_clearance_m: float = 0.045
    # Stacked C4-2 objects can keep moving for more than the former 1.6 s
    # release window.  Evaluate containment only after the contact solver has
    # had enough time to settle the free body against existing contents.
    release_hold_duration_s: float = 5.0
    release_event_offset_s: float = 0.1
    tracking_joint_tolerance_rad: float = 0.01
    tracking_max_wait_s: float = 3.0
    tracking_required_consecutive_ticks: int = 5
    candidate_count: int = 6
    occupancy_clearance_m: float = 0.002

    def __post_init__(self) -> None:
        nonnegative = (
            "inset_margin_m",
            "hover_clearance_m",
            "hover_clearance_object_height_ratio",
            "insertion_depth_m",
            "insertion_depth_object_height_ratio",
            "retreat_clearance_m",
            "release_event_offset_s",
            "occupancy_clearance_m",
        )
        positive = (
            "minimum_container_height_m",
            "minimum_container_to_object_height_ratio",
            "release_hold_duration_s",
            "tracking_joint_tolerance_rad",
            "tracking_max_wait_s",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"packing profile {name} must be finite and nonnegative")
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"packing profile {name} must be finite and positive")
        if self.tracking_required_consecutive_ticks < 1:
            raise ValueError("tracking_required_consecutive_ticks must be positive")
        if not 1 <= self.candidate_count <= len(_NORMALIZED_XY_CANDIDATES):
            raise ValueError(
                f"candidate_count must be between 1 and {len(_NORMALIZED_XY_CANDIDATES)}"
            )

    @classmethod
    def from_request(cls, request: MotionPlanRequest) -> "PackingProfile":
        """Merge world defaults with task-local overrides."""

        allowed = {field.name for field in fields(cls)}
        values: dict[str, Any] = {}
        for metadata in (request.world.metadata, request.task.metadata):
            raw = metadata.get(PACKING_PROFILE_METADATA_KEY)
            if raw is None:
                continue
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"{PACKING_PROFILE_METADATA_KEY!r} metadata must be a mapping"
                )
            unknown = set(raw) - allowed
            if unknown:
                raise ValueError(
                    "unknown packing profile fields: " + ", ".join(sorted(unknown))
                )
            values.update(raw)
        integer_fields = {
            "candidate_count",
            "tracking_required_consecutive_ticks",
        }
        for name, value in list(values.items()):
            if name in integer_fields:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"packing profile {name} must be an integer")
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"packing profile {name} must be numeric")
                values[name] = float(value)
        return replace(cls(), **values)


def _positive_xyz(raw: object) -> tuple[float, float, float] | None:
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != 3
    ):
        return None
    try:
        result = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and value > 0.0 for value in result):
        return None
    return result


def _xyz(raw: object) -> tuple[float, float, float] | None:
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != 3
    ):
        return None
    try:
        result = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in result):
        return None
    return result


def _dimensions(record: object) -> tuple[float, float, float] | None:
    if not isinstance(record, Mapping):
        return None
    return _positive_xyz(record.get("dimensions_m"))


def _anchor_z(record: Mapping[str, Any], name: str) -> float | None:
    anchors = record.get("anchors")
    if not isinstance(anchors, Mapping):
        return None
    raw = anchors.get(name)
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != 3
    ):
        return None
    try:
        value = float(raw[2])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _region_top_z(
    region: Mapping[str, Any],
    dimensions: tuple[float, float, float],
) -> float:
    top = _anchor_z(region, "top_center")
    if top is not None:
        return top
    center = _anchor_z(region, "center")
    return (center or 0.0) + dimensions[2] / 2.0


@dataclass(frozen=True)
class PackingPoseCandidate:
    orientation_xyzw: tuple[float, float, float, float]
    dimensions_m: tuple[float, float, float]
    center_offset_m: tuple[float, float, float]

    @property
    def axis_roll(self) -> tuple[tuple[float, float, float], float]:
        rotation = quaternion_matrix_xyzw(self.orientation_xyzw)
        axis = rotation[:, 2]
        base = tool_rotation_from_axis(axis, 0.0)
        roll = math.atan2(
            float(np.dot(base[:, 1], rotation[:, 0])),
            float(np.dot(base[:, 0], rotation[:, 0])),
        )
        return tuple(float(value) for value in axis), roll


def _orientation_candidates(record: object) -> tuple[PackingPoseCandidate, ...]:
    if not isinstance(record, Mapping):
        return ()
    metadata = record.get("packing_metadata")
    if not isinstance(metadata, Mapping):
        return ()
    raw_candidates = metadata.get("orientation_candidates")
    if not isinstance(raw_candidates, Sequence) or isinstance(
        raw_candidates, (str, bytes)
    ):
        return ()
    result: list[PackingPoseCandidate] = []
    reflection_rotations = (
        np.eye(3),
        np.diag((1.0, -1.0, -1.0)),
        np.diag((-1.0, 1.0, -1.0)),
        np.diag((-1.0, -1.0, 1.0)),
    )
    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            continue
        dimensions = _positive_xyz(raw.get("dimensions_m"))
        center = _xyz(raw.get("center_offset_m"))
        quaternion = raw.get("orientation_xyzw")
        if (
            dimensions is None
            or center is None
            or not isinstance(quaternion, Sequence)
            or isinstance(quaternion, (str, bytes))
            or len(quaternion) != 4
        ):
            continue
        try:
            quaternion_values = tuple(float(value) for value in quaternion)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in quaternion_values):
            continue
        rotation = quaternion_matrix_xyzw(quaternion_values)
        center_vector = np.asarray(center, dtype=float)
        for reflection in reflection_rotations:
            reflected_rotation = reflection @ rotation
            reflected_center = reflection @ center_vector
            result.append(
                PackingPoseCandidate(
                    orientation_xyzw=matrix_quaternion_xyzw(reflected_rotation),
                    dimensions_m=dimensions,
                    center_offset_m=tuple(
                        float(value) for value in reflected_center
                    ),
                )
            )
    return tuple(result)


def _normalized_xy_candidates(record: object) -> tuple[tuple[float, float], ...]:
    """Return object-specific placement preferences followed by safe defaults."""
    preferred: list[tuple[float, float]] = []
    metadata = record.get("packing_metadata") if isinstance(record, Mapping) else None
    raw = metadata.get("normalized_xy_candidates") if isinstance(metadata, Mapping) else None
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes))
                or len(item) != 2
            ):
                continue
            try:
                pair = (float(item[0]), float(item[1]))
            except (TypeError, ValueError):
                continue
            if all(
                math.isfinite(value) and -1.0 <= value <= 1.0
                for value in pair
            ) and pair not in preferred:
                preferred.append(pair)
    for pair in _NORMALIZED_XY_CANDIDATES:
        if pair not in preferred:
            preferred.append(pair)
    return tuple(preferred)


def _packing_clearance(record: object, key: str, default: float) -> float:
    metadata = record.get("packing_metadata") if isinstance(record, Mapping) else None
    raw = metadata.get(key) if isinstance(metadata, Mapping) else None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value >= 0.0 else default


def _record_pose(
    record: object,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(record, Mapping):
        return None
    pose = record.get("pose")
    if not isinstance(pose, Mapping):
        return None
    position = _xyz(pose.get("position_m"))
    quaternion = pose.get("orientation_xyzw")
    if (
        position is None
        or not isinstance(quaternion, Sequence)
        or isinstance(quaternion, (str, bytes))
        or len(quaternion) != 4
    ):
        return None
    try:
        rotation = quaternion_matrix_xyzw(tuple(float(value) for value in quaternion))
    except (TypeError, ValueError):
        return None
    return np.asarray(position, dtype=float), rotation


def _record_collision_points(record: object) -> np.ndarray | None:
    if not isinstance(record, Mapping):
        return None
    raw = record.get("collision_points_m")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        try:
            points = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            points = np.empty((0, 3))
        if points.ndim == 2 and points.shape[1:] == (3,) and len(points) > 0 and (
            np.all(np.isfinite(points))
        ):
            return points
    dimensions = _dimensions(record)
    if dimensions is None:
        return None
    half = 0.5 * np.asarray(dimensions, dtype=float)
    return np.asarray(
        [
            (sx * half[0], sy * half[1], sz * half[2])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class PackingFootprint:
    object_id: str
    lower_xy_m: tuple[float, float]
    upper_xy_m: tuple[float, float]


def _occupied_footprints(
    request: MotionPlanRequest,
    *,
    target_id: str,
    region_id: str,
    region_dimensions_m: tuple[float, float, float],
    interior_center_m: tuple[float, float, float],
) -> tuple[PackingFootprint, ...]:
    region = request.world.objects.get(region_id)
    region_pose = _record_pose(region)
    if region_pose is None:
        return ()
    region_position, region_rotation = region_pose
    interior_center = np.asarray(interior_center_m, dtype=float)
    interior_half = 0.5 * np.asarray(region_dimensions_m, dtype=float)
    interior_lower = interior_center - interior_half
    interior_upper = interior_center + interior_half
    occupied: list[PackingFootprint] = []
    for object_id, record in request.world.objects.items():
        if object_id in {target_id, region_id}:
            continue
        object_pose = _record_pose(record)
        points = _record_collision_points(record)
        if object_pose is None or points is None:
            continue
        object_position, object_rotation = object_pose
        world_points = object_position + points @ object_rotation.T
        region_points = (world_points - region_position) @ region_rotation
        lower = np.min(region_points, axis=0)
        upper = np.max(region_points, axis=0)
        if not np.all(upper >= interior_lower) or not np.all(lower <= interior_upper):
            continue
        occupied.append(
            PackingFootprint(
                object_id=str(object_id),
                lower_xy_m=(float(lower[0]), float(lower[1])),
                upper_xy_m=(float(upper[0]), float(upper[1])),
            )
        )
    return tuple(sorted(occupied, key=lambda item: item.object_id))


@dataclass(frozen=True)
class PackingTarget:
    x_m: float
    y_m: float
    hover_z_m: float
    release_z_m: float
    retreat_z_m: float
    pose: PackingPoseCandidate | None = None
    blocking_object_ids: tuple[str, ...] = ()
    release_clearance_m: float | None = None


@dataclass(frozen=True)
class PackingBinding:
    object_id: str
    region_id: str
    object_dimensions_m: tuple[float, float, float]
    region_dimensions_m: tuple[float, float, float]
    region_top_z_m: float
    profile: PackingProfile
    # Never propose geometry with less clearance than the collision validator
    # will require. Keeping these effective values on the binding prevents a
    # profile tuned with a small inset from contradicting the request.
    inset_margin_m: float
    occupancy_clearance_m: float
    pose_candidates: tuple[PackingPoseCandidate, ...] = ()
    normalized_xy_candidates: tuple[tuple[float, float], ...] = (
        _NORMALIZED_XY_CANDIDATES
    )
    pose_hover_clearance_m: float = 0.025
    pose_release_clearance_m: float = 0.006
    occupied_footprints: tuple[PackingFootprint, ...] = ()

    @classmethod
    def resolve(cls, request: MotionPlanRequest) -> "PackingBinding | None":
        operation = task_operation(request.task)
        if operation not in {"TRANSPORT", "PLACE", "PLACE_ON"}:
            return None
        object_id = held_pose_subject(request)
        region_id = request.task.goal.target_region_id
        if object_id is None or not region_id or object_id == region_id:
            return None
        object_record = request.world.objects.get(object_id)
        region_record = request.world.objects.get(region_id)
        if not isinstance(region_record, Mapping):
            return None
        object_dimensions = _dimensions(object_record)
        region_dimensions = _dimensions(region_record)
        if object_dimensions is None or region_dimensions is None:
            return None
        profile = PackingProfile.from_request(request)
        inset_margin = max(
            profile.inset_margin_m,
            float(request.constraints.collision_margin_m),
        )
        occupancy_clearance = max(
            profile.occupancy_clearance_m,
            float(request.constraints.collision_margin_m),
        )
        region_metadata = region_record.get("packing_metadata")
        if isinstance(region_metadata, Mapping):
            interior_dimensions = _positive_xyz(
                region_metadata.get("interior_dimensions_m")
            )
            if interior_dimensions is not None:
                region_dimensions = interior_dimensions
            raw_top = region_metadata.get("opening_top_z_m")
            try:
                metadata_top = float(raw_top)
            except (TypeError, ValueError):
                metadata_top = math.nan
            interior_center = _xyz(region_metadata.get("interior_center_m"))
        else:
            metadata_top = math.nan
            interior_center = None
        if interior_center is None:
            interior_center = (0.0, 0.0, region_dimensions[2] / 2.0)
        pose_candidates = tuple(
            candidate
            for candidate in _orientation_candidates(object_record)
            if all(
                candidate.dimensions_m[axis]
                + 2.0 * inset_margin
                < region_dimensions[axis]
                for axis in range(3)
            )
        )
        fit_dimensions = (
            pose_candidates[0].dimensions_m
            if pose_candidates
            else object_dimensions
        )
        if region_dimensions[2] < max(
            profile.minimum_container_height_m,
            fit_dimensions[2]
            * profile.minimum_container_to_object_height_ratio,
        ):
            return None
        safe_x = (
            region_dimensions[0] / 2.0
            - fit_dimensions[0] / 2.0
            - inset_margin
        )
        safe_y = (
            region_dimensions[1] / 2.0
            - fit_dimensions[1] / 2.0
            - inset_margin
        )
        if safe_x <= 0.0 or safe_y <= 0.0:
            return None
        occupied = _occupied_footprints(
            request,
            target_id=object_id,
            region_id=region_id,
            region_dimensions_m=region_dimensions,
            interior_center_m=interior_center,
        )
        return cls(
            object_id=object_id,
            region_id=region_id,
            object_dimensions_m=object_dimensions,
            region_dimensions_m=region_dimensions,
            region_top_z_m=(
                metadata_top
                if math.isfinite(metadata_top)
                else _region_top_z(region_record, region_dimensions)
            ),
            profile=profile,
            inset_margin_m=inset_margin,
            occupancy_clearance_m=occupancy_clearance,
            pose_candidates=pose_candidates,
            normalized_xy_candidates=_normalized_xy_candidates(object_record),
            pose_hover_clearance_m=_packing_clearance(
                object_record,
                "pose_hover_clearance_m",
                profile.hover_clearance_m,
            ),
            pose_release_clearance_m=_packing_clearance(
                object_record,
                "pose_release_clearance_m",
                0.006,
            ),
            occupied_footprints=occupied,
        )

    @property
    def safe_half_extents_xy_m(self) -> tuple[float, float]:
        return (
            self.region_dimensions_m[0] / 2.0
            - self.object_dimensions_m[0] / 2.0
            - self.inset_margin_m,
            self.region_dimensions_m[1] / 2.0
            - self.object_dimensions_m[1] / 2.0
            - self.inset_margin_m,
        )

    @property
    def hover_center_z_m(self) -> float:
        object_height = self.object_dimensions_m[2]
        clearance = max(
            self.profile.hover_clearance_m,
            object_height * self.profile.hover_clearance_object_height_ratio,
        )
        return self.region_top_z_m + object_height + clearance

    @property
    def release_center_z_m(self) -> float:
        object_height = self.object_dimensions_m[2]
        insertion = min(
            self.profile.insertion_depth_m,
            object_height * self.profile.insertion_depth_object_height_ratio,
        )
        return self.region_top_z_m + object_height / 2.0 - insertion

    @property
    def retreat_z_m(self) -> float:
        return (
            self.region_top_z_m
            + self.object_dimensions_m[2]
            + self.profile.retreat_clearance_m
        )

    def centers(self, *, release_variants: bool = False) -> list[PackingTarget]:
        if not self.pose_candidates:
            safe_x, safe_y = self.safe_half_extents_xy_m
            candidates = [
                PackingTarget(
                    x_m=x_fraction * safe_x,
                    y_m=y_fraction * safe_y,
                    hover_z_m=self.hover_center_z_m,
                    release_z_m=self.release_center_z_m,
                    retreat_z_m=self.retreat_z_m,
                )
                for x_fraction, y_fraction in self.normalized_xy_candidates
            ]
            return self._rank_by_occupancy(candidates)

        result: list[PackingTarget] = []
        for index, (x_fraction, y_fraction) in enumerate(
            self.normalized_xy_candidates
        ):
            pose = self.pose_candidates[index % len(self.pose_candidates)]
            safe_x = (
                self.region_dimensions_m[0] / 2.0
                - pose.dimensions_m[0] / 2.0
                - self.inset_margin_m
            )
            safe_y = (
                self.region_dimensions_m[1] / 2.0
                - pose.dimensions_m[1] / 2.0
                - self.inset_margin_m
            )
            center_x = x_fraction * safe_x
            center_y = y_fraction * safe_y
            center_offset = pose.center_offset_m
            object_height = pose.dimensions_m[2]
            clearance = max(
                self.pose_hover_clearance_m,
                object_height * self.profile.hover_clearance_object_height_ratio,
            )
            release_clearances = [self.pose_release_clearance_m]
            if release_variants:
                # A diagonally fitting object can clear the opening while the
                # mounted hand or wrist cannot. Offer an elevated release that
                # stays just below the hover pose and relies on the configured
                # physical settle hold, rather than lowering the hand between
                # the walls. The value scales with object geometry.
                elevated = max(
                    self.pose_release_clearance_m,
                    min(0.9 * clearance, 0.2 * object_height),
                )
                if elevated > self.pose_release_clearance_m + 1e-6:
                    release_clearances.append(elevated)
            for release_clearance in release_clearances:
                result.append(
                    PackingTarget(
                        x_m=center_x - center_offset[0],
                        y_m=center_y - center_offset[1],
                        hover_z_m=(
                            self.region_top_z_m
                            + object_height / 2.0
                            + clearance
                            - center_offset[2]
                        ),
                        release_z_m=(
                            self.region_top_z_m
                            + object_height / 2.0
                            + release_clearance
                            - center_offset[2]
                        ),
                        retreat_z_m=(
                            self.region_top_z_m
                            + object_height / 2.0
                            + release_clearance
                            - center_offset[2]
                            + self.profile.retreat_clearance_m
                        ),
                        pose=pose,
                        release_clearance_m=release_clearance,
                    )
                )
        return self._rank_by_occupancy(result)

    def _rank_by_occupancy(
        self, candidates: Sequence[PackingTarget]
    ) -> list[PackingTarget]:
        """Prefer the first non-overlapping slots while retaining fallbacks."""

        ranked: list[tuple[int, float, int, PackingTarget]] = []
        clearance = self.occupancy_clearance_m
        for index, target in enumerate(candidates):
            pose = target.pose
            dimensions = pose.dimensions_m if pose is not None else self.object_dimensions_m
            offset = pose.center_offset_m if pose is not None else (0.0, 0.0, 0.0)
            center_x = target.x_m + offset[0]
            center_y = target.y_m + offset[1]
            lower_x = center_x - dimensions[0] / 2.0
            upper_x = center_x + dimensions[0] / 2.0
            lower_y = center_y - dimensions[1] / 2.0
            upper_y = center_y + dimensions[1] / 2.0
            blockers: list[str] = []
            overlap_area = 0.0
            for occupied in self.occupied_footprints:
                overlap_x = min(upper_x, occupied.upper_xy_m[0] + clearance) - max(
                    lower_x, occupied.lower_xy_m[0] - clearance
                )
                overlap_y = min(upper_y, occupied.upper_xy_m[1] + clearance) - max(
                    lower_y, occupied.lower_xy_m[1] - clearance
                )
                if overlap_x > 0.0 and overlap_y > 0.0:
                    blockers.append(occupied.object_id)
                    overlap_area += overlap_x * overlap_y
            candidate = replace(
                target,
                blocking_object_ids=tuple(sorted(blockers)),
            )
            ranked.append((len(blockers), overlap_area, index, candidate))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in ranked[: self.profile.candidate_count]]


def _identity(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _provenance(request: MotionPlanRequest, label: str):
    digest = _identity(f"{request.request_id}|{label}|packing-geometry-v1")
    return (
        ArtifactProvenance(
            artifact_id=f"keyframe-plan-artifact:{digest}",
            artifact_type="KeyframePlanArtifact",
            produced_by=ModuleName.MOTION_PLANNER,
            invocation_id=f"packing-geometry:{request.request_id}",
            input_artifact_ids=[request.provenance.artifact_id],
            metadata={"policy": "attachment_aware_packing_geometry_v1"},
        ),
        digest,
    )


def _strategy_provenance(request: MotionPlanRequest, index: int):
    digest = _identity(
        f"{request.request_id}|candidate|{index}|packing-geometry-v1"
    )
    return StrategyGenerationProvenance(
        generator_kind=StrategyGeneratorKind.TASK_GEOMETRY,
        generator_id="ATTACHMENT_AWARE_PACKING_GEOMETRY_V1",
        input_hash=digest,
        attempt_index=1,
    )


def _anchor(
    request: MotionPlanRequest,
    binding: PackingBinding,
    name: str,
    position: tuple[float, float, float],
) -> None:
    region = request.world.objects[binding.region_id]
    region.setdefault("anchors", {})[name] = list(position)


def _held_metadata(binding: PackingBinding) -> dict[str, str]:
    return {
        POSE_SUBJECT_KEY: ATTACHED_OBJECT_POSE_SUBJECT,
        POSE_SUBJECT_OBJECT_ID_KEY: binding.object_id,
    }


def _object_keyframe(
    binding: PackingBinding,
    *,
    keyframe_id: str,
    kind: KeyframeType,
    anchor: str,
    planner: KeyframePlannerType,
    pose: PackingPoseCandidate | None = None,
    events: tuple[KeyframeEventType, ...] = (),
    motion_role: str,
    blocking_object_ids: tuple[str, ...] = (),
    release_clearance_m: float | None = None,
) -> RelativeKeyframeSpec:
    metadata: dict[str, Any] = _held_metadata(binding)
    metadata.update(
        packing_motion_role=motion_role,
        packing_candidate_blockers=list(blocking_object_ids),
        packing_occupied_object_ids=[
            footprint.object_id for footprint in binding.occupied_footprints
        ],
        packing_inset_margin_m=binding.inset_margin_m,
        packing_occupancy_clearance_m=binding.occupancy_clearance_m,
    )
    if release_clearance_m is not None:
        metadata["packing_release_clearance_m"] = release_clearance_m
    if events:
        profile = binding.profile
        metadata.update(
            event_target_id=binding.object_id,
            allow_release_contact=True,
            hold_duration_after_s=profile.release_hold_duration_s,
            event_time_offsets_s={
                event.value: profile.release_event_offset_s for event in events
            },
            tracking_settle={
                "joint_tolerance_rad": profile.tracking_joint_tolerance_rad,
                "max_wait_s": profile.tracking_max_wait_s,
                "required_consecutive_ticks": (
                    profile.tracking_required_consecutive_ticks
                ),
            },
        )
    if pose is None:
        approach_axis = (0.0, 0.0, 1.0)
        roll_rad = 0.0
    else:
        approach_axis, roll_rad = pose.axis_roll
        metadata["packing_orientation_xyzw"] = list(pose.orientation_xyzw)
        metadata["packing_bounds_dimensions_m"] = list(pose.dimensions_m)
    return RelativeKeyframeSpec(
        keyframe_id=keyframe_id,
        keyframe_type=kind,
        frame_ref=f"object:{binding.region_id}",
        anchor=anchor,
        approach_axis_xyz=approach_axis,
        tool_axis_to_align="+z",
        offset_along_approach_m=0.0,
        roll_rad=roll_rad,
        planner=planner,
        events_after=list(events),
        metadata=metadata,
    )


def _eef_retreat_keyframe(
    request: MotionPlanRequest,
    binding: PackingBinding,
    *,
    place_keyframe: RelativeKeyframeSpec,
    keyframe_id: str,
    anchor: str,
    blocking_object_ids: tuple[str, ...],
    release_clearance_m: float | None,
) -> RelativeKeyframeSpec:
    """Retreat from the release EE pose along the container's local +Z axis."""

    place_object_pose = RelativePoseResolver(request.world).resolve(place_keyframe)
    place_eef_pose = retarget_resolved_pose(
        request.world,
        place_keyframe,
        place_object_pose,
    )
    region_pose = _record_pose(request.world.objects[binding.region_id])
    if region_pose is None:  # Binding resolution already requires this pose.
        raise ValueError(f"packing region {binding.region_id!r} has no usable pose")
    region_position, region_rotation = region_pose
    release_eef_local = region_rotation.T @ (
        np.asarray(place_eef_pose.position_m, dtype=float) - region_position
    )
    retreat_eef_local = release_eef_local + np.asarray(
        (0.0, 0.0, binding.profile.retreat_clearance_m),
        dtype=float,
    )
    _anchor(
        request,
        binding,
        anchor,
        tuple(float(value) for value in retreat_eef_local),
    )
    return RelativeKeyframeSpec(
        keyframe_id=keyframe_id,
        keyframe_type=KeyframeType.RETREAT,
        frame_ref=f"object:{binding.region_id}",
        anchor=anchor,
        approach_axis_xyz=(0.0, 0.0, 1.0),
        tool_axis_to_align="+z",
        offset_along_approach_m=0.0,
        roll_rad=0.0,
        planner=KeyframePlannerType.CARTESIAN,
        metadata={
            "packing_orientation_xyzw": list(place_eef_pose.orientation_xyzw),
            "packing_motion_role": "CONSTRAINED_RETREAT",
            "packing_candidate_blockers": list(blocking_object_ids),
            "packing_occupied_object_ids": [
                footprint.object_id for footprint in binding.occupied_footprints
            ],
            "packing_inset_margin_m": binding.inset_margin_m,
            "packing_occupancy_clearance_m": binding.occupancy_clearance_m,
            "packing_release_clearance_m": release_clearance_m,
        },
    )


def _transport(
    request: MotionPlanRequest,
    binding: PackingBinding,
) -> KeyframePlanArtifact:
    candidates: list[KeyframePlanCandidate] = []
    for index, target in enumerate(
        binding.centers(), start=1
    ):
        anchor = f"packing_transport_{index}"
        _anchor(
            request,
            binding,
            anchor,
            (target.x_m, target.y_m, target.hover_z_m),
        )
        prefix = f"{request.task.subgoal_id}:packing_transport_{index}"
        candidates.append(
            KeyframePlanCandidate(
                strategy_id=prefix,
                keyframes=[
                    _object_keyframe(
                        binding,
                        keyframe_id=f"{prefix}:1:transfer",
                        kind=KeyframeType.TRANSFER,
                        anchor=anchor,
                        planner=KeyframePlannerType.SAMPLING_BASED,
                        pose=target.pose,
                        motion_role="FREE_SPACE_TRANSFER",
                        blocking_object_ids=target.blocking_object_ids,
                        release_clearance_m=target.release_clearance_m,
                    ),
                    _object_keyframe(
                        binding,
                        keyframe_id=f"{prefix}:2:pre_place",
                        kind=KeyframeType.PRE_PLACE,
                        anchor=anchor,
                        planner=KeyframePlannerType.SAMPLING_BASED,
                        pose=target.pose,
                        motion_role="FREE_SPACE_ALIGNMENT",
                        blocking_object_ids=target.blocking_object_ids,
                        release_clearance_m=target.release_clearance_m,
                    ),
                ],
                rationale=(
                    f"Move attached {binding.object_id} above {binding.region_id} "
                    "using a geometry-scaled container-local candidate."
                ),
                provenance=_strategy_provenance(request, index),
            )
        )
    provenance, digest = _provenance(request, "transport")
    return KeyframePlanArtifact(
        artifact_id=f"keyframe-plan:{digest}",
        provenance=provenance,
        scene_signature=request.world.scene.signature,
        subgoal_id=request.task.subgoal_id,
        candidates=candidates,
    )


def _place(
    request: MotionPlanRequest,
    binding: PackingBinding,
) -> KeyframePlanArtifact:
    candidates: list[KeyframePlanCandidate] = []
    events = (KeyframeEventType.DETACH_OBJECT, KeyframeEventType.GRIPPER_OPEN)
    for index, target in enumerate(
        binding.centers(release_variants=True), start=1
    ):
        high = f"packing_place_high_{index}"
        low = f"packing_release_{index}"
        retreat = f"packing_retreat_{index}"
        _anchor(
            request,
            binding,
            high,
            (target.x_m, target.y_m, target.hover_z_m),
        )
        _anchor(
            request,
            binding,
            low,
            (target.x_m, target.y_m, target.release_z_m),
        )
        prefix = f"{request.task.subgoal_id}:packing_place_{index}"
        place_keyframe = _object_keyframe(
            binding,
            keyframe_id=f"{prefix}:2:release",
            kind=KeyframeType.PLACE,
            anchor=low,
            planner=KeyframePlannerType.CARTESIAN,
            pose=target.pose,
            events=events,
            motion_role="CONSTRAINED_INSERTION",
            blocking_object_ids=target.blocking_object_ids,
            release_clearance_m=target.release_clearance_m,
        )
        if target.pose is not None:
            _anchor(
                request,
                binding,
                retreat,
                (target.x_m, target.y_m, target.retreat_z_m),
            )
            retreat_keyframe = _object_keyframe(
                binding,
                keyframe_id=f"{prefix}:3:retreat",
                kind=KeyframeType.RETREAT,
                anchor=retreat,
                planner=KeyframePlannerType.CARTESIAN,
                pose=target.pose,
                motion_role="CONSTRAINED_RETREAT",
                blocking_object_ids=target.blocking_object_ids,
                release_clearance_m=target.release_clearance_m,
            )
        else:
            retreat_keyframe = _eef_retreat_keyframe(
                request,
                binding,
                place_keyframe=place_keyframe,
                keyframe_id=f"{prefix}:3:retreat",
                anchor=retreat,
                blocking_object_ids=target.blocking_object_ids,
                release_clearance_m=target.release_clearance_m,
            )
        candidates.append(
            KeyframePlanCandidate(
                strategy_id=prefix,
                keyframes=[
                    _object_keyframe(
                        binding,
                        keyframe_id=f"{prefix}:1:pre_place",
                        kind=KeyframeType.PRE_PLACE,
                        anchor=high,
                        # Free-space alignment may need a substantial payload
                        # reorientation, so it remains sampling based.
                        planner=KeyframePlannerType.SAMPLING_BASED,
                        pose=target.pose,
                        motion_role="FREE_SPACE_ALIGNMENT",
                        blocking_object_ids=target.blocking_object_ids,
                        release_clearance_m=target.release_clearance_m,
                    ),
                    place_keyframe,
                    retreat_keyframe,
                ],
                rationale=(
                    f"Lower {binding.object_id} through the opening of "
                    f"{binding.region_id}, release it, then retreat."
                ),
                provenance=_strategy_provenance(request, index),
            )
        )
    provenance, digest = _provenance(request, "place")
    return KeyframePlanArtifact(
        artifact_id=f"keyframe-plan:{digest}",
        provenance=provenance,
        scene_signature=request.world.scene.signature,
        subgoal_id=request.task.subgoal_id,
        candidates=candidates,
    )


class PackingKeyframeProvider:
    """Generate generic container-packing poses, with provider fallback."""

    def __init__(self, fallback):
        self.fallback = fallback

    def generate(self, request: MotionPlanRequest) -> KeyframePlanArtifact:
        binding = PackingBinding.resolve(request)
        if binding is None:
            return self.fallback.generate(request)
        if task_operation(request.task) == "TRANSPORT":
            return _transport(request, binding)
        return _place(request, binding)


__all__ = [
    "PACKING_PROFILE_METADATA_KEY",
    "PackingBinding",
    "PackingFootprint",
    "PackingKeyframeProvider",
    "PackingProfile",
]
