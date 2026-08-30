"""Geometry-grounded push-to-region planning primitives.

The planner is independent of concrete task IDs and tool names.  A selected
``ToolContactPatch`` plus observed object/region geometry is enough to generate
approach, engage, push, and retract keyframes.  Tool-to-EEF grasp transforms can
be handled by supplying a different ``ContactPoseResolver``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tuj.m4_motion.geometry import (
    matrix_quaternion_xyzw,
    quaternion_matrix_xyzw,
    tool_rotation_from_axis,
)
from tuj.m4_motion.profiles import PushPlanningProfile
from tuj.m4_motion.schema import (
    ArtifactProvenance,
    KeyframePlanArtifact,
    KeyframePlanCandidate,
    KeyframePlannerType,
    KeyframeType,
    ModuleName,
    MotionPlanRequest,
    Pose,
    RelativeKeyframeSpec,
    StrategyGenerationProvenance,
    StrategyGeneratorKind,
    ToolContactPatch,
    WorldSnapshot,
)
from tuj.m4_motion.tool_affordance import (
    ToolAffordanceProvider,
    select_contact_patch,
)


class PushToRegionError(ValueError):
    """Observed geometry or requested contact intent cannot produce a push."""


@dataclass(frozen=True, slots=True)
class PushToRegionGeometry:
    target_id: str
    region_id: str
    push_direction_world: tuple[float, float, float]
    start_contact_position_world_m: tuple[float, float, float]
    end_contact_position_world_m: tuple[float, float, float]
    target_goal_center_world_m: tuple[float, float, float]
    required_push_distance_m: float


def _record_geometry(
    world: WorldSnapshot,
    object_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    record = world.objects.get(object_id)
    if not isinstance(record, Mapping):
        raise PushToRegionError(f"object {object_id!r} is missing from the world")
    pose = record.get("pose")
    if not isinstance(pose, Mapping):
        raise PushToRegionError(f"object {object_id!r} has no pose")
    position = np.asarray(pose.get("position_m"), dtype=float)
    dimensions = np.asarray(record.get("dimensions_m"), dtype=float)
    if position.shape != (3,) or dimensions.shape != (3,):
        raise PushToRegionError(
            f"object {object_id!r} requires 3D position and dimensions"
        )
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(dimensions)):
        raise PushToRegionError(f"object {object_id!r} geometry must be finite")
    if np.any(dimensions <= 0.0):
        raise PushToRegionError(f"object {object_id!r} dimensions must be positive")
    return position, dimensions


def _record_position(world: WorldSnapshot, object_id: str) -> np.ndarray:
    record = world.objects.get(object_id)
    pose = record.get("pose") if isinstance(record, Mapping) else None
    position = (
        np.asarray(pose.get("position_m"), dtype=float)
        if isinstance(pose, Mapping)
        else np.asarray((), dtype=float)
    )
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise PushToRegionError(f"object {object_id!r} requires a finite 3D position")
    return position


def _support_radius_xy(half_size: np.ndarray, direction: np.ndarray) -> float:
    return float(np.dot(np.abs(direction), half_size))


def push_to_region_geometry(
    world: WorldSnapshot,
    *,
    target_id: str,
    region_id: str,
    profile: PushPlanningProfile,
    maximum_push_distance_m: float | None = None,
) -> PushToRegionGeometry:
    """Ground a radial push using full object and region footprints."""

    target_position, target_size = _record_geometry(world, target_id)
    region_position, region_size = _record_geometry(world, region_id)
    outward = target_position[:2] - region_position[:2]
    distance = float(np.linalg.norm(outward))
    if distance <= 1e-9:
        raise PushToRegionError(f"target {target_id!r} is already at region center")
    outward /= distance
    push_xy = -outward
    usable_half_size = region_size[:2] * 0.5 - target_size[:2] * 0.5
    if np.any(usable_half_size <= profile.goal_inset_margin_m):
        raise PushToRegionError(
            f"region {region_id!r} is too small for target {target_id!r}"
        )
    radial_limits = [
        float(usable_half_size[axis] / abs(outward[axis]))
        for axis in range(2)
        if abs(float(outward[axis])) > 1e-9
    ]
    boundary_radius = min(radial_limits) - profile.goal_inset_margin_m
    goal_radius = max(0.0, boundary_radius)
    target_goal_xy = region_position[:2] + outward * goal_radius
    required = max(0.0, distance - goal_radius)
    if maximum_push_distance_m is not None:
        if maximum_push_distance_m <= 0.0:
            raise PushToRegionError("maximum push distance must be positive")
        required = min(required, maximum_push_distance_m)
        target_goal_xy = target_position[:2] + push_xy * required

    target_support = _support_radius_xy(target_size[:2] * 0.5, outward)
    start_xy = (
        target_position[:2]
        + outward * (target_support + profile.approach_standoff_m)
    )
    end_xy = target_goal_xy + outward * target_support
    contact_z = float(
        target_position[2]
        + (profile.contact_height_fraction - 0.5) * target_size[2]
    )
    return PushToRegionGeometry(
        target_id=target_id,
        region_id=region_id,
        push_direction_world=(float(push_xy[0]), float(push_xy[1]), 0.0),
        start_contact_position_world_m=(float(start_xy[0]), float(start_xy[1]), contact_z),
        end_contact_position_world_m=(float(end_xy[0]), float(end_xy[1]), contact_z),
        target_goal_center_world_m=(
            float(target_goal_xy[0]),
            float(target_goal_xy[1]),
            float(target_position[2]),
        ),
        required_push_distance_m=required,
    )


def target_fully_inside_region(
    world: WorldSnapshot,
    *,
    target_id: str,
    region_id: str,
    inset_margin_m: float = 0.0,
) -> bool:
    target_position, target_size = _record_geometry(world, target_id)
    region_position, region_size = _record_geometry(world, region_id)
    available = region_size[:2] * 0.5 - target_size[:2] * 0.5 - inset_margin_m
    return bool(
        np.all(available >= 0.0)
        and np.all(np.abs(target_position[:2] - region_position[:2]) <= available)
    )


def order_targets_around_region(
    world: WorldSnapshot,
    target_ids: Sequence[str],
    region_id: str,
) -> tuple[str, ...]:
    region_position = _record_position(world, region_id)

    def key(target_id: str) -> tuple[float, float, str]:
        position = _record_position(world, target_id)
        delta = position[:2] - region_position[:2]
        return (
            float(math.atan2(delta[1], delta[0])),
            -float(np.linalg.norm(delta)),
            target_id,
        )

    return tuple(sorted(dict.fromkeys(target_ids), key=key))


def cleanup_target_ids(
    world: WorldSnapshot,
    target_ids: Sequence[str],
    *,
    region_id: str,
    reference_positions_m: Mapping[str, Sequence[float]],
    maximum_support_error_m: float,
) -> tuple[str, ...]:
    if maximum_support_error_m < 0.0:
        raise ValueError("maximum support error must be non-negative")
    result: list[str] = []
    for target_id in target_ids:
        position, _ = _record_geometry(world, target_id)
        reference = np.asarray(reference_positions_m[target_id], dtype=float)
        if reference.shape != (3,):
            raise ValueError(f"reference position for {target_id!r} must be 3D")
        if (
            not target_fully_inside_region(
                world, target_id=target_id, region_id=region_id
            )
            or abs(float(position[2] - reference[2])) > maximum_support_error_m
        ):
            result.append(target_id)
    return tuple(result)


def reduced_contact_step_distance(
    current_distance_m: float,
    *,
    minimum_distance_m: float,
    retry_scale: float,
) -> float:
    if current_distance_m <= 0.0 or minimum_distance_m <= 0.0:
        raise ValueError("contact step distances must be positive")
    if not 0.0 < retry_scale <= 1.0:
        raise ValueError("retry scale must be within (0, 1]")
    return min(
        current_distance_m,
        max(minimum_distance_m, current_distance_m * retry_scale),
    )


class ContactPoseResolver(Protocol):
    def resolve(
        self,
        patch: ToolContactPatch,
        *,
        contact_position_world_m: Sequence[float],
        push_direction_world: Sequence[float],
    ) -> Pose: ...


class DirectToolContactPoseResolver:
    """Resolve a tool-frame pose by aligning patch normal with push direction.

    It is appropriate when the planned reference frame is the tool frame.  A
    held-tool integration should provide a resolver that additionally inverts
    the live tool-in-EEF grasp transform.
    """

    def resolve(
        self,
        patch: ToolContactPatch,
        *,
        contact_position_world_m: Sequence[float],
        push_direction_world: Sequence[float],
    ) -> Pose:
        local_normal = np.asarray(patch.normal_in_tool_xyz, dtype=float)
        local_tangent = np.asarray(
            patch.tangent_in_tool_xyz or (1.0, 0.0, 0.0), dtype=float
        )
        local_binormal = np.cross(local_normal, local_tangent)
        local_basis = np.column_stack(
            (local_normal, local_tangent, local_binormal)
        )
        if abs(float(np.linalg.det(local_basis))) <= 1e-6:
            raise PushToRegionError(f"patch {patch.patch_id!r} has a degenerate frame")

        world_normal = np.asarray(push_direction_world, dtype=float)
        world_normal /= np.linalg.norm(world_normal)
        vertical = np.asarray((0.0, 0.0, 1.0), dtype=float)
        world_tangent = vertical - float(np.dot(vertical, world_normal)) * world_normal
        if np.linalg.norm(world_tangent) <= 1e-9:
            world_tangent = np.asarray((1.0, 0.0, 0.0), dtype=float)
        world_tangent /= np.linalg.norm(world_tangent)
        world_binormal = np.cross(world_normal, world_tangent)
        world_basis = np.column_stack(
            (world_normal, world_tangent, world_binormal)
        )
        rotation = world_basis @ np.linalg.inv(local_basis)
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        contact = np.asarray(contact_position_world_m, dtype=float)
        patch_offset = np.asarray(patch.position_in_tool_m, dtype=float)
        tool_position = contact - rotation @ patch_offset
        return Pose(
            frame_id="world",
            position_m=tuple(float(value) for value in tool_position),
            orientation_xyzw=matrix_quaternion_xyzw(rotation),
        )


def _orientation_axis_roll(pose: Pose) -> tuple[tuple[float, float, float], float]:
    rotation = quaternion_matrix_xyzw(pose.orientation_xyzw)
    world_axis = rotation[:, 2]
    base = tool_rotation_from_axis(world_axis, 0.0)
    relative = base.T @ rotation
    roll = float(math.atan2(relative[1, 0], relative[0, 0]))
    # The virtual reference frame already carries ``rotation``; +Z in that
    # local frame resolves to ``world_axis`` exactly once.
    return (0.0, 0.0, 1.0), roll


class PushToRegionStrategyProvider:
    """Generate deterministic contact-patch-aware push keyframes."""

    def __init__(
        self,
        affordances: ToolAffordanceProvider,
        *,
        profile: PushPlanningProfile | None = None,
        pose_resolver: ContactPoseResolver | None = None,
    ) -> None:
        self._affordances = affordances
        self._profile = profile or PushPlanningProfile()
        self._pose_resolver = pose_resolver or DirectToolContactPoseResolver()

    @staticmethod
    def _digest(value: object) -> str:
        rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    def generate(self, request: MotionPlanRequest) -> KeyframePlanArtifact:
        task = request.task
        if task.contact is None:
            raise PushToRegionError("push-to-region task requires a contact spec")
        if not task.tool:
            raise PushToRegionError("push-to-region task requires a selected tool")
        region_id = task.goal.target_region_id
        if not region_id:
            raise PushToRegionError("push-to-region task requires target_region_id")
        if not task.target_ids:
            raise PushToRegionError("push-to-region task requires at least one target")
        path_pattern = task.contact.path_pattern.strip().upper()
        if path_pattern not in {"AUTO", "RADIAL"}:
            raise PushToRegionError(
                f"path pattern {task.contact.path_pattern!r} is not implemented"
            )
        patch = select_contact_patch(
            self._affordances.patches(task.tool, request.world),
            task.contact,
        )
        ordered = order_targets_around_region(
            request.world, task.target_ids, region_id
        )
        keyframes: list[RelativeKeyframeSpec] = []
        frame_ids: list[str] = []
        for target_id in ordered:
            geometry = push_to_region_geometry(
                request.world,
                target_id=target_id,
                region_id=region_id,
                profile=self._profile,
            )
            start = self._pose_resolver.resolve(
                patch,
                contact_position_world_m=geometry.start_contact_position_world_m,
                push_direction_world=geometry.push_direction_world,
            )
            end = self._pose_resolver.resolve(
                patch,
                contact_position_world_m=geometry.end_contact_position_world_m,
                push_direction_world=geometry.push_direction_world,
            )
            poses = {
                "hover_start": start.model_copy(
                    update={
                        "position_m": (
                            start.position_m[0],
                            start.position_m[1],
                            start.position_m[2] + self._profile.hover_height_m,
                        )
                    }
                ),
                "engage": start,
                "push": end,
                "hover_end": end.model_copy(
                    update={
                        "position_m": (
                            end.position_m[0],
                            end.position_m[1],
                            end.position_m[2] + self._profile.hover_height_m,
                        )
                    }
                ),
            }
            for suffix, pose in poses.items():
                frame_id = f"contact_push_{target_id}_{suffix}"
                request.world.objects[frame_id] = {
                    "pose": pose.model_dump(mode="json"),
                    "virtual_reference_frame": True,
                    "target_id": target_id,
                    "contact_patch_id": patch.patch_id,
                }
                frame_ids.append(frame_id)
                axis, roll = _orientation_axis_roll(pose)
                kind = {
                    "hover_start": KeyframeType.TRANSFER,
                    "engage": KeyframeType.CUSTOM,
                    "push": KeyframeType.CUSTOM,
                    "hover_end": KeyframeType.RETREAT,
                }[suffix]
                planner = (
                    KeyframePlannerType.SAMPLING_BASED
                    if suffix == "hover_start"
                    else KeyframePlannerType.CARTESIAN
                )
                keyframes.append(
                    RelativeKeyframeSpec(
                        keyframe_id=f"{task.subgoal_id}:{target_id}:{suffix}",
                        keyframe_type=kind,
                        frame_ref=f"object:{frame_id}",
                        anchor="center",
                        approach_axis_xyz=axis,
                        tool_axis_to_align="+z",
                        offset_along_approach_m=0.0,
                        roll_rad=roll,
                        planner=planner,
                        metadata={
                            "target_id": target_id,
                            "region_id": region_id,
                            "contact_patch_id": patch.patch_id,
                            "required_push_distance_m": geometry.required_push_distance_m,
                            "maintain_contact": task.contact.maintain_contact,
                        },
                    )
                )
        digest = self._digest(
            {
                "request": request.model_dump(mode="json"),
                "patch": patch.model_dump(mode="json"),
                "profile": self._profile,
                "frames": frame_ids,
            }
        )
        return KeyframePlanArtifact(
            artifact_id=f"keyframe-plan:push-to-region:{digest[:24]}",
            provenance=ArtifactProvenance(
                artifact_id=f"keyframe-plan-artifact:push-to-region:{digest[:24]}",
                artifact_type="KeyframePlanArtifact",
                produced_by=ModuleName.MOTION_PLANNER,
                invocation_id=f"push-to-region:{digest[:20]}",
                input_artifact_ids=[request.provenance.artifact_id],
            ),
            scene_signature=request.world.scene.signature,
            subgoal_id=task.subgoal_id,
            candidates=[
                KeyframePlanCandidate(
                    strategy_id=f"{task.subgoal_id}:push-to-region:{patch.patch_id}",
                    keyframes=keyframes,
                    rationale=(
                        "Use observed target and region footprints with the selected "
                        f"{patch.surface_type.value} contact patch."
                    ),
                    provenance=StrategyGenerationProvenance(
                        generator_kind=StrategyGeneratorKind.TASK_GEOMETRY,
                        generator_id="PUSH_TO_REGION_V1",
                        input_hash=digest,
                        attempt_index=1,
                    ),
                    metadata={
                        "contact_patch": patch.model_dump(mode="json"),
                        "target_order": list(ordered),
                        "path_pattern": "RADIAL",
                    },
                )
            ],
        )


__all__ = [
    "ContactPoseResolver",
    "DirectToolContactPoseResolver",
    "PushToRegionError",
    "PushToRegionGeometry",
    "PushToRegionStrategyProvider",
    "cleanup_target_ids",
    "order_targets_around_region",
    "push_to_region_geometry",
    "reduced_contact_step_distance",
    "target_fully_inside_region",
]
