"""Observed-geometry side pushes for an explicit single-target contact spec.

This module generates motion waypoints only. It never edits object state,
attachment physics, feasibility verdicts, or the task's goal evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from .contact_keyframe_validation import (
    ContactKeyframeGeometryError,
    _contact_engagement_tcp_z_m,
    _held_tool_engaging_offsets_m,
    _hover_working_floor_z_m,
    _MIN_HELD_TOOL_AXIS_DOT,
    _retarget_tcp_world_xy,
)
from .geometry import RelativePoseResolver, quaternion_matrix_xyzw
from .profiles import PushPlanningProfile
from .push_to_region import _record_box_geometry
from .schema import KeyframeType, MotionPlanRequest, RelativeKeyframeSpec

_PHASES = (
    KeyframeType.PRE_CONTACT, KeyframeType.CONTACT_START,
    KeyframeType.CONTACT_SWEEP, KeyframeType.CONTACT_END, KeyframeType.RETREAT,
)


@dataclass(frozen=True)
class NativeSidePushGeometry:
    target_id: str
    region_id: str
    target_center_world_m: tuple[float, float, float]
    target_goal_center_world_m: tuple[float, float, float]
    push_direction_xy: tuple[float, float]
    working_patch_offset_xy_m: tuple[float, float]
    patch_start_xy_m: tuple[float, float]
    patch_engage_xy_m: tuple[float, float]
    patch_end_xy_m: tuple[float, float]
    contact_tcp_z_m: float
    hover_tcp_z_m: float
    target_trailing_projection_m: float
    patch_leading_projection_m: float


def _target_points(request: MotionPlanRequest, target_id: str) -> tuple[np.ndarray, np.ndarray]:
    position, rotation, center, size = _record_box_geometry(request.world, target_id)
    points = np.asarray(request.world.objects[target_id].get("collision_points_m", []), dtype=float)
    if not (points.ndim == 2 and points.shape[1:] == (3,) and len(points) >= 4
            and np.isfinite(points).all()):
        signs = np.asarray([(x, y, z) for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
        points = center + signs * size * .5
    return position + center @ rotation.T, position + points @ rotation.T


def native_side_push_geometry(
    request: MotionPlanRequest, *, eef_orientation_xyzw: Sequence[float],
    profile: PushPlanningProfile | None = None,
) -> NativeSidePushGeometry:
    """Place the observed engaging patch behind the target's trailing face.

    Target/region rotations and actual collision footprints determine the
    destination. Contact height remains the common sweep engagement height.
    The supplied attitude is held fixed; no angle search is performed here.
    """
    task = request.task
    if task.contact is None or task.contact.path_pattern.upper() != "SIDE_PUSH":
        raise ContactKeyframeGeometryError("native side push requires explicit SIDE_PUSH contact path")
    if task.contact.primitive.lower() != "sweep" or len(task.target_ids) != 1:
        raise ContactKeyframeGeometryError("native side push requires one sweep target")
    target_id = task.target_ids[0]
    region_id = task.goal.target_region_id
    if not region_id:
        raise ContactKeyframeGeometryError("native side push requires a target region")
    state = request.world.robot_state
    if not task.tool or task.tool not in {state.attached_object_id, state.held_tool_id}:
        raise ContactKeyframeGeometryError("native side push requires the selected tool to be held")
    if state.eef_pose is None:
        raise ContactKeyframeGeometryError("native side push requires the live EEF pose")
    rotation = quaternion_matrix_xyzw(eef_orientation_xyzw)
    live_rotation = quaternion_matrix_xyzw(state.eef_pose.orientation_xyzw)
    if float(np.dot(rotation[:, 2], live_rotation[:, 2])) < _MIN_HELD_TOOL_AXIS_DOT - 1e-9:
        raise ContactKeyframeGeometryError("native side push attitude exceeds the existing held-tool axis bound")
    patch = _held_tool_engaging_offsets_m(request, eef_orientation_xyzw=eef_orientation_xyzw)
    if patch is None or len(patch) == 0:
        raise ContactKeyframeGeometryError("native side push requires observed grasp and engaging collision geometry")
    profile = profile or PushPlanningProfile()
    target_center, points_world = _target_points(request, target_id)
    region_position, region_rotation, region_center, region_size = _record_box_geometry(request.world, region_id)
    if not np.allclose(region_rotation[2], (0., 0., 1.), atol=1e-6):
        raise ContactKeyframeGeometryError("native side push requires a horizontal region")
    region_origin = region_position + region_rotation @ region_center
    target_center_local = region_rotation.T @ (target_center - region_origin)
    relative = (points_world - target_center) @ region_rotation
    lower = -region_size[:2] * .5 + profile.goal_inset_margin_m - relative[:, :2].min(axis=0)
    upper = region_size[:2] * .5 - profile.goal_inset_margin_m - relative[:, :2].max(axis=0)
    if np.any(lower > upper):
        raise ContactKeyframeGeometryError("target footprint cannot fit inside region at its current orientation")
    goal_local = target_center_local.copy()
    goal_local[:2] = np.clip(target_center_local[:2], lower, upper)
    goal_center = region_origin + region_rotation @ goal_local
    direction = goal_center[:2] - target_center[:2]
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-9:
        raise ContactKeyframeGeometryError("target already occupies the planned interior footprint")
    direction /= distance
    patch_offset = patch[:, :2].mean(axis=0)
    trailing = float(((points_world[:, :2] - target_center[:2]) @ direction).min())
    leading = float(((patch[:, :2] - patch_offset) @ direction).max())
    engage = target_center[:2] + direction * (trailing - leading)
    start = engage - direction * profile.approach_standoff_m
    end = goal_center[:2] + direction * (trailing - leading)
    z = _contact_engagement_tcp_z_m(request, eef_orientation_xyzw=eef_orientation_xyzw)
    if z is None or not np.isfinite(z):
        raise ContactKeyframeGeometryError("native side push has no finite common engagement height")
    hover = _hover_working_floor_z_m(request)
    hover = max(float(z) + profile.hover_height_m, float(z) if hover is None else hover)
    return NativeSidePushGeometry(
        target_id, region_id, tuple(target_center), tuple(goal_center), tuple(direction),
        tuple(patch_offset), tuple(start), tuple(engage), tuple(end), float(z), float(hover), trailing, leading,
    )


def canonicalize_native_side_push(
    request: MotionPlanRequest, keyframes: Sequence[RelativeKeyframeSpec], *,
    resolver: RelativePoseResolver | None = None,
) -> list[RelativeKeyframeSpec]:
    """Recompute an explicit SIDE_PUSH path from current observed geometry.

    No metadata authorizes an exemption. Even pre-marked input frames receive
    the same geometric construction and remain subject to normal IK/collisions.
    """
    if tuple(frame.keyframe_type for frame in keyframes) != _PHASES:
        raise ContactKeyframeGeometryError("SIDE_PUSH requires the existing five ordered sweep phases")
    resolver = resolver or RelativePoseResolver(request.world)
    poses = [resolver.resolve(frame) for frame in keyframes]
    attitude = poses[1].orientation_xyzw
    rotation = quaternion_matrix_xyzw(attitude)
    if any(not np.allclose(quaternion_matrix_xyzw(p.orientation_xyzw), rotation, atol=1e-7) for p in poses):
        raise ContactKeyframeGeometryError("SIDE_PUSH requires one fixed attitude for all five phases")
    geometry = native_side_push_geometry(request, eef_orientation_xyzw=attitude)
    patch_xy = [geometry.patch_start_xy_m, geometry.patch_start_xy_m,
                geometry.patch_engage_xy_m, geometry.patch_end_xy_m, geometry.patch_end_xy_m]
    result = []
    for frame, xy in zip(keyframes, patch_xy):
        z = geometry.hover_tcp_z_m if frame.keyframe_type in {KeyframeType.PRE_CONTACT, KeyframeType.RETREAT} else geometry.contact_tcp_z_m
        updated = _retarget_tcp_world_xy(
            frame, resolver=resolver,
            target_xy=np.asarray(xy) - np.asarray(geometry.working_patch_offset_xy_m),
            target_z=z, packing_xyzw=attitude, metadata_flag="native_side_push_geometry",
        )
        if updated is None:
            raise ContactKeyframeGeometryError("could not represent native side-push pose")
        result.append(updated)
    return result


def generate_native_side_push_keyframes(
    request: MotionPlanRequest, *, eef_orientation_xyzw: Sequence[float],
) -> list[RelativeKeyframeSpec]:
    """Create a standard five-phase template, then apply the same contract."""
    frames = [RelativeKeyframeSpec(
        keyframe_id=f"{request.task.subgoal_id}:native-side-push:{index}",
        keyframe_type=kind, frame_ref="world", anchor="origin",
        approach_axis_xyz=(0., 0., 1.), tool_axis_to_align="+z",
        offset_along_approach_m=1., planner="CARTESIAN",
        metadata={"packing_orientation_xyzw": list(eef_orientation_xyzw)},
    ) for index, kind in enumerate(_PHASES)]
    return canonicalize_native_side_push(request, frames)
