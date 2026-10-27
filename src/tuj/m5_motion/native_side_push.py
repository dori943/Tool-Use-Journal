"""Observed-geometry side pushes for an explicit single-target contact spec.

This module generates motion waypoints only. It never edits object state,
attachment physics, feasibility verdicts, or the task's goal evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from scipy.spatial import ConvexHull, QhullError

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
    approach_direction_xy: tuple[float, float]
    translation_kind: str
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


def _face_normal_translation(
    points_world: np.ndarray, target_center: np.ndarray,
    region_rotation: np.ndarray, center_local: np.ndarray,
    lower: np.ndarray, upper: np.ndarray, goal_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Find the shortest face-normal ray that enters the target-fit region.

    A diagonal normal to a box corner has no central face contact. Select an
    actual observed footprint edge instead. If no normal ray reaches the
    target-fit region, return a face-normal approach for a subsequent goal
    translation; this path still requires full IK and collision validation.
    """
    try:
        hull = ConvexHull(points_world[:, :2])
    except QhullError as error:
        raise ContactKeyframeGeometryError("side push requires a nondegenerate observed footprint") from error
    polygon = points_world[hull.vertices, :2]
    candidates, facing = [], []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length <= 1e-12:
            continue
        # ConvexHull vertices are counterclockwise; left normal points inward.
        direction = np.array([-edge[1], edge[0]]) / length
        facing.append((float(direction @ goal_direction), -index, direction, (start + end) * .5))
        local_direction = region_rotation[:2, :2].T @ direction
        minimum, maximum = 0., float("inf")
        for axis in range(2):
            velocity = float(local_direction[axis])
            if abs(velocity) <= 1e-12:
                if center_local[axis] < lower[axis] - 1e-12 or center_local[axis] > upper[axis] + 1e-12:
                    maximum = -1.
                    break
                continue
            distances = ((lower[axis] - center_local[axis]) / velocity,
                         (upper[axis] - center_local[axis]) / velocity)
            minimum = max(minimum, min(distances))
            maximum = min(maximum, max(distances))
        if minimum > 1e-9 and minimum <= maximum + 1e-12:
            candidates.append((minimum, index, direction, (start + end) * .5))
    if not candidates:
        # The five-phase contract can approach the central face normally and
        # then translate toward the goal. This changes the contact path, never
        # the target state; the entire turning path must still pass validation.
        _, _, direction, face_center = max(facing, key=lambda candidate: candidate[:2])
        return direction, face_center, None
    distance, _, direction, face_center = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    return direction, face_center, float(distance)


def _convex_xy(points: np.ndarray) -> np.ndarray:
    try:
        return points[ConvexHull(points).vertices]
    except QhullError as error:
        raise ContactKeyframeGeometryError("side push requires finite nondegenerate obstacle footprints") from error


def _convex_polygon_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between closed convex XY footprints, including edges."""
    def cross(p, q):
        return float(p[0] * q[1] - p[1] * q[0])

    def inside(point, polygon):
        return all(cross(end-start, point-start) >= -1e-12
                   for start, end in zip(polygon, np.roll(polygon, -1, axis=0)))

    if inside(a[0], b) or inside(b[0], a):
        return 0.
    for start, end in zip(a, np.roll(a, -1, axis=0)):
        for other_start, other_end in zip(b, np.roll(b, -1, axis=0)):
            direction, other_direction = end-start, other_end-other_start
            denominator = cross(direction, other_direction)
            if abs(denominator) > 1e-12:
                delta = other_start-start
                if (-1e-12 <= cross(delta, other_direction)/denominator <= 1.+1e-12
                        and -1e-12 <= cross(delta, direction)/denominator <= 1.+1e-12):
                    return 0.

    def point_edge_distance(point, start, end):
        edge = end-start
        fraction = np.clip(np.dot(point-start, edge)/np.dot(edge, edge), 0., 1.)
        return float(np.linalg.norm(point-start-fraction*edge))

    return min(point_edge_distance(point, start, end)
               for first, second in ((a, b), (b, a)) for point in first
               for start, end in zip(second, np.roll(second, -1, axis=0)))


def _free_nominal_target_goal(
    request: MotionPlanRequest, *, target_center: np.ndarray,
    target_points: np.ndarray, original_goal: np.ndarray,
    region_origin: np.ndarray, region_rotation: np.ndarray,
    lower: np.ndarray, upper: np.ndarray,
) -> np.ndarray:
    """Keep a clear original path; otherwise try nine target-fit region anchors.

    The convex hull of the initial and translated footprints bounds the whole
    nominal rigid target translation, not just its final position. Obstacles
    are observed non-target scene objects in the target's vertical band. The
    region itself is the designated destination surface. This only filters
    nominal target occupancy: the ordinary compiler must still validate the
    entire robot and held-tool path using their full collision geometry.
    """
    margin = request.constraints.collision_margin_m
    excluded = {request.task.tool, *request.task.target_ids, request.task.goal.target_region_id}
    obstacles = []
    for object_id, record in request.world.objects.items():
        if object_id in excluded or record.get("collision_enabled") is False:
            continue
        try:
            _, points = _target_points(request, object_id)
        except (KeyError, TypeError, ValueError) as error:
            raise ContactKeyframeGeometryError(f"side push obstacle {object_id!r} has no usable observed geometry") from error
        # Objects above or below the entire target sweep do not occupy its
        # plane. Keep the unchanged requested clearance in this broad phase.
        if (points[:, 2].max() < target_points[:, 2].min()-margin
                or points[:, 2].min() > target_points[:, 2].max()+margin):
            continue
        obstacles.append(_convex_xy(points[:, :2]))

    def clear(goal):
        sweep = _convex_xy(np.vstack((target_points[:, :2],
            target_points[:, :2]+goal[:2]-target_center[:2])))
        return all(_convex_polygon_distance(sweep, obstacle) >= margin-1e-9
                   for obstacle in obstacles)

    if clear(original_goal):
        return original_goal
    local = region_rotation.T @ (target_center-region_origin)
    candidates = []
    for ix, x in enumerate((0., .5, 1.)):
        for iy, y in enumerate((0., .5, 1.)):
            candidate = local.copy()
            candidate[:2] = lower+(upper-lower)*np.array((x, y))
            goal = region_origin+region_rotation @ candidate
            # Rounded distance only breaks numerically equivalent ordering;
            # no coordinates, fit margins or collision distances are rounded.
            candidates.append((round(float(np.linalg.norm(goal[:2]-target_center[:2])), 12), ix*3+iy, goal))
    for _, _, goal in sorted(candidates, key=lambda item: item[:2]):
        if clear(goal):
            return goal
    raise ContactKeyframeGeometryError("side push has no free nominal target path among the original and nine region endpoints")


def native_side_push_geometry(
    request: MotionPlanRequest, *, eef_orientation_xyzw: Sequence[float],
    profile: PushPlanningProfile | None = None,
) -> NativeSidePushGeometry:
    """Align the actual leading patch feature with an observed face center.

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
    if np.all(target_center_local[:2] >= lower) and np.all(target_center_local[:2] <= upper):
        raise ContactKeyframeGeometryError("target already occupies the planned interior footprint")
    goal_local = target_center_local.copy()
    goal_local[:2] = np.clip(target_center_local[:2], lower, upper)
    goal_center = region_origin + region_rotation @ goal_local
    direction = goal_center[:2] - target_center[:2]
    direction /= np.linalg.norm(direction)
    approach_direction, face_center, distance = _face_normal_translation(
        points_world, target_center, region_rotation, target_center_local, lower, upper, direction)
    if distance is not None:
        direction = approach_direction
        goal_center = target_center.copy()
        goal_center[:2] += direction * distance
        translation_kind = "FACE_NORMAL"
    else:
        translation_kind = "FACE_NORMAL_APPROACH_THEN_GOAL_TRANSLATION"
    free_goal = _free_nominal_target_goal(
        request, target_center=target_center, target_points=points_world,
        original_goal=goal_center, region_origin=region_origin,
        region_rotation=region_rotation, lower=lower, upper=upper)
    if not np.allclose(free_goal, goal_center, atol=1e-12, rtol=0.):
        goal_center = free_goal
        direction = goal_center[:2]-target_center[:2]
        direction /= np.linalg.norm(direction)
        translation_kind = "FACE_NORMAL_APPROACH_THEN_FREE_REGION_TRANSLATION"
    projections = patch[:, :2] @ approach_direction
    # Center the actual leading feature, not the whole patch centroid. A
    # skewed patch can put its front vertex far to one side of that centroid.
    leading_feature = patch[np.isclose(projections, projections.max(), atol=1e-9, rtol=0.)]
    patch_offset = leading_feature[:, :2].mean(axis=0)
    trailing = float((face_center - target_center[:2]) @ approach_direction)
    leading = float(((patch[:, :2] - patch[:, :2].mean(axis=0)) @ approach_direction).max())
    engage = face_center
    start = engage - approach_direction * profile.approach_standoff_m
    end = engage + goal_center[:2] - target_center[:2]
    z = _contact_engagement_tcp_z_m(request, eef_orientation_xyzw=eef_orientation_xyzw)
    if z is None or not np.isfinite(z):
        raise ContactKeyframeGeometryError("native side push has no finite common engagement height")
    hover = _hover_working_floor_z_m(request)
    hover = max(float(z) + profile.hover_height_m, float(z) if hover is None else hover)
    return NativeSidePushGeometry(
        target_id, region_id, tuple(target_center), tuple(goal_center), tuple(direction),
        tuple(approach_direction), translation_kind, tuple(patch_offset), tuple(start), tuple(engage), tuple(end), float(z), float(hover), trailing, leading,
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
