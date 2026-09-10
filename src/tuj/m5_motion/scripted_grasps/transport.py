"""Ground M4's implicit region transport/place without changing the physical grasp.

The emitted ``held_transport_goal`` / ``held_place_goal`` describe where the
*held object* must end up (object body origin, object orientation), not where
the gripper goes.  The keyframe generator tags the matching keyframes with
``pose_subject=ATTACHED_OBJECT`` so the compiler retargets the object pose to
an EEF pose through the measured grasp transform
(:func:`tuj.m5_motion.attachment_retarget.retarget_resolved_pose`).  Object
space keeps both the scripted live runtime (``retention``) and the generic
``run.py`` pipeline (attachment metadata on the WorldSnapshot) on one contract.

Transport ends above the region with rim clearance; place ends with the object
bbox a release clearance above the region's interior floor.  Both pick the
same free XY spot so several objects can share one tray without stacking.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from tuj.m5_motion.attachment_retarget import (
    ATTACHED_OBJECT_POSE_SUBJECT, attachment_transform, held_pose_subject)
from tuj.m5_motion.geometry import tool_rotation_from_axis
from tuj.m5_motion.schema import Pose
from tuj.m5_motion.task_semantics import normalize_action
from .frames import inverse, transform


HELD_TRANSPORT_GOAL_ANCHOR = 'held_transport_goal'
HELD_TRANSPORT_START_ANCHOR = 'held_transport_start'
HELD_PLACE_GOAL_ANCHOR = 'held_place_goal'
HELD_PLACE_START_ANCHOR = 'held_place_start'
# Rim/wall allowance when searching the region interior for a free spot and the
# fallback floor thickness when the region has no usable collision points.
REGION_WALL_ALLOWANCE_M = 0.02
REGION_FLOOR_FALLBACK_M = 0.005
FREE_SPOT_GRID_M = 0.01


def _action(task):
    return normalize_action(task.metadata.get('operation') or task.action_type)


def _is_transport(task):
    return _action(task) in {'TRANSPORT', 'MOVE'}


def _is_region_place(task):
    # RETURN_TOOL keeps its rack-dock semantics; only scene placements into a
    # region are grounded in object space here.
    action = _action(task)
    return action in {'PLACE', 'RELEASE'} or action.startswith('PLACE_')


def _implicit_region_goal(task, object_id):
    if not task.goal.target_region_id or object_id is None or task.target_ids != [object_id]:
        return False
    # Explicit task poses remain authoritative. Only replace M4's implicit
    # fallback to the carried object's current pose, which is not a destination.
    if task.metadata.get('action_parameters', {}).get('target_pose') is not None or task.grasp is not None:
        return False
    return True


def _held_body_from_world(request, object_id):
    """Current held-object body pose from the EEF pose and the grasp transform."""
    state = request.world.robot_state
    if state.eef_pose is None or state.eef_pose.frame_id != 'world':
        raise ValueError('TRANSPORT_EEF_POSE_REQUIRED')
    grasp = attachment_transform(request.world, object_id)
    T_WE = transform(state.eef_pose.position_m, quaternion_xyzw=state.eef_pose.orientation_xyzw)
    T_EB = transform(grasp.position_in_reference_m, quaternion_xyzw=grasp.orientation_in_reference_xyzw)
    record = request.world.objects.get(object_id) or {}
    if 'dimensions_m' not in record:
        raise ValueError('TRANSPORT_HELD_OBJECT_GEOMETRY_REQUIRED')
    center_in_body = np.asarray(record.get('anchors', {}).get('center', [0., 0., 0.]), dtype=float)
    return T_WE @ T_EB, center_in_body, np.asarray(record['dimensions_m'], dtype=float), T_WE


class _Grounding:
    """Shared region + held-object geometry for one implicit region goal."""

    def __init__(self, request, retention, object_id):
        task = request.task
        record = request.world.objects.get(task.goal.target_region_id)
        if not record or 'pose' not in record or 'dimensions_m' not in record:
            raise ValueError('TRANSPORT_REGION_GEOMETRY_REQUIRED')
        pose = record['pose']
        if pose.get('frame_id', 'world') != 'world':
            raise ValueError('TRANSPORT_REGION_WORLD_FRAME_REQUIRED')
        self.request, self.task, self.record, self.object_id = request, task, record, object_id
        self.T_WR = transform(pose['position_m'], rotation=Rotation.from_quat(pose['orientation_xyzw']).as_matrix())
        self.region_dims = np.asarray(record['dimensions_m'], dtype=float)
        self.region_center_local = np.asarray(record.get('anchors', {}).get('center', [0., 0., 0.]), dtype=float)
        self.region_world = (self.T_WR @ np.r_[self.region_center_local, 1.])[:3]
        self.region_half = np.abs(self.T_WR[:3, :3]) @ self.region_dims / 2.
        if retention is not None:
            c = retention.context
            self.T_WB = c.body_pose()
            self.center_in_body = np.asarray(c.center_in_body, dtype=float)
            self.local_size = np.asarray(c.local_size, dtype=float)
            self.T_WE = c.grip_pose()
            self.source = 'LIVE_GRASP_AND_DESTINATION_BBOX'
        else:
            self.T_WB, self.center_in_body, self.local_size, self.T_WE = _held_body_from_world(request, object_id)
            self.source = 'WORLD_GRASP_TRANSFORM_AND_DESTINATION_BBOX'
        self.center = (self.T_WB @ np.r_[self.center_in_body, 1.])[:3]
        self.half = np.abs(self.T_WB[:3, :3]) @ self.local_size / 2.

    def bottom_below_origin(self):
        """Depth of the object's lowest point under its body origin (world z).

        Uses the object's own collision vertices in the current (preserved)
        orientation when the record carries them, so a tilted or asymmetric
        object is released just above the floor instead of hovering by the
        conservative bbox half height.  Falls back to the rotated bbox bottom.
        """
        record = self.request.world.objects.get(self.object_id) or {}
        points = record.get('collision_points_m') if isinstance(record, dict) else None
        if points is not None and len(points) >= 4:
            P = np.asarray(points, dtype=float)
            return float(-(P @ self.T_WB[2, :3]).min())
        return float(self.half[2] - (self.T_WB[:3, :3] @ self.center_in_body)[2])

    # -- region interior -----------------------------------------------------
    def floor_top_world_z(self):
        """Top of the region's interior floor (world z).

        Uses the innermost collision vertices: the smallest central footprint
        fraction that still contains points isolates the floor slab from the
        walls, and its highest point at or below the body center is the floor
        top.  Falls back to bottom + a nominal slab thickness.
        """
        bottom_local = self.region_center_local[2] - self.region_dims[2] / 2.
        floor_local = bottom_local + REGION_FLOOR_FALLBACK_M
        points = self.record.get('collision_points_m')
        if points is not None and len(points) >= 8:
            P = np.asarray(points, dtype=float)
            rel = P - self.region_center_local
            for fraction in (.15, .2, .25, .3, .35, .4, .45):
                mask = (np.abs(rel[:, 0]) <= self.region_dims[0] * fraction) & \
                       (np.abs(rel[:, 1]) <= self.region_dims[1] * fraction)
                if mask.sum() >= 4:
                    lower = P[mask, 2][P[mask, 2] <= self.region_center_local[2] + 1e-9]
                    if lower.size:
                        floor_local = max(floor_local, float(lower.max()))
                    break
        return float((self.T_WR @ np.r_[self.region_center_local[0], self.region_center_local[1], floor_local, 1.])[2])

    def _occupants(self):
        """World XY footprints of scene objects already inside the region."""
        out = []
        region_bottom = self.region_world[2] - self.region_half[2]
        for other_id, other in self.request.world.objects.items():
            if other_id in {self.object_id, self.task.goal.target_region_id}:
                continue
            if not isinstance(other, dict) or 'pose' not in other or 'dimensions_m' not in other:
                continue
            pose = other['pose']
            if pose.get('frame_id', 'world') != 'world':
                continue
            p = np.asarray(pose['position_m'], dtype=float)
            R = Rotation.from_quat(pose['orientation_xyzw']).as_matrix()
            c = np.asarray(other.get('anchors', {}).get('center', [0., 0., 0.]), dtype=float)
            center = p + R @ c
            half = np.abs(R[:2, :]) @ np.asarray(other['dimensions_m'], dtype=float) / 2.
            inside = np.all(np.abs(center[:2] - self.region_world[:2]) <= self.region_half[:2])
            if inside and center[2] >= region_bottom - 1e-6:
                out.append((center[:2], half))
        return out

    def free_destination_xy(self):
        """Region-center XY, or the nearest interior spot clear of occupants."""
        margin = max(.01, float(self.request.constraints.collision_margin_m) * 2.)
        occupants = self._occupants()
        mine = self.half[:2]
        center_xy = self.region_world[:2]
        if not occupants:
            return center_xy

        def clearance(xy):
            # Minimum per-occupant gap: how far this footprint clears each
            # occupant along its least-separated axis (negative = overlap).
            gaps = [
                float(np.max(np.abs(xy - c) - (h + mine)))
                for c, h in occupants
            ]
            return min(gaps) if gaps else float("inf")

        def free(xy):
            return clearance(xy) >= margin

        if free(center_xy):
            return center_xy
        # Search the reachable interior (clamp the range to >= 0 per axis so an
        # object wider than the interior on one axis still sweeps the other).
        limit = np.maximum(
            self.region_half[:2] - REGION_WALL_ALLOWANCE_M - mine, 0.0
        )
        def _axis_offsets(extent: float) -> np.ndarray:
            values = np.arange(-extent, extent + 1e-9, FREE_SPOT_GRID_M)
            return values if values.size else np.array([0.0])

        xs = _axis_offsets(float(limit[0]))
        ys = _axis_offsets(float(limit[1]))
        cells = [
            center_xy + np.array([float(dx), float(dy)])
            for dx in xs
            for dy in ys
        ]
        # Prefer the nearest spot that meets the full margin; if none does
        # (the region is too crowded for this footprint), fall back to the
        # spot that maximises clearance from occupants — spreading objects to
        # opposite ends of the region — instead of stacking on the centre.
        clear_cells = [xy for xy in cells if free(xy)]
        if clear_cells:
            return min(
                clear_cells,
                key=lambda xy: float(np.sum((xy - center_xy) ** 2)),
            )
        return max(cells, key=clearance)

    # -- publication ---------------------------------------------------------
    def publish(self, goal_key, start_key, desired_center, extra):
        # Object-space destination: the body origin translated so the bbox center
        # lands on desired_center; the grasp orientation is preserved, except for
        # an optional 90 deg yaw about the region's vertical that puts the held
        # object's SHORT footprint side along the region's SHORT axis (long side
        # along the long axis). A narrow region (e.g. a tray) otherwise leaves no
        # margin when the object's long axis lands across it, so settle drift on
        # release overhangs the rim (target_fully_inside_region failure). The yaw
        # is vertical-only, so the object's height/floor geometry is unchanged.
        destination = self.T_WB.copy()
        region_rotation = self.T_WR[:3, :3]
        object_in_region = region_rotation.T @ destination[:3, :3]
        short_axis = int(np.argmin(self.region_dims[:2]))
        long_axis = 1 - short_axis
        extent_along_short = float(
            np.abs(object_in_region[short_axis, :]) @ self.local_size
        )
        extent_if_rotated = float(
            np.abs(object_in_region[long_axis, :]) @ self.local_size
        )
        if extent_if_rotated + 1e-6 < extent_along_short:
            yaw_about_region_z = np.array(
                [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
            )
            destination[:3, :3] = region_rotation @ (
                yaw_about_region_z @ object_in_region
            )
        # Re-anchor the (possibly rotated) body so its bbox center lands exactly
        # on desired_center.  Equivalent to the previous translate-only path when
        # no yaw is applied.
        destination[:3, 3] = desired_center - destination[:3, :3] @ self.center_in_body
        anchors = self.record.setdefault('anchors', {})
        anchors[goal_key] = (inverse(self.T_WR) @ destination)[:3, 3].tolist()
        anchors[start_key] = (inverse(self.T_WR) @ self.T_WB)[:3, 3].tolist()
        z, x = destination[:3, 2], destination[:3, 0]
        base = tool_rotation_from_axis(z, 0.)
        # The approach axis must point outward (up) so a positive
        # offset_along_approach_m lifts the object away from the region.  The
        # object's z axis is aligned with it either way: '+z' when the held
        # object's z already points up, '-z' when it points down.
        if float(z[2]) >= 0.:
            approach_world, tool_axis = z, '+z'
        else:
            approach_world, tool_axis = -z, '-z'
        self.task.metadata[goal_key] = {
            'frame_ref': 'object:' + self.task.goal.target_region_id, 'anchor': goal_key,
            'start_anchor': start_key,
            'approach_axis_xyz': (self.T_WR[:3, :3].T @ approach_world).tolist(),
            'tool_axis_to_align': tool_axis, 'roll_rad': math.atan2(float(x @ base[:, 1]), float(x @ base[:, 0])),
            'offset_along_approach_m': 0., 'preserve_grasp_orientation': True,
            'pose_subject': ATTACHED_OBJECT_POSE_SUBJECT,
            'object_orientation_xyzw': Rotation.from_matrix(destination[:3, :3]).as_quat().tolist(),
            'eef_orientation_xyzw': Rotation.from_matrix(self.T_WE[:3, :3]).as_quat().tolist(),
            'object_id': self.object_id, 'source': self.source, **extra}
        return destination


def _grounding_for(request, retention, predicate):
    task = request.task
    if retention is not None:
        object_id = retention.entry.object_id
        implicit = task.metadata.get('scripted_m4_implicit_object_pose', False)
    else:
        object_id = held_pose_subject(request)
        implicit = True
    if not implicit or not predicate(task) or not _implicit_region_goal(task, object_id):
        return None
    return _Grounding(request, retention, object_id)


def ground_held_transport(request, retention=None):
    """Carry the held object to a free spot above the region with rim clearance."""
    if request.task.metadata.get(HELD_TRANSPORT_GOAL_ANCHOR) is not None:
        return
    g = _grounding_for(request, retention, _is_transport)
    if g is None:
        return
    desired_center = g.region_world.copy()
    desired_center[:2] = g.free_destination_xy()
    # Keep the measured object bbox clear of the rim without an arbitrary 5 cm
    # standoff that can place a reachable kitchen destination outside UR5e reach.
    clearance = max(.02, request.constraints.collision_margin_m * 2.)
    desired_center[2] = max(g.center[2], g.region_world[2] + g.region_half[2] + g.half[2] + clearance)
    g.task.goal.target_pose = None
    g.publish(HELD_TRANSPORT_GOAL_ANCHOR, HELD_TRANSPORT_START_ANCHOR, desired_center, {})


def ground_held_place(request, retention=None):
    """Lower the held object onto the region's interior floor at a free spot.

    Publishes ``held_place_goal`` and replaces M4's implicit fallback
    ``goal.target_pose`` (the object's stale pre-grasp pose) with the grounded
    destination so the released collision body and later subgoals see the
    object where it was actually put down.
    """
    if request.task.metadata.get(HELD_PLACE_GOAL_ANCHOR) is not None:
        return
    g = _grounding_for(request, retention, _is_region_place)
    if g is None:
        return
    release_clearance = max(REGION_FLOOR_FALLBACK_M, float(request.constraints.collision_margin_m))
    desired_center = g.region_world.copy()
    desired_center[:2] = g.free_destination_xy()
    origin_z = g.floor_top_world_z() + release_clearance + g.bottom_below_origin()
    desired_center[2] = g.center[2] + (origin_z - g.T_WB[2, 3])
    destination = g.publish(
        HELD_PLACE_GOAL_ANCHOR, HELD_PLACE_START_ANCHOR, desired_center,
        {'release_clearance_m': release_clearance})
    g.task.goal.target_pose = Pose(
        frame_id='world',
        position_m=tuple(float(v) for v in destination[:3, 3]),
        orientation_xyzw=tuple(float(v) for v in Rotation.from_matrix(destination[:3, :3]).as_quat()))


def ground_held_region_goal(request, retention=None):
    """Ground whichever implicit region goal the task carries (transport or place)."""
    ground_held_transport(request, retention)
    ground_held_place(request, retention)
