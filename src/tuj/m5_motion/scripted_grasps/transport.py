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
bbox a release clearance above the region floor.  Both pick the same XY spot --
the slot M2 assigned inside the region when the plan divided it among several
objects, else the region centre -- and move off it when occupied footprints
block the seat.  Nested platforms already in the region (e.g. a plate on a
tray) are stacking-forbidden supports: a small contact tolerance is allowed,
but seats that rest on them are not preferred.
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
# Body-frame XY aspect for nested platforms vs thin utensil strips.  Broad
# occupants inside a destination (plate/mug on a tray) are stacking-forbidden
# supports: seat search may graze them within OCCUPANT_CONTACT_TOLERANCE_M but
# must not rank a seat that rests fully on them.  Thin strips stay hard blockers.
THIN_STRIP_MAX_SHORT_M = 0.06
THIN_STRIP_MAX_ASPECT = 0.4
OCCUPANT_CONTACT_TOLERANCE_M = 0.005
# Occupants whose tops agree to within this share the load of what is put
# on them; below it only the higher one is touched.
SUPPORT_LEVEL_TOLERANCE_M = 0.002
# Rejected place seats must move at least this far in XY before reuse.
REJECTED_PLACE_EPS_M = 0.005
# Geom-label tokens that identify moving robot / EE collision bodies (not scene
# objects). Matched case-insensitively as substrings of validator geom names.
_ROBOT_MOTION_LABEL_MARKERS = (
    "hand",
    "gripper",
    "finger",
    "palm",
    "wrist",
    "flange",
    "robot0_",
    "ur5e",
    "forearm",
    "wrist_3",
)
# VacuumGripper ``vac_cup`` cylinder half-height from
# ``scripts/assets/vacuum_gripper.xml`` (``size="0.03 0.012"``).  The seal
# face coincides with the grip TCP; held-place still reserves this axial
# cup size above the support so vac EE collision clears the floor the same
# way multi-finger place reserves ``finger_below`` — not only the object bbox.
VACUUM_CUP_HALF_HEIGHT_M = 0.012
# Radial extent of the same ``vac_cup`` collision cylinder.  Held-object slot
# clipping reserves this footprint too, because during PLACE the cup remains
# attached to an off-centre plate grasp and must clear the container rim itself.
VACUUM_CUP_RADIUS_M = 0.030


def _is_soft_contact_occupant(dimensions_m):
    """True for broad nested platforms; false for thin utensil strips.

    Soft-contact occupants forbid stacking but allow a small XY graze.  Thin
    strips remain hard blockers (no intentional contact budget).
    """
    xy = np.sort(np.asarray(dimensions_m[:2], dtype=float))
    short, long = float(xy[0]), float(xy[1])
    if short < THIN_STRIP_MAX_SHORT_M and short / max(long, 1e-9) < THIN_STRIP_MAX_ASPECT:
        return False
    return True


CONCEPTUAL_TOOL_REST_ID = 'tool_rest'
CONCEPTUAL_TOOL_HOME_GOAL = 'conceptual_tool_home_goal'


def _materialize_conceptual_tool_home(request, retention):
    """Expose the pre-grasp tool pose as geometry for conceptual ``tool_rest``.

    M2 deliberately represents a tool's return location with the semantic ID
    ``tool_rest``.  Scripted live snapshots track the held body's current pose,
    so unlike predicted planning worlds they cannot recover its original pose
    after acquisition.  The grasp context owns that measured pre-grasp pose;
    publish a non-colliding virtual floor around it for the ordinary region
    transport/place grounding path.
    """
    task = request.task
    region_id = task.goal.target_region_id
    if region_id != CONCEPTUAL_TOOL_REST_ID or retention is None:
        return
    existing = request.world.objects.get(region_id)
    if isinstance(existing, dict) and {'pose', 'dimensions_m'} <= set(existing):
        return
    entry = getattr(retention, 'entry', None)
    object_id = getattr(entry, 'scene_object_id', None) or getattr(entry, 'object_id', None)
    if object_id is None or task.target_ids != [object_id]:
        return
    if held_pose_subject(request) != object_id:
        raise ValueError('TRANSPORT_TOOL_HOME_RETENTION_MISMATCH')
    try:
        world_transform = attachment_transform(request.world, object_id)
        retained_transform = retention.reference_transform()
        same_reference = (
            retained_transform.object_id == world_transform.object_id
            and retained_transform.reference_kind == world_transform.reference_kind
            and retained_transform.reference_name == world_transform.reference_name
        )
        same_position = np.allclose(
            retained_transform.position_in_reference_m,
            world_transform.position_in_reference_m,
            atol=1e-6,
            rtol=0.,
        )
        same_orientation = np.allclose(
            Rotation.from_quat(
                retained_transform.orientation_in_reference_xyzw
            ).as_matrix(),
            Rotation.from_quat(
                world_transform.orientation_in_reference_xyzw
            ).as_matrix(),
            atol=1e-6,
            rtol=0.,
        )
    except (AttributeError, TypeError, ValueError):
        raise ValueError('TRANSPORT_TOOL_HOME_RETENTION_MISMATCH') from None
    if not (same_reference and same_position and same_orientation):
        raise ValueError('TRANSPORT_TOOL_HOME_RETENTION_MISMATCH')
    context = getattr(retention, 'context', None)
    initial_body = np.asarray(getattr(context, 'initial_body', None), dtype=float)
    center_in_body = np.asarray(getattr(context, 'center_in_body', None), dtype=float)
    local_size = np.asarray(getattr(context, 'local_size', None), dtype=float)
    initial_bottom = float(getattr(context, 'initial_bottom', float('nan')))
    support_top = float(getattr(context, 'support_top_z', float('nan')))
    if (
        initial_body.shape != (4, 4)
        or center_in_body.shape != (3,)
        or local_size.shape != (3,)
        or not np.all(np.isfinite(initial_body))
        or not np.all(np.isfinite(center_in_body))
        or not np.all(np.isfinite(local_size))
        or np.any(local_size <= 0.)
        or not math.isfinite(initial_bottom)
        or not math.isfinite(support_top)
    ):
        raise ValueError('TRANSPORT_TOOL_HOME_POSE_REQUIRED')
    initial_center = (initial_body @ np.r_[center_in_body, 1.])[:3]
    # Both heights are world-z metres measured by the grasp context. Contact
    # settling can put the initial object bottom slightly inside its support;
    # that penetration must not reduce the later collision clearance.
    home_floor = max(initial_bottom, support_top)
    half_xy = np.abs(initial_body[:2, :3]) @ local_size / 2.
    wall = max(
        REGION_WALL_ALLOWANCE_M,
        float(request.constraints.collision_margin_m) * 2.,
    )
    floor = REGION_FLOOR_FALLBACK_M
    request.world.objects[region_id] = {
        'pose': {
            'frame_id': 'world',
            'position_m': [
                float(initial_center[0]),
                float(initial_center[1]),
                home_floor - floor / 2.,
            ],
            'orientation_xyzw': [0., 0., 0., 1.],
        },
        'dimensions_m': [
            float(2. * (half_xy[0] + wall)),
            float(2. * (half_xy[1] + wall)),
            floor,
        ],
        'anchors': {'center': [0., 0., 0.]},
        'collision_enabled': False,
        'metadata': {
            'conceptual_tool_home': True,
            'object_id': object_id,
            'initial_bottom_world_z_m': initial_bottom,
            'support_top_world_z_m': support_top,
            'support_correction_m': home_floor - initial_bottom,
            'object_orientation_xyzw': Rotation.from_matrix(
                initial_body[:3, :3]
            ).as_quat().tolist(),
        },
    }
    task.metadata[CONCEPTUAL_TOOL_HOME_GOAL] = True


def _action(task):
    return normalize_action(task.metadata.get('operation') or task.action_type)


def _request_uses_vacuum(request):
    ee = str(request.task.ee or '').strip().lower()
    return ee in {'vac', 'vacuum'}


def _vacuum_cup_half_height_m():
    """Axial vac-cup collision half-size used for held-place EE clearance."""

    return float(VACUUM_CUP_HALF_HEIGHT_M)


def _vacuum_place_release_clearance_m(g, collision_margin_m, base_clearance):
    """Object↔support gap for vac place while the payload is still attached.

    Thin plates seat with planner margin alone. Thicker/irregular vac payloads
    (live bread_b on plate_b) still intersect the support during the attached
    PLACE settle keyframe, so absolute-joint tracking never reaches KF1_2
    (~0.036 m / 0.083 rad vs 0.005 m / 0.05 rad gates). Pad with the held
    object's vertical half-extent plus a small tilt term from the lateral
    footprint — geometry-driven, not object-id specific.
    """

    margin = float(collision_margin_m)
    vertical = float(np.asarray(g.half, dtype=float)[2])
    lateral = float(np.linalg.norm(np.asarray(g.half, dtype=float)[:2]))
    geometry_pad = vertical + 0.15 * lateral
    return max(float(base_clearance), margin, geometry_pad)


def _retargeted_tcp_world_z(grip_pose, body_pose, destination_body):
    """World-z of the grip TCP that realizes ``destination_body`` under T_GB."""

    t_gb = inverse(grip_pose) @ body_pose
    t_we = destination_body @ inverse(t_gb)
    return float(t_we[2, 3])


def _raise_place_for_vacuum_ee_clearance(
        g, destination, *, support_z, release_clearance, collision_margin_m):
    """Raise an object-space place pose until the vac TCP clears the support.

    Object seating (bbox above support by ``release_clearance``) is the floor;
    vacuum EE clearance is applied only as an additional lift when the
    measured grasp transform would put the cup/TCP inside the planner margin.
    """

    margin = float(collision_margin_m)
    tcp_z = _retargeted_tcp_world_z(g.T_WE, g.T_WB, destination)
    required_tcp_z = (
        float(support_z) + _vacuum_cup_half_height_m() + max(0., margin)
    )
    lift = required_tcp_z - tcp_z
    if lift <= 1e-12:
        return destination, float(release_clearance), 0.
    raised = destination.copy()
    raised[2, 3] = float(destination[2, 3]) + lift
    anchors = g.record.setdefault('anchors', {})
    anchors[HELD_PLACE_GOAL_ANCHOR] = (inverse(g.T_WR) @ raised)[:3, 3].tolist()
    hint = g.task.metadata.get(HELD_PLACE_GOAL_ANCHOR)
    if isinstance(hint, dict):
        hint['release_clearance_m'] = float(release_clearance) + lift
        hint['vacuum_ee_clearance_lift_m'] = float(lift)
        hint['vacuum_ee_cup_half_height_m'] = _vacuum_cup_half_height_m()
    return raised, float(release_clearance) + lift, float(lift)


def _is_transport(task):
    return _action(task) in {'TRANSPORT', 'MOVE'}


def _is_region_place(task):
    # RETURN_TOOL keeps its rack-dock semantics; only scene placements into a
    # region are grounded in object space here.
    action = _action(task)
    return (
        action in {'PLACE', 'RELEASE'}
        or action.startswith('PLACE_')
        or (
            action in {'RETURN_TOOL', 'TERMINAL_RETURN_TOOL'}
            and task.metadata.get(CONCEPTUAL_TOOL_HOME_GOAL) is True
        )
    )


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
        metadata = record.get('metadata') if isinstance(record, dict) else None
        home_orientation = (
            metadata.get('object_orientation_xyzw')
            if isinstance(metadata, dict) and metadata.get('conceptual_tool_home') is True
            else None
        )
        self.destination_rotation = (
            Rotation.from_quat(home_orientation).as_matrix()
            if isinstance(home_orientation, (list, tuple)) and len(home_orientation) == 4
            else self.T_WB[:3, :3].copy()
        )
        self.preserve_destination_rotation = home_orientation is not None
        self.half = np.abs(self.destination_rotation) @ self.local_size / 2.

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
            return float(-(P @ self.destination_rotation[2, :]).min())
        return float(
            self.half[2] - (self.destination_rotation @ self.center_in_body)[2]
        )

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
        """World XY footprints (and tops) of scene objects already in the region.

        Each entry is ``(center_xy, half_xy, top_z, soft_contact)``.
        ``soft_contact`` marks broad nested platforms (plate/mug on a tray):
        stacking onto them is forbidden, but seat search may graze within
        ``OCCUPANT_CONTACT_TOLERANCE_M``.  Thin strips are hard blockers.
        """
        cached = getattr(self, '_occupant_cache', None)
        if cached is not None:
            return cached
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
            dims = np.asarray(other['dimensions_m'], dtype=float)
            half = np.abs(R[:2, :]) @ dims / 2.
            # Destination regions often sit inside a larger container (plate on
            # tray).  The container center can fall inside the region AABB while
            # its footprint dwarfs the region, so treating it as a packed item
            # makes every interior seat look blocked and free-spot search then
            # stacks the held object onto utensils already on the plate
            # (live bread_a → plate_a with tray_a counted as an occupant).
            if np.any(half > self.region_half[:2] + 1e-9):
                continue
            inside = np.all(np.abs(center[:2] - self.region_world[:2]) <= self.region_half[:2])
            if inside and center[2] >= region_bottom - 1e-6:
                top = float(center[2] + (np.abs(R[2, :]) @ dims) / 2.)
                out.append((center[:2], half, top, _is_soft_contact_occupant(dims)))
        self._occupant_cache = out
        return out

    def _slot_xy(self):
        """World XY of the slot the plan assigned inside this region, if any.

        M2 divides a shared container among the objects that go into it and
        publishes each one's spot as a normalized offset of the interior half
        extent, so the upstream scene scale (point-cloud AABB) need not match
        the executed one.  Without it every object aims at the same region
        centre and lands on whatever was put there first.
        """
        params = self.task.metadata.get('action_parameters')
        slot = (params or {}).get('placement_slot') if isinstance(params, dict) else None
        if not isinstance(slot, dict):
            return None
        uv = slot.get('uv')
        if not (isinstance(uv, (list, tuple)) and len(uv) >= 2):
            return None
        # ``uv`` is expressed in the region frame.  Using the world-axis AABB
        # (``region_half``) here inflates a rotated region and then applies the
        # local offset along the wrong axes.  That placed a bread slot on the
        # rim of a rotated C3_2 plate even though the planned slot was interior.
        inner = np.maximum(
            self.region_dims[:2] * .5 - REGION_WALL_ALLOWANCE_M, 0.)
        object_in_region = self.T_WR[:3, :3].T @ self.destination_rotation
        placed_object_in_region = object_in_region
        object_half = (
            np.abs(object_in_region[:2, :]) @ self.local_size * .5)
        # ``publish`` may rotate the held body by 90 degrees to fit the
        # region's short axis.  Slot clipping happens before publish, so reserve
        # the footprint of that orientation as well; otherwise the later yaw
        # can turn a valid center into a rim overhang.
        short_axis = int(np.argmin(self.region_dims[:2]))
        long_axis = 1 - short_axis
        extent_along_short = float(
            np.abs(object_in_region[short_axis, :]) @ self.local_size)
        extent_if_rotated = float(
            np.abs(object_in_region[long_axis, :]) @ self.local_size)
        if (
            not self.preserve_destination_rotation
            and extent_if_rotated + 1e-6 < extent_along_short
        ):
            yaw_about_region_z = np.array(
                [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
            )
            rotated = yaw_about_region_z @ object_in_region
            placed_object_in_region = rotated
            object_half = np.maximum(
                object_half,
                np.abs(rotated[:2, :]) @ self.local_size * .5,
            )
        limit = np.maximum(inner - object_half, 0.)
        lower, upper = -limit, limit
        if _request_uses_vacuum(self.request):
            # T_BE is the measured body→TCP transform.  Preserve the actual
            # off-centre suction station when determining whether the cup, not
            # only the payload bbox, fits inside the destination region.
            t_be = inverse(self.T_WB) @ self.T_WE
            # Slot / free-spot XY is the object bbox center, while T_BE is
            # expressed from the body origin.  Convert the cup into the center
            # frame before clipping or an off-centre vac grasp undershoots the
            # rim pad by ``center_in_body``.
            cup_from_center = (
                np.asarray(t_be[:3, 3], dtype=float)
                - np.asarray(self.center_in_body, dtype=float)
            )
            cup_offset = (placed_object_in_region @ cup_from_center)[:2]
            # Tray collision meshes (e.g. rim geoms) sit inset from the AABB that
            # backs ``region_dims``. wall_allowance alone can leave vac_cup
            # inside the planner margin of those meshes (live plate_a: ~0.08 mm
            # short of 5 mm). Reserve 2x collision margin beyond the cup disk.
            mesh_pad = max(
                2.0 * float(self.request.constraints.collision_margin_m), 0.01)
            # Extra millimetre covers publish yaw / center-vs-origin numerics so
            # the final TCP disk stays inside the padded limit, not on it.
            cup_limit = np.maximum(
                inner - VACUUM_CUP_RADIUS_M - mesh_pad - 0.001, 0.)
            lower = np.maximum(lower, -cup_limit - cup_offset)
            upper = np.minimum(upper, cup_limit - cup_offset)
        requested = np.clip(
            np.asarray(uv[:2], dtype=float), -1., 1.) * inner
        if np.any(lower > upper):
            # The region cannot contain both footprints; retain the object-safe
            # interval and let ordinary collision validation report infeasibility.
            lower, upper = -limit, limit
        local_xy = self.region_center_local[:2] + np.clip(
            requested, lower, upper)
        local = self.region_center_local.copy()
        local[:2] = local_xy
        return (self.T_WR @ np.r_[local, 1.])[:2]

    def free_destination_xy(self):
        """The plan's slot when it is clear enough, else the least-overlapping seat.

        Nested platforms already inside the region are stacking-forbidden: a
        small contact tolerance is allowed, but the search never prefers resting
        fully on them.  Thin occupants remain hard blockers.  Returning the
        anchor whenever it was taken put every object at the same place
        (c3_1: mug into the plate).
        """
        occupants = self._occupants()
        mine = self.half[:2]
        margin = max(.01, float(self.request.constraints.collision_margin_m) * 2.)

        def clearance(xy):
            """Smallest per-object AABB separation; >= 0 means no contact."""
            if not occupants:
                return float('inf')
            return min(float(np.max(np.abs(xy - c) - (h + mine + margin)))
                       for c, h, _t, _soft in occupants)

        def acceptable(xy):
            """Clear enough: hard blockers need gap>=0; soft platforms allow a graze."""
            if not occupants:
                return True
            for c, h, _t, soft in occupants:
                gap = float(np.max(np.abs(xy - c) - (h + mine + margin)))
                limit = -OCCUPANT_CONTACT_TOLERANCE_M if soft else 0.
                if gap < limit:
                    return False
            return True

        anchor = self._slot_xy()
        if anchor is None:
            anchor = self.region_world[:2]
        if acceptable(anchor):
            return anchor
        limit = self.region_half[:2] - REGION_WALL_ALLOWANCE_M - mine
        if np.any(limit <= 0.):
            return anchor
        center_xy = self.region_world[:2]
        xs = np.arange(-limit[0], limit[0] + 1e-9, FREE_SPOT_GRID_M)
        ys = np.arange(-limit[1], limit[1] + 1e-9, FREE_SPOT_GRID_M)
        best, best_key = anchor, None
        for dx, dy in sorted(((dx, dy) for dx in xs for dy in ys),
                             key=lambda v: (v[0] - (anchor - center_xy)[0]) ** 2
                             + (v[1] - (anchor - center_xy)[1]) ** 2):
            xy = center_xy + np.array([dx, dy])
            if acceptable(xy):
                return xy
            # No seat within contact policy: minimize penetration (max gap).
            # Do not rank by resting on a stacking-forbidden platform.
            key = (clearance(xy), -float(np.sum((xy - anchor) ** 2)))
            if best_key is None or key > best_key:
                best, best_key = xy, key
        return best

    def place_xy_away_from(
        self,
        partner_xy,
        *,
        rejected_xys=(),
        min_shift_m=0.0,
        current_xy=None,
        toward_partner=False,
    ):
        """Next in-region place XY farther from ``partner_xy`` than rejected seats.

        Used by collision-feedback repair when an EE/hand margin violation is
        measured against a region occupant: keep the object AABB clear, stay
        inside the interior, and do not reuse prior place seats.

        When ``toward_partner`` is set (destination region itself is the
        collision partner), move toward the region center instead of away from
        it — fleeing the center drives the cup further into the rim.
        """
        partner = np.asarray(partner_xy, dtype=float).reshape(2)
        rejected = [
            np.asarray(item, dtype=float).reshape(2)
            for item in rejected_xys
            if item is not None
        ]
        margin = max(.01, float(self.request.constraints.collision_margin_m) * 2.)
        mine = self.half[:2]
        min_shift = max(0.0, float(min_shift_m))
        anchor = (
            np.asarray(current_xy, dtype=float).reshape(2)
            if current_xy is not None
            else self.free_destination_xy()
        )

        def occupants_acceptable(xy):
            occupants = self._occupants()
            if not occupants:
                return True
            for c, h, _t, soft in occupants:
                gap = float(np.max(np.abs(xy - c) - (h + mine + margin)))
                limit = -OCCUPANT_CONTACT_TOLERANCE_M if soft else 0.
                if gap < limit:
                    return False
            return True

        def usable(xy):
            limit = self.region_half[:2] - REGION_WALL_ALLOWANCE_M - mine
            if np.any(limit <= 0.):
                return False
            if np.any(np.abs(xy - self.region_world[:2]) > limit + 1e-9):
                return False
            if not occupants_acceptable(xy):
                return False
            for prior in rejected:
                if float(np.linalg.norm(xy - prior)) < REJECTED_PLACE_EPS_M:
                    return False
            anchor_dist = float(np.linalg.norm(anchor - partner))
            cand_dist = float(np.linalg.norm(xy - partner))
            if toward_partner:
                # Must move at least min_shift closer to the region center.
                if anchor_dist - cand_dist + 1e-9 < min_shift:
                    return False
            elif cand_dist + 1e-9 < anchor_dist + min_shift:
                return False
            return True

        if toward_partner:
            away = partner - anchor
        else:
            away = anchor - partner
        away_norm = float(np.linalg.norm(away))
        if away_norm < 1e-9:
            away = self.region_world[:2] - partner
            away_norm = float(np.linalg.norm(away))
        if away_norm < 1e-9:
            away = np.array([1.0, 0.0])
            away_norm = 1.0
        away = away / away_norm
        # Prefer flee rays first so a small margin deficit yields a distinct seat.
        radii = []
        step = max(FREE_SPOT_GRID_M, min_shift if min_shift > 0 else FREE_SPOT_GRID_M)
        reach = float(np.linalg.norm(self.region_half[:2])) + step
        r = max(step, min_shift)
        while r <= reach + 1e-9:
            radii.append(r)
            r += step
        angles = (0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.57, -1.57)
        for radius in radii:
            for angle in angles:
                c, s = math.cos(angle), math.sin(angle)
                direction = np.array(
                    [away[0] * c - away[1] * s, away[0] * s + away[1] * c]
                )
                xy = anchor + direction * radius
                if usable(xy):
                    return xy
        # Fall back to a full interior grid ranked by distance from the partner.
        limit = self.region_half[:2] - REGION_WALL_ALLOWANCE_M - mine
        if np.any(limit <= 0.):
            return None
        center_xy = self.region_world[:2]
        xs = np.arange(-limit[0], limit[0] + 1e-9, FREE_SPOT_GRID_M)
        ys = np.arange(-limit[1], limit[1] + 1e-9, FREE_SPOT_GRID_M)
        if toward_partner:
            ranked = sorted(
                (center_xy + np.array([dx, dy]) for dx in xs for dy in ys),
                key=lambda xy: (
                    float(np.linalg.norm(xy - partner)),
                    float(np.linalg.norm(xy - anchor)),
                ),
            )
        else:
            ranked = sorted(
                (center_xy + np.array([dx, dy]) for dx in xs for dy in ys),
                key=lambda xy: (
                    -float(np.linalg.norm(xy - partner)),
                    float(np.linalg.norm(xy - anchor)),
                ),
            )
        for xy in ranked:
            if usable(xy):
                return xy
        return None

    def _supported_fraction(self, xy):
        """Deprecated stacking score; region occupants are never stacking supports.

        Kept for call sites/tests that probe overlap geometry.  Always returns
        0 because nested platforms are stacking-forbidden (tray seat search may
        graze them, but must not rest on them).
        """
        del xy
        return 0.

    def interior_top_world_z(self):
        """Highest point of anything already inside the region (rim if empty).

        What the carried object has to fly over is not the rim but whatever is
        stacked in there.  Clearing the rim alone drove the bread through the
        mug standing on the plate (c3_1).
        """
        rim = self.region_world[2] + self.region_half[2]
        return max([rim, *(t for _c, _h, t, _soft in self._occupants())])

    def support_top_world_z(self, xy):
        """Region-floor support height; occupants are never stacking surfaces.

        Nested platforms inside the destination remain XY exclusion (with a
        soft-contact budget).  Releasing on their tops would stack the held
        object onto a plate that already occupies a tray seat.
        """
        del xy
        return self.floor_top_world_z(), False

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
        destination[:3, :3] = self.destination_rotation
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
        if (
            not self.preserve_destination_rotation
            and extent_if_rotated + 1e-6 < extent_along_short
        ):
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
        object_id = retention.entry.scene_object_id
        implicit = task.metadata.get('scripted_m4_implicit_object_pose', False)
    else:
        object_id = held_pose_subject(request)
        implicit = True
    if not implicit or not predicate(task) or not _implicit_region_goal(task, object_id):
        return None
    return _Grounding(request, retention, object_id)


def ground_held_transport(request, retention=None):
    """Carry the held object to a free spot above the region with rim clearance."""
    _materialize_conceptual_tool_home(request, retention)
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
    desired_center[2] = max(g.center[2],
                            g.interior_top_world_z() + g.half[2] + clearance)
    g.task.goal.target_pose = None
    g.publish(HELD_TRANSPORT_GOAL_ANCHOR, HELD_TRANSPORT_START_ANCHOR, desired_center, {})


def ground_held_place(request, retention=None):
    """Lower the held object onto the region's interior floor at a free spot.

    Publishes ``held_place_goal`` and replaces M4's implicit fallback
    ``goal.target_pose`` (the object's stale pre-grasp pose) with the grounded
    destination so the released collision body and later subgoals see the
    object where it was actually put down.
    """
    _materialize_conceptual_tool_home(request, retention)
    if request.task.metadata.get(HELD_PLACE_GOAL_ANCHOR) is not None:
        return
    g = _grounding_for(request, retention, _is_region_place)
    if g is None:
        return
    _seat_held_place_at(g, g.free_destination_xy())


def _seat_held_place_at(g, desired_xy):
    """Publish held_place_goal / target_pose for bbox center ``desired_xy``."""
    request = g.request
    release_clearance = max(REGION_FLOOR_FALLBACK_M, float(request.constraints.collision_margin_m))
    # Multi-finger fingertips reach below the grip TCP into the region floor.
    # Seat the held object high enough that the retargeted TCP clears that
    # immersion plus the planner margin (same constant as acquire enclosure).
    from tuj.m5_motion.grasp_geometry import (
        _request_uses_multi_finger,
        multi_finger_finger_below_tcp_m,
    )
    if _request_uses_multi_finger(request):
        release_clearance = max(
            release_clearance,
            float(multi_finger_finger_below_tcp_m())
            + float(request.constraints.collision_margin_m),
        )
    if _request_uses_vacuum(request):
        release_clearance = _vacuum_place_release_clearance_m(
            g, request.constraints.collision_margin_m, release_clearance)
    if g.task.metadata.get(CONCEPTUAL_TOOL_HOME_GOAL) is True:
        # A nominal pose exactly on the collision boundary can discard the
        # continuous IK branch for sub-millimetre solver error. Reserve the
        # declared positional accuracy above the unchanged collision margin.
        release_clearance = max(
            release_clearance,
            float(request.constraints.collision_margin_m)
            + float(request.constraints.position_tolerance_m),
        )
    desired_center = g.region_world.copy()
    desired_center[:2] = np.asarray(desired_xy, dtype=float).reshape(2)
    support_z, stacked = g.support_top_world_z(desired_center[:2])
    if stacked:
        # Releasing exactly at the required margin above another object leaves
        # the state validity check on the boundary; double it so the stacked
        # release keyframe is inside the margin, not on it.
        release_clearance = max(release_clearance,
                                float(request.constraints.collision_margin_m) * 2.)
    origin_z = support_z + release_clearance + g.bottom_below_origin()
    desired_center[2] = g.center[2] + (origin_z - g.T_WB[2, 3])
    destination = g.publish(
        HELD_PLACE_GOAL_ANCHOR, HELD_PLACE_START_ANCHOR, desired_center,
        {
            'release_clearance_m': release_clearance,
            'destination_center_xy_m': [
                float(desired_center[0]),
                float(desired_center[1]),
            ],
        })
    if _request_uses_vacuum(request):
        destination, release_clearance, _lift = _raise_place_for_vacuum_ee_clearance(
            g, destination,
            support_z=support_z,
            release_clearance=release_clearance,
            collision_margin_m=request.constraints.collision_margin_m,
        )
        hint = g.task.metadata.get(HELD_PLACE_GOAL_ANCHOR)
        if isinstance(hint, dict):
            hint['destination_center_xy_m'] = [
                float(desired_center[0]),
                float(desired_center[1]),
            ]
            hint['release_clearance_m'] = float(release_clearance)
    g.task.goal.target_pose = Pose(
        frame_id='world',
        position_m=tuple(float(v) for v in destination[:3, 3]),
        orientation_xyzw=tuple(float(v) for v in Rotation.from_matrix(destination[:3, :3]).as_quat()))
    return desired_center[:2]


def _label_matches_object_id(label: str, object_id: str) -> bool:
    return bool(
        label == object_id
        or label.startswith(f"{object_id}_")
        or label.startswith(f"{object_id}.")
    )


def _is_robot_motion_collision_label(label: str, request) -> bool:
    lower = str(label).lower()
    if any(marker in lower for marker in _ROBOT_MOTION_LABEL_MARKERS):
        return True
    ee = request.task.ee or request.world.metadata.get("physical_active_ee")
    if isinstance(ee, str) and ee:
        return label == ee or label.startswith(f"{ee}_") or label.startswith(f"{ee}.")
    return False


def _object_id_for_collision_label(label: str, request) -> str | None:
    best = None
    for object_id in request.world.objects:
        if not _label_matches_object_id(label, object_id):
            continue
        if best is None or len(object_id) > len(best):
            best = object_id
    return best


def _held_place_center_xy(request) -> np.ndarray | None:
    hint = request.task.metadata.get(HELD_PLACE_GOAL_ANCHOR)
    if isinstance(hint, dict):
        raw = hint.get("destination_center_xy_m")
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            return np.asarray(raw[:2], dtype=float)
    pose = request.task.goal.target_pose
    if pose is not None and pose.frame_id == "world":
        return np.asarray(pose.position_m[:2], dtype=float)
    return None


def _clear_held_place_grounding(request) -> None:
    request.task.metadata.pop(HELD_PLACE_GOAL_ANCHOR, None)
    request.task.metadata.pop(HELD_PLACE_START_ANCHOR, None)
    region_id = request.task.goal.target_region_id
    record = request.world.objects.get(region_id) if region_id else None
    if isinstance(record, dict):
        anchors = record.get("anchors")
        if isinstance(anchors, dict):
            anchors.pop(HELD_PLACE_GOAL_ANCHOR, None)
            anchors.pop(HELD_PLACE_START_ANCHOR, None)


def _place_margin_partner_from_feedback(request, feedback):
    """Pick the scene partner of an EE/hand margin violation, if any."""
    if not isinstance(feedback, dict):
        return None
    strategies = feedback.get("failed_strategies")
    if not isinstance(strategies, list):
        return None
    held_id = held_pose_subject(request)
    region_id = request.task.goal.target_region_id
    best = None
    for strategy in strategies:
        if not isinstance(strategy, dict):
            continue
        observations = strategy.get("collision_observations")
        if not isinstance(observations, list):
            continue
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            try:
                measured = float(observation.get("measured_clearance_m"))
                required = float(observation.get("required_clearance_m"))
            except (TypeError, ValueError):
                continue
            if (
                not math.isfinite(measured)
                or not math.isfinite(required)
                or required < 0.0
                or measured >= required
            ):
                continue
            labels = (
                str(observation.get("geometry_a", "")),
                str(observation.get("geometry_b", "")),
            )
            robot_labels = [
                label for label in labels
                if _is_robot_motion_collision_label(label, request)
            ]
            if not robot_labels:
                continue
            scene_labels = [label for label in labels if label not in robot_labels]
            partner_id = None
            for label in scene_labels:
                partner_id = _object_id_for_collision_label(label, request)
                if partner_id is not None:
                    break
            if partner_id is None or partner_id == held_id:
                continue
            record = request.world.objects.get(partner_id)
            if not isinstance(record, dict) or "pose" not in record:
                continue
            pose = record["pose"]
            if pose.get("frame_id", "world") != "world":
                continue
            center = np.asarray(pose["position_m"], dtype=float)[:2].copy()
            anchor = record.get("anchors", {}).get("center") if isinstance(
                record.get("anchors"), dict
            ) else None
            if anchor is not None:
                rotation = Rotation.from_quat(pose["orientation_xyzw"]).as_matrix()
                center = center + (rotation @ np.asarray(anchor, dtype=float))[:2]
            deficit = max(0.0, required - measured)
            # Prefer the worst deficit; break ties toward region occupants.
            rank = (deficit, 1.0 if partner_id == region_id else 0.0)
            if best is None or rank > best[0]:
                best = (rank, center, deficit, partner_id)
    if best is None:
        return None
    _rank, partner_xy, deficit, partner_id = best
    return {
        "partner_xy": partner_xy,
        "deficit_m": float(deficit),
        "partner_id": partner_id,
    }


def reground_held_place_from_collision_feedback(request, feedback) -> bool:
    """Rewrite ``held_place_goal`` away from a colliding partner for repair.

    Returns True when a distinct in-region place seat was published. Real
    penetration still fails later validation; this only diversifies the PLACE
    XY that collision-feedback retries would otherwise freeze.
    """
    if not _is_region_place(request.task):
        return False
    partner = _place_margin_partner_from_feedback(request, feedback)
    if partner is None:
        return False
    g = _grounding_for(request, None, _is_region_place)
    if g is None:
        return False
    rejected = []
    raw_rejected = feedback.get("rejected_place_xy_m") if isinstance(feedback, dict) else None
    if isinstance(raw_rejected, list):
        for item in raw_rejected:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                rejected.append([float(item[0]), float(item[1])])
    current = _held_place_center_xy(request)
    if current is not None:
        rejected.append([float(current[0]), float(current[1])])
    min_shift = float(partner["deficit_m"]) + float(
        request.constraints.collision_margin_m
    )
    # Destination-region partners are rim/floor geoms: move toward the region
    # center. Fleeing the region center pushes an off-centre vac cup deeper
    # into the wall that already failed margin.
    toward_region = (
        partner["partner_id"] == request.task.goal.target_region_id
    )
    next_xy = g.place_xy_away_from(
        partner["partner_xy"],
        rejected_xys=rejected,
        min_shift_m=min_shift,
        current_xy=current,
        toward_partner=toward_region,
    )
    if next_xy is None:
        return False
    _clear_held_place_grounding(request)
    seated = _seat_held_place_at(g, next_xy)
    if isinstance(feedback, dict):
        feedback["rejected_place_xy_m"] = rejected
        feedback["reground_place_partner_id"] = partner["partner_id"]
        feedback["reground_place_xy_m"] = [float(seated[0]), float(seated[1])]
    return True


def ground_held_region_goal(request, retention=None):
    """Ground whichever implicit region goal the task carries (transport or place)."""
    ground_held_transport(request, retention)
    ground_held_place(request, retention)
