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
# Extra place height accumulated when a repair measured the hand inside the
# robot mount deeper than the fixture tolerance. Not applied on the first seat.
_HELD_PLACE_MOUNT_RELEASE_LIFT_KEY = 'held_place_mount_release_lift_m'
HELD_PLACE_START_ANCHOR = 'held_place_start'
# Rim/wall allowance when searching the region interior for a free spot and the
# fallback floor thickness when the region has no usable collision points.
REGION_WALL_ALLOWANCE_M = 0.02
REGION_FLOOR_FALLBACK_M = 0.005
# Thickness of the band at the floor top whose vertices define the interior
# footprint.  Wide enough to catch a slab meshed a few millimetres apart,
# narrow enough to exclude the wall above it.
_FLOOR_BAND_M = 0.004
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
        self.retention = retention
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
        from tuj.m5_motion.ee_path_keepout import keepout_for_task
        self.ee_path_keepout = keepout_for_task(task)
        from .container_orientation import configure_packing_orientation
        configure_packing_orientation(self)

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
        floor_local = self.floor_top_local()
        return float((self.T_WR @ np.r_[self.region_center_local[0], self.region_center_local[1], floor_local, 1.])[2])

    def floor_top_local(self):
        """Floor top in region-local z; shared with interior_half_xy()."""
        cached = getattr(self, '_floor_local_cache', None)
        if cached is not None:
            return cached
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
                    # 0912: 중심 이하만 보는 이 필터는 용기의 벽을 걸러내려는
                    # 것인데, 속이 찬 물체 위에 얹는 place_on 에서는 물체의
                    # 한가운데를 지지면으로 돌려준다. c2_2 가 터키를 빵 위에
                    # 올리는 단계에서 EEF 를 z=0.945646 (빵 중심 0.94602, 윗면
                    # 0.95748) 에 두어 흡착컵이 빵을 7.8~9.6mm 파고들었고 전략
                    # 11개가 전부 기각됐다.
                    #
                    # 둘은 같은 점군으로 갈린다. 용기는 테두리가 중앙 발자국
                    # 바깥에 있어 중앙 점들이 중심 이하에서 끝나고, 속이 찬
                    # 물체는 중앙 점들이 꼭대기까지 올라간다. 중앙 최고점이
                    # 중심보다 위면 그 점이 곧 얹는 면이다.
                    central = P[mask, 2]
                    highest = float(central.max())
                    if highest > self.region_center_local[2] + 1e-9:
                        floor_local = highest
                    else:
                        lower = central[central <= self.region_center_local[2] + 1e-9]
                        if lower.size:
                            floor_local = max(floor_local, float(lower.max()))
                    break
        self._floor_local_cache = float(floor_local)
        return self._floor_local_cache

    def interior_half_xy(self):
        """Region interior half extent (XY), measured rather than assumed.

        ``dimensions_m`` is the region's bounding box, and a tray whose rim
        flares outward sets that box by the rim alone: blue_tray measures
        0.1966 m in y at the top band and about 0.172 m at every band below it.
        Subtracting a fixed wall allowance from the box therefore reports
        interior that does not exist (0.1766 m in y against a floor that ends
        at 0.1672 m), and a place aimed there rests the object on the wall --
        13.6 mm above its commanded pose, pushed sideways until one finger pad
        unloads and the grasp monitor stops the task (c3_1 apple).

        The floor slab is the honest interior: take the collision vertices at
        the floor top and use their extent.  Falls back to the bounding box
        when a region carries no collision geometry, so regions without points
        behave exactly as before.
        """
        cached = getattr(self, '_interior_half_cache', None)
        if cached is not None:
            return cached
        half = self.region_half[:2].copy()
        points = self.record.get('collision_points_m')
        if points is not None and len(points) >= 8:
            P = np.asarray(points, dtype=float)
            floor_local = self.floor_top_local()
            band = P[np.abs(P[:, 2] - floor_local) <= _FLOOR_BAND_M]
            if len(band) >= 4:
                measured = np.abs(band[:, :2] - self.region_center_local[:2]).max(axis=0)
                half = np.minimum(half, measured)
        self._interior_half_cache = half
        return half

    def interior_half_xy_local(self):
        """Measured floor half extents expressed in the region's own axes."""
        half = self.region_dims[:2] / 2.
        points = self.record.get('collision_points_m')
        if points is not None and len(points) >= 8:
            points = np.asarray(points, dtype=float)
            band = points[np.abs(points[:, 2] - self.floor_top_local()) <= _FLOOR_BAND_M]
            if len(band) >= 4:
                half = np.minimum(half,
                    np.abs(band[:, :2] - self.region_center_local[:2]).max(axis=0))
        return half

    def usable_half_xy(self):
        """Interior minus the wall allowance: where a footprint may be centred."""
        return np.maximum(self.interior_half_xy() - REGION_WALL_ALLOWANCE_M, 0.)

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
        # M2 packs observed world-axis AABBs and checks each offset against a
        # world-axis EE sweep. Its slots therefore retain those axes, including
        # older saved plans identified by the producer tag. Generic untagged
        # slots retain their historical region-local meaning.
        slot_frame = slot.get('frame') or (
            'world_xy' if slot.get('source') == 'm2_container_layout' else 'region_xy')
        if slot_frame not in {'world_xy', 'region_xy'}:
            raise ValueError(f'PLACEMENT_SLOT_FRAME_UNSUPPORTED: {slot_frame}')
        # Boundary/cup clipping always remains region-local. A world AABB is
        # only the normalization scale for world-axis slots, never a permission
        # to seat an object outside the rotated region's physical footprint.
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
        normalized = np.clip(np.asarray(uv[:2], dtype=float), -1., 1.)
        if slot_frame == 'world_xy':
            world_inner = np.maximum(self.region_half[:2] - REGION_WALL_ALLOWANCE_M, 0.)
            axes = self.T_WR[:2, :2]
            if abs(float(np.linalg.det(axes))) < 1e-9:
                raise ValueError('PLACEMENT_SLOT_FRAME_SINGULAR')
            requested = np.linalg.solve(axes, normalized * world_inner)
        else:
            requested = normalized * inner
        if np.any(lower > upper):
            # The region cannot contain both footprints; retain the object-safe
            # interval and let ordinary collision validation report infeasibility.
            lower, upper = -limit, limit
        local_xy = self.region_center_local[:2] + np.clip(
            requested, lower, upper)
        local = self.region_center_local.copy()
        local[:2] = local_xy
        return (self.T_WR @ np.r_[local, 1.])[:2]

    def ee_path_clearance(self, xy):
        """Gap between the seated object and the fixed EE-rack arm sweep.

        ``inf`` when no keep-out is bound.  The footprint is the larger half
        extent on both axes because ``publish`` may still yaw the object 90
        degrees after the seat is chosen.
        """
        field = self.ee_path_keepout
        if field is None:
            return math.inf
        from tuj.common.ee_keepout import placement_margin_m
        r = float(np.max(self.half[:2]))
        top = self.floor_top_world_z() + 2. * float(self.half[2])
        margin = placement_margin_m(self.request.constraints.collision_margin_m)
        return field.clearance(xy, (r, r), top, margin)

    def _initialize_seat_yaw(self):
        if (getattr(self, '_seat_yaw_initialized', False)
                or self.record.get('packing_metadata', {}).get('kind') == 'CONTAINER'):
            return
        self._seat_yaw_initialized = True
        # Choose the ordinary tray-aligned yaw BEFORE computing seat bounds.
        # Publishing a different yaw after search invalidates its occupancy
        # checks (a horizontal spoon becomes a blocked vertical spoon).
        region_rotation = self.T_WR[:3, :3]
        object_in_region = region_rotation.T @ self.destination_rotation
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
            self.destination_rotation = region_rotation @ (
                yaw_about_region_z @ object_in_region
            )
        self.half = np.abs(self.destination_rotation) @ self.local_size / 2.

    def free_destination_xy(self):
        """Search the same upright yaw alternatives used by goal publication.

        A preferred long-axis alignment can leave no seat in an occupied tray.
        Try its orthogonal yaw with the SAME footprint, wall, occupant, and
        keepout checks. Explicit tool-home and packing orientations stay fixed.
        Every resulting pose still goes through ordinary IK/collision planning.
        """
        self._initialize_seat_yaw()
        try:
            return self._free_destination_fixed_orientation()
        except ValueError as error:
            if (not str(error).startswith('PLACEMENT_SPACE_EXHAUSTED')
                    or self.preserve_destination_rotation
                    or self.record.get('packing_metadata', {}).get('kind') == 'CONTAINER'):
                raise
            original_error = error
        rotation, half = self.destination_rotation.copy(), self.half.copy()
        region = self.T_WR[:3, :3]
        yaw = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.destination_rotation = region @ yaw @ region.T @ rotation
        self.half = np.abs(self.destination_rotation) @ self.local_size / 2.
        self.preserve_destination_rotation = True
        try:
            xy = self._free_destination_fixed_orientation()
        except ValueError:
            self.destination_rotation, self.half = rotation, half
            self.preserve_destination_rotation = False
            raise original_error
        self.task.metadata['region_yaw_search'] = {
            'quarter_turns_from_preferred': 1,
            'reason': 'preferred orientation has no unoccupied seat',
            'clearance_policy_changed': False,
        }
        return xy

    def _free_destination_fixed_orientation(self):
        """Seat search that also keeps clear of the fixed EE-rack arm sweep.

        A seat under the stored attach/return trajectories is accepted here
        and fails only at the next EE exchange, after the rest of the plan has
        run (c3_2 mug_a).  Seats outside the sweep are preferred; when the
        region has none, the unconstrained choice is kept and recorded so the
        conflict is visible before the exchange rejects it.
        """
        def outside_sweep(xy):
            return self.ee_path_clearance(xy) >= 0.

        if self.ee_path_keepout is None:
            xy = self._free_destination_xy(None)
            if xy is None:
                raise ValueError('PLACEMENT_SPACE_EXHAUSTED: no unoccupied seat')
            return xy
        xy = self._free_destination_xy(outside_sweep)
        unavoidable = xy is None
        if unavoidable:
            xy = self._free_destination_xy(None)
        if xy is None:
            raise ValueError('PLACEMENT_SPACE_EXHAUSTED: no unoccupied seat')
        note = {'clearance_m': round(float(self.ee_path_clearance(xy)), 4)}
        if unavoidable:
            note['unavoidable'] = True
        anchor = self._slot_xy()
        if anchor is not None and not outside_sweep(anchor):
            note['slot_xy_in_sweep'] = [round(float(v), 4) for v in anchor]
        self.task.metadata['ee_path_keepout_seat'] = note
        return xy

    def seat_clear_of_occupants(self, xy, *, ignore_occupants=False):
        if ignore_occupants:
            return True
        margin = max(.01, float(self.request.constraints.collision_margin_m) * 2.)
        for center, half, _top, soft in self._occupants():
            gap = float(np.max(np.abs(xy - center) - (half + self.half[:2] + margin)))
            limit = -OCCUPANT_CONTACT_TOLERANCE_M if soft else 0.
            if gap < limit:
                return False
        return True

    def _m2_layout_slot(self):
        params = self.task.metadata.get('action_parameters')
        slot = (params or {}).get('placement_slot') if isinstance(params, dict) else None
        if isinstance(slot, dict) and slot.get('source') == 'm2_container_layout':
            return slot
        return None

    def _free_destination_xy(self, seat_allowed):
        """Find an unoccupied seat, or return None. Never accept overlap as fallback."""
        # The measured floor is a physical boundary; the extra 20 mm inset
        # in usable_half_xy() is only the existing conservative search domain.
        # Applying that inset to preferred slots would reject valid broad
        # plates even when their measured rim clearance exceeds the request's
        # collision margin. Conversely, outer-bbox slot clipping alone can
        # overhang a flared tray's narrower floor.
        support = self.task.metadata.get('support_placement') or {}
        on_support = (_action(self.task) == 'PLACE_ON' or (
            support.get('relation') == 'ON'
            and support.get('source') == 'SELECTED_PLACE_ON'
            and support.get('object_id') == self.object_id
            and support.get('region_id') == self.task.goal.target_region_id))
        # ON permits footprint overhang (e.g. cheese on a narrower sandwich
        # layer). Keep its centre above the explicit support; full-footprint
        # containment is reserved for placements IN a region. Collision and
        # measured post-execution support checks remain unchanged.
        center_xy = self.region_world[:2]
        # Floor measurements, footprint extents and seat offsets must all use
        # region-local axes. A world AABB mixes the long and short sides when
        # the region is rotated, accepting overhangs and rejecting valid slots.
        seat_axes = self.T_WR[:2, :2]
        if abs(float(np.linalg.det(seat_axes))) < 1e-9:
            return None
        seat_origin = center_xy
        region_half = self.interior_half_xy_local()
        object_in_region = self.T_WR[:3, :3].T @ self.destination_rotation
        footprint = np.abs(object_in_region[:2, :]) @ self.local_size / 2.
        if on_support:
            floor = self.floor_top_local()
            seat_origin = (self.T_WR @ np.r_[self.region_center_local[:2], floor, 1.])[:2]
            footprint = np.zeros(2)
        physical_limit = (region_half - footprint
                          - float(self.request.constraints.collision_margin_m))
        if np.any(physical_limit < -1e-9):
            return None
        physical_limit = np.maximum(physical_limit, 0.)

        def acceptable(xy, *, ignore_occupants=False):
            relative = np.linalg.solve(seat_axes, xy - seat_origin)
            if np.any(np.abs(relative) > physical_limit + 1e-9):
                return False
            if seat_allowed is not None and not seat_allowed(xy):
                return False
            return self.seat_clear_of_occupants(
                xy, ignore_occupants=ignore_occupants)

        anchor = self._slot_xy()
        if anchor is None:
            anchor = center_xy
        if acceptable(anchor):
            return anchor
        # M2 assigns non-conflicting seats in its own packing model; runtime
        # AABBs of already-placed co-region objects (plate, utensils) often
        # still overlap those seats. Honour the planned seat inside the
        # measured interior; free-grid search keeps full occupant blocking.
        if self._m2_layout_slot() is not None and acceptable(
                anchor, ignore_occupants=True):
            return anchor
        limit = (physical_limit if on_support else
                 np.maximum(region_half - REGION_WALL_ALLOWANCE_M, 0.) - footprint)
        if np.any(limit < -1e-9):
            return None
        limit = np.maximum(limit, 0.)
        xs = np.arange(-limit[0], limit[0] + 1e-9, FREE_SPOT_GRID_M)
        ys = np.arange(-limit[1], limit[1] + 1e-9, FREE_SPOT_GRID_M)
        candidates = [seat_origin + seat_axes @ np.array([dx, dy])
                      for dx in xs for dy in ys]
        for xy in sorted(candidates, key=lambda xy: float(np.sum((xy - anchor) ** 2))):
            if acceptable(xy):
                return xy
        return None

    def place_xy_away_from(
        self,
        partner_xy,
        *,
        rejected_xys=(),
        min_shift_m=0.0,
        current_xy=None,
        toward_partner=False,
        avoid_partner_xys=(),
    ):
        """Next in-region place XY that clears a measured collision partner.

        Used by collision-feedback repair when an EE/hand/held-payload margin
        violation is measured against a region occupant: stay inside the
        interior, clear hard (thin-strip) blockers, satisfy ``min_shift_m``
        relative to the measured partner, keep prior flee partners from
        collapsing back onto the held object, and do not reuse prior seats.

        Soft co-region platforms (plate/mug) keep the M2 packing rule — their
        inflated AABB must not void a repair seat. Physics already supplied the
        partner and deficit; maximizing center distance alone would drive the
        seat into the next soft body, so candidates prefer the nearest seat
        that still meets the shift while staying far from other soft bodies.

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
        avoid = [
            np.asarray(item, dtype=float).reshape(2)
            for item in avoid_partner_xys
            if item is not None
        ]
        margin = max(.01, float(self.request.constraints.collision_margin_m) * 2.)
        # Packed trays often need a later repair to step slightly back toward a
        # prior soft partner (apple) to leave the next one (plate). Allow a
        # small collapse of that prior distance without undoing the flee.
        avoid_slack = max(
            4.0 * float(self.request.constraints.collision_margin_m),
            2.0 * REJECTED_PLACE_EPS_M,
        )
        mine = self.half[:2]
        min_shift = max(0.0, float(min_shift_m))
        anchor = (
            np.asarray(current_xy, dtype=float).reshape(2)
            if current_xy is not None
            else self.free_destination_xy()
        )
        # A repair must not trade a hand-margin miss for the EE-rack sweep,
        # unless the seat it repairs was already inside it (no clear seat).
        keep_sweep_clear = self.ee_path_clearance(anchor) >= 0.
        occupants = self._occupants()

        def other_soft_separation(xy):
            best = None
            for c, _h, _t, soft in occupants:
                if not soft:
                    continue
                if float(np.linalg.norm(c - partner)) < REJECTED_PLACE_EPS_M:
                    continue
                dist = float(np.linalg.norm(xy - c))
                if best is None or dist < best:
                    best = dist
            return 0.0 if best is None else best

        def occupants_acceptable(xy):
            if not occupants:
                return True
            for c, h, _t, soft in occupants:
                # Soft footprints are advisory only; thin strips stay hard.
                if soft:
                    continue
                gap = float(np.max(np.abs(xy - c) - (h + mine + margin)))
                if gap < 0.:
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
            if keep_sweep_clear and self.ee_path_clearance(xy) < 0.:
                return False
            for prior in rejected:
                if float(np.linalg.norm(xy - prior)) < REJECTED_PLACE_EPS_M:
                    return False
            for prior_partner in avoid:
                if (
                    float(np.linalg.norm(xy - prior_partner)) + 1e-9
                    < float(np.linalg.norm(anchor - prior_partner)) - avoid_slack
                ):
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

        def candidate_key(xy):
            if toward_partner:
                return (
                    float(np.linalg.norm(xy - partner)),
                    float(np.linalg.norm(xy - anchor)),
                )
            # Prefer a short move that still meets min_shift, and among those
            # stay far from other soft co-region bodies (avoid apple→plate).
            return (
                float(np.linalg.norm(xy - anchor)),
                -other_soft_separation(xy),
                -float(np.linalg.norm(xy - partner)),
            )

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
            best_xy = None
            best_key = None
            for angle in angles:
                c, s = math.cos(angle), math.sin(angle)
                direction = np.array(
                    [away[0] * c - away[1] * s, away[0] * s + away[1] * c]
                )
                xy = anchor + direction * radius
                if not usable(xy):
                    continue
                key = candidate_key(xy)
                if best_key is None or key < best_key:
                    best_xy = xy
                    best_key = key
            if best_xy is not None:
                return best_xy
        # Fall back to a full interior grid with the same ranking.
        limit = self.region_half[:2] - REGION_WALL_ALLOWANCE_M - mine
        if np.any(limit <= 0.):
            return None
        center_xy = self.region_world[:2]
        xs = np.arange(-limit[0], limit[0] + 1e-9, FREE_SPOT_GRID_M)
        ys = np.arange(-limit[1], limit[1] + 1e-9, FREE_SPOT_GRID_M)
        ranked = sorted(
            (center_xy + np.array([dx, dy]) for dx in xs for dy in ys),
            key=candidate_key,
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
        floor = self.floor_top_world_z()
        container = self.record.get('packing_metadata', {})
        if _action(self.task) == 'PLACE_ON' and container.get('kind') == 'CONTAINER':
            # Rest on the container's opening plane, not its interior contents.
            # Use the same explicit rim metadata as PackingBinding.
            rim = float(container['opening_top_z_m'])
            if not math.isfinite(rim) or not np.allclose(self.T_WR[:3, 2], [0., 0., 1.], atol=1e-6):
                raise ValueError('PLACE_ON_UPRIGHT_CONTAINER_RIM_REQUIRED')
            floor = max(floor, float(self.T_WR[2, 3] + rim))
        del xy
        return floor, False

    # -- publication ---------------------------------------------------------
    def publish(self, goal_key, start_key, desired_center, extra):
        # Publish exactly the orientation whose footprint passed seat search.
        # The preferred tray yaw and its orthogonal alternative are selected
        # before occupancy checks, never after a free XY has been chosen.
        destination = self.T_WB.copy()
        destination[:3, :3] = self.destination_rotation
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
            'start_object_orientation_xyzw': Rotation.from_matrix(self.T_WB[:3, :3]).as_quat().tolist(),
            'eef_orientation_xyzw': Rotation.from_matrix(self.T_WE[:3, :3]).as_quat().tolist(),
            'object_id': self.object_id, 'source': self.source, **extra}
        if goal_key == HELD_PLACE_GOAL_ANCHOR:
            # Withdrawal targets the empty hand, not the held object's origin.
            # This measured entry is a candidate; post-release collision checks
            # must still validate the open hand and the entire return path.
            entry_eef_anchor = goal_key + '_entry_eef'
            anchors[entry_eef_anchor] = (inverse(self.T_WR) @ self.T_WE)[:3, 3].tolist()
            self.task.metadata[goal_key]['entry_eef_anchor'] = entry_eef_anchor
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


def _uses_volume_packing_entry(g):
    from .packing_position import POSITION_POLICY
    return (hasattr(g, 'packing_bounds')
            and g.record.get('packing_metadata', {}).get('kind') == 'CONTAINER'
            and g.record['packing_metadata'].get('position_policy') == POSITION_POLICY)


def _is_conceptual_tool_home(g):
    """True when the destination is a measured tool-return pose, not packing."""
    metadata = g.record.get('metadata') if isinstance(g.record, dict) else None
    return isinstance(metadata, dict) and metadata.get('conceptual_tool_home') is True


def _region_destination_xy(g):
    """Select an entry proposal for explicit 3D packing, or a free floor seat.

    A collision-mesh packing orientation with NEAREST_CLEAR_VOLUME is released
    above the opening. Requiring a disjoint 2D floor footprint before its 3D
    volume/IK checks prevents layering over already packed objects. This branch
    supplies only a preferred XY: packing_destination_center applies the exact
    vertex bounds, and the existing volume preference, IK, path, release and
    measured containment gates remain authoritative.

    Conceptual tool home is a measured return pose, not a shared packing region.
    Occupants left under that footprint (e.g. after a sweep) must not trigger
    packing seat search; residual under-tool contact uses the ordinary
    breakaway / allowed-collision path at execution.
    """
    if _is_conceptual_tool_home(g):
        return g.region_world[:2].copy()
    if _uses_volume_packing_entry(g):
        preferred = g._slot_xy()
        return g.region_world[:2].copy() if preferred is None else preferred
    return g.free_destination_xy()


def transport_destination_center(g):
    """Shared exact center for transport grounding and orientation IK probes."""
    desired_center = g.region_world.copy()
    desired_center[:2] = _region_destination_xy(g)
    # Keep the measured object bbox clear of the rim without an arbitrary 5 cm
    # standoff that can place a reachable kitchen destination outside UR5e reach.
    clearance = max(.02, g.request.constraints.collision_margin_m * 2.)
    desired_center[2] = max(g.center[2],
                            g.interior_top_world_z() + g.half[2] + clearance)
    from .container_orientation import packing_destination_center
    return packing_destination_center(g, desired_center, place=False)


def ground_held_transport(request, retention=None):
    """Carry the held object to a free spot above the region with rim clearance."""
    _materialize_conceptual_tool_home(request, retention)
    if request.task.metadata.get(HELD_TRANSPORT_GOAL_ANCHOR) is not None:
        return
    g = _grounding_for(request, retention, _is_transport)
    if g is None:
        return
    desired_center = transport_destination_center(g)
    g.task.goal.target_pose = None
    destination = g.publish(HELD_TRANSPORT_GOAL_ANCHOR, HELD_TRANSPORT_START_ANCHOR, desired_center, {})
    from .container_release import clear_container_rim
    raised, evidence = clear_container_rim(g, destination, retention)
    if evidence:
        desired_center[2] += raised[2, 3] - destination[2, 3]
        g.publish(HELD_TRANSPORT_GOAL_ANCHOR, HELD_TRANSPORT_START_ANCHOR, desired_center, evidence)


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
    xy = _region_destination_xy(g)
    # Floor-level lateral jaw clearance is not the release model for an
    # explicit above-opening drop. Its open-hand corridor and measured rim
    # clearance are checked by the packing path and _seat_held_place_at.
    # Measured tool-home returns keep the fixed pose; under-footprint debris
    # is handled by breakaway rather than lateral seat shifts.
    if not _uses_volume_packing_entry(g) and not _is_conceptual_tool_home(g):
        xy = _clear_finger_overhang(g, xy)
    _seat_held_place_at(g, xy)


def _grip_beside_payload(g) -> bool:
    """True when the grip site sits at the payload silhouette, not over its center."""

    from tuj.m5_motion.grasp_geometry import multi_finger_finger_below_tcp_m

    offset = np.asarray(g.T_WE[:2, 3], dtype=float) - np.asarray(g.center[:2], dtype=float)
    silhouette_gap = float(np.max(np.abs(offset) - np.asarray(g.half[:2], dtype=float)))
    return silhouette_gap > -float(multi_finger_finger_below_tcp_m())


def _clear_finger_overhang(g, desired_xy):
    """Move a side grasp off a neighbor that the fingertips would still enter.

    The object footprint can clear an occupant while the jaws, parked beside
    the payload, still reach that occupant. Shift away by the shortfall and
    stay inside the region.
    """

    from tuj.m5_motion.grasp_geometry import multi_finger_finger_below_tcp_m

    if not _grip_beside_payload(g):
        return desired_xy
    occupants = g._occupants()
    if not occupants:
        return desired_xy
    mine = np.asarray(g.half[:2], dtype=float)
    seat = np.asarray(desired_xy, dtype=float).reshape(2)
    nearest = None
    nearest_gap = None
    for center, half, _top, _soft in occupants:
        gap = float(np.max(np.abs(seat - center) - (np.asarray(half, dtype=float) + mine)))
        if nearest_gap is None or gap < nearest_gap:
            nearest_gap = gap
            nearest = np.asarray(center, dtype=float).reshape(2)
    reach = float(multi_finger_finger_below_tcp_m())
    margin = float(g.request.constraints.collision_margin_m)
    if nearest is None or nearest_gap is None or nearest_gap >= reach + margin:
        return desired_xy
    shortfall = reach + margin - nearest_gap
    shifted = g.place_xy_away_from(
        nearest,
        min_shift_m=shortfall,
        current_xy=seat,
    )
    def fingers_clear(xy):
        return all(float(np.max(np.abs(xy - center) - (np.asarray(half) + mine)))
                   >= reach + margin - 1e-9 for center, half, _top, _soft in occupants)

    if shifted is not None and g.seat_clear_of_occupants(shifted) and fingers_clear(shifted):
        return shifted
    away = seat - nearest
    norm = float(np.linalg.norm(away))
    if norm <= 1e-9:
        raise ValueError("PLACEMENT_FINGER_SPACE_EXHAUSTED")
    away /= norm
    limit = g.usable_half_xy() - mine
    for direction in (
        away,
        np.array([0.0, math.copysign(1.0, away[1])]),
        np.array([math.copysign(1.0, away[0]), 0.0]),
    ):
        candidate = seat + direction * shortfall
        offset = candidate - g.region_world[:2]
        if (np.all(np.abs(offset) <= limit + 1e-9)
                and g.seat_clear_of_occupants(candidate)
                and fingers_clear(candidate)):
            return candidate
    raise ValueError("PLACEMENT_FINGER_SPACE_EXHAUSTED")


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
    # A thin parallel jaw is not raised on the first seat. Releasing it by the
    # inner-finger remainder lets the payload fall into the robot mount after
    # the fingers open. A later repair adds height only after the hand is
    # measured deeper than the fixture tolerance.
    mount_lift = request.task.metadata.get(_HELD_PLACE_MOUNT_RELEASE_LIFT_KEY)
    if isinstance(mount_lift, (int, float)) and math.isfinite(mount_lift) and mount_lift > 0.0:
        release_clearance += float(mount_lift)
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
    desired_center[2] = origin_z + float(g.destination_rotation[2, :] @ g.center_in_body)
    from .container_orientation import packing_destination_center
    desired_center = packing_destination_center(g, desired_center, place=True)
    place_extra = {
        'release_clearance_m': release_clearance,
        'destination_center_xy_m': [
            float(desired_center[0]),
            float(desired_center[1]),
        ],
    }
    destination = g.publish(
        HELD_PLACE_GOAL_ANCHOR, HELD_PLACE_START_ANCHOR, desired_center,
        place_extra)
    if _request_uses_vacuum(request):
        destination, release_clearance, lift = _raise_place_for_vacuum_ee_clearance(
            g, destination,
            support_z=support_z,
            release_clearance=release_clearance,
            collision_margin_m=request.constraints.collision_margin_m,
        )
        desired_center[2] += lift
        place_extra['release_clearance_m'] = float(release_clearance)
        if lift > 0.:
            place_extra.update({
                'vacuum_ee_clearance_lift_m': float(lift),
                'vacuum_ee_cup_half_height_m': _vacuum_cup_half_height_m(),
            })
    from .container_release import clear_container_rim
    raised, evidence = clear_container_rim(g, destination, g.retention)
    if evidence:
        desired_center[2] += raised[2, 3] - destination[2, 3]
        destination = g.publish(HELD_PLACE_GOAL_ANCHOR, HELD_PLACE_START_ANCHOR,
            desired_center, {**place_extra, **evidence})
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


def _place_body_hits_mount(label: str, request) -> bool:
    """Hand or held payload: both travel with the place seat."""

    if _is_robot_motion_collision_label(label, request):
        return True
    held_id = held_pose_subject(request)
    return bool(held_id and _label_matches_object_id(str(label), held_id))


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
    """Pick the scene partner of a place-body margin violation, if any.

    Place-body covers the moving hand/EE and the held payload: both travel
    with the place seat, so a held↔occupant clearance miss needs the same XY
    diversification as a hand↔occupant miss.

    Cumulative repair history may still list an older, deeper miss. Prefer
    partners from the newest ``source_repair_attempt`` and skip partners the
    seat already fled, so repair cannot keep marching away from a cleared
    occupant into the next collision (and eventually the robot mount).
    """
    if not isinstance(feedback, dict):
        return None
    strategies = feedback.get("failed_strategies")
    if not isinstance(strategies, list):
        return None
    held_id = held_pose_subject(request)
    region_id = request.task.goal.target_region_id
    avoided_ids = {
        str(item)
        for item in (feedback.get("avoid_place_partner_ids") or [])
        if isinstance(item, str) and item
    }
    candidates = []
    latest_attempt = None
    for strategy in strategies:
        if not isinstance(strategy, dict):
            continue
        try:
            source_attempt = int(strategy.get("source_repair_attempt", -1))
        except (TypeError, ValueError):
            source_attempt = -1
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
            moving_labels = [
                label for label in labels
                if _place_body_hits_mount(label, request)
            ]
            if not moving_labels:
                continue
            scene_labels = [label for label in labels if label not in moving_labels]
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
            if latest_attempt is None or source_attempt > latest_attempt:
                latest_attempt = source_attempt
            candidates.append(
                (source_attempt, deficit, partner_id == region_id, center, partner_id)
            )
    if not candidates:
        return None
    preferred = [
        item for item in candidates if item[0] == latest_attempt
    ] or candidates
    # Skip already-fled partners when the newest batch has another option;
    # if the newest batch only repeats the same partner, allow another flee.
    active = [
        item for item in preferred if item[4] not in avoided_ids
    ] or preferred
    _source, deficit, _region_tie, partner_xy, partner_id = max(
        active,
        key=lambda item: (item[1], 1.0 if item[2] else 0.0),
    )
    return {
        "partner_xy": partner_xy,
        "deficit_m": float(deficit),
        "partner_id": partner_id,
    }


def _mount_penetration_shortfall_m(request, feedback) -> float | None:
    """How far a hand/mount pair sits past the negative fixture tolerance.

    Scene-margin violations keep a positive required clearance and are seated
    in XY. A negative required clearance is the robot-mount penetration limit.
    """

    if not isinstance(feedback, dict):
        return None
    strategies = feedback.get("failed_strategies")
    if not isinstance(strategies, list):
        return None
    shortfall = 0.0
    found = False
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
                or required >= 0.0
                or measured >= required
            ):
                continue
            labels = (
                str(observation.get("geometry_a", "")),
                str(observation.get("geometry_b", "")),
            )
            if not any(
                _place_body_hits_mount(label, request) for label in labels
            ):
                continue
            found = True
            shortfall = max(shortfall, required - measured)
    if not found:
        return None
    return float(shortfall)


def _robot_base_xy(request) -> np.ndarray | None:
    """World XY of the robot base the mount stands on, when the scene recorded it."""

    raw = request.world.metadata.get("robot_base_world_m")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    xy = np.asarray(raw[:2], dtype=float)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        return None
    return xy


def _shift_held_place_off_mount(request, feedback, shortfall_m: float) -> bool:
    """Move the place seat away from the robot base, keeping the low release.

    A same-XY lift clears the mount by dropping the payload from the finger
    depth. That fall leaves a tall grasp outside a region that the low seat
    still contained. Sliding off the mount keeps the original release height.
    """

    base_xy = _robot_base_xy(request)
    if base_xy is None:
        return False
    g = _grounding_for(request, None, _is_region_place)
    if g is None:
        return False
    current = _held_place_center_xy(request)
    if current is None:
        return False
    rejected = []
    raw_rejected = feedback.get("rejected_place_xy_m") if isinstance(feedback, dict) else None
    if isinstance(raw_rejected, list):
        for item in raw_rejected:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                rejected.append([float(item[0]), float(item[1])])
    rejected.append([float(current[0]), float(current[1])])
    from tuj.m5_motion.grasp_geometry import (
        multi_finger_finger_below_tcp_m,
        parallel_jaw_inner_finger_extra_below_tcp_m,
    )

    margin = float(request.constraints.collision_margin_m)
    # One step has to clear the inner finger that still overlaps the mount
    # after the distal pad has moved past it.
    min_shift = max(
        float(shortfall_m) + margin,
        float(multi_finger_finger_below_tcp_m())
        + float(parallel_jaw_inner_finger_extra_below_tcp_m()),
    )
    avoid = []
    raw_avoid = feedback.get("avoid_place_partner_xy_m") if isinstance(feedback, dict) else None
    if isinstance(raw_avoid, list):
        for item in raw_avoid:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                avoid.append([float(item[0]), float(item[1])])
    next_xy = g.place_xy_away_from(
        base_xy,
        rejected_xys=rejected,
        min_shift_m=min_shift,
        current_xy=current,
        avoid_partner_xys=avoid,
    )
    if next_xy is None:
        return False
    request.task.metadata.pop(_HELD_PLACE_MOUNT_RELEASE_LIFT_KEY, None)
    _clear_held_place_grounding(request)
    seated = _seat_held_place_at(g, next_xy)
    if isinstance(feedback, dict):
        feedback["rejected_place_xy_m"] = rejected
        feedback["reground_place_partner_id"] = "robot_base"
        feedback["reground_place_xy_m"] = [float(seated[0]), float(seated[1])]
    return True


def _raise_held_place_for_mount(request, feedback, shortfall_m: float) -> bool:
    """Republish the same place XY high enough to leave the mount tolerance."""

    g = _grounding_for(request, None, _is_region_place)
    if g is None:
        return False
    current = _held_place_center_xy(request)
    if current is None:
        return False
    previous = request.task.metadata.get(_HELD_PLACE_MOUNT_RELEASE_LIFT_KEY, 0.0)
    try:
        previous = float(previous)
    except (TypeError, ValueError):
        previous = 0.0
    if not math.isfinite(previous) or previous < 0.0:
        previous = 0.0
    from tuj.m5_motion.grasp_geometry import (
        multi_finger_finger_below_tcp_m,
        parallel_jaw_inner_finger_extra_below_tcp_m,
    )

    margin = float(request.constraints.collision_margin_m)
    # The leftover past the mount tolerance is only the last millimetre. The
    # inner finger hangs below the distal pad, so the repair covers both.
    step = max(
        float(shortfall_m) + margin,
        float(multi_finger_finger_below_tcp_m())
        + float(parallel_jaw_inner_finger_extra_below_tcp_m()),
    )
    lift = previous + step
    request.task.metadata[_HELD_PLACE_MOUNT_RELEASE_LIFT_KEY] = lift
    _clear_held_place_grounding(request)
    _seat_held_place_at(g, current)
    if isinstance(feedback, dict):
        feedback["reground_mount_release_lift_m"] = lift
    return True


def reground_held_place_from_collision_feedback(request, feedback) -> bool:
    """Rewrite ``held_place_goal`` away from a colliding partner for repair.

    Returns True when a distinct in-region place seat was published. Real
    penetration still fails later validation; this only diversifies the PLACE
    XY that collision-feedback retries would otherwise freeze. A hand that is
    already past the robot-mount tolerance moves away from the robot base at
    the same release height. The same XY is raised only when that seat does
    not fit in the region.

    Soft co-region partners are fled in XY once. A second miss against the same
    partner raises the release instead: further XY shifts walk into the other
    packed occupant (apple↔plate) and waste the repair budget on the mount.
    """
    if not _is_region_place(request.task):
        return False
    shortfall = _mount_penetration_shortfall_m(request, feedback)
    if shortfall is not None:
        if _shift_held_place_off_mount(request, feedback, shortfall):
            return True
        return _raise_held_place_for_mount(request, feedback, shortfall)
    partner = _place_margin_partner_from_feedback(request, feedback)
    if partner is None:
        return False
    avoided_ids = {
        str(item)
        for item in (
            (feedback.get("avoid_place_partner_ids") or [])
            if isinstance(feedback, dict)
            else []
        )
        if isinstance(item, str) and item
    }
    toward_region = (
        partner["partner_id"] == request.task.goal.target_region_id
    )
    # Already fled this soft/scene partner once: lift instead of marching XY
    # into the next packed occupant.
    if partner["partner_id"] in avoided_ids and not toward_region:
        raised = _raise_held_place_for_mount(
            request, feedback, float(partner["deficit_m"])
        )
        if raised and isinstance(feedback, dict):
            feedback["reground_place_partner_id"] = partner["partner_id"]
            feedback["reground_place_lift_only"] = True
        return raised
    g = _grounding_for(request, None, _is_region_place)
    if g is None:
        return False
    rejected = []
    raw_rejected = feedback.get("rejected_place_xy_m") if isinstance(feedback, dict) else None
    if isinstance(raw_rejected, list):
        for item in raw_rejected:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                rejected.append([float(item[0]), float(item[1])])
    avoid = []
    raw_avoid = (
        feedback.get("avoid_place_partner_xy_m") if isinstance(feedback, dict) else None
    )
    if isinstance(raw_avoid, list):
        for item in raw_avoid:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                avoid.append([float(item[0]), float(item[1])])
    current = _held_place_center_xy(request)
    if current is not None:
        rejected.append([float(current[0]), float(current[1])])
    min_shift = float(partner["deficit_m"]) + float(
        request.constraints.collision_margin_m
    )
    # Destination-region partners are rim/floor geoms: move toward the region
    # center. Fleeing the region center pushes an off-centre vac cup deeper
    # into the wall that already failed margin.
    next_xy = g.place_xy_away_from(
        partner["partner_xy"],
        rejected_xys=rejected,
        min_shift_m=min_shift,
        current_xy=current,
        toward_partner=toward_region,
        avoid_partner_xys=avoid,
    )
    if next_xy is None:
        raised = _raise_held_place_for_mount(
            request, feedback, float(partner["deficit_m"])
        )
        if raised and isinstance(feedback, dict):
            feedback["reground_place_partner_id"] = partner["partner_id"]
            feedback["reground_place_lift_only"] = True
        return raised
    _clear_held_place_grounding(request)
    seated = _seat_held_place_at(g, next_xy)
    if isinstance(feedback, dict):
        feedback["rejected_place_xy_m"] = rejected
        feedback["reground_place_partner_id"] = partner["partner_id"]
        feedback["reground_place_xy_m"] = [float(seated[0]), float(seated[1])]
        # Remember this partner so the next repair batch cannot re-flee it or
        # walk back into it while clearing a newer collision.
        partner_xy = [
            float(partner["partner_xy"][0]),
            float(partner["partner_xy"][1]),
        ]
        if not any(
            float(np.linalg.norm(np.asarray(prior, dtype=float) - partner["partner_xy"]))
            < REJECTED_PLACE_EPS_M
            for prior in avoid
        ):
            avoid.append(partner_xy)
        feedback["avoid_place_partner_xy_m"] = avoid
        avoided_ids_list = [
            str(item)
            for item in (feedback.get("avoid_place_partner_ids") or [])
            if isinstance(item, str) and item
        ]
        if partner["partner_id"] not in avoided_ids_list:
            avoided_ids_list.append(str(partner["partner_id"]))
        feedback["avoid_place_partner_ids"] = avoided_ids_list
    return True


def ground_held_region_goal(request, retention=None):
    """Ground whichever implicit region goal the task carries (transport or place)."""
    ground_held_transport(request, retention)
    ground_held_place(request, retention)
