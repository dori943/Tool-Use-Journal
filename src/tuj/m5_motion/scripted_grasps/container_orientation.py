"""Restore environment-owned packing orientations in object-space grounding."""
import numpy as np
from scipy.spatial.transform import Rotation
from tuj.m5_motion.packing import _orientation_candidates


def configure_packing_orientation(g):
    record = g.request.world.objects.get(g.object_id, {})
    metadata = record.get('packing_metadata', {})
    container = g.record.get('packing_metadata', {})
    if container.get('kind') != 'CONTAINER':
        return
    if metadata.get('stable_face_policy') == 'MINIMUM_HEIGHT':
        configure_stable_face(g, record, container)
        return
    if not metadata.get('orientation_candidates'):
        return
    if metadata.get('orientation_frame') != 'TARGET_REGION':
        raise ValueError('PACKING_ORIENTATION_FRAME_REQUIRED')
    if not np.allclose(g.T_WR[:3, 2], [0., 0., 1.], atol=1e-6):
        raise ValueError('PACKING_UPRIGHT_CONTAINER_REQUIRED')
    points = np.asarray(record.get('collision_points_m'), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('PACKING_COLLISION_VERTICES_REQUIRED')
    dimensions = np.asarray(container['interior_dimensions_m'], dtype=float)
    margin = float(g.request.constraints.collision_margin_m)
    current = g.T_WR[:3, :3].T @ g.T_WB[:3, :3]
    choices = []
    for candidate in _orientation_candidates(record):
        rotation = Rotation.from_quat(candidate.orientation_xyzw).as_matrix()
        rotated = points @ rotation.T
        lower, upper = rotated.min(0), rotated.max(0)
        if np.any(upper - lower + 2. * margin > dimensions):
            continue
        angle = Rotation.from_matrix(current.T @ rotation).magnitude()
        choices.append((angle, rotation, lower, upper))
    if not choices:
        raise ValueError('PACKING_NO_INTERIOR_FIT_ORIENTATION')
    _, rotation, lower, upper = min(choices, key=lambda row: row[0])
    g.destination_rotation = g.T_WR[:3, :3] @ rotation
    g.preserve_destination_rotation = True
    g.half = np.ptp(points @ g.destination_rotation.T, axis=0) / 2.
    g.packing_bounds = (lower, upper)


def packing_destination_center(g, desired_center, *, place):
    if not hasattr(g, 'packing_bounds'):
        return desired_center
    container = g.record['packing_metadata']
    lower, upper = g.packing_bounds
    center = np.asarray(container['interior_center_m'], dtype=float)
    half = np.asarray(container['interior_dimensions_m'], dtype=float) / 2.
    margin = float(g.request.constraints.collision_margin_m)
    # Clamp the requested body origin using collision vertices, not an unrotated
    # object bbox. The measured grasp transform is still used by retargeting.
    body = desired_center - g.destination_rotation @ g.center_in_body
    local = g.T_WR[:3, :3].T @ (body - g.T_WR[:3, 3])
    local[:2] = np.clip(local[:2], (center-half+margin-lower)[:2],
                       (center+half-margin-upper)[:2])
    clearance = max(margin, .005) if place else max(.02, 2. * margin)
    # Release above the opening, matching the existing PackingBinding policy.
    # Physical settling and the unchanged 3D containment gate determine success.
    rim_world = g.T_WR[2, 3] + float(container['opening_top_z_m'])
    obstacle_top = max(rim_world, g.interior_top_world_z())
    minimum_z = obstacle_top - g.T_WR[2, 3] + clearance - lower[2]
    local[2] = minimum_z if place else max(local[2], minimum_z)
    body = g.T_WR[:3, 3] + g.T_WR[:3, :3] @ local
    return body + g.destination_rotation @ g.center_in_body


def configure_stable_face(g, record, container):
    """Use explicitly declared stable box faces; never infer this for meshes."""
    from itertools import permutations, product
    if not np.allclose(g.T_WR[:3, 2], [0., 0., 1.], atol=1e-6):
        raise ValueError('PACKING_UPRIGHT_CONTAINER_REQUIRED')
    points = np.asarray(record.get('collision_points_m'), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('PACKING_COLLISION_VERTICES_REQUIRED')
    dims = np.asarray(container['interior_dimensions_m'])
    margin = float(g.request.constraints.collision_margin_m)
    current = g.T_WR[:3, :3].T @ g.T_WB[:3, :3]
    choices = []
    retained = getattr(getattr(g, 'retention', None), 'stable_packing_rotation', None)
    if retained is not None:
        g.destination_rotation = np.asarray(retained).copy()
        g.half = np.ptp(points @ g.destination_rotation.T, axis=0) / 2.
        g.preserve_destination_rotation = True
        g.stable_face_xy = g.free_destination_xy()
        return
    for axes in permutations(range(3)):
        for signs in product((-1., 1.), repeat=3):
            local = np.eye(3)[:, axes] @ np.diag(signs)
            if np.linalg.det(local) < 0.:
                continue
            extents = np.ptp(points @ local.T, axis=0)
            if np.any(extents + 2. * margin > dims):
                continue
            rotation = g.T_WR[:3, :3] @ local
            g.destination_rotation = rotation
            g.half = np.ptp(points @ rotation.T, axis=0) / 2.
            xy = g.free_destination_xy()
            support, _ = g.support_top_world_z(xy)
            if not stable_face_has_ik(g, xy):
                continue
            angle = Rotation.from_matrix(current.T @ local).magnitude()
            # Quantization only breaks floating-point symmetry of equal faces.
            key = (round(float(extents[2]), 9), round(float(support), 9), angle)
            choices.append((key, rotation.copy(), g.half.copy(), xy.copy()))
    if not choices:
        raise ValueError('PACKING_NO_REACHABLE_STABLE_FACE_FITS')
    _, g.destination_rotation, g.half, g.stable_face_xy = min(choices, key=lambda row: row[0])
    g.preserve_destination_rotation = True
    if getattr(g, 'retention', None) is not None:
        g.retention.stable_packing_rotation = g.destination_rotation.copy()


def stable_face_has_ik(g, xy):
    """Prefilter full measured-grasp targets; path collision checks remain mandatory."""
    retention = getattr(g, 'retention', None)
    if retention is None:
        return True
    from .frames import inverse
    c = retention.context
    clearance = max(.02, float(g.request.constraints.collision_margin_m) * 2.)
    center = np.r_[xy, max(g.center[2], g.interior_top_world_z() + g.half[2] + clearance)]
    target = np.eye(4)
    target[:3, :3] = g.destination_rotation
    target[:3, 3] = center - g.destination_rotation @ g.center_in_body
    from .container_release import clear_container_rim
    target, _ = clear_container_rim(g, target, retention)
    target = target @ inverse(inverse(g.T_WE) @ g.T_WB)
    solved = c.kinematics.solve_all_ik(target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_quat(),
                                       seed_qpos=c.data.qpos[c.arm_ids])
    g.task.metadata.setdefault('stable_face_ik_candidates', []).append({
        'object_orientation_xyzw': Rotation.from_matrix(g.destination_rotation).as_quat().tolist(),
        'raw_ik_count': len(solved.solutions),
    })
    return bool(solved.solutions)
