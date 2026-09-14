"""Certify shallow convex overlap when a native distance query rejects attach.

Projection bounds validate penetration only. They are not a replacement for
Euclidean proximity, so recovery also requires a loaded physical contact.
"""
import math

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError


def projection_penetration_certificate(points_a, points_b, limit, padding):
    """A separating translation along any unit axis bounds penetration above.

    Checking only hull face normals is sufficient for a certificate, but may
    conservatively reject an edge/edge case. No exact SAT distance is claimed.
    """
    arrays = [np.asarray(p, dtype=float) for p in (points_a, points_b)]
    if (not math.isfinite(limit) or limit < 0 or not math.isfinite(padding)
            or padding < 0 or any(p.ndim != 2 or p.shape[1] != 3 or len(p) < 4
                                  or not np.isfinite(p).all() for p in arrays)):
        return None
    origin = arrays[0][0].copy()
    a, b = [p - origin for p in arrays]
    try:
        axes = np.concatenate([ConvexHull(p).equations[:, :3] for p in (a, b)])
    except QhullError:
        return None
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    pa, pb = a @ axes.T, b @ axes.T
    gaps = np.maximum(pa.min(axis=0) - pb.max(axis=0),
                      pb.min(axis=0) - pa.max(axis=0))
    index = int(np.argmax(gaps))
    # Account for arithmetic on world transforms as well as the engine's
    # existing CCD tolerance. Padding makes certification stricter.
    scale = max(1., *(float(np.abs(p).max()) for p in arrays))
    padding = max(padding, 64 * np.finfo(float).eps * scale)
    bound = float(gaps[index]) - padding
    return {
        'certified': bound >= -limit,
        'axis_world': axes[index].tolist(),
        'signed_projection_lower_bound_m': bound,
        'penetration_upper_bound_m': max(0., -bound),
        'numerical_padding_m': padding,
        'penetration_limit_m': limit,
    }


def certify_attachment_penetration(model, data, ee_geoms, object_geoms, limit):
    """Validate every offending pair, preserving the original gate on failure."""
    from tuj.m5_motion.tool_use_journal import _geom_local_points

    ee_geoms, object_geoms = set(ee_geoms), set(object_geoms)
    audit = {'certified': False, 'reason': 'PHYSICAL_CONTACT_REQUIRED',
             'physical_contacts': [], 'offending_pairs': [],
             'penetration_limit_m': limit, 'simulation_time_s': float(data.time)}
    for i, contact in enumerate(data.contact[:data.ncon]):
        a, b = int(contact.geom1), int(contact.geom2)
        if not ((a in ee_geoms and b in object_geoms)
                or (b in ee_geoms and a in object_geoms)):
            continue
        distance = float(contact.dist)
        if not math.isfinite(distance) or distance < -limit:
            audit['reason'] = 'PHYSICAL_PENETRATION_EXCEEDS_LIMIT'
            return audit
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        if distance <= 0 and contact.efc_address >= 0 and np.isfinite(force).all() and force[0] > 0:
            audit['physical_contacts'].append({
                'geoms': [model.geom(a).name, model.geom(b).name],
                'distance_m': distance, 'normal_force_n': float(force[0])})
    if not audit['physical_contacts']:
        return audit

    def world_vertices(gid):
        if model.geom_type[gid] not in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_BOX):
            return None
        local = _geom_local_points(model, gid)
        if local is None:
            return None
        return local @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]

    for a in sorted(ee_geoms):
        for b in sorted(object_geoms):
            raw = float(mujoco.mj_geomDistance(model, data, a, b, 10., None))
            if not math.isfinite(raw):
                audit['reason'] = 'NONFINITE_NATIVE_DISTANCE'
                return audit
            if raw >= -limit:
                continue
            row = {'geoms': [model.geom(a).name, model.geom(b).name],
                   'raw_distance_m': raw, 'certificate': None}
            audit['offending_pairs'].append(row)
            points = [world_vertices(g) for g in (a, b)]
            if any(p is None for p in points):
                audit['reason'] = 'UNSUPPORTED_COLLISION_GEOMETRY'
                return audit
            certificate = projection_penetration_certificate(
                *points, limit, float(model.opt.ccd_tolerance))
            row['certificate'] = certificate
            if certificate is None or not certificate['certified']:
                audit['reason'] = 'PENETRATION_NOT_CERTIFIED'
                return audit
    if not audit['offending_pairs']:
        audit['reason'] = 'NATIVE_REJECTION_NOT_REPRODUCED'
        return audit
    audit.update(certified=True, reason='CONTACT_AND_CONVEX_PROJECTION_VERIFIED',
                 proximity_source='LOADED_PHYSICAL_CONTACT',
                 raw_distance_retained=True)
    return audit
