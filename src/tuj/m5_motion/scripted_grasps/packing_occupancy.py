"""Occupied-volume preference only; sampled poses are never motion targets.

Static overlap is not a settling or collision certificate. Normal planning,
release and measured containment gates must still pass in the live simulator.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from tuj.m5_motion.push_to_region import target_fully_inside_region


def packing_occupants(g):
    return sorted(name for name, record in g.request.world.objects.items()
        if name != g.object_id
        and record.get('packing_metadata', {}).get('kind') == 'PACKABLE_OBJECT'
        and target_fully_inside_region(g.request.world, target_id=name,
            region_id=g.task.goal.target_region_id, include_vertical=True))


def packing_translation_bounds(g):
    """Body-origin limits in the container frame, with existing collision margin."""
    metadata = g.record['packing_metadata']
    center = np.asarray(metadata['interior_center_m'], dtype=float)
    half = np.asarray(metadata['interior_dimensions_m'], dtype=float) / 2.
    margin = float(g.request.constraints.collision_margin_m)
    lower, upper = g.packing_bounds
    if not math.isfinite(margin) or margin < 0.:
        raise ValueError('PACKING_PROBE_INVALID_RESOLUTION_OR_MARGIN')
    low, high = center - half + margin - lower, center + half - margin - upper
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high < low):
        raise ValueError('PACKING_PROBE_NO_VERTICAL_FIT')
    return low, high


def packing_overlap_preference(g, occupants, *, body_xy_in_region=None):
    """Minimum over sampled Z of the worst occupant contact depth at each Z.

XY remains the exact grounded transport XY. Probe Z spans collision-vertex
    containment bounds at the request's position resolution, including endpoints.
    This looks for a less obstructed resting height, not a collision-free descent.
    Excessive sample counts fail explicitly instead of silently coarsening the grid.
The result ranks existing IK-feasible orientations, without admitting any
colliding pose to execution or changing the existing release height.
"""
    import mujoco
    from .transport import transport_destination_center
    c = g.retention.context
    model, data = c.model, c.data
    joint = int(model.body_jntadr[c.body_id])
    if (joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
            or model.body_parentid[c.body_id] != 0):
        raise ValueError('PACKING_PROBE_WORLD_FREE_JOINT_REQUIRED')
    address = int(model.jnt_qposadr[joint])
    occupant_geoms = set()
    for name in occupants:
        body = int(c.env.obj_body_id[name])
        occupant_geoms.update(i for i in range(model.ngeom)
                             if c.descendant(int(model.geom_bodyid[i]), body))
    if not occupant_geoms:
        raise ValueError('PACKING_PROBE_OCCUPANT_GEOMETRY_REQUIRED')
    margin = float(g.request.constraints.collision_margin_m)
    resolution = float(g.request.constraints.position_tolerance_m)
    if not math.isfinite(resolution) or resolution <= 0 or not math.isfinite(margin) or margin < 0:
        raise ValueError('PACKING_PROBE_INVALID_RESOLUTION_OR_MARGIN')
    minimum, maximum = packing_translation_bounds(g)
    low, high = float(minimum[2]), float(maximum[2])
    if not math.isfinite(low + high) or high < low:
        raise ValueError('PACKING_PROBE_NO_VERTICAL_FIT')
    count = max(2, 1 + math.ceil((high - low) / resolution))
    if count > 4096:
        raise ValueError('PACKING_PROBE_SAMPLE_BUDGET_EXCEEDED')
    body = transport_destination_center(g) - g.destination_rotation @ g.center_in_body
    local = g.T_WR[:3, :3].T @ (body - g.T_WR[:3, 3])
    if body_xy_in_region is not None:
        xy = np.asarray(body_xy_in_region, dtype=float)
        if (xy.shape != (2,) or not np.isfinite(xy).all()
                or np.any(xy < minimum[:2]) or np.any(xy > maximum[:2])):
            raise ValueError('PACKING_PROBE_XY_OUTSIDE_BOUNDS')
        local[:2] = xy
    quaternion = Rotation.from_matrix(g.destination_rotation).as_quat()[[3, 0, 1, 2]]
    probe = mujoco.MjData(model)
    samples = []
    for z in np.linspace(low, high, count):
        mujoco.mj_resetData(model, probe)
        for field in ('qpos', 'qvel', 'ctrl', 'act'):
            getattr(probe, field)[:] = getattr(data, field)
        probe.time = data.time
        local[2] = z
        probe.qpos[address:address + 3] = g.T_WR[:3, 3] + g.T_WR[:3, :3] @ local
        probe.qpos[address + 3:address + 7] = quaternion
        mujoco.mj_forward(model, probe)
        depths = []
        for contact in probe.contact[:probe.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            if ((a in c.object_geoms and b in occupant_geoms)
                    or (b in c.object_geoms and a in occupant_geoms)):
                depths.append(max(0., -float(contact.dist)))
        samples.append({'body_z_in_region_m': float(z),
                        'max_penetration_m': max(depths, default=0.)})
    score = min(s['max_penetration_m'] for s in samples)
    g.task.metadata.setdefault('packing_occupancy_preferences', []).append({
        'basis': 'STATIC_OCCUPIED_VOLUME_RANKING_ONLY',
        'orientation_xyzw': Rotation.from_matrix(g.destination_rotation).as_quat().tolist(),
        'body_xy_in_region_m': local[:2].tolist(),
        'occupant_ids': occupants, 'sample_count': count,
        'score_m': score, 'samples': samples})
    return score


def bounded_packing_xy(g, value):
    """Snap only floating-point roundoff at an already margin-inset boundary."""
    minimum, maximum = packing_translation_bounds(g)
    xy = np.asarray(value, dtype=float)
    roundoff = 32. * np.finfo(float).eps * max(1., float(np.abs(np.r_[minimum, maximum]).max()))
    if (xy.shape != (2,) or not np.isfinite(xy).all()
            or np.any(xy < minimum[:2] - roundoff)
            or np.any(xy > maximum[:2] + roundoff)):
        raise ValueError('PACKING_POSITION_OUTSIDE_BOUNDS')
    return np.clip(xy, minimum[:2], maximum[:2])
