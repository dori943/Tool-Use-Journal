"""Measured hand opening envelope for collision-safe container release goals."""
import math
import numpy as np

from .frames import inverse


def opening_points_in_body(context):
    """Sample a private kinematic copy; never alter the live simulator state.

    This envelope proposes a goal, not a collision exemption. The ordinary
    planner and live collision checks still validate the complete motion.
    """
    from tuj.m5_motion.tool_use_journal import _geom_local_points
    c = context
    probe = c.mj.MjData(c.model)
    probe.qpos[:] = c.data.qpos
    addresses = [int(c.model.jnt_qposadr[c.model.joint(n).id]) for n in c.gripper.joints]
    from .open_geometry import open_joint_positions
    configured = open_joint_positions(c)
    opened = np.array([configured[n] for n in c.gripper.joints])
    closed = c.data.qpos[addresses].copy()
    count = max(2, 1 + math.ceil(float(np.max(np.abs(opened - closed))) / .05))
    body_from_world = inverse(c.body_pose())
    geometry = [(gid, _geom_local_points(c.model, gid)) for gid in sorted(c.gripper_geoms)
                if c.model.geom_contype[gid] or c.model.geom_conaffinity[gid]]
    points = []
    for alpha in np.linspace(0., 1., count):
        probe.qpos[addresses] = closed + alpha * (opened - closed)
        c.mj.mj_forward(c.model, probe)
        for gid, local in geometry:
            if local is None:
                continue
            world = local @ probe.geom_xmat[gid].reshape(3, 3).T + probe.geom_xpos[gid]
            points.append(world @ body_from_world[:3, :3].T + body_from_world[:3, 3])
    if not points:
        raise ValueError('CONTAINER_RELEASE_HAND_GEOMETRY_REQUIRED')
    return np.concatenate(points), count


def rim_clearance_lift(points_body, destination, region_pose, interior_center,
                       interior_dimensions, rim_z, margin):
    """Return vertical lift only if the hand envelope crosses an interior wall."""
    world = np.asarray(points_body) @ destination[:3, :3].T + destination[:3, 3]
    local = (world - region_pose[:3, 3]) @ region_pose[:3, :3]
    center, half = np.asarray(interior_center), np.asarray(interior_dimensions) / 2.
    # This vertical insertion policy requires an upright container.
    if not np.allclose(region_pose[:3, 2], [0., 0., 1.], atol=1e-6):
        return 0.
    low, high = local.min(axis=0), local.max(axis=0)
    crosses_wall = np.any(low[:2] < center[:2] - half[:2] + margin) or np.any(high[:2] > center[:2] + half[:2] - margin)
    if not crosses_wall:
        return 0.
    return max(0., float(rim_z + margin - low[2]))


def clear_container_rim(grounding, destination, retention):
    metadata = grounding.record.get('packing_metadata', {})
    if retention is None or metadata.get('kind') != 'CONTAINER':
        return destination, {}
    points, samples = opening_points_in_body(retention.context)
    # A nominal target on the collision boundary leaves no room for the
    # request's declared tracking error. This raises the goal, never the gate.
    goal_margin = (float(grounding.request.constraints.collision_margin_m)
                   + float(grounding.request.constraints.position_tolerance_m))
    lift = rim_clearance_lift(points, destination, grounding.T_WR,
        metadata['interior_center_m'], metadata['interior_dimensions_m'],
        metadata['opening_top_z_m'], goal_margin)
    result = destination.copy()
    result[2, 3] += lift
    return result, {'container_release_lift_m': lift,
                    'container_goal_clearance_m': goal_margin,
                    'release_geometry_source': 'LIVE_HAND_OPENING_ENVELOPE',
                    'opening_envelope_samples': samples}
