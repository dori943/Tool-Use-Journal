"""Derive an edge grasp whose open hand clears an object support plane.

This is a grasp proposal, not a contact exemption or a release certificate.
The normal physical acquisition, motion, and placement checks remain required.
"""
import math

import numpy as np


def shift_targets_for_open_clearance(targets, object_points, hand_points, axis, margin):
    """Translate the grasp along a body-local axis using measured geometry."""
    axis = np.asarray(axis, dtype=float)
    if (axis.shape != (3,) or not np.isfinite(axis).all()
            or not np.isclose(np.linalg.norm(axis), 1., atol=1e-9, rtol=0.)):
        raise ValueError('RELEASE_GRASP_UNIT_AXIS_REQUIRED')
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError('RELEASE_GRASP_POSITIVE_MARGIN_REQUIRED')
    arrays = [np.asarray(points, dtype=float) for points in (object_points, hand_points)]
    if any(p.ndim != 2 or p.shape[1] != 3 or len(p) < 4
           or not np.isfinite(p).all() for p in arrays):
        raise ValueError('RELEASE_GRASP_COLLISION_GEOMETRY_REQUIRED')
    object_points, hand_points = arrays
    T_BC, T_CG = targets['T_BC'], targets['T_CG']
    T_BG = T_BC @ T_CG
    hand_in_body = hand_points @ T_BG[:3, :3].T + T_BG[:3, 3]
    object_edge = float(np.max(object_points @ axis))
    hand_edge = float(np.min(hand_in_body @ axis))
    shift = max(0., object_edge + margin - hand_edge)
    body_rotation = targets['T_WC'][:3, :3] @ T_BC[:3, :3].T
    world_delta = body_rotation @ (axis * shift)
    result = {name: value.copy() for name, value in targets.items()}
    for name in ('PRE_GRASP', 'GRASP', 'LIFT'):
        result[name][:3, 3] += world_delta
    result['T_CG'][:3, 3] += T_BC[:3, :3].T @ (axis * shift)
    return result, {
        'source': 'PUBLIC_OPEN_HAND_AND_OBJECT_COLLISION_GEOMETRY',
        'axis_in_object': axis.tolist(), 'clearance_m': margin,
        'object_support_plane_m': object_edge, 'original_open_hand_plane_m': hand_edge,
        'grasp_shift_m': shift, 'proposed_open_clearance_m': hand_edge + shift - object_edge,
        'resolved_grasp_in_center': result['T_CG'].tolist(),
        'validation_scope': 'GRASP_PROPOSAL_REQUIRES_PHYSICAL_VALIDATION',
    }


def resolve_release_clearance_targets(context, targets):
    c = context
    axis = getattr(c.recipe, 'open_hand_clearance_axis', None)
    if axis is None:
        return targets, None
    from tuj.m5_motion.tool_use_journal import _geom_local_points
    from .open_geometry import open_joint_positions

    probe = c.mj.MjData(c.model)
    probe.qpos[:] = c.data.qpos
    for name, value in open_joint_positions(c).items():
        probe.qpos[c.model.jnt_qposadr[c.model.joint(name).id]] = value
    c.mj.mj_forward(c.model, probe)

    def points_in_frame(data, gids, position, rotation):
        parts = []
        for gid in sorted(gids):
            if not (c.model.geom_contype[gid] or c.model.geom_conaffinity[gid]):
                continue
            local = _geom_local_points(c.model, gid)
            if local is None:
                raise ValueError('RELEASE_GRASP_UNSUPPORTED_COLLISION_GEOMETRY')
            world = local @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            parts.append((world - position) @ rotation)
        if not parts:
            raise ValueError('RELEASE_GRASP_COLLISION_GEOMETRY_REQUIRED')
        return np.concatenate(parts)

    body = c.body_pose()
    object_points = points_in_frame(c.data, c.object_geoms, body[:3, 3], body[:3, :3])
    hand_points = points_in_frame(probe, c.gripper_geoms, probe.site_xpos[c.site_id],
                                 probe.site_xmat[c.site_id].reshape(3, 3))
    # This is hand/object relative clearance. Reserve at least the allowed
    # physical grasp slip as well as the request's existing collision margin.
    margin = max(c.recipe.maximum_slip_m, getattr(c, 'request_collision_margin_m', 0.))
    return shift_targets_for_open_clearance(targets, object_points, hand_points, axis, margin)
