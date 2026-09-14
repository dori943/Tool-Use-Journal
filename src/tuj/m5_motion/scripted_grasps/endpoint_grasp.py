"""Support clearance for explicitly requested contact-region endpoint grasps."""
import math
import numpy as np


def lift_targets_above_support(targets, hand_points_in_grip, support_top, margin):
    points = np.asarray(hand_points_in_grip, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all()):
        raise ValueError('ENDPOINT_GRASP_HAND_GEOMETRY_REQUIRED')
    if not math.isfinite(support_top) or not math.isfinite(margin) or margin <= 0:
        raise ValueError('ENDPOINT_GRASP_FINITE_SUPPORT_AND_MARGIN_REQUIRED')
    grasp = targets['GRASP']
    minimum = float((points @ grasp[:3, :3].T + grasp[:3, 3])[:, 2].min())
    rise = max(0., support_top + margin - minimum)
    delta = np.array([0., 0., rise])
    result = {name: value.copy() for name, value in targets.items()}
    for name in ('GRASP', 'PRE_GRASP', 'LIFT'):
        result[name][:3, 3] += delta
    result['T_CG'][:3, 3] += targets['T_WC'][:3, :3].T @ delta
    return result, {'source': 'LIVE_PRESHAPED_HAND_AND_SUPPORT_GEOMETRY',
                    'support_top_world_z_m': support_top, 'clearance_m': margin,
                    'original_hand_min_world_z_m': minimum, 'world_up_translation_m': rise,
                    'validation_scope': 'PROPOSAL_REQUIRES_PHYSICAL_GRASP_VALIDATION'}


def resolve_endpoint_support_clearance(c, targets):
    from tuj.m5_motion.tool_use_journal import _geom_local_points
    grip = c.grip_pose()
    points = []
    for gid in sorted(c.gripper_geoms):
        if not (c.model.geom_contype[gid] or c.model.geom_conaffinity[gid]):
            continue
        local = _geom_local_points(c.model, gid)
        if local is None:
            raise ValueError('ENDPOINT_GRASP_UNSUPPORTED_HAND_GEOMETRY')
        world = local @ c.data.geom_xmat[gid].reshape(3, 3).T + c.data.geom_xpos[gid]
        points.append((world - grip[:3, 3]) @ grip[:3, :3])
    if not points:
        raise ValueError('ENDPOINT_GRASP_HAND_GEOMETRY_REQUIRED')
    if not math.isfinite(c.request_collision_margin_m) or c.request_collision_margin_m < 0:
        raise ValueError('ENDPOINT_GRASP_INVALID_REQUEST_MARGIN')
    margin = max(c.recipe.maximum_slip_m, c.request_collision_margin_m)
    result, evidence = lift_targets_above_support(targets, np.concatenate(points), c.support_top_z, margin)
    evidence.update(policy=c.recipe.contact_region_endpoint_policy,
                    long_axis=int(np.argmax(c.local_size)))
    return result, evidence
