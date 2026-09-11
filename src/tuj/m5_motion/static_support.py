"""Measured support contacts across coplanar fixed box surface fragments."""
from collections.abc import Mapping

import mujoco
import numpy as np

from .support_distance import _place_free_object


def coplanar_static_support_contacts(model, baseline_qpos, body_names, world, target, primary):
    """Return exact current support distances without mutating the live model.

    Only horizontal fixed boxes represented by snapshot obstacles participate.
    A primary support is required; this does not invent support from proximity.
    Penetration is returned unmodified so the existing policy's hard limit
    rejects overpenetration, including on an additional support fragment.
    """
    gravity = np.asarray(model.opt.gravity, dtype=float)
    if np.linalg.norm(gravity) == 0 or not np.allclose(
            -gravity / np.linalg.norm(gravity), [0., 0., 1.], atol=1e-8, rtol=0):
        return {}
    data = mujoco.MjData(model)
    data.qpos[:] = baseline_qpos
    target_geoms = _place_free_object(model, data, body_names, world.objects, target)
    if not target_geoms:
        return {}
    mujoco.mj_forward(model, data)
    boxes = {}
    for obstacle in world.obstacles:
        if not isinstance(obstacle, Mapping) or obstacle.get('collision_enabled_in_source') is False:
            continue
        name = obstacle.get('obstacle_id')
        if not isinstance(name, str):
            continue
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0 or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        body = int(model.geom_bodyid[gid])
        movable = False
        while body:
            movable |= bool(model.body_jntnum[body])
            body = int(model.body_parentid[body])
        if movable:
            continue
        rotation = data.geom_xmat[gid].reshape(3, 3)
        if not np.isclose(abs(rotation[2, 2]), 1., atol=1e-8, rtol=0):
            continue
        half = np.abs(rotation) @ model.geom_size[gid]
        boxes[name] = (gid, data.geom_xpos[gid] - half, data.geom_xpos[gid] + half)
    if not primary or any(s not in boxes for s in primary):
        return {}
    level = boxes[primary[0]][2][2]
    if any(not np.isclose(boxes[s][2][2], level, atol=1e-8, rtol=0) for s in primary):
        return {}
    lower, upper = [], []
    for gid in target_geoms:
        rotation = data.geom_xmat[gid].reshape(3, 3)
        center = data.geom_xpos[gid] + rotation @ model.geom_aabb[gid, :3]
        half = np.abs(rotation) @ model.geom_aabb[gid, 3:]
        lower.append(center - half)
        upper.append(center + half)
    lo, hi = np.min(lower, axis=0), np.max(upper, axis=0)
    candidates = {gid: name for name, (gid, bmin, bmax) in boxes.items()
                  if np.isclose(bmax[2], level, atol=1e-8, rtol=0)
                  and np.all(np.minimum(hi[:2], bmax[:2]) > np.maximum(lo[:2], bmin[:2]))}
    target_geoms = set(target_geoms)
    contacts = {}
    for contact in data.contact:
        first, second = int(contact.geom1), int(contact.geom2)
        if first in candidates and second in target_geoms:
            support, sign = first, 1
        elif second in candidates and first in target_geoms:
            support, sign = second, -1
        else:
            continue
        # mjContact.frame[:3] points geom1 -> geom2. This tolerance only
        # handles round-off in a horizontal normal, not a collision margin.
        normal = sign * np.asarray(contact.frame[:3])
        if contact.dist > 0 or not np.allclose(normal, [0., 0., 1.], atol=1e-8, rtol=0):
            continue
        name = candidates[support]
        contacts[name] = min(contacts.get(name, float('inf')), float(contact.dist))
    if not all(s in contacts for s in primary):
        return {}
    # Contact manifolds can report a different depth than the exact pair
    # distance (notably box-box contacts). Preserve the policy's geometric
    # penetration limit by measuring every target geom against the surface.
    distances = {}
    for name in contacts:
        gid = boxes[name][0]
        distance = min(float(mujoco.mj_geomDistance(model, data, g, gid, .1, None))
                       for g in target_geoms)
        if np.isfinite(distance) and distance <= 0:
            distances[name] = distance
    return distances if all(s in distances for s in primary) else {}
