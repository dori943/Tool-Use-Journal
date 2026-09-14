"""Identify only measured object contacts that support it against gravity."""
import numpy as np


def measured_support_geoms(c):
    gravity = np.asarray(c.model.opt.gravity, dtype=float)
    length = np.linalg.norm(gravity)
    if length == 0.:
        return set()
    up = -gravity / length
    body = c.body_pose()
    center = body[:3, 3] + body[:3, :3] @ c.center_in_body
    result = set()
    for contact in c.data.contact[:c.data.ncon]:
        a, b = int(contact.geom1), int(contact.geom2)
        if contact.dist > 0. or ((a in c.object_geoms) == (b in c.object_geoms)):
            continue
        other = b if a in c.object_geoms else a
        if other in c.robot_geoms or other in c.gripper_geoms:
            continue
        # A fixed child of a moving body is not a static support.
        if int(c.model.body_weldid[int(c.model.geom_bodyid[other])]) != 0:
            continue
        normal = np.asarray(contact.frame).reshape(3, 3)[0]
        if a in c.object_geoms:
            normal = -normal
        if float(normal @ up) <= 1e-6 or float((center - contact.pos) @ up) <= 0.:
            continue
        result.add(other)
    return result
