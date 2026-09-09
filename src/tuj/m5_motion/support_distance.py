"""Native signed clearance for selected rigid objects at snapshot poses."""
from collections.abc import Mapping

import mujoco
import numpy as np


def refine_support_request(request, compiler):
    """Refine already selected support evidence without changing its policy."""
    metadata = request.task.metadata
    selectors = metadata.get('support_collision_selectors')
    target = request.task.goal.target_object_id
    measure = getattr(compiler, 'initial_object_clearance', None)
    if (not target or not isinstance(selectors, list) or not selectors
            or not all(isinstance(value, str) for value in selectors)
            or not callable(measure)):
        return request
    native_contacts = {}
    contact_query = getattr(compiler, 'initial_static_support_contacts', None)
    if (metadata.get('support_collision_policy') == 'AUTO_INITIAL_SUPPORT_V1'
            and metadata.get('support_collision_detection_source') == 'world.obstacles.aabb'
            and callable(contact_query)):
        native_contacts = contact_query(request.world, target, selectors)
    # Keep the primary overlap policy. Extend it only when every original
    # selector and each additional coplanar fragment has native support contact.
    expanded = bool(native_contacts) and all(s in native_contacts for s in selectors)
    if expanded:
        selectors = [*selectors, *sorted(set(native_contacts) - set(selectors))]
        distances = [native_contacts[s] for s in selectors]
    else:
        distances = [measure(request.world.objects, target, selector) for selector in selectors]
    if any(value is None for value in distances):
        return request
    if not all(np.isfinite(value) for value in distances):
        raise ValueError('non-finite native support clearance')
    refined = request.model_copy(deep=True)
    output = refined.task.metadata
    output['support_bound_clearance_m'] = metadata.get('support_initial_clearance_m')
    output['support_initial_clearance_m'] = min(distances)
    output['support_clearance_source'] = 'mujoco.native_geom_distance'
    if expanded:
        output['support_collision_selectors'] = selectors
        output['support_native_contact_distances_m'] = dict(native_contacts)
    if output.get('support_collision_policy') == 'AUTO_INITIAL_SUPPORT_V1':
        tolerance = output.get('support_contact_tolerance_m', request.constraints.collision_margin_m)
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('support contact tolerance must be finite and nonnegative')
        if min(distances) > tolerance:
            output['support_collision_selectors'] = []
    return refined


def _place_free_object(model, data, body_names, objects, object_id):
    """Place a bound free object in scratch data and return its collision geoms."""
    record = objects.get(object_id)
    if object_id not in body_names or not isinstance(record, Mapping):
        return None
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_names[object_id])
    if body < 0:
        return None
    joints = np.flatnonzero(model.jnt_bodyid == body)
    if len(joints) != 1 or model.jnt_type[joints[0]] != mujoco.mjtJoint.mjJNT_FREE:
        return None
    pose = record.get('pose')
    if not isinstance(pose, Mapping):
        return None
    position = np.asarray(pose.get('position_m'), dtype=float)
    quat = np.asarray(pose.get('orientation_xyzw'), dtype=float)
    if (pose.get('frame_id') != 'world' or position.shape != (3,) or quat.shape != (4,)
            or not np.isfinite(position).all() or not np.isfinite(quat).all()
            or np.linalg.norm(quat) < 1e-12):
        raise ValueError('support clearance requires finite world poses')
    quat = quat / np.linalg.norm(quat)
    address = model.jnt_qposadr[joints[0]]
    data.qpos[address:address + 7] = [*position, quat[3], *quat[:3]]
    descendants = {body}
    for candidate in range(body + 1, model.nbody):
        if int(model.body_parentid[candidate]) in descendants:
            descendants.add(candidate)
    return [i for i in range(model.ngeom)
            if int(model.geom_bodyid[i]) in descendants
            and (model.geom_contype[i] or model.geom_conaffinity[i])]


def object_pair_clearance(model, baseline_qpos, body_names, objects, first, second):
    """Return exact minimum geom distance, or None when no exact binding exists.

    Work is confined to a new MjData. Bounds remain a conservative fallback;
    an unsupported binding must never become a fictitious zero penetration.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = baseline_qpos
    groups = []
    supported = {int(getattr(mujoco.mjtGeom, 'mjGEOM_' + kind))
                 for kind in ('SPHERE', 'CAPSULE', 'ELLIPSOID', 'CYLINDER', 'BOX', 'MESH')}
    for object_id in (first, second):
        geoms = _place_free_object(model, data, body_names, objects, object_id)
        if not geoms or any(int(model.geom_type[i]) not in supported for i in geoms):
            return None
        groups.append(geoms)
    mujoco.mj_forward(model, data)
    search_limit = 0.1
    distance = min(float(mujoco.mj_geomDistance(model, data, a, b, search_limit, None))
                   for a in groups[0] for b in groups[1])
    if not np.isfinite(distance):
        raise ValueError('non-finite native support clearance')
    return distance if distance < search_limit else None
