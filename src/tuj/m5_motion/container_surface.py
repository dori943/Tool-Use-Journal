"""Verify a solid cover on a container opening, separately from its contents."""
from itertools import product
import math

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .push_to_region import target_fully_inside_region
from .task_semantics import task_operation


def is_container_surface_task(task, objects):
    region = objects.get(task.goal.target_region_id, {})
    return (task_operation(task) == 'PLACE_ON'
            and region.get('packing_metadata', {}).get('kind') == 'CONTAINER')


def packable_ids(objects):
    return sorted(name for name, record in objects.items()
                  if record.get('packing_metadata', {}).get('kind') == 'PACKABLE_OBJECT')


def _array(value, shape):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('CONTAINER_SURFACE_GEOMETRY_REQUIRED')
    return result


def _pose(record):
    pose = record.get('pose', {})
    if pose.get('frame_id', 'world') != 'world':
        raise ValueError('CONTAINER_SURFACE_WORLD_POSE_REQUIRED')
    return (_array(pose.get('position_m'), (3,)),
            Rotation.from_quat(_array(pose.get('orientation_xyzw'), (4,))).as_matrix())


def cover_geometry(objects, target_id, region_id, position_tolerance_m, orientation_tolerance_rad):
    """Require the solid bottom face to cover the opening and rest at rim level.

    A convex hull of arbitrary mesh/bbox points would fill holes incorrectly.
    Only an actual single compiled BOX supplies the solid face used here.
    """
    if any(not math.isfinite(v) or v <= 0 for v in (position_tolerance_m, orientation_tolerance_rad)):
        raise ValueError('CONTAINER_SURFACE_TOLERANCE_REQUIRED')
    target, region = objects[target_id], objects[region_id]
    geometry = target.get('solid_box_geometry', {})
    if geometry.get('source') != 'MUJOCO_BOX':
        raise ValueError('CONTAINER_SURFACE_SOLID_BOX_REQUIRED')
    half = _array(geometry.get('half_size_m'), (3,))
    center = _array(geometry.get('center_in_body_m'), (3,))
    local_rotation = _array(geometry.get('rotation_in_body'), (3, 3))
    if (np.any(half <= 0) or not np.allclose(local_rotation.T @ local_rotation, np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(local_rotation), 1., atol=1e-8)):
        raise ValueError('CONTAINER_SURFACE_INVALID_SOLID_BOX')
    target_p, target_r = _pose(target)
    region_p, region_r = _pose(region)
    if not np.allclose(region_r[:, 2], [0., 0., 1.], atol=1e-8, rtol=0):
        raise ValueError('CONTAINER_SURFACE_UPRIGHT_REGION_REQUIRED')
    metadata = region.get('packing_metadata', {})
    dimensions = _array(metadata.get('interior_dimensions_m'), (3,))
    interior_center = _array(metadata.get('interior_center_m'), (3,))
    rim = float(metadata['opening_top_z_m'])
    if np.any(dimensions <= 0) or not math.isfinite(rim):
        raise ValueError('CONTAINER_SURFACE_OPENING_REQUIRED')
    rotation = region_r.T @ target_r @ local_rotation
    position = region_r.T @ (target_p + target_r @ center - region_p)
    axis = int(np.argmax(np.abs(rotation[2])))
    normal_angle = float(np.arccos(np.clip(abs(rotation[2, axis]), 0., 1.)))
    other = [i for i in range(3) if i != axis]
    face = []
    for signs in product((-1., 1.), repeat=2):
        corner = np.zeros(3)
        corner[axis] = -np.sign(rotation[2, axis]) * half[axis]
        corner[other] = np.asarray(signs) * half[other]
        face.append(position + rotation @ corner)
    face = np.asarray(face)
    opening = interior_center[:2] + np.asarray(list(product((-1., 1.), repeat=2))) * dimensions[:2] / 2.
    hull = ConvexHull(face[:, :2])
    # Only floating-point roundoff is tolerated horizontally, not overhang.
    numerical = 64. * np.finfo(float).eps * max(1., float(np.max(np.abs(face))))
    covered = bool(np.all(opening @ hull.equations[:, :2].T + hull.equations[:, 2] <= numerical))
    bottom_gap = float(np.max(np.abs(face[:, 2] - rim)))
    return {'succeeded': covered and bottom_gap <= position_tolerance_m
            and normal_angle <= orientation_tolerance_rad,
            'target_id': target_id, 'region_id': region_id,
            'opening_covered': covered, 'max_bottom_rim_error_m': bottom_gap,
            'face_normal_error_rad': normal_angle,
            'bottom_face_in_region_m': face.tolist(), 'source': 'MEASURED_SOLID_BOX_FACE'}


def measured_surface_support(model, data, target_body, region_body, rim_world_z,
                             position_tolerance_m, orientation_tolerance_rad):
    """Count loaded, upward container contacts at the rim; never infer contact."""
    import mujoco
    def under(body, root):
        while body and body != root:
            body = int(model.body_parentid[body])
        return body == root
    count, load = 0, 0.
    for i, contact in enumerate(data.contact[:data.ncon]):
        a, b = int(contact.geom1), int(contact.geom2)
        if a < 0 or b < 0 or contact.dist > 0:
            continue
        ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
        if under(ba, region_body) and under(bb, target_body):
            sign = 1.
        elif under(bb, region_body) and under(ba, target_body):
            sign = -1.
        else:
            continue
        upward = sign * float(contact.frame[2])
        if (upward < math.cos(orientation_tolerance_rad)
                or abs(float(contact.pos[2]) - rim_world_z) > position_tolerance_m):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        if force[0] > 0:
            count += 1
            load += float(force[0]) * upward
    return {'contact_count': count, 'upward_force_n': load}


def surface_report_result(request, report, world):
    """Use fresh geometry and the final physical settle evidence for this request."""
    target, region = request.task.goal.target_object_id, request.task.goal.target_region_id
    geometry = cover_geometry(world.objects, target, region,
        request.constraints.position_tolerance_m, request.constraints.orientation_tolerance_rad)
    required = packable_ids(request.world.objects)
    outside = [name for name in required if not target_fully_inside_region(
        world, target_id=name, region_id=region, include_vertical=True)]
    tracking = report.metadata.get('segment_tracking', [])
    custom = (tracking[-1].get('custom_settle') or {}) if tracking else {}
    settle = custom.get('container_settle', {})
    support = settle.get('surface_support', {})
    verified = (settle.get('request_id') == request.request_id
        and settle.get('surface_target_id') == target and settle.get('region_id') == region
        and settle.get('succeeded') is True and support.get('contact_count', 0) > 0
        and support.get('upward_force_n', 0) > 0)
    return {'succeeded': geometry['succeeded'] and not outside and verified,
            'cover': geometry, 'outside_target_ids': outside,
            'required_inside_target_ids': required, 'physical_settle_verified': verified}
