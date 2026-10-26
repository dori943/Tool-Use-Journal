"""Geometry and measured attachment grounded compression candidates.

No task or object name selects coordinates. A homogeneous isochoric uniaxial
plastic estimate supplies a trial compression depth, never a success result.
Actual contact and residual deformation must be evaluated after unloading.
"""
import hashlib
import math
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from .geometry import tool_rotation_from_axis
from .schema import (ArtifactProvenance, KeyframePlanArtifact, KeyframePlanCandidate,
                     RelativeKeyframeSpec, StrategyGenerationProvenance)


def is_flatten_contact(task):
    return task.contact is not None and task.contact.primitive.lower() == 'flatten'


def compression_trial_height(initial_height, configuration):
    low, high = configuration['height_reduction_fraction']
    if not 0 < low <= high < 1:
        raise ValueError('invalid requested thickness reduction')
    material = configuration['material']
    # Uniaxial isochoric J2 elastic log-strain at yield = sigma_y / (3 mu).
    # Use this estimate only to propose a physical experiment; rebound is measured.
    elastic_recovery = math.exp(material['yield_stress_pa'] / (3 * material['shear_modulus_pa']))
    return initial_height * (1 - (low + high) / 2) / elastic_recovery


def flattening_outcome(record):
    metrics = record.get('deformable_metrics') or {}
    config = record.get('flattening_configuration') or {}
    contact = metrics.get('tool_contact') or {}
    if not metrics or not config or not contact:
        return False, {'reason': 'missing measured deformation or tool contact'}
    ratio = 1 - metrics['height_m'] / metrics['initial_height_m']
    low, high = config['height_reduction_fraction']
    checks = {'thickness_reduction': low <= ratio <= high,
              'tool_contact': contact.get('impulse_ns', 0) > 0,
              'unloaded': contact.get('combined_current_force_n', contact.get('current_force_n', 0)) == 0
                          and contact.get('unloaded_duration_s', 0) + 1e-9 >= config['unloaded_observation_s'],
              'volume_preserved': abs(metrics['volume_ratio'] - 1) <= config['volume_tolerance_fraction'],
              'force_within_limit': contact.get('combined_peak_force_n', contact.get('peak_force_n', float('inf'))) <= contact.get('force_limit_n', 0),
              'tool_caused_reduction': metrics['height_m'] < contact['start_height_m']}
    return all(checks.values()), {'checks': checks, 'height_reduction_fraction': ratio,
                                'metrics': metrics}


def _matrix(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose['orientation_xyzw']).as_matrix()
    result[:3, 3] = pose['position_m']
    return result


def contact_face_center(points, normal_axis):
    """Center the broad observed footprint, rather than a narrow tool handle.

    The largest inscribed disk of the projected collision hull proposes a
    contact patch. The hull is only a planning approximation; executed contact
    and deformation remain the source of success evidence. For uniform faces,
    choose the disk center nearest the bounding-box center deterministically.
    """
    points = np.asarray(points, dtype=float)
    axes = [axis for axis in range(3) if axis != normal_axis]
    projected = points[:, axes]
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    try:
        hull = ConvexHull(projected)
    except QhullError as error:
        raise ValueError('flatten requires a nondegenerate observed tool face') from error
    normals, limits = hull.equations[:, :2], -hull.equations[:, 2]
    disk = linprog([0., 0., -1.],
        A_ub=np.column_stack((normals, np.ones(len(normals)))), b_ub=limits,
        bounds=[(None, None), (None, None), (0., None)], method='highs')
    if not disk.success or disk.x[2] <= 0:
        raise ValueError('flatten could not locate a supported tool contact patch')
    radius = disk.x[2]
    preferred = center[axes]
    nearest = minimize(lambda x: np.sum((x - preferred) ** 2), disk.x[:2],
        jac=lambda x: 2 * (x - preferred), method='SLSQP',
        constraints=[{'type': 'ineq', 'fun': lambda x: limits - radius - normals @ x,
                      'jac': lambda x: -normals}], options={'ftol': 1e-12})
    if not nearest.success or np.max(normals @ nearest.x + radius - limits) > 1e-8:
        raise ValueError('flatten contact-patch center optimization failed')
    center[axes] = nearest.x
    return center


def measured_contact_face(triangles, center, normal_axis, side):
    """Intersect the proposed patch with an actual collision face on one side.

    A projected hull may bridge a gap between components. Only a triangle
    containing the patch center can supply its height and normal; a hull alone
    cannot certify either. Face winding is irrelevant to the outward side.
    """
    axes = [axis for axis in range(3) if axis != normal_axis]
    hits = []
    for triangle in triangles:
        edges = triangle[1:] - triangle[0]
        projected = edges[:, axes].T
        area = abs(float(np.linalg.det(projected)))
        if area <= 64 * np.finfo(float).eps:
            continue
        coordinates = np.linalg.solve(projected, center[axes] - triangle[0, axes])
        barycentric = np.r_[1. - coordinates.sum(), coordinates]
        if np.min(barycentric) < -1e-9:
            continue
        point = barycentric @ triangle
        normal = np.cross(edges[0], edges[1])
        normal /= np.linalg.norm(normal)
        if normal[normal_axis] * side < 0:
            normal *= -1
        hits.append((point, normal, area))
    if not hits:
        raise ValueError('flatten patch has no measured collision face')
    outer = max(side * point[normal_axis] for point, _, _ in hits)
    hits = [hit for hit in hits if outer - side * hit[0][normal_axis] <= 1e-9]
    point, normal, _ = max(hits, key=lambda hit: (hit[2], *hit[1]))
    return point, normal


class FlattenContactProvider:
    def generate(self, request):
        task, world = request.task, request.world
        if len(task.target_ids) != 1 or not task.tool:
            raise ValueError('flatten requires one deformable target and a held tool')
        target_id = task.target_ids[0]
        target, tool = world.objects[target_id], world.objects[task.tool]
        metrics, config = target['deformable_metrics'], target['flattening_configuration']
        attachment = world.metadata['attached_object_transforms'][task.tool]
        grasp = _matrix({'position_m': attachment['position_in_reference_m'],
                         'orientation_xyzw': attachment['orientation_in_reference_xyzw']})
        points = np.asarray(tool['collision_points_m'], dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError('finite observed tool collision points required')
        triangles = np.asarray(tool.get('collision_surface_triangles_m'), dtype=float)
        if (tool.get('collision_surface_source') != 'COMPILED_MUJOCO_COLLISION_FACES'
                or triangles.ndim != 3 or triangles.shape[1:] != (3, 3)
                or not len(triangles) or not np.isfinite(triangles).all()):
            raise ValueError('flatten requires measured collision surface triangles')
        size = np.ptp(points, axis=0)
        long_axis = int(np.argmax(size))
        normal_axis = min((i for i in range(3) if i != long_axis), key=lambda i: size[i])
        center = contact_face_center(points, normal_axis)
        current = _matrix(tool['pose'])[:3, :3]
        rotations = []
        for direction in ([1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]):
            for sign in (-1., 1.):
                face, normal = measured_contact_face(triangles, center, normal_axis, sign)
                along = np.eye(3)[long_axis] - normal[long_axis] * normal
                along /= np.linalg.norm(along)
                local_basis = np.column_stack((along, np.cross(normal, along), normal))
                down = np.array([0., 0., -1.])
                world_basis = np.column_stack((direction, np.cross(down, direction), down))
                rotation = world_basis @ local_basis.T
                rotations.append((rotation, face, normal))
        rotations.sort(key=lambda item: Rotation.from_matrix(current.T @ item[0]).magnitude())
        lower, upper = np.array(metrics['bounds_min_m']), np.array(metrics['bounds_max_m'])
        target_center = (lower + upper) / 2
        compressed_height = compression_trial_height(metrics['initial_height_m'], config)
        margin = request.constraints.collision_margin_m
        if compressed_height <= margin:
            raise ValueError('compression trial would violate support clearance')
        digest = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        candidates = []
        for index, (rotation, face, normal) in enumerate(rotations[:4]):
            legacy_up = np.eye(3)[normal_axis] * -np.sign(normal[normal_axis])
            legacy_height_error = float((face - center) @ legacy_up
                - np.min((points - center) @ legacy_up))
            hover = upper[2] + max(size) + margin
            keys = []
            for stage, work_z in [('hover', hover), ('engage', upper[2]),
                                  ('press', lower[2] + compressed_height), ('unload', hover)]:
                tool_pose = np.eye(4)
                tool_pose[:3, :3] = rotation
                contact_center = np.array([target_center[0], target_center[1], work_z])
                tool_pose[:3, 3] = contact_center - rotation @ face
                eef = tool_pose @ np.linalg.inv(grasp)
                frame = f'flatten-frame:{digest[:12]}:{index}:{stage}'
                world.objects[frame] = {'pose': {'frame_id': 'world',
                    'position_m': eef[:3, 3].tolist(),
                    'orientation_xyzw': Rotation.from_matrix(eef[:3, :3]).as_quat().tolist()},
                    'virtual_reference_frame': True, 'anchors': {'center': [0., 0., 0.]}}
                base = tool_rotation_from_axis(eef[:3, 2], 0.)
                roll = math.atan2(float(eef[:3, 0] @ base[:, 1]), float(eef[:3, 0] @ base[:, 0]))
                keys.append(RelativeKeyframeSpec(keyframe_id=f'{task.subgoal_id}:{index}:{stage}',
                    keyframe_type='TRANSFER' if stage == 'hover' else 'RETREAT' if stage == 'unload' else 'CUSTOM',
                    frame_ref='object:' + frame, anchor='center', approach_axis_xyz=(0, 0, 1),
                    tool_axis_to_align='+z', roll_rad=roll,
                    planner='SAMPLING_BASED' if stage == 'hover' else 'CARTESIAN',
                    metadata={'flatten_stage': stage, 'target_id': target_id,
                              'hold_duration_after_s': config['unloaded_observation_s'] if stage in ('press', 'unload') else 0.}))
            candidates.append(KeyframePlanCandidate(strategy_id=f'flatten:{index}', keyframes=keys,
                rationale='Measured tool geometry and actual grasp transform; trial depth from constitutive model.',
                provenance=StrategyGenerationProvenance(generator_kind='TASK_GEOMETRY',
                    generator_id='CONTACT_COMPRESSION_V3', input_hash=digest, attempt_index=1),
                metadata={'trial_compressed_height_m': compressed_height, 'target_id': target_id,
                          'contact_face_center_local_m': face.tolist(),
                          'contact_face_normal_local': normal.tolist(),
                          'contact_face_source': tool['collision_surface_source'],
                          'legacy_contact_face_height_error_m': legacy_height_error}))
        return KeyframePlanArtifact(artifact_id='flatten:' + digest[:20],
            provenance=ArtifactProvenance(artifact_id='flatten-artifact:' + digest[:20],
                artifact_type='KeyframePlanArtifact', produced_by='MOTION_PLANNER', invocation_id=digest[:20],
                input_artifact_ids=[request.provenance.artifact_id]),
            scene_signature=world.scene.signature, subgoal_id=task.subgoal_id, candidates=candidates)
