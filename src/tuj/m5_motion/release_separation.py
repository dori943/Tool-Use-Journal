"""Measure exact held-geometry pairs for a nonpenetrating withdrawal."""
import math

import mujoco
import numpy as np

from .attachment_retarget import attachment_transform, object_pose_for_end_effector_pose
from .geometry import matrix_quaternion_xyzw
from .schema import Pose
from .support_distance import _place_free_object


def release_geometry_pairs(model, baseline_qpos, body_names, mounted_body, world, target, margin):
    if not math.isfinite(margin) or margin <= 0:
        return []
    if target != world.robot_state.attached_object_id:
        return []
    transform = attachment_transform(world, target)
    data = mujoco.MjData(model)
    data.qpos[:] = baseline_qpos
    for name, value in zip(world.robot_state.joint_names, world.robot_state.joint_positions_rad):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            return []
        data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)
    kind = mujoco.mjtObj.mjOBJ_SITE if transform.reference_kind == 'site' else mujoco.mjtObj.mjOBJ_BODY
    rid = mujoco.mj_name2id(model, kind, transform.reference_name)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mounted_body)
    if rid < 0 or root < 0:
        return []
    pos = data.site_xpos[rid] if transform.reference_kind == 'site' else data.xpos[rid]
    rot = data.site_xmat[rid] if transform.reference_kind == 'site' else data.xmat[rid]
    object_pose = object_pose_for_end_effector_pose(Pose(
        frame_id='world', position_m=tuple(pos),
        orientation_xyzw=matrix_quaternion_xyzw(rot.reshape(3, 3)),
    ), transform)
    objects = dict(world.objects)
    objects[target] = {**dict(objects[target]), 'pose': object_pose.model_dump(mode='json')}
    target_geoms = _place_free_object(model, data, body_names, objects, target)
    mujoco.mj_forward(model, data)
    def mounted(gid):
        bid = int(model.geom_bodyid[gid])
        while bid:
            if bid == root:
                return True
            bid = int(model.body_parentid[bid])
        return False
    pairs = {}
    for ee in range(model.ngeom):
        if not mounted(ee) or not (model.geom_contype[ee] or model.geom_conaffinity[ee]):
            continue
        for obj in target_geoms:
            distance = float(mujoco.mj_geomDistance(model, data, ee, obj, margin, None))
            if math.isfinite(distance) and 0 <= distance < margin:
                a, b = model.geom(ee).name, model.geom(obj).name
                if a and b:
                    # Mesh decomposition is not a contact semantic boundary.
                    # Identify the close EE surface, then require nonpenetration
                    # against every part of this same held object on withdrawal.
                    record = pairs.setdefault(a, {
                        'selectors': [a, target], 'initial_distance_m': distance,
                        'minimum_distance_m': 0.0, 'measured_target_geoms': [],
                    })
                    record['initial_distance_m'] = min(record['initial_distance_m'], distance)
                    record['measured_target_geoms'].append(b)
    return list(pairs.values())


def bind_release_separation(compiler, request, artifact, contexts, target):
    """Scope measured surface/object nonpenetration to the first withdrawal."""
    from .schema import KeyframeEventType, KeyframeType
    from .tool_use_journal_planning import ToolUseJournalCollisionBindingError

    if request.world.robot_state.attached_object_id != target:
        return
    measure = getattr(compiler, 'initial_release_geometry_pairs', None)
    pairs = measure(request.world, target, request.constraints.collision_margin_m) if callable(measure) else []
    if not pairs:
        return
    for candidate in artifact.candidates:
        place = next(k for k in candidate.keyframes if KeyframeEventType.DETACH_OBJECT in k.events_after)
        detached_id = place.collision_context_after_events_id
        detached = contexts[detached_id]
        # Preserve pre-existing explicit release policies.
        if detached.allowed_collision_pairs:
            continue
        index = candidate.keyframes.index(place) + 1
        if index >= len(candidate.keyframes):
            raise ToolUseJournalCollisionBindingError('bounded release has no RETREAT')
        retreat = candidate.keyframes[index]
        if retreat.keyframe_type is not KeyframeType.RETREAT:
            raise ToolUseJournalCollisionBindingError('bounded release requires an immediate RETREAT')
        release_id = detached_id.replace('object-detached:', 'object-release-separation:', 1)
        contexts[release_id] = detached.model_copy(update={
            'context_id': release_id,
            'allowed_collision_pairs': [tuple(item['selectors']) for item in pairs],
            'metadata': {'bounded_collision_allowances': pairs,
                         'release_separation': 'MEASURED_NONPENETRATING_FIRST_RETREAT'},
        })
        place.collision_context_after_events_id = release_id
        retreat.collision_context_id = release_id
        retreat.collision_context_after_events_id = detached_id
        retreat.metadata['validate_endpoint_after_context'] = True
