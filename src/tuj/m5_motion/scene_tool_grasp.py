"""Contact candidates for free scene tools; rack docking remains untouched.

A scene tool is a free object, even when the symbolic action is PICK_TOOL.
These candidates remove model-proposed airborne GRASP offsets, align a parallel
jaw across the tool, and wait for closure before requesting normal attachment.
IK, collisions, runtime attachment distance and all task criteria remain final
validators. Candidate generation itself is never evidence of a physical grasp.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import numpy as np

from .geometry import quaternion_matrix_xyzw, tool_rotation_from_axis
from .grasp_geometry import support_clearance_context
from .profiles import PhysicalGraspProfile
from .schema import KeyframeEventType, KeyframeType
from .task_semantics import task_operation

SOURCE = 'FREE_SCENE_TOOL_CONTACT_V1'
ANCHOR = 'scene_tool_contact'


def is_free_scene_tool_pick(request):
    if task_operation(request.task) != 'PICK_TOOL':
        return False
    target = request.task.goal.target_object_id or request.task.tool
    record = request.world.objects.get(target)
    if target in request.world.rack:
        return False
    if isinstance(record, Mapping) and any(record.get(k) for k in ('dock_pose', 'rack_slot')):
        return False
    return (
        isinstance(record, Mapping)
        and isinstance(record.get('free_joint_name'), str)
        and bool(record['free_joint_name'])
        and record.get('collision_enabled') is not False
    )


def bind_scene_tool_contact(artifact, request):
    if not is_free_scene_tool_pick(request) or str(request.task.ee).upper() != '2F':
        return artifact
    if request.task.grasp is not None:
        return artifact  # An explicit grasp remains authoritative.
    target = request.task.goal.target_object_id or request.task.tool
    record = request.world.objects[target]
    support = support_clearance_context(record, request, target)
    margin = float(request.constraints.collision_margin_m)
    if support is None or support.under_clearance_m > margin:
        return artifact
    pose = record.get('pose') or {}
    dims = np.asarray(record.get('dimensions_m'), dtype=float)
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0.):
        raise ValueError('SCENE_TOOL_CONTACT_GEOMETRY_REQUIRED')
    rotation = quaternion_matrix_xyzw(pose['orientation_xyzw'])
    position = np.asarray(pose['position_m'], dtype=float)
    anchors = dict(record.get('anchors') or {})
    center = np.asarray(anchors.get('center', [0., 0., 0.]), dtype=float)
    approach = rotation.T @ np.array([0., 0., 1.])
    points = np.asarray(record.get('collision_points_m', []), dtype=float)
    if points.ndim == 2 and points.shape[1] == 3 and len(points) >= 4 and np.isfinite(points).all():
        top_z = float((points @ rotation.T + position)[:, 2].max())
    else:
        top_z = float((position + rotation @ center)[2] + np.abs(rotation[2]) @ dims / 2.)
    contact = center + approach * (top_z - float((position + rotation @ center)[2]))
    anchors[ANCHOR] = contact.tolist()
    request.world.objects[target] = {**record, 'anchors': anchors}
    # Parallel-jaw X is the closing direction, as in TwoFingerOpposedContactBinder.
    vertical_axis = int(np.argmax(np.abs(rotation[2])))
    lateral_axes = [i for i in range(3) if i != vertical_axis]
    short_axis = min(lateral_axes, key=lambda i: dims[i])
    closing = rotation[:, short_axis].copy();closing[2] = 0.
    closing /= np.linalg.norm(closing)
    zero = tool_rotation_from_axis(np.array([0., 0., -1.]), 0.)
    roll = math.atan2(float(closing @ zero[:, 1]), float(closing @ zero[:, 0]))
    profile = PhysicalGraspProfile.from_mapping(request.task.metadata.get('grasp_profile'))
    wait = profile.grasp_hold_duration_s
    # Existing collision margin supplies the finite vertical sampling scale.
    # Lower candidates are preferred; higher ones must still pass the unchanged
    # runtime proximity gate after real closure. No feasibility flag is written.
    offsets = tuple(dict.fromkeys([0., margin, 2. * margin, 3. * margin]))
    candidates = []
    for candidate in artifact.candidates:
        allowed = {KeyframeType.PRE_GRASP, KeyframeType.GRASP, KeyframeType.LIFT, KeyframeType.RETREAT}
        if any(k.keyframe_type not in allowed or k.frame_ref not in {'world', f'object:{target}'} for k in candidate.keyframes):
            candidates.append(candidate)
            continue
        if sum(k.keyframe_type is KeyframeType.GRASP for k in candidate.keyframes) != 1:
            candidates.append(candidate)
            continue
        grasp = next((k for k in candidate.keyframes if k.keyframe_type is KeyframeType.GRASP), None)
        if grasp is None:
            continue
        for offset in offsets:
            frames = []
            grasp_meta = dict(grasp.metadata)
            event_offsets = dict(grasp_meta.get('event_time_offsets_s') or {})
            events = list(grasp.events_after)
            if KeyframeEventType.GRIPPER_CLOSE not in events or KeyframeEventType.ATTACH_OBJECT not in events:
                continue
            close_at = float(event_offsets.get('GRIPPER_CLOSE', 0.))
            event_offsets['GRIPPER_CLOSE'] = close_at
            event_offsets['ATTACH_OBJECT'] = max(float(event_offsets.get('ATTACH_OBJECT', 0.)), close_at + wait)
            grasp_meta.update(contact_geometry_source=SOURCE, prior_grasp_offset_m=float(grasp.offset_along_approach_m), scene_tool_contact_offset_m=offset,
                              hold_duration_after_s=max(float(grasp_meta.get('hold_duration_after_s', 0.)), event_offsets['ATTACH_OBJECT']), event_time_offsets_s=event_offsets)
            frame_base = dict(frame_ref=f'object:{target}', anchor=ANCHOR, approach_axis_xyz=tuple(approach), tool_axis_to_align='-z', roll_rad=roll)
            clean_meta = {k:v for k,v in grasp_meta.items() if k not in {'event_target_id','event_parameters','event_time_offsets_s','hold_duration_after_s'}}
            pre = grasp.model_copy(update={**frame_base, 'keyframe_id':grasp.keyframe_id+':scene-pre', 'keyframe_type':KeyframeType.PRE_GRASP, 'offset_along_approach_m':max(.08, request.task.goal.approach_distance_m or 0.), 'events_after':[], 'metadata':clean_meta})
            frames.append(pre)
            frames.append(grasp.model_copy(update={**frame_base, 'offset_along_approach_m':offset, 'metadata':grasp_meta}))
            follower = next((k for k in candidate.keyframes if k.keyframe_type in {KeyframeType.LIFT, KeyframeType.RETREAT}), None)
            if follower is None:
                continue
            frames.append(follower.model_copy(update={**frame_base, 'keyframe_type':KeyframeType.LIFT, 'offset_along_approach_m':max(.12, float(follower.offset_along_approach_m)), 'metadata':{**follower.metadata,'contact_geometry_source':SOURCE}}))
            candidates.append(candidate.model_copy(update={'strategy_id':candidate.strategy_id+f':scene-contact-{offset:.6f}', 'keyframes':frames, 'metadata':{**candidate.metadata,'geometry_binder':SOURCE,'contact_candidate_only':True}, 'rationale':candidate.rationale+' Free scene-tool contact candidate; normal physics validation required.'}))
            if len(candidates) == 20:
                break
        if len(candidates) == 20:
            break
    if not candidates:
        return artifact
    digest = hashlib.sha256((artifact.artifact_id + str(target) + SOURCE).encode()).hexdigest()[:16]
    return artifact.model_copy(update={'artifact_id':artifact.artifact_id+':scene-contact:'+digest, 'candidates':candidates, 'provenance':artifact.provenance.model_copy(update={'metadata':{**artifact.provenance.metadata,'geometry_binder':SOURCE,'source_keyframe_artifact_id':artifact.artifact_id}})})
