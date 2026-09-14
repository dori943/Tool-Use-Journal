"""Geometry-grounded horizontal tool-tip extraction candidates.

A thin tool held by the gripper must reach a target (e.g. a card) wedged in a
narrow horizontal interval between two appliances and drag it out the open end.
The default VLM keyframe strategy anchors the whole gripper at the target pose,
so the gripper hand / forearm / held tool collide with the appliances and the
island.  This provider instead anchors the TOOL TIP at the target and back-
computes the end-effector pose from the measured grasp transform, so the hand
stays offset behind the open end while only the thin tool tip enters the gap.

No task or object name selects coordinates: the insertion axis, the target
location and the tool tip are all read from the live world geometry and the
measured grasp.  Stage offsets are tunable through ``task.contact.metadata``
with scene-derived defaults.
"""
import hashlib
import math
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import tool_rotation_from_axis
from .schema import (ArtifactProvenance, KeyframePlanArtifact, KeyframePlanCandidate,
                     RelativeKeyframeSpec, StrategyGenerationProvenance)


def is_extract_contact(task):
    return task.contact is not None and task.contact.primitive.lower() == 'extract'


def _matrix(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose['orientation_xyzw']).as_matrix()
    result[:3, 3] = pose['position_m']
    return result


def _unit(vector):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError('degenerate direction')
    return np.asarray(vector, dtype=float) / norm


class ExtractContactProvider:
    def generate(self, request):
        task, world = request.task, request.world
        if len(task.target_ids) != 1 or not task.tool:
            raise ValueError('extract requires one target and a held tool')
        target_id = task.target_ids[0]
        target, tool = world.objects[target_id], world.objects[task.tool]
        meta = dict(task.contact.metadata or {})

        # Measured grasp transform (tool relative to the end-effector).
        attachment = world.metadata['attached_object_transforms'][task.tool]
        grasp = _matrix({'position_m': attachment['position_in_reference_m'],
                         'orientation_xyzw': attachment['orientation_in_reference_xyzw']})

        # Tool geometry from the observed collision points (object frame).
        points = np.asarray(tool['collision_points_m'], dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('finite observed tool collision points required')
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        size = np.ptp(points, axis=0)
        long_axis = int(np.argmax(size))
        thin_axis = int(np.argmin(size))
        width_axis = next(i for i in range(3) if i not in (long_axis, thin_axis))
        half_long = float(size[long_axis]) / 2

        # Target location and the horizontal insertion axis.  The tool enters
        # from the open end (the robot side) and slides toward the target; use
        # the horizontal vector from the currently held tool to the target as
        # the insertion direction (robot-side -> target), which is robust to the
        # exact gap orientation.
        target_pose = _matrix(target['pose'])
        target_center = target_pose[:3, 3] + target_pose[:3, :3] @ np.asarray(
            target.get('anchors', {}).get('center', [0., 0., 0.]), dtype=float)
        if 'insertion_axis_xyz' in meta:
            insert_dir = _unit(meta['insertion_axis_xyz'])
        else:
            # The extraction channel is the target's own longest HORIZONTAL axis
            # (the card slides out along its length), oriented to point away from
            # the held tool -- i.e. deeper into the gap, so the tip travels this
            # way to reach the target and the drag reverses it toward the open end.
            axes = target_pose[:3, :3].T  # rows = world directions of target-local axes
            horizontal = [(i, np.linalg.norm(axes[i][:2])) for i in range(3)]
            span = np.ptp(np.asarray(target['collision_points_m'], dtype=float), axis=0) \
                if 'collision_points_m' in target else np.asarray(target.get('dimensions_m', [1., 1., 1.]))
            # longest target extent whose axis is (near) horizontal
            channel_axis = max((i for i, h in horizontal if h > 0.5),
                               key=lambda i: float(span[i]))
            direction = axes[channel_axis].copy()
            direction[2] = 0.
            direction = _unit(direction)
            tool_here = _matrix(tool['pose'])[:3, 3]
            if float(direction @ (target_center - tool_here)) < 0:
                direction = -direction
            insert_dir = direction

        up = np.array([0., 0., 1.])
        # Desired tool orientation: long axis along the insertion direction (tip
        # into the gap), held FLAT -- the thin dimension VERTICAL so the tool's
        # low vertical profile (~7 mm) clears the narrow horizontal ENTRANCE gap
        # (a ~9 mm-tall slot under the clearance roof), and its width lies
        # horizontally across the wider internal channel.  The gripper is held
        # back outside the entrance (end grip), so the fingers stay in front of
        # the appliances and do not need to fit inside the narrow slot.
        rotation = np.zeros((3, 3))
        rotation[:, long_axis] = insert_dir
        rotation[:, thin_axis] = up
        rotation[:, width_axis] = np.cross(rotation[:, long_axis], rotation[:, thin_axis])
        if np.linalg.det(rotation) < 0:
            rotation[:, width_axis] *= -1
        # Re-orthonormalize (insert_dir and up need not be exactly orthogonal).
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt

        # --- Target (card) geometry along the extraction axis and vertically ---
        # The card lies FLAT on the counter.  A flat blade held at the card's own
        # level only ABUTS the card edge (both boxes are coplanar on a hard
        # surface), so sliding the blade back just slips it off and the card
        # stays put.  To actually extract it the blade must get a face BEHIND the
        # card's far edge and pull, which is exactly what the scene requires: the
        # card's FAR edge is REQUIRED_REACH deep, so a long-enough tool reaches
        # past the card and hooks it back.
        tgt_pts = np.asarray(target.get('collision_points_m', []), dtype=float)
        if tgt_pts.ndim == 2 and tgt_pts.shape[0] >= 2 and np.isfinite(tgt_pts).all():
            tproj = tgt_pts @ insert_dir  # projection along insert (extent is offset-invariant)
            card_half_insert = float((tproj.max() - tproj.min()) / 2)
            card_half_thick = float(np.ptp(tgt_pts[:, 2]) / 2)
        else:
            card_half_insert, card_half_thick = 0.027, 0.001
        half_thin = float(size[thin_axis]) / 2  # blade half-thickness

        # Hook geometry.  Slide the blade in ABOVE the card (underside just at the
        # card top -- still under the ~9 mm roof, since blade top = card_top +
        # blade_thickness stays below the roof for a thin tool), carry the tip
        # PAST the card's far edge, then LOWER the blade to counter level so a
        # vertical blade face sits behind that far edge; dragging back then
        # catches the card and pulls it out through the open end.
        # Tip sits a little PAST the card's far edge as a backstop.  The high
        # tool<->card friction is what actually drags the card (it moves WITH the
        # covering blade, staying flat), but with the tip only AT the far edge the
        # blade slid off the card partway (observed ~0.09 m then the card was left
        # behind).  A small past-edge margin keeps the card's far edge behind the
        # tip so it cannot slip out from under the blade; because friction carries
        # the card at the blade's speed, the tip rarely has to push, so it does
        # not tip the card up the way it did under LOW friction.
        hook_margin = float(meta.get('hook_margin_m', 0.010))
        hook_past = card_half_insert + hook_margin           # tip just past the far edge
        # Glide height (relative to the card CENTRE, which is target_center z):
        # blade underside just at the card top -> raise the blade centre by the
        # card half-thickness plus the blade half-thickness.
        over_card = float(meta.get('over_card_dz_m', card_half_thick + half_thin))
        # Contact height for the hook/drag.  The blade underside rides at the
        # card's MID height (a bit ABOVE the counter), pressing DOWN on the top
        # part of the thin card rather than dropping to the counter through it.
        # Dropping the blade to the counter (bottom at the surface) scooped the
        # thin card UP onto the blade -- it then rode the blade the whole sweep
        # to the hand (observed drag travel ~0.40 m, card perched on the blade).
        # Pinning the card's top instead keeps the blade ABOVE the card at all
        # times, so the card is pushed FLAT along the counter and cannot climb
        # onto the blade; it is left on the counter when the blade lifts off.
        # (Blade bottom = target_centre_z + contact_dz - half_thin; with
        # contact_dz = half_thin the bottom sits at the card centre height, i.e.
        # pressing the card's upper half.)
        contact_dz = float(meta.get('contact_dz_m', half_thin))
        # Enter/leave OUTSIDE the mouth so the vertical descent to glide height
        # happens in open space, not under the roof (the roof blocks a vertical
        # move).  The mouth is well beyond the card's near edge; stand off past it.
        approach_standoff = float(meta.get('approach_standoff_m', 0.22))
        # Pull the card fully out past the mouth (drag the caught far edge back
        # beyond the open end).
        drag_out = float(meta.get('drag_out_m', 0.24))
        lift_after = float(meta.get('retract_lift_m', 0.10))
        height_offset = float(meta.get('tool_height_offset_m', 0.0))
        approach_hover = float(meta.get('approach_hover_m', 0.14))
        hold_after = float(meta.get('hold_duration_after_s', 0.3))

        # name,        tip_s (along insert from card centre),  dz (from card z),  type,       planner
        stages = [('approach', -approach_standoff, approach_hover, 'TRANSFER', 'SAMPLING_BASED'),
                  ('descend',  -approach_standoff, over_card,      'CUSTOM',   'CARTESIAN'),
                  ('glide',    +hook_past,         over_card,      'CUSTOM',   'CARTESIAN'),
                  ('hook',     +hook_past,         contact_dz,     'CUSTOM',   'CARTESIAN'),
                  ('drag',     -drag_out,          contact_dz,     'CUSTOM',   'CARTESIAN'),
                  ('retract',  -drag_out,          lift_after,     'RETREAT',  'SAMPLING_BASED')]

        margin = request.constraints.collision_margin_m
        digest = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        tip_object = center + np.eye(3)[long_axis] * half_long  # +long extreme

        keys = []
        for tip_s, dz, ktype, planner in [(s[1], s[2], s[3], s[4]) for s in stages]:
            stage = stages[len(keys)][0]
            tip_world = target_center + insert_dir * tip_s + up * (height_offset + dz)
            # Place the tool object so its tip lands at tip_world, then recover
            # the end-effector pose from the measured grasp transform.
            tool_pose = np.eye(4)
            tool_pose[:3, :3] = rotation
            tool_pose[:3, 3] = tip_world - rotation @ tip_object
            eef = tool_pose @ np.linalg.inv(grasp)
            frame = f'extract-frame:{digest[:12]}:{stage}'
            world.objects[frame] = {'pose': {'frame_id': 'world',
                'position_m': eef[:3, 3].tolist(),
                'orientation_xyzw': Rotation.from_matrix(eef[:3, :3]).as_quat().tolist()},
                'virtual_reference_frame': True, 'anchors': {'center': [0., 0., 0.]}}
            base = tool_rotation_from_axis(eef[:3, 2], 0.)
            roll = math.atan2(float(eef[:3, 0] @ base[:, 1]), float(eef[:3, 0] @ base[:, 0]))
            keys.append(RelativeKeyframeSpec(keyframe_id=f'{task.subgoal_id}:{stage}',
                keyframe_type=ktype, frame_ref='object:' + frame, anchor='center',
                approach_axis_xyz=(0, 0, 1), tool_axis_to_align='+z', roll_rad=roll,
                planner=planner,
                metadata={'extract_stage': stage, 'target_id': target_id,
                          'tip_offset_m': tip_s,
                          'hold_duration_after_s': hold_after if stage in ('hook', 'drag') else 0.}))

        candidate = KeyframePlanCandidate(strategy_id='extract:tip_hook', keyframes=keys,
            rationale='Blade slid in above the card, tip carried past the card far edge, then '
                      'lowered behind that edge and dragged back so the blade face catches the '
                      'card and pulls it out through the open end.',
            provenance=StrategyGenerationProvenance(generator_kind='TASK_GEOMETRY',
                generator_id='CONTACT_EXTRACT_V1', input_hash=digest, attempt_index=1),
            metadata={'target_id': target_id, 'insertion_axis_xyz': insert_dir.tolist()})
        return KeyframePlanArtifact(artifact_id='extract:' + digest[:20],
            provenance=ArtifactProvenance(artifact_id='extract-artifact:' + digest[:20],
                artifact_type='KeyframePlanArtifact', produced_by='MOTION_PLANNER', invocation_id=digest[:20],
                input_artifact_ids=[request.provenance.artifact_id]),
            scene_signature=world.scene.signature, subgoal_id=task.subgoal_id, candidates=[candidate])
