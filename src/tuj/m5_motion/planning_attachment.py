"""Measured planning geometry for a physically compliant held object.

The runtime's formation transform remains the penalty weld's rest pose. A live
planning snapshot instead needs the current observed body-to-reference pose.
This module only reads state; it never rebases the weld or moves a body.
"""
from __future__ import annotations

import numpy as np

from tuj.m5_motion.geometry import matrix_quaternion_xyzw
from tuj.m5_motion.tool_use_journal_planning import attached_object_transform_from_state


def planning_attachment_transform(runtime, adapter):
    """Return planning transform and provenance, retaining rigid paths exactly."""
    attachment = runtime.attachment
    if attachment is None:
        return None, None
    rest = attached_object_transform_from_state(attachment)
    mode = str(getattr(attachment.mode, 'value', attachment.mode))
    if mode != 'BREAKABLE_WELD':
        return rest, None
    body_id = adapter.object_body_ids[attachment.object_id]
    object_position = np.asarray(adapter.data.xpos[body_id], dtype=float).copy()
    object_rotation = np.asarray(adapter.data.xmat[body_id], dtype=float).reshape(3, 3).copy()
    reference_position, reference_rotation = runtime._reference_pose(
        runtime.env, attachment.reference_kind, attachment.reference_name
    )
    reference_position = np.asarray(reference_position, dtype=float)
    reference_rotation = np.asarray(reference_rotation, dtype=float).reshape(3, 3)
    values = (object_position, object_rotation, reference_position, reference_rotation)
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError('non-finite measured attachment pose')
    position = reference_rotation.T @ (object_position - reference_position)
    rotation = reference_rotation.T @ object_rotation
    measured = rest.model_copy(update={
        'position_in_reference_m': tuple(float(value) for value in position),
        'orientation_in_reference_xyzw': matrix_quaternion_xyzw(rotation),
    })
    rest_rotation = np.asarray(attachment.rotation_in_reference, dtype=float)
    angle_cos = np.clip((np.trace(rest_rotation.T @ rotation) - 1.) / 2., -1., 1.)
    provenance = {
        'source': 'MEASURED_CURRENT_REFERENCE_TO_OBJECT',
        'attachment_mode': mode,
        'runtime_rest_transform_unchanged': True,
        'formation_rest_transform': rest.model_dump(mode='json'),
        'measured_planning_transform': measured.model_dump(mode='json'),
        'position_deviation_from_rest_m': float(np.linalg.norm(
            position - np.asarray(rest.position_in_reference_m))),
        'orientation_deviation_from_rest_rad': float(np.arccos(angle_cos)),
    }
    return measured, provenance
