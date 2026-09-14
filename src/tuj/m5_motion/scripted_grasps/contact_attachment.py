"""Opt-in finger attachment after measured bilateral contact acquisition."""
from dataclasses import asdict

import numpy as np
from scipy.spatial.transform import Rotation

from .frames import inverse
from .runtime import GraspFailure, save_json


def attach_after_stable_contact(context):
    c = context
    if c.recipe.ee_id != '2F' or c.recipe.finger_attachment_policy != 'STABLE_CONTACT':
        raise GraspFailure('STABLE_CONTACT_ATTACHMENT_NOT_CONFIGURED')
    if not c.ready():
        raise GraspFailure('BILATERAL_CONTACT_GATE_NOT_PASSED')
    # mj_step can leave derived poses at the pre-integration state. Refresh
    # before capturing; attach_object also forwards without advancing time.
    c.mj.mj_forward(c.model, c.data)
    c.sample()
    if not c.ready():
        raise GraspFailure('BILATERAL_CONTACT_GATE_NOT_PASSED')
    before = c.body_pose().copy()
    relative = inverse(c.grip_pose()) @ before
    action = np.asarray(c.gripper.current_action).copy()
    time_s = float(c.data.time)
    c.runtime.command_gripper(engaged=True, suction=False, command=1.)
    attachment = c.runtime.attach_object(c.object_id)
    # These are numerical identity checks, not a physical grasp tolerance.
    if (float(c.data.time) != time_s
            or not np.allclose(c.body_pose(), before, rtol=0., atol=1e-10)
            or not np.array_equal(c.gripper.current_action, action)):
        raise GraspFailure('ATTACH_CHANGED_MEASURED_STATE')
    c.runtime.capture_gripper_hold()
    c.two_finger_force_hold = False
    record = {'policy': 'STABLE_CONTACT', 'validation_basis': 'CONTACT_GATED_KINEMATIC_ATTACHMENT',
              'time_s': time_s, 'T_GB_at_attach': relative.copy(),
              'finger_action_at_attach': action.copy(),
              'contact_before_attach': c.trace[-1],
              'attachment': asdict(attachment),
              'world_pose_jump_m': float(np.linalg.norm(c.body_pose()[:3, 3] - before[:3, 3])),
              'relative_pose_source': 'ACTUAL_CONTACT_POSE'}
    c.finger_attachment_record = record
    save_json(c.output / 'finger_attachment.json', record)
    return -float(action.mean())


def audit_contact_attachment(context, row):
    record = getattr(context, 'finger_attachment_record', None)
    if record is None:
        return
    if not row['attachment_active']:
        raise GraspFailure('FINGER_ATTACHMENT_LOST')
    reference = np.asarray(record['T_GB_at_attach'])
    actual = np.asarray(row['T_GB'])
    row['attachment_position_error_m'] = float(np.linalg.norm(actual[:3, 3] - reference[:3, 3]))
    row['attachment_angle_error_deg'] = float(np.rad2deg(
        Rotation.from_matrix(reference[:3, :3].T @ actual[:3, :3]).magnitude()))
    if (row['attachment_position_error_m'] > context.recipe.maximum_slip_m
            or row['attachment_angle_error_deg'] > context.recipe.maximum_slip_deg):
        raise GraspFailure('FINGER_ATTACHMENT_POSE_ERROR')
