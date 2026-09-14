"""Contracts for the explicit stable-contact attachment policy."""
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.m5_motion.scripted_grasps.contact_attachment import attach_after_stable_contact, audit_contact_attachment
from tuj.m5_motion.scripted_grasps.objects.whisk import whisk_recipe
from tuj.m5_motion.scripted_grasps.frames import inverse
from tuj.m5_motion.scripted_grasps.runtime import GraspFailure


@dataclass
class Attachment:
    object_id: str


def fixture(tmp_path):
    grip = np.eye(4)
    grip[:3, :3] = Rotation.from_euler('xyz', [20, -30, 70], degrees=True).as_matrix()
    grip[:3, 3] = [.4, .2, 1.1]
    body = np.eye(4)
    body[:3, :3] = Rotation.from_euler('xyz', [-15, 25, 10], degrees=True).as_matrix()
    body[:3, 3] = [.43, .21, 1.02]
    runtime = SimpleNamespace(command_gripper=Mock(), capture_gripper_hold=Mock(),
                              attach_object=Mock(return_value=Attachment('whisk')))
    return SimpleNamespace(recipe=whisk_recipe(), ready=lambda: True, output=tmp_path,
                           runtime=runtime, object_id='whisk', trace=[{'finger_contacts': ['left', 'right']}],
                           body_pose=lambda: body.copy(), grip_pose=lambda: grip.copy(),
                           mj=SimpleNamespace(mj_forward=Mock()), model=object(), sample=Mock(),
                           data=SimpleNamespace(time=12.), gripper=SimpleNamespace(current_action=np.array([.58])),
                           two_finger_force_hold=True)


def test_capture_actual_rotated_pose_and_action(tmp_path):
    c = fixture(tmp_path)
    expected = inverse(c.grip_pose()) @ c.body_pose()
    opening = attach_after_stable_contact(c)
    np.testing.assert_allclose(c.finger_attachment_record['T_GB_at_attach'], expected)
    np.testing.assert_array_equal(c.finger_attachment_record['finger_action_at_attach'], [.58])
    assert opening == -.58 and c.two_finger_force_hold is False
    assert c.finger_attachment_record['world_pose_jump_m'] == 0.
    c.runtime.capture_gripper_hold.assert_called_once()
    c.mj.mj_forward.assert_called_once_with(c.model, c.data)


def test_refreshed_contact_gate_must_still_pass(tmp_path):
    c = fixture(tmp_path)
    c.ready = Mock(side_effect=[True, False])
    with pytest.raises(GraspFailure, match='CONTACT_GATE'):
        attach_after_stable_contact(c)
    c.runtime.attach_object.assert_not_called()


def test_refresh_precedes_capture_of_measured_pose(tmp_path):
    c = fixture(tmp_path)
    latest = c.body_pose(); latest[0, 3] += 4.5e-7
    def forward(*_):
        c.body_pose = lambda: latest.copy()
    c.mj.mj_forward.side_effect = forward
    attach_after_stable_contact(c)
    np.testing.assert_allclose(c.finger_attachment_record['T_GB_at_attach'],
                               inverse(c.grip_pose()) @ latest, rtol=0., atol=1e-12)


def test_unstable_contact_never_attaches(tmp_path):
    c = fixture(tmp_path)
    c.ready = lambda: False
    with pytest.raises(GraspFailure, match='CONTACT_GATE'):
        attach_after_stable_contact(c)
    c.runtime.attach_object.assert_not_called()


@pytest.mark.parametrize('mutation', ['pose', 'action', 'time'])
def test_attach_cannot_mutate_measured_state(tmp_path, mutation):
    c = fixture(tmp_path)
    def attach(_):
        if mutation == 'pose':
            changed = c.body_pose(); changed[0, 3] += .001
            c.body_pose = lambda: changed
        elif mutation == 'action':
            c.gripper.current_action[:] = 1.
        else:
            c.data.time += .02
        return Attachment('whisk')
    c.runtime.attach_object.side_effect = attach
    with pytest.raises(GraspFailure, match='ATTACH_CHANGED_MEASURED_STATE'):
        attach_after_stable_contact(c)


def test_attach_time_pose_is_checked_and_loss_fails(tmp_path):
    c = fixture(tmp_path)
    attach_after_stable_contact(c)
    row = {'attachment_active': True, 'T_GB': inverse(c.grip_pose()) @ c.body_pose()}
    audit_contact_attachment(c, row)
    assert row['attachment_position_error_m'] == 0.
    row['T_GB'][0, 3] += .006
    with pytest.raises(GraspFailure, match='POSE_ERROR'):
        audit_contact_attachment(c, row)
    row['attachment_active'] = False
    with pytest.raises(GraspFailure, match='ATTACHMENT_LOST'):
        audit_contact_attachment(c, row)


def test_policy_is_explicit_and_two_finger_only():
    from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe
    assert CatalogRecipe('object', 'c1_1', '2F', (.1, .1, .1)).finger_attachment_policy == 'FREE_HOLD_THEN_ATTACH'
    with pytest.raises(ValueError, match='policy'):
        replace(whisk_recipe(), finger_attachment_policy='ALWAYS_ATTACH')
    with pytest.raises(ValueError, match='bilateral'):
        replace(whisk_recipe(), ee_id='3F')
