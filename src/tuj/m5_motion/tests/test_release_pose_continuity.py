"""Detached collision geometry must stay where the release candidate puts it."""
import math

import numpy as np
import pytest

from tuj.m5_motion.geometry import RelativePoseResolver, quaternion_matrix_xyzw
from tuj.m5_motion.schema import AttachedObjectTransform, GoalType, KeyframeEventType, KeyframeType, MotionGoal, Pose
from tuj.m5_motion.tests.test_tool_use_journal_planning import _artifact, _factory, _keyframe, _request


@pytest.mark.parametrize('tagged', [False, True])
@pytest.mark.parametrize('roll', [0., math.pi / 2])
def test_candidate_release_pose_is_continuous(tagged, roll):
    transform = AttachedObjectTransform(
        object_id='bottle', free_joint_name='bottle_free',
        reference_kind='site', reference_name='grip_site',
        position_in_reference_m=(.02, -.03, .1),
        orientation_in_reference_xyzw=(math.sqrt(.5), 0., 0., math.sqrt(.5)),
    )
    request = _request(MotionGoal(
        goal_type=GoalType.POSE, target_object_id='bottle',
        target_pose=Pose(frame_id='world', position_m=(9., 9., 9.), orientation_xyzw=(0., 0., 0., 1.)),
    ), action_type='PLACE', attached=transform)
    place = _keyframe('place', KeyframeType.PLACE, events=(KeyframeEventType.DETACH_OBJECT,))
    place.roll_rad = roll
    if tagged:
        place.metadata.update(pose_subject='ATTACHED_OBJECT', pose_subject_object_id='bottle')
    source = _artifact((place, _keyframe('retreat', KeyframeType.RETREAT)))
    setup = _factory().prepare(request, source)
    bound = setup.keyframe_artifact.candidates[0].keyframes[0]
    detached = setup.collision_contexts[bound.collision_context_after_events_id]
    actual = next(v.pose for v in detached.free_object_poses if v.object_id == 'bottle')
    resolved = RelativePoseResolver(request.world).resolve(place)
    rotation = quaternion_matrix_xyzw(resolved.orientation_xyzw)
    expected_position = np.array(resolved.position_m)
    expected_rotation = rotation
    if not tagged:
        expected_position = expected_position + rotation @ np.array(transform.position_in_reference_m)
        expected_rotation = rotation @ quaternion_matrix_xyzw(transform.orientation_in_reference_xyzw)
    assert actual.position_m == pytest.approx(expected_position)
    assert quaternion_matrix_xyzw(actual.orientation_xyzw) == pytest.approx(expected_rotation)
    assert not detached.allowed_collision_pairs
    assert not detached.attached_object_ids
    assert request.task.goal.target_pose.position_m == (9., 9., 9.)
