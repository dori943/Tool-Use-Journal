from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.m5_motion.contact_keyframe_validation import (
    ContactKeyframeGeometryError, _contact_engagement_tcp_z_m,
    _held_tool_engaging_offsets_m,
)
from tuj.m5_motion.geometry import RelativePoseResolver
from tuj.m5_motion.native_side_push import (
    canonicalize_native_side_push, generate_native_side_push_keyframes,
    native_side_push_geometry,
)
from tuj.m5_motion.profiles import PushPlanningProfile
from tuj.m5_motion.push_to_region import target_fully_inside_region
from tuj.m5_motion.tests.test_contact_keyframe_validation import _sweep_request


def _request():
    request = _sweep_request()
    request.task.contact.path_pattern = "SIDE_PUSH"
    request.world.robot_state.attached_object_id = "tool_pusher"
    request.world.robot_state.held_tool_id = "tool_pusher"
    request.world.objects['tool_pusher']['collision_points_m'] = list(product((-.02, .08), (-.015, .025), (-.004, .004)))
    request.world.metadata['attached_object_transforms'] = {'tool_pusher': {
        'object_id': 'tool_pusher', 'reference_kind': 'site', 'reference_name': 'eef',
        'free_joint_name': 'tool_free', 'position_in_reference_m': [.03, -.015, .01],
        'orientation_in_reference_xyzw': Rotation.from_euler('z', 30, degrees=True).as_quat().tolist(),
    }}
    target = request.world.objects['block_a']
    target['dimensions_m'] = [.02, .035, .012]
    target['pose']['position_m'] = [-.10, .05, .8058]
    target['pose']['orientation_xyzw'] = Rotation.from_euler('z', 37, degrees=True).as_quat().tolist()
    region = request.world.objects['collect_zone']
    region['pose']['position_m'] = [.15, .10, .801]
    region['pose']['orientation_xyzw'] = Rotation.from_euler('z', 63, degrees=True).as_quat().tolist()
    region['dimensions_m'] = [.20, .11, .001]
    return request


def _attitude(request):
    return (Rotation.from_quat(request.world.robot_state.eef_pose.orientation_xyzw)
            * Rotation.from_euler('x', 15, degrees=True)).as_quat()


def test_rotated_footprints_reach_region_without_changing_world_or_height():
    request = _request(); before = request.model_dump_json(); attitude = _attitude(request)
    geometry = native_side_push_geometry(request, eef_orientation_xyzw=attitude)
    frames = generate_native_side_push_keyframes(request, eef_orientation_xyzw=attitude)
    resolver = RelativePoseResolver(request.world)
    assert request.model_dump_json() == before
    assert geometry.contact_tcp_z_m == pytest.approx(_contact_engagement_tcp_z_m(request, eef_orientation_xyzw=attitude))
    assert all(resolver.resolve(f).position_m[2] == pytest.approx(geometry.contact_tcp_z_m) for f in frames[1:4])
    result = request.world.model_copy(deep=True)
    translation = np.array(geometry.target_goal_center_world_m)-np.array(geometry.target_center_world_m)
    result.objects['block_a']['pose']['position_m'] = (np.array(result.objects['block_a']['pose']['position_m'])+translation).tolist()
    assert target_fully_inside_region(result, target_id='block_a', region_id='collect_zone', contact_tolerance_m=0.)


def test_start_is_behind_full_target_and_patch_leading_extent_with_asymmetric_grasp():
    request = _request(); attitude = _attitude(request)
    geometry = native_side_push_geometry(request, eef_orientation_xyzw=attitude)
    frames = generate_native_side_push_keyframes(request, eef_orientation_xyzw=attitude)
    pose = RelativePoseResolver(request.world).resolve(frames[1])
    cloud = _held_tool_engaging_offsets_m(request, eef_orientation_xyzw=attitude)
    direction = np.array(geometry.push_direction_xy)
    front = ((np.array(pose.position_m[:2])+cloud[:, :2]-np.array(geometry.target_center_world_m[:2])) @ direction).max()
    assert geometry.target_trailing_projection_m-front == pytest.approx(PushPlanningProfile().approach_standoff_m)
    assert not np.allclose(pose.position_m[:2], geometry.patch_start_xy_m)


def test_recanonicalization_is_idempotent_and_ignores_forged_xy_metadata():
    request = _request(); frames = generate_native_side_push_keyframes(request, eef_orientation_xyzw=_attitude(request))
    again = canonicalize_native_side_push(request, frames)
    resolver = RelativePoseResolver(request.world)
    for a, b in zip(frames, again):
        assert resolver.resolve(a).position_m == pytest.approx(resolver.resolve(b).position_m)
    forged = [f.model_copy(deep=True) for f in frames]
    forged[1].offset_along_approach_m += .2
    forged[1].metadata['native_side_push_geometry'] = True
    corrected = canonicalize_native_side_push(request, forged)
    assert resolver.resolve(corrected[1]).position_m == pytest.approx(resolver.resolve(frames[1]).position_m)


@pytest.mark.parametrize('case', ['multi', 'missing_grasp', 'oversize', 'unsafe_tilt', 'auto'])
def test_invalid_requests_do_not_silently_get_a_feasible_push(case):
    request = _request(); attitude = _attitude(request)
    if case == 'multi': request.task.target_ids.append('other_target')
    elif case == 'missing_grasp': request.world.metadata.clear()
    elif case == 'oversize': request.world.objects['block_a']['dimensions_m'] = [1., 1., .012]
    elif case == 'unsafe_tilt': attitude = [0., 0., 0., 1.]
    elif case == 'auto': request.task.contact.path_pattern = 'AUTO'
    with pytest.raises(ContactKeyframeGeometryError):
        generate_native_side_push_keyframes(request, eef_orientation_xyzw=attitude)


def test_changing_attitude_mid_push_is_rejected():
    request = _request(); frames = generate_native_side_push_keyframes(request, eef_orientation_xyzw=_attitude(request))
    frames[2].metadata['packing_orientation_xyzw'] = request.world.robot_state.eef_pose.orientation_xyzw
    with pytest.raises(ContactKeyframeGeometryError, match='fixed attitude'):
        canonicalize_native_side_push(request, frames)
