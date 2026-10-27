from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.m5_motion.compiler import FirstFeasibleStrategyCompiler
from tuj.m5_motion.contact_keyframe_validation import ContactKeyframeGeometryError
from tuj.m5_motion.geometry import RelativePoseResolver
from tuj.m5_motion.kinematics import IKResult, IKSolutionSet
from tuj.m5_motion.native_contact_preparation import build_native_contact_preparation
from tuj.m5_motion.native_side_push import native_side_push_geometry
from tuj.m5_motion.plan_builder import MotionPlanBuilder, MotionPlanBuildError
from tuj.m5_motion.profiles import GraspExecutionProfile
from tuj.m5_motion.schema import (
    CollisionContext, JointDynamicLimit, KeyframePlanCandidate, ModuleName,
    StrategyGenerationProvenance, StrategyGeneratorKind,
)
from tuj.m5_motion.strategy import EdgePlanResult
from tuj.m5_motion.tests.test_native_side_push import _request, _attitude


def _prepared():
    request = _request()
    request.task.metadata.update(ee_capabilities=['multi_finger_contact'], operation='SWEEP',
                                 attach_target=True, held_transport_goal={'bad': 'stale'})
    request.constraints.position_tolerance_m = .003
    request.constraints.orientation_tolerance_rad = .027
    request.constraints.joint_limits = {name: JointDynamicLimit(
        max_velocity_rad_s=1., max_acceleration_rad_s2=2., max_jerk_rad_s3=10.)
        for name in request.world.robot_state.joint_names}
    return build_native_contact_preparation(request, eef_orientation_xyzw=_attitude(request))


@pytest.mark.parametrize('live_height', [.85, 1.2])
def test_prepare_keeps_xy_lifts_before_rotation_and_does_not_modify_original(live_height):
    source = _request();live=source.world.robot_state.eef_pose
    live.position_m = (*live.position_m[:2], live_height)
    source.task.metadata.update(ee_capabilities=['multi_finger_contact'], operation='SWEEP',
                                attach_target=True, held_transport_goal={'stale': True})
    source.constraints.position_tolerance_m = .003
    source.constraints.orientation_tolerance_rad = .027
    before = source.model_dump_json();attitude=_attitude(source)
    geometry = native_side_push_geometry(source, eef_orientation_xyzw=attitude)
    prepared, frames = build_native_contact_preparation(source, eef_orientation_xyzw=attitude)
    assert source.model_dump_json() == before
    assert prepared.world == source.world and prepared.world is not source.world
    assert prepared.constraints == source.constraints and prepared.constraints is not source.constraints
    assert prepared.options == source.options
    assert prepared.task.action_type == 'MOVE' and prepared.task.contact is None
    assert prepared.task.target_ids == prepared.task.allowed_touch_objects == []
    assert prepared.task.grasp is None
    assert prepared.task.ee == source.task.ee and prepared.task.tool == source.task.tool
    assert prepared.task.metadata['ee_capabilities'] == ['multi_finger_contact']
    assert prepared.task.metadata['native_contact_preparation'] is True
    assert all(k not in prepared.task.metadata for k in ('operation', 'attach_target', 'held_transport_goal'))
    assert prepared.task.goal.target_object_id is prepared.task.goal.target_region_id is None
    assert prepared.request_id != source.request_id and prepared.task.subgoal_id != source.task.subgoal_id
    poses = [RelativePoseResolver(prepared.world).resolve(f) for f in frames]
    for pose, frame in zip(poses, frames):
        assert pose.position_m == pytest.approx((*live.position_m[:2], max(live_height, geometry.hover_tcp_z_m)))
        assert frame.keyframe_type.value == 'TRANSFER'
        assert frame.events_after == [] and frame.collision_context_id is None
        assert 'pose_subject' not in frame.metadata
    assert Rotation.from_quat(poses[0].orientation_xyzw).as_matrix() == pytest.approx(Rotation.from_quat(live.orientation_xyzw).as_matrix())
    assert Rotation.from_quat(poses[1].orientation_xyzw).as_matrix() == pytest.approx(Rotation.from_quat(attitude).as_matrix())
    assert poses[-1].position_m == pytest.approx(prepared.task.goal.target_pose.position_m)
    assert frames[-1].metadata['hold_duration_after_s'] == GraspExecutionProfile().hold_duration_s
    assert frames[-1].metadata['tracking_settle'] == {'eef_tolerance_m': .003, 'eef_orientation_tolerance_rad': .027}
    # Preparation output owns its data; changing it cannot rewrite the task/inputs.
    prepared.task.metadata['ee_capabilities'].append('test')
    prepared.world.objects['block_a']['dimensions_m'][0] += .1
    assert source.model_dump_json() == before


@pytest.mark.parametrize('case', ['not_side', 'no_pose', 'not_held', 'no_grasp', 'bad_frame', 'unsafe_attitude'])
def test_preparation_does_not_bypass_native_geometry_contract(case):
    source=_request();attitude=_attitude(source)
    if case=='not_side': source.task.contact.path_pattern='AUTO'
    elif case=='no_pose': source.world.robot_state.eef_pose=None
    elif case=='not_held': source.world.robot_state.held_tool_id=source.world.robot_state.attached_object_id=None
    elif case=='no_grasp': source.world.metadata.clear()
    elif case=='bad_frame': source.world.robot_state.eef_pose.frame_id='other_frame'
    else: attitude=[0.,0.,0.,1.]
    with pytest.raises(ContactKeyframeGeometryError):
        build_native_contact_preparation(source,eef_orientation_xyzw=attitude)


def _compile_preparation(prepared, frames, *, valid=True):
    calls=[]
    class Kinematics:
        def solve_all_ik(self, world_pos, orientation_xyzw, **kwargs):
            calls.append(tuple(world_pos))
            q=np.array(prepared.world.robot_state.joint_positions_rad);q[0]+=.01*len(calls)
            return IKSolutionSet(solutions=(IKResult(solved=True, qpos=tuple(q), branch_id='ok',
                position_error_m=0.,orientation_error_rad=0.),),best_position_error_m=0.,
                best_orientation_error_rad=0.,attempted_seeds=1,enumeration_complete=True)
    class Edges:
        def plan(self, source, target, source_keyframe, target_keyframe):
            return EdgePlanResult(valid=True,joint_path=(tuple(source),tuple(target)))
    candidate=KeyframePlanCandidate(strategy_id='prep',keyframes=frames,
        provenance=StrategyGenerationProvenance(generator_kind=StrategyGeneratorKind.TASK_GEOMETRY,
            generator_id='native-preparation-test',input_hash='test'))
    result=FirstFeasibleStrategyCompiler(Kinematics()).compile(prepared.world,[candidate],
        start_joint_config=prepared.world.robot_state.joint_positions_rad,
        state_validator=lambda *_:valid,edge_planner=Edges(),request=prepared)
    return result,calls


def _build_preparation_plan(*, final_valid=True):
    prepared,frames=_prepared();compiled,calls=_compile_preparation(prepared,frames)
    assert compiled.solved and len(calls)==2
    context=CollisionContext(context_id='held',scene_state_id='scene',active_ee=prepared.task.ee,
        attached_object_ids=[prepared.task.tool],collision_model_version='test')
    checked=[]
    def validate(waypoints, collision_context):
        checked.append(collision_context)
        return final_valid
    plan=MotionPlanBuilder().build(prepared,compiled.connected,plan_id='prep-plan',
        provenance=prepared.provenance.model_copy(update={'artifact_type':'MotionPlan','produced_by':ModuleName.MOTION_PLANNER}),
        collision_contexts={'held':context},initial_collision_context_id='held',final_segment_validator=validate)
    assert checked
    return plan


def test_preparation_compiles_and_keeps_hold_tracking_and_collision_guards():
    plan=_build_preparation_plan()
    assert plan.metadata['native_contact_preparation'] is True
    assert 'contact_primitive' not in plan.metadata and 'target_ids' not in plan.metadata
    assert plan.events == []
    final=plan.segments[-1]
    assert final.metadata['hold_duration_after_s']==1.
    assert final.end_time_s-final.metadata['motion_end_time_s']==pytest.approx(1.)
    assert final.metadata['tracking_settle']=={'eef_tolerance_m':.003,'eef_orientation_tolerance_rad':.027,
        'max_wait_s':2.,'required_consecutive_ticks':3}
    assert plan.expected_final_state.attached_object_id=='tool_pusher'
    for segment in plan.segments:
        assert segment.collision_checked
        assert segment.collision_context_before.allowed_collision_pairs==[]


def test_preparation_still_rejects_invalid_ik_states_and_final_collisions():
    prepared,frames=_prepared();result,calls=_compile_preparation(prepared,frames,valid=False)
    assert calls and not result.solved
    assert result.attempts[-1].failure_code=='NO_VALID_IK_BRANCH'
    with pytest.raises(MotionPlanBuildError):
        _build_preparation_plan(final_valid=False)


@pytest.mark.parametrize('live_height', [.85, 1.2])
def test_local_preparation_moves_to_tool_contact_start_hover_without_changing_task(live_height):
    from tuj.m5_motion import native_contact_preparation as preparation
    source = _request()
    source.world.robot_state.eef_pose.position_m = (.31, -.24, live_height)
    source.constraints.position_tolerance_m = .003
    source.constraints.orientation_tolerance_rad = .027
    source.task.metadata.update(ee_capabilities=['multi_finger_contact'], attach_target=True)
    before = source.model_dump_json(); attitude = _attitude(source)
    geometry = native_side_push_geometry(source, eef_orientation_xyzw=attitude)
    prepared, frames = preparation.build_native_contact_local_preparation(source, eef_orientation_xyzw=attitude)
    poses = [RelativePoseResolver(prepared.world).resolve(f) for f in frames]
    expected_xy = np.asarray(geometry.patch_start_xy_m) - geometry.working_patch_offset_xy_m
    expected_z = max(live_height, geometry.hover_tcp_z_m)
    assert poses[0].position_m == pytest.approx((.31, -.24, expected_z))
    assert poses[-1].position_m == pytest.approx((*expected_xy, expected_z))
    assert poses[-1].position_m == pytest.approx(prepared.task.goal.target_pose.position_m)
    assert Rotation.from_quat(poses[-1].orientation_xyzw).as_matrix() == pytest.approx(Rotation.from_quat(attitude).as_matrix())
    assert source.model_dump_json() == before
    assert prepared.world == source.world and prepared.world is not source.world
    assert prepared.constraints == source.constraints and prepared.options == source.options
    assert prepared.task.action_type == 'MOVE' and prepared.task.contact is None
    assert prepared.task.target_ids == prepared.task.allowed_touch_objects == []
    assert prepared.task.goal.target_region_id is prepared.task.goal.target_object_id is None
    assert prepared.task.metadata['native_contact_preparation'] is True
    assert prepared.task.metadata['requires_measured_replanning_after_execution'] is True
    assert 'attach_target' not in prepared.task.metadata
    assert frames[-1].metadata['hold_duration_after_s'] == GraspExecutionProfile().hold_duration_s
    assert frames[-1].metadata['tracking_settle'] == {'eef_tolerance_m': .003, 'eef_orientation_tolerance_rad': .027}
    assert all(frame.events_after == [] and frame.collision_context_id is None for frame in frames)


def test_local_preparation_uses_supplied_observation_and_never_predicts_post_move_world():
    from tuj.m5_motion import native_contact_preparation as preparation
    source = _request(); attitude = _attitude(source)
    first, _ = preparation.build_native_contact_local_preparation(source, eef_orientation_xyzw=attitude)
    measured = source.model_copy(deep=True)
    measured.world.metadata['attached_object_transforms']['tool_pusher']['position_in_reference_m'][0] += .004
    second, _ = preparation.build_native_contact_local_preparation(measured, eef_orientation_xyzw=attitude)
    assert second.world == measured.world
    assert first.world == source.world
    assert not np.allclose(first.task.goal.target_pose.position_m, second.task.goal.target_pose.position_m)
    assert second.task.metadata['requires_measured_replanning_after_execution'] is True
    # The returned MOVE has no contact keyframes. Its caller must observe the
    # executed state and request a fresh SIDE_PUSH, not treat this as collection.
    assert second.task.contact is None and second.task.action_type == 'MOVE'


def test_local_preparation_keeps_normal_ik_collision_validation_and_native_contract():
    from tuj.m5_motion import native_contact_preparation as preparation
    source = _request(); attitude = _attitude(source)
    prepared, frames = preparation.build_native_contact_local_preparation(source, eef_orientation_xyzw=attitude)
    valid, valid_calls = _compile_preparation(prepared, frames)
    assert valid.solved and len(valid_calls) == 2
    compiled, calls = _compile_preparation(prepared, frames, valid=False)
    assert calls and not compiled.solved
    assert compiled.attempts[-1].failure_code == 'NO_VALID_IK_BRANCH'
    source.task.contact.path_pattern = 'AUTO'
    with pytest.raises(ContactKeyframeGeometryError):
        preparation.build_native_contact_local_preparation(source, eef_orientation_xyzw=attitude)
