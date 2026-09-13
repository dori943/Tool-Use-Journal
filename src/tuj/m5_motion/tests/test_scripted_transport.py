from types import SimpleNamespace
import pytest
import numpy as np
from scipy.spatial.transform import Rotation
from tuj.m5_motion.attachment_retarget import ATTACHED_OBJECT_POSE_SUBJECT
from tuj.m5_motion.schema import AttachedObjectTransform, Pose
from tuj.m5_motion.scripted_grasps.frames import inverse, transform
from tuj.m5_motion.scripted_grasps.transport import (
    ground_held_region_goal,
    ground_held_transport,
)
from tuj.m5_motion.geometry import tool_rotation_from_axis
from tuj.m5_motion.tests.test_scripted_grasps import request_for,ENTRIES


REGION=Rotation.from_euler('z',37,degrees=True).as_matrix()
REGION_POSITION=[-.5,.3,.7]
BODY=transform([.2,.1,.8],rotation=Rotation.from_euler('z',25,degrees=True).as_matrix())
GRIP=transform([.24,.09,.89],rotation=Rotation.from_euler('xyz',[180,0,30],degrees=True).as_matrix())
CENTER_IN_BODY=np.array([.01,-.02,0.])
LOCAL_SIZE=np.array([.08,.08,.09])
HOME=transform([-.35,.27,.76],rotation=Rotation.from_euler('z',15,degrees=True).as_matrix())
HOME_CENTER=(HOME@np.r_[CENTER_IN_BODY,1.])[:3]
HOME_BOTTOM=float(HOME_CENTER[2]-np.abs(HOME[2,:3])@LOCAL_SIZE/2.)


def _home_retention(request):
    T_GB=inverse(GRIP)@BODY
    held=AttachedObjectTransform(
        object_id=ENTRIES[4].object_id,
        free_joint_name=ENTRIES[4].object_id+'_joint0',
        reference_kind='site',
        reference_name='gripper0_right_grip_site',
        position_in_reference_m=tuple(T_GB[:3,3]),
        orientation_in_reference_xyzw=tuple(
            Rotation.from_matrix(T_GB[:3,:3]).as_quat()),
    )
    request.world.robot_state.attached_object_id=ENTRIES[4].object_id
    request.world.robot_state.held_tool_id=ENTRIES[4].object_id
    request.world.metadata['attached_object_transforms']={
        ENTRIES[4].object_id:held.model_dump(mode='json')}
    context=SimpleNamespace(body_pose=lambda:BODY,grip_pose=lambda:GRIP,
        center_in_body=CENTER_IN_BODY,local_size=LOCAL_SIZE,
        initial_body=HOME,initial_bottom=HOME_BOTTOM,support_top_z=HOME_BOTTOM)
    return SimpleNamespace(
        context=context,
        entry=ENTRIES[4],
        transform=lambda:held,
        reference_transform=lambda:held,
    )


def _tray_request(margin):
    request=request_for(ENTRIES[4],action='transport')
    request.constraints.collision_margin_m=margin
    request.task.goal.target_region_id='tray'
    request.world.objects['tray']={'pose':{'frame_id':'world','position_m':REGION_POSITION,
        'orientation_xyzw':Rotation.from_matrix(REGION).as_quat().tolist()},
        'dimensions_m':[.3,.2,.1],'anchors':{'center':[.02,0.,0.]}}
    return request


def _assert_object_space_goal(request,clearance):
    hint=request.task.metadata['held_transport_goal']
    rec=request.world.objects['tray']
    # The anchor is the held object's body origin; its bbox center must sit
    # over the region center and above the rim + half height + clearance.
    destination=REGION@rec['anchors'][hint['anchor']]+REGION_POSITION
    carried_center=destination+BODY[:3,:3]@CENTER_IN_BODY
    np.testing.assert_allclose(carried_center[:2],(REGION@np.array([.02,0,0])+REGION_POSITION)[:2])
    assert carried_center[2]>=.7+.05+.045+clearance-1e-12
    start=REGION@rec['anchors'][hint['start_anchor']]+REGION_POSITION
    np.testing.assert_allclose(start,BODY[:3,3],atol=1e-10)
    # Orientation encodes the OBJECT rotation, not the gripper's.
    approach_world=REGION@hint['approach_axis_xyz']
    assert approach_world[2]>0, 'approach axis must point outward (up) so positive offsets lift the object'
    aligned=approach_world if hint['tool_axis_to_align']=='+z' else -approach_world
    np.testing.assert_allclose(tool_rotation_from_axis(aligned,hint['roll_rad']),BODY[:3,:3],atol=1e-10)
    np.testing.assert_allclose(Rotation.from_quat(hint['object_orientation_xyzw']).as_matrix(),BODY[:3,:3],atol=1e-10)
    assert hint['pose_subject']==ATTACHED_OBJECT_POSE_SUBJECT
    assert hint['preserve_grasp_orientation'] is True
    assert hint['object_id']==ENTRIES[4].object_id
    assert request.task.goal.target_pose is None
    assert hint['frame_ref']=='object:tray'


@pytest.mark.parametrize('margin,clearance',[(.005,.02),(.03,.06)])
def test_region_transport_preserves_measured_grasp_offset_and_orientation(margin,clearance):
    request=_tray_request(margin)
    request.task.metadata['scripted_m4_implicit_object_pose']=True
    context=SimpleNamespace(body_pose=lambda:BODY,grip_pose=lambda:GRIP,
        center_in_body=CENTER_IN_BODY,local_size=LOCAL_SIZE)
    retention=SimpleNamespace(context=context,entry=ENTRIES[4])
    ground_held_transport(request,retention)
    _assert_object_space_goal(request,clearance)
    assert request.task.metadata['held_transport_goal']['source']=='LIVE_GRASP_AND_DESTINATION_BBOX'


@pytest.mark.parametrize('margin,clearance',[(.005,.02),(.03,.06)])
def test_generic_pipeline_grounds_transport_from_eef_pose_and_grasp_transform(margin,clearance):
    request=_tray_request(margin)
    object_id=ENTRIES[4].object_id
    # No scripted retention: the world carries the EEF pose and the recorded
    # contact-friction grasp transform (T_grip_body) exactly like run.py.
    T_GB=inverse(GRIP)@BODY
    request.world.robot_state.held_tool_id=object_id
    request.world.robot_state.eef_pose=Pose(frame_id='world',position_m=tuple(GRIP[:3,3]),
        orientation_xyzw=tuple(Rotation.from_matrix(GRIP[:3,:3]).as_quat()))
    request.world.metadata['contact_friction_held_objects']={object_id:{
        'object_id':object_id,'free_joint_name':f'{object_id}_joint0',
        'reference_kind':'site','reference_name':'gripper0_right_grip_site',
        'position_in_reference_m':T_GB[:3,3].tolist(),
        'orientation_in_reference_xyzw':Rotation.from_matrix(T_GB[:3,:3]).as_quat().tolist()}}
    request.world.objects[object_id]={'pose':{'frame_id':'world','position_m':[0.,0.,0.],
        'orientation_xyzw':[0.,0.,0.,1.]},'dimensions_m':LOCAL_SIZE.tolist(),
        'anchors':{'center':CENTER_IN_BODY.tolist()}}
    ground_held_transport(request)
    _assert_object_space_goal(request,clearance)
    assert request.task.metadata['held_transport_goal']['source']=='WORLD_GRASP_TRANSFORM_AND_DESTINATION_BBOX'


def test_generic_pipeline_requires_a_held_object_and_eef_pose():
    request=_tray_request(.005)
    before=request.model_dump()
    ground_held_transport(request)
    assert request.model_dump()==before
    request.world.robot_state.held_tool_id=ENTRIES[4].object_id
    request.world.metadata['contact_friction_held_objects']={}
    with pytest.raises(ValueError,match='TRANSPORT_EEF_POSE_REQUIRED'):
        ground_held_transport(request)


def test_explicit_transport_target_is_not_replaced_by_a_bbox_default():
    request=request_for(ENTRIES[4],action='transport')
    request.task.metadata['scripted_m4_implicit_object_pose']=True
    request.task.goal.target_region_id='tray'
    request.task.metadata['action_parameters']={'target_pose':{'explicit':True}}
    before=request.model_dump()
    ground_held_transport(request,SimpleNamespace(entry=ENTRIES[4]))
    assert request.model_dump()==before


def test_direct_caller_pose_is_authoritative_without_an_m4_fallback_marker():
    request=request_for(ENTRIES[4],action='transport')
    request.task.goal.target_region_id='tray'
    before=request.model_dump()
    ground_held_transport(request,SimpleNamespace(entry=ENTRIES[4]))
    assert request.model_dump()==before


def test_conceptual_tool_rest_transports_to_measured_pregrasp_home():
    request=request_for(ENTRIES[4],action='transport')
    request.task.goal.target_region_id='tool_rest'
    request.task.metadata['scripted_m4_implicit_object_pose']=True
    ground_held_transport(request,_home_retention(request))

    record=request.world.objects['tool_rest']
    assert record['collision_enabled'] is False
    assert record['metadata']['conceptual_tool_home'] is True
    hint=request.task.metadata['held_transport_goal']
    origin=np.asarray(record['pose']['position_m'])+np.asarray(
        record['anchors'][hint['anchor']])
    np.testing.assert_allclose(origin[:2],HOME[:2,3],atol=1e-10)
    assert origin[2]>HOME[2,3]
    np.testing.assert_allclose(
        Rotation.from_quat(hint['object_orientation_xyzw']).as_matrix(),
        HOME[:3,:3],
        atol=1e-10,
    )


def test_conceptual_tool_rest_returns_to_measured_pregrasp_home():
    request=request_for(ENTRIES[4],action='RETURN_TOOL')
    request.task.goal.target_region_id='tool_rest'
    request.task.metadata['scripted_m4_implicit_object_pose']=True
    ground_held_region_goal(request,_home_retention(request))

    assert 'held_transport_goal' not in request.task.metadata
    hint=request.task.metadata['held_place_goal']
    assert hint['object_id']==ENTRIES[4].object_id
    goal=request.task.goal.target_pose
    assert goal is not None
    np.testing.assert_allclose(goal.position_m[:2],HOME[:2,3],atol=1e-10)
    assert goal.position_m[2]>HOME[2,3]
    np.testing.assert_allclose(
        Rotation.from_quat(goal.orientation_xyzw).as_matrix(),
        HOME[:3,:3],
        atol=1e-10,
    )


@pytest.mark.parametrize('support_delta', [0., .000130453943949, -.001])
@pytest.mark.parametrize('margin,tolerance', [(.005,.001),(.005,.005),(.008,.002)])
def test_tool_return_clearance_uses_actual_support_not_contact_penetration(support_delta,margin,tolerance):
    request=request_for(ENTRIES[4],action='RETURN_TOOL')
    request.task.goal.target_region_id='tool_rest'
    request.task.metadata['scripted_m4_implicit_object_pose']=True
    request.constraints.collision_margin_m=margin
    request.constraints.position_tolerance_m=tolerance
    retention=_home_retention(request)
    retention.context.support_top_z=HOME_BOTTOM+support_delta
    ground_held_region_goal(request,retention)
    goal=request.task.goal.target_pose
    rotation=Rotation.from_quat(goal.orientation_xyzw).as_matrix()
    center=np.asarray(goal.position_m)+rotation@CENTER_IN_BODY
    bottom=center[2]-np.abs(rotation[2,:])@LOCAL_SIZE/2.
    expected_floor=max(HOME_BOTTOM,retention.context.support_top_z)
    assert bottom==pytest.approx(expected_floor+margin+tolerance,abs=1e-10)
    assert bottom-tolerance-retention.context.support_top_z>=margin-1e-10
    np.testing.assert_allclose(goal.position_m[:2],HOME[:2,3],atol=1e-10)
    np.testing.assert_allclose(rotation,HOME[:3,:3],atol=1e-10)


def test_tool_return_rejects_nonfinite_support_height():
    request=request_for(ENTRIES[4],action='RETURN_TOOL')
    request.task.goal.target_region_id='tool_rest'
    retention=_home_retention(request)
    retention.context.support_top_z=float('nan')
    with pytest.raises(ValueError,match='TRANSPORT_TOOL_HOME_POSE_REQUIRED'):
        ground_held_region_goal(request,retention)


def test_conceptual_tool_rest_rejects_mismatched_retention_transform():
    request=request_for(ENTRIES[4],action='transport')
    request.task.goal.target_region_id='tool_rest'
    request.task.metadata['scripted_m4_implicit_object_pose']=True
    retention=_home_retention(request)
    wrong=retention.reference_transform().model_copy(update={
        'position_in_reference_m':(9.,9.,9.)})
    retention.reference_transform=lambda:wrong

    with pytest.raises(ValueError,match='TRANSPORT_TOOL_HOME_RETENTION_MISMATCH'):
        ground_held_transport(request,retention)
    assert 'tool_rest' not in request.world.objects


def test_conceptual_tool_rest_ignores_bounded_live_pose_drift_for_provenance():
    request=request_for(ENTRIES[4],action='RETURN_TOOL')
    request.task.goal.target_region_id='tool_rest'
    request.task.metadata['scripted_m4_implicit_object_pose']=True
    retention=_home_retention(request)
    drifted=retention.transform().model_copy(update={
        'position_in_reference_m':tuple(
            np.asarray(retention.transform().position_in_reference_m)
            + np.array([2e-6,0.,0.]))})
    retention.transform=lambda:drifted

    ground_held_region_goal(request,retention)

    assert request.task.metadata['held_place_goal']['object_id']==ENTRIES[4].object_id


def test_static_packing_box_is_exposed_as_a_physical_goal_region(monkeypatch):
    from tuj.m5_motion import tool_use_journal as module
    from tuj.m5_motion.tests import test_tool_use_journal as fixtures
    source=fixtures._source_xml
    def with_box(ee):
        return source(ee).replace('</worldbody>',
            '<body name="packing_box" pos=".4 .2 .8"><geom name="packing_box_floor" type="box" size=".1 .15 .01"/></body></worldbody>')
    monkeypatch.setattr(fixtures,'_source_xml',with_box)
    monkeypatch.setattr(module,'_environment_name',lambda env:'C4_2_DiagonalFitPacking')
    env=fixtures._fake_env('2F')
    adapter=module.ToolUseJournalEnvironmentAdapter(env)
    world=adapter.world_snapshot()
    assert 'packing_box' not in env.obj_body_id
    record=world.objects['packing_box']
    assert record['body_name']=='packing_box'
    assert record['collision_enabled']
    assert 'free_joint_name' not in record
    np.testing.assert_allclose(record['dimensions_m'],[.2,.3,.02])


def _tray_collision_points(inner_floor_z=-0.029, rim_z=0.0317, bottom_z=-0.0317, half=(0.11, 0.19)):
    """Convex-hull style vertices of a tray: outer shell + an inner floor slab."""
    pts=[]
    for sx in (-1,1):
        for sy in (-1,1):
            pts.append([sx*half[0],sy*half[1],bottom_z]); pts.append([sx*half[0],sy*half[1],rim_z])
            pts.append([sx*half[0]*.9,sy*half[1]*.9,bottom_z]); pts.append([sx*half[0]*.9,sy*half[1]*.9,inner_floor_z])
            pts.append([sx*half[0]*.35,sy*half[1]*.35,bottom_z]); pts.append([sx*half[0]*.35,sy*half[1]*.35,inner_floor_z])
    return pts


def _generic_held(request,object_id):
    T_GB=inverse(GRIP)@BODY
    request.world.robot_state.held_tool_id=object_id
    request.world.robot_state.eef_pose=Pose(frame_id='world',position_m=tuple(GRIP[:3,3]),
        orientation_xyzw=tuple(Rotation.from_matrix(GRIP[:3,:3]).as_quat()))
    request.world.metadata['contact_friction_held_objects']={object_id:{
        'object_id':object_id,'free_joint_name':f'{object_id}_joint0',
        'reference_kind':'site','reference_name':'gripper0_right_grip_site',
        'position_in_reference_m':T_GB[:3,3].tolist(),
        'orientation_in_reference_xyzw':Rotation.from_matrix(T_GB[:3,:3]).as_quat().tolist()}}
    request.world.objects[object_id]={'pose':{'frame_id':'world','position_m':[0.,0.,0.],
        'orientation_xyzw':[0.,0.,0.,1.]},'dimensions_m':LOCAL_SIZE.tolist(),
        'anchors':{'center':CENTER_IN_BODY.tolist()}}


def test_generic_place_rests_the_object_on_the_region_floor_and_rewrites_the_goal_pose():
    from tuj.m5_motion.scripted_grasps.transport import ground_held_place
    request=_tray_request(.005)
    request.task.action_type='place'
    request.world.objects['tray']['collision_points_m']=_tray_collision_points()
    _generic_held(request,ENTRIES[4].object_id)
    ground_held_place(request)
    hint=request.task.metadata['held_place_goal']
    rec=request.world.objects['tray']
    destination=REGION@rec['anchors'][hint['anchor']]+REGION_POSITION
    carried_center=destination+BODY[:3,:3]@CENTER_IN_BODY
    floor_top_world=REGION_POSITION[2]-0.029
    half_height=float(np.abs(BODY[2,:3])@LOCAL_SIZE/2.)
    np.testing.assert_allclose(carried_center[:2],(REGION@np.array([.02,0,0])+REGION_POSITION)[:2])
    np.testing.assert_allclose(carried_center[2],floor_top_world+.005+half_height,atol=1e-9)
    assert hint['release_clearance_m']==pytest.approx(.005)
    assert hint['pose_subject']==ATTACHED_OBJECT_POSE_SUBJECT
    np.testing.assert_allclose(Rotation.from_quat(hint['object_orientation_xyzw']).as_matrix(),BODY[:3,:3],atol=1e-10)
    np.testing.assert_allclose(Rotation.from_quat(hint['eef_orientation_xyzw']).as_matrix(),GRIP[:3,:3],atol=1e-10)
    # M4's stale fallback pose is replaced by the grounded destination.
    goal=request.task.goal.target_pose
    assert goal is not None and goal.frame_id=='world'
    np.testing.assert_allclose(goal.position_m,destination,atol=1e-9)
    np.testing.assert_allclose(Rotation.from_quat(goal.orientation_xyzw).as_matrix(),BODY[:3,:3],atol=1e-10)
    assert 'held_transport_goal' not in request.task.metadata


def test_place_without_collision_points_uses_a_nominal_floor_slab():
    from tuj.m5_motion.scripted_grasps.transport import ground_held_place
    request=_tray_request(.005)
    request.task.action_type='place'
    _generic_held(request,ENTRIES[4].object_id)
    ground_held_place(request)
    hint=request.task.metadata['held_place_goal']
    destination=REGION@request.world.objects['tray']['anchors'][hint['anchor']]+REGION_POSITION
    carried_center=destination+BODY[:3,:3]@CENTER_IN_BODY
    half_height=float(np.abs(BODY[2,:3])@LOCAL_SIZE/2.)
    np.testing.assert_allclose(carried_center[2],REGION_POSITION[2]-.05+.005+.005+half_height,atol=1e-9)


def test_region_goals_avoid_objects_already_inside_the_region():
    from tuj.m5_motion.scripted_grasps.transport import (
        ground_held_place, OCCUPANT_CONTACT_TOLERANCE_M,
    )
    request=_tray_request(.005)
    request.task.action_type='place'
    request.world.objects['tray']['dimensions_m']=[.6,.4,.1]
    _generic_held(request,ENTRIES[4].object_id)
    # A plate already rests at the tray center; the spoon must land beside it.
    plate_center=REGION@np.array([.02,0,0])+REGION_POSITION
    request.world.objects['plate']={'pose':{'frame_id':'world','position_m':plate_center.tolist(),
        'orientation_xyzw':[0.,0.,0.,1.]},'dimensions_m':[.12,.12,.01],'anchors':{'center':[0.,0.,0.]}}
    ground_held_place(request)
    hint=request.task.metadata['held_place_goal']
    destination=REGION@request.world.objects['tray']['anchors'][hint['anchor']]+REGION_POSITION
    carried_center=destination+BODY[:3,:3]@CENTER_IN_BODY
    my_half=np.abs(BODY[:2,:3])@LOCAL_SIZE/2.
    # Soft nested platforms allow a small graze inside the planner margin.
    margin=max(.01,2.*.005)
    gap=np.abs(carried_center[:2]-plate_center[:2])-(my_half+np.array([.06,.06])+margin)
    assert gap.max()>=-OCCUPANT_CONTACT_TOLERANCE_M-1e-9, gap
    # ...but stays inside the tray footprint minus the wall allowance.
    tray_half=np.abs(REGION[:2,:])@np.array([.6,.4,.1])/2.
    assert np.all(np.abs(carried_center[:2]-np.asarray(REGION_POSITION[:2])-(REGION@np.array([.02,0,0]))[:2])<=tray_half-my_half-.02+1e-9)


def test_return_tool_is_not_grounded_as_a_region_place():
    from tuj.m5_motion.scripted_grasps.transport import ground_held_region_goal
    request=_tray_request(.005)
    request.task.action_type='RETURN_TOOL'
    _generic_held(request,ENTRIES[4].object_id)
    before=request.model_dump()
    ground_held_region_goal(request)
    assert request.model_dump()==before


def _plate_in_tray(request,half_xy=(.06,.06),thickness=.01):
    """A plate resting at the tray center, the spot every place aims at."""
    plate_center=REGION@np.array([.02,0,0])+REGION_POSITION
    request.world.objects['plate']={'pose':{'frame_id':'world',
        'position_m':plate_center.tolist(),'orientation_xyzw':[0.,0.,0.,1.]},
        'dimensions_m':[half_xy[0]*2,half_xy[1]*2,thickness],'anchors':{'center':[0.,0.,0.]}}
    return plate_center


def test_place_uses_the_slot_the_plan_assigned_inside_the_region():
    """M2 divides a shared container; the place must honour its own slot."""
    from tuj.m5_motion.scripted_grasps.transport import ground_held_place
    request=_tray_request(.005)
    request.task.action_type='place'
    request.world.objects['tray']['dimensions_m']=[.6,.4,.1]
    _generic_held(request,ENTRIES[4].object_id)
    request.task.metadata['action_parameters']={'placement_slot':{
        'region':'obj_tray_tray','uv':[.5,-.5],'source':'m2_container_layout'}}
    ground_held_place(request)
    hint=request.task.metadata['held_place_goal']
    destination=REGION@request.world.objects['tray']['anchors'][hint['anchor']]+REGION_POSITION
    carried_center=destination+BODY[:3,:3]@CENTER_IN_BODY
    region_center=(REGION@np.array([.02,0,0])+REGION_POSITION)[:2]
    inner=np.array([.6,.4])*.5-.02
    expected=region_center+(REGION@np.r_[np.array([.5,-.5])*inner,0.])[:2]
    np.testing.assert_allclose(carried_center[:2],expected,atol=1e-9)


def test_rotated_region_slot_clips_in_region_frame_not_world_aabb():
    """A near-edge slot must remain inside a rotated region's local footprint."""
    from tuj.m5_motion.scripted_grasps.transport import ground_held_place
    request=_tray_request(.005)
    request.task.action_type='place'
    request.world.objects['tray']['dimensions_m']=[.16755,.16709,.01]
    _generic_held(request,ENTRIES[4].object_id)
    request.world.objects[ENTRIES[4].object_id]['dimensions_m']=[.06421,.09998,.05342]
    request.task.metadata['action_parameters']={'placement_slot':{
        'region':'obj_tray_tray','uv':[-.4869,0.],
        'source':'m2_container_layout'}}
    ground_held_place(request)
    hint=request.task.metadata['held_place_goal']
    destination=REGION@request.world.objects['tray']['anchors'][hint['anchor']]+REGION_POSITION
    destination_rotation=Rotation.from_quat(
        hint['object_orientation_xyzw']).as_matrix()
    carried_center=destination+destination_rotation@CENTER_IN_BODY
    center_local=(REGION.T@(
        np.r_[carried_center[:2],REGION_POSITION[2]]-np.asarray(REGION_POSITION)
    ))[:2]
    object_in_region=REGION.T@destination_rotation
    object_half=np.abs(object_in_region[:2,:])@np.array([.06421,.09998,.05342])*.5
    limit=np.array([.16755,.16709])*.5-.02-object_half
    assert np.all(np.abs(center_local-np.array([.02,0.]))<=limit+1e-9)


def test_tray_place_avoids_stacking_on_nested_platform():
    """Plate on a tray is a stacking-forbidden support; seat the tray floor."""
    from tuj.m5_motion.scripted_grasps.transport import (
        ground_held_place, OCCUPANT_CONTACT_TOLERANCE_M,
    )
    request = _tray_request(.005)
    request.task.action_type = 'place'
    request.world.objects['tray']['dimensions_m'] = [.6, .4, .1]
    _generic_held(request, ENTRIES[4].object_id)
    # Plate covers the default centre slot but leaves free tray floor beside it.
    plate_center = _plate_in_tray(request, half_xy=(.08, .08), thickness=.01)
    request.task.metadata['action_parameters'] = {'placement_slot': {
        'region': 'obj_tray_tray', 'uv': [0., 0.], 'source': 'm2_container_layout'}}
    ground_held_place(request)
    hint = request.task.metadata['held_place_goal']
    destination = REGION @ request.world.objects['tray']['anchors'][hint['anchor']] + REGION_POSITION
    carried_center = destination + BODY[:3, :3] @ CENTER_IN_BODY
    my_half = np.abs(BODY[:2, :3]) @ LOCAL_SIZE / 2.
    plate_half = np.array([.08, .08])
    gap = np.max(np.abs(carried_center[:2] - plate_center[:2]) - (my_half + plate_half))
    # Soft contact allowed; deep seating on the plate is not.
    assert gap >= -OCCUPANT_CONTACT_TOLERANCE_M - 1e-9, gap
    half_height = float(np.abs(BODY[2, :3]) @ LOCAL_SIZE / 2.)
    # Region floor, not the plate top.
    assert carried_center[2] - half_height < plate_center[2] + .005 + 1e-6


def test_a_full_region_prefers_least_overlap_not_stacking_on_occupant():
    """No clear tray floor left: minimize penetration, still do not stack on plate."""
    from tuj.m5_motion.scripted_grasps.transport import ground_held_place
    request = _tray_request(.005)
    request.task.action_type = 'place'
    _generic_held(request, ENTRIES[4].object_id)
    plate_center = _plate_in_tray(request, half_xy=(.14, .09), thickness=.01)
    ground_held_place(request)
    hint = request.task.metadata['held_place_goal']
    destination = REGION @ request.world.objects['tray']['anchors'][hint['anchor']] + REGION_POSITION
    carried_center = destination + BODY[:3, :3] @ CENTER_IN_BODY
    my_half = np.abs(BODY[:2, :3]) @ LOCAL_SIZE / 2.
    # Must not sit fully on the plate (old stacking behaviour).
    assert np.any(
        np.abs(carried_center[:2] - plate_center[:2]) + my_half
        > np.array([.14, .09]) + 1e-9
    )
    half_height = float(np.abs(BODY[2, :3]) @ LOCAL_SIZE / 2.)
    plate_top = plate_center[2] + .005
    # Released on the tray floor, not stacked on the plate top.
    assert carried_center[2] - half_height < plate_top - 1e-6


def test_tray_place_keeps_clear_of_taller_neighbour_without_stacking_on_plate():
    """Mug/plate on a tray: prefer free floor; never rest on the nested plate."""
    from tuj.m5_motion.scripted_grasps.transport import (
        ground_held_place, OCCUPANT_CONTACT_TOLERANCE_M,
    )
    request = _tray_request(.005)
    request.task.action_type = 'place'
    # Wide tray with plate+mug on the -x half and free floor on +x.
    request.world.objects['tray'] = {
        'pose': {'frame_id': 'world', 'position_m': REGION_POSITION,
                 'orientation_xyzw': [0., 0., 0., 1.]},
        'dimensions_m': [.60, .50, .064], 'anchors': {'center': [0., 0., 0.]}}
    plate_center = np.asarray(REGION_POSITION) + np.array([-.08, 0., -.01])
    request.world.objects['plate'] = {
        'pose': {'frame_id': 'world', 'position_m': plate_center.tolist(),
                 'orientation_xyzw': [0., 0., 0., 1.]},
        'dimensions_m': [.22, .40, .012], 'anchors': {'center': [0., 0., 0.]}}
    mug_center = np.asarray(REGION_POSITION) + np.array([-.18, 0., .04])
    request.world.objects['mug'] = {
        'pose': {'frame_id': 'world', 'position_m': mug_center.tolist(),
                 'orientation_xyzw': [0., 0., 0., 1.]},
        'dimensions_m': [.09, .09, .09], 'anchors': {'center': [0., 0., 0.]}}
    _generic_held(request, ENTRIES[4].object_id)
    request.task.metadata['action_parameters'] = {'placement_slot': {
        'region': 'obj_tray_tray', 'uv': [-.5, 0.], 'source': 'm2_container_layout'}}
    ground_held_place(request)
    hint = request.task.metadata['held_place_goal']
    destination = np.asarray(
        request.world.objects['tray']['anchors'][hint['anchor']]) + REGION_POSITION
    carried_center = destination + BODY[:3, :3] @ CENTER_IN_BODY
    my_half = np.abs(BODY[:2, :3]) @ LOCAL_SIZE / 2.
    mug_gap = np.max(
        np.abs(carried_center[:2] - mug_center[:2]) - (my_half + np.array([.045, .045])))
    plate_gap = np.max(
        np.abs(carried_center[:2] - plate_center[:2])
        - (my_half + np.array([.11, .20])))
    # Mug is a soft nested platform too (broad); both may graze within tolerance.
    assert mug_gap >= -OCCUPANT_CONTACT_TOLERANCE_M - 1e-9, mug_gap
    assert plate_gap >= -OCCUPANT_CONTACT_TOLERANCE_M - 1e-9, plate_gap
    half_height = float(np.abs(BODY[2, :3]) @ LOCAL_SIZE / 2.)
    plate_top = plate_center[2] + .006
    assert carried_center[2] - half_height < plate_top - 1e-6
    # Prefer the free +x tray floor over the plate centre.
    assert carried_center[0] > plate_center[0]


def _vacuum_place_request(margin=.005, *, object_id='held_vac', support_floor_z=-.029,
                          local_size=(.16,.16,.01), body_z=.80, grip_z=.795):
    """World-held vacuum place: thin top-grasp so TCP sits near the object bottom."""
    from tuj.m5_motion.scripted_grasps.registry import ENTRIES as _ENTRIES
    vac_entry = next(e for e in _ENTRIES if e.ee == 'vac')
    request = request_for(vac_entry, action='place', object_id=object_id)
    request.constraints.collision_margin_m = margin
    request.task.goal.target_region_id = 'tray'
    request.task.ee = 'vac'
    request.world.metadata['physical_active_ee'] = 'vac'
    request.world.objects['tray'] = {
        'pose': {'frame_id': 'world', 'position_m': REGION_POSITION,
                 'orientation_xyzw': Rotation.from_matrix(REGION).as_quat().tolist()},
        'dimensions_m': [.3, .2, .1], 'anchors': {'center': [.02, 0., 0.]},
        'collision_points_m': _tray_collision_points(inner_floor_z=support_floor_z),
    }
    body = transform([.2, .1, body_z],
                     rotation=Rotation.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix())
    grip = transform([.2, .1, grip_z],
                     rotation=Rotation.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix())
    size = np.asarray(local_size, dtype=float)
    center = np.zeros(3)
    T_GB = inverse(grip) @ body
    request.world.robot_state.held_tool_id = object_id
    request.world.robot_state.attached_object_id = object_id
    request.world.robot_state.eef_pose = Pose(
        frame_id='world', position_m=tuple(grip[:3, 3]),
        orientation_xyzw=tuple(Rotation.from_matrix(grip[:3, :3]).as_quat()))
    payload = {
        'object_id': object_id, 'free_joint_name': f'{object_id}_joint0',
        'reference_kind': 'site', 'reference_name': 'gripper0_right_grip_site',
        'position_in_reference_m': T_GB[:3, 3].tolist(),
        'orientation_in_reference_xyzw': Rotation.from_matrix(T_GB[:3, :3]).as_quat().tolist(),
    }
    request.world.metadata['attached_object_transforms'] = {object_id: payload}
    request.world.objects[object_id] = {
        'pose': {'frame_id': 'world', 'position_m': body[:3, 3].tolist(),
                 'orientation_xyzw': Rotation.from_matrix(body[:3, :3]).as_quat().tolist()},
        'dimensions_m': size.tolist(), 'anchors': {'center': center.tolist()},
    }
    return request, body, grip, size


def test_vacuum_held_place_raises_tcp_above_support_by_cup_half_height_and_margin():
    from tuj.m5_motion.scripted_grasps.transport import (
        VACUUM_CUP_HALF_HEIGHT_M, ground_held_place, _retargeted_tcp_world_z,
        _vacuum_place_release_clearance_m,
    )
    from types import SimpleNamespace
    margin = .005
    request, body, grip, size = _vacuum_place_request(margin)
    ground_held_place(request)
    hint = request.task.metadata['held_place_goal']
    destination = transform(
        request.task.goal.target_pose.position_m,
        quaternion_xyzw=request.task.goal.target_pose.orientation_xyzw,
    )
    tcp_z = _retargeted_tcp_world_z(grip, body, destination)
    floor_top_world = REGION_POSITION[2] - 0.029
    assert tcp_z >= floor_top_world + VACUUM_CUP_HALF_HEIGHT_M + margin - 1e-9
    half = np.abs(body[:3, :3]) @ (size / 2.)
    geo = _vacuum_place_release_clearance_m(
        SimpleNamespace(half=half), margin, margin)
    assert hint['release_clearance_m'] >= geo - 1e-12
    assert 'plate_b' not in str(hint)
    assert 'c3_2' not in str(hint).lower()


def test_vacuum_held_place_keeps_object_seating_then_adds_ee_lift_only_when_needed():
    from tuj.m5_motion.scripted_grasps.transport import (
        VACUUM_CUP_HALF_HEIGHT_M, ground_held_place, _retargeted_tcp_world_z,
        _vacuum_place_release_clearance_m,
    )
    from types import SimpleNamespace
    margin = .005
    # Grip already far above the object: object seating alone must clear EE.
    request, body, grip, size = _vacuum_place_request(
        margin, body_z=.90, grip_z=.98, local_size=(.08, .08, .09))
    ground_held_place(request)
    hint = request.task.metadata['held_place_goal']
    destination = transform(
        request.task.goal.target_pose.position_m,
        quaternion_xyzw=request.task.goal.target_pose.orientation_xyzw,
    )
    floor_top_world = REGION_POSITION[2] - 0.029
    assert hint.get('vacuum_ee_clearance_lift_m', 0.) == pytest.approx(0.)
    half = np.abs(body[:3, :3]) @ (size / 2.)
    geo = _vacuum_place_release_clearance_m(
        SimpleNamespace(half=half), margin, margin)
    assert hint['release_clearance_m'] == pytest.approx(geo)
    tcp_z = _retargeted_tcp_world_z(grip, body, destination)
    assert tcp_z >= floor_top_world + VACUUM_CUP_HALF_HEIGHT_M + margin - 1e-9


def test_vacuum_place_release_clearance_scales_with_held_object_extent():
    """Bread-like extents need more attached-place gap than thin plates."""
    from types import SimpleNamespace
    from tuj.m5_motion.scripted_grasps.transport import (
        _vacuum_place_release_clearance_m,
    )
    margin = .005
    plate = _vacuum_place_release_clearance_m(
        SimpleNamespace(half=np.array([.08, .08, .005])), margin, margin)
    bread = _vacuum_place_release_clearance_m(
        SimpleNamespace(half=np.array([.032, .05, .027])), margin, margin)
    assert plate == pytest.approx(margin + 0.15 * np.linalg.norm([.08, .08]))
    assert bread > plate
    assert bread == pytest.approx(.027 + 0.15 * np.linalg.norm([.032, .05]))
    # Live bread place timed out at ~36 mm tracking error with 10 mm seat.
    assert bread >= .03


def test_vacuum_held_place_tracks_higher_support_floor():
    from tuj.m5_motion.scripted_grasps.transport import (
        VACUUM_CUP_HALF_HEIGHT_M, ground_held_place, _retargeted_tcp_world_z,
    )
    margin = .005
    raised_floor = -0.010
    request, body, grip, size = _vacuum_place_request(
        margin, support_floor_z=raised_floor, body_z=.80, grip_z=.795,
        local_size=(.16, .16, .01))
    ground_held_place(request)
    destination = transform(
        request.task.goal.target_pose.position_m,
        quaternion_xyzw=request.task.goal.target_pose.orientation_xyzw,
    )
    floor_top_world = REGION_POSITION[2] + raised_floor
    tcp_z = _retargeted_tcp_world_z(grip, body, destination)
    assert tcp_z >= floor_top_world + VACUUM_CUP_HALF_HEIGHT_M + margin - 1e-9


def test_vacuum_held_place_logic_is_not_tied_to_a_scene_instance_name():
    from tuj.m5_motion.scripted_grasps.transport import ground_held_place
    for object_id in ('lid', 'held_vac_a', 'held_vac_b'):
        request, _body, _grip, _size = _vacuum_place_request(object_id=object_id)
        ground_held_place(request)
        hint = request.task.metadata['held_place_goal']
        assert hint['object_id'] == object_id
        assert request.task.ee == 'vac'


def test_vacuum_place_slot_reserves_off_center_cup_footprint_from_region_rim():
    from tuj.m5_motion.scripted_grasps.transport import (
        REGION_WALL_ALLOWANCE_M, VACUUM_CUP_RADIUS_M, ground_held_place,
    )
    request, body, grip, _size = _vacuum_place_request(
        object_id='generic_plate', local_size=(.1675, .1671, .0102),
        body_z=.90, grip_z=.895)
    request.task.metadata['action_parameters']={'placement_slot':{
        'region':'obj_tray_tray','uv':[-.49,0.],
        'source':'m2_container_layout'}}
    # Reproduce a rim grasp: the cup is displaced from the plate body center.
    grip[:3, 3] = body[:3, 3] + body[:3, :3] @ np.array([.048, 0., -.005])
    request.world.robot_state.eef_pose = Pose(
        frame_id='world', position_m=tuple(grip[:3, 3]),
        orientation_xyzw=tuple(Rotation.from_matrix(grip[:3, :3]).as_quat()))
    t_gb=inverse(grip)@body
    request.world.metadata['attached_object_transforms']['generic_plate'].update({
        'position_in_reference_m':t_gb[:3,3].tolist(),
        'orientation_in_reference_xyzw':Rotation.from_matrix(
            t_gb[:3,:3]).as_quat().tolist(),
    })
    ground_held_place(request)
    destination=transform(
        request.task.goal.target_pose.position_m,
        quaternion_xyzw=request.task.goal.target_pose.orientation_xyzw)
    t_be=inverse(body)@grip
    tcp=destination@t_be
    tcp_local=REGION.T@(tcp[:3,3]-np.asarray(REGION_POSITION))
    tray_dims=np.asarray(request.world.objects['tray']['dimensions_m'])
    margin=float(request.constraints.collision_margin_m)
    mesh_pad=max(2.0*margin, .01)
    limit=(
        tray_dims[:2]*.5
        -REGION_WALL_ALLOWANCE_M
        -VACUUM_CUP_RADIUS_M
        -mesh_pad
        -0.001
    )
    assert np.all(np.abs(tcp_local[:2]-np.array([.02,0.]))<=limit+1e-6)


def test_vacuum_plate_place_slot_pulls_eccentric_cup_off_live_tray_rim():
    """Live plate_a failure: AABB cup clip was satisfied but rim geom was not."""
    import json
    from pathlib import Path
    from tuj.m5_motion.schema import MotionPlanRequest
    from tuj.m5_motion.scripted_grasps.transport import (
        ground_held_place, HELD_PLACE_GOAL_ANCHOR, VACUUM_CUP_RADIUS_M,
        REGION_WALL_ALLOWANCE_M, _grounding_for, _is_region_place,
    )
    from tuj.m5_motion.scripted_grasps.frames import inverse, transform

    root = Path('output/c3_2/m5/live/run-20260913-162930-c5faac98')
    if not root.exists():
        pytest.skip('live failure artifact not present')
    man = json.loads((root / 'live-execution-manifest.json').read_text(encoding='utf-8'))
    raw = json.loads(Path(man['steps'][3]['request']).read_text(encoding='utf-8'))
    req = MotionPlanRequest.model_validate(raw)
    req.task.metadata.pop(HELD_PLACE_GOAL_ANCHOR, None)
    ground_held_place(req)
    g = _grounding_for(req, None, _is_region_place)
    dest = transform(
        req.task.goal.target_pose.position_m,
        quaternion_xyzw=req.task.goal.target_pose.orientation_xyzw,
    )
    t_be = inverse(g.T_WB) @ g.T_WE
    tcp_local = (inverse(g.T_WR) @ np.r_[(dest @ t_be)[:3, 3], 1])[:3]
    mesh_pad = max(2.0 * float(req.constraints.collision_margin_m), 0.01)
    limit = (
        g.region_dims[:2] * .5
        - REGION_WALL_ALLOWANCE_M
        - VACUUM_CUP_RADIUS_M
        - mesh_pad
        - 0.001
    )
    assert np.all(
        np.abs(tcp_local[:2] - g.region_center_local[:2]) <= limit + 1e-6
    )
    # Must move off the failing M2 uv seat (cup was at ~-0.123 m in region y).
    assert abs(float(tcp_local[1] - g.region_center_local[1])) < 0.123 - 1e-4



def test_packed_region_place_ignores_enclosing_container_as_occupant():
    """Tray under a plate must not force bread onto fork/spoon already on the plate."""
    import json
    from pathlib import Path
    from tuj.m5_motion.schema import MotionPlanRequest
    from tuj.m5_motion.scripted_grasps.transport import (
        ground_held_place, HELD_PLACE_GOAL_ANCHOR, _grounding_for, _is_region_place,
    )

    root = Path('output/c3_2/m5/live/run-20260913-170640-631341be')
    if not root.exists():
        pytest.skip('live failure artifact not present')
    man = json.loads((root / 'live-execution-manifest.json').read_text(encoding='utf-8'))
    raw = json.loads(Path(man['steps'][23]['request']).read_text(encoding='utf-8'))
    req = MotionPlanRequest.model_validate(raw)
    g = _grounding_for(req, None, _is_region_place)
    occupants = g._occupants()
    assert all(
        float(np.linalg.norm(c - g.region_world[:2])) < 0.2
        for c, _h, _t, _s in occupants
    )
    # Enclosing tray is no longer an occupant of plate_a.
    tray = np.asarray(raw['world']['objects']['tray_a']['pose']['position_m'][:2])
    assert all(float(np.linalg.norm(c - tray)) > 1e-6 for c, _h, _t, _s in occupants)
    # Utensils on the plate are hard blockers (not soft nested platforms).
    assert occupants and all(not soft for _c, _h, _t, soft in occupants)
    fork = np.asarray(raw['world']['objects']['fork_a']['pose']['position_m'][:2])
    failing = np.asarray([2.197513799663473, -3.2477456935174187])
    req.task.metadata.pop(HELD_PLACE_GOAL_ANCHOR, None)
    req.world.objects['plate_a'].setdefault('anchors', {}).pop(
        HELD_PLACE_GOAL_ANCHOR, None)
    ground_held_place(req)
    seat = np.asarray(
        req.task.metadata[HELD_PLACE_GOAL_ANCHOR]['destination_center_xy_m'])
    assert float(np.linalg.norm(seat - fork)) > float(
        np.linalg.norm(failing - fork)) + 0.04


def test_non_vacuum_held_place_ignores_vacuum_ee_clearance_branch():
    """Existing 2F spoon place seating must not gain vacuum cup lift."""
    from tuj.m5_motion.scripted_grasps.transport import ground_held_place
    request = _tray_request(.005)
    request.task.action_type = 'place'
    request.world.objects['tray']['collision_points_m'] = _tray_collision_points()
    _generic_held(request, ENTRIES[4].object_id)
    ground_held_place(request)
    hint = request.task.metadata['held_place_goal']
    assert hint['release_clearance_m'] == pytest.approx(.005)
    assert 'vacuum_ee_clearance_lift_m' not in hint


def _crowded_place_request():
    """Held object + in-region neighbor for place-margin repair tests."""
    from tuj.m5_motion.scripted_grasps.transport import ground_held_place
    request = _tray_request(.005)
    request.task.action_type = 'place'
    request.task.metadata['scripted_m4_implicit_object_pose'] = True
    request.world.objects['tray']['collision_points_m'] = _tray_collision_points()
    object_id = ENTRIES[4].object_id
    _generic_held(request, object_id)
    slot_xy = (REGION @ np.array([.02, 0., 0.]) + REGION_POSITION)[:2]
    neighbor_xy = slot_xy + np.array([.03, .0])
    request.world.objects['neighbor'] = {
        'object_id': 'neighbor',
        'free_joint_name': 'neighbor_free',
        'pose': {
            'frame_id': 'world',
            'position_m': [float(neighbor_xy[0]), float(neighbor_xy[1]), float(REGION_POSITION[2])],
            'orientation_xyzw': [0., 0., 0., 1.],
        },
        'dimensions_m': [.04, .04, .02],
        'anchors': {'center': [0., 0., 0.]},
    }
    ground_held_place(request)
    return request, slot_xy, neighbor_xy


def test_reground_held_place_moves_away_from_hand_neighbor_margin_partner():
    from tuj.m5_motion.scripted_grasps.transport import (
        reground_held_place_from_collision_feedback,
    )
    request, _slot_xy, neighbor_xy = _crowded_place_request()
    first = np.asarray(
        request.task.metadata['held_place_goal']['destination_center_xy_m'],
        dtype=float,
    )
    feedback = {
        'contract_version': 'COLLISION_REPAIR_V1',
        'repair_attempt': 1,
        'maximum_repair_attempts': 2,
        'required_collision_margin_m': 0.005,
        'failed_strategies': [{
            'source_repair_attempt': 0,
            'strategy_id': 'place-1',
            'failure_code': 'COLLISION_FILTERED_ALL',
            'ik_diagnostics': [],
            'collision_observations': [{
                'geometry_a': 'gripper0_right_hand_collision',
                'geometry_b': 'neighbor_g0',
                'measured_clearance_m': 0.003635,
                'required_clearance_m': 0.005,
            }],
        }],
    }
    assert reground_held_place_from_collision_feedback(request, feedback) is True
    second = np.asarray(
        request.task.metadata['held_place_goal']['destination_center_xy_m'],
        dtype=float,
    )
    assert float(np.linalg.norm(second - first)) >= 0.005 - 1e-9
    assert float(np.linalg.norm(second - neighbor_xy)) > float(
        np.linalg.norm(first - neighbor_xy)
    ) - 1e-9
    assert feedback['reground_place_partner_id'] == 'neighbor'
    assert feedback['rejected_place_xy_m']
    feedback2 = {
        **feedback,
        'repair_attempt': 2,
        'rejected_place_xy_m': list(feedback['rejected_place_xy_m']),
    }
    assert reground_held_place_from_collision_feedback(request, feedback2) is True
    third = np.asarray(
        request.task.metadata['held_place_goal']['destination_center_xy_m'],
        dtype=float,
    )
    assert float(np.linalg.norm(third - second)) >= 0.005 - 1e-9
    assert float(np.linalg.norm(third - first)) >= 0.005 - 1e-9


def test_reground_held_place_ignores_non_robot_margin_observations():
    from tuj.m5_motion.scripted_grasps.transport import (
        reground_held_place_from_collision_feedback,
    )
    request, _slot_xy, _neighbor_xy = _crowded_place_request()
    before = request.task.metadata['held_place_goal']['destination_center_xy_m']
    feedback = {
        'failed_strategies': [{
            'collision_observations': [{
                'geometry_a': 'held_object_g0',
                'geometry_b': 'neighbor_g0',
                'measured_clearance_m': 0.001,
                'required_clearance_m': 0.005,
            }],
        }],
    }
    assert reground_held_place_from_collision_feedback(request, feedback) is False
    assert request.task.metadata['held_place_goal']['destination_center_xy_m'] == before
