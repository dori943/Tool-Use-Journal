from types import SimpleNamespace
import pytest
import numpy as np
from scipy.spatial.transform import Rotation
from tuj.m5_motion.attachment_retarget import ATTACHED_OBJECT_POSE_SUBJECT
from tuj.m5_motion.schema import Pose
from tuj.m5_motion.scripted_grasps.frames import inverse, transform
from tuj.m5_motion.scripted_grasps.transport import ground_held_transport
from tuj.m5_motion.geometry import tool_rotation_from_axis
from tuj.m5_motion.tests.test_scripted_grasps import request_for,ENTRIES


REGION=Rotation.from_euler('z',37,degrees=True).as_matrix()
REGION_POSITION=[-.5,.3,.7]
BODY=transform([.2,.1,.8],rotation=Rotation.from_euler('z',25,degrees=True).as_matrix())
GRIP=transform([.24,.09,.89],rotation=Rotation.from_euler('xyz',[180,0,30],degrees=True).as_matrix())
CENTER_IN_BODY=np.array([.01,-.02,0.])
LOCAL_SIZE=np.array([.08,.08,.09])


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
    from tuj.m5_motion.scripted_grasps.transport import ground_held_place
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
    gap=np.abs(carried_center[:2]-plate_center[:2])-(my_half+np.array([.06,.06]))
    assert gap.max()>=.01-1e-9, gap
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
    tray_half=np.abs(REGION[:2,:])@np.array([.6,.4,.1])/2.
    region_center=(REGION@np.array([.02,0,0])+REGION_POSITION)[:2]
    expected=region_center+np.array([.5,-.5])*(tray_half-.02)
    np.testing.assert_allclose(carried_center[:2],expected,atol=1e-9)


def test_a_full_region_stacks_squarely_instead_of_being_driven_into_the_occupant():
    """No free spot left: rest on the occupant's top, fully supported."""
    from tuj.m5_motion.scripted_grasps.transport import ground_held_place
    request=_tray_request(.005)
    request.task.action_type='place'
    _generic_held(request,ENTRIES[4].object_id)
    # A plate that covers the whole tray interior — nothing can land beside it.
    plate_center=_plate_in_tray(request,half_xy=(.14,.09),thickness=.01)
    ground_held_place(request)
    hint=request.task.metadata['held_place_goal']
    destination=REGION@request.world.objects['tray']['anchors'][hint['anchor']]+REGION_POSITION
    carried_center=destination+BODY[:3,:3]@CENTER_IN_BODY
    my_half=np.abs(BODY[:2,:3])@LOCAL_SIZE/2.
    # Fully on the plate, not hanging off its rim.
    assert np.all(np.abs(carried_center[:2]-plate_center[:2])+my_half
                  <=np.array([.14,.09])+1e-9)
    # Released above the plate's top, not the tray floor, and clear of the margin.
    half_height=float(np.abs(BODY[2,:3])@LOCAL_SIZE/2.)
    plate_top=plate_center[2]+.005
    assert carried_center[2]-half_height>=plate_top+.005-1e-9
    assert hint['release_clearance_m']>=.01-1e-12


def test_a_full_region_stacks_on_the_flat_support_not_astride_a_taller_neighbour():
    """The resting surface is the highest overlapped occupant, not the widest.

    c3_1: the bread's slot lay on the plate, which the search rejected as
    occupied, and then ranked a spot that was fully on the plate *by area* yet
    straddled the mug standing on it.  Only the mug ever touches the bread
    there, so the release had to clear the mug's rim by a few millimetres and
    the place never settled.
    """
    from tuj.m5_motion.scripted_grasps.transport import ground_held_place
    request=_tray_request(.005)
    request.task.action_type='place'
    # An axis-aligned tray holding a wide flat plate with a tall mug standing
    # on one end of it -- the c3_1 layout, with no free spot left beside them.
    request.world.objects['tray']={'pose':{'frame_id':'world','position_m':REGION_POSITION,
        'orientation_xyzw':[0.,0.,0.,1.]},'dimensions_m':[.40,.50,.064],
        'anchors':{'center':[0.,0.,0.]}}
    plate_center=np.asarray(REGION_POSITION)+np.array([.03,0.,-.01])
    request.world.objects['plate']={'pose':{'frame_id':'world',
        'position_m':plate_center.tolist(),'orientation_xyzw':[0.,0.,0.,1.]},
        'dimensions_m':[.27,.50,.012],'anchors':{'center':[0.,0.,0.]}}
    mug_center=np.asarray(REGION_POSITION)+np.array([-.07,0.,.04])
    request.world.objects['mug']={'pose':{'frame_id':'world',
        'position_m':mug_center.tolist(),'orientation_xyzw':[0.,0.,0.,1.]},
        'dimensions_m':[.09,.09,.09],'anchors':{'center':[0.,0.,0.]}}
    _generic_held(request,ENTRIES[4].object_id)
    request.task.metadata['action_parameters']={'placement_slot':{
        'region':'obj_tray_tray','uv':[.9,0.],'source':'m2_container_layout'}}
    ground_held_place(request)
    hint=request.task.metadata['held_place_goal']
    destination=np.asarray(request.world.objects['tray']['anchors'][hint['anchor']])+REGION_POSITION
    carried_center=destination+BODY[:3,:3]@CENTER_IN_BODY
    my_half=np.abs(BODY[:2,:3])@LOCAL_SIZE/2.
    # Clear of the mug's footprint, so nothing is balanced on its rim...
    gap=np.abs(carried_center[:2]-mug_center[:2])-(my_half+np.array([.045,.045]))
    assert gap.max()>0., gap
    # ...and released just over the plate rather than over the mug's top.
    half_height=float(np.abs(BODY[2,:3])@LOCAL_SIZE/2.)
    plate_top=plate_center[2]+.006
    assert carried_center[2]-half_height==pytest.approx(plate_top+.01)
    assert carried_center[2]-half_height<mug_center[2]+.045
