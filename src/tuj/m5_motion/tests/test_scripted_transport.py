from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from tuj.m5_motion.scripted_grasps.frames import transform
from tuj.m5_motion.scripted_grasps.transport import ground_held_transport
from tuj.m5_motion.geometry import tool_rotation_from_axis
from tuj.m5_motion.tests.test_scripted_grasps import request_for,ENTRIES


def test_region_transport_preserves_measured_grasp_offset_and_orientation():
    request=request_for(ENTRIES[4],action='transport')
    request.task.metadata['scripted_m4_implicit_object_pose']=True
    request.task.goal.target_region_id='tray'
    region=Rotation.from_euler('z',37,degrees=True).as_matrix()
    request.world.objects['tray']={'pose':{'frame_id':'world','position_m':[-.5,.3,.7],
        'orientation_xyzw':Rotation.from_matrix(region).as_quat().tolist()},
        'dimensions_m':[.3,.2,.1],'anchors':{'center':[.02,0.,0.]}}
    body=transform([.2,.1,.8],rotation=Rotation.from_euler('z',25,degrees=True).as_matrix())
    grip=transform([.24,.09,.89],rotation=Rotation.from_euler('xyz',[180,0,30],degrees=True).as_matrix())
    center_in_body=np.array([.01,-.02,0.])
    context=SimpleNamespace(body_pose=lambda:body,grip_pose=lambda:grip,
        center_in_body=center_in_body,local_size=np.array([.08,.08,.09]))
    retention=SimpleNamespace(context=context,entry=ENTRIES[4])
    ground_held_transport(request,retention)
    hint=request.task.metadata['held_transport_goal']
    rec=request.world.objects['tray']
    destination=region@rec['anchors'][hint['anchor']]+[-.5,.3,.7]
    object_center=(body@np.r_[center_in_body,1.])[:3]
    carried_center=object_center+(destination-grip[:3,3])
    np.testing.assert_allclose(carried_center[:2],(region@np.array([.02,0,0])+[-.5,.3,.7])[:2])
    assert carried_center[2]>=.7+.05+.045+.05-1e-12
    np.testing.assert_allclose(tool_rotation_from_axis(-region@hint['approach_axis_xyz'],hint['roll_rad']),grip[:3,:3],atol=1e-10)
    assert request.task.goal.target_pose is None
    assert hint['frame_ref']=='object:tray'


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
