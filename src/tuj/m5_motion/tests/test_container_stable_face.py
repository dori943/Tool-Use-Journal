from itertools import product
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tuj.m5_motion.tests.test_scripted_transport import _tray_request, _generic_held, ENTRIES
from tuj.m5_motion.scripted_grasps.transport import _Grounding, ground_held_place


def request():
    req=_tray_request(.005);req.task.action_type='place';oid=ENTRIES[4].object_id
    _generic_held(req,oid)
    obj=req.world.objects[oid];obj['dimensions_m']=[.04,.12,.15]
    obj['anchors']['center']=[.01,-.02,.015]
    obj['collision_points_m']=(np.array(list(product([-1,1],repeat=3)))*np.array([.04,.12,.15])/2+obj['anchors']['center']).tolist()
    obj['packing_metadata']={'stable_face_policy':'MINIMUM_HEIGHT'}
    reg=req.world.objects['tray'];reg['dimensions_m']=[.32,.26,.21];reg['anchors']['center']=[0.,0.,.105]
    reg['packing_metadata']={'kind':'CONTAINER','interior_dimensions_m':[.3,.24,.2],'interior_center_m':[0.,0.,.105],'opening_top_z_m':.205}
    return req,oid


def test_stable_face_minimizes_height_with_nonzero_body_center():
    req,oid=request();g=_Grounding(req,None,oid);floor=g.floor_top_world_z()
    ground_held_place(req)
    pose=req.task.goal.target_pose;rot=Rotation.from_quat(pose.orientation_xyzw).as_matrix()
    points=np.array(req.world.objects[oid]['collision_points_m'])@rot.T+pose.position_m
    assert np.ptp(points[:,2])==pytest.approx(.04)
    assert points[:,2].min()==pytest.approx(floor+.005)
    assert np.linalg.det(rot)==pytest.approx(1.)


def test_unmarked_object_keeps_existing_orientation():
    req,oid=request();req.world.objects[oid]['packing_metadata']={}
    g=_Grounding(req,None,oid)
    np.testing.assert_allclose(g.destination_rotation,g.T_WB[:3,:3])


def test_stable_face_uses_free_floor_before_stacking():
    req,oid=request();g=_Grounding(req,None,oid)
    # The selected first face is stable and does not invent extra support height.
    assert g.support_top_world_z(g.stable_face_xy)[1] is False
    assert g.half[2]==pytest.approx(.02)


def test_live_ik_filter_uses_measured_object_to_hand_transform(monkeypatch):
    from types import SimpleNamespace
    from tuj.m5_motion.scripted_grasps.container_orientation import stable_face_has_ik
    monkeypatch.setattr('tuj.m5_motion.scripted_grasps.container_release.clear_container_rim',
                        lambda g, target, retention: (target, {}))
    req,oid=request();g=_Grounding(req,None,oid)
    calls=[]
    def solve(position, quaternion, **kwargs):
        calls.append((np.array(position),Rotation.from_quat(quaternion).as_matrix()))
        return SimpleNamespace(solutions=[])
    c=SimpleNamespace(kinematics=SimpleNamespace(solve_all_ik=solve),
                      data=SimpleNamespace(qpos=np.zeros(6)),arm_ids=np.arange(6))
    g.retention=SimpleNamespace(context=c)
    assert not stable_face_has_ik(g,g.stable_face_xy)
    pos,rot=calls[0]
    hand=np.eye(4);hand[:3,:3]=rot;hand[:3,3]=pos
    # Forward reconstruction must recover the desired object, including its
    # nonzero body-center offset, rather than mistaking object pose for hand pose.
    obj=hand@np.linalg.inv(g.T_WE)@g.T_WB
    np.testing.assert_allclose(obj[:3,:3],g.destination_rotation,atol=1e-12)
    center=obj[:3,3]+obj[:3,:3]@g.center_in_body
    np.testing.assert_allclose(center[:2],g.stable_face_xy,atol=1e-12)
    assert req.task.metadata['stable_face_ik_candidates'][0]['raw_ik_count']==0
