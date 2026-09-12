from types import SimpleNamespace as NS
from itertools import product
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tuj.m5_motion.scripted_grasps.container_orientation import configure_packing_orientation, packing_destination_center


def fixture(yaw):
    P = np.array(list(product([-.15, .15], [-.01, .01], [-.01, .01]))) + [.02, .01, -.005]
    R = Rotation.from_euler('zy', [45., 25.], degrees=True).as_matrix()
    q = Rotation.from_matrix(R).as_quat().tolist()
    rec = {'collision_points_m': P.tolist(), 'packing_metadata': {
        'orientation_frame': 'TARGET_REGION', 'orientation_candidates': [{
        'orientation_xyzw': q, 'dimensions_m': [.1,.1,.1], 'center_offset_m': [0.,0.,0.]}]}}
    T = np.eye(4); T[:3,:3] = Rotation.from_euler('z', yaw).as_matrix(); T[:3,3] = [1.,-2.,.9]
    return NS(object_id='rod', request=NS(world=NS(objects={'rod':rec}), constraints=NS(collision_margin_m=.005)),
        record={'packing_metadata': {'kind':'CONTAINER','interior_dimensions_m':[.26,.26,.26],
                'interior_center_m':[.01,-.02,.14],'opening_top_z_m':.27}},
        T_WR=T, T_WB=T.copy(), center_in_body=np.array([.02,.01,-.005]),
        destination_rotation=T[:3,:3].copy(), interior_top_world_z=lambda:1.17), P


@pytest.mark.parametrize('yaw',[0.,.7,-1.2])
def test_metadata_rotation_uses_real_vertices_and_region_frame(yaw):
    g,P=fixture(yaw);configure_packing_orientation(g)
    wanted=np.array([4.,-4.,1.5])
    center=packing_destination_center(g,wanted,place=True)
    origin=center-g.destination_rotation@g.center_in_body
    points=(P@g.destination_rotation.T+origin-g.T_WR[:3,3])@g.T_WR[:3,:3]
    bounds=g.record['packing_metadata'];mid=np.array(bounds['interior_center_m']);half=np.array(bounds['interior_dimensions_m'])/2
    assert np.all(points.min(0)[:2]>= (mid-half+.005)[:2]-1e-10)
    assert np.all(points.max(0)[:2]<= (mid+half-.005)[:2]+1e-10)
    assert points[:,2].min()==pytest.approx(.275)
    chosen=g.destination_rotation.copy();g.T_WB[:3,:3]=chosen
    configure_packing_orientation(g)
    np.testing.assert_allclose(chosen,g.destination_rotation,atol=1e-10)


def test_bad_metadata_dimensions_cannot_allow_oversize_geometry():
    g,_=fixture(0.);g.request.world.objects['rod']['collision_points_m']=np.array(list(product([-1.,1.],repeat=3))).tolist()
    with pytest.raises(ValueError,match='NO_INTERIOR_FIT'):configure_packing_orientation(g)


def test_other_objects_keep_existing_path():
    g,_=fixture(0.);g.request.world.objects['rod']['packing_metadata']={}
    configure_packing_orientation(g)
    assert not hasattr(g,'packing_bounds')


def test_unreachable_nearest_orientation_uses_next_fit(monkeypatch):
    from tuj.m5_motion.scripted_grasps import container_orientation as module
    g, _ = fixture(.7)
    g.task = NS(action_type='TRANSPORT', metadata={})
    g.retention = NS()
    original = g.destination_rotation.copy()
    calls = []
    def reachable(probe):
        np.testing.assert_array_equal(g.destination_rotation, original)
        calls.append(probe.destination_rotation.copy())
        return len(calls) == 2
    monkeypatch.setattr(module, 'packing_transport_has_ik', reachable)
    configure_packing_orientation(g)
    assert len(calls) == 2
    np.testing.assert_allclose(g.destination_rotation, calls[1])
    assert not np.allclose(calls[0], calls[1])


def test_all_fitting_orientations_unreachable_fail(monkeypatch):
    from tuj.m5_motion.scripted_grasps import container_orientation as module
    g, _ = fixture(0.)
    g.task = NS(action_type='TRANSPORT', metadata={}); g.retention = NS()
    original = g.destination_rotation.copy()
    monkeypatch.setattr(module, 'packing_transport_has_ik', lambda probe: False)
    with pytest.raises(ValueError, match='NO_REACHABLE_FIT'):
        configure_packing_orientation(g)
    np.testing.assert_array_equal(g.destination_rotation, original)
    assert not hasattr(g, 'packing_bounds')


@pytest.mark.parametrize('yaw', [-.7, .5])
def test_ik_probe_equals_published_transport_goal_with_rotated_region(monkeypatch, yaw):
    from tuj.m5_motion.tests.test_container_stable_face import request
    from tuj.m5_motion.scripted_grasps.transport import _Grounding, ground_held_transport
    from tuj.m5_motion.scripted_grasps.frames import inverse, transform
    req, oid = request()
    req.task.action_type = 'TRANSPORT'
    req.task.metadata['scripted_m4_implicit_object_pose'] = True
    req.world.objects[oid]['packing_metadata'] = {
        'orientation_frame': 'TARGET_REGION', 'orientation_candidates': [{
            'orientation_xyzw': [0., 0., 0., 1.],
            'dimensions_m': [.04, .12, .15], 'center_offset_m': [.01, -.02, .015]}]}
    region = req.world.objects['tray']
    region['pose']['orientation_xyzw'] = Rotation.from_euler('z', yaw).as_quat().tolist()
    initial = _Grounding(req, None, oid)
    calls = []
    def solve(position, quaternion, **kwargs):
        calls.append(transform(position, quaternion_xyzw=quaternion))
        return NS(solutions=[object()])
    c = NS(body_pose=lambda: initial.T_WB.copy(), grip_pose=lambda: initial.T_WE.copy(),
           center_in_body=initial.center_in_body, local_size=initial.local_size,
           kinematics=NS(solve_all_ik=solve), data=NS(qpos=np.zeros(6)), arm_ids=np.arange(6))
    retention = NS(entry=NS(object_id=oid), context=c)
    def rim(g, destination, retention):
        raised = destination.copy(); raised[2, 3] += .013
        return raised, {'container_release_lift_m': .013}
    monkeypatch.setattr('tuj.m5_motion.scripted_grasps.container_release.clear_container_rim', rim)
    ground_held_transport(req, retention)
    hint = req.task.metadata['held_transport_goal']
    world_from_region = transform(region['pose']['position_m'],
                                  quaternion_xyzw=region['pose']['orientation_xyzw'])
    position = (world_from_region @ np.r_[region['anchors'][hint['anchor']], 1.])[:3]
    body = transform(position, quaternion_xyzw=hint['object_orientation_xyzw'])
    expected = body @ inverse(inverse(initial.T_WE) @ initial.T_WB)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], expected, rtol=0., atol=1e-12)
