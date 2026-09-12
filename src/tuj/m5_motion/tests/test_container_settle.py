from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
from tuj.m5_motion.container_settle import evaluate_container_settle, configure_container_settle


def fixture():
    pose={'frame_id':'world','position_m':[0.,0.,0.], 'orientation_xyzw':[0.,0.,0.,1.]}
    box={'pose':deepcopy(pose),'dimensions_m':[1.,1.,1.], 'packing_metadata':{'kind':'CONTAINER','interior_dimensions_m':[1.,1.,1.],'interior_center_m':[0.,0.,0.]}}
    item={'pose':deepcopy(pose),'dimensions_m':[.1,.1,.1], 'packing_metadata':{'kind':'PACKABLE_OBJECT'}}
    config={'region_id':'box','target_ids':['new','prior'],'objects':{'box':box,'new':item,'prior':deepcopy(item)},'position_tolerance_m':.005,'orientation_tolerance_rad':.01}
    poses={n:deepcopy(v['pose']) for n,v in config['objects'].items()}
    return config,poses


def test_first_observation_does_not_establish_stability():
    config,poses=fixture()
    assert not evaluate_container_settle(config,poses,None)['succeeded']
    assert evaluate_container_settle(config,poses,deepcopy(poses))['succeeded']


@pytest.mark.parametrize('name',['new','prior'])
def test_every_required_object_must_remain_inside(name):
    config,poses=fixture();poses[name]['position_m'][0]=.6
    assert not evaluate_container_settle(config,poses,deepcopy(poses))['succeeded']


@pytest.mark.parametrize('kwargs',[{'attached':True},{'hand_contacts':1}])
def test_supported_or_attached_object_is_not_settled(kwargs):
    config,poses=fixture()
    assert not evaluate_container_settle(config,poses,deepcopy(poses),**kwargs)['succeeded']


def test_moving_object_inside_is_not_stable_and_geometry_is_unchanged():
    config,poses=fixture();before=deepcopy(config);previous=deepcopy(poses)
    poses['new']['position_m'][0]=.02
    result=evaluate_container_settle(config,poses,previous)
    assert not result['succeeded'] and not result['stable']
    assert config==before


def test_configuration_preserves_arm_tolerances_and_includes_prior_items():
    config,poses=fixture()
    request=NS(task=NS(goal=NS(target_region_id='box',target_object_id='new'),metadata={'held_place_goal':{}}),world=NS(objects=config['objects'],metadata={'free_object_settle_seconds':5.}),constraints=NS(position_tolerance_m=.005,orientation_tolerance_rad=.01))
    request.task.metadata['held_place_goal']={'anchor':'destination'}
    segment=NS(segment_type=NS(value='RETREAT'),metadata={'tracking_settle':{'eef_tolerance_m':.003}})
    plan=NS(events=[NS(event_type=NS(value='DETACH_OBJECT'),target_id='new')],segments=[segment])
    configure_container_settle(request,plan)
    assert segment.metadata['tracking_settle']['eef_tolerance_m']==.003
    assert segment.metadata['tracking_settle']['max_wait_s']==5.
    assert segment.metadata['container_settle']['target_ids']==['new','prior']
    assert segment.metadata['container_settle']['objects']==config['objects']
