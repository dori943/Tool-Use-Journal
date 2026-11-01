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


@pytest.mark.parametrize('missing', [['new'], ['prior'], ['box'], ['box', 'new', 'prior']])
def test_missing_current_pose_cannot_reuse_starting_geometry(missing):
    config, poses = fixture(); before = deepcopy(config); previous = deepcopy(poses)
    for name in missing:
        poses.pop(name)
    result = evaluate_container_settle(config, poses, previous)
    assert not result['succeeded'] and not result['stable']
    assert result['missing_pose_ids'] == sorted(missing)
    assert config == before


@pytest.mark.parametrize('missing', [['new'], ['prior'], ['new', 'prior']])
def test_missing_previous_target_pose_cannot_establish_stability(missing):
    config, poses = fixture(); previous = deepcopy(poses)
    for name in missing:
        previous.pop(name)
    result = evaluate_container_settle(config, poses, previous)
    assert not result['succeeded'] and not result['stable']
    assert result['missing_previous_pose_ids'] == sorted(missing)


def test_two_empty_observations_do_not_establish_stability():
    config, _ = fixture()
    result = evaluate_container_settle(config, {}, {})
    assert not result['succeeded'] and not result['stable']
    assert result['missing_pose_ids'] == ['box', 'new', 'prior']
    assert result['missing_previous_pose_ids'] == ['new', 'prior']


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


def test_measured_velocity_rejects_slow_drift_hidden_by_adjacent_ticks():
    from tuj.m5_motion.container_settle import measured_settle_velocity
    config={'settle_horizon_s':5., 'target_ids':['baguette'],
            'position_tolerance_m':.005,'orientation_tolerance_rad':.01}
    # Recorded E-run velocity: the 20 ms displacement is small, but the
    # object was still rotating/sliding and later left the container.
    velocity=[-.1589217726,-.0587637727,.0091471992,
              -.0033192076,.0066145808,-.0148342135]
    result=measured_settle_velocity(config,{'baguette':velocity})
    assert result['succeeded'] is False
    assert result['projected_position_drift_m'] > .08
    assert result['projected_orientation_drift_rad'] > .8
    assert measured_settle_velocity(config,{'baguette':[0.,0.,0.,0.,0.,0.]})['succeeded']
    assert measured_settle_velocity(config,{})['succeeded'] is False
    assert measured_settle_velocity(config,{'baguette':[float('nan')]*6})['succeeded'] is False


def test_velocity_evidence_identifies_distinct_translation_and_rotation_failures():
    from tuj.m5_motion.container_settle import measured_settle_velocity
    config = {'settle_horizon_s': 5., 'target_ids': ['still', 'sliding', 'turning'],
              'position_tolerance_m': .005, 'orientation_tolerance_rad': .05}
    velocities = {'still': [0.] * 6, 'sliding': [0., 0., 0., .002, 0., 0.],
                  'turning': [0., -.02, 0., 0., 0., 0.]}
    before = deepcopy((config, velocities))
    result = measured_settle_velocity(config, velocities)
    assert not result['succeeded']
    assert result['unstable_object_ids'] == ['sliding', 'turning']
    assert result['per_object']['still']['succeeded']
    assert result['per_object']['sliding']['projected_position_drift_m'] == pytest.approx(.01)
    assert result['per_object']['turning']['projected_orientation_drift_rad'] == pytest.approx(.1)
    assert result['per_object']['turning']['angular_velocity_world_rad_s'] == [0., -.02, 0.]
    assert result['max_linear_speed_m_s'] == .002
    assert result['max_angular_speed_rad_s'] == .02
    assert (config, velocities) == before
