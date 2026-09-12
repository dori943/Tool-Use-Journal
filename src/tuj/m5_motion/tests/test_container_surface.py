from copy import deepcopy
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.m5_motion.container_surface import (cover_geometry, measured_surface_support,
    surface_report_result, is_container_surface_task)
from tuj.m5_motion.container_settle import configure_container_settle, evaluate_container_settle
from tuj.m5_motion.tool_use_journal import _object_record


def fixture(yaw=0., origin=(0., 0., 0.)):
    r = Rotation.from_euler('z', yaw)
    pose = lambda z: {'frame_id': 'world', 'position_m': (np.asarray(origin) + [0., 0., z]).tolist(),
                     'orientation_xyzw': r.as_quat().tolist()}
    objects = {
        'container': {'pose': pose(0), 'dimensions_m': [.2, .26, .24],
            'anchors': {'center': [0, 0, .12]},
            'packing_metadata': {'kind': 'CONTAINER', 'interior_dimensions_m': [.18, .24, .23],
                'interior_center_m': [0., 0., .125], 'opening_top_z_m': .24}},
        'cover': {'pose': pose(.246), 'dimensions_m': [.21, .27, .012],
            'solid_box_geometry': {'source': 'MUJOCO_BOX', 'half_size_m': [.105, .135, .006],
                'center_in_body_m': [0., 0., 0.], 'rotation_in_body': np.eye(3).tolist()}},
        'contents': {'pose': pose(.06), 'dimensions_m': [.05, .05, .05],
            'packing_metadata': {'kind': 'PACKABLE_OBJECT'}}}
    config = {'target_ids': ['contents', 'cover'], 'surface_target_id': 'cover', 'region_id': 'container',
        'request_id': 'request1', 'objects': objects, 'position_tolerance_m': .005,
        'orientation_tolerance_rad': .05}
    return objects, config


@pytest.mark.parametrize('yaw,origin', [(0., (0., 0., 0.)), (.7, (2., -3., .92))])
def test_cover_in_measured_region_frame(yaw, origin):
    objects, _ = fixture(yaw, origin)
    assert cover_geometry(objects, 'cover', 'container', .005, .05)['succeeded']


@pytest.mark.parametrize('failure', ['off_edge', 'hover', 'inside', 'tilt', 'hollow_unknown', 'wrong_source'])
def test_bad_cover_is_not_accepted(failure):
    objects, _ = fixture()
    target = objects['cover']
    if failure == 'off_edge': target['pose']['position_m'][0] += .02
    if failure == 'hover': target['pose']['position_m'][2] += .03
    if failure == 'inside': target['pose']['position_m'][2] -= .08
    if failure == 'tilt': target['pose']['orientation_xyzw'] = Rotation.from_euler('x', .15).as_quat().tolist()
    if failure == 'hollow_unknown': target.pop('solid_box_geometry')
    if failure == 'wrong_source': target['solid_box_geometry']['source'] = 'BBOX'
    if failure in ('hollow_unknown', 'wrong_source'):
        with pytest.raises(ValueError, match='SOLID_BOX_REQUIRED'):
            cover_geometry(objects, 'cover', 'container', .005, .05)
    else:
        assert not cover_geometry(objects, 'cover', 'container', .005, .05)['succeeded']


def test_compiled_box_descriptor_accounts_for_nested_rotated_geom():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="object" pos="1 2 3" euler="0 0 20">
        <freejoint/><body pos=".02 .03 .04" euler="20 0 0"><geom type="box" size=".1 .13 .006"/>
        <geom type="sphere" size=".2" contype="0" conaffinity="0"/></body></body></worldbody></mujoco>''')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    record = _object_record(model, data, 'cover', model.body('object').id)
    box = record['solid_box_geometry']
    np.testing.assert_allclose(box['center_in_body_m'], [.02, .03, .04], atol=1e-12)
    np.testing.assert_allclose(box['rotation_in_body'], Rotation.from_euler('x', 20, degrees=True).as_matrix(), atol=1e-12)
    assert box['half_size_m'] == [.1, .13, .006]
    model.geom_contype[1] = 1
    assert 'solid_box_geometry' not in _object_record(model, data, 'cover', model.body('object').id)


@pytest.mark.parametrize('reversed_order', [False, True])
def test_support_requires_loaded_upward_contact_at_rim(reversed_order):
    floor = '<body name="support"><geom type="box" size=".1 .13 .01" pos="0 0 .23"/></body>'
    cover = '<body name="cover" pos="0 0 .2459"><freejoint/><geom type="box" size=".105 .135 .006" mass=".1"/></body>'
    bodies = cover + floor if reversed_order else floor + cover
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>' + bodies + '</worldbody></mujoco>')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    args = (model, data, model.body('cover').id, model.body('support').id, .24, .005, .05)
    result = measured_surface_support(*args)
    assert result['contact_count'] > 0 and result['upward_force_n'] > 0
    data.qpos[2] += .1; mujoco.mj_forward(model, data)
    assert measured_surface_support(*args)['contact_count'] == 0


@pytest.mark.parametrize('failure', ['escaped_contents', 'held', 'hand_contact', 'unsupported', 'moving', 'first_sample'])
def test_cover_does_not_hide_contents_or_physical_failure(failure):
    objects, config = fixture()
    poses = {name: deepcopy(record['pose']) for name, record in objects.items()}
    previous = deepcopy(poses)
    support = {'contact_count': 4, 'upward_force_n': 1.}
    kwargs = {'attached': False, 'hand_contacts': 0, 'surface_support': support}
    if failure == 'escaped_contents': poses['contents']['position_m'][0] = .2
    if failure == 'held': kwargs['attached'] = True
    if failure == 'hand_contact': kwargs['hand_contacts'] = 1
    if failure == 'unsupported': kwargs['surface_support'] = None
    if failure == 'moving': previous['cover']['position_m'][0] += .02
    if failure == 'first_sample': previous = None
    assert not evaluate_container_settle(config, poses, previous, **kwargs)['succeeded']


def test_closing_configuration_includes_already_escaped_packable():
    objects, config = fixture(); objects['contents']['pose']['position_m'][0] = .2
    task = NS(action_type='place_on', metadata={'held_place_goal': {'anchor': 'goal'}},
              goal=NS(target_region_id='container', target_object_id='cover'))
    request = NS(request_id='request1', task=task, world=NS(objects=objects, metadata={'free_object_settle_seconds': 5.}),
                 constraints=NS(position_tolerance_m=.005, orientation_tolerance_rad=.05))
    segment = NS(segment_type=NS(value='RETREAT'), metadata={})
    plan = NS(events=[NS(event_type=NS(value='DETACH_OBJECT'), target_id='cover')], segments=[segment])
    configure_container_settle(request, plan)
    assert segment.metadata['container_settle']['target_ids'] == ['contents', 'cover']
    assert segment.metadata['container_settle']['surface_target_id'] == 'cover'
    task.action_type = 'place'
    assert not is_container_surface_task(task, objects)


def test_final_evaluator_requires_same_request_physical_evidence_and_fresh_contents():
    objects, config = fixture()
    poses = {name: deepcopy(record['pose']) for name, record in objects.items()}
    settle = evaluate_container_settle(config, poses, deepcopy(poses),
        surface_support={'contact_count': 4, 'upward_force_n': 1.})
    assert settle['succeeded']
    request = NS(request_id='request1', task=NS(goal=NS(target_object_id='cover', target_region_id='container')),
        world=NS(objects=objects), constraints=NS(position_tolerance_m=.005, orientation_tolerance_rad=.05))
    report = NS(metadata={'segment_tracking': [{'custom_settle': {'container_settle': settle}}]})
    assert surface_report_result(request, report, NS(objects=objects))['succeeded']
    changed = deepcopy(objects); changed['contents']['pose']['position_m'][0] = .2
    assert not surface_report_result(request, report, NS(objects=changed))['succeeded']
    request.request_id = 'different'
    assert not surface_report_result(request, report, NS(objects=objects))['succeeded']


@pytest.mark.parametrize('failure', [None, 'attached', 'escaped', 'missing_evidence'])
def test_task_evaluator_routes_on_relation_to_cover_and_contents(failure):
    from tuj.m5_motion.contact_evaluation import TaskAwareGoalEvaluator
    from tuj.m5_motion.execution import GoalEvaluationStatus
    from tuj.m5_motion.tests.test_contact_manipulation import _request, _report
    objects, config = fixture()
    request = _request(); report = _report()
    request.task.action_type = 'place_on'; request.task.contact = None
    request.task.target_ids = ['cover']; request.task.goal.target_object_id = 'cover'
    request.task.goal.target_region_id = 'container'; request.world.objects = objects
    request.constraints.position_tolerance_m = .005
    request.constraints.orientation_tolerance_rad = .05
    config['request_id'] = request.request_id
    poses = {name: deepcopy(record['pose']) for name, record in objects.items()}
    settle = evaluate_container_settle(config, poses, deepcopy(poses),
        surface_support={'contact_count': 4, 'upward_force_n': 1.})
    report.metadata['segment_tracking'] = [{'custom_settle': {'container_settle': settle}}]
    if failure == 'attached': report.final_robot_state.attached_object_id = 'cover'
    if failure == 'escaped': objects['contents']['pose']['position_m'][0] = .2
    if failure == 'missing_evidence': report.metadata['segment_tracking'] = []
    outcome = TaskAwareGoalEvaluator().evaluate(request, report, request.world)
    assert outcome.status == (GoalEvaluationStatus.SATISFIED if failure is None else GoalEvaluationStatus.FAILED)
