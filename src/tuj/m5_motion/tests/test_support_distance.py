"""Tilted supports must use physical clearance without relaxing penetration limits."""
import copy

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.m5_motion.support_distance import object_pair_clearance, refine_support_request
from tuj.m5_motion.tests.test_tool_use_journal_planning import _Compiler, _factory, _request
from tuj.m5_motion.schema import GoalType, MotionGoal
from tuj.m5_motion.tool_use_journal_planning import ToolUseJournalCollisionBindingError


def pair(penetration):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="lower"><freejoint/><geom type="box" size=".055 .055 .01"/></body>
      <body name="upper"><freejoint/><geom type="box" size=".055 .055 .01"/></body>
    </worldbody></mujoco>''')
    rotation = Rotation.from_euler('xy', [.02, -.01])
    normal = rotation.apply([0, 0, 1])
    objects = {}
    for name, pos in [('other', np.array([1., 2., 3.])),
                      ('bottle', np.array([1., 2., 3.]) + normal * (.02 - penetration))]:
        objects[name] = {'pose': {'frame_id': 'world', 'position_m': pos.tolist(),
                                  'orientation_xyzw': rotation.as_quat().tolist()}}
    return model, {'other': 'lower', 'bottle': 'upper'}, objects


@pytest.mark.parametrize('penetration', [.000017, .0025])
def test_signed_distance_at_snapshot_poses_and_no_baseline_mutation(penetration):
    model, names, objects = pair(penetration)
    baseline = model.qpos0.copy()
    before = copy.deepcopy(objects)
    distance = object_pair_clearance(model, baseline, names, objects, 'bottle', 'other')
    assert distance == pytest.approx(-penetration, abs=1e-8)
    np.testing.assert_array_equal(baseline, model.qpos0)
    assert objects == before


@pytest.mark.parametrize('penetration, rejected', [(.000017, False), (.0025, True)])
@pytest.mark.parametrize('policy', ['EXPLICIT_TASK_METADATA', 'AUTO_INITIAL_SUPPORT_V1'])
def test_support_policy_keeps_one_mm_limit_with_native_distance(penetration, rejected, policy):
    model, names, objects = pair(penetration)
    class Compiler(_Compiler):
        def initial_object_clearance(self, records, first, second):
            return object_pair_clearance(model, model.qpos0, names, records, first, second)
    request = _request(MotionGoal(goal_type=GoalType.POSE, target_object_id='bottle'), action_type='PICK')
    request.world.objects.update(objects)
    request.task.metadata.update(support_collision_selectors=['other'], support_initial_clearance_m=-.00335,
                                 support_collision_policy=policy, support_horizontal_overlap_ratio=1.)
    refined = refine_support_request(request, Compiler())
    assert request.task.metadata['support_initial_clearance_m'] == -.00335
    assert refined.task.metadata['support_bound_clearance_m'] == -.00335
    if rejected:
        with pytest.raises(ToolUseJournalCollisionBindingError, match='1 mm hard limit'):
            _factory(Compiler())._pick_support_policy(refined, 'bottle')
    else:
        selectors, evidence, maximum = _factory(Compiler())._pick_support_policy(refined, 'bottle')
        assert selectors == ['other']
        assert maximum == .001
        assert evidence['support_initial_clearance_m'] == pytest.approx(-penetration, abs=1e-8)


def test_missing_binding_retains_conservative_rejection():
    class Compiler(_Compiler):
        def initial_object_clearance(self, *args):
            return None
    request = _request(MotionGoal(goal_type=GoalType.POSE, target_object_id='bottle'), action_type='PICK')
    request.task.metadata.update(support_collision_selectors=['other'], support_initial_clearance_m=-.00335)
    assert refine_support_request(request, Compiler()) is request
    with pytest.raises(ToolUseJournalCollisionBindingError, match='1 mm hard limit'):
        _factory(Compiler())._pick_support_policy(request, 'bottle')
