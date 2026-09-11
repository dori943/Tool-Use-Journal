"""Support fragments must be measured contacts, never broad collision bypasses."""
import copy
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from tuj.m5_motion.static_support import coplanar_static_support_contacts
from tuj.m5_motion.support_distance import refine_support_request
from tuj.m5_motion.schema import GoalType, MotionGoal
from tuj.m5_motion.tests.test_tool_use_journal_planning import _Compiler, _factory, _request
from tuj.m5_motion.tool_use_journal_planning import ToolUseJournalCollisionBindingError


def scene(penetration=.00001, *, tilt_right=False):
    xml = '''<mujoco><worldbody>
      <geom name="left" type="box" pos="-.06 0 .005" size=".06 .08 .005"/>
      <geom name="right" type="box" pos=".06 0 .005" size=".06 .08 .005"/>
      <geom name="far" type="box" pos=".5 0 .005" size=".06 .08 .005"/>
      <geom name="wall" type="box" pos=".08 0 .1" size=".005 .08 .1"/>
      <body name="moving" pos="0 .06 0"><freejoint/>
        <geom name="moving_surface" type="box" pos="0 0 .005" size=".08 .02 .005"/>
      </body>
      <body name="payload"><freejoint/>
        <geom name="payload_geom" type="box" size=".08 .05 .005" mass=".1"/>
      </body>
    </worldbody></mujoco>'''
    if tilt_right:
        xml = xml.replace('name="right"', 'name="right" quat="0.9987502604 0.0499791693 0 0"')
    model = mujoco.MjModel.from_xml_string(xml)
    world = SimpleNamespace(objects={'bottle': {'pose': {
        'frame_id': 'world', 'position_m': [0, 0, .015-penetration],
        'orientation_xyzw': [0, 0, 0, 1]}}}, obstacles=[
            {'obstacle_id': name, 'collision_enabled_in_source': True}
            for name in ['left', 'right', 'far', 'wall', 'moving_surface']])
    return model, world


def measure(model, world, primary=('left',)):
    return coplanar_static_support_contacts(
        model, model.qpos0.copy(), {'bottle': 'payload'}, world, 'bottle', list(primary))


def test_multiple_support_fragments_exclude_wall_far_and_movable_geoms():
    model, world = scene()
    before_objects = copy.deepcopy(world.objects)
    before_qpos = model.qpos0.copy()
    result = measure(model, world)
    assert set(result) == {'left', 'right'}
    assert all(v == pytest.approx(-.00001, abs=1e-8) for v in result.values())
    assert world.objects == before_objects
    np.testing.assert_array_equal(model.qpos0, before_qpos)


@pytest.mark.parametrize('primary', [('missing',), ('moving_surface',), ('wall',), ('left', 'far')])
def test_unbound_noncontact_or_nonhorizontal_primary_cannot_expand(primary):
    model, world = scene()
    assert measure(model, world, primary) == {}


def test_proximity_and_positive_margin_contact_are_not_support():
    model, world = scene(penetration=-.0001)
    for gid in range(model.ngeom):
        model.geom_margin[gid] = .001
    assert measure(model, world) == {}


def test_disabled_and_tilted_extra_surfaces_cannot_expand():
    model, world = scene()
    world.obstacles[1]['collision_enabled_in_source'] = False
    assert set(measure(model, world)) == {'left'}
    model, world = scene(tilt_right=True)
    assert set(measure(model, world)) == {'left'}


def test_no_gravity_does_not_invent_upward_support():
    model, world = scene()
    model.opt.gravity[:] = 0
    assert measure(model, world) == {}


@pytest.mark.parametrize('penetration,rejected', [(.00001, False), (.0015, True), (.0025, True)])
def test_refinement_retains_original_overlap_and_penetration_guards(penetration, rejected):
    model, world = scene(penetration)
    class Compiler(_Compiler):
        def initial_static_support_contacts(self, snapshot, target, primary):
            return coplanar_static_support_contacts(
                model, model.qpos0, {'bottle': 'payload'}, snapshot, target, primary)
        def initial_object_clearance(self, *args):
            return None
    request = _request(MotionGoal(goal_type=GoalType.POSE, target_object_id='bottle'), action_type='PICK')
    request.world.objects.update(world.objects)
    request.world.obstacles = world.obstacles
    request.task.metadata.update(support_collision_selectors=['left'],
        support_collision_policy='AUTO_INITIAL_SUPPORT_V1',
        support_collision_detection_source='world.obstacles.aabb',
        support_initial_clearance_m=-penetration,
        support_horizontal_overlap_ratio=.5)
    refined = refine_support_request(request, Compiler())
    assert request.task.metadata['support_collision_selectors'] == ['left']
    assert refined.task.metadata['support_collision_selectors'] == ['left', 'right']
    if rejected:
        with pytest.raises(ToolUseJournalCollisionBindingError, match='1 mm hard limit'):
            _factory(Compiler())._pick_support_policy(refined, 'bottle')
    else:
        selectors, evidence, bound = _factory(Compiler())._pick_support_policy(refined, 'bottle')
        assert selectors == ['left', 'right']
        assert bound == .001
        assert evidence['support_horizontal_overlap_ratio'] == .5
        refined.task.metadata['support_horizontal_overlap_ratio'] = .1
        with pytest.raises(ToolUseJournalCollisionBindingError, match='horizontal overlap'):
            _factory(Compiler())._pick_support_policy(refined, 'bottle')


def test_explicit_support_metadata_does_not_expand():
    class Compiler(_Compiler):
        def initial_static_support_contacts(self, *args):
            raise AssertionError('explicit supports must not be augmented')
        def initial_object_clearance(self, *args):
            return None
    request = _request(MotionGoal(goal_type=GoalType.POSE, target_object_id='bottle'), action_type='PICK')
    request.task.metadata.update(support_collision_selectors=['left'],
                                 support_collision_policy='EXPLICIT_TASK_METADATA')
    assert refine_support_request(request, Compiler()) is request
