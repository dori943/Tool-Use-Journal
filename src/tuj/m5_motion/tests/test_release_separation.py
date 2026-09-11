import mujoco
import numpy as np
import pytest

from tuj.m5_motion.release_separation import release_geometry_pairs
from tuj.m5_motion.schema import AttachedObjectTransform, GoalType, KeyframeEventType, KeyframeType, MotionGoal, Pose
from tuj.m5_motion.tests.test_tool_use_journal_planning import _Compiler, _artifact, _factory, _keyframe, _request, _world
from tuj.m5_motion.tool_use_journal_planning import ToolUseJournalCollisionBindingError


def transform(offset=.022):
    return AttachedObjectTransform(object_id='bottle', free_joint_name='bottle_free',
        reference_kind='site', reference_name='grip', position_in_reference_m=(offset, 0., 0.),
        orientation_in_reference_xyzw=(0., 0., 0., 1.))


@pytest.mark.parametrize('offset,expected', [(.022, True), (.019, False), (.04, False)])
def test_measures_only_nonpenetrating_close_pairs(offset, expected):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="robot"><joint name="j1" axis="0 0 1"/><joint name="j2" axis="0 1 0"/>
        <body name="ee"><site name="grip"/><geom name="pad" type="box" size=".01 .01 .01"/>
          <geom name="shaft" type="box" pos="-.2 0 0" size=".01 .01 .01"/></body></body>
      <body name="bottle"><freejoint name="bottle_free"/><geom name="object" type="box" size=".01 .01 .01"/></body>
    </worldbody></mujoco>''')
    qpos = model.qpos0.copy()
    world = _world(attached=transform(offset))
    pairs = release_geometry_pairs(model, qpos, {'bottle': 'bottle'}, 'ee', world, 'bottle', .005)
    assert bool(pairs) == expected
    if expected:
        assert pairs[0]['selectors'] == ['pad', 'bottle']
        assert pairs[0]['measured_target_geoms'] == ['object']
        assert pairs[0]['minimum_distance_m'] == 0
        assert pairs[0]['initial_distance_m'] == pytest.approx(.002)
    assert np.array_equal(qpos, model.qpos0)
    world.robot_state.attached_object_id = None
    assert release_geometry_pairs(model, qpos, {'bottle': 'bottle'}, 'ee', world, 'bottle', .005) == []


@pytest.mark.parametrize('retreat_type', [KeyframeType.RETREAT, KeyframeType.TRANSFER])
def test_bounded_policy_ends_at_first_retreat(retreat_type):
    compiler = _Compiler()
    compiler.initial_release_geometry_pairs = lambda *a: [
        {'selectors': ['pad', 'bottle_geom'], 'minimum_distance_m': 0., 'initial_distance_m': .002}]
    req = _request(MotionGoal(goal_type=GoalType.POSE, target_object_id='bottle',
        target_pose=Pose(frame_id='world', position_m=(0., 0., 0.), orientation_xyzw=(0., 0., 0., 1.))),
        action_type='PLACE', attached=transform())
    art = _artifact((_keyframe('place', KeyframeType.PLACE, events=(KeyframeEventType.DETACH_OBJECT,)),
        _keyframe('retreat', retreat_type), _keyframe('clear', KeyframeType.TRANSFER)))
    if retreat_type != KeyframeType.RETREAT:
        with pytest.raises(ToolUseJournalCollisionBindingError, match='immediate RETREAT'):
            _factory(compiler).prepare(req, art)
        return
    setup = _factory(compiler).prepare(req, art)
    place, retreat, clear = setup.keyframe_artifact.candidates[0].keyframes
    sep = setup.collision_contexts[retreat.collision_context_id]
    assert sep.allowed_collision_pairs == [('pad', 'bottle_geom')]
    assert sep.metadata['bounded_collision_allowances'][0]['minimum_distance_m'] == 0
    assert retreat.metadata['validate_endpoint_after_context'] is True
    assert retreat.collision_context_after_events_id == clear.collision_context_id
    assert not setup.collision_contexts[clear.collision_context_id].allowed_collision_pairs


def test_release_pair_still_rejects_penetration_and_restores_margin():
    from tuj.m5_motion.mujoco_collision import MuJoCoCollisionValidator
    from tuj.m5_motion.schema import CollisionContext
    from tuj.m5_motion.tests.test_mujoco_collision import _SCENE

    model = mujoco.MjModel.from_xml_string(_SCENE)
    validator = MuJoCoCollisionValidator(model, joint_names=('slide',),
        robot_root_body_name='robot_root', collision_margin_m=.005,
        entity_geoms={'target': ('wall_col',)}, collision_model_version='release-test')
    normal = CollisionContext(context_id='normal', collision_model_version='release-test')
    release = normal.model_copy(update={'context_id': 'release',
        'allowed_collision_pairs': [('robot_col', 'target')],
        'metadata': {'bounded_collision_allowances': [
            {'selectors': ['robot_col', 'target'], 'minimum_distance_m': 0.}]}})
    assert validator.check((.296,), context=release).valid
    assert not validator.check((.296,), context=normal).valid
    assert not validator.check((.301,), context=release).valid
    assert validator.check((.29,), context=normal).valid
