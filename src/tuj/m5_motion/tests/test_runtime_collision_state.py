import mujoco
import numpy as np
import pytest

from tuj.m5_motion.mujoco_collision import MuJoCoCollisionValidator, MuJoCoCollisionModelRegistry
from tuj.m5_motion.runtime_collision_state import copy_runtime_configuration
from tuj.m5_motion.schema import AttachedObjectTransform, CollisionContext
from tuj.m5_motion.tests.test_mujoco_collision import _ATTACHMENT_SCENE


def scene():
    model = mujoco.MjModel.from_xml_string(_ATTACHMENT_SCENE)
    data = mujoco.MjData(model)
    context = CollisionContext(
        context_id="live-test", collision_model_version="test",
        attached_object_ids=["part"],
        attached_object_transforms=[AttachedObjectTransform(
            object_id="part", free_joint_name="part_free", reference_name="robot_root",
            position_in_reference_m=(0.2, 0, 0), orientation_in_reference_xyzw=(0, 0, 0, 1),
        )],
    )
    validator = MuJoCoCollisionValidator(
        mujoco.MjModel.from_xml_string(_ATTACHMENT_SCENE), joint_names=("arm_slide",),
        robot_root_body_name="robot_root", collision_margin_m=.02,
        collision_model_version="test", entity_geoms={"part": ("part",)},
    )
    registry = MuJoCoCollisionModelRegistry(
        {"test": validator}, collision_contexts={context.context_id: context},
        default_model_version="test",
    )
    return model, data, context, registry


@pytest.mark.parametrize("part_x,finger_y", [(.58, 0), (.2, .23)])
def test_detects_actual_grasp_slip_and_finger_motion_without_changing_live_state(part_x, finger_y):
    model, data, context, registry = scene()
    data.qpos[model.jnt_qposadr[model.joint("part_free").id]] = part_x
    data.qpos[model.jnt_qposadr[model.joint("finger_slide").id]] = finger_y
    mujoco.mj_forward(model, data)
    before, policy = data.qpos.copy(), context.model_dump()
    assert registry.check((0,), context=context).valid
    actual = registry.check((0,), context=context, runtime_state=(model, data))
    assert not actual.valid
    assert actual.min_clearance_m < 0
    np.testing.assert_array_equal(data.qpos, before)
    assert context.model_dump() == policy


def test_missing_runtime_joint_fails_closed():
    _, _, context, registry = scene()
    model = mujoco.MjModel.from_xml_string(_ATTACHMENT_SCENE.replace('name="part_free"', 'name="wrong"'))
    result = registry.check((0,), context=context, runtime_state=(model, mujoco.MjData(model)))
    assert not result.valid
    assert result.failure_code == "RUNTIME_COLLISION_STATE_UNAVAILABLE"


def test_controller_probe_uses_runtime_state_and_preserves_bounded_allowance():
    from types import SimpleNamespace as NS
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalControllerTrajectoryPlayer

    model, data, context, registry = scene()
    context.allowed_collision_pairs = [("part", "carry_obstacle")]
    context.metadata["bounded_collision_allowances"] = [
        {"selectors": ["part", "carry_obstacle"], "minimum_distance_m": -.001}
    ]
    env = NS(sim=NS(model=NS(_model=model), data=NS(_data=data)))
    player = ToolUseJournalControllerTrajectoryPlayer(NS(env=env), collision_probe=registry)
    address = model.jnt_qposadr[model.joint("part_free").id]
    data.qpos[address] = .5505
    assert player._check_collision((0,), context=context).valid
    data.qpos[address] = .552
    result = player._check_collision((0,), context=context)
    assert result.failure_code == "BOUNDED_COLLISION_VIOLATION"


def test_mocap_obstacle_position_is_copied_without_mutating_source():
    xml = '<mujoco><worldbody><body name="obstacle" mocap="true"><geom size=".1"/></body></worldbody></mujoco>'
    source, target = mujoco.MjModel.from_xml_string(xml), mujoco.MjModel.from_xml_string(xml)
    live, isolated = mujoco.MjData(source), mujoco.MjData(target)
    live.mocap_pos[:] = (1, 2, 3)
    copy_runtime_configuration(source, live, target, isolated)
    np.testing.assert_array_equal(isolated.mocap_pos, live.mocap_pos)
    isolated.mocap_pos[:] = 0
    np.testing.assert_array_equal(live.mocap_pos, [[1, 2, 3]])


def test_reordered_joint_ids_are_mapped_by_name_and_invalid_values_rejected():
    source = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="a"/><geom size=".1"/></body><body><joint name="b"/><geom size=".1"/></body></worldbody></mujoco>')
    target = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="b"/><geom size=".1"/></body><body><joint name="a"/><geom size=".1"/></body></worldbody></mujoco>')
    live, isolated = mujoco.MjData(source), mujoco.MjData(target)
    live.qpos[:] = (.2, .4)
    copy_runtime_configuration(source, live, target, isolated)
    np.testing.assert_array_equal(isolated.qpos, [.4, .2])
    live.qpos[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        copy_runtime_configuration(source, live, target, isolated)
    np.testing.assert_array_equal(isolated.qpos, [.4, .2])
