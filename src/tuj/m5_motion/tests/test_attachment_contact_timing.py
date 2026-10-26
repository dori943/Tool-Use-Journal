"""Contact forces must stay paired with the contact list from their solve."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from tuj.m5_motion.tests.test_tool_use_journal import _fake_env, MOUNTED_ROOTS
from tuj.m5_motion.tool_use_journal_runtime import (
    AttachmentMode, BreakableWeldConfig, ToolUseJournalAttachmentBroken,
    ToolUseJournalControllerTrajectoryPlayer, ToolUseJournalEERuntime,
)


def contact_runtime(*, lite_physics=True, debounce=1, startup=0, loss_grace=5):
    runtime = ToolUseJournalEERuntime(_fake_env('2F'), _fake_env)
    # The plane belongs to the mounted EE so the real attachment contact
    # collector reads this small scene without replacing mj_contactForce.
    xml = f'''<mujoco><option timestep=".002"/><worldbody>
      <body name="robot0_right_hand"><body name="{MOUNTED_ROOTS['2F']}">
        <geom name="gripper0_right_left_fingerpad_collision" type="plane" size="2 2 .1"/>
      </body></body>
      <body name="heavy" pos="-.5 0 .099"><freejoint name="heavy_joint"/>
        <geom name="heavy" type="sphere" size=".1" mass="100"/></body>
      <body name="held" pos="0 0 .099"><freejoint name="held_joint"/>
        <geom name="held" type="sphere" size=".1" mass=".01"/></body>
      <body name="other" pos=".5 0 1"><freejoint name="other_joint"/>
        <geom name="other" type="sphere" size=".1" mass=".01"/></body>
    </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = runtime.env
    env.sim = SimpleNamespace(model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data),
        step1=lambda: mujoco.mj_step1(model, data), step2=lambda: mujoco.mj_step2(model, data),
        forward=lambda: mujoco.mj_forward(model, data), step=lambda: mujoco.mj_step(model, data))
    env.model = SimpleNamespace(get_xml=lambda: xml)
    env.obj_body_id = {'held': model.body('held').id}
    env.control_timestep = env.model_timestep = float(model.opt.timestep)
    env.lite_physics = lite_physics
    env._pre_action = lambda action, policy_step: None
    runtime.command_gripper(engaged=True, suction=False)
    runtime.attach_object('held', attachment_mode=AttachmentMode.BREAKABLE_WELD,
        breakable_weld=BreakableWeldConfig(max_contact_force_n=10., startup_grace_steps=startup,
            break_debounce_steps=debounce, contact_loss_grace_steps=loss_grace))
    player = object.__new__(ToolUseJournalControllerTrajectoryPlayer)
    player.runtime = runtime
    return runtime, player, model, data


def replace_other_contact(model, data):
    # Keep two contacts, but replace heavy-floor with other-floor. The held
    # contact's constraint address changes from 4 to 0 without moving held.
    data.qpos[model.jnt_qposadr[model.joint('heavy_joint').id] + 2] = 1.
    data.qpos[model.jnt_qposadr[model.joint('other_joint').id] + 2] = .099


def test_real_step1_pairs_fresh_contacts_with_old_constraint_forces():
    runtime, _, model, data = contact_runtime()
    try:
        solved = runtime.object_contact_metrics('held')
        replace_other_contact(model, data)
        mujoco.mj_step1(model, data)
        unresolved = runtime.object_contact_metrics('held')
        assert unresolved.total_force_n > 1000. * solved.total_force_n
        mujoco.mj_step2(model, data)
        resolved = runtime.object_contact_metrics('held')
        assert resolved.total_force_n == pytest.approx(solved.total_force_n)
    finally:
        runtime.close()


@pytest.mark.parametrize('lite_physics', [False, True])
def test_controller_does_not_break_from_another_objects_old_force(lite_physics):
    runtime, player, model, data = contact_runtime(lite_physics=lite_physics)
    try:
        config = runtime.attachment.breakable_weld
        masses, friction, timestep = model.body_mass.copy(), model.geom_friction.copy(), model.opt.timestep
        replace_other_contact(model, data)
        player._advance_controller(np.zeros(1))
        assert runtime.attached_object_id == 'held'
        assert runtime.attachment_contact_metrics.total_force_n < config.max_contact_force_n
        assert runtime._breakable_runtime.step_count == 1
        assert runtime._breakable_runtime.violation_steps == 0
        assert runtime.attachment.breakable_weld == config
        np.testing.assert_array_equal(model.body_mass, masses)
        np.testing.assert_array_equal(model.geom_friction, friction)
        assert model.opt.timestep == timestep
    finally:
        runtime.close()


def test_split_prepare_uses_solved_sample_without_changing_state_or_limits():
    runtime, _, model, data = contact_runtime()
    try:
        solved = runtime.attachment_contact_metrics
        attachment = runtime.attachment
        replace_other_contact(model, data)
        mujoco.mj_step1(model, data)
        before_qpos, before_qvel, before_time = data.qpos.copy(), data.qvel.copy(), data.time
        runtime.prepare_attachment_step(contact_forces_resolved=False)
        assert runtime.attachment_contact_metrics == solved
        assert runtime.attachment is attachment
        np.testing.assert_array_equal(data.qpos, before_qpos)
        np.testing.assert_array_equal(data.qvel, before_qvel)
        assert data.time == before_time
    finally:
        runtime.close()


@pytest.mark.parametrize('startup', [0, 2])
def test_solved_overload_still_breaks_with_existing_grace_and_debounce(startup):
    runtime, player, model, data = contact_runtime(debounce=2, startup=startup)
    try:
        data.xfrc_applied[model.body('held').id, 2] = -100.
        first_violation_step = max(2, startup + 1)
        for _ in range(first_violation_step - 1):
            player._advance_controller(np.zeros(1))
            assert runtime.attachment_contact_metrics.total_force_n > 10.
            assert runtime._breakable_runtime.violation_steps == 0
        player._advance_controller(np.zeros(1))
        assert runtime._breakable_runtime.step_count == first_violation_step
        assert runtime._breakable_runtime.violation_steps == 1
        with pytest.raises(ToolUseJournalAttachmentBroken, match='CONTACT_FORCE_LIMIT'):
            player._advance_controller(np.zeros(1))
        assert runtime.last_attachment_break.contact_force_n > 10.
    finally:
        runtime.close()


def test_solved_contact_loss_is_checked_at_the_next_prepare():
    runtime, player, model, data = contact_runtime(loss_grace=0)
    try:
        data.qpos[model.jnt_qposadr[model.joint('held_joint').id] + 2] += .005
        player._advance_controller(np.zeros(1))
        assert runtime.attachment_contact_metrics.contact_count == 0
        with pytest.raises(ToolUseJournalAttachmentBroken, match='CONTACT_LOST'):
            player._advance_controller(np.zeros(1))
    finally:
        runtime.close()


def test_new_attachment_replaces_old_solved_sample_and_counters():
    runtime, player, model, data = contact_runtime()
    try:
        data.xfrc_applied[model.body('held').id, 2] = -100.
        player._advance_controller(np.zeros(1))
        assert runtime.attachment_contact_metrics.total_force_n > 10.
        config = runtime.attachment.breakable_weld
        runtime.detach_object()
        data.xfrc_applied[:] = 0.
        data.qvel[:] = 0.
        data.qpos[model.jnt_qposadr[model.joint('held_joint').id] + 2] = .099
        runtime.attach_object('held', attachment_mode=AttachmentMode.BREAKABLE_WELD,
            breakable_weld=config)
        assert runtime.attachment_contact_metrics.total_force_n < 10.
        assert runtime.attachment_contact_metrics == runtime.object_contact_metrics('held')
        assert runtime._breakable_runtime.step_count == 0
        assert runtime._breakable_runtime.violation_steps == 0
    finally:
        runtime.close()


@pytest.mark.parametrize('via_controller', [False, True])
def test_direct_and_full_forward_still_check_current_solved_overload(via_controller):
    runtime, player, model, data = contact_runtime(lite_physics=False)
    try:
        data.xfrc_applied[model.body('held').id, 2] = -100.
        mujoco.mj_forward(model, data)
        with pytest.raises(ToolUseJournalAttachmentBroken, match='CONTACT_FORCE_LIMIT'):
            if via_controller:
                player._advance_controller(np.zeros(1))
            else:
                runtime.prepare_attachment_step()
    finally:
        runtime.close()


@pytest.mark.parametrize('failure', ['POSITION_ERROR_LIMIT', 'ORIENTATION_ERROR_LIMIT', 'WELD_FORCE_LIMIT'])
def test_split_prepare_still_checks_current_pose_and_weld_demand(failure):
    runtime, _, model, data = contact_runtime()
    try:
        joint = model.joint('held_joint').id
        adr = model.jnt_qposadr[joint]
        if failure == 'POSITION_ERROR_LIMIT':
            data.qpos[adr] += .06
        elif failure == 'ORIENTATION_ERROR_LIMIT':
            data.qpos[adr + 3:adr + 7] = [np.cos(.5), np.sin(.5), 0., 0.]
        else:
            data.qvel[model.jnt_dofadr[joint]] = 100.
        mujoco.mj_step1(model, data)
        with pytest.raises(ToolUseJournalAttachmentBroken, match=failure):
            runtime.prepare_attachment_step(contact_forces_resolved=False)
    finally:
        runtime.close()
