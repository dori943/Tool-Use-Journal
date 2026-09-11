import json

import mujoco
import numpy as np
import pytest

from tuj.m5_motion.runtime_checkpoint import RuntimeCheckpointError, save_runtime_checkpoint, restore_runtime_checkpoint
from tuj.m5_motion.tests.test_tool_use_journal import _fake_env
from tuj.m5_motion.tool_use_journal_runtime import AttachmentMode, BreakableWeldConfig, ToolUseJournalEERuntime, ToolUseJournalAttachmentBroken


def elastic_runtime():
    runtime = ToolUseJournalEERuntime(_fake_env('2F'), _fake_env)
    m, d = runtime.env.sim.model._model, runtime.env.sim.data._data
    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot0_right_hand')
    body, joint, _ = runtime._object_free_joint(runtime.env, 'apple')
    adr = int(m.jnt_qposadr[joint])
    d.qpos[adr:adr+3] = d.xpos[hand]
    mujoco.mj_forward(m, d)
    runtime.command_gripper(engaged=True, suction=False)
    runtime.attach_object('apple', max_attach_distance_m=.05,
        max_attach_penetration_m=.05, attachment_mode=AttachmentMode.BREAKABLE_WELD,
        breakable_weld=BreakableWeldConfig(require_contact=False, max_orientation_error_rad=.6))
    d.qpos[adr+3:adr+7] = [np.cos(.04), np.sin(.04), 0., 0.]
    mujoco.mj_forward(m, d)
    runtime._breakable_runtime.step_count = 42
    runtime._breakable_runtime.violation_steps = 2
    runtime._breakable_runtime.contact_loss_steps = 1
    return runtime, adr


def test_elastic_pose_and_break_counters_roundtrip_without_changing_rest_pose(tmp_path):
    runtime, adr = elastic_runtime()
    try:
        d = runtime.env.sim.data._data
        state = d.qpos.copy()
        rest = runtime.attachment
        path = save_runtime_checkpoint(tmp_path/'elastic.json', runtime)
        runtime.detach_object()
        d.qpos[adr] += .02
        restore_runtime_checkpoint(path, runtime)
        np.testing.assert_array_equal(d.qpos, state)
        assert runtime.attachment == rest
        assert runtime._breakable_runtime.step_count == 42
        assert runtime._breakable_runtime.violation_steps == 2
        assert runtime._breakable_runtime.contact_loss_steps == 1
    finally:
        runtime.close()


def test_restored_pending_break_is_not_given_a_new_grace_period(tmp_path):
    runtime, adr = elastic_runtime()
    try:
        m, d = runtime.env.sim.model._model, runtime.env.sim.data._data
        d.qpos[adr] += runtime.attachment.breakable_weld.max_position_error_m + .01
        mujoco.mj_forward(m, d)
        path = save_runtime_checkpoint(tmp_path/'pending.json', runtime)
        runtime.detach_object()
        restore_runtime_checkpoint(path, runtime)
        with pytest.raises(ToolUseJournalAttachmentBroken, match='POSITION_ERROR_LIMIT'):
            runtime.prepare_attachment_step()
    finally:
        runtime.close()


def test_legacy_checkpoint_keeps_rigid_pose_gate(tmp_path):
    runtime, _ = elastic_runtime()
    try:
        path = save_runtime_checkpoint(tmp_path/'legacy.json', runtime)
        runtime.detach_object()
        raw = json.loads(path.read_text(encoding='utf-8'))
        raw['logical_state'].pop('attachment_observation')
        raw['logical_state'].pop('breakable_runtime')
        path.write_text(json.dumps(raw), encoding='utf-8')
        with pytest.raises(RuntimeCheckpointError, match='pose mismatch'):
            restore_runtime_checkpoint(path, runtime)
    finally:
        runtime.close()


def test_invalid_observation_rotation_rejected(tmp_path):
    runtime, _ = elastic_runtime()
    try:
        path = save_runtime_checkpoint(tmp_path/'bad_rotation.json', runtime)
        runtime.detach_object()
        raw = json.loads(path.read_text(encoding='utf-8'))
        raw['logical_state']['attachment_observation']['rotation_in_reference'][0][0] = 2.
        path.write_text(json.dumps(raw), encoding='utf-8')
        with pytest.raises(RuntimeCheckpointError, match='observed attachment pose'):
            restore_runtime_checkpoint(path, runtime)
    finally:
        runtime.close()


def test_corrupt_observed_pose_rejects_and_rolls_back(tmp_path):
    runtime, adr = elastic_runtime()
    try:
        path = save_runtime_checkpoint(tmp_path/'corrupt.json', runtime)
        runtime.detach_object()
        raw = json.loads(path.read_text(encoding='utf-8'))
        raw['logical_state']['attachment_observation']['position_in_reference_m'][0] += .02
        path.write_text(json.dumps(raw), encoding='utf-8')
        d = runtime.env.sim.data._data
        d.qpos[adr] += .01
        before = d.qpos.copy()
        with pytest.raises(RuntimeCheckpointError):
            restore_runtime_checkpoint(path, runtime)
        np.testing.assert_array_equal(d.qpos, before)
    finally:
        runtime.close()


@pytest.mark.parametrize('bad', [-1, True, 1.5])
def test_invalid_counter_rejected_before_restore(tmp_path, bad):
    runtime, _ = elastic_runtime()
    try:
        path = save_runtime_checkpoint(tmp_path/'counter.json', runtime)
        runtime.detach_object()
        raw = json.loads(path.read_text(encoding='utf-8'))
        raw['logical_state']['breakable_runtime']['violation_steps'] = bad
        path.write_text(json.dumps(raw), encoding='utf-8')
        with pytest.raises(RuntimeCheckpointError):
            restore_runtime_checkpoint(path, runtime)
    finally:
        runtime.close()
