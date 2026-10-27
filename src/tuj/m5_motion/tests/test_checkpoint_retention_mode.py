"""Restoration must not add a different grasp model to a valid saved hold."""
from dataclasses import replace

import numpy as np
import pytest

from tuj.m5_motion.runtime_checkpoint import save_runtime_checkpoint, restore_runtime_checkpoint
from tuj.m5_motion.tests.test_elastic_checkpoint import elastic_runtime
from tuj.m5_motion.tests.test_tool_use_journal import _fake_env
from tuj.m5_motion.tool_use_journal_runtime import (
    AttachmentMode, ToolUseJournalEERuntime,
)


@pytest.mark.parametrize('mode', [AttachmentMode.BREAKABLE_WELD, AttachmentMode.KINEMATIC])
@pytest.mark.parametrize('held_tool', [False, True])
def test_attachment_checkpoint_restores_only_its_original_retention_mode(tmp_path, mode, held_tool):
    source, _ = elastic_runtime()
    restored = ToolUseJournalEERuntime(_fake_env('2F'), _fake_env)
    try:
        if mode is AttachmentMode.KINEMATIC:
            source.detach_object()
            source.attach_object('apple', max_attach_distance_m=.05,
                max_attach_penetration_m=.05, attachment_mode=mode)
        if held_tool:
            source.mark_attached_object_as_tool('apple')
        original_attachment = source.attachment
        source_state = source.env.sim.data._data.qpos.copy()
        path = save_runtime_checkpoint(tmp_path/'attachment.json', source)
        restore_runtime_checkpoint(path, restored)

        assert restored.attachment == original_attachment
        np.testing.assert_allclose(restored.env.sim.data._data.qpos, source_state, atol=1e-12, rtol=0)
        assert restored.held_tool_id == ('apple' if held_tool else None)
        assert restored._contact_friction_retention is None
        assert (restored._breakable_runtime is not None) is (mode is AttachmentMode.BREAKABLE_WELD)
        if mode is AttachmentMode.BREAKABLE_WELD:
            assert restored._breakable_runtime.step_count == source._breakable_runtime.step_count
            assert restored._breakable_runtime.violation_steps == source._breakable_runtime.violation_steps
    finally:
        source.close()
        restored.close()


def test_unattached_contact_hold_still_restores_contact_friction_monitor(tmp_path, monkeypatch):
    source = ToolUseJournalEERuntime(_fake_env('2F'), _fake_env)
    restored = ToolUseJournalEERuntime(_fake_env('2F'), _fake_env)
    try:
        source.command_gripper(engaged=True, suction=False)
        source.mark_contact_friction_object_as_tool('apple')
        path = save_runtime_checkpoint(tmp_path/'contact.json', source)
        # This test isolates monitor selection; the ordinary missing-bilateral
        # contact rejection is independently covered by the checkpoint tests.
        contact = replace(restored.object_contact_metrics('apple'),
            contact_count=2, contact_groups=('left_finger', 'right_finger'))
        monkeypatch.setattr(restored, 'object_contact_metrics', lambda _: contact)
        restore_runtime_checkpoint(path, restored)
        assert restored.attachment is None
        assert restored._breakable_runtime is None
        assert restored.held_tool_id == 'apple'
        assert restored._contact_friction_retention.object_id == 'apple'
    finally:
        source.close()
        restored.close()


def test_empty_hold_does_not_create_any_retention_monitor(tmp_path):
    runtime = ToolUseJournalEERuntime(_fake_env('2F'), _fake_env)
    try:
        path = save_runtime_checkpoint(tmp_path/'empty.json', runtime)
        restore_runtime_checkpoint(path, runtime)
        assert runtime.attachment is None
        assert runtime.held_tool_id is None
        assert runtime._breakable_runtime is None
        assert runtime._contact_friction_retention is None
    finally:
        runtime.close()
