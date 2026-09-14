"""Vacuum attachment locks the intended target and suppresses native adhesion."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tuj.m5_motion.tool_use_journal_runtime import (
    AttachmentMode,
    AttachedObjectState,
    ToolUseJournalControllerTrajectoryPlayer,
    ToolUseJournalRuntimeError,
)


def _attachment(object_id: str = "plate") -> AttachedObjectState:
    return AttachedObjectState(
        object_id=object_id,
        free_joint_name=f"{object_id}_joint",
        reference_kind="site",
        reference_name="grip_site",
        position_in_reference_m=(0.0, 0.0, 0.05),
        rotation_in_reference=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        attach_distance_m=0.001,
        mode=AttachmentMode.KINEMATIC,
    )


def test_vac_gripper_action_suppresses_adhesion_while_attached():
    runtime = SimpleNamespace(
        _active_ee="vac",
        _attachment=_attachment("plate"),
    )
    # Bind methods from the real class without constructing a full env.
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime

    runtime.native_adhesion_suppressed = (
        ToolUseJournalEERuntime.native_adhesion_suppressed.__get__(runtime)
    )
    runtime.vac_gripper_action_for_command = (
        ToolUseJournalEERuntime.vac_gripper_action_for_command.__get__(runtime)
    )
    runtime.attached_object_id = "plate"

    assert runtime.native_adhesion_suppressed is True
    # Logical suction-on command must not drive native adhesion after lock.
    assert runtime.vac_gripper_action_for_command(1.0) == -1.0
    assert runtime.vac_gripper_action_for_command(0.5) == -1.0


def test_vac_gripper_action_passes_through_before_attach():
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime

    runtime = SimpleNamespace(_active_ee="vac", _attachment=None)
    runtime.native_adhesion_suppressed = (
        ToolUseJournalEERuntime.native_adhesion_suppressed.__get__(runtime)
    )
    runtime.vac_gripper_action_for_command = (
        ToolUseJournalEERuntime.vac_gripper_action_for_command.__get__(runtime)
    )
    assert runtime.native_adhesion_suppressed is False
    assert runtime.vac_gripper_action_for_command(0.8) == pytest.approx(0.8)


def test_controller_action_forces_adhesion_off_when_plate_attached():
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime

    gripper = SimpleNamespace(action_is_absolute=True, current_action=np.zeros(1))
    robot = SimpleNamespace(
        part_controllers={
            "right": SimpleNamespace(name="JOINT_POSITION", input_type="absolute")
        },
        composite_controller=SimpleNamespace(
            _action_split_indexes={"right": (0, 1), "right_gripper": (1, 2)}
        ),
        robot_model=SimpleNamespace(joints=["arm"]),
        action_dim=2,
        gripper={"right": gripper},
    )
    runtime = SimpleNamespace(
        env=SimpleNamespace(robots=[robot]),
        gripper_command=1.0,
        _captured_gripper_action=None,
        _active_ee="vac",
        _attachment=_attachment("plate"),
        suppress_native_adhesion_actuators=lambda: None,
        log_vac_non_target_cup_contacts=lambda: None,
    )
    runtime.native_adhesion_suppressed = (
        ToolUseJournalEERuntime.native_adhesion_suppressed.__get__(runtime)
    )
    runtime.vac_gripper_action_for_command = (
        ToolUseJournalEERuntime.vac_gripper_action_for_command.__get__(runtime)
    )
    player = ToolUseJournalControllerTrajectoryPlayer(runtime)
    action = player._controller_action(
        SimpleNamespace(joint_names=["arm"]), [0.0]
    )
    assert action[1] == pytest.approx(-1.0)


def test_second_attach_is_rejected_while_locked(monkeypatch):
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime

    calls: list[dict] = []

    runtime = SimpleNamespace(
        _active_ee="vac",
        _attachment=_attachment("plate"),
        _held_tool_id="plate",
        _grasp_engaged=True,
        env=object(),
    )

    def log(**kwargs):
        calls.append(kwargs)

    runtime.log_vac_attach_diagnostic = log
    with pytest.raises(ToolUseJournalRuntimeError, match="already attached"):
        ToolUseJournalEERuntime.attach_object(runtime, "block_3")
    assert calls and calls[0]["action"] == "REJECT"
    assert "attachment_locked" in calls[0]["reason"]


def test_finger_ee_unaffected_by_vac_adhesion_gate():
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime

    runtime = SimpleNamespace(_active_ee="2F", _attachment=None)
    runtime.native_adhesion_suppressed = (
        ToolUseJournalEERuntime.native_adhesion_suppressed.__get__(runtime)
    )
    runtime.vac_gripper_action_for_command = (
        ToolUseJournalEERuntime.vac_gripper_action_for_command.__get__(runtime)
    )
    assert runtime.native_adhesion_suppressed is False
    assert runtime.vac_gripper_action_for_command(1.0) == pytest.approx(1.0)


def test_catalog_step_maps_opening_through_adhesion_gate():
    """Catalog vac step uses -opening as action; lock must still force -1."""

    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime

    runtime = SimpleNamespace(
        _active_ee="vac",
        _attachment=_attachment("plate_b"),
    )
    runtime.native_adhesion_suppressed = (
        ToolUseJournalEERuntime.native_adhesion_suppressed.__get__(runtime)
    )
    runtime.vac_gripper_action_for_command = (
        ToolUseJournalEERuntime.vac_gripper_action_for_command.__get__(runtime)
    )
    # hold_opening = -1 → raw action +1; after lock → -1 (adhesion off).
    opening = -1.0
    raw = -float(opening)
    assert raw == pytest.approx(1.0)
    assert runtime.vac_gripper_action_for_command(raw) == pytest.approx(-1.0)
