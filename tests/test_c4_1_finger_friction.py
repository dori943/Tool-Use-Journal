"""The declared 3F pad material must apply to tips/pads, not links or the rack."""
from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime


def test_3f_friction_targets_actual_jaco_contact_names_only():
    selected = [f"gripper0_right_{finger}_{surface}_collision"
                for finger in ("thumb", "index", "pinky") for surface in ("tip", "pad")]
    excluded = ["gripper0_right_thumb_proximal_collision", "gripper0_right_thumb_distal_collision",
                "gripper0_right_thumb_pad_visual", "rack_3F_thumb_tip_collision",
                "gripper0_right_left_fingertip_collision", "table_collision"]
    xml = "<mujoco><worldbody>" + "".join(
        f'<geom name="{name}" type="box" size=".01 .01 .01" pos="{i*.03} 0 0"/>'
        for i, name in enumerate(selected+excluded)) + "</worldbody></mujoco>"
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
    runtime = object.__new__(ToolUseJournalEERuntime)
    runtime._active_ee = "3F"; runtime._closed = False
    runtime._env = SimpleNamespace(sim=SimpleNamespace(
        model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data)))
    before_friction, before_condim = model.geom_friction.copy(), model.geom_condim.copy()
    names = runtime.set_finger_gripper_contact_friction((2., .1, .02))
    assert set(names) == set(selected)
    for name in selected:
        gid=model.geom(name).id
        assert model.geom_friction[gid] == pytest.approx((2., .1, .02))
        assert model.geom_condim[gid] == 6
    for name in excluded:
        gid=model.geom(name).id
        np.testing.assert_array_equal(model.geom_friction[gid], before_friction[gid])
        assert model.geom_condim[gid] == before_condim[gid]
