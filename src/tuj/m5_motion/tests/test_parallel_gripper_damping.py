"""The calibrated linkage must settle under a constant actuator target."""
import numpy as np
from pathlib import Path
import time
from robosuite.models.grippers import Robotiq85Gripper

from tuj.m5_motion.scripted_grasps.spoon_hand_model import (
    exclude_parallel_2f_linkage_selfcontact,
    repair_spoon_parallel_2f_xml,
)


def test_parallel_gripper_constant_target_settles_without_relaxing_limits(tmp_path):
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime
    from tuj.m5_motion.scripted_grasps.context import bind_context
    from tuj.m5_motion.scripted_grasps.registry import ENTRIES

    # Use the assembled scene: a free-standing XML gripper has different
    # self-contact exclusions and cannot reproduce the borrowed M5 controller.
    repository = Path(__file__).resolve().parents[4]
    runtime = ToolUseJournalEERuntime.from_repository_for_controller(
        repository, 'C4_2_DiagonalFitPacking', active_ee='2F', seed=0,
        scripted_grasps=True, ignore_done=True, has_renderer=False,
        has_offscreen_renderer=False, use_camera_obs=False)
    try:
        entry = next(e for e in ENTRIES if e.object_id == 'cereal')
        context = bind_context(runtime, entry, tmp_path / 'grasp', seed=0)
        context.execution_started = time.monotonic()
        context.max_runtime_s = 120
        assert np.all(context.model.actuator_biasprm[context.gripper_actuator_ids, 2] < 0)
        opening, _ = context.preshape()
        target = context.recipe.preshape_aperture_m
        q = context.data.qpos[context.arm_ids].copy()
        for _ in range(20):
            context.step(q, opening)
            context.audit_hand_range()
            # Same sustained aperture acceptance as the production recovery.
            assert abs(runtime.fingerpad_separation_m() - target) <= .003
    finally:
        runtime.close()


def test_parallel_gripper_keeps_explicit_existing_actuator_damping():
    gripper = Robotiq85Gripper()
    actuators = gripper.root.findall('./actuator/position')
    actuators[0].set('kv', '.2')
    actuators[1].set('dampratio', '.7')
    repair_spoon_parallel_2f_xml(gripper.root, gripper.naming_prefix)
    assert actuators[0].get('kv') == '.2'
    assert actuators[0].get('dampratio') is None
    assert actuators[1].get('dampratio') == '.7'
