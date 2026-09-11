"""Canonical rack paths retain their wrist reference when motion IK uses a TCP."""
from pathlib import Path

import numpy as np
import pytest

from tuj.m5_motion.precomputed_ee_attach import (
    EEAttachTrajectoryTemplate,
    validate_portable_start_pose,
)
from tuj.m5_motion.tool_use_journal import (
    ToolUseJournalEnvironmentAdapter,
    make_tool_use_journal_env,
)
from tuj.m5_motion.tool_use_journal_planning import ToolUseJournalMotionRequestPlanner


def test_actual_bare_robot_validates_wrist_artifact_and_rejects_displaced_rack():
    root = Path(__file__).resolve().parents[4]
    env = make_tool_use_journal_env(root, "C3_1_ObjectSorting", active_ee=None, seed=0)
    try:
        env.reset()
        adapter = ToolUseJournalEnvironmentAdapter(env)
        world = adapter.world_snapshot()
        planner = ToolUseJournalMotionRequestPlanner.from_environment(
            env, root, provider=object(), log=None
        )
        template = EEAttachTrajectoryTemplate.model_validate_json(
            (root / "configs/precomputed_ee_paths/ee_rack/bare_to_vac.json").read_text(encoding="utf-8")
        )
        fk = planner.precomputed_ee_attach_planner.forward_kinematics
        validate_portable_start_pose(world, template, forward_kinematics=fk)
        # The motion TCP really is different; a zero offset fixture would miss
        # the production regression caused by NullGripper's 90-degree rotation.
        _, wrist_q = fk.forward_pose_world(template.start_joint_positions_rad)
        _, tcp_q = adapter.make_kinematics().forward_pose_world(template.start_joint_positions_rad)
        assert 2 * np.arccos(abs(np.dot(wrist_q, tcp_q))) == pytest.approx(np.pi / 2)
        with pytest.raises(ValueError, match="orientation error"):
            validate_portable_start_pose(world, template, forward_kinematics=adapter.make_kinematics())
        moved = world.model_copy(deep=True)
        for rack in moved.rack.values():
            pose = rack["dock_pose"]
            pose["position_m"] = [pose["position_m"][0] + 0.1, *pose["position_m"][1:]]
        with pytest.raises(ValueError, match="position error"):
            validate_portable_start_pose(moved, template, forward_kinematics=fk)
    finally:
        env.close()
