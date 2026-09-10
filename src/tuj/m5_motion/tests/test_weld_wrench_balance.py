import mujoco
import numpy as np
import pytest

from tuj.m5_motion.tests.test_tool_use_journal import _fake_env
from tuj.m5_motion.tool_use_journal_runtime import (
    AttachmentMode, BreakableWeldConfig, ToolUseJournalEERuntime,
)


@pytest.mark.parametrize('offset', [(0.003, 0., 0.), (0., -.002, .001)])
def test_internal_weld_conserves_momentum_with_offset_mass_centers(offset):
    runtime = ToolUseJournalEERuntime(_fake_env('2F'), _fake_env)
    try:
        model, data = runtime.env.sim.model._model, runtime.env.sim.data._data
        hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot0_right_hand')
        body, joint, _ = runtime._object_free_joint(runtime.env, 'apple')
        adr = int(model.jnt_qposadr[joint])
        # Different body-frame/CoM origins, as in mesh-based real EE assets.
        model.body_ipos[hand] = [.02, -.03, .01]
        model.body_ipos[body] = [-.01, .02, .015]
        model.body_sameframe[hand] = 0
        model.body_sameframe[body] = 0
        data.qpos[adr:adr+3] = data.xpos[hand]
        mujoco.mj_forward(model, data)
        runtime.command_gripper(engaged=True, suction=False)
        runtime.attach_object('apple', max_attach_distance_m=.05,
            max_attach_penetration_m=.05, attachment_mode=AttachmentMode.BREAKABLE_WELD,
            breakable_weld=BreakableWeldConfig(require_contact=False))
        data.qpos[adr:adr+3] += np.array(offset)
        mujoco.mj_forward(model, data)
        runtime.prepare_attachment_step()
        state = runtime._breakable_runtime
        a, b = state.object_body_id, state.reference_body_id
        wa, wb = state.applied_object_wrench, state.applied_reference_wrench
        assert np.linalg.norm(data.xipos[a] - data.xpos[a]) > .001
        assert np.linalg.norm(wa[:3]) > 0
        np.testing.assert_allclose(wa[:3] + wb[:3], 0, atol=1e-12)
        # Total torque about an arbitrary world point, not either body origin.
        origin = np.array([.1, -.2, .3])
        torque = wa[3:] + wb[3:] + np.cross(data.xipos[a]-origin, wa[:3]) + np.cross(data.xipos[b]-origin, wb[:3])
        np.testing.assert_allclose(torque, 0, atol=1e-12)
    finally:
        runtime.close()
