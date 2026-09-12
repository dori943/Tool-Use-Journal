import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.m5_motion.tests.test_tool_use_journal import _fake_env, _source_xml, ARM_JOINTS
from tuj.m5_motion.tool_use_journal_runtime import (
    AttachmentMode, BreakableWeldConfig, ToolUseJournalEERuntime,
)


@pytest.mark.parametrize('reference_kind', ['body', 'site'])
@pytest.mark.parametrize('integrate', [False, True])
def test_projection_uses_current_hand_pose_after_qpos_changes(reference_kind, integrate):
    env = _fake_env('2F')
    if reference_kind == 'site':
        xml = _source_xml('2F').replace(
            '<body name="robot0_right_hand" pos="0 0 0.08">',
            '<body name="robot0_right_hand" pos="0 0 0.08">'
            '<site name="test_grip" pos=".003 .004 .002" euler="20 10 30"/>')
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        env.sim.model._model, env.sim.data._data = model, data
        env.robots[0].gripper['right'].important_sites = {'grip_site': 'test_grip'}
        mujoco.mj_forward(model, data)
    runtime = ToolUseJournalEERuntime(env, _fake_env)
    try:
        m, d = env.sim.model._model, env.sim.data._data
        body, joint, _ = runtime._object_free_joint(env, 'apple')
        adr, dof = int(m.jnt_qposadr[joint]), int(m.jnt_dofadr[joint])
        _, _, p, _ = runtime._grasp_reference(env)
        d.qpos[adr:adr + 3] = p + [.02, .003, .002]
        q = Rotation.from_euler('xyz', [13, 27, -17], degrees=True).as_quat()
        d.qpos[adr + 3:adr + 7] = q[[3, 0, 1, 2]]
        mujoco.mj_forward(m, d)
        runtime.command_gripper(engaged=True, suction=False)
        attached = runtime.attach_object('apple', max_attach_distance_m=.05, max_attach_penetration_m=.05)
        assert attached.reference_kind == reference_kind
        arm_joint = m.joint(ARM_JOINTS[0]).id
        if integrate:
            d.qvel[int(m.jnt_dofadr[arm_joint])] = 1.
            mujoco.mj_step(m, d)
        else:
            d.qpos[int(m.jnt_qposadr[arm_joint])] += .01
        qpos, qvel, ctrl, time = d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), float(d.time)
        runtime.synchronize_attached_object()
        p, r = runtime._reference_pose(env, attached.reference_kind, attached.reference_name)
        np.testing.assert_allclose(d.xpos[body], p + r @ attached.position_in_reference_m, atol=1e-12, rtol=0)
        np.testing.assert_allclose(d.xmat[body].reshape(3, 3), r @ attached.rotation_in_reference, atol=1e-12, rtol=0)
        keep = np.ones(m.nq, dtype=bool)
        keep[adr:adr + 7] = False
        np.testing.assert_array_equal(d.qpos[keep], qpos[keep])
        np.testing.assert_array_equal(d.ctrl, ctrl)
        assert float(d.time) == time
        keep_vel = np.ones(m.nv, dtype=bool)
        keep_vel[dof:dof + 6] = False
        np.testing.assert_array_equal(d.qvel[keep_vel], qvel[keep_vel])
        # Independent finite-difference oracle: advance the reference's joint
        # configuration and differentiate the projected object root pose.
        probe = mujoco.MjData(m)
        probe.qpos[:] = d.qpos
        epsilon = 1e-7
        mujoco.mj_integratePos(m, probe.qpos, d.qvel, epsilon)
        mujoco.mj_kinematics(m, probe)
        if reference_kind == 'site':
            idx = m.site(attached.reference_name).id
            next_p, next_r = probe.site_xpos[idx], probe.site_xmat[idx].reshape(3, 3)
        else:
            idx = m.body(attached.reference_name).id
            next_p, next_r = probe.xpos[idx], probe.xmat[idx].reshape(3, 3)
        next_position = next_p + next_r @ attached.position_in_reference_m
        next_rotation = next_r @ attached.rotation_in_reference
        current_rotation = d.xmat[body].reshape(3, 3)
        expected_twist = np.r_[
            (next_position - d.xpos[body]) / epsilon,
            Rotation.from_matrix(current_rotation.T @ next_rotation).as_rotvec() / epsilon,
        ]
        np.testing.assert_allclose(d.qvel[dof:dof + 6], expected_twist, atol=1e-7, rtol=1e-5)
        if not integrate:
            np.testing.assert_allclose(d.qvel[dof:dof + 6], 0., atol=1e-12)
        else:
            assert np.linalg.norm(d.qvel[dof:dof + 6]) > .1
        # Release keeps the established zero-velocity policy and the world pose.
        release_pose = d.qpos[adr:adr + 7].copy()
        runtime.detach_object('apple')
        np.testing.assert_array_equal(d.qpos[adr:adr + 7], release_pose)
        np.testing.assert_array_equal(d.qvel[dof:dof + 6], 0.)
    finally:
        runtime.close()


def test_breakable_attachment_does_not_project_or_refresh_its_reference():
    runtime = ToolUseJournalEERuntime(_fake_env('2F'), _fake_env)
    try:
        m, d = runtime.env.sim.model._model, runtime.env.sim.data._data
        _, joint, _ = runtime._object_free_joint(runtime.env, 'apple')
        adr = int(m.jnt_qposadr[joint])
        d.qpos[adr:adr + 3] = d.xpos[m.body('robot0_right_hand').id]
        mujoco.mj_forward(m, d)
        runtime.command_gripper(engaged=True, suction=False)
        runtime.attach_object('apple', max_attach_distance_m=.05, max_attach_penetration_m=.05,
                              attachment_mode=AttachmentMode.BREAKABLE_WELD,
                              breakable_weld=BreakableWeldConfig(require_contact=False))
        d.qpos[int(m.jnt_qposadr[m.joint(ARM_JOINTS[0]).id])] += .01
        qpos, xpos = d.qpos.copy(), d.xpos.copy()
        runtime.synchronize_attached_object()
        np.testing.assert_array_equal(d.qpos, qpos)
        np.testing.assert_array_equal(d.xpos, xpos)
    finally:
        runtime.close()
