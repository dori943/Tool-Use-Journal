"""Opt-in calibrated hand topology, applied only while constructing M5 envs."""
from copy import deepcopy

from .catalog_timing import timing_xml


def _preserve_offscreen_buffer(env):
    """Restore the offscreen framebuffer size after the model is rebuilt.

    robosuite sizes ``visual/global`` from the requested camera dimensions when
    it first builds the environment; reloading the model here discards that and
    leaves MuJoCo's 640x480 default. Rendering a larger frame out of a smaller
    buffer reads past the rows that exist, which is what turned recorded video
    into vertical-striped noise between otherwise correct frames (c3_1 at
    960x540).
    """
    import xml.etree.ElementTree as ET

    def largest(value, fallback):
        if isinstance(value, (list, tuple)):
            values = [int(item) for item in value if item]
            return max(values) if values else fallback
        return int(value) if value else fallback

    width = largest(getattr(env, "camera_widths", None), 640)
    height = largest(getattr(env, "camera_heights", None), 480)
    root = getattr(getattr(env, "model", None), "root", None)
    if root is None:
        return
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_element = visual.find("global")
    if global_element is None:
        global_element = ET.SubElement(visual, "global")
    global_element.set("offwidth", str(max(width, 640)))
    global_element.set("offheight", str(max(height, 480)))


def configure_environment(env, environment, ee):
    """Install the lab's numerical hand corrections before the first reset."""
    from .spoon_hand_model import (exclude_parallel_2f_linkage_selfcontact,
        repair_spoon_hand_xml, repair_spoon_parallel_2f_xml)
    from tuj.m5_motion.tool_use_journal_runtime import tool_use_journal_joint_position_controller_config

    kitchen = environment != "C1_1_LegoSweep"
    if not kitchen and ee not in {"3F", "vac"}:
        # C1_1 already accepts the M5 joint controller during construction.
        # Reloading its robot model changes the calibrated hand/controller
        # initialization, so the native plate profile needs no rebuild.
        env.scripted_grasp_profile = {"environment": environment, "ee": ee,
            "correction": {"policy": "NATIVE_HAND", "source_assets_changed": False}}
        return env
    # 0911 충돌 해결: bare -> EE 랙 경로 캐시는 tabletop(ee_rack)과 부엌
    # (ee_rack_kitchen) 모두 BARE_HOME 에서 녹화돼 있다. 그래서 맨손으로
    # 시작하는 실행(ee is None)은 절대 덮어쓰면 안 된다. EE 를 이미 달고
    # 시작하는 실행은 랙 경로를 타지 않으므로 부엌 홈을 써도 안전하다.
    if kitchen and ee is not None:
        # Only a run that already carries an EE may start from the scripted
        # kitchen home. A bare start fetches its EE from the rack, and every
        # commissioned rack path begins at TOOL_USE_JOURNAL_BARE_HOME_QPOS --
        # overriding it here put the arm 1.99 rad from that seam and no cached
        # path could start (c3_1: START_STATE_MISMATCH on bare->vac). Tabletop
        # and kitchen rack caches both expect that bare home when ee is None.
        env.robot_configs[0]["initial_qpos"] = [0., -1.8, 1.2, -.97, -1.57, 0.]
    corrected = not (environment == "C1_2_DoughFlatten" and ee == "3F")
    env._load_model()
    _preserve_offscreen_buffer(env)
    correction = {"policy": "NATIVE_HAND", "source_assets_changed": False}
    if corrected and ee in {"2F", "3F"}:
        prefix = env.robots[0].gripper["right"].naming_prefix
        repair = repair_spoon_parallel_2f_xml if ee == "2F" else repair_spoon_hand_xml
        correction = repair(env.model.root, prefix)
        if ee == "2F":
            correction["excluded_linkage_contacts"] = (
                exclude_parallel_2f_linkage_selfcontact(env.model.root, prefix))
    env.set_xml_processor(lambda xml: timing_xml(xml, .001, "implicitfast"))
    env._initialize_sim()
    env.hard_reset = False
    env.scripted_grasp_profile = {"environment": environment, "ee": ee, "correction": correction}
    env.catalog_hand_correction = env.spoon_hand_correction = correction
    reset = env.reset

    def reset_with_controller(*args, **kwargs):
        observation = reset(*args, **kwargs)
        robot = env.robots[0]
        config = tool_use_journal_joint_position_controller_config(kp=150. if kitchen or ee == "vac" else 50.)
        robot.composite_controller_config = config
        robot.part_controller_config = deepcopy(config["body_parts"])
        robot._load_controller()
        env.model_timestep = float(env.sim.model.opt.timestep)
        if corrected and ee == "2F":
            # Establish the corrected linkage's rest state during environment
            # initialization, never in a borrowed grasp context.
            import numpy as np
            gripper = robot.gripper["right"]
            model, data = env.sim.model, env.sim.data
            ids = np.array([model.joint_name2id(n) for n in gripper.joints])
            data.qpos[model.jnt_qposadr[ids]] = 0.
            data.qvel[model.jnt_dofadr[ids]] = 0.
            data.ctrl[[model.actuator_name2id(n) for n in gripper.actuators]] = 0.
            gripper.current_action = np.full(gripper.dof, -1.)
            env.sim.forward()
        settle_tool_use_journal_free_objects(env, duration_s=2.)
        return observation

    env.reset = reset_with_controller
    return env


def settle_tool_use_journal_free_objects(
    env: object,
    *,
    duration_s: float = 5.0,
) -> int:
    """Advance free bodies to a deterministic resting pose with the robot fixed.

    Object placement initializers write poses and call ``mj_forward`` but do not
    integrate gravity. Planning directly from that transient state is unsafe
    for non-flat tools, which can rotate before the robot reaches them. During
    settling, every non-free joint is restored after each MuJoCo step so only
    object free joints evolve.
    """

    import math
    if not math.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError("duration_s must be finite and non-negative")
    from tuj.m5_motion.tool_use_journal import _raw_model_data
    import math
    import mujoco
    import numpy as np
    model, data = _raw_model_data(env)
    if duration_s == 0.0:
        return 0
    timestep_s = float(model.opt.timestep)
    steps = max(1, int(math.ceil(duration_s / timestep_s)))
    free_qpos = np.zeros(int(model.nq), dtype=bool)
    free_dofs = np.zeros(int(model.nv), dtype=bool)
    for joint_id in range(int(model.njnt)):
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        free_qpos[qpos_address : qpos_address + 7] = True
        free_dofs[dof_address : dof_address + 6] = True
    materials = tuple(getattr(env, "deformable_runtimes", {}).values())
    for material in materials:
        free_qpos[material.qpos] = True
        free_dofs[material.dofs] = True
    fixed_qpos = ~free_qpos
    fixed_dofs = ~free_dofs
    fixed_positions = np.asarray(data.qpos[fixed_qpos], dtype=float).copy()
    previous_ctrl = np.asarray(data.ctrl, dtype=float).copy()
    data.ctrl[:] = 0.0
    try:
        for _ in range(steps):
            if materials:
                mujoco.mj_step1(model, data)
                for material in materials:
                    material.prepare_forces()
                mujoco.mj_step2(model, data)
            else:
                mujoco.mj_step(model, data)
            data.qpos[fixed_qpos] = fixed_positions
            data.qvel[fixed_dofs] = 0.0
        mujoco.mj_forward(model, data)
    finally:
        data.ctrl[:] = previous_ctrl
    return steps
