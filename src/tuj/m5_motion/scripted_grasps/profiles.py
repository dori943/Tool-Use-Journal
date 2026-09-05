"""Opt-in calibrated hand topology, applied only while constructing M5 envs."""
from copy import deepcopy

from .catalog_timing import timing_xml


def configure_environment(env, environment, ee):
    """Install the lab's numerical hand corrections before the first reset."""
    from .spoon_hand_model import repair_spoon_hand_xml, repair_spoon_parallel_2f_xml
    from tuj.m5_motion.tool_use_journal_runtime import tool_use_journal_joint_position_controller_config

    kitchen = environment != "C1_1_LegoSweep"
    if not kitchen and ee != "3F":
        # C1_1 already accepts the M5 joint controller during construction.
        # Reloading its robot model changes the calibrated hand/controller
        # initialization, so the native plate profile needs no rebuild.
        env.scripted_grasp_profile = {"environment": environment, "ee": ee,
            "correction": {"policy": "NATIVE_HAND", "source_assets_changed": False}}
        return env
    if kitchen:
        env.robot_configs[0]["initial_qpos"] = [0., -1.8, 1.2, -.97, -1.57, 0.]
    corrected = not (environment == "C1_2_DoughFlatten" and ee == "3F")
    env._load_model()
    correction = {"policy": "NATIVE_HAND", "source_assets_changed": False}
    if corrected and ee in {"2F", "3F"}:
        prefix = env.robots[0].gripper["right"].naming_prefix
        repair = repair_spoon_parallel_2f_xml if ee == "2F" else repair_spoon_hand_xml
        correction = repair(env.model.root, prefix)
    env.set_xml_processor(lambda xml: timing_xml(xml, .001, "implicitfast"))
    env._initialize_sim()
    env.hard_reset = False
    env.scripted_grasp_profile = {"environment": environment, "ee": ee, "correction": correction}
    env.catalog_hand_correction = env.spoon_hand_correction = correction
    reset = env.reset

    def reset_with_controller(*args, **kwargs):
        observation = reset(*args, **kwargs)
        robot = env.robots[0]
        config = tool_use_journal_joint_position_controller_config(kp=150. if kitchen else 50.)
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
    fixed_qpos = ~free_qpos
    fixed_dofs = ~free_dofs
    fixed_positions = np.asarray(data.qpos[fixed_qpos], dtype=float).copy()
    previous_ctrl = np.asarray(data.ctrl, dtype=float).copy()
    data.ctrl[:] = 0.0
    try:
        for _ in range(steps):
            mujoco.mj_step(model, data)
            data.qpos[fixed_qpos] = fixed_positions
            data.qvel[fixed_dofs] = 0.0
        mujoco.mj_forward(model, data)
    finally:
        data.ctrl[:] = previous_ctrl
    return steps
