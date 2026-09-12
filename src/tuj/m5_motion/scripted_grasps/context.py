"""Bind calibrated grasp drivers to a borrowed M5 runtime without resetting it."""
from importlib import import_module
from pathlib import Path
from types import MethodType

import numpy as np

from .frames import inverse, pose_dict
from .runtime import GraspFailure, save_json


def bind_context(runtime, entry, output, *, seed=0, request=None):
    import mujoco
    from tuj.m5_motion.tool_use_journal import ToolUseJournalEnvironmentAdapter, _object_record
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalControllerTrajectoryPlayer
    from .ik_continuity import ContinuousIK
    from .catalog_runtime import CatalogContext

    recipe = entry.recipe()
    if runtime.active_ee != entry.ee:
        raise GraspFailure(f"MOUNT_REQUIRED: {entry.object_id} requires {entry.ee}")
    if runtime.held_tool_id is not None or runtime.attachment is not None:
        raise GraspFailure("RELEASE_CURRENT_OBJECT_BEFORE_ACQUIRE")
    env = runtime.env
    profile = getattr(env, "scripted_grasp_profile", None)
    if profile is None or profile["environment"] != entry.environment or profile["ee"] != entry.ee:
        raise GraspFailure("SCRIPTED_ENVIRONMENT_PROFILE_REQUIRED")
    module = import_module(f"{__package__}." + ("runtime" if entry.driver == "plate" else entry.driver + "_runtime"))
    cls = getattr(module, entry.driver.title() + "Context")
    c = cls()
    c.runtime, c.env, c.recipe, c.object_id = runtime, env, recipe, entry.object_id
    c.output = Path(output)
    c.output.mkdir(parents=True, exist_ok=False)
    c.mj = mujoco
    c.adapter = ToolUseJournalEnvironmentAdapter(env)
    c.model, c.data = c.adapter.model, c.adapter.data
    c.robot = env.robots[0]
    c.gripper = c.robot.gripper["right"]
    if type(c.gripper).__name__ != recipe.model_class:
        raise GraspFailure("UNSUPPORTED_EE_MODEL")
    c.arm_ids = np.asarray(c.robot._ref_joint_pos_indexes)
    c.hand_joint_ids = np.array([c.model.joint(n).id for n in c.gripper.joints], dtype=int)
    c.body_id = int(env.obj_body_id[entry.object_id])
    c.site_id = c.model.site(c.gripper.important_sites["grip_site"]).id
    jid = int(c.model.body_jntadr[c.body_id])
    if jid < 0 or c.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise GraspFailure("TARGET_HAS_NO_FREE_JOINT")
    c.object_qadr = c.plate_qadr = int(c.model.jnt_qposadr[jid])
    c.object_dadr = c.plate_dadr = int(c.model.jnt_dofadr[jid])
    record = _object_record(c.model, c.data, entry.object_id, c.body_id)
    c.center_in_body = np.asarray(record["anchors"]["center"])
    c.local_size = c.collision_local_size = np.asarray(record["dimensions_m"])
    # Plate uses a frozen partial-view calibration; changed visible bbox cannot
    # displace its rim anchor. The full compiled bbox separately checks asset size.
    c.observed_size = np.asarray(getattr(recipe, "calibrated_local_bbox_size_m", c.local_size))
    c.gripper_geoms = {i for i in range(c.model.ngeom) if c.model.geom(i).name.startswith(c.gripper.naming_prefix)}
    c.object_geoms = c.plate_geoms = {i for i in range(c.model.ngeom) if c.descendant(int(c.model.geom_bodyid[i]), c.body_id)}
    names = ("left", "right") if entry.ee == "2F" else ("thumb", "index", "pinky")
    if entry.ee == "vac":
        c.finger_groups = {"suction": {c.model.geom(c.gripper.naming_prefix + "vac_cup").id}}
    elif entry.driver == "plate":
        c.finger_groups = {n: {c.model.geom(g).id for g in c.gripper.important_geoms[n + "_finger"]} for n in names}
    else:
        c.finger_groups = {n: {i for i in c.gripper_geoms if c.model.geom(i).name.startswith(c.gripper.naming_prefix + n + "_") and c.model.geom_contype[i]} for n in names}
    c.finger_geoms = set.union(*c.finger_groups.values())
    c.handle_geoms = {i for i in c.object_geoms if c.model.geom_contype[i]}
    if entry.driver in {"spoon", "spatula"}:
        c.handle_geoms = {i for i in c.handle_geoms if c.model.geom_dataid[i] >= 0 and c.model.mesh(int(c.model.geom_dataid[i])).name.endswith(entry.object_id + "_collision_mesh_0")}
        if not c.handle_geoms:
            raise GraspFailure("HANDLE_GEOMETRY_NOT_FOUND")
    root = c.model.body(c.robot.robot_model.root_body).id
    c.robot_geoms = {i for i in range(c.model.ngeom) if c.descendant(int(c.model.geom_bodyid[i]), root)}
    c.fixed_mount_pairs = {tuple(sorted((int(con.geom1), int(con.geom2)))) for con in c.data.contact[:c.data.ncon]
        if con.dist < 0 and (int(con.geom1) in c.robot_geoms or int(con.geom2) in c.robot_geoms)
        and int(con.geom1) not in c.gripper_geoms and int(con.geom2) not in c.gripper_geoms}
    c.support_gids = CatalogContext.find_support_geoms(c)
    c.support_gid = min(c.support_gids)
    c.support_top_z = max(CatalogContext.geom_top_height(c, gid)
        for gid in c.support_gids)
    c.thin_contact_profile = None
    if hasattr(recipe, "thin_contact_timeconstant_s"):
        relevant = c.gripper_geoms | c.object_geoms | c.support_gids
        original = {c.model.geom(i).name: c.model.geom_solref[i].tolist() for i in relevant}
        solimp = getattr(recipe, "thin_contact_solimp", None)
        original_solimp = (
            {c.model.geom(i).name: c.model.geom_solimp[i].tolist() for i in relevant}
            if solimp is not None else None)
        for i in relevant:
            c.model.geom_solref[i] = [recipe.thin_contact_timeconstant_s, 1.]
            if solimp is not None:
                c.model.geom_solimp[i][:3] = list(solimp)
        c.thin_contact_profile = {"timeconstant_s": recipe.thin_contact_timeconstant_s, "original_solref": original,
            "solimp": list(solimp) if solimp is not None else None, "original_solimp": original_solimp}
    c.vacuum_cup_margin_profile = None
    cup_margin = getattr(recipe, "vacuum_cup_margin_m", None)
    if cup_margin is not None and entry.ee == "vac":
        # 흡착이 받침을 통해 딸려 올라가지 않도록 컵의 접촉 탐지 범위를 좁힌다.
        # gap 을 margin 과 같게 유지해야 접촉력 발생 시점(dist<0)이 안 바뀐다.
        cup_ids = sorted(c.finger_groups["suction"])
        c.vacuum_cup_margin_profile = {"margin_m": float(cup_margin),
            "original": {c.model.geom(i).name: [float(c.model.geom_margin[i]), float(c.model.geom_gap[i])] for i in cup_ids}}
        for i in cup_ids:
            c.model.geom_margin[i] = float(cup_margin)
            c.model.geom_gap[i] = float(cup_margin)
    c.ee_spec = next(s for s in env.robot_spec["ee_pool"] if s["ee_id"] == entry.ee)
    c.gripper_actuator_ids = np.array([i for i in range(c.model.nu) if c.model.actuator(i).name.startswith(c.gripper.naming_prefix)])
    if entry.driver == "plate":
        runtime.set_finger_gripper_contact_friction(c.ee_spec["fingerpad_friction"])
    c.initial_body, c.initial_bottom = c.body_pose(), c.bottom_height()
    c.seed, c.frequency, c.tick = seed, 50, 0
    c.scenario = {"seed": seed, "mode": "BORROWED_M5_RUNTIME", "request_id": getattr(request, "request_id", None)}
    c.trace, c.plans, c.stage = [], [], "INITIALIZE"
    c.video, c.writer, c.camera = False, None, None
    c.player = ToolUseJournalControllerTrajectoryPlayer(runtime)
    c.kinematics = ContinuousIK(c.adapter.make_kinematics(), c.data.qpos[c.arm_ids])
    c.probe = mujoco.MjData(c.model)
    c.carried_pose = None
    c.support_released = False
    c.vacuum_attachment_record = None
    c.three_finger_force_hold, c.two_finger_force_hold = False, False
    c.three_finger_commands, c.two_finger_command = None, 0.
    # Shared 3F force-hold clamp bounds. The runtime step reads these, but only
    # the spoon path (execute_spoon) initialised them; a catalog 3F grasp (bread
    # /apple/mug) hit AttributeError. Initialise here for every context; the
    # step leaves the clamp inactive while they stay None.
    c.three_finger_hold_command_min = c.three_finger_hold_command_max = None
    c.physics_steps_audited, c.maximum_physics_joint_error = 0, 0.
    c.max_runtime_s = c.execution_started = None
    c.timing = {"physics_timestep_s": float(c.model.opt.timestep), "control_timestep_s": float(env.control_timestep), "clock_source": "MUJOCO_DATA_TIME"}
    if not np.isclose(env.control_timestep, .02) or not np.isclose(env.model_timestep, c.model.opt.timestep):
        raise GraspFailure("CONTROL_TIMING_MISMATCH")

    def record_input(self):
        save_json(self.output / "input.json", {"object_id": entry.object_id, "recipe": recipe.to_dict(),
            "scenario": self.scenario, "T_WB": self.body_pose(), "center_in_body_m": self.center_in_body,
            "local_bbox_size_m": self.local_size, "geometry_source": "M5_COMPILED_OBJECT_RECORD",
            "pose_source": "M5_LIVE_RUNTIME", "object_material_inputs": [], "learned_model_calls": 0})
    c.record_input = MethodType(record_input, c)
    return c


def execute_grasp(runtime, entry, output, *, seed=0, request=None):
    """Call one object function and retain real state only after validation."""
    c = bind_context(runtime, entry, output, seed=seed, request=request)
    arm_controller = getattr(getattr(c, 'robot', None), 'part_controllers', {}).get('right')
    original_arm_gains = None
    if arm_controller is not None and hasattr(c.recipe, 'arm_kp'):
        original_arm_gains = (arm_controller.kp.copy(), arm_controller.kd.copy())
        damping_ratio = arm_controller.kd / (2. * np.sqrt(arm_controller.kp))
        runtime.set_joint_position_controller_gains(
            kp=c.recipe.arm_kp, damping_ratio=float(np.asarray(damping_ratio).mean()))
    finish = runtime.finish_attachment_step
    audit = getattr(c, "audit_hand_range", None)
    if audit is not None:
        def finish_audited():
            finish()
            audit()
        runtime.finish_attachment_step = finish_audited
    try:
        result = entry.function()(c)
    finally:
        runtime.finish_attachment_step = finish
        if original_arm_gains is not None:
            arm_controller.kp, arm_controller.kd = original_arm_gains
    if result["status"] != "SUCCESS":
        raise GraspFailure(f"{entry.object_id}: {result.get('failure_stage')}: {result.get('failure_reason')}")
    result["acquisition_status"] = result["status"]
    result["acquisition_attachment_used"] = result.get("attachment_used", False)
    runtime.command_gripper(engaged=True, suction=entry.ee == "vac", command=1.)
    try:
        if runtime.attachment is None:
            # Acquisition must pass its physical hold validation before the
            # existing runtime applies its contact-gated kinematic attachment.
            runtime.attach_object(entry.object_id)
        if runtime.attachment.object_id != entry.object_id:
            raise GraspFailure("POST_GRASP_ATTACHMENT_OBJECT_MISMATCH")
        runtime.mark_attached_object_as_tool(entry.object_id)
    except Exception as error:
        result.update(status="FAILED", failure_stage="POST_GRASP_ATTACHMENT",
                      failure_reason=str(error), error_type=type(error).__name__,
                      attachment_used=runtime.attachment is not None)
        save_json(c.output / "result.json", result)
        raise
    result.update(attachment_used=True, attachment_mode=runtime.attachment.mode.value,
                  attachment_phase="POST_VALIDATED_GRASP" if not result["acquisition_attachment_used"] else "ACQUISITION")
    from .retention import GraspRetention
    runtime.scripted_grasp_retention = GraspRetention(c, entry)
    result.update(final_robot_q=c.data.qpos[c.arm_ids].tolist(), object_pose_in_gripper=pose_dict(inverse(c.grip_pose()) @ c.body_pose()))
    save_json(c.output / "result.json", result)
    return result
