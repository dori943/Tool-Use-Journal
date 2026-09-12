"""Maintain a validated physical grasp while normal M5 arm plans execute."""
import numpy as np
from scipy.spatial.transform import Rotation

from .frames import inverse
from .runtime import GraspFailure
from .spoon_hand_model import bound_spoon_3f_commands


class GraspRetention:
    def __init__(self, context, entry):
        self.context, self.entry = context, entry
        self.reference = inverse(context.grip_pose()) @ context.body_pose()
        self.commands = np.asarray(context.gripper.current_action).copy()
        self.loss_started = None
        self.samples = []
        self.forces = self._forces()
        self.original_arm_gains = None
        gain = getattr(context.recipe, 'post_grasp_arm_kp', None)
        if gain is not None:
            controller = context.robot.part_controllers['right']
            self.original_arm_gains = (controller, controller.kp.copy(), controller.kd.copy())
            damping_ratio = controller.kd / (2. * np.sqrt(controller.kp))
            controller.kp = np.full_like(controller.kp, gain)
            controller.kd = 2. * np.sqrt(controller.kp) * damping_ratio
        if entry.driver == "plate":
            # The acquisition-only collision checks must not reject deliberate
            # tool contacts in a later task; M5's scoped collision probe owns them.
            context.physical_monitor.sample = context.original_monitor_sample

    def close(self):
        if self.original_arm_gains is not None:
            controller, kp, kd = self.original_arm_gains
            controller.kp, controller.kd = kp, kd
            self.original_arm_gains = None

    def _forces(self):
        c = self.context
        forces = {n: 0. for n in c.finger_groups}
        for i, contact in enumerate(c.data.contact[:c.data.ncon]):
            if contact.dist > 0:
                continue
            if hasattr(c, "contact_in_region") and not c.contact_in_region(contact.pos):
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            for name, geoms in c.finger_groups.items():
                if (a in geoms and b in c.handle_geoms) or (b in geoms and a in c.handle_geoms):
                    force = np.zeros(6)
                    c.mj.mj_contactForce(c.model, c.data, i, force)
                    forces[name] += max(0., float(force[0]))
        return forces

    def _snap_vac_arm_to_controller_goal(self):
        """Overwrite vac arm joints to the absolute JOINT_POSITION goal.

        Live bread_b stays attached but absolute-joint PD sags ~35 mm / 0.083 rad
        through M5 transport/place, so DETACH settle never converges. Plate_vac
        tracks; bread does not. Snapping to ``goal_qpos`` after each control tick
        (then re-syncing the kinematic attach) makes held vac M5 playback match
        the commanded waypoints the same way catalog LEVEL/HOLD already does.
        """

        c = self.context
        robot = getattr(c, 'robot', None)
        if robot is None or getattr(c, 'arm_ids', None) is None:
            return False
        if getattr(c, 'mj', None) is None or getattr(c, 'data', None) is None:
            return False
        if getattr(c, 'model', None) is None:
            return False
        controllers = getattr(robot, 'part_controllers', None) or {}
        controller = controllers.get('right')
        if controller is None:
            return False
        goal = getattr(controller, 'goal_qpos', None)
        if goal is None:
            return False
        goal = np.asarray(goal, dtype=float).reshape(-1)
        arm = np.asarray(c.arm_ids, dtype=int)
        if goal.shape != arm.shape:
            return False
        c.data.qpos[arm] = goal
        vel = getattr(robot, '_ref_joint_vel_indexes', None)
        if vel is not None:
            c.data.qvel[np.asarray(vel, dtype=int)] = 0.0
        c.mj.mj_fwdPosition(c.model, c.data)
        runtime = getattr(c, 'runtime', None)
        if runtime is not None and getattr(runtime, 'attachment', None) is not None:
            runtime.synchronize_attached_object()
            dadr = getattr(c, 'object_dadr', None)
            if dadr is not None:
                c.data.qvel[int(dadr):int(dadr) + 6] = 0.0
            c.mj.mj_fwdPosition(c.model, c.data)
        return True

    def before_tick(self, action):
        c, recipe = self.context, self.context.recipe
        if c.runtime.env is not c.env:
            raise GraspFailure("RUNTIME_CHANGED_WITH_HELD_OBJECT")
        if self.entry.ee == "vac" or self.entry.driver == "plate":
            return action
        action = np.asarray(action).copy()
        if c.three_finger_force_hold:
            measured = np.array([self.forces[n] for n in ("thumb", "index", "pinky")])
            from .spatula_runtime import update_three_finger_commands
            if not getattr(recipe, 'hold_finger_positions', False):
                self.commands = update_three_finger_commands(self.commands, measured, recipe)
                command_min=getattr(c,'three_finger_hold_command_min',None)
                if command_min is not None:
                    self.commands=np.clip(self.commands,command_min,
                        c.three_finger_hold_command_max)
            if self.entry.driver == "catalog":
                self.commands = bound_spoon_3f_commands(self.commands)
        elif c.two_finger_force_hold:
            delta = np.clip(recipe.two_finger_force_gain * (recipe.two_finger_force_target_n - min(self.forces.values())), -.005, .005)
            self.commands = np.clip(self.commands + delta, -1., 1.)
        c.gripper.current_action = self.commands.copy()
        lo, hi = c.robot.composite_controller._action_split_indexes["right_gripper"]
        action[lo:hi] = 0.
        return action

    def audit_substep(self):
        audit = getattr(self.context, "audit_hand_range", None)
        if audit is not None:
            audit()

    def after_tick(self, time_s):
        c = self.context
        if self.entry.ee == "vac":
            self._snap_vac_arm_to_controller_goal()
        self.forces = self._forces()
        actual = inverse(c.grip_pose()) @ c.body_pose()
        slip = float(np.linalg.norm(actual[:3, 3] - self.reference[:3, 3]))
        angle = float(np.rad2deg(Rotation.from_matrix(self.reference[:3, :3].T @ actual[:3, :3]).magnitude()))
        # Attachments are keyed by scene instance (fruit_a / plate_b), not recipe
        # type id. Vac multi-instance holds compare those ids alone.
        attached = (
            getattr(c.runtime, "attached_object_id", None) == self.entry.scene_object_id
            and getattr(c.runtime, "attachment", None) is not None
        )
        if self.entry.ee == "vac":
            contact = (
                getattr(c.runtime, "attached_object_id", None)
                == self.entry.scene_object_id
            )
        elif attached:
            # Post-validated kinematic 3F/2F carry: finger pads can unload under
            # M5 dynamics while slip stays tiny; requiring all finger forces then
            # false-triggers SCRIPTED_GRASP_CONTACT_LOST (c3_2 fruit_a transport).
            contact = True
        else:
            contact = all(force > .01 for force in self.forces.values())
        self.loss_started = None if contact else (time_s if self.loss_started is None else self.loss_started)
        self.samples.append({
            "time_s": time_s,
            "contact": contact,
            "slip_m": slip,
            "slip_deg": angle,
            "finger_force_n": self.forces.copy(),
            "attached": bool(attached),
        })
        if self.entry.driver == "plate":
            c.original_monitor_sample(time_s)
        if self.loss_started is not None and time_s - self.loss_started > .10:
            raise GraspFailure("SCRIPTED_GRASP_CONTACT_LOST")
        if slip > .005 or angle > 5.:
            raise GraspFailure(f"SCRIPTED_GRASP_SLIPPED: {slip:.6f} m, {angle:.3f} deg")

    def transform(self):
        from tuj.m5_motion.schema import AttachedObjectTransform
        c = self.context
        actual = inverse(c.grip_pose()) @ c.body_pose()
        joint_id = int(c.model.body_jntadr[c.body_id])
        return AttachedObjectTransform(object_id=self.entry.scene_object_id, reference_kind="site",
            free_joint_name=c.model.joint(joint_id).name,
            reference_name=c.gripper.important_sites["grip_site"],
            position_in_reference_m=tuple(actual[:3, 3]),
            orientation_in_reference_xyzw=tuple(Rotation.from_matrix(actual[:3, :3]).as_quat()))

    def reference_transform(self):
        """Return the immutable transform captured when retention began."""
        from tuj.m5_motion.schema import AttachedObjectTransform
        c = self.context
        joint_id = int(c.model.body_jntadr[c.body_id])
        return AttachedObjectTransform(
            object_id=self.entry.object_id,
            reference_kind="site",
            free_joint_name=c.model.joint(joint_id).name,
            reference_name=c.gripper.important_sites["grip_site"],
            position_in_reference_m=tuple(self.reference[:3, 3]),
            orientation_in_reference_xyzw=tuple(
                Rotation.from_matrix(self.reference[:3, :3]).as_quat()
            ),
        )
