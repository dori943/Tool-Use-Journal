"""Contact-gated vacuum attachment using the existing M5 runtime."""
from dataclasses import asdict, replace

import numpy as np
from scipy.spatial.transform import Rotation

from tuj.m5_motion.scripted_grasps.frames import inverse, pose_dict
from tuj.m5_motion.scripted_grasps.runtime import GraspFailure, save_json

VACUUM_POLICY = 'CONTACT_GATED_KINEMATIC_ATTACH'
# Contact-gate proves cup contact; CLOSE suction can still tilt/depress a free
# body into support. Attachment continuity uses the controller GRASP snapshot.
GRASP_RELATIVE_POSE_SOURCE = 'CONTROLLER_GRASP_OBJECT_POSE'
# AABB/soft-contact noise floor: treat clearances inside this as already clear
# (matches suction surface eps scale; does not widen LIFT collision whitelist).
VACUUM_SUPPORT_CONTACT_EPS_M = 1e-4
# Post-breakaway clearance target / early-LIFT max exempted plate↔support
# penetration in CatalogContext.bad_contacts. Soft-contact noise on C1_1 table
# plate vac routinely reports ~2.03 mm; keep a half-millimetre margin above the
# historical 2 mm pad so LIFT does not fail on the first tick after attach.
VACUUM_SUPPORT_CLEARANCE_PAD_M = 0.0025
# Extra headroom so early-LIFT controller dip after kinematic breakaway does not
# re-immerse the held object. Live bread_b: post-breakaway TCP dipped ~9.5 mm
# on the first LIFT ticks (clr +1.9 mm → -7.1 mm).
VACUUM_POST_BREAKAWAY_LIFT_DIP_MARGIN_M = 0.01
# Soft resting immersion deeper than this is treated as an invalid scene state.
MAX_VACUUM_SUPPORT_BREAKAWAY_M = 0.02
# Live bread_b vac grasp inherits ~7° object tilt. Absolute JOINT_POSITION
# playback then sags ~0.083 rad / ~35 mm during M5 transport and fails the
# place DETACH settle gate (0.05 rad / 5 mm). Level the tool after LIFT when
# tilt exceeds this threshold so held M5 motions track like plate_vac.
VACUUM_POST_LIFT_LEVEL_TILT_RAD = np.deg2rad(2.0)


def object_support_penetration_m(context):
    """Max MuJoCo penetration between the held object and known support geoms."""

    support = set(context.support_geom_names())
    object_names = {context.model.geom(int(g)).name for g in context.object_geoms}
    penetration = 0.0
    data = context.data
    for con in data.contact[: int(data.ncon)]:
        if float(con.dist) >= 0.0:
            continue
        names = {
            context.model.geom(int(con.geom1)).name,
            context.model.geom(int(con.geom2)).name,
        }
        if names & support and names & object_names:
            penetration = max(penetration, -float(con.dist))
    return float(penetration)


def support_bottom_clearance_m(context):
    """Signed clearance of the held object's AABB bottom above the support top."""

    return float(context.bottom_height() - context.support_top_z)


def vacuum_support_breakaway_lift_m(clearance_m, mesh_penetration_m=0.0):
    """Bounded lift so a vac-held object clears known support.

    Uses the worse of AABB bottom immersion and direct object↔support mesh
    penetration. Always leaves ``pad + post-breakaway dip margin`` of clearance
    so the first impedance LIFT ticks cannot re-immerse the object when the
    AABB looked "clear" but soft contacts remain (live C1_1 plate↔table).
    Returns 0 when already at/above that target. Raises when required lift
    exceeds the safety bound (deep/invalid immersion, not soft resting contact).
    """

    clearance = float(clearance_m)
    mesh_pen = max(0.0, float(mesh_penetration_m))
    desired_clearance = (
        VACUUM_SUPPORT_CLEARANCE_PAD_M
        + VACUUM_POST_BREAKAWAY_LIFT_DIP_MARGIN_M
    )
    lift = max(0.0, desired_clearance - clearance)
    if mesh_pen > VACUUM_SUPPORT_CONTACT_EPS_M:
        lift = max(lift, mesh_pen + desired_clearance)
    if lift <= VACUUM_SUPPORT_CONTACT_EPS_M:
        return 0.0
    if lift > MAX_VACUUM_SUPPORT_BREAKAWAY_M + 1e-12:
        raise GraspFailure(
            'SUPPORT_BREAKAWAY_EXCEEDS_BOUND: '
            f'need {lift:.6f} m > max {MAX_VACUUM_SUPPORT_BREAKAWAY_M:.6f} m'
        )
    return float(lift)


def _breakaway_direction(grip_pose):
    """Unit direction away from a horizontal support along the tool approach."""

    grip_z = np.asarray(grip_pose[:3, 2], dtype=float)
    world_up = np.array([0.0, 0.0, 1.0])
    if float(grip_z @ world_up) < -0.5:
        direction = -grip_z
    else:
        direction = world_up
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1e-12:
        return world_up
    return direction / norm


def _zero_arm_qvel(context):
    vel = getattr(context.robot, '_ref_joint_vel_indexes', None)
    if vel is not None:
        context.data.qvel[np.asarray(vel, dtype=int)] = 0.0


def _sync_absolute_joint_goal(context, q):
    """Align the absolute JOINT_POSITION goal with a kinematically set arm q."""

    controllers = getattr(context.robot, 'part_controllers', None) or {}
    controller = controllers.get('right')
    if controller is None:
        return
    q = np.asarray(q, dtype=float).reshape(-1)
    if getattr(controller, 'input_type', None) == 'absolute':
        controller.goal_qpos = q.copy()
    update = getattr(controller, 'update', None)
    if callable(update):
        update()


def _apply_vac_arm_waypoint(context, waypoint):
    """Set arm joints, zero velocity, FK, and sync the kinematic vac attach."""

    arm = np.asarray(context.arm_ids, dtype=int)
    context.data.qpos[arm] = np.asarray(waypoint, dtype=float)
    _zero_arm_qvel(context)
    context.mj.mj_fwdPosition(context.model, context.data)
    if context.runtime.attachment is not None:
        context.runtime.synchronize_attached_object()
        dadr = getattr(context, 'object_dadr', None)
        if dadr is not None:
            context.data.qvel[int(dadr):int(dadr) + 6] = 0.0
        context.mj.mj_fwdPosition(context.model, context.data)
    context.mj.mj_collision(context.model, context.data)


def _play_vac_kinematic_path(context, path, stage, opening):
    """FK each waypoint, then one controller tick so video captures the climb.

    Pure qpos writes skip ``_advance_controller`` / ``scripted_render``, so a
    long LIFT looks like a teleport in recorded video. Sync the absolute joint
    goal and ``step`` once per waypoint to keep support-safe kinematics while
    emitting frames.
    """

    context.stage = stage
    q_final = None
    for waypoint in np.asarray(path, dtype=float):
        waypoint = np.asarray(waypoint, dtype=float).reshape(-1)
        _apply_vac_arm_waypoint(context, waypoint)
        _sync_absolute_joint_goal(context, waypoint)
        context.step(waypoint, opening)
        context.sample()
        q_final = waypoint
    if q_final is None:
        raise GraspFailure(f'{stage}_KINEMATIC_PATH_EMPTY')
    return q_final


def _kinematic_breakaway_to(context, target, opening):
    """Play BREAKAWAY by FK + attach sync, with rendered controller ticks.

    Live bread: kinematic clear to +12 mm, then PD settle/LIFT tracked back
    toward the pre-breakaway pose (~20 mm TCP dip) and re-immersed meshes.
    """

    path = np.asarray(context.plan_to(target, 'BREAKAWAY', cartesian=True), dtype=float)
    return _play_vac_kinematic_path(context, path, 'BREAKAWAY', opening)


def move_vacuum_cartesian_kinematic(context, target, stage, opening, settle_steps=50):
    """Cartesian vac move via FK playback, then PD settle once far from support.

    Used for LIFT after support breakaway: absolute joint PD tracks back into
    the pre-breakaway configuration for the first ticks (~20 mm TCP dip on
    bread_b), so early LIFT must not rely on impedance near the island.
    Each waypoint is followed by one controller tick so recorded video shows a
    continuous climb instead of a teleport.
    """

    path = np.asarray(context.plan_to(target, stage, cartesian=True), dtype=float)
    q = _play_vac_kinematic_path(context, path, stage, opening)
    for _ in range(int(settle_steps)):
        context.step(q, opening)
    return q


def settle_vacuum_arm_tracking(context, q, opening, tol_rad=0.01, max_steps=250):
    """Hold an absolute joint target until PD tracking converges (or budget).

    Kinematic vac LIFT can leave the absolute JOINT_POSITION goal / actual pair
    far enough that the next M5 contact-sensitive settle gate times out.
    """

    q = np.asarray(q, dtype=float).reshape(-1)
    _sync_absolute_joint_goal(context, q)
    arm = np.asarray(context.arm_ids, dtype=int)
    last_err = float('inf')
    streak = 0
    for _ in range(int(max_steps)):
        context.step(q, opening)
        last_err = float(np.linalg.norm(np.asarray(context.data.qpos[arm], dtype=float) - q))
        if last_err <= float(tol_rad):
            streak += 1
            if streak >= 3:
                return last_err
        else:
            streak = 0
    return last_err


def vacuum_tool_tilt_from_world_down_rad(grip_pose):
    """Angle between grip -Z and world -Z (vac cup approach)."""

    z = np.asarray(grip_pose, dtype=float)[:3, 2]
    return float(np.arccos(np.clip(float(-z[2]), -1.0, 1.0)))


def world_down_vacuum_grip_pose(grip_pose, body_pose=None):
    """Level tool -Z to world -Z; keep TCP or the held body origin fixed.

    Fixed-TCP leveling swings a thick attached payload (live bread_b) and can
    drop ``lift_m`` below ``minimum_lift_m``. When ``body_pose`` is given, the
    TCP is translated so ``grip @ T_GB`` keeps the body origin in place while
    only the shared orientation is leveled.
    """

    grip = np.asarray(grip_pose, dtype=float).copy()
    z_new = np.array([0.0, 0.0, -1.0], dtype=float)
    x_ref = grip[:3, 0]
    x_proj = x_ref - z_new * float(x_ref @ z_new)
    if float(np.linalg.norm(x_proj)) < 1e-8:
        x_ref = grip[:3, 1]
        x_proj = x_ref - z_new * float(x_ref @ z_new)
    if float(np.linalg.norm(x_proj)) < 1e-8:
        x_proj = np.array([1.0, 0.0, 0.0], dtype=float)
    x_new = x_proj / np.linalg.norm(x_proj)
    y_new = np.cross(z_new, x_new)
    y_new = y_new / np.linalg.norm(y_new)
    x_new = np.cross(y_new, z_new)
    leveled = np.eye(4, dtype=float)
    leveled[:3, 0] = x_new
    leveled[:3, 1] = y_new
    leveled[:3, 2] = z_new
    if body_pose is None:
        leveled[:3, 3] = grip[:3, 3]
    else:
        body = np.asarray(body_pose, dtype=float)
        t_gb = (inverse(grip) @ body)[:3, 3]
        leveled[:3, 3] = body[:3, 3] - leveled[:3, :3] @ t_gb
    return leveled


def straighten_vacuum_tool_world_down(context, q, opening):
    """Level a tilted vac tool after LIFT so M5 held motions can track.

    Skips when already within ``VACUUM_POST_LIFT_LEVEL_TILT_RAD``. Uses
    kinematic Cartesian playback only: live bread_b showed that a post-LEVEL
    absolute-joint PD settle sagged ~35 mm / 0.083 rad and cut HOLD
    ``lift_m`` from 0.124 to 0.090. Vertical tool (like plate_vac) is what
    M5 can track; do not re-settle into the tilted equilibrium here.
    """

    grip = np.asarray(context.grip_pose(), dtype=float)
    tilt = vacuum_tool_tilt_from_world_down_rad(grip)
    record = {
        'applied': False,
        'tilt_before_rad': tilt,
        'tilt_before_deg': float(np.rad2deg(tilt)),
        'threshold_rad': float(VACUUM_POST_LIFT_LEVEL_TILT_RAD),
    }
    if tilt <= float(VACUUM_POST_LIFT_LEVEL_TILT_RAD) + 1e-12:
        save_json(context.output / 'vacuum_tool_level.json', record)
        return np.asarray(q, dtype=float), record
    body = np.asarray(context.body_pose(), dtype=float)
    initial_z = float(np.asarray(context.initial_body, dtype=float)[2, 3])
    lift_before = float(body[2, 3] - initial_z)
    record.update({
        'lift_before_m': lift_before,
        'body_z_before_m': float(body[2, 3]),
    })
    target = world_down_vacuum_grip_pose(grip, body_pose=body)
    # settle_steps=0: PD after FK reintroduces the transport lag that LEVEL
    # exists to remove (live: tilt 3.9° → 4.8°, body z -35 mm).
    q = move_vacuum_cartesian_kinematic(
        context, target, 'LEVEL', opening, settle_steps=0)
    body_mid = np.asarray(context.body_pose(), dtype=float)
    lift_mid = float(body_mid[2, 3] - initial_z)
    # Cartesian mid-segments can still dip the payload; restore body z.
    restore_m = lift_before - lift_mid
    if restore_m > 1e-4:
        raised = np.asarray(context.grip_pose(), dtype=float).copy()
        raised[2, 3] = float(raised[2, 3]) + restore_m
        q = move_vacuum_cartesian_kinematic(
            context, raised, 'LEVEL', opening, settle_steps=0)
        record['height_restore_m'] = float(restore_m)
    _sync_absolute_joint_goal(context, q)
    body_after = np.asarray(context.body_pose(), dtype=float)
    tilt_after = vacuum_tool_tilt_from_world_down_rad(context.grip_pose())
    lift_after = float(body_after[2, 3] - initial_z)
    record.update({
        'applied': True,
        'tilt_after_rad': tilt_after,
        'tilt_after_deg': float(np.rad2deg(tilt_after)),
        'lift_after_m': lift_after,
        'body_z_after_m': float(body_after[2, 3]),
        'tcp_delta_m': (
            np.asarray(context.grip_pose(), dtype=float)[:3, 3] - grip[:3, 3]
        ).tolist(),
        'mode': 'KINEMATIC_PATH_PRESERVE_BODY_ORIGIN',
        'pd_settle_after_level': False,
    })
    if getattr(context, 'carried_pose', None) is not None:
        context.carried_pose = inverse(context.grip_pose()) @ context.body_pose()
    save_json(context.output / 'vacuum_tool_level.json', record)
    return q, record


def run_kinematic_vacuum_hold(context, q, opening, duration_s):
    """Hold a vac arm pose by re-snapping FK after each control tick.

    Live bread_b: after LEVEL, impedance ``run_timed_hold`` reintroduced the
    ~35 mm / 0.083 rad lag and failed HOLD ``minimum_lift_m``. Advance the
    controller clock, then overwrite arm joints + attach sync so SETTLE/HOLD
    samples keep the leveled lift. Leave ``goal_qpos`` synced for M5.
    """

    q = np.asarray(q, dtype=float).reshape(-1)
    _sync_absolute_joint_goal(context, q)
    start = float(context.data.time)
    rows = []
    splits = context.robot.composite_controller._action_split_indexes
    while float(context.data.time) - start < float(duration_s) - 1e-9:
        action = np.zeros(context.robot.action_dim)
        lo, hi = splits['right']
        action[lo:hi] = q
        lo, hi = splits['right_gripper']
        gripper_cmd = -float(opening)
        if hasattr(context.runtime, 'vac_gripper_action_for_command'):
            gripper_cmd = float(
                context.runtime.vac_gripper_action_for_command(gripper_cmd)
            )
        action[lo:hi] = gripper_cmd
        context.player._advance_controller(action)
        _apply_vac_arm_waypoint(context, q)
        suppress = getattr(context.runtime, 'suppress_native_adhesion_actuators', None)
        if callable(suppress):
            suppress()
        rows.append(context.sample())
    _sync_absolute_joint_goal(context, q)
    return rows, float(context.data.time) - start


def _vacuum_desired_support_clearance_m():
    return (
        VACUUM_SUPPORT_CLEARANCE_PAD_M
        + VACUUM_POST_BREAKAWAY_LIFT_DIP_MARGIN_M
    )


def breakaway_vacuum_from_support(context, q, opening):
    """If vac-held object is below the LIFT dip-safe clearance, lift before LIFT.

    Skips only when AABB clearance already meets pad+dip-margin and mesh
    penetration is within eps. Uses residual, geometry-driven chunks bounded by
    ``MAX_VACUUM_SUPPORT_BREAKAWAY_M``. Does not rewrite free-object initial
    poses or widen collision thresholds.
    """

    desired = _vacuum_desired_support_clearance_m()
    clearance = support_bottom_clearance_m(context)
    mesh_pen = object_support_penetration_m(context)
    record = {
        'clearance_before_m': clearance,
        'mesh_penetration_before_m': mesh_pen,
        'lift_m': 0.0,
        'applied': False,
        'pad_m': VACUUM_SUPPORT_CLEARANCE_PAD_M,
        'lift_dip_margin_m': VACUUM_POST_BREAKAWAY_LIFT_DIP_MARGIN_M,
        'desired_clearance_m': desired,
        'max_breakaway_m': MAX_VACUUM_SUPPORT_BREAKAWAY_M,
        'chunks': [],
        'mode': 'KINEMATIC_PATH',
    }
    total_lift = 0.0
    # Residual loop covers tilt / AABB vs mesh mismatch after one chunk.
    for _ in range(4):
        chunk = vacuum_support_breakaway_lift_m(clearance, mesh_pen)
        if chunk <= 0.0:
            break
        if total_lift + chunk > MAX_VACUUM_SUPPORT_BREAKAWAY_M + 1e-12:
            raise GraspFailure(
                'SUPPORT_BREAKAWAY_EXCEEDS_BOUND: '
                f'need {total_lift + chunk:.6f} m > max '
                f'{MAX_VACUUM_SUPPORT_BREAKAWAY_M:.6f} m'
            )
        grip = np.asarray(context.grip_pose(), dtype=float)
        direction = _breakaway_direction(grip)
        target = grip.copy()
        target[:3, 3] = grip[:3, 3] + direction * chunk
        q = _kinematic_breakaway_to(context, target, opening)
        clearance_after = support_bottom_clearance_m(context)
        mesh_after = object_support_penetration_m(context)
        improved = clearance_after - clearance
        record['chunks'].append({
            'commanded_lift_m': chunk,
            'clearance_before_m': clearance,
            'clearance_after_m': clearance_after,
            'mesh_penetration_before_m': mesh_pen,
            'mesh_penetration_after_m': mesh_after,
            'direction_xyz': direction.tolist(),
        })
        total_lift += chunk
        record['applied'] = True
        record['lift_m'] = total_lift
        record['direction_xyz'] = direction.tolist()
        if improved < 0.25 * chunk and mesh_after > mesh_pen - 0.25 * chunk:
            raise GraspFailure(
                'SUPPORT_BREAKAWAY_STUCK: '
                f'clearance {clearance:.6f}→{clearance_after:.6f} m '
                f'mesh_pen {mesh_pen:.6f}→{mesh_after:.6f} m '
                f'after commanded {chunk:.6f} m'
            )
        clearance = clearance_after
        mesh_pen = mesh_after
        if (
            clearance >= desired - 1e-4
            and mesh_pen <= VACUUM_SUPPORT_CONTACT_EPS_M
        ):
            break
    else:
        clearance = support_bottom_clearance_m(context)
        mesh_pen = object_support_penetration_m(context)

    record['clearance_after_m'] = clearance
    record['mesh_penetration_after_m'] = mesh_pen
    if record['applied'] and (
        clearance < desired - 1e-4
        or mesh_pen > VACUUM_SUPPORT_CONTACT_EPS_M
    ):
        raise GraspFailure(
            'SUPPORT_BREAKAWAY_INSUFFICIENT: '
            f'clearance {clearance:.6f} m mesh_pen {mesh_pen:.6f} m '
            f'after lift {total_lift:.6f} m'
        )
    if record['applied']:
        save_json(context.output / 'vacuum_support_breakaway.json', record)
    return q, record


def apply_free_object_pose(context, T_WB):
    """Write a free-joint object world pose and forward kinematics."""
    T_WB = np.asarray(T_WB, dtype=float)
    if T_WB.shape != (4, 4):
        raise ValueError('object pose must be a 4x4 transform')
    adr = int(context.object_qadr)
    dadr = int(context.object_dadr)
    context.data.qpos[adr:adr + 3] = T_WB[:3, 3]
    quat_xyzw = Rotation.from_matrix(T_WB[:3, :3]).as_quat()
    context.data.qpos[adr + 3:adr + 7] = quat_xyzw[[3, 0, 1, 2]]
    context.data.qvel[dadr:dadr + 6] = 0.0
    context.mj.mj_forward(context.model, context.data)
    return context.body_pose()


def grasp_continuous_relative_pose(grip_pose, object_pose):
    """T_GB such that grip @ T_GB == object (controller GRASP continuity)."""
    return inverse(np.asarray(grip_pose, dtype=float)) @ np.asarray(
        object_pose, dtype=float
    )


def rebind_kinematic_attachment_to_object_pose(context, desired_T_WB):
    """Replace the frozen attach transform so synchronize yields ``desired_T_WB``.

    ``attach_object`` must already have succeeded (contact-distance gate). This
    only rewrites the kinematic relative pose; it does not change weld/physics.
    """
    attachment = context.runtime.attachment
    if attachment is None:
        raise GraspFailure('VACUUM_ATTACHMENT_LOST')
    rel = grasp_continuous_relative_pose(context.grip_pose(), desired_T_WB)
    updated = replace(
        attachment,
        position_in_reference_m=tuple(float(v) for v in rel[:3, 3]),
        rotation_in_reference=tuple(
            tuple(float(v) for v in row) for row in rel[:3, :3]
        ),
    )
    context.runtime._attachment = updated
    context.runtime.synchronize_attached_object()
    return updated, rel


def attach_vacuum(context):
    if context.recipe.ee_id != 'vac':
        raise ValueError('VACUUM_EE_REQUIRED')
    if not context.ready():
        raise GraspFailure('VACUUM_CONTACT_GATE_NOT_PASSED')
    if getattr(context, 'grasp_object_pose', None) is None:
        raise GraspFailure('GRASP_OBJECT_POSE_MISSING')
    if getattr(context, 'grasp_T_GB', None) is None:
        raise GraspFailure('GRASP_RELATIVE_POSE_MISSING')

    command = 2. * context.recipe.suction_command - 1.
    context.runtime.command_gripper(engaged=True, suction=True, command=command)

    pose_before_attach = np.asarray(context.body_pose(), dtype=float).copy()
    grip_before_attach = np.asarray(context.grip_pose(), dtype=float).copy()
    # Attach on the live contact pair (distance gate). Continuity rebind to the
    # GRASP object pose happens after TCP recovery in the catalog runner.
    attachment = context.runtime.attach_object(
        context.object_id,
        attachment_mode='KINEMATIC',
        max_attach_distance_m=.002,
        max_attach_penetration_m=.002,
    )
    # Lock intended target: kinematic weld holds the object; zero adhesion so
    # the cup cannot suction additional free bodies during later tool-use.
    suppress = getattr(context.runtime, 'suppress_native_adhesion_actuators', None)
    if callable(suppress):
        suppress()
    pose_after_attach = np.asarray(context.body_pose(), dtype=float).copy()
    jump = pose_after_attach[:3, 3] - pose_before_attach[:3, 3]
    T_GB = inverse(context.grip_pose()) @ context.body_pose()

    record = {
        'policy': VACUUM_POLICY,
        'time_s': float(context.data.time),
        'attachment': asdict(attachment),
        'contact_before_attach': context.trace[-1],
        'T_GB_at_attach': T_GB,
        'grasp_T_GB': np.asarray(context.grasp_T_GB, dtype=float),
        'relative_pose_source': 'ACTUAL_CONTACT_POSE_PENDING_GRASP_REBIND',
        'object_pose_before_attach': pose_dict(pose_before_attach),
        'object_pose_after_attach': pose_dict(pose_after_attach),
        'grip_pose_at_attach': pose_dict(grip_before_attach),
        'grasp_object_pose': pose_dict(context.grasp_object_pose),
        'raw_attach_translation_m': jump.tolist(),
        'raw_attach_translation_norm_m': float(np.linalg.norm(jump)),
    }
    context.vacuum_attachment_record = record
    save_json(context.output / 'vacuum_attachment.json', record)
    return attachment


def finalize_vacuum_grasp_continuity(context):
    """Recover GRASP TCP tracking, then freeze attach to the GRASP object pose."""
    if context.runtime.attachment is None:
        raise GraspFailure('VACUUM_ATTACHMENT_LOST')
    pose_before = np.asarray(context.body_pose(), dtype=float).copy()
    grip_before = np.asarray(context.grip_pose(), dtype=float).copy()
    attachment, rel = rebind_kinematic_attachment_to_object_pose(
        context, context.grasp_object_pose
    )
    pose_after = np.asarray(context.body_pose(), dtype=float).copy()
    jump = pose_after[:3, 3] - pose_before[:3, 3]
    record = dict(context.vacuum_attachment_record or {})
    record.update({
        'attachment': asdict(attachment),
        'T_GB_at_attach': rel,
        'relative_pose_source': GRASP_RELATIVE_POSE_SOURCE,
        'grip_pose_before_rebind': pose_dict(grip_before),
        'object_pose_before_rebind': pose_dict(pose_before),
        'object_pose_after_grasp_rebind': pose_dict(pose_after),
        'attach_rebind_translation_m': jump.tolist(),
        'attach_rebind_translation_norm_m': float(np.linalg.norm(jump)),
        'time_s_rebind': float(context.data.time),
    })
    context.vacuum_attachment_record = record
    save_json(context.output / 'vacuum_attachment.json', record)
    return attachment


def release_vacuum(context):
    if context.recipe.ee_id != 'vac':
        raise ValueError('VACUUM_EE_REQUIRED')
    attached = context.runtime.attachment is not None
    if attached:
        context.runtime.detach_object(context.object_id)
    context.runtime.command_gripper(engaged=False, suction=True, command=-1.)
    # Public release takes effect even when no subsequent arm step is requested.
    ids = [context.model.actuator(n).id for n in context.gripper.actuators]
    context.data.ctrl[ids] = 0.
    context.carried_pose = None
    context.stage = 'RELEASE'
    context.mj.mj_forward(context.model, context.data)
    return {
        'detached': attached,
        'attachment_active': False,
        'suction_ctrl': context.data.ctrl[ids].tolist(),
        'time_s': float(context.data.time),
    }
