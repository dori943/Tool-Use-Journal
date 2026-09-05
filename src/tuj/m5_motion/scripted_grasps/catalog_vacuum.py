"""Contact-gated vacuum attachment using the existing M5 runtime."""
from dataclasses import asdict
from tuj.m5_motion.scripted_grasps.runtime import GraspFailure,save_json

VACUUM_POLICY = 'CONTACT_GATED_KINEMATIC_ATTACH'


def attach_vacuum(context):
    if context.recipe.ee_id != 'vac':
        raise ValueError('VACUUM_EE_REQUIRED')
    if not context.ready():
        raise GraspFailure('VACUUM_CONTACT_GATE_NOT_PASSED')
    command = 2. * context.recipe.suction_command - 1.
    context.runtime.command_gripper(engaged=True, suction=True, command=command)
    attachment = context.runtime.attach_object(context.object_id,
        attachment_mode='KINEMATIC', max_attach_distance_m=.002,
        max_attach_penetration_m=.002)
    record = {'policy':VACUUM_POLICY, 'time_s':float(context.data.time),
        'attachment':asdict(attachment), 'contact_before_attach':context.trace[-1],
        'T_GB_at_attach':context.trace[-1]['T_GB'],
        'relative_pose_source':'ACTUAL_CONTACT_POSE'}
    context.vacuum_attachment_record = record
    save_json(context.output/'vacuum_attachment.json', record)
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
    return {'detached':attached, 'attachment_active':False,
            'suction_ctrl':context.data.ctrl[ids].tolist(), 'time_s':float(context.data.time)}
