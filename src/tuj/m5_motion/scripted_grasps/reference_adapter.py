"""Object-relative scripted targets with the successful reference controller.

The existing M5 example provides motion compilation and contact feedback. No
M4 artifact, learned keyframe generation, or object-material force calculation
is invoked by this adapter.
"""
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation
from tuj.m5_motion.scripted_grasps.settings import REPOSITORY
from tuj.m5_motion.scripted_grasps.frames import transform, inverse
from tuj.m5_motion.scripted_grasps.objects.plate import build_calibrated_plate_targets
from tuj.m5_motion.scripted_grasps.ik_continuity import PlanningAdapter


def bind_object_keyframe(context, world, body, target, stage):
    """Bind a complete world goal to an object anchor without losing roll."""
    from tuj.m5_motion.geometry import RelativePoseResolver
    key,_=context.target_keyframe(stage,target)
    anchor='scripted_'+stage.lower()
    world.objects['plate']['anchors'][anchor]=(inverse(body)@target)[:3,3].tolist()
    key=key.model_copy(update={'frame_ref':'object:plate','anchor':anchor,
        'approach_axis_xyz':tuple(body[:3,:3].T@target[:3,2]),
        'events_after':[], 'metadata':{'event_target_id':'plate'}})
    resolved=RelativePoseResolver(world).resolve(key)
    if not np.allclose(transform(resolved.position_m,resolved.orientation_xyzw),target,atol=1e-7):
        raise ValueError('REFERENCE_POSE_ADAPTER_MISMATCH')
    return key


def execute(context, recipe):
    from tuj.m5_motion.scripted_grasps.runtime import save_json, GraspFailure
    from tuj.m5_motion.examples import c1_1_openai_motion_run as ref
    from tuj.m5_motion.schema import (MotionPlanRequest, ArtifactProvenance,
        MotionTask, MotionGoal, GraspSpec, KeyframePlanArtifact,
        KeyframePlanCandidate, StrategyGenerationProvenance)
    from tuj.m5_motion.tool_use_journal_runtime import (ToolUseJournalEERuntime,
        ToolUseJournalControllerTrajectoryPlayer)
    from tuj.m5_motion.tool_use_journal import ToolUseJournalCollisionModelCompiler
    from tuj.m5_motion.tool_use_journal_planning import ToolUseJournalCollisionContextFactory

    c=context
    c.recipe=recipe
    if type(c.gripper).__name__ != recipe.model_class:
        raise GraspFailure('UNSUPPORTED_EE_MODEL')
    if not np.allclose(c.collision_local_size, recipe.asset_dimensions_m, atol=1e-5, rtol=0):
        raise GraspFailure('ASSET_SIZE_REQUIRES_SEPARATE_CALIBRATION')
    runtime=c.runtime
    runtime.set_finger_gripper_contact_friction(c.ee_spec['fingerpad_friction'])
    c.record_input()
    c.stage='PRESHAPE'
    aperture=ToolUseJournalControllerTrajectoryPlayer(runtime).preshape_finger_gripper_to_aperture(
        target_aperture_m=float(recipe.calibrated_local_bbox_size_m[2]+recipe.preshape_clearance_m),
        tolerance_m=.002,final_settle_ticks=25,allow_wider_discrete_state=True)
    print(f'[reference preshape] pad separation={aperture:.4f}m',flush=True)
    world=c.adapter.world_snapshot()
    pose=world.objects['plate']['pose']
    body=transform(pose['position_m'],pose['orientation_xyzw'])
    targets=build_calibrated_plate_targets(body,c.center_in_body,c.observed_size,recipe)
    targets['calibration']['observation_source']='FROZEN_REFERENCE_BBOX_WITH_CURRENT_FULL_ASSET_SIZE_CHECK'
    targets['calibration']['current_observed_local_bbox_size_m']=None
    save_json(c.output/'targets.json',targets)
    # Exclude material metadata from the motion request. Geometry and robot
    # state are sufficient for deterministic target generation and planning.
    world.metadata={k:v for k,v in world.metadata.items() if k not in {'tool_physical_metadata'}}
    for record in world.objects.values():
        record.pop('physical_metadata', None)
    task=MotionTask(task_id='scripted:plate',subgoal_id='scripted:plate:pick',
        action_type='PICK',ee='2F',tool='plate',target_ids=['plate'],
        grasp=GraspSpec(grasp_id=recipe.recipe_id,owner_kind='tool',owner_id='plate',source='scripted_plate'),
        goal=MotionGoal(goal_type='POSE',target_object_id='plate',
            approach_distance_m=recipe.approach_distance_m,retreat_distance_m=recipe.lift_distance_m),
        allowed_touch_objects=['plate'],
        metadata={'operation':'PICK_TOOL','attach_target':True,'grasp_execution_mode':'CONTACT_FRICTION'})
    provenance=ArtifactProvenance(artifact_id='scripted:plate:request',artifact_type='MotionPlanRequest',
        produced_by='GRASP_PLANNER',invocation_id=c.output.name,
        metadata={'generator':'OBJECT_CENTER_SCRIPT','learned_model_calls':0,
            'joint_angle_policy':'NEAREST_EQUIVALENT_WITHIN_UR5E_LIMITS',
            'pose_policy':'FIXED_OBJECT_CENTER_POSE'})
    request=MotionPlanRequest(request_id='scripted:plate',provenance=provenance,world=world,task=task,
        constraints=ref._constraints(list(world.robot_state.joint_names)),
        options=ref._options(c.seed,allowed_planning_time_s=12.,rrt_max_iterations=5000))
    request.constraints.allowed_collision_pairs=[('2F','table*'),('plate','table*')]
    keys=[]
    for stage in ['PRE_GRASP','GRASP','LIFT']:
        keys.append(bind_object_keyframe(c,request.world,body,targets[stage],stage))
    digest=hashlib.sha256(json.dumps({'recipe':recipe.to_dict(),'body_pose':body.tolist(),
        'calibrated_T_CG':targets['T_CG'].tolist(),
        'local_bbox':c.observed_size.tolist()},sort_keys=True).encode()).hexdigest()
    artifact=KeyframePlanArtifact(artifact_id='scripted:plate:'+digest[:16],
        provenance=provenance.model_copy(update={'artifact_type':'KeyframePlanArtifact',
            'artifact_id':'scripted:plate:keyframes'}),scene_signature=world.scene.signature,
        subgoal_id=task.subgoal_id,candidates=[KeyframePlanCandidate(strategy_id=recipe.recipe_id,
            keyframes=keys,rationale='Fixed plate rim recipe transformed by the current object pose.',
            provenance=StrategyGenerationProvenance(generator_kind='TEMPLATE',
                generator_id='SCRIPTED_PLATE_CENTER_FRAME',input_hash=digest))])
    artifact=ref._with_contact_friction_grasp(artifact,hold_duration_s=recipe.close_hold_s,
        lift_hold_duration_s=recipe.post_lift_settle_s+recipe.final_hold_s,gripper_close_rate=1.)
    ref._write_model(c.output/'motion_request.json',request)
    ref._write_model(c.output/'scripted_keyframes.json',artifact)
    compiler=ToolUseJournalCollisionModelCompiler.from_repository(c.env,REPOSITORY,seed=c.seed,
        ignore_done=True,use_camera_obs=False,has_offscreen_renderer=False)
    kind,name,_,_=runtime._grasp_reference(c.env)
    factory=ToolUseJournalCollisionContextFactory(compiler,attachment_reference_name=name,
        attachment_reference_kind=kind)
    c.stage='PLAN'
    print('[reference planner] compiling current object-relative targets',flush=True)
    planning,registry=ref._plan(request,artifact,
        PlanningAdapter(c.adapter,world.robot_state.joint_positions_rad),factory)
    plan=planning.plan
    ref._write_model(c.output/'motion_plan.json',plan)
    from tuj.m5_motion.scripted_grasps.plan_validation import audit_endpoints
    pose_errors = audit_endpoints(c, plan, targets)
    save_json(c.output/'plan_pose_errors.json', pose_errors)
    if ({row['keyframe'] for row in pose_errors} != {'PRE_GRASP','GRASP','LIFT'} or
            any(row['position_error_m'] > .0001 or row['orientation_error_deg'] > .05 for row in pose_errors)):
        raise GraspFailure('PLANNED_EE_POSE_PRECISION_FAILED')
    print(f'[reference plan] {len(plan.segments)} segments, {plan.duration_s:.2f}s',flush=True)
    profile=ref._load_motion_profile(REPOSITORY/'src/tuj/m5_motion/examples/c1_1_physical_grasp_profile.json')
    monitor=ref._PhysicalGraspMonitor(runtime=runtime,object_id='plate',
        initial_position_m=tuple(c.initial_body[:3,3]),min_lift_m=recipe.minimum_lift_m,
        object_dimensions_m=tuple(c.collision_local_size),table_surface_z_m=c.initial_bottom,
        min_bottom_clearance_m=.01,required_final_hold_s=recipe.final_hold_s,
        contact_loss_grace_s=.06,required_contact_ticks=5,contact_freeze_ticks=2,
        closure_actuator_kp=recipe.closure_kp,max_closure_actuator_kp=recipe.maximum_closure_kp,
        force_feedback_gain=recipe.force_feedback_gain,
        retention_target_normal_force_n=recipe.contact_force_target_n,
        max_grip_force_n=float(c.ee_spec['grip_force_n']),sliding_friction=float(c.ee_spec['fingerpad_friction'][0]),
        min_contact_separation_m=.003,min_normal_opposition=.5,max_friction_utilization=.95,
        contact_follow_gain=profile.pick_contact_follow_gain,
        contact_follow_max_m=profile.pick_contact_follow_max_m,
        contact_follow_activation_ticks=profile.pick_contact_follow_activation_ticks,
        contact_follow_max_tick_m=profile.pick_contact_follow_max_tick_m,
        contact_follow_max_joint_step_rad=profile.pick_contact_follow_max_joint_step_rad,
        regrasp_roll_rad=profile.pick_regrasp_roll_rad,
        regrasp_roll_rate_rad_s=profile.pick_regrasp_roll_rate_rad_s,
        regrasp_min_separation_ratio=profile.pick_regrasp_min_separation_ratio)
    c.physical_monitor=monitor
    original_sample=monitor.sample
    c.original_monitor_sample=original_sample
    def sample(time_s):
        original_sample(time_s)
        stage=monitor.active_keyframe_id or 'PRE_GRASP'
        c.stage=stage
        sides,bad=c.contacts()
        # The native nominal-path probe can run less often because this
        # actual contact check still runs on every 50 Hz controller tick.
        for con in c.data.contact[:c.data.ncon]:
            a,b=int(con.geom1),int(con.geom2)
            if con.dist>=-.001 or not ((a in c.plate_geoms) ^ (b in c.plate_geoms)):
                continue
            other=b if a in c.plate_geoms else a
            if other in c.gripper_geoms or c.model.geom(other).name=='table_collision':
                continue
            bad.append({'geoms':[c.model.geom(a).name,c.model.geom(b).name],
                'penetration_m':-float(con.dist)})
        row={'time_s':float(time_s),'stage':stage,'q':c.data.qpos[c.arm_ids].tolist(),
            'finger_contacts':sides,'normal_contact_force_n':c.contact_force(),
            'gripper_kp':float(c.model.actuator_gainprm[c.gripper_actuator_ids[0],0]),
            'gripper_actuator_targets':c.data.ctrl[c.gripper_actuator_ids].tolist(),
            'lift_m':float(c.body_pose()[2,3]-c.initial_body[2,3]),
            'bottom_clearance_m':c.bottom_height()-c.initial_bottom,
            'T_GB':(inverse(c.grip_pose())@c.body_pose()).tolist(),'bad_contacts':bad,
            'contact_gate':getattr(c,'contact_gate_state',None),
            'finger_table_penetration_m':max([0.]+[-float(con.dist) for con in c.data.contact[:c.data.ncon]
                if c.finger_table_contact(int(con.geom1),int(con.geom2))])}
        c.trace.append(row)
        c.tick+=1
        if c.tick%250==0:
            print(f"[physical] {stage}: lift={row['lift_m']:.3f}m, contacts={sides}",flush=True)
        if c.video and c.tick%5==0:
            if c.writer is None:
                import imageio.v2 as imageio
                c.writer=imageio.get_writer(str(c.output/'execution.mp4'),fps=10,codec='libx264',quality=7)
            c.writer.append_data(c.env.sim.render(width=640,height=480,camera_name='agentview')[::-1])
        if bad:
            raise GraspFailure('UNEXPECTED_COLLISION: '+str(bad[:2]))
    monitor.sample=sample
    try:
        from tuj.m5_motion.scripted_grasps.contact_gate import run
        _,report=run(c,runtime,plan,registry,'scripted:plate:'+c.output.name,monitor,targets['T_WC'])
        ref._write_model(c.output/'execution_report.json',report)
    finally:
        save_json(c.output/'physical_grasp_trace.json',monitor.samples)
    validation=monitor.summary()
    save_json(c.output/'physical_grasp_validation.json',validation)
    if str(report.status.value)!='SUCCESS':
        reason=report.failure.message if report.failure is not None else str(report.status.value)
        if 'PRE_LIFT_CONTACT_NOT_STABLE' in reason:
            c.stage='PRE_LIFT_CHECK'
        raise GraspFailure(reason)
    if not c.trace:
        raise GraspFailure('NO_CONTROLLER_SAMPLES')
    end=c.trace[-1]['time_s']
    hold=[s for s in c.trace if s['time_s']>end-recipe.final_hold_s+1e-6]
    ref_pose=np.asarray(hold[0]['T_GB'])
    slip=max(np.linalg.norm(np.asarray(s['T_GB'])[:3,3]-ref_pose[:3,3]) for s in hold)
    rotation=max(Rotation.from_matrix(ref_pose[:3,:3].T@np.asarray(s['T_GB'])[:3,:3]).magnitude() for s in hold)
    metrics={'minimum_hold_lift_m':min(s['lift_m'] for s in hold),
        'minimum_bottom_clearance_m':min(s['bottom_clearance_m'] for s in hold),
        'bilateral_contact_fraction':sum(len(s['finger_contacts'])==2 for s in hold)/len(hold),
        'bilateral_contact_fraction_during_lift_and_hold':validation['bilateral_contact_ratio_after_formation'],
        'max_slip_m':float(slip),'max_slip_deg':float(np.rad2deg(rotation)),'hold_s':recipe.final_hold_s,
        'maximum_gripper_table_penetration_m':max(s['finger_table_penetration_m'] for s in c.trace),
        'mean_hold_normal_force_n':float(np.mean([s['normal_contact_force_n'] for s in hold]))}
    success=(validation['status']=='SUCCESS' and str(report.status.value)=='SUCCESS' and
        metrics['minimum_hold_lift_m']>=recipe.minimum_lift_m and metrics['minimum_bottom_clearance_m']>=.01 and
        metrics['bilateral_contact_fraction']>=.95 and slip<=recipe.max_slip_m and
        np.rad2deg(rotation)<=recipe.max_slip_deg and runtime.attachment is None)
    return {'status':'SUCCESS' if success else 'FAILED','recipe':recipe.to_dict(),'scenario':c.scenario,
        'metrics':metrics,'failure_reason':None if success else 'REFERENCE_HOLD_VALIDATION_FAILED',
        'controller':'OBJECT_FRAME_ROBOT_RESPONSE_WITH_CONTACT_GATE','preshape_pad_separation_m':aperture,
        'joint_angle_policy':'NEAREST_EQUIVALENT_WITHIN_UR5E_LIMITS',
        'pose_policy':'FIXED_OBJECT_CENTER_POSE'}
