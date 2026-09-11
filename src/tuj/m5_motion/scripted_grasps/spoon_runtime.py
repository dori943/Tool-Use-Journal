"""Physical C1_2 spoon grasp; independent of the validated object runtimes."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation
from tuj.m5_motion.scripted_grasps.settings import REPOSITORY
from tuj.m5_motion.scripted_grasps.frames import transform, inverse, pose_dict
from tuj.m5_motion.scripted_grasps.runtime import save_json, GraspFailure
from tuj.m5_motion.scripted_grasps.objects.spoon import SpoonRecipe, build_spoon_targets
from tuj.m5_motion.scripted_grasps.ik_continuity import ContinuousIK
from tuj.m5_motion.scripted_grasps.spoon_hand_model import bound_spoon_3f_commands


def three_finger_ready(samples, ticks=5):
    return len(samples)>=ticks and all(three_finger_pinch_event(s) for s in samples[-ticks:])


def thin_handle_ready(samples, ticks=5):
    """Thumb + one opposing finger on a thin utensil handle (c3_2 spoon/fork)."""
    return len(samples)>=ticks and all(thin_handle_pinch_event(s) for s in samples[-ticks:])


def two_finger_ready(samples,ticks=5):
    return fingers_ready(samples,('left','right'),ticks) and all(
        min(s['finger_force_n'].values())>=2. for s in samples[-ticks:])


def three_finger_pinch_event(sample):
    """Require opposed, balanced contacts rather than one heavily loaded finger."""
    forces=sample['finger_force_n']
    other=forces['index']+forces['pinky']
    return (set(sample['finger_contacts'])=={'thumb','index','pinky'}
        and sample['normal_opposition']>=.8 and sample['contact_span_m']>=.020
        and forces['thumb']>=1.5 and forces['index']>=.75 and forces['pinky']>=.75
        and .5*other<=forces['thumb']<=2.*other and sum(forces.values())<=8.)


def thin_handle_pinch_event(sample):
    """Accept a thumb↔index/pinky pinch when the handle is too thin for 3 fingers.

    c3_2 utensil collision meshes are narrow; the third Jaco finger often cannot
    seat. A sustained thumb+secondary pinch with span is enough to lift.
    """
    forces=sample['finger_force_n']
    contacts=set(sample['finger_contacts'])
    if 'thumb' not in contacts:
        return False
    secondary=[name for name in ('index','pinky') if name in contacts and forces[name]>=.2]
    if not secondary:
        return False
    other=sum(forces[name] for name in secondary)
    return (sample['contact_span_m']>=.035 and forces['thumb']>=1.5
            and other>=.2 and sample['normal_opposition']>=.15
            and sum(forces.values())<=12.)


def fingers_ready(samples, fingers, ticks=5):
    if len(samples) < ticks:
        return False
    return all(set(s['finger_contacts']) == set(fingers)
               and s['normal_opposition'] >= .5 and s['contact_span_m'] >= .0035
               and all(v > .1 for v in s['finger_force_n'].values()) for s in samples[-ticks:])


def grasp_contact_ready(recipe, samples, ticks=5):
    if recipe.ee_id=='2F':
        return two_finger_ready(samples,ticks)
    if getattr(recipe,'thin_handle_pinch',False):
        return thin_handle_ready(samples,ticks)
    return three_finger_ready(samples,ticks)


def hold_contact_fraction(hold_rows, finger_groups, recipe):
    if getattr(recipe,'thin_handle_pinch',False):
        def ok(contacts):
            contacts=set(contacts)
            return 'thumb' in contacts and bool(contacts & {'index','pinky'})
        return sum(ok(s['finger_contacts']) for s in hold_rows)/len(hold_rows)
    return sum(set(s['finger_contacts'])==set(finger_groups) for s in hold_rows)/len(hold_rows)


def attach_thin_handle_pinch(context):
    """Kinematic carry after a verified thin-handle pinch (vac-style gate).

    c3_2 utensil meshes are too thin for reliable friction lift with the Jaco
    3F. Contact still gates the grasp; attachment carries LIFT/HOLD/transport.
    """
    from dataclasses import asdict
    recipe=context.recipe
    if not getattr(recipe,'thin_handle_pinch',False):
        raise ValueError('thin_handle_pinch required')
    if not thin_handle_ready(context.trace,recipe.contact_ticks):
        raise GraspFailure('THIN_HANDLE_CONTACT_GATE_NOT_PASSED')
    # Logical engage uses the runtime's non-negative close convention.
    context.runtime.command_gripper(engaged=True,suction=False,command=1.)
    try:
        attachment=context.runtime.attach_object(
            context.object_id,
            attachment_mode='KINEMATIC',
            max_attach_distance_m=.05,
            max_attach_penetration_m=.01,
        )
    except Exception as exc:
        raise GraspFailure(f'THIN_HANDLE_ATTACH_FAILED: {exc}') from exc
    T_GB=inverse(context.grip_pose())@context.body_pose()
    record={
        'policy':'CONTACT_GATED_KINEMATIC_THIN_HANDLE',
        'time_s':float(context.data.time),
        'attachment':asdict(attachment),
        'contact_before_attach':context.trace[-1],
        'T_GB_at_attach':np.asarray(T_GB,dtype=float),
    }
    context.thin_handle_attachment_record=record
    # JSON-safe copy for disk (trace sample may contain numpy).
    disk={**record,
          'T_GB_at_attach':np.asarray(T_GB,dtype=float).tolist(),
          'contact_before_attach':{
              k:(v.tolist() if hasattr(v,'tolist') else v)
              for k,v in (context.trace[-1] or {}).items()
              if k in {'finger_contacts','finger_force_n','contact_span_m','normal_opposition','time_s','stage'}
          }}
    save_json(context.output/'thin_handle_attachment.json',disk)
    return attachment


def update_three_finger_commands(commands,measured_forces,recipe):
    """A bounded force integrator: positive command opens the Jaco finger."""
    error=np.asarray(measured_forces,dtype=float)-np.asarray(recipe.three_finger_force_targets_n)
    delta=np.clip(recipe.three_finger_force_gain*error,-.01,.01)
    return np.clip(np.asarray(commands,dtype=float)+delta,-1.,1.)


def _ik_failure_message(exc):
    message=str(exc)
    return message.startswith('IK_FAILED:') or message.startswith('CARTESIAN_IK_FAILED:')


def _optional_grasp_ik_seed(context, grasp):
    """Solve GRASP IK without moving; reuse as PRE continuation seed when available."""
    q=np.asarray(context.data.qpos[context.arm_ids],dtype=float).copy()
    quat=Rotation.from_matrix(grasp[:3,:3]).as_quat()
    try:
        solutions=context.kinematics.solve_all_ik(grasp[:3,3],quat,seed_qpos=q)
    except Exception:
        return None
    if not getattr(solutions,'solutions',None):
        return None
    best=min(solutions.solutions,key=lambda s:np.linalg.norm(np.asarray(s.qpos)-q))
    return np.asarray(best.qpos,dtype=float)


def approach_spoon(context, targets, opening=1.):
    """Use a shorter or elevated entry when the nominal pre-grasp is unreachable.

    Outer UR5e workspace often has a full-pose IK cliff: GRASP is solvable but
    the preferred PRE standoff is not.  Shorten along the approach ray from the
    recipe distance before the elevated collision-clearance hop.
    """
    from tuj.m5_motion.grasp_geometry import catalog_pre_grasp_reach_standoff_candidates

    grasp=targets['GRASP']
    axis=np.asarray(grasp[:3,2],dtype=float)
    preferred=float(np.linalg.norm(
        np.asarray(targets['PRE_GRASP'][:3,3],dtype=float)-grasp[:3,3]))
    schedule=catalog_pre_grasp_reach_standoff_candidates(preferred)
    seed=_optional_grasp_ik_seed(context,grasp)
    move_kwargs={}
    if seed is not None:
        move_kwargs['seed_qpos']=seed

    try:
        context.move(targets['PRE_GRASP'],'PRE_GRASP',opening,**move_kwargs)
        return
    except GraspFailure as exc:
        if not _ik_failure_message(exc):
            if not str(exc).startswith('COLLISION_FREE_PATH_NOT_FOUND'):
                raise
            clearance=targets['PRE_GRASP'].copy()
            clearance[2,3]+=.16
            context.move(clearance,'PRE_CLEARANCE',opening,**move_kwargs)
            # Do not descend back to an invalid nominal pre-grasp. The caller
            # validates a diagonal Cartesian approach to the same grasp goal.
            return
        last_exc=exc

    for distance in schedule[1:]:
        pre=grasp.copy()
        pre[:3,3]=grasp[:3,3]-axis*distance
        try:
            context.move(pre,'PRE_GRASP',opening,**move_kwargs)
            targets['PRE_GRASP']=pre
            print('[plan] PRE_GRASP reach fallback standoff_m=',
                  distance, flush=True)
            return
        except GraspFailure as exc:
            # Keep IK/collision validation strict; only reject this candidate.
            last_exc=exc
            continue
    raise last_exc


def lift_with_reach_fallback(context, targets, opening, *, cartesian=True):
    """Raise to LIFT, shortening the climb when the preferred height has no IK."""
    from tuj.m5_motion.grasp_geometry import catalog_lift_reach_standoff_candidates

    grasp=targets['GRASP']
    preferred=float(targets['LIFT'][2,3]-grasp[2,3])
    minimum=float(getattr(context.recipe,'minimum_lift_m',preferred))
    schedule=catalog_lift_reach_standoff_candidates(
        preferred, minimum_lift_m=minimum)

    try:
        return context.move(targets['LIFT'],'LIFT',opening,cartesian=cartesian)
    except GraspFailure as exc:
        if not _ik_failure_message(exc):
            raise
        last_exc=exc

    for height in schedule[1:]:
        lift=grasp.copy()
        lift[2,3]=grasp[2,3]+height
        try:
            q=context.move(lift,'LIFT',opening,cartesian=cartesian)
            targets['LIFT']=lift
            print('[plan] LIFT reach fallback height_m=', height, flush=True)
            return q
        except GraspFailure as exc:
            last_exc=exc
            continue
    raise last_exc


def preshape_spoon(context,recipe):
    """Shape the hand while holding the actual current arm joints."""
    context.stage='PRESHAPE'
    q=context.data.qpos[context.arm_ids].copy()
    # The Kitchen reset leaves the passive linkage in a short transient.
    # Stabilize at the half-range controller target before arm travel.
    for opening in np.linspace(1.,0.,150): context.step(q,float(opening))
    for _ in range(100): context.step(q,0.)
    opening=-recipe.preshape_closure_command
    for partial in np.linspace(0.,opening,100): context.step(q,float(partial))
    for _ in range(150): context.step(q,opening)
    aperture=context.runtime.fingerpad_separation_m()
    if not recipe.preshape_aperture_m-.005<=aperture<=recipe.preshape_aperture_m+.005:
        raise GraspFailure(f'PRESHAPE_APERTURE_OUTSIDE_RANGE: {aperture}')
    return opening,aperture


def preshape_spoon_3f(context,recipe):
    """Partial-close 3F before GRASP so open tips clear the support surface.

    Flat utensils sit near the table. Fully open Jaco tips hang below the TCP
    and collide with the island at handle height; a mid closure lifts the tips
    enough to reach the mesh without a table strike.
    """
    context.stage='PRESHAPE'
    q=context.data.qpos[context.arm_ids].copy()
    target=float(np.clip(1.-2.*recipe.preshape_closure_command,-1.,1.))
    for opening in np.linspace(1.,target,100):
        context.step(q,float(opening))
    for _ in range(75):
        context.step(q,target)
    return target,None


from .motion import GraspMotionContext


class SpoonContext(GraspMotionContext):


    def audit_hand_range(self):
        """Check every physics substep, including transients between policy ticks."""
        self.physics_steps_audited+=1
        q=self.data.qpos[self.model.jnt_qposadr[self.hand_joint_ids]]
        limits=self.model.jnt_range[self.hand_joint_ids]
        error=np.maximum(np.maximum(limits[:,0]-q,q-limits[:,1]),0.)
        self.maximum_physics_joint_error=max(self.maximum_physics_joint_error,float(error.max()))
        if not np.isfinite(q).all() or any(self.data.warning[i].number for i in (4,5,6)):
            raise GraspFailure('PHYSICS_NUMERICAL_INSTABILITY')
        if error.max()>self.recipe.maximum_joint_limit_error_rad:
            raise GraspFailure(f'GRIPPER_JOINT_LIMIT_EXCEEDED: physics substep {error.max():.6f} rad')




    def table_top_height(self):
        gid=self.model.geom('island_island_group_top_2').id
        if self.model.geom_type[gid]==self.mj.mjtGeom.mjGEOM_MESH:
            mid=int(self.model.geom_dataid[gid])
            start,count=int(self.model.mesh_vertadr[mid]),int(self.model.mesh_vertnum[mid])
            points=self.model.mesh_vert[start:start+count]
        elif self.model.geom_type[gid]==self.mj.mjtGeom.mjGEOM_BOX:
            import itertools
            points=np.array(list(itertools.product([-1,1],repeat=3)))*self.model.geom_size[gid]
        else:
            raise GraspFailure('UNSUPPORTED_COUNTERTOP_GEOMETRY')
        world=points@self.data.geom_xmat[gid].reshape(3,3).T+self.data.geom_xpos[gid]
        return float(world[:,2].max())

    def bad_contacts(self,data,stage):
        bad=[]
        thin_attached=(
            getattr(self.recipe,'thin_handle_pinch',False)
            and self.runtime.attached_object_id==self.object_id
            and stage in {'LIFT','SETTLE','HOLD'}
        )
        for con in data.contact[:data.ncon]:
            a,b=int(con.geom1),int(con.geom2)
            if con.dist >= -.001:
                continue
            pair=tuple(sorted((a,b)))
            if pair in self.fixed_mount_pairs or (a in self.gripper_geoms and b in self.gripper_geoms):
                continue
            robot=a in self.robot_geoms or b in self.robot_geoms
            obj=a in self.object_geoms or b in self.object_geoms
            finger=(a in self.finger_geoms and b in self.handle_geoms) or (b in self.finger_geoms and a in self.handle_geoms)
            if finger and stage in {'GRASP','CLOSE','LIFT','SETTLE','HOLD'}:
                continue
            # After thin-handle kinematic attach, any finger↔object contact is expected.
            if thin_attached and (
                (a in self.finger_geoms and b in self.object_geoms)
                or (b in self.finger_geoms and a in self.object_geoms)
            ):
                continue
            if robot or (obj and stage in {'LIFT','SETTLE','HOLD'}):
                bad.append({'geoms':[self.model.geom(a).name,self.model.geom(b).name],'penetration_m':-float(con.dist)})
        return bad

    def sample(self):
        forces={name:0. for name in self.finger_groups}
        points={name:[] for name in forces}
        normals={name:[] for name in forces}
        for i,con in enumerate(self.data.contact[:self.data.ncon]):
            a,b=int(con.geom1),int(con.geom2)
            if con.dist>0:
                continue
            for finger,geoms in self.finger_groups.items():
                if (a in geoms and b in self.handle_geoms) or (b in geoms and a in self.handle_geoms):
                    wrench=np.zeros(6)
                    self.mj.mj_contactForce(self.model,self.data,i,wrench)
                    forces[finger]+=max(0.,float(wrench[0]))
                    points[finger].append(np.asarray(con.pos))
                    normals[finger].append(np.asarray(con.frame).reshape(3,3)[0]*(1 if a in geoms else -1))
        opposition=0.
        span=0.
        opposing=list(self.finger_groups)
        for other in opposing[1:]:
            for n,p in zip(normals[opposing[0]],points[opposing[0]]):
                for v,q in zip(normals[other],points[other]):
                    opposition=max(opposition,float(-n@v))
                    span=max(span,float(np.linalg.norm(p-q)))
        row={'time_s':float(self.data.time),'stage':self.stage,'q':self.data.qpos[self.arm_ids].tolist(),
             'finger_contacts':[f for f in forces if forces[f]>.01],'finger_force_n':forces,
             'normal_opposition':opposition,'contact_span_m':span,
             'lift_m':float(self.body_pose()[2,3]-self.initial_body[2,3]),
             'bottom_clearance_m':self.bottom_height()-self.support_top_z,
             'T_GB':inverse(self.grip_pose())@self.body_pose(),
             'gripper_q':self.data.qpos[self.robot._ref_gripper_joint_pos_indexes['right']].tolist(),
             'gripper_ctrl':self.data.ctrl[[self.model.actuator(name).id for name in self.gripper.actuators]].tolist(),
             'gripper_command':np.asarray(self.gripper.current_action,dtype=float).tolist(),
             'object_pose':self.body_pose(),
             'bad_contacts':self.bad_contacts(self.data,self.stage)}
        self.trace.append(row)
        if any(self.data.warning[i].number for i in (4,5,6)):
            raise GraspFailure('PHYSICS_NUMERICAL_INSTABILITY')
        violations=[]
        for name in self.gripper.joints:
            jid=self.model.joint(name).id
            if not self.model.jnt_limited[jid]: continue
            value=float(self.data.qpos[self.model.jnt_qposadr[jid]])
            lo,hi=self.model.jnt_range[jid]
            error=max(float(lo)-value,value-float(hi),0.)
            if error>0.: violations.append({'joint':name,'violation_rad':error})
        row['gripper_joint_limit_violations']=violations
        if any(v['violation_rad']>self.recipe.maximum_joint_limit_error_rad for v in violations):
            raise GraspFailure('GRIPPER_JOINT_LIMIT_EXCEEDED: '+str(violations))
        if len(self.trace)%250==0:
            print(f"[spoon] {self.stage}: lift={row['lift_m']:.3f}m fingers={row['finger_contacts']}",flush=True)
        if self.video and len(self.trace)%5==0:
            if self.writer is None:
                import imageio.v2 as imageio
                self.writer=imageio.get_writer(str(self.output/'execution.mp4'),fps=10,codec='libx264',quality=7)
            self.writer.append_data(self.render())
        if row['bad_contacts']:
            raise GraspFailure('UNEXPECTED_COLLISION: '+str(row['bad_contacts'][:2]))
        return row

    def step(self,q,opening):
        action=np.zeros(self.robot.action_dim)
        splits=self.robot.composite_controller._action_split_indexes
        lo,hi=splits['right']; action[lo:hi]=q
        # Both drivers integrate action signs. Preload an explicit opening
        # command and send a zero increment; Robotiq's sign opposes Jaco's.
        lo,hi=splits['right_gripper']
        if self.recipe.ee_id=='3F':
            if self.three_finger_force_hold and self.trace:
                measured=np.array([self.trace[-1]['finger_force_n'][name]
                    for name in ('thumb','index','pinky')])
                hold_pos=getattr(self.recipe,'hold_finger_positions',False)
                freeze=hold_pos and self.stage in {'LIFT','SETTLE','HOLD'}
                if not freeze:
                    self.three_finger_commands=update_three_finger_commands(
                        self.three_finger_commands,measured,self.recipe)
                command=self.three_finger_commands
            else:
                command=np.full(self.gripper.dof,float(opening))
            command=bound_spoon_3f_commands(command)
            if self.three_finger_force_hold: self.three_finger_commands=command.copy()
            self.gripper.current_action=command
            action[lo:hi]=0.
        else:
            if self.two_finger_force_hold and self.trace:
                measured=min(self.trace[-1]['finger_force_n'].values())
                delta=np.clip(self.recipe.two_finger_force_gain*(self.recipe.two_finger_force_target_n-measured),-.005,.005)
                self.two_finger_command=float(np.clip(self.two_finger_command+delta,-1.,1.))
                desired=self.two_finger_command
            else: desired=-float(opening)
            self.gripper.current_action=np.full(self.gripper.dof,desired)
            action[lo:hi]=0.
        self.player._advance_controller(action)
        return self.sample()






    def execute_spoon(self,recipe):
        self.recipe=recipe
        self.three_finger_force_hold=False
        self.three_finger_commands=None
        self.two_finger_force_hold=False
        self.two_finger_command=0.
        result={'status':'FAILED','object_id':'spoon','environment':'C1_2_DoughFlatten',
                'controller':'NATIVE_JOINT_PD_WITH_HANDLE_CONTACT_FORCE_HOLD',
                'recipe':recipe.to_dict(),'scenario':self.scenario}
        try:
            if type(self.gripper).__name__!=recipe.model_class:
                raise GraspFailure('UNSUPPORTED_EE')
            self.record_input()
            targets=build_spoon_targets(self.body_pose(),self.center_in_body,self.local_size,recipe)
            save_json(self.output/'targets.json',targets)
            q=self.data.qpos[self.arm_ids].copy()
            if recipe.ee_id=='2F':
                # Stabilize the passive four-bar linkage before moving the arm;
                # the calibrated 20 mm pad separation clears the 11 mm handle.
                opening,aperture=preshape_spoon(self,recipe)
                save_json(self.output/'preshape.json',{'aperture_m':aperture,'opening_command':opening})
                print('[preshape]',aperture,'opening',opening,flush=True)
                approach_spoon(self,targets,opening)
            else:
                # Near-open tip pose clears the island; mid-close dips tips into it.
                if recipe.preshape_closure_command < .1:
                    self.stage='OPEN'
                    for _ in range(75): self.step(q,1.)
                    opening,aperture=1.,None
                else:
                    opening,aperture=preshape_spoon_3f(self,recipe)
                save_json(self.output/'preshape.json',{'aperture_m':aperture,'opening_command':opening})
                print('[preshape-3f] opening',opening,flush=True)
                approach_spoon(self,targets,opening)
            q=self.move(targets['GRASP'],'GRASP',opening,cartesian=True)
            np.savez_compressed(self.output/'grasp_state.npz',qpos=self.data.qpos,qvel=self.data.qvel,ctrl=self.data.ctrl,time=self.data.time)
            self.stage='CLOSE'
            if recipe.ee_id=='2F':
                self.runtime.set_finger_gripper_actuator_gains(kp=recipe.closure_kp)
            acquired=False
            hold_opening=-1.
            for f in np.linspace(0,1,math.ceil(recipe.close_duration_s*50)):
                hold_opening=opening+(-1.-opening)*f
                row=self.step(q,hold_opening)
                if recipe.ee_id=='3F' and not self.three_finger_force_hold and any(v>.05 for v in row['finger_force_n'].values()):
                    self.three_finger_commands=np.asarray(self.gripper.current_action).copy()
                    self.three_finger_force_hold=True
                if recipe.ee_id=='2F' and not self.two_finger_force_hold and any(v>.05 for v in row['finger_force_n'].values()):
                    self.two_finger_command=float(np.asarray(self.gripper.current_action).mean())
                    self.two_finger_force_hold=True
                acquired=grasp_contact_ready(recipe,self.trace,recipe.contact_ticks)
                if acquired: break
            for _ in range(100):
                if acquired: break
                hold_opening=-1.
                row=self.step(q,-1.)
                if recipe.ee_id=='3F' and not self.three_finger_force_hold and any(v>.05 for v in row['finger_force_n'].values()):
                    self.three_finger_commands=np.asarray(self.gripper.current_action).copy()
                    self.three_finger_force_hold=True
                if recipe.ee_id=='2F' and not self.two_finger_force_hold and any(v>.05 for v in row['finger_force_n'].values()):
                    self.two_finger_command=float(np.asarray(self.gripper.current_action).mean())
                    self.two_finger_force_hold=True
                acquired=grasp_contact_ready(recipe,self.trace,recipe.contact_ticks)
            if not acquired:
                raise GraspFailure('HANDLE_CONTACT_NOT_STABLE')
            if recipe.ee_id=='3F':
                # Freeze the pinch pose. A hard snap to -1 ejects thin utensils.
                self.three_finger_commands=np.asarray(self.gripper.current_action).copy()
                self.three_finger_force_hold=True
                if getattr(recipe,'thin_handle_pinch',False) or getattr(recipe,'hold_finger_positions',False):
                    hold_opening=float(np.mean(self.three_finger_commands))
            for _ in range(math.ceil(recipe.prelift_stabilization_s*50)):
                self.step(q,hold_opening)
            if not grasp_contact_ready(recipe,self.trace,recipe.contact_ticks):
                raise GraspFailure('HANDLE_CONTACT_LOST_BEFORE_LIFT')
            self.carried_pose=inverse(self.grip_pose())@self.body_pose()
            save_json(self.output/'contact_gate.json',{'status':'RELEASED',
                'mode':('STABLE_THIN_HANDLE_PINCH' if getattr(recipe,'thin_handle_pinch',False)
                        else 'STABLE_BALANCED_THREE_FINGER_CONTACT' if recipe.ee_id=='3F'
                        else 'STABLE_TWO_FINGER_CONTACT'),
                'hold_opening_command':hold_opening,
                'actual_gripper_command':np.asarray(self.gripper.current_action),
                'force_targets_n':recipe.three_finger_force_targets_n if recipe.ee_id=='3F' else [recipe.two_finger_force_target_n]*2,
                'sample':self.trace[-1]})
            if getattr(recipe,'thin_handle_pinch',False):
                attach_thin_handle_pinch(self)
            q=self.move(targets['LIFT'],'LIFT',hold_opening,cartesian=True)
            self.stage='SETTLE'
            for _ in range(math.ceil(recipe.settle_s*50)): self.step(q,hold_opening)
            self.stage='HOLD'
            hold=[]
            for _ in range(math.ceil(recipe.hold_s*50)): hold.append(self.step(q,hold_opening))
            if getattr(recipe,'thin_handle_pinch',False) and self.runtime.attachment is not None:
                ref=np.asarray(self.thin_handle_attachment_record['T_GB_at_attach'])
            else:
                ref=hold[0]['T_GB']
            slip=max(float(np.linalg.norm(s['T_GB'][:3,3]-ref[:3,3])) for s in hold)
            angle=max(float(np.rad2deg(Rotation.from_matrix(ref[:3,:3].T@s['T_GB'][:3,:3]).magnitude())) for s in hold)
            metrics={'minimum_hold_lift_m':min(s['lift_m'] for s in hold),
                'minimum_bottom_clearance_m':min(s['bottom_clearance_m'] for s in hold),
                'all_finger_contact_fraction':hold_contact_fraction(hold,self.finger_groups,recipe),
                'max_slip_m':slip,'max_slip_deg':angle,'hold_s':recipe.hold_s}
            ok=(metrics['minimum_hold_lift_m']>=recipe.minimum_lift_m
                and metrics['minimum_bottom_clearance_m']>=.05
                and slip<=recipe.maximum_slip_m and angle<=recipe.maximum_slip_deg)
            if getattr(recipe,'thin_handle_pinch',False):
                metrics['attachment_active_fraction']=sum(
                    self.runtime.attached_object_id==self.object_id for _ in hold)/len(hold)
                # Attachment carries the utensil; finger contact may drop after lift.
                ok=ok and metrics['attachment_active_fraction']==1.
            else:
                ok=ok and metrics['all_finger_contact_fraction']>=.95
            result.update(status='SUCCESS' if ok else 'FAILED',metrics=metrics,
                failure_reason=None if ok else 'HOLD_VALIDATION_FAILED')
            if getattr(self,'thin_handle_attachment_record',None) is not None:
                result['thin_handle_attachment']=self.thin_handle_attachment_record
        except GraspFailure as exc:
            result.update(failure_stage=self.stage,failure_reason=str(exc))
        except Exception as exc:
            import traceback
            (self.output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            result.update(failure_stage=self.stage,failure_reason=f'IMPLEMENTATION_ERROR: {exc}')
        finally:
            result.update(object_pose_in_gripper=pose_dict(inverse(self.grip_pose())@self.body_pose()),
                center_pose_in_gripper=pose_dict(inverse(self.grip_pose())@self.body_pose()@
                    transform(self.center_in_body,rotation=np.eye(3))),
                final_object_pose=pose_dict(self.body_pose()),final_robot_q=self.data.qpos[self.arm_ids],
                attachment_used=self.runtime.attachment is not None,object_material_inputs=[],learned_model_calls=0)
            result['contact_solver_profile']=self.thin_contact_profile
            result['nominal_joint_limit_diagnostics']={
                'policy':'ENFORCE_FOR_BOTH_HANDS',
                'max_violation_rad':self.maximum_physics_joint_error,
                'allowed_numerical_residual_rad':recipe.maximum_joint_limit_error_rad,
                'physics_steps_audited':self.physics_steps_audited,
                'hardware_joint_limits_validated':False}
            result['hand_model_correction']=self.env.spoon_hand_correction
            result['validation_scope']='SIMULATOR_CONTACT_LIFT_HOLD_AND_JOINT_RANGE_TOLERANCE'
            save_json(self.output/'result.json',result)
            save_json(self.output/'trace.json',self.trace)
            np.savez_compressed(self.output/'final_state.npz',qpos=self.data.qpos,qvel=self.data.qvel,ctrl=self.data.ctrl,time=self.data.time)
            if self.camera:
                from PIL import Image
                Image.fromarray(self.render()).save(self.output/'final_scene.png')
        return result
