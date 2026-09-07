"""Physical C1_2 spatula handle grasp; isolated from the validated bottle runtime."""
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
from tuj.m5_motion.scripted_grasps.objects.spatula import SpatulaRecipe, build_spatula_targets
from tuj.m5_motion.scripted_grasps.ik_continuity import ContinuousIK


def three_finger_ready(samples, ticks=5):
    return fingers_ready(samples,('thumb','index','pinky'),ticks)


def three_finger_pinch_event(sample):
    """Detect the brief force-closure state before thin-handle over-closing."""
    forces=sample['finger_force_n']
    return (set(sample['finger_contacts'])=={'thumb','index','pinky'}
        and sample['normal_opposition']>=.8 and sample['contact_span_m']>=.020
        and all(forces[name]>.05 for name in ('thumb','index','pinky')))


def fingers_ready(samples, fingers, ticks=5):
    if len(samples) < ticks:
        return False
    return all(set(s['finger_contacts']) == set(fingers)
               and s['normal_opposition'] >= .5 and s['contact_span_m'] >= .010
               and all(v > .1 for v in s['finger_force_n'].values()) for s in samples[-ticks:])


def update_three_finger_commands(commands,measured_forces,recipe):
    """A bounded force integrator: positive command opens the Jaco finger."""
    error=np.asarray(measured_forces,dtype=float)-np.asarray(recipe.three_finger_force_targets_n)
    # Small load-sharing errors otherwise integrate forever, opening one
    # finger until the thin handle loses its opposed contact.
    deadband=getattr(recipe,'three_finger_force_deadband_n',0.)
    error=np.sign(error)*np.maximum(np.abs(error)-deadband,0.)
    delta=np.clip(recipe.three_finger_force_gain*error,-.01,.01)
    return np.clip(np.asarray(commands,dtype=float)+delta,-1.,1.)


def approach_spatula(context, targets, opening=1.):
    """Use an elevated entry if the nominal pre-grasp cannot be reached safely."""
    try:
        context.move(targets['PRE_GRASP'],'PRE_GRASP',opening)
    except GraspFailure as exc:
        if not str(exc).startswith('COLLISION_FREE_PATH_NOT_FOUND'):
            raise
        clearance=targets['PRE_GRASP'].copy()
        clearance[2,3]+=.16
        context.move(clearance,'PRE_CLEARANCE',opening)
        # Do not descend back to an invalid nominal pre-grasp. The caller
        # validates a diagonal Cartesian approach to the same grasp goal.


def preshape_spatula(context,recipe):
    """Shape the hand while holding the actual current arm joints."""
    context.stage='PRESHAPE'
    q=context.data.qpos[context.arm_ids].copy()
    # The Kitchen reset leaves the passive linkage in a short transient.
    # Stabilize at the half-range controller target before arm travel.
    context.gripper.current_action=np.zeros(context.gripper.dof)
    for _ in range(250): context.step(q,0.)
    opening=-recipe.preshape_closure_command
    context.gripper.current_action=np.full(context.gripper.dof,recipe.preshape_closure_command)
    for _ in range(250): context.step(q,opening)
    aperture=context.runtime.fingerpad_separation_m()
    if not recipe.preshape_aperture_m-.005<=aperture<=recipe.preshape_aperture_m+.005:
        raise GraspFailure(f'PRESHAPE_APERTURE_OUTSIDE_RANGE: {aperture}')
    return opening,aperture


from .motion import GraspMotionContext


class SpatulaContext(GraspMotionContext):





    def bad_contacts(self,data,stage):
        bad=[]
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
             'bottom_clearance_m':self.bottom_height()-self.initial_bottom,
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
            if error>.10: violations.append({'joint':name,'violation_rad':error})
        row['gripper_joint_limit_violations']=violations
        # Jaco's distal coordinates use the same negative operating branch as
        # the already validated bottle grasp, despite the XML's nominal range.
        # Keep them as telemetry; the Robotiq 2F guard remains enforceable.
        if violations and self.recipe.ee_id=='2F':
            raise GraspFailure('GRIPPER_JOINT_LIMIT_EXCEEDED: '+str(violations))
        if len(self.trace)%250==0:
            print(f"[spatula] {self.stage}: lift={row['lift_m']:.3f}m fingers={row['finger_contacts']}",flush=True)
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
                if not (self.recipe.hold_finger_positions and self.stage in {'SETTLE','HOLD'}):
                    self.three_finger_commands=update_three_finger_commands(
                        self.three_finger_commands,measured,self.recipe)
                command=self.three_finger_commands
            else:
                command=np.full(self.gripper.dof,float(opening))
            self.gripper.current_action=np.asarray(command,dtype=float)
            action[lo:hi]=0.
        else:
            desired=-float(opening)
            current=float(np.asarray(self.gripper.current_action).mean())
            action[lo:hi]=np.sign(desired-current) if abs(desired-current)>.09 else 0.
        self.player._advance_controller(action)
        return self.sample()






    def execute_spatula(self,recipe):
        self.recipe=recipe
        self.three_finger_force_hold=False
        self.three_finger_commands=None
        result={'status':'FAILED','object_id':'spatula','environment':'C1_2_DoughFlatten',
                'controller':'NATIVE_JOINT_PD_WITH_HANDLE_CONTACT_FORCE_HOLD',
                'recipe':recipe.to_dict(),'scenario':self.scenario}
        try:
            if type(self.gripper).__name__!=recipe.model_class:
                raise GraspFailure('UNSUPPORTED_EE')
            self.record_input()
            targets=build_spatula_targets(self.body_pose(),self.center_in_body,self.local_size,recipe)
            save_json(self.output/'targets.json',targets)
            q=self.data.qpos[self.arm_ids].copy()
            if recipe.ee_id=='2F':
                # Stabilize the passive four-bar linkage before moving the arm;
                # the 25 mm aperture still clears the 16 mm handle.
                opening,aperture=preshape_spatula(self,recipe)
                save_json(self.output/'preshape.json',{'aperture_m':aperture,'opening_command':opening})
                print('[preshape]',aperture,'opening',opening,flush=True)
                approach_spatula(self,targets,opening)
            else:
                self.stage='OPEN'
                for _ in range(75): self.step(q,1.)
                opening=1.
                approach_spatula(self,targets,opening)
            q=self.move(targets['GRASP'],'GRASP',opening,cartesian=True)
            np.savez_compressed(self.output/'grasp_state.npz',qpos=self.data.qpos,qvel=self.data.qvel,ctrl=self.data.ctrl,time=self.data.time)
            self.stage='CLOSE'
            if recipe.ee_id=='2F':
                self.runtime.set_finger_gripper_actuator_gains(kp=recipe.closure_kp)
            acquired=False
            hold_opening=-1.
            for f in np.linspace(0,1,math.ceil(recipe.close_duration_s*50)):
                hold_opening=-1. if recipe.ee_id=='2F' else 1.-2*f
                row=self.step(q,hold_opening)
                acquired=(three_finger_pinch_event(row) if recipe.ee_id=='3F'
                    else fingers_ready(self.trace,self.finger_groups,recipe.contact_ticks))
                if acquired: break
            for _ in range(100):
                if acquired: break
                hold_opening=-1.
                row=self.step(q,-1.)
                acquired=(three_finger_pinch_event(row) if recipe.ee_id=='3F'
                    else fingers_ready(self.trace,self.finger_groups,recipe.contact_ticks))
            if not acquired:
                raise GraspFailure('HANDLE_CONTACT_NOT_STABLE')
            if recipe.ee_id=='3F':
                self.three_finger_commands=np.full(3,hold_opening)
                self.three_finger_force_hold=True
            self.carried_pose=inverse(self.grip_pose())@self.body_pose()
            save_json(self.output/'contact_gate.json',{'status':'RELEASED',
                'mode':'THIN_HANDLE_FORCE_CLOSURE_EVENT' if recipe.ee_id=='3F' else 'STABLE_TWO_FINGER_CONTACT',
                'hold_opening_command':hold_opening,
                'force_targets_n':recipe.three_finger_force_targets_n if recipe.ee_id=='3F' else None,
                'sample':self.trace[-1]})
            q=self.move(targets['LIFT'],'LIFT',hold_opening,cartesian=True)
            self.stage='SETTLE'
            for _ in range(math.ceil(recipe.settle_s*50)): self.step(q,hold_opening)
            self.stage='HOLD'
            hold=[]
            for _ in range(math.ceil(recipe.hold_s*50)): hold.append(self.step(q,hold_opening))
            reference=hold[0]['T_GB']
            slip=max(float(np.linalg.norm(s['T_GB'][:3,3]-reference[:3,3])) for s in hold)
            angle=max(float(np.rad2deg(Rotation.from_matrix(reference[:3,:3].T@s['T_GB'][:3,:3]).magnitude())) for s in hold)
            metrics={'minimum_hold_lift_m':min(s['lift_m'] for s in hold),
                'minimum_bottom_clearance_m':min(s['bottom_clearance_m'] for s in hold),
                'all_finger_contact_fraction':sum(set(s['finger_contacts'])==set(self.finger_groups) for s in hold)/len(hold),
                'max_slip_m':slip,'max_slip_deg':angle,'hold_s':recipe.hold_s}
            success=(metrics['minimum_hold_lift_m']>=recipe.minimum_lift_m and metrics['minimum_bottom_clearance_m']>=.05
                and metrics['all_finger_contact_fraction']>=.95 and slip<=recipe.maximum_slip_m and angle<=recipe.maximum_slip_deg)
            result.update(status='SUCCESS' if success else 'FAILED',metrics=metrics,
                failure_reason=None if success else 'HOLD_VALIDATION_FAILED')
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
            save_json(self.output/'result.json',result)
            save_json(self.output/'trace.json',self.trace)
            np.savez_compressed(self.output/'final_state.npz',qpos=self.data.qpos,qvel=self.data.qvel,ctrl=self.data.ctrl,time=self.data.time)
            if self.camera:
                from PIL import Image
                Image.fromarray(self.render()).save(self.output/'final_scene.png')
        return result
