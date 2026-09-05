"""Physical C1_2 bottle grasp; commanded joints only, no object attachment."""
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
from tuj.m5_motion.scripted_grasps.objects.bottle import BottleRecipe, build_bottle_targets
from tuj.m5_motion.scripted_grasps.ik_continuity import ContinuousIK


def three_finger_ready(samples, ticks=5):
    if len(samples) < ticks:
        return False
    return all(set(s['finger_contacts']) == {'thumb','index','pinky'}
               and s['normal_opposition'] >= .5 and s['contact_span_m'] >= .025
               and all(v > .1 for v in s['finger_force_n'].values()) for s in samples[-ticks:])


def approach_bottle(context, targets):
    """Use an elevated entry if the nominal pre-grasp cannot be reached safely."""
    try:
        context.move(targets['PRE_GRASP'],'PRE_GRASP',1.)
    except GraspFailure as exc:
        if not str(exc).startswith('COLLISION_FREE_PATH_NOT_FOUND'):
            raise
        clearance=targets['PRE_GRASP'].copy()
        clearance[2,3]+=.16
        context.move(clearance,'PRE_CLEARANCE',1.)
        # Do not descend back to an invalid nominal pre-grasp. The caller
        # validates a diagonal Cartesian approach to the same grasp goal.


from .motion import GraspMotionContext


class BottleContext(GraspMotionContext):





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
            finger=(a in self.finger_geoms and b in self.object_geoms) or (b in self.finger_geoms and a in self.object_geoms)
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
                if (a in geoms and b in self.object_geoms) or (b in geoms and a in self.object_geoms):
                    wrench=np.zeros(6)
                    self.mj.mj_contactForce(self.model,self.data,i,wrench)
                    forces[finger]+=max(0.,float(wrench[0]))
                    points[finger].append(np.asarray(con.pos))
                    normals[finger].append(np.asarray(con.frame).reshape(3,3)[0]*(1 if a in geoms else -1))
        opposition=0.
        span=0.
        for other in ('index','pinky'):
            for n,p in zip(normals['thumb'],points['thumb']):
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
             'object_pose':self.body_pose(),
             'bad_contacts':self.bad_contacts(self.data,self.stage)}
        self.trace.append(row)
        if len(self.trace)%250==0:
            print(f"[bottle] {self.stage}: lift={row['lift_m']:.3f}m fingers={row['finger_contacts']}",flush=True)
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
        # Jaco format_action integrates the sign of an input into current_action.
        # Set the desired normalized opening command, then send zero increment.
        self.gripper.current_action=np.full(self.gripper.dof,float(opening))
        lo,hi=splits['right_gripper']; action[lo:hi]=0.
        self.player._advance_controller(action)
        return self.sample()






    def execute_bottle(self,recipe):
        self.recipe=recipe
        result={'status':'FAILED','object_id':'bottle','environment':'C1_2_DoughFlatten',
                'controller':'NATIVE_JOINT_PD_WITH_THREE_FINGER_GATE',
                'recipe':recipe.to_dict(),'scenario':self.scenario}
        try:
            if type(self.gripper).__name__!=recipe.model_class:
                raise GraspFailure('UNSUPPORTED_EE')
            self.record_input()
            targets=build_bottle_targets(self.body_pose(),self.center_in_body,self.local_size,recipe)
            save_json(self.output/'targets.json',targets)
            self.stage='OPEN'
            q=self.data.qpos[self.arm_ids].copy()
            # Jaco joint coordinate increases to open (opposite to Robotiq).
            for _ in range(75): self.step(q,1.)
            approach_bottle(self,targets)
            q=self.move(targets['GRASP'],'GRASP',1.,cartesian=True)
            self.stage='CLOSE'
            for f in np.linspace(0,1,math.ceil(recipe.close_duration_s*50)):
                self.step(q,1.-2*f)
            for _ in range(100):
                if three_finger_ready(self.trace,recipe.contact_ticks): break
                self.step(q,-1.)
            if not three_finger_ready(self.trace,recipe.contact_ticks):
                raise GraspFailure('THREE_FINGER_CONTACT_NOT_STABLE')
            self.carried_pose=inverse(self.grip_pose())@self.body_pose()
            save_json(self.output/'contact_gate.json',{'status':'RELEASED','sample':self.trace[-1]})
            q=self.move(targets['LIFT'],'LIFT',-1.,cartesian=True)
            self.stage='SETTLE'
            for _ in range(math.ceil(recipe.settle_s*50)): self.step(q,-1.)
            self.stage='HOLD'
            hold=[]
            for _ in range(math.ceil(recipe.hold_s*50)): hold.append(self.step(q,-1.))
            reference=hold[0]['T_GB']
            slip=max(float(np.linalg.norm(s['T_GB'][:3,3]-reference[:3,3])) for s in hold)
            angle=max(float(np.rad2deg(Rotation.from_matrix(reference[:3,:3].T@s['T_GB'][:3,:3]).magnitude())) for s in hold)
            metrics={'minimum_hold_lift_m':min(s['lift_m'] for s in hold),
                'minimum_bottom_clearance_m':min(s['bottom_clearance_m'] for s in hold),
                'three_finger_contact_fraction':sum(len(s['finger_contacts'])==3 for s in hold)/len(hold),
                'max_slip_m':slip,'max_slip_deg':angle,'hold_s':recipe.hold_s}
            success=(metrics['minimum_hold_lift_m']>=recipe.minimum_lift_m and metrics['minimum_bottom_clearance_m']>=.05
                and metrics['three_finger_contact_fraction']>=.95 and slip<=recipe.maximum_slip_m and angle<=recipe.maximum_slip_deg)
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
            save_json(self.output/'result.json',result)
            save_json(self.output/'trace.json',self.trace)
            np.savez_compressed(self.output/'final_state.npz',qpos=self.data.qpos,qvel=self.data.qvel,ctrl=self.data.ctrl,time=self.data.time)
            if self.camera:
                from PIL import Image
                Image.fromarray(self.render()).save(self.output/'final_scene.png')
        return result
