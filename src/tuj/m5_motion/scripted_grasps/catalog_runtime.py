"""Scripted grasp runner for object-specific catalog recipes.

Reuses the already tested joint/Cartesian planner without changing spoon code.
Fingers use contact forces; vacuum attaches after verified cup contact.
"""
from pathlib import Path
import importlib.util
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation
from tuj.m5_motion.scripted_grasps.settings import REPOSITORY
from tuj.m5_motion.scripted_grasps.frames import transform,inverse,pose_dict
from tuj.m5_motion.scripted_grasps.runtime import save_json,GraspFailure
from tuj.m5_motion.scripted_grasps.ik_continuity import ContinuousIK
from tuj.m5_motion.scripted_grasps.catalog_types import build_catalog_targets
from tuj.m5_motion.scripted_grasps.catalog_timing import RUNTIME_VERSION,synchronize_timing,check_control_elapsed,run_timed_hold
from tuj.m5_motion.scripted_grasps.spoon_runtime import SpoonContext,approach_spoon


class CatalogContext(SpoonContext):

    def find_support_geom(self):
        for name in ('table_collision','island_island_group_top_2'):
            try:return self.model.geom(name).id
            except KeyError:pass
        raise GraspFailure('SUPPORT_GEOMETRY_NOT_FOUND')

    def geom_top_height(self,gid):
        import itertools
        if self.model.geom_type[gid]==self.mj.mjtGeom.mjGEOM_MESH:
            mid=int(self.model.geom_dataid[gid]);start=int(self.model.mesh_vertadr[mid]);count=int(self.model.mesh_vertnum[mid])
            vertices=self.model.mesh_vert[start:start+count]
        elif self.model.geom_type[gid]==self.mj.mjtGeom.mjGEOM_BOX:
            vertices=np.array(list(itertools.product([-1,1],repeat=3)))*self.model.geom_size[gid]
        else:raise GraspFailure('UNSUPPORTED_SUPPORT_GEOMETRY')
        return float((vertices@self.data.geom_xmat[gid].reshape(3,3).T+self.data.geom_xpos[gid])[:,2].max())

    def audit_hand_range(self):
        if len(self.hand_joint_ids):return super().audit_hand_range()
        self.physics_steps_audited+=1
        if any(self.data.warning[i].number for i in (4,5,6)):raise GraspFailure('PHYSICS_NUMERICAL_INSTABILITY')

    def step(self,q,opening):
        if self.recipe.ee_id!='vac':return super().step(q,opening)
        if float(opening)>=1. and self.runtime.attachment is not None:
            self.release_vacuum()
        action=np.zeros(self.robot.action_dim);splits=self.robot.composite_controller._action_split_indexes
        lo,hi=splits['right'];action[lo:hi]=q
        lo,hi=splits['right_gripper'];action[lo:hi]=-float(opening)
        self.player._advance_controller(action)
        return self.sample()

    def release_vacuum(self):
        """Disable suction and release the object at its current world pose."""
        from tuj.m5_motion.scripted_grasps.catalog_vacuum import release_vacuum
        return release_vacuum(self)

    def contact_in_region(self,position):
        body=self.body_pose()
        local=((np.asarray(position)-body[:3,3])@body[:3,:3]-self.center_in_body)/self.local_size
        return bool(np.all(local>=self.recipe.contact_region_min) and np.all(local<=self.recipe.contact_region_max))

    def bad_contacts(self,data,stage):
        bad=super().bad_contacts(data,stage)
        if stage!='LIFT' or self.support_released:return bad
        height=float(self.body_pose(data)[2,3]-self.initial_body[2,3])
        if not -.002<=height<=.005:return bad
        support=self.model.geom(self.support_gid).name
        object_names={self.model.geom(g).name for g in self.object_geoms}
        return [c for c in bad if not (support in c['geoms'] and any(n in object_names for n in c['geoms']) and c['penetration_m']<=.002)]

    def sample(self):
        if self.stage=='LIFT' and self.body_pose()[2,3]-self.initial_body[2,3]>.005:self.support_released=True
        forces={n:0. for n in self.finger_groups};points={n:[] for n in forces};normals={n:[] for n in forces}
        for i,c in enumerate(self.data.contact[:self.data.ncon]):
            a,b=int(c.geom1),int(c.geom2)
            if c.dist>0:continue
            if not ((a in self.finger_geoms and b in self.handle_geoms) or (b in self.finger_geoms and a in self.handle_geoms)):continue
            if not self.contact_in_region(c.pos):continue
            for name,geoms in self.finger_groups.items():
                if (a in geoms and b in self.handle_geoms) or (b in geoms and a in self.handle_geoms):
                    wrench=np.zeros(6);self.mj.mj_contactForce(self.model,self.data,i,wrench)
                    forces[name]+=max(0.,float(wrench[0]));points[name].append(np.asarray(c.pos))
                    normals[name].append(np.asarray(c.frame).reshape(3,3)[0]*(1 if a in geoms else -1))
        opposition=0.;span=0.;first=list(forces)[0]
        for other in list(forces)[1:]:
            for n,p in zip(normals[first],points[first]):
                for v,q in zip(normals[other],points[other]):
                    opposition=max(opposition,float(-n@v));span=max(span,float(np.linalg.norm(p-q)))
        suction_alignment=min((abs(float(n@self.grip_pose()[:3,2])) for n in normals[first]),default=0.)
        body=self.body_pose()
        contact_centers={name:((np.mean(group,axis=0)-body[:3,3])@body[:3,:3]-self.center_in_body).tolist()
                         for name,group in points.items() if group}
        row={'time_s':float(self.data.time),'stage':self.stage,'q':self.data.qpos[self.arm_ids],
            'attachment_active':self.runtime.attached_object_id==self.object_id,
            'finger_contacts':[n for n in forces if forces[n]>.01],'finger_force_n':forces,
            'contact_count':sum(map(len,points.values())),'normal_opposition':opposition,'contact_span_m':span,
            'suction_alignment':suction_alignment,'lift_m':float(self.body_pose()[2,3]-self.initial_body[2,3]),
            'finger_contact_centers_from_object_center_m':contact_centers,
            'bottom_clearance_m':self.bottom_height()-self.support_top_z,'T_GB':inverse(self.grip_pose())@self.body_pose(),
            'gripper_q':self.data.qpos[self.model.jnt_qposadr[self.hand_joint_ids]],
            'gripper_ctrl':self.data.ctrl[[self.model.actuator(n).id for n in self.gripper.actuators]],
            'object_pose':self.body_pose(),'bad_contacts':self.bad_contacts(self.data,self.stage)}
        self.trace.append(row)
        lift_rows=self.trace[-10:]
        if self.recipe.ee_id=='vac' and self.vacuum_attachment_record is not None and self.stage in {'CLOSE','LIFT','SETTLE','HOLD'}:
            if not row['attachment_active']:raise GraspFailure('VACUUM_ATTACHMENT_LOST')
            reference=np.asarray(self.vacuum_attachment_record['T_GB_at_attach'])
            row['attachment_position_error_m']=float(np.linalg.norm(row['T_GB'][:3,3]-reference[:3,3]))
            row['attachment_angle_error_deg']=float(np.rad2deg(Rotation.from_matrix(reference[:3,:3].T@row['T_GB'][:3,:3]).magnitude()))
            if row['attachment_position_error_m']>self.recipe.maximum_slip_m or row['attachment_angle_error_deg']>self.recipe.maximum_slip_deg:
                raise GraspFailure('VACUUM_ATTACHMENT_POSE_ERROR')
        if self.recipe.ee_id!='vac' and len(lift_rows)==10 and all(s['stage']=='LIFT' and set(s['finger_contacts'])!=set(self.finger_groups) for s in lift_rows):
            raise GraspFailure('CONTACT_LOST_DURING_LIFT')
        if self.max_runtime_s is not None and time.monotonic()-self.execution_started>self.max_runtime_s:
            raise GraspFailure('TIME_BUDGET_EXCEEDED')
        if len(self.trace)%250==0:print('[catalog]',self.object_id,self.stage,f"lift={row['lift_m']:.3f}",row['finger_contacts'],flush=True)
        if self.video and len(self.trace)%5==0:
            if self.writer is None:
                import imageio.v2 as imageio
                self.writer=imageio.get_writer(str(self.output/'execution.mp4'),fps=10,codec='libx264',quality=7)
            self.writer.append_data(self.render())
        if row['bad_contacts']:raise GraspFailure('UNEXPECTED_COLLISION: '+str(row['bad_contacts'][:2]))
        return row


    def ready(self):
        if len(self.trace)<self.recipe.contact_ticks:return False
        for row in self.trace[-self.recipe.contact_ticks:]:
            if set(row['finger_contacts'])!=set(self.finger_groups):return False
            if self.recipe.ee_id=='vac':
                if row['contact_count']<3 or row['suction_alignment']<.9 or min(row['gripper_ctrl'])<.5:return False
            elif row['normal_opposition']<.5 or row['contact_span_m']<.0035 or min(row['finger_force_n'].values())<1.:
                return False
        return True

    def engage_feedback(self,row):
        if not any(v>.05 for v in row['finger_force_n'].values()):return
        if self.recipe.ee_id=='2F' and not self.two_finger_force_hold:
            self.two_finger_command=float(np.asarray(self.gripper.current_action).mean());self.two_finger_force_hold=True
        if self.recipe.ee_id=='3F' and not self.three_finger_force_hold:
            self.three_finger_commands=np.asarray(self.gripper.current_action).copy();self.three_finger_force_hold=True

    def preshape(self):
        self.stage='PRESHAPE';q=self.data.qpos[self.arm_ids].copy()
        opening=-self.recipe.preshape_closure_command
        for value in np.linspace(1.,opening,100):self.step(q,float(value))
        for _ in range(75):
            aperture=self.runtime.fingerpad_separation_m()
            correction=np.clip(10.*(self.recipe.preshape_aperture_m-aperture),-.025,.025)
            opening=float(np.clip(opening+correction,-1.,1.))
            self.step(q,opening)
        aperture=self.runtime.fingerpad_separation_m()
        if abs(aperture-self.recipe.preshape_aperture_m)>.005:
            # The original fixed wait can end during a finger oscillation.
            # Recover with a slower aperture loop and require sustained arrival.
            recovery=[];stable=0;start=float(self.data.time)
            for _ in range(200):
                correction=np.clip(self.recipe.preshape_aperture_m-aperture,-.005,.005)
                opening=float(np.clip(opening+correction,-1.,1.))
                self.step(q,opening)
                aperture=self.runtime.fingerpad_separation_m()
                stable=stable+1 if abs(aperture-self.recipe.preshape_aperture_m)<=.003 else 0
                recovery.append({'time_s':float(self.data.time),'aperture_m':aperture,'opening':opening})
                if stable>=20:break
            save_json(self.output/'preshape_recovery.json',{'policy':'SLOW_APERTURE_FEEDBACK',
                'elapsed_s':float(self.data.time)-start,'stable_ticks':stable,'samples':recovery})
            if stable<20:raise GraspFailure(f'PRESHAPE_APERTURE_NOT_SETTLED: {aperture}')
        if abs(aperture-self.recipe.preshape_aperture_m)>.005:
            raise GraspFailure(f'PRESHAPE_APERTURE_OUTSIDE_RANGE: {aperture}')
        return opening,aperture

    def execute_object(self,recipe):
        if recipe!=self.recipe:raise ValueError('Context must be initialized with the same recipe')
        self.execution_started=time.monotonic()
        result={'status':'FAILED','object_id':self.object_id,'task_id':recipe.task_id,'ee_id':recipe.ee_id,
            'recipe':recipe.to_dict(),'scenario':self.scenario,'runtime_version':RUNTIME_VERSION,'timing':self.timing}
        try:
            self.record_input()
            targets=build_catalog_targets(self.body_pose(),self.center_in_body,self.local_size,recipe)
            save_json(self.output/'targets.json',targets)
            q=self.data.qpos[self.arm_ids].copy()
            if recipe.ee_id=='2F':opening,aperture=self.preshape()
            else:
                self.stage='OPEN';opening=1.
                for _ in range(75):self.step(q,opening)
                aperture=None
            save_json(self.output/'preshape.json',{'aperture_m':aperture,'opening_command':opening})
            approach_spoon(self,targets,opening)
            q=self.move(targets['GRASP'],'GRASP',opening,cartesian=True)
            np.savez_compressed(self.output/'grasp_state.npz',qpos=self.data.qpos,qvel=self.data.qvel,ctrl=self.data.ctrl,time=self.data.time)
            self.stage='CLOSE'
            if recipe.ee_id=='2F':self.runtime.set_finger_gripper_actuator_gains(kp=recipe.closure_kp)
            acquired=False;hold_opening=-1.
            for f in np.linspace(0,1,math.ceil(recipe.close_duration_s*50)):
                hold_opening=opening+(-1.-opening)*f
                row=self.step(q,hold_opening);self.engage_feedback(row)
                if self.ready():acquired=True;break
            for _ in range(100):
                if acquired:break
                row=self.step(q,-1.);self.engage_feedback(row);acquired=self.ready();hold_opening=-1.
            if not acquired:raise GraspFailure('GRASP_CONTACT_NOT_STABLE')
            if recipe.ee_id=='vac':
                from tuj.m5_motion.scripted_grasps.catalog_vacuum import attach_vacuum
                attach_vacuum(self)
                hold_opening=1.-2.*recipe.suction_command
            run_timed_hold(self,q,hold_opening,recipe.prelift_stabilization_s)
            if recipe.ee_id=='vac':
                if self.runtime.attached_object_id!=self.object_id:raise GraspFailure('VACUUM_ATTACHMENT_LOST')
            elif not self.ready():raise GraspFailure('CONTACT_LOST_BEFORE_LIFT')
            self.carried_pose=inverse(self.grip_pose())@self.body_pose()
            save_json(self.output/'contact_gate.json',{'status':'PASSED','sample':self.trace[-1]})
            q=self.move(targets['LIFT'],'LIFT',hold_opening,cartesian=True)
            self.stage='SETTLE'
            run_timed_hold(self,q,hold_opening,recipe.settle_s)
            self.stage='HOLD';hold,measured_hold_s=run_timed_hold(self,q,hold_opening,recipe.hold_s)
            ref=np.asarray(self.vacuum_attachment_record['T_GB_at_attach']) if recipe.ee_id=='vac' else hold[0]['T_GB']
            slip=max(float(np.linalg.norm(s['T_GB'][:3,3]-ref[:3,3])) for s in hold)
            angle=max(float(np.rad2deg(Rotation.from_matrix(ref[:3,:3].T@s['T_GB'][:3,:3]).magnitude())) for s in hold)
            metrics={'minimum_hold_lift_m':min(s['lift_m'] for s in hold),
                'minimum_bottom_clearance_m':min(s['bottom_clearance_m'] for s in hold),
                'all_finger_contact_fraction':sum(set(s['finger_contacts'])==set(self.finger_groups) for s in hold)/len(hold),
                'max_slip_m':slip,'max_slip_deg':angle,'hold_s':measured_hold_s,'requested_hold_s':recipe.hold_s}
            ok=metrics['minimum_hold_lift_m']>=recipe.minimum_lift_m and metrics['minimum_bottom_clearance_m']>=.05 and slip<=recipe.maximum_slip_m and angle<=recipe.maximum_slip_deg
            if recipe.ee_id=='vac':
                metrics['attachment_active_fraction']=sum(s['attachment_active'] for s in hold)/len(hold)
                metrics['validation_basis']='CONTACT_GATED_KINEMATIC_ATTACHMENT'
                metrics['pose_error_reference']='ATTACH_TIME'
                ok=ok and metrics['attachment_active_fraction']==1.
            else:ok=ok and metrics['all_finger_contact_fraction']>=.95
            result.update(status='SUCCESS' if ok else 'FAILED',metrics=metrics,failure_reason=None if ok else 'HOLD_VALIDATION_FAILED')
        except Exception as exc:
            import traceback
            (self.output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            result.update(failure_stage=self.stage,failure_reason=str(exc),error_type=type(exc).__name__)
        finally:
            result.update(object_pose_in_gripper=pose_dict(inverse(self.grip_pose())@self.body_pose()),
                final_object_pose=pose_dict(self.body_pose()),attachment_used=self.runtime.attachment is not None,
                object_material_inputs=[],learned_model_calls=0,hand_model_correction=self.env.catalog_hand_correction,
                nominal_joint_limit_diagnostics={'max_violation_rad':self.maximum_physics_joint_error,
                    'allowed_numerical_residual_rad':recipe.maximum_joint_limit_error_rad,'physics_steps_audited':self.physics_steps_audited})
            if recipe.ee_id=='vac':
                from tuj.m5_motion.scripted_grasps.catalog_vacuum import VACUUM_POLICY
                result.update(vacuum_policy=VACUUM_POLICY,vacuum_attachment=self.vacuum_attachment_record)
            save_json(self.output/'result.json',result);save_json(self.output/'trace.json',self.trace)
            np.savez_compressed(self.output/'final_state.npz',qpos=self.data.qpos,qvel=self.data.qvel,ctrl=self.data.ctrl,time=self.data.time)
            if self.camera:
                from PIL import Image
                Image.fromarray(self.render()).save(self.output/'final_scene.png')
        return result
