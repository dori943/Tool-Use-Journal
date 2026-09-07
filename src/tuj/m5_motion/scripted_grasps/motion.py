"""Shared calibrated motion stages using M5 IK, RRT and Cartesian validation."""
from types import SimpleNamespace
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .frames import transform
from .runtime import save_json, GraspFailure


class GraspMotionContext:
    def descendant(self,body,parent):
        while body and body!=parent:
            body=int(self.model.body_parentid[body])
        return body==parent

    def body_pose(self,data=None):
        d=self.data if data is None else data
        return transform(d.xpos[self.body_id],rotation=d.xmat[self.body_id].reshape(3,3))

    def grip_pose(self,data=None):
        d=self.data if data is None else data
        return transform(d.site_xpos[self.site_id],rotation=d.site_xmat[self.site_id].reshape(3,3))

    def bottom_height(self):
        body=self.body_pose()
        center=body[:3,3]+body[:3,:3]@self.center_in_body
        return float(center[2]-np.abs(body[2,:3])@(self.local_size/2))

    def valid_state(self,q,key):
        self.probe.qpos[:]=self.data.qpos
        self.probe.qpos[self.arm_ids]=q
        self.mj.mj_fwdPosition(self.model,self.probe)
        if self.carried_pose is not None and key.keyframe_id=='LIFT':
            body=self.grip_pose(self.probe)@self.carried_pose
            self.probe.qpos[self.object_qadr:self.object_qadr+3]=body[:3,3]
            quat=Rotation.from_matrix(body[:3,:3]).as_quat()
            self.probe.qpos[self.object_qadr+3:self.object_qadr+7]=quat[[3,0,1,2]]
            self.mj.mj_fwdPosition(self.model,self.probe)
        self.last_planning_collision=self.bad_contacts(self.probe,key.keyframe_id)
        return not self.last_planning_collision

    def plan_to(self,target,stage,cartesian=False):
        from tuj.m5_motion.path_planning import RRTConnectEdgePlanner,validate_joint_segment
        self.stage='PLAN_'+stage
        q=self.data.qpos[self.arm_ids].copy()
        key=SimpleNamespace(keyframe_id=stage)
        quat=Rotation.from_matrix(target[:3,:3]).as_quat()
        solutions=self.kinematics.solve_all_ik(target[:3,3],quat,seed_qpos=q)
        candidates=sorted(solutions.solutions,key=lambda s:np.linalg.norm(np.asarray(s.qpos)-q))
        if not candidates:
            raise GraspFailure('IK_FAILED: '+stage+' '+solutions.detail)
        path=None
        if cartesian:
            start=self.grip_pose()
            count=max(1,math.ceil(np.linalg.norm(target[:3,3]-start[:3,3])/.005))
            from scipy.spatial.transform import Slerp
            slerp=Slerp([0,1],Rotation.from_matrix([start[:3,:3],target[:3,:3]]))
            path=[q.tolist()]
            for frac in np.linspace(0,1,count+1)[1:]:
                pos=(1-frac)*start[:3,3]+frac*target[:3,3]
                choices=self.kinematics.solve_all_ik(pos,slerp(frac).as_quat(),seed_qpos=path[-1])
                if not choices.solved:
                    raise GraspFailure('CARTESIAN_IK_FAILED: '+stage)
                solution=min(choices.solutions,key=lambda s:np.linalg.norm(np.asarray(s.qpos)-path[-1]))
                from tuj.m5_motion.scripted_grasps.ik_continuity import bounded_angles_near
                nextq=bounded_angles_near(solution.qpos,path[-1],self.kinematics.joint_limits_rad)
                edge=validate_joint_segment(path[-1],nextq,key,self.valid_state,max_joint_step_rad=.025,wrap_joints=False)
                if not edge.valid:
                    raise GraspFailure('CARTESIAN_COLLISION: '+stage+' '+edge.detail+' '+str(self.last_planning_collision[:1]))
                path.extend(edge.joint_path[1:])
        else:
            planner=RRTConnectEdgePlanner(self.valid_state,self.kinematics.joint_limits_rad,
                random_seed=self.seed,timeout_s=12,max_iterations=5000,wrap_joints=False)
            attempts=[]
            for solution in candidates:
                if not self.valid_state(solution.qpos,key):
                    attempts.append({'q':solution.qpos,'rejection':'GOAL_COLLISION','contacts':self.last_planning_collision})
                    continue
                edge=planner.plan(tuple(q),solution.qpos,None,key)
                if edge.valid:
                    path=edge.joint_path
                    break
                attempts.append({'q':solution.qpos,'rejection':edge.failure_code,'detail':edge.detail})
            if path is None:
                save_json(self.output/(stage.lower()+'_planning_failure.json'),attempts)
                print('[planning rejected]',stage,
                    [{'reason':a['rejection'],'first_contact':a.get('contacts',[None])[0]}
                     for a in attempts],flush=True)
                raise GraspFailure('COLLISION_FREE_PATH_NOT_FOUND: '+stage)
        self.probe.qpos[:]=self.data.qpos
        self.probe.qpos[self.arm_ids]=path[-1]
        self.mj.mj_kinematics(self.model,self.probe)
        actual=self.grip_pose(self.probe)
        errors={'position_error_m':float(np.linalg.norm(actual[:3,3]-target[:3,3])),
            'orientation_error_deg':float(np.rad2deg(Rotation.from_matrix(target[:3,:3].T@actual[:3,:3]).magnitude()))}
        if errors['position_error_m']>.0001 or errors['orientation_error_deg']>.05:
            raise GraspFailure('FULL_MODEL_FK_VALIDATION_FAILED')
        self.plans.append({'stage':stage,'target':target,'path':path,'errors':errors,
            'planner':'CARTESIAN' if cartesian else 'RRT_CONNECT','start_q':q,
            'carried_object_for_collision_only':self.carried_pose})
        save_json(self.output/'motion_plan.json',self.plans)
        print('[plan]',stage,len(path),'points',errors,flush=True)
        return np.asarray(path)

    def move(self,target,stage,opening,cartesian=False):
        path=self.plan_to(target,stage,cartesian)
        self.stage=stage
        lengths=np.linalg.norm(np.diff(path,axis=0),axis=1)
        arc=np.r_[0.,np.cumsum(lengths)]
        duration=max(1.,float(arc[-1])/self.recipe.joint_speed_rad_s*1.5)
        if cartesian:
            duration=max(duration,np.linalg.norm(target[:3,3]-self.grip_pose()[:3,3])/self.recipe.cartesian_speed_m_s*1.5)
        for f in np.linspace(0,1,math.ceil(duration*50)+1)[1:]:
            u=f*f*(3-2*f)*arc[-1]
            q=np.array([np.interp(u,arc,path[:,j]) for j in range(6)])
            self.step(q,opening)
        for _ in range(50):
            self.step(path[-1],opening)
        if self.camera:
            from PIL import Image
            Image.fromarray(self.render()).save(self.output/(stage.lower()+'.png'))
        return path[-1]
