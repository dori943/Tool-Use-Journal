"""Scripted grasp runner for object-specific catalog recipes.

Reuses the already tested joint/Cartesian planner without changing spoon code.
Fingers use contact forces, with explicit opt-in attachment after stable contact.
Vacuum attaches after verified cup contact.
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
from tuj.m5_motion.scripted_grasps.spoon_runtime import (
    SpoonContext, approach_spoon, grasp_contact_ready, hold_contact_fraction,
    thin_handle_pinch_event, attach_thin_handle_pinch, attach_catalog_kinematic_carry,
    thin_handle_lateral_retry_schedule, recipe_with_thin_handle_lateral,
    thin_handle_nominal_lateral_m, prepare_thin_handle_close_retry,
)


class CatalogContext(SpoonContext):

    def find_support_geoms(self):
        from .support_geometry import measured_support_geoms
        measured = measured_support_geoms(self)
        if measured:
            return measured
        try:return {self.model.geom('table_collision').id}
        except KeyError:pass
        geoms={i for i in range(self.model.ngeom)
            if self.model.geom(i).name.startswith('island_island_group_top_')
            and not self.model.geom(i).name.endswith('_visual')}
        if geoms:return geoms
        raise GraspFailure('SUPPORT_GEOMETRY_NOT_FOUND')

    def support_geom_names(self):
        """All known support surfaces present in the compiled scene."""
        gids = getattr(self, 'support_gids', None)
        if gids is None:
            gids = self.find_support_geoms()
        names=[self.model.geom(int(gid)).name for gid in gids]
        if not names:
            raise GraspFailure('SUPPORT_GEOMETRY_NOT_FOUND')
        return names

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
        lo,hi=splits['right_gripper']
        # opening=-1 means suction-on for catalog vac. After kinematic attach,
        # force native adhesion off so non-target bodies are not pulled.
        gripper_cmd=-float(opening)
        if hasattr(self.runtime,'vac_gripper_action_for_command'):
            gripper_cmd=float(self.runtime.vac_gripper_action_for_command(gripper_cmd))
        action[lo:hi]=gripper_cmd
        self.player._advance_controller(action)
        suppress=getattr(self.runtime,'suppress_native_adhesion_actuators',None)
        if callable(suppress):
            suppress()
        log_contacts=getattr(self.runtime,'log_vac_non_target_cup_contacts',None)
        if callable(log_contacts):
            log_contacts()
        return self.sample()

    def release_vacuum(self):
        """Disable suction and release the object at its current world pose."""
        from tuj.m5_motion.scripted_grasps.catalog_vacuum import release_vacuum
        return release_vacuum(self)

    def contact_in_region(self,position):
        body=self.body_pose()
        local=((np.asarray(position)-body[:3,3])@body[:3,:3]-self.center_in_body)/self.local_size
        return bool(np.all(local>=self.recipe.contact_region_min) and np.all(local<=self.recipe.contact_region_max))

    def valid_state(self,q,key):
        """Carry the vac-held object during BREAKAWAY the same way as LIFT."""
        self.probe.qpos[:]=self.data.qpos
        self.probe.qpos[self.arm_ids]=q
        self.mj.mj_fwdPosition(self.model,self.probe)
        if self.carried_pose is not None and key.keyframe_id in {'LIFT','BREAKAWAY','LEVEL'}:
            body=self.grip_pose(self.probe)@self.carried_pose
            self.probe.qpos[self.object_qadr:self.object_qadr+3]=body[:3,3]
            quat=Rotation.from_matrix(body[:3,:3]).as_quat()
            self.probe.qpos[self.object_qadr+3:self.object_qadr+7]=quat[[3,0,1,2]]
            self.mj.mj_fwdPosition(self.model,self.probe)
        self.last_planning_collision=self.bad_contacts(self.probe,key.keyframe_id)
        return not self.last_planning_collision

    def bad_contacts(self,data,stage):
        from tuj.m5_motion.scripted_grasps.catalog_vacuum import (
            EARLY_LIFT_OBJECT_SUPPORT_PENETRATION_M,
        )
        bad=super().bad_contacts(data,stage)
        if stage in {'BREAKAWAY','LEVEL'}:
            # Spoon exemptions cover GRASP/CLOSE/LIFT/... but not BREAKAWAY /
            # LEVEL. Vac cup↔held-object contact is expected while attached.
            return [c for c in bad if not (
                any(self.model.geom(g).name in c['geoms'] for g in self.finger_geoms)
                and any(self.model.geom(g).name in c['geoms'] for g in self.handle_geoms)
            )]
        if stage!='LIFT' or self.support_released:return bad
        height=float(self.body_pose(data)[2,3]-self.initial_body[2,3])
        if not -.002<=height<=.005:return bad
        support_names={self.model.geom(g).name for g in self.support_gids}
        object_names={self.model.geom(g).name for g in self.object_geoms}
        return [c for c in bad if not (
            support_names.intersection(c['geoms'])
            and any(n in object_names for n in c['geoms'])
            and c['penetration_m']<=max(
                EARLY_LIFT_OBJECT_SUPPORT_PENETRATION_M,
                self.recipe.maximum_support_separation_penetration_m)
        )]

    def sample(self):
        if self.stage=='LIFT' and not self.support_released:
            height=self.body_pose()[2,3]-self.initial_body[2,3]
            touching=any(c.dist<=0 and (
                (int(c.geom1) in self.support_gids and int(c.geom2) in self.object_geoms)
                or (int(c.geom2) in self.support_gids and int(c.geom1) in self.object_geoms))
                for c in self.data.contact[:self.data.ncon])
            # Once the lifted object has physically cleared its original
            # support, any later re-contact is an ordinary collision.
            if height>.002 and not touching:self.support_released=True
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
        # 0912: 이 지표는 접촉 법선 중 최솟값이라 "모든 접촉이 평평한가" 를
        # 묻는다. 접촉이 하나뿐인 접시에서는 그게 곧 밀착 여부지만, 빵처럼
        # 메시 표면이 울퉁불퉁해 접촉이 34개 나오는 면에서는 가장자리 미세
        # 경사 하나가 값을 끌어내린다 (빵은 월드 수직과 0.06 도로 평평한데
        # 최솟값이 0.784 로 나와 게이트에 막혔다). 판정은 그대로 두고 분포만
        # 같이 남겨, 통계의 문제인지 컵이 실제로 경사에 앉은 것인지 가른다.
        alignments=[abs(float(n@self.grip_pose()[:3,2])) for n in normals[first]]
        suction_alignment=min(alignments,default=0.)
        suction_alignment_mean=float(np.mean(alignments)) if alignments else 0.
        suction_aligned_fraction=(
            float(sum(a>=.9 for a in alignments))/len(alignments) if alignments else 0.)
        body=self.body_pose()
        contact_centers={name:((np.mean(group,axis=0)-body[:3,3])@body[:3,:3]-self.center_in_body).tolist()
                         for name,group in points.items() if group}
        row={'time_s':float(self.data.time),'stage':self.stage,'q':self.data.qpos[self.arm_ids],
            'attachment_active':self.runtime.attached_object_id==self.object_id,
            'finger_contacts':[n for n in forces if forces[n]>.01],'finger_force_n':forces,
            'contact_count':sum(map(len,points.values())),'normal_opposition':opposition,'contact_span_m':span,
            'suction_alignment':suction_alignment,'suction_alignment_mean':suction_alignment_mean,'suction_aligned_fraction':suction_aligned_fraction,'lift_m':float(self.body_pose()[2,3]-self.initial_body[2,3]),
            'finger_contact_centers_from_object_center_m':contact_centers,
            'bottom_clearance_m':self.bottom_height()-self.support_top_z,'T_GB':inverse(self.grip_pose())@self.body_pose(),
            'gripper_q':self.data.qpos[self.model.jnt_qposadr[self.hand_joint_ids]],
            'gripper_ctrl':self.data.ctrl[[self.model.actuator(n).id for n in self.gripper.actuators]],
            'object_pose':self.body_pose(),'bad_contacts':self.bad_contacts(self.data,self.stage)}
        self.trace.append(row)
        from .contact_attachment import audit_contact_attachment
        audit_contact_attachment(self, row)
        lift_rows=self.trace[-10:]
        if self.recipe.ee_id=='vac' and self.vacuum_attachment_record is not None and self.stage in {'CLOSE','BREAKAWAY','LIFT','LEVEL','SETTLE','HOLD'}:
            if not row['attachment_active']:raise GraspFailure('VACUUM_ATTACHMENT_LOST')
            reference=np.asarray(self.vacuum_attachment_record['T_GB_at_attach'])
            row['attachment_position_error_m']=float(np.linalg.norm(row['T_GB'][:3,3]-reference[:3,3]))
            row['attachment_angle_error_deg']=float(np.rad2deg(Rotation.from_matrix(reference[:3,:3].T@row['T_GB'][:3,:3]).magnitude()))
            if row['attachment_position_error_m']>self.recipe.maximum_slip_m or row['attachment_angle_error_deg']>self.recipe.maximum_slip_deg:
                raise GraspFailure('VACUUM_ATTACHMENT_POSE_ERROR')
        if self.recipe.ee_id!='vac' and len(lift_rows)==10 and all(s['stage']=='LIFT' for s in lift_rows):
            # After kinematic attach, finger pads may unload while the free joint
            # is welded to the grip site (thin-handle utensils and mug carry).
            if self.runtime.attached_object_id==self.object_id:
                if all(not s.get('attachment_active') for s in lift_rows):
                    raise GraspFailure('KINEMATIC_ATTACHMENT_LOST_DURING_LIFT')
            elif getattr(self.recipe,'thin_handle_pinch',False):
                if all(not thin_handle_pinch_event(s) for s in lift_rows):
                    raise GraspFailure('CONTACT_LOST_DURING_LIFT')
            elif all(set(s['finger_contacts'])!=set(self.finger_groups) for s in lift_rows):
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
        if getattr(self.recipe,'thin_handle_pinch',False):
            return grasp_contact_ready(self.recipe,self.trace,self.recipe.contact_ticks)
        if len(self.trace)<self.recipe.contact_ticks:return False
        for row in self.trace[-self.recipe.contact_ticks:]:
            if set(row['finger_contacts'])!=set(self.finger_groups):return False
            if self.recipe.ee_id=='vac':
                # 0912: 예전 조건은 suction_alignment(접촉 법선 정렬의 최솟값)
                # 이었다. 접촉이 하나뿐인 접시에서는 그게 곧 밀착 여부지만,
                # 메시 표면이 거친 면에서는 가장자리 미세 경사 하나가 값을
                # 끌어내린다. c2_2 의 빵은 몸체 z 가 월드 수직과 0.06 도이고
                # 컵도 수직으로 내려왔는데, 접촉 34개 중 32개가 평균 0.987 로
                # 정렬된 상태에서 최솟값이 0.784 로 나와 막혔다.
                #
                # 밀봉이 성립하는 조건은 "모든 접촉이 평평한가" 가 아니라
                # "컵 면이 평평한 자리에 앉았는가" 다. 평균과 정렬 비율을 함께
                # 요구한다. 컵이 모서리에 걸치면 둘 다 같이 떨어지므로 판정은
                # 느슨해지지 않는다. 접촉이 하나인 접시는 평균=비율=최솟값이라
                # 기존 동작 그대로다. 최솟값은 진단용으로 계속 기록한다.
                if (row['contact_count']<self.recipe.minimum_vacuum_contact_count
                        or row['suction_alignment_mean']<.9
                        or row['suction_aligned_fraction']<.8
                        or min(row['gripper_ctrl'])<.5):return False
            elif row['normal_opposition']<.5 or row['contact_span_m']<.0035 or min(row['finger_force_n'].values())<1.:
                return False
        return True

    def engage_feedback(self,row):
        if not any(v>.05 for v in row['finger_force_n'].values()):return
        if self.recipe.ee_id=='2F' and not self.two_finger_force_hold:
            self.two_finger_command=float(np.asarray(self.gripper.current_action).mean());self.two_finger_force_hold=True
        # hold_finger_positions freezes commands at acquire; starting the force
        # servo on first contact opens overloaded fingers before CLOSE finishes.
        if (self.recipe.ee_id=='3F' and not self.three_finger_force_hold
                and not getattr(self.recipe,'hold_finger_positions',False)):
            self.three_finger_commands=np.asarray(self.gripper.current_action).copy();self.three_finger_force_hold=True

    def _close_fingers_until_ready(self,q,opening,recipe):
        """Ramp CLOSE and return ``(acquired, hold_opening, q)``."""
        self.stage='CLOSE'
        acquired=False
        hold_opening=-1.
        for f in np.linspace(0,1,math.ceil(recipe.close_duration_s*50)):
            hold_opening=opening+(-1.-opening)*f
            row=self.step(q,hold_opening);self.engage_feedback(row)
            if self.ready():
                acquired=True
                break
        for _ in range(100):
            if acquired:
                break
            row=self.step(q,-1.);self.engage_feedback(row)
            acquired=self.ready()
            hold_opening=-1.
        return acquired,hold_opening,q

    def preshape(self):
        self.stage='PRESHAPE';q=self.data.qpos[self.arm_ids].copy()
        opening=-self.recipe.preshape_closure_command
        # 0912: 한계값(1.)에서 목표로 직행하고 정착 구간이 없었다. Robotiq 85 는
        # 패시브 링키지라 직전 동작이 남긴 과도 상태를 안고 들어오면 그 구간에서
        # 관절이 한계를 타고 넘고, 모든 물리 서브스텝을 보는 audit_hand_range 가
        # 이를 잡는다 (c3_1 머그 PRESHAPE 0.011472 rad, 허용 0.01). 파지 하나만
        # 도는 스모크는 과도 상태가 작아 표가 안 났다.
        #
        # spoon_runtime.preshape_spoon 이 같은 문제를 이미 풀어 두었고 (주석:
        # "Stabilize at the half-range controller target before arm travel"),
        # 6개 경로에서 검증됐다. 중간(0.)을 거쳐 거기서 멈춰 링키지를 안정시킨
        # 뒤 목표로 가고 다시 멈추는 그 순서를 그대로 쓴다.
        for value in np.linspace(1.,0.,150):self.step(q,float(value))
        for _ in range(100):self.step(q,0.)
        for value in np.linspace(0.,opening,100):self.step(q,float(value))
        for _ in range(150):self.step(q,opening)
        # 0912: 여기에 이득 10, 틱당 +-0.025 짜리 빠른 루프가 75틱 먼저 돌았다.
        # 명령이 손가락 응답보다 빠르게 움직여 목표 개구를 지나치고 기계적
        # 스토퍼를 파고들었다 (c3_1 머그: 명령이 +0.60 에서 -0.7733 까지 가고
        # 개구 86.87mm, Robotiq 85 최대 행정 85mm, 관절이 하한 0 을 0.010987 rad
        # 침범해 audit_hand_range 가 중단). 아래 느린 루프는 그 빠른 루프가
        # 오실레이션으로 끝나는 것을 수습하려고 이미 있던 것이고, 이득 1 에
        # 틱당 +-0.005, 도달을 20틱 연속으로 요구한다. 빠른 단계를 없애고
        # 처음부터 이쪽으로 수렴시킨다. 스푼 2F 는 애초에 피드백 루프 없이
        # 열린 루프 명령과 정착만으로 개구를 맞춘다 (preshape_spoon).
        aperture=self.runtime.fingerpad_separation_m()
        if abs(aperture-self.recipe.preshape_aperture_m)>.005:
            recovery=[];stable=0;start=float(self.data.time)
            for _ in range(300):
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

    def apply_post_grasp_arm_gains(self):
        """Raise arm tracking gains for post-attach lift (recipe post_grasp_arm_kp)."""
        gain=self.recipe.post_grasp_arm_kp
        if gain is None:
            return
        controller=self.robot.part_controllers['right']
        if getattr(self,'_catalog_arm_gains_original',None) is None:
            self._catalog_arm_gains_original=(
                controller, controller.kp.copy(), controller.kd.copy())
        damping_ratio=controller.kd/(2.*np.sqrt(np.maximum(controller.kp,1e-8)))
        controller.kp=np.full_like(controller.kp,float(gain))
        controller.kd=2.*np.sqrt(controller.kp)*damping_ratio

    def restore_arm_gains(self):
        original=getattr(self,'_catalog_arm_gains_original',None)
        if original is None:
            return
        controller,kp,kd=original
        controller.kp,controller.kd=kp,kd
        self._catalog_arm_gains_original=None

    def execute_object(self,recipe):
        if recipe!=self.recipe:raise ValueError('Context must be initialized with the same recipe')
        if recipe.ee_id=='vac' and recipe.object_id=='plate' and recipe.task_id=='c3_2':
            # Rim fraction is size-relative; retune to the live AABB so a
            # PLATE_SCALE edit cannot seat the cup on the recessed dish.
            from tuj.m5_motion.scripted_grasps.objects.plate_vac import (
                tune_recipe_to_measured_size)
            recipe=tune_recipe_to_measured_size(recipe,self.local_size)
            self.recipe=recipe
        self.execution_started=time.monotonic()
        self.three_finger_force_hold=False
        self.three_finger_commands=None
        self.three_finger_hold_command_min=None
        self.three_finger_hold_command_max=None
        self.two_finger_force_hold=False
        self.two_finger_command=0.
        result={'status':'FAILED','object_id':self.object_id,'task_id':recipe.task_id,'ee_id':recipe.ee_id,
            'recipe':recipe.to_dict(),'scenario':self.scenario,'runtime_version':RUNTIME_VERSION,'timing':self.timing}
        try:
            self.record_input()
            if recipe.ee_id == 'vac':
                from tuj.m5_motion.scripted_grasps.catalog_types import build_collision_surface_vacuum_targets
                targets=build_collision_surface_vacuum_targets(
                    self.body_pose(),self.center_in_body,self.local_size,recipe,
                    self.object_record)
            else:
                targets=build_catalog_targets(self.body_pose(),self.center_in_body,self.local_size,recipe)
            from .release_grasp import resolve_release_clearance_targets
            targets, release_clearance = resolve_release_clearance_targets(self, targets)
            if release_clearance is not None:
                save_json(self.output/'release_grasp_geometry.json',release_clearance)
            save_json(self.output/'targets.json',targets)
            q=self.data.qpos[self.arm_ids].copy()
            if recipe.ee_id=='2F':
                opening,aperture=self.preshape()
            elif recipe.ee_id=='3F':
                # Near-open tip pose clears island/tray; mid-close dips tips into them.
                if recipe.preshape_closure_command < .1:
                    self.stage='OPEN';opening=1.
                    for _ in range(75):self.step(q,opening)
                else:
                    self.stage='PRESHAPE'
                    opening=float(np.clip(1.-2.*recipe.preshape_closure_command,-1.,1.))
                    for value in np.linspace(1.,opening,100):self.step(q,float(value))
                    for _ in range(75):self.step(q,opening)
                aperture=None
            else:
                self.stage='OPEN';opening=1.
                for _ in range(75):self.step(q,opening)
                aperture=None
            save_json(self.output/'preshape.json',{'aperture_m':aperture,'opening_command':opening})
            if recipe.contact_region_endpoint_policy is not None:
                from .endpoint_grasp import resolve_endpoint_support_clearance
                # Measure the real preshaped hand and the current object pose.
                targets=build_catalog_targets(self.body_pose(),self.center_in_body,self.local_size,recipe)
                targets, endpoint_clearance = resolve_endpoint_support_clearance(self, targets)
                save_json(self.output/'endpoint_grasp_geometry.json',endpoint_clearance)
                save_json(self.output/'targets.json',targets)
            approach_spoon(self,targets,opening)
            q=self.move(targets['GRASP'],'GRASP',opening,cartesian=True)
            # Snapshot for vac attach continuity: CLOSE suction can depress the
            # free body into support; attach must not freeze that crushed pose.
            self.grasp_grip_pose=self.grip_pose().copy()
            self.grasp_object_pose=self.body_pose().copy()
            self.grasp_T_GB=inverse(self.grasp_grip_pose)@self.grasp_object_pose
            np.savez_compressed(self.output/'grasp_state.npz',qpos=self.data.qpos,qvel=self.data.qvel,ctrl=self.data.ctrl,time=self.data.time,
                grasp_T_GB=self.grasp_T_GB,grasp_object_pose=self.grasp_object_pose,grasp_grip_pose=self.grasp_grip_pose)
            if recipe.ee_id=='2F':self.runtime.set_finger_gripper_actuator_gains(kp=recipe.closure_kp)
            close_attempts=[]
            acquired,hold_opening,q=self._close_fingers_until_ready(q,opening,recipe)
            close_attempts.append({
                'lateral_m':thin_handle_nominal_lateral_m(recipe) if getattr(recipe,'thin_handle_pinch',False) else None,
                'acquired':bool(acquired),
            })
            if (not acquired) and getattr(recipe,'thin_handle_pinch',False):
                for lateral in thin_handle_lateral_retry_schedule(recipe):
                    q=prepare_thin_handle_close_retry(self,q,opening,targets)
                    recipe=recipe_with_thin_handle_lateral(recipe,lateral)
                    self.recipe=recipe
                    targets=build_catalog_targets(
                        self.body_pose(),self.center_in_body,self.local_size,recipe)
                    save_json(self.output/'targets.json',targets)
                    q=self.move(targets['PRE_GRASP'],'PRE_GRASP',opening,cartesian=True)
                    q=self.move(targets['GRASP'],'GRASP',opening,cartesian=True)
                    self.grasp_grip_pose=self.grip_pose().copy()
                    self.grasp_object_pose=self.body_pose().copy()
                    self.grasp_T_GB=inverse(self.grasp_grip_pose)@self.grasp_object_pose
                    acquired,hold_opening,q=self._close_fingers_until_ready(q,opening,recipe)
                    close_attempts.append({
                        'lateral_m':float(lateral),'acquired':bool(acquired)})
                    if acquired:
                        break
            if getattr(recipe,'thin_handle_pinch',False) and (
                    len(close_attempts)>1 or not acquired):
                save_json(self.output/'thin_handle_close_retry.json',{
                    'attempts':close_attempts,
                    'final_lateral_m':thin_handle_nominal_lateral_m(recipe),
                })
            if not acquired:raise GraspFailure('GRASP_CONTACT_NOT_STABLE')
            result['recipe']=recipe.to_dict()
            if recipe.ee_id=='3F' and (
                getattr(recipe,'thin_handle_pinch',False) or getattr(recipe,'hold_finger_positions',False)
            ):
                self.three_finger_commands=np.asarray(self.gripper.current_action).copy()
                self.three_finger_force_hold=True
                hold_opening=float(np.mean(self.three_finger_commands))
            if recipe.ee_id=='vac':
                from tuj.m5_motion.scripted_grasps.catalog_vacuum import (
                    attach_vacuum, finalize_vacuum_grasp_continuity)
                attach_vacuum(self)
                hold_opening=1.-2.*recipe.suction_command
            # Attach while the CLOSE gate sample is still in the trace tail.
            # Thin utensil pad forces often bleed off during prelift hold even
            # with frozen finger commands; kinematic carry is the retention plan.
            if getattr(recipe,'thin_handle_pinch',False):
                attach_thin_handle_pinch(self)
            elif (recipe.ee_id=='3F'
                  and getattr(recipe,'hold_finger_positions',False)):
                # Larger enclosure bodies (c3_2 mug/fruit) unload fingers under
                # force servo / inertia; contact still gates, attachment carries.
                attach_catalog_kinematic_carry(self)
            run_timed_hold(self,q,hold_opening,recipe.prelift_stabilization_s)
            if recipe.ee_id=='vac':
                if self.runtime.attached_object_id!=self.object_id:raise GraspFailure('VACUUM_ATTACHMENT_LOST')
                # Recover GRASP joint tracking (no Cartesian reseat through the
                # cup/object pair), then rebind kinematic attach to the GRASP
                # object world pose so CLOSE depression is not carried into LIFT.
                self.stage='CLOSE'
                for _ in range(75):
                    self.step(q,hold_opening)
                finalize_vacuum_grasp_continuity(self)
                save_json(self.output/'grasp_reseat.json',{
                    'object_pose':pose_dict(self.body_pose()),
                    'grip_pose':pose_dict(self.grip_pose()),
                    'T_GB':inverse(self.grip_pose())@self.body_pose(),
                    'grasp_T_GB':self.grasp_T_GB,
                })
            elif self.runtime.attachment is None and not self.ready():
                raise GraspFailure('CONTACT_LOST_BEFORE_LIFT')
            if recipe.finger_attachment_policy == 'STABLE_CONTACT':
                from .contact_attachment import attach_after_stable_contact
                hold_opening = attach_after_stable_contact(self)
            self.apply_post_grasp_arm_gains()
            # Snapshot grasp relative pose before any post-attach breakaway so
            # BREAKAWAY/LIFT collision probes carry the held object with the TCP.
            self.carried_pose=inverse(self.grip_pose())@self.body_pose()
            breakaway={'applied':False}
            kinematic_carry = (
                recipe.ee_id=='vac'
                or getattr(self.runtime,'attachment',None) is not None
            )
            if kinematic_carry:
                from tuj.m5_motion.scripted_grasps.catalog_vacuum import (
                    EARLY_LIFT_OBJECT_SUPPORT_PENETRATION_M,
                    breakaway_vacuum_from_support,
                    move_vacuum_cartesian_kinematic,
                    settle_vacuum_arm_tracking)
                # Clear residual support immersion before the long LIFT path so
                # early-LIFT controller dip cannot deepen object↔island contacts
                # past the early-LIFT exemption window.  Vac keeps the dip
                # margin; 3F enclosure omits it (live mug_b exceeded the 20 mm
                # bound when CLOSE immersion + vac dip were stacked).
                if recipe.ee_id=='vac':
                    q, breakaway = breakaway_vacuum_from_support(
                        self, q, hold_opening)
                    result['vacuum_support_breakaway'] = breakaway
                else:
                    q, breakaway = breakaway_vacuum_from_support(
                        self, q, hold_opening,
                        pad_m=EARLY_LIFT_OBJECT_SUPPORT_PENETRATION_M,
                        dip_margin_m=0.0,
                    )
                result['support_breakaway'] = breakaway
                if breakaway.get('applied'):
                    self.carried_pose=inverse(self.grip_pose())@self.body_pose()
                    # Re-base early-LIFT height against the cleared pose. Otherwise
                    # post-breakaway height already exceeds +5 mm and permanently
                    # disables the early support-contact filter before LIFT dips.
                    self.initial_body=self.body_pose()
                    self.initial_bottom=self.bottom_height()
                    self.support_released=False
            save_json(self.output/'contact_gate.json',{'status':'PASSED','sample':self.sample()})
            if recipe.ee_id=='vac':
                # Impedance-only LIFT dips TCP into the table even when AABB looked
                # clear (live plate↔table ~2.5 mm). Always climb by FK first, then
                # finish with Cartesian move so absolute joint PD is tracking
                # before HOLD / transport settle gates.
                move_vacuum_cartesian_kinematic(
                    self, targets['LIFT'], 'LIFT', hold_opening, settle_steps=10)
                q=self.move(targets['LIFT'],'LIFT',hold_opening,cartesian=True)
                settle_vacuum_arm_tracking(self, q, hold_opening)
            else:
                q=self.move(targets['LIFT'],'LIFT',hold_opening,cartesian=True)
                if (recipe.ee_id=='3F'
                        and getattr(self.runtime,'attachment',None) is not None):
                    from tuj.m5_motion.scripted_grasps.catalog_vacuum import (
                        straighten_tool_world_down)
                    # Live fruit_b: ~6° post-grasp tilt made place PRE_PLACE
                    # offset along a skewed approach; precise IK then failed.
                    def _catalog_level_move(ctx, target, stage, opening_cmd):
                        return ctx.move(target, stage, opening_cmd, cartesian=True)
                    q, level = straighten_tool_world_down(
                        self, q, hold_opening, move_fn=_catalog_level_move,
                        artifact_name='tool_level.json')
                    result['tool_level'] = level
            if recipe.ee_id=='vac':
                from tuj.m5_motion.scripted_grasps.catalog_vacuum import (
                    run_kinematic_vacuum_hold,
                    straighten_vacuum_tool_world_down)
                # Bread-like tilted vac grasps leave ~0.083 rad M5 tracking lag;
                # level tool -Z to world -Z before HOLD so transport/place settle.
                q, level = straighten_vacuum_tool_world_down(
                    self, q, hold_opening)
                result['vacuum_tool_level'] = level
                if level.get('applied'):
                    # Impedance SETTLE/HOLD after LEVEL re-sags bread (~35 mm).
                    self.stage='SETTLE'
                    run_kinematic_vacuum_hold(
                        self, q, hold_opening, recipe.settle_s)
                    self.stage='HOLD'
                    hold, measured_hold_s = run_kinematic_vacuum_hold(
                        self, q, hold_opening, recipe.hold_s)
                else:
                    self.stage='SETTLE'
                    run_timed_hold(self, q, hold_opening, recipe.settle_s)
                    self.stage='HOLD'
                    hold, measured_hold_s = run_timed_hold(
                        self, q, hold_opening, recipe.hold_s)
            else:
                self.stage='SETTLE'
                run_timed_hold(self,q,hold_opening,recipe.settle_s)
                self.stage='HOLD';hold,measured_hold_s=run_timed_hold(self,q,hold_opening,recipe.hold_s)
            ref=np.asarray(self.vacuum_attachment_record['T_GB_at_attach']) if recipe.ee_id=='vac' else (
                np.asarray(self.thin_handle_attachment_record['T_GB_at_attach'])
                if getattr(self,'thin_handle_attachment_record',None) is not None else hold[0]['T_GB'])
            if getattr(self, 'finger_attachment_record', None) is not None:
                ref = np.asarray(self.finger_attachment_record['T_GB_at_attach'])
            slip=max(float(np.linalg.norm(s['T_GB'][:3,3]-ref[:3,3])) for s in hold)
            angle=max(float(np.rad2deg(Rotation.from_matrix(ref[:3,:3].T@s['T_GB'][:3,:3]).magnitude())) for s in hold)
            metrics={'minimum_hold_lift_m':min(s['lift_m'] for s in hold),
                'minimum_bottom_clearance_m':min(s['bottom_clearance_m'] for s in hold),
                'all_finger_contact_fraction':hold_contact_fraction(hold,self.finger_groups,recipe),
                'max_slip_m':slip,'max_slip_deg':angle,'hold_s':measured_hold_s,'requested_hold_s':recipe.hold_s}
            ok=metrics['minimum_hold_lift_m']>=recipe.minimum_lift_m and metrics['minimum_bottom_clearance_m']>=.05 and slip<=recipe.maximum_slip_m and angle<=recipe.maximum_slip_deg
            if recipe.ee_id=='vac':
                metrics['attachment_active_fraction']=sum(s['attachment_active'] for s in hold)/len(hold)
                metrics['validation_basis']='CONTACT_GATED_KINEMATIC_ATTACHMENT'
                metrics['pose_error_reference']='ATTACH_TIME'
                ok=ok and metrics['attachment_active_fraction']==1.
            elif getattr(self,'thin_handle_attachment_record',None) is not None:
                metrics['attachment_active_fraction']=sum(
                    self.runtime.attached_object_id==self.object_id for _ in hold)/len(hold)
                metrics['validation_basis']=self.thin_handle_attachment_record.get(
                    'policy','CONTACT_GATED_KINEMATIC_ATTACH')
                ok=ok and metrics['attachment_active_fraction']==1.
            else:ok=ok and metrics['all_finger_contact_fraction']>=.95
            if getattr(self, 'finger_attachment_record', None) is not None:
                metrics['attachment_active_fraction'] = sum(s['attachment_active'] for s in hold)/len(hold)
                metrics['validation_basis'] = 'CONTACT_GATED_KINEMATIC_ATTACHMENT'
                metrics['pose_error_reference'] = 'ATTACH_TIME'
                ok = ok and metrics['attachment_active_fraction'] == 1.
            result.update(status='SUCCESS' if ok else 'FAILED',metrics=metrics,failure_reason=None if ok else 'HOLD_VALIDATION_FAILED')
            if getattr(self,'thin_handle_attachment_record',None) is not None:
                result['thin_handle_attachment']=self.thin_handle_attachment_record
        except Exception as exc:
            import traceback
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            result.update(failure_stage=self.stage,failure_reason=str(exc),error_type=type(exc).__name__)
        finally:
            self.restore_arm_gains()
            result.update(object_pose_in_gripper=pose_dict(inverse(self.grip_pose())@self.body_pose()),
                final_object_pose=pose_dict(self.body_pose()),attachment_used=self.runtime.attachment is not None,
                object_material_inputs=[],learned_model_calls=0,hand_model_correction=self.env.catalog_hand_correction,
                nominal_joint_limit_diagnostics={'max_violation_rad':self.maximum_physics_joint_error,
                    'allowed_numerical_residual_rad':recipe.maximum_joint_limit_error_rad,'physics_steps_audited':self.physics_steps_audited})
            if recipe.ee_id=='vac':
                from tuj.m5_motion.scripted_grasps.catalog_vacuum import VACUUM_POLICY
                result.update(vacuum_policy=VACUUM_POLICY,vacuum_attachment=self.vacuum_attachment_record)
            if getattr(self, 'finger_attachment_record', None) is not None:
                result['finger_attachment'] = self.finger_attachment_record
            save_json(self.output/'result.json',result);save_json(self.output/'trace.json',self.trace)
            np.savez_compressed(self.output/'final_state.npz',qpos=self.data.qpos,qvel=self.data.qvel,ctrl=self.data.ctrl,time=self.data.time)
            if self.camera:
                from PIL import Image
                Image.fromarray(self.render()).save(self.output/'final_scene.png')
        return result
