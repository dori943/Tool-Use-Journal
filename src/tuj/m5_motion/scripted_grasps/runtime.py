"""Calibrated plate driver bound to a caller-owned M5 runtime.

No model calls or material-dependent force calculation. Live object dynamics
remain physical; collision probes use separate MjData. Binding is in context.py.
"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation
from tuj.m5_motion.scripted_grasps.settings import REPOSITORY
from tuj.m5_motion.scripted_grasps.frames import transform, inverse, pose_dict
from tuj.m5_motion.scripted_grasps.objects.plate import PlateRecipe, build_plate_targets

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else
            x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")

class GraspFailure(RuntimeError):
    pass

class PlateContext:

    def descendant(self, body, parent):
        while body != 0 and body != parent:
            body = int(self.model.body_parentid[body])
        return body == parent

    def body_pose(self, data=None):
        data = self.data if data is None else data
        return transform(data.xpos[self.body_id], rotation=data.xmat[self.body_id].reshape(3,3))

    def grip_pose(self, data=None):
        data = self.data if data is None else data
        return transform(data.site_xpos[self.site_id], rotation=data.site_xmat[self.site_id].reshape(3,3))

    def bottom_height(self):
        pose = self.body_pose()
        center = pose[:3, 3] + pose[:3, :3] @ self.center_in_body
        return float(center[2] - np.abs(pose[2, :3]) @ (self.collision_local_size / 2))

    def contact_force(self):
        normal=0.
        for index,c in enumerate(self.data.contact[:self.data.ncon]):
            a,b=int(c.geom1),int(c.geom2)
            if (a in self.finger_geoms and b in self.plate_geoms) or (b in self.finger_geoms and a in self.plate_geoms):
                wrench=np.zeros(6)
                self.mj.mj_contactForce(self.model,self.data,index,wrench)
                normal+=abs(float(wrench[0]))
        return normal

    def finger_table_contact(self,a,b):
        # The reference permits the complete 2F linkage to meet the support
        # surface during the compliant side grasp, including inner knuckles.
        return ((a in self.gripper_geoms and self.model.geom(b).name=='table_collision') or
                (b in self.gripper_geoms and self.model.geom(a).name=='table_collision'))

    def contacts(self, data=None):
        data = self.data if data is None else data
        sides, bad = set(), []
        for contact in data.contact[:data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            if contact.dist > 0:
                continue
            # Internal linkage contacts belong to the unchanged EE model;
            # obstacle checks still include every gripper-to-scene contact.
            if a in self.gripper_geoms and b in self.gripper_geoms:
                continue
            for side, group in self.finger_groups.items():
                if (a in group and b in self.plate_geoms) or (b in group and a in self.plate_geoms):
                    sides.add(side)
            if contact.dist < -0.001 and (a in self.robot_geoms or b in self.robot_geoms):
                allowed = ((a in self.finger_geoms and b in self.plate_geoms) or
                           (b in self.finger_geoms and a in self.plate_geoms))
                if self.finger_table_contact(a,b) and self.stage in {'GRASP','CLOSE','LIFT'}:
                    allowed = contact.dist >= -self.recipe.maximum_actual_table_penetration_m
                if not allowed:
                    bad.append({"geoms": [self.model.geom(a).name, self.model.geom(b).name],
                                "penetration_m": -float(contact.dist)})
        return sorted(sides), bad


    def target_keyframe(self, stage, target):
        from tuj.m5_motion.schema import RelativeKeyframeSpec, KeyframeType
        from tuj.m5_motion.geometry import tool_rotation_from_axis, RelativePoseResolver
        z, x = target[:3,2], target[:3,0]
        base = tool_rotation_from_axis(z, 0.)
        roll = math.atan2(x @ base[:,1], x @ base[:,0])
        key = RelativeKeyframeSpec(keyframe_id=stage, keyframe_type=getattr(KeyframeType, stage),
            frame_ref="object:target", anchor="center", approach_axis_xyz=tuple(z),
            tool_axis_to_align="+z", roll_rad=roll, planner="CARTESIAN")
        world = SimpleNamespace(objects={"target": {"pose": {"position_m": target[:3,3].tolist(),
            "orientation_xyzw": [0,0,0,1]}, "anchors": {"center": [0,0,0]}}}, rack={})
        resolved = RelativePoseResolver(world).resolve(key)
        actual = transform(resolved.position_m, resolved.orientation_xyzw)
        if not np.allclose(actual, target, atol=1e-7):
            raise GraspFailure("POSE_ADAPTER_MISMATCH")
        return key, world

    def execute_plate(self, recipe):
        self.recipe=recipe
        result={"status":"FAILED", "recipe":recipe.to_dict(), "scenario":self.scenario}
        try:
            from tuj.m5_motion.scripted_grasps.reference_adapter import execute
            result=execute(self,recipe)
            return result
        except GraspFailure as exc:
            result.update(failure_stage=self.stage,failure_reason=str(exc))
        except Exception as exc:
            import traceback
            (self.output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            result.update(failure_stage=self.stage,failure_reason=f"IMPLEMENTATION_ERROR: {type(exc).__name__}: {exc}")
        finally:
            result.update(final_object_pose=pose_dict(self.body_pose()),
                final_robot_q=self.data.qpos[self.arm_ids].tolist(),
                object_pose_in_gripper=pose_dict(inverse(self.grip_pose())@self.body_pose()),
                center_pose_in_gripper=pose_dict(inverse(self.grip_pose())@self.body_pose()@
                    transform(self.center_in_body,rotation=np.eye(3))),
                reference_frame=self.gripper.important_sites['grip_site'],source_snapshot=str(REPOSITORY),
                object_material_inputs=[],learned_model_calls=0,attachment_used=False,
                ee_contact_configuration={'source':'configs/robot_spec.json',
                    'fingerpad_friction':self.ee_spec['fingerpad_friction'],'contact_dimension':6})
            save_json(self.output/'result.json',result)
            save_json(self.output/'trace.json',self.trace)
            np.savez_compressed(self.output/'final_state.npz',qpos=self.data.qpos,
                qvel=self.data.qvel,ctrl=self.data.ctrl,time=self.data.time)
        return result
