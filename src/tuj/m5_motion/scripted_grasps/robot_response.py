"""Keep the calibrated robot contact response in the object's centre frame.

Only robot inertia and kinematics enter this correction. It changes actuator
torques through the existing controller; it never changes object state.
"""
import json
from pathlib import Path
import numpy as np
from tuj.m5_motion.scripted_grasps.runtime import save_json


def spatial_rotation(rotation):
    axes=np.zeros((6,6))
    axes[:3,:3]=axes[3:,3:]=rotation
    return axes


def object_frame_metric(jacobian,robot_mass,object_rotation):
    inverse=np.linalg.inv(jacobian)
    axes=spatial_rotation(object_rotation)
    world_metric=inverse.T@robot_mass@inverse
    return axes.T@world_metric@axes


class ObjectFrameRobotResponse:
    def __init__(self,context,object_rotation):
        c=self.context=context
        self.controller=c.robot.part_controllers['right']
        self.original=self.controller.run_controller
        self.dof_ids=[int(c.model.jnt_dofadr[c.model.joint(n).id]) for n in c.robot.robot_model.joints]
        probe=c.mj.MjData(c.model)
        probe.qpos[:]=c.data.qpos
        record=json.loads((Path(__file__).parent/'calibrations/plate_robot_response.json').read_text())
        probe.qpos[c.arm_ids]=record['reference_joint_positions_rad']
        c.mj.mj_forward(c.model,probe)
        jac=self.jacobian(probe)
        mass=np.zeros((c.model.nv,c.model.nv))
        c.mj.mj_fullM(c.model,mass,probe.qM)
        self.local_metric=object_frame_metric(jac,mass[np.ix_(self.dof_ids,self.dof_ids)],
            np.array(record['object_frame_rotation_world']))
        axes=spatial_rotation(np.asarray(object_rotation))
        self.metric=axes@self.local_metric@axes.T
        self.metric=(self.metric+self.metric.T)/2
        if np.min(np.linalg.eigvalsh(self.metric))<=0:
            raise ValueError('Robot response metric must be positive definite')
        self.steps=self.clipped_steps=0
        self.blend=0.
        save_json(c.output/'robot_response.json',{'calibration':record,
            'metric_in_object_frame':self.local_metric,'mode':'joint_metric',
            'input_dynamics':'ROBOT_ONLY','object_material_inputs':[]})

    def jacobian(self,data):
        c=self.context
        jp,jr=np.zeros((3,c.model.nv)),np.zeros((3,c.model.nv))
        c.mj.mj_jacSite(c.model,data,jp,jr,c.site_id)
        return np.vstack([jp[:,self.dof_ids],jr[:,self.dof_ids]])

    def run_controller(self):
        c=self.context
        native=self.original()
        if c.stage not in {'GRASP','LIFT'}:
            return native
        ctrl=self.controller
        bias=ctrl.torque_compensation
        acceleration=np.linalg.solve(ctrl.mass_matrix,native-bias)
        jac=self.jacobian(c.data)
        torque=jac.T@self.metric@jac@acceleration+bias
        self.blend=min(1.,self.blend+float(c.env.model_timestep)/.1)
        torque=self.blend*torque+(1-self.blend)*native
        low,high=c.robot.torque_limits
        clipped=np.clip(torque,low,high)
        self.steps+=1
        self.clipped_steps+=int(np.any(np.abs(clipped-torque)>1e-8))
        ctrl.torques=clipped
        return clipped

    def __enter__(self):
        self.controller.run_controller=self.run_controller
        return self

    def __exit__(self,*_):
        self.controller.run_controller=self.original
        save_json(self.context.output/'robot_response_execution.json',{
            'physics_steps':self.steps,'torque_clipped_steps':self.clipped_steps,'controller_restored':True})
