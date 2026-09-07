"""Require consecutive opposed contacts before starting the planned lift."""
import numpy as np
from tuj.m5_motion.examples import c1_1_openai_motion_run as ref
from tuj.m5_motion.scripted_grasps.runtime import GraspFailure,save_json
from tuj.m5_motion.scripted_grasps.robot_response import ObjectFrameRobotResponse


def contact_ready(samples,ticks=5,min_span_m=.0035):
    if len(samples)<ticks:
        return False
    return all(bool(s.get('opposed_contact')) and s.get('contact_point_separation_m',0)>=min_span_m
        and s.get('normal_opposition',-1)>=.5 and s.get('normal_force_n',0)>1.
        for s in samples[-ticks:])


class ContactGatedPlayer(ref._PhysicalGraspControllerTrajectoryPlayer):
    def __init__(self,*args,context,**kwargs):
        super().__init__(*args,**kwargs)
        self.context=context
        self.lift_released=False
        self.gate_wait=0.
        self.gate_record={'status':'NOT_REACHED'}

    def _plan_time_step_s(self,*,segment,control_timestep_s):
        step=super()._plan_time_step_s(segment=segment,control_timestep_s=control_timestep_s)
        if segment.metadata.get('keyframe_id')=='LIFT' and not self.lift_released:
            recipe=self.context.recipe
            ready=contact_ready(self._physical_grasp_monitor.samples,
                ticks=recipe.prelift_contact_ticks,min_span_m=recipe.prelift_contact_span_m)
            if ready:
                self.lift_released=True
                latest=self._physical_grasp_monitor.samples[-1]
                self.gate_record={'status':'RELEASED','wait_s':self.gate_wait,
                    'contact_span_m':latest['contact_point_separation_m'],
                    'normal_opposition':latest['normal_opposition']}
            else:
                self.gate_wait+=control_timestep_s
                self.gate_record={'status':'WAITING','wait_s':self.gate_wait}
                if self.gate_wait>recipe.prelift_max_wait_s:
                    self.gate_record['status']='FAILED'
                    raise GraspFailure('PRE_LIFT_CONTACT_NOT_STABLE')
                self.context.contact_gate_state=self.gate_record.copy()
                return 0.
        self.context.contact_gate_state=self.gate_record.copy()
        return step


def run(context,runtime,plan,registry,run_id,monitor,center_pose):
    simulation_run=ref.SimulationRun(run_id=run_id,
        provenance=ref._provenance(run_id+':artifact','SimulationRun',ref.ModuleName.SIMULATOR,plan.provenance.artifact_id),
        plan=plan,config=ref.SimulationConfig(physics_timestep_s=float(context.env.model_timestep),
            control_timestep_s=float(context.env.control_timestep),realtime_factor=0.,
            max_duration_s=max(30.,float(plan.duration_s)+context.recipe.prelift_max_wait_s+5.),
            terminate_on_collision=True,render=False,random_seed=context.seed))
    player=ContactGatedPlayer(runtime,context=context,monitor=monitor,
        collision_probe=registry,collision_check_stride=context.recipe.execution_collision_check_stride)
    try:
        with ObjectFrameRobotResponse(context,np.asarray(center_pose)[:3,:3]):
            return simulation_run,player.execute(simulation_run)
    finally:
        save_json(context.output/'contact_gate.json',player.gate_record)
