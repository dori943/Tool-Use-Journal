"""Precise full-pose IK with bounded revolute-angle continuity."""
from dataclasses import replace
import math


def bounded_angles_near(q, seed, limits):
    if not len(q)==len(seed)==len(limits):
        raise ValueError('Joint angles, seed, and limits must have matching dimensions')
    result=[]
    for angle,reference,(lower,upper) in zip(q,seed,limits):
        first=math.ceil((lower-angle)/(2*math.pi)-1e-10)
        last=math.floor((upper-angle)/(2*math.pi)+1e-10)
        candidates=[angle+2*math.pi*k for k in range(first,last+1)]
        if not candidates:
            raise ValueError('IK angle has no equivalent value within joint limits')
        nearest=min(candidates,key=lambda v:abs(v-reference))
        result.append(float(min(upper,max(lower,nearest))))
    return tuple(result)


class ContinuousIK:
    """Adapter for the six revolute joints of this UR5e model."""
    def __init__(self, kinematics, initial_q):
        self.inner=kinematics
        self.initial_q=tuple(initial_q)

    def __getattr__(self, name):
        return getattr(self.inner,name)

    def set_reference_qpos(self, qpos):
        if len(qpos) != len(self.initial_q):
            raise ValueError('Live IK reference has the wrong joint count')
        self.initial_q = tuple(qpos)

    def solve_all_ik(self, *args, **kwargs):
        kwargs.setdefault('seed_qpos', self.initial_q)
        kwargs['position_tolerance_m']=min(kwargs.get('position_tolerance_m', 5e-3), 1e-5)
        kwargs['orientation_tolerance_rad']=min(kwargs.get('orientation_tolerance_rad', 5e-2), 5e-5)
        kwargs['max_iterations']=max(kwargs.get('max_iterations',180),240)
        solutions=self.inner.solve_all_ik(*args,**kwargs)
        seed=kwargs.get('seed_qpos')
        if seed is None:
            seed=self.initial_q
        adjusted=tuple(replace(s,qpos=bounded_angles_near(s.qpos,seed,self.inner.joint_limits_rad))
            for s in solutions.solutions)
        return replace(solutions,solutions=adjusted,
            solver_id=solutions.solver_id+'+PRECISE_FULL_POSE+BOUNDED_ANGLE_CONTINUITY')


class PlanningAdapter:
    def __init__(self, adapter, initial_q):
        self.inner=adapter
        self.initial_q=tuple(initial_q)

    def make_kinematics(self):
        return ContinuousIK(self.inner.make_kinematics(),self.initial_q)
