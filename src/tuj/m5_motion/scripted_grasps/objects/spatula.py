"""Handle grasp for the C1_2 spatula, in the geometric-center frame."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.transform import Rotation
from tuj.m5_motion.scripted_grasps.frames import transform


@dataclass(frozen=True)
class SpatulaRecipe:
    recipe_id: str = 'spatula_3f_short_handle_position_hold_v2'
    ee_id: str = '3F'
    model_class: str = 'JacoThreeFingerDexterousGripper'
    handle_fraction: float = -.12
    lateral_offset_m: float = -.003
    height_offset_m: float = .017
    tilt_deg: float = 0.
    approach_distance_m: float = .12
    lift_distance_m: float = .15
    arm_kp: float = 150.
    close_duration_s: float = 2.
    closure_kp: float = 20.
    preshape_aperture_m: float = .025
    preshape_closure_command: float = .5
    physics_timestep_s: float = .001
    physics_integrator: str = 'implicitfast'
    thin_contact_timeconstant_s: float = .004
    three_finger_force_targets_n: tuple = (3.,1.5,1.5)
    three_finger_force_gain: float = .002
    three_finger_force_deadband_n: float = 0.
    hold_finger_positions: bool = True
    post_grasp_velocity_scaling: float | None = .1
    post_grasp_acceleration_scaling: float | None = .1
    settle_s: float = 1.
    hold_s: float = 5.
    joint_speed_rad_s: float = .35
    cartesian_speed_m_s: float = .04
    minimum_lift_m: float = .10
    maximum_slip_m: float = .005
    maximum_slip_deg: float = 5.
    contact_ticks: int = 5

    def __post_init__(self):
        values=asdict(self)
        for key,value in values.items():
            if isinstance(value,(float,int)) and not np.isfinite(value):
                raise ValueError(f'{key} must be finite')
        models={'2F':'Robotiq85Gripper','3F':'JacoThreeFingerDexterousGripper'}
        if models.get(self.ee_id)!=self.model_class:
            raise ValueError('UNSUPPORTED_EE: spatula recipe requires matching 2F or 3F')
        if not -.38 <= self.handle_fraction <= -.08:
            raise ValueError('Target must lie within the handle')
        if abs(self.lateral_offset_m)>.025 or not -.005<=self.height_offset_m<=.07 or abs(self.tilt_deg)>60:
            raise ValueError('Offset outside spatula calibration range')
        for key in ('approach_distance_m','lift_distance_m','arm_kp','close_duration_s','closure_kp','preshape_aperture_m','thin_contact_timeconstant_s','settle_s','hold_s',
                    'joint_speed_rad_s','cartesian_speed_m_s','minimum_lift_m','maximum_slip_m','maximum_slip_deg'):
            if values[key]<=0: raise ValueError(f'{key} must be positive')
        if not isinstance(self.contact_ticks,int) or self.contact_ticks<1:
            raise ValueError('contact_ticks must be a positive integer')
        if not .02<=self.preshape_aperture_m<=.06: raise ValueError('Invalid handle pre-shape aperture')
        if not 0<self.preshape_closure_command<1: raise ValueError('Invalid handle pre-shape command')
        if self.physics_timestep_s not in (.0005,.001,.002): raise ValueError('Unsupported physics timestep')
        if self.physics_integrator not in ('Euler','implicitfast'): raise ValueError('Unsupported physics integrator')
        if self.thin_contact_timeconstant_s < 2*self.physics_timestep_s:
            raise ValueError('Thin contact time constant must be at least two physics steps')
        if len(self.three_finger_force_targets_n)!=3 or any(v<=0 or not np.isfinite(v) for v in self.three_finger_force_targets_n):
            raise ValueError('Invalid three-finger force targets')
        if not 0 < self.three_finger_force_gain <= .01:
            raise ValueError('Invalid three-finger force gain')
        if not 0 <= self.three_finger_force_deadband_n < min(self.three_finger_force_targets_n):
            raise ValueError('Invalid force deadband')
        for value in (self.post_grasp_velocity_scaling, self.post_grasp_acceleration_scaling):
            if value is not None and not 0 < value <= 1:
                raise ValueError('Invalid post-grasp motion scaling')

    def to_dict(self): return asdict(self)


def build_spatula_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    r=recipe or SpatulaRecipe()
    size=np.asarray(local_size_m,dtype=float)
    if size.shape!=(3,) or not np.isfinite(size).all() or not (.04<size[0]<.06 and .23<size[1]<.27 and .003<size[2]<.008):
        raise ValueError('UNSUPPORTED_SPATULA_GEOMETRY')
    center=np.asarray(T_WB)@transform(center_in_body_m,rotation=np.eye(3))
    # Asset +Y is the blade, -Y is the handle. Approach the thin handle from
    # above, with opposed fingers closing across its narrow width.
    rotation=Rotation.from_euler('y',r.tilt_deg,degrees=True).as_matrix()@np.diag([1.,-1.,-1.])
    relative=transform([r.lateral_offset_m,r.handle_fraction*size[1],r.height_offset_m],rotation=rotation)
    grasp=center@relative
    pre=grasp.copy(); pre[:3,3]-=grasp[:3,2]*r.approach_distance_m
    lift=grasp.copy(); lift[2,3]+=r.lift_distance_m
    return {'T_WC':center,'T_CG':relative,'PRE_GRASP':pre,'GRASP':grasp,'LIFT':lift}


def grasp_spatula(context,object_id='spatula',recipe=None):
    if object_id!='spatula': raise ValueError('This recipe supports the C1_2 spatula')
    return context.execute_spatula(recipe or SpatulaRecipe())
