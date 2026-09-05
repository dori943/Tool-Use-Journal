"""Handle grasp for the C1_2 spoon, in the geometric-center frame."""
from dataclasses import dataclass, asdict, replace
import numpy as np
from scipy.spatial.transform import Rotation
from tuj.m5_motion.scripted_grasps.frames import transform


@dataclass(frozen=True)
class SpoonRecipe:
    recipe_id: str = 'spoon_2f_handle_center_v2'
    ee_id: str = '2F'
    model_class: str = 'Robotiq85Gripper'
    handle_fraction: float = -.08
    lateral_offset_m: float = 0.
    height_offset_m: float = .015
    tilt_deg: float = .8
    handle_roll_deg: float = -8.8
    approach_yaw_deg: float = 0.
    approach_distance_m: float = .12
    lift_distance_m: float = .15
    arm_kp: float = 150.
    close_duration_s: float = 4.
    closure_kp: float = 20.
    preshape_aperture_m: float = .020
    preshape_closure_command: float = .625
    physics_timestep_s: float = .001
    physics_integrator: str = 'implicitfast'
    thin_contact_timeconstant_s: float = .004
    three_finger_force_targets_n: tuple = (3.,1.5,1.5)
    three_finger_force_gain: float = .002
    prelift_stabilization_s: float = .5
    two_finger_parallel_linkage: bool = True
    two_finger_force_target_n: float = 5.
    two_finger_force_gain: float = .002
    settle_s: float = 1.
    hold_s: float = 5.
    joint_speed_rad_s: float = .35
    cartesian_speed_m_s: float = .04
    minimum_lift_m: float = .10
    maximum_slip_m: float = .005
    maximum_slip_deg: float = 5.
    contact_ticks: int = 5
    maximum_joint_limit_error_rad: float = .01

    def __post_init__(self):
        values=asdict(self)
        for key,value in values.items():
            if isinstance(value,(float,int)) and not np.isfinite(value):
                raise ValueError(f'{key} must be finite')
        models={'2F':'Robotiq85Gripper','3F':'JacoThreeFingerDexterousGripper'}
        if models.get(self.ee_id)!=self.model_class:
            raise ValueError('UNSUPPORTED_EE: spoon recipe requires matching 2F or 3F')
        if not -.38 <= self.handle_fraction <= -.08:
            raise ValueError('Target must lie within the handle')
        if abs(self.lateral_offset_m)>.025 or not -.005<=self.height_offset_m<=.07 or abs(self.tilt_deg)>75 or abs(self.handle_roll_deg)>30 or abs(self.approach_yaw_deg)>180:
            raise ValueError('Offset outside spoon calibration range')
        for key in ('approach_distance_m','lift_distance_m','arm_kp','close_duration_s','closure_kp','preshape_aperture_m','thin_contact_timeconstant_s','settle_s','hold_s','prelift_stabilization_s',
                    'joint_speed_rad_s','cartesian_speed_m_s','minimum_lift_m','maximum_slip_m','maximum_slip_deg'):
            if values[key]<=0: raise ValueError(f'{key} must be positive')
        if not isinstance(self.contact_ticks,int) or self.contact_ticks<1:
            raise ValueError('contact_ticks must be a positive integer')
        if not .008<=self.preshape_aperture_m<=.06: raise ValueError('Invalid handle pre-shape aperture')
        if not 0<self.preshape_closure_command<1: raise ValueError('Invalid handle pre-shape command')
        if self.physics_timestep_s not in (.0005,.001,.002): raise ValueError('Unsupported physics timestep')
        if self.physics_integrator not in ('Euler','implicitfast'): raise ValueError('Unsupported physics integrator')
        if self.thin_contact_timeconstant_s < 2*self.physics_timestep_s:
            raise ValueError('Thin contact time constant must be at least two physics steps')
        if len(self.three_finger_force_targets_n)!=3 or any(v<=0 or not np.isfinite(v) for v in self.three_finger_force_targets_n):
            raise ValueError('Invalid three-finger force targets')
        if not 0 < self.three_finger_force_gain <= .01:
            raise ValueError('Invalid three-finger force gain')
        if self.two_finger_force_target_n<=0 or not 0<self.two_finger_force_gain<=.01:
            raise ValueError('Invalid two-finger force feedback')
        if not 0 < self.maximum_joint_limit_error_rad <= .01:
            raise ValueError('Joint-limit residual tolerance must be at most .01 rad')

    def to_dict(self): return asdict(self)


def spoon_recipe(ee_id='2F'):
    """Return the hand-specific pose; validation records identify passed runs."""
    if ee_id=='2F': return SpoonRecipe()
    if ee_id=='3F':
        return replace(SpoonRecipe(),recipe_id='spoon_3f_handle_center_v2_unvalidated',
            ee_id='3F',model_class='JacoThreeFingerDexterousGripper',lateral_offset_m=-.002,height_offset_m=.027,
            handle_fraction=-.18,preshape_aperture_m=.025,preshape_closure_command=.5,
            two_finger_parallel_linkage=False)
    raise ValueError('UNSUPPORTED_EE: choose 2F or 3F')


def build_spoon_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    r=recipe or SpoonRecipe()
    size=np.asarray(local_size_m,dtype=float)
    if size.shape!=(3,) or not np.isfinite(size).all() or not (.05<size[0]<.065 and .16<size[1]<.185 and .025<size[2]<.035):
        raise ValueError('UNSUPPORTED_SPOON_GEOMETRY')
    center=np.asarray(T_WB)@transform(center_in_body_m,rotation=np.eye(3))
    # Asset +Y is the bowl, -Y is the handle. Approach the narrow handle from
    # above, with opposed fingers closing across its narrow width.
    rotation=(Rotation.from_euler('x',r.handle_roll_deg,degrees=True).as_matrix()
        @Rotation.from_euler('z',r.approach_yaw_deg,degrees=True).as_matrix()
        @Rotation.from_euler('y',r.tilt_deg,degrees=True).as_matrix()@np.diag([1.,-1.,-1.]))
    relative=transform([r.lateral_offset_m,r.handle_fraction*size[1],r.height_offset_m],rotation=rotation)
    grasp=center@relative
    pre=grasp.copy(); pre[:3,3]-=grasp[:3,2]*r.approach_distance_m
    lift=grasp.copy(); lift[2,3]+=r.lift_distance_m
    return {'T_WC':center,'T_CG':relative,'PRE_GRASP':pre,'GRASP':grasp,'LIFT':lift}


def grasp_spoon(context,object_id='spoon',recipe=None):
    if object_id!='spoon': raise ValueError('This recipe supports the C1_2 spoon')
    return context.execute_spoon(recipe or getattr(context,'recipe',None) or SpoonRecipe())
