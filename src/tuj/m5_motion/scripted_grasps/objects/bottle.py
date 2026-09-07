"""C1_2 bottle side grasp in its geometric-center frame; no learned inference."""
from dataclasses import dataclass, asdict
import numpy as np
from tuj.m5_motion.scripted_grasps.frames import transform


@dataclass(frozen=True)
class BottleRecipe:
    recipe_id: str = 'bottle_3f_side_center_v1'
    ee_id: str = '3F'
    model_class: str = 'JacoThreeFingerDexterousGripper'
    # +X approach from the robot-facing side; x/y/z are object-center axes.
    approach_azimuth_deg: float = 0.
    roll_deg: float = 0.
    depth_offset_m: float = 0.
    # Center the body height and compensate the Jaco thumb/pair asymmetry.
    lateral_offset_m: float = -.003
    height_offset_m: float = 0.
    approach_distance_m: float = .12
    lift_distance_m: float = .15
    arm_kp: float = 150.
    close_duration_s: float = 2.
    settle_s: float = 1.
    hold_s: float = 5.
    joint_speed_rad_s: float = .35
    cartesian_speed_m_s: float = .04
    minimum_lift_m: float = .10
    maximum_slip_m: float = .005
    maximum_slip_deg: float = 5.
    contact_ticks: int = 5

    def __post_init__(self):
        values = asdict(self)
        for name,value in values.items():
            if isinstance(value,(float,int)) and not np.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if self.ee_id != '3F' or self.model_class != 'JacoThreeFingerDexterousGripper':
            raise ValueError('UNSUPPORTED_EE: bottle side grasp requires Jaco 3F')
        for name in ('approach_distance_m','lift_distance_m','arm_kp','close_duration_s',
                     'settle_s','hold_s','joint_speed_rad_s','cartesian_speed_m_s',
                     'minimum_lift_m','maximum_slip_m','maximum_slip_deg'):
            if values[name] <= 0:
                raise ValueError(f'{name} must be positive')
        if not isinstance(self.contact_ticks,int) or self.contact_ticks < 1:
            raise ValueError('contact_ticks must be a positive integer')
        if abs(self.height_offset_m) > .04 or abs(self.depth_offset_m) > .04 or abs(self.lateral_offset_m) > .025:
            raise ValueError('Offset outside bottle body calibration range')

    def to_dict(self):
        return asdict(self)


def build_bottle_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    r = recipe or BottleRecipe()
    size = np.asarray(local_size_m,dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or not (.035 < min(size[:2]) <= max(size[:2]) < .07 and .11 < size[2] < .16):
        raise ValueError('UNSUPPORTED_BOTTLE_GEOMETRY')
    angle = np.deg2rad(r.approach_azimuth_deg)
    z = np.array([np.cos(angle),np.sin(angle),0.])
    y = np.array([0.,0.,1.])
    x = np.cross(y,z)
    roll = np.deg2rad(r.roll_deg)
    rotation = np.column_stack((np.cos(roll)*x+np.sin(roll)*y,
                               -np.sin(roll)*x+np.cos(roll)*y,z))
    center = np.asarray(T_WB) @ transform(center_in_body_m,rotation=np.eye(3))
    offset = z*r.depth_offset_m + x*r.lateral_offset_m + [0.,0.,r.height_offset_m]
    relative = transform(offset,rotation=rotation)
    grasp = center @ relative
    pre = grasp.copy()
    pre[:3,3] -= grasp[:3,2]*r.approach_distance_m
    lift = grasp.copy()
    lift[2,3] += r.lift_distance_m
    return {'T_WC':center,'T_CG':relative,'PRE_GRASP':pre,'GRASP':grasp,'LIFT':lift}


def grasp_bottle(context, object_id='bottle', recipe=None):
    if object_id != 'bottle':
        raise ValueError('This recipe supports the C1_2 bottle')
    return context.execute_bottle(recipe or BottleRecipe())
