"""Scene-relative spoon handle grasps, in the geometric-center frame."""
from dataclasses import dataclass, asdict, replace
import numpy as np
from scipy.spatial.transform import Rotation
from tuj.m5_motion.scripted_grasps.frames import transform


# Compiled C3_2 spoon AABB (spoon_a / spoon_b). Used only when asset='c3_2'.
C3_2_SPOON_EXPECTED_SIZE_M = (0.044984882, 0.137213218, 0.023900813)


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
    three_finger_force_deadband_n: float = .5
    three_finger_joint_limit_timeconstant_s: float = .002
    three_finger_joint_limit_impedance: tuple = (.9999,.9999,.001,.5,2.)
    three_finger_joint_armature_kg_m2: float = .0001
    three_finger_hold_close_margin: float = .05
    three_finger_hold_open_margin: float = .005
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
    # None keeps the legacy C1_2 / C3_1 hard size gate in build_spoon_targets.
    expected_size_m: tuple | None = None
    # When True, 3F CLOSE/HOLD accept thumb+index/pinky pinch (thin utensils).
    thin_handle_pinch: bool = False
    # Freeze finger commands after pinch (spatula pattern); avoids force-servo unload.
    hold_finger_positions: bool = False

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
        if self.thin_handle_pinch and self.ee_id!='3F':
            raise ValueError('thin_handle_pinch requires 3F')
        if self.hold_finger_positions and self.ee_id!='3F':
            raise ValueError('hold_finger_positions requires 3F')
        for key in ('approach_distance_m','lift_distance_m','arm_kp','close_duration_s','closure_kp','preshape_aperture_m','thin_contact_timeconstant_s','settle_s','hold_s','prelift_stabilization_s',
                    'joint_speed_rad_s','cartesian_speed_m_s','minimum_lift_m','maximum_slip_m','maximum_slip_deg'):
            if values[key]<=0: raise ValueError(f'{key} must be positive')
        if not isinstance(self.contact_ticks,int) or self.contact_ticks<1:
            raise ValueError('contact_ticks must be a positive integer')
        if not isinstance(self.thin_handle_pinch,bool) or not isinstance(self.hold_finger_positions,bool):
            raise ValueError('Invalid boolean recipe flag')
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
        if not 0 <= self.three_finger_force_deadband_n < min(
                self.three_finger_force_targets_n):
            raise ValueError('Invalid three-finger force deadband')
        if not 0 < self.three_finger_joint_limit_timeconstant_s <= .01:
            raise ValueError('Invalid three-finger joint-limit time constant')
        if len(self.three_finger_joint_limit_impedance)!=5 or not all(
                np.isfinite(v) for v in self.three_finger_joint_limit_impedance):
            raise ValueError('Invalid three-finger joint-limit impedance')
        if not 0 <= self.three_finger_joint_armature_kg_m2 <= .01:
            raise ValueError('Invalid three-finger joint armature')
        if not 0 < self.three_finger_hold_close_margin <= .1:
            raise ValueError('Invalid three-finger hold close margin')
        if not 0 <= self.three_finger_hold_open_margin <= .02:
            raise ValueError('Invalid three-finger hold open margin')
        if self.two_finger_force_target_n<=0 or not 0<self.two_finger_force_gain<=.01:
            raise ValueError('Invalid two-finger force feedback')
        if not 0 < self.maximum_joint_limit_error_rad <= .01:
            raise ValueError('Joint-limit residual tolerance must be at most .01 rad')
        if self.expected_size_m is not None:
            size=np.asarray(self.expected_size_m,dtype=float)
            if size.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0):
                raise ValueError('Invalid expected spoon geometry')

    def to_dict(self): return asdict(self)


def spoon_recipe(ee_id='2F', environment=None, *, asset='default'):
    """Return the hand-specific pose; validation records identify passed runs.

    ``asset='default'`` preserves C1_2 / C3_1 geometry gates. ``asset='c3_2'``
    selects the breakfast-tray spoon AABB and 3F handle station.
    ``environment`` selects object-sorting 3F calibrations from origin/main.
    """
    if environment is not None and str(environment).startswith('C3_2'):
        asset = 'c3_2'
    if asset == 'c3_2':
        if ee_id == '2F':
            return replace(
                spoon_recipe('2F'),
                recipe_id='spoon_2f_c3_2_handle_center_v1_experimental',
                expected_size_m=C3_2_SPOON_EXPECTED_SIZE_M,
                handle_fraction=-.20,
            )
        if ee_id != '3F':
            raise ValueError('UNSUPPORTED_EE: c3_2 spoon supports 2F/3F recipes')
        # c3_2: yaw-90 thumb+index pinch on the single handle collision mesh.
        # Full 3F enclosure is not achievable on this thin asset; see thin_handle_pinch.
        return replace(
            spoon_recipe('3F'),
            recipe_id='spoon_3f_c3_2_handle_center_v2',
            expected_size_m=C3_2_SPOON_EXPECTED_SIZE_M,
            handle_fraction=-.15,
            height_offset_m=.014,
            lateral_offset_m=-.004,
            tilt_deg=5.,
            handle_roll_deg=0.,
            approach_yaw_deg=90.,
            preshape_aperture_m=.025,
            # Near-open approach: mid-close dips tips into the island.
            preshape_closure_command=.05,
            thin_handle_pinch=True,
            hold_finger_positions=True,
            three_finger_force_targets_n=(3.0, 1.5, 0.5),
            # Keep deadband strictly below the lowest finger target (pinky 0.5).
            three_finger_force_deadband_n=0.25,
        )
    if ee_id=='2F': return SpoonRecipe()
    if ee_id=='3F':
        recipe=replace(SpoonRecipe(),recipe_id='spoon_3f_handle_center_v2',
            ee_id='3F',model_class='JacoThreeFingerDexterousGripper',lateral_offset_m=-.002,height_offset_m=.027,
            handle_fraction=-.18,tilt_deg=0.,handle_roll_deg=0.,close_duration_s=2.,
            preshape_aperture_m=.025,preshape_closure_command=.5,
            two_finger_parallel_linkage=False)
        if environment in {'C2_1_ObjectSorting','C3_1_ObjectSorting'}:
            recipe=replace(recipe,
                recipe_id='spoon_3f_object_sorting_handle_center_v5',
                handle_fraction=-.08,lateral_offset_m=-.0035,
                three_finger_hold_close_margin=.1)
        return recipe
    raise ValueError('UNSUPPORTED_EE: choose 2F or 3F')


def build_spoon_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    r=recipe or SpoonRecipe()
    size=np.asarray(local_size_m,dtype=float)
    if size.shape!=(3,) or not np.isfinite(size).all():
        raise ValueError('UNSUPPORTED_SPOON_GEOMETRY')
    if r.expected_size_m is not None:
        expected=np.asarray(r.expected_size_m,dtype=float)
        if np.any(np.abs(size-expected)>np.maximum(.004,expected*.12)):
            raise ValueError('UNSUPPORTED_SPOON_GEOMETRY')
    elif not (.05<size[0]<.065 and .16<size[1]<.185 and .025<size[2]<.035):
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


def grasp_spoon(context,object_id=None,recipe=None):
    from tuj.m5_motion.scripted_grasps.catalog_types import _instance_matches_type
    object_id = object_id or getattr(context, 'object_id', 'spoon')
    if not _instance_matches_type(object_id, 'spoon'):
        raise ValueError('This recipe supports the spoon asset (including spoon_a/spoon_b)')
    return context.execute_spoon(recipe or getattr(context,'recipe',None) or SpoonRecipe())
