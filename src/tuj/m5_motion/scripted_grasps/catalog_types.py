"""Object-centered recipes shared by the additional twelve grasp functions."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from tuj.m5_motion.scripted_grasps.frames import transform


@dataclass(frozen=True)
class CatalogRecipe:
    object_id: str
    task_id: str
    ee_id: str
    expected_size_m: tuple
    offset_fraction: tuple = (0.,0.,0.)
    offset_m: tuple = (0.,0.,0.)
    rotation_xyz_deg: tuple = (0.,0.,0.)
    # Body-local normalized bbox region; only contacts here count as grasping.
    contact_region_min: tuple = (-.6,-.6,-.6)
    contact_region_max: tuple = (.6,.6,.6)
    preshape_aperture_m: float = .05
    preshape_closure_command: float = .16
    approach_distance_m: float = .12
    lift_distance_m: float = .18
    arm_kp: float = 150.
    post_grasp_arm_kp: float | None = None
    close_duration_s: float = 4.
    closure_kp: float = 20.
    physics_timestep_s: float = .001
    physics_integrator: str = 'implicitfast'
    thin_contact_timeconstant_s: float = .004
    two_finger_parallel_linkage: bool = True
    two_finger_force_target_n: float = 5.
    two_finger_force_gain: float = .002
    three_finger_force_targets_n: tuple = (6.,3.,3.)
    three_finger_force_gain: float = .002
    prelift_stabilization_s: float = .5
    settle_s: float = 1.
    hold_s: float = 5.
    joint_speed_rad_s: float = .45
    cartesian_speed_m_s: float = .05
    minimum_lift_m: float = .10
    maximum_slip_m: float = .005
    maximum_slip_deg: float = 5.
    contact_ticks: int = 5
    maximum_joint_limit_error_rad: float = .01
    suction_command: float = 1.

    @property
    def model_class(self):
        return {'2F':'Robotiq85Gripper','3F':'JacoThreeFingerDexterousGripper','vac':'VacuumGripper'}[self.ee_id]

    @property
    def recipe_id(self):
        version='v2_attach' if self.ee_id=='vac' else 'v1'
        return f'{self.object_id}_{self.ee_id.lower()}_center_{version}'

    def __post_init__(self):
        if self.ee_id not in {'2F','3F','vac'}: raise ValueError('UNSUPPORTED_EE')
        if self.task_id not in {'c1_1','c1_2','c2_1','c2_2','c4_2'}: raise ValueError('UNSUPPORTED_TASK')
        for name in ('expected_size_m','offset_fraction','offset_m','rotation_xyz_deg','contact_region_min','contact_region_max'):
            value=np.asarray(getattr(self,name),dtype=float)
            if value.shape!=(3,) or not np.isfinite(value).all(): raise ValueError(f'Invalid {name}')
        if np.any(np.asarray(self.expected_size_m)<=0): raise ValueError('Invalid expected geometry')
        if np.any(np.asarray(self.contact_region_min)>=self.contact_region_max): raise ValueError('Invalid contact region')
        for name,value in asdict(self).items():
            if isinstance(value,(float,int)) and not np.isfinite(value): raise ValueError(f'Invalid {name}')
        for name in ('approach_distance_m','lift_distance_m','hold_s','settle_s','arm_kp','closure_kp','close_duration_s',
                     'maximum_slip_m','maximum_slip_deg','minimum_lift_m','two_finger_force_target_n'):
            if getattr(self,name)<=0: raise ValueError(f'{name} must be positive')
        if not 0<self.maximum_joint_limit_error_rad<=.01: raise ValueError('Invalid joint tolerance')
        if self.post_grasp_arm_kp is not None and not 0 < self.post_grasp_arm_kp <= 300:
            raise ValueError('Invalid post-grasp arm gain')
        if not 0<=self.suction_command<=1: raise ValueError('Invalid suction command')
        if self.ee_id=='vac' and self.suction_command<.5: raise ValueError('Vacuum attachment requires suction command >= .5')
        if self.physics_timestep_s not in (.0005,.001,.002): raise ValueError('Invalid timestep')
        if self.physics_integrator!='implicitfast': raise ValueError('Invalid integrator')
        if not isinstance(self.contact_ticks,int) or self.contact_ticks<1: raise ValueError('Invalid contact ticks')

    def to_dict(self):
        result={**asdict(self),'model_class':self.model_class,'recipe_id':self.recipe_id}
        if self.ee_id=='vac':result['vacuum_attachment_mode']='KINEMATIC'
        return result


def build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe):
    body=np.asarray(T_WB,dtype=float); size=np.asarray(local_size_m,dtype=float)
    if body.shape!=(4,4) or not np.isfinite(body).all(): raise ValueError('Invalid body pose')
    if not np.allclose(body[3],[0,0,0,1]) or not np.allclose(body[:3,:3].T@body[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(body[:3,:3])<.999:
        raise ValueError('Invalid rigid pose')
    expected=np.asarray(recipe.expected_size_m)
    if size.shape!=(3,) or not np.isfinite(size).all() or np.any(np.abs(size-expected)>np.maximum(.004,expected*.12)):
        raise ValueError('UNSUPPORTED_OBJECT_GEOMETRY')
    T_BC=transform(center_in_body_m,rotation=np.eye(3))
    T_WC=body@T_BC
    rotation=Rotation.from_euler('xyz',recipe.rotation_xyz_deg,degrees=True).as_matrix()@np.diag([1.,-1.,-1.])
    T_CG=transform(size*np.asarray(recipe.offset_fraction)+np.asarray(recipe.offset_m),rotation=rotation)
    grasp=T_WC@T_CG
    pre=grasp.copy();pre[:3,3]-=grasp[:3,2]*recipe.approach_distance_m
    lift=grasp.copy();lift[2,3]+=recipe.lift_distance_m
    return {'T_BC':T_BC,'T_WC':T_WC,'T_CG':T_CG,'PRE_GRASP':pre,'GRASP':grasp,'LIFT':lift}


def dispatch_grasp(context,object_id,recipe,expected_id):
    if object_id!=expected_id or recipe.object_id!=expected_id: raise ValueError('WRONG_TARGET_OBJECT')
    return context.execute_object(recipe)
