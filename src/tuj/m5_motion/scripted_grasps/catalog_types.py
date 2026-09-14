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
    linear_lift_start: bool = False
    arm_kp: float = 150.
    post_grasp_arm_kp: float | None = None
    close_duration_s: float = 4.
    closure_kp: float = 20.
    physics_timestep_s: float = .001
    physics_integrator: str = 'implicitfast'
    thin_contact_timeconstant_s: float = .004
    # 충돌 geom 의 solimp. 자산이 선언한 solimp 는 contype=0 인 visual geom 에만
    # 붙어 있어서 실제로 충돌하는 geom 은 MuJoCo 기본값(.9 .95 .001)으로 떨어진다.
    # 그 물렁한 접촉 때문에 가벼운 슬라이스가 흡착(gain 80 N)에 컵 안으로 끌려
    # 들어간다. cheese 자산만 충돌 geom 에 .998 을 제대로 달고 있다.
    # None 이면 기존 동작 그대로다.
    thin_contact_solimp: tuple | None = None
    # 흡착컵 geom 의 margin/gap. 자산 기본값은 10 mm 인데, MuJoCo 흡착은 margin
    # 안의 모든 접촉에 작용하므로 얇은 슬라이스를 집으면 그 아래 받침까지 같이
    # 빨아올린다 (토마토 4.74 mm 를 .5 mm 눌러 앉으면 도마 윗면이 4.2 mm,
    # 터키 2.52 mm 면 2.0 mm 거리다). None 이면 자산 값 그대로다.
    vacuum_cup_margin_m: float | None = None
    two_finger_parallel_linkage: bool = True
    two_finger_force_target_n: float = 5.
    two_finger_force_gain: float = .002
    three_finger_force_targets_n: tuple = (6.,3.,3.)
    three_finger_force_gain: float = .002
    # Force-integrator deadband (N) shared by the 3F force hold
    # (update_three_finger_commands).  Upstream added this to the spoon recipe
    # but not to CatalogRecipe, so a catalog 3F grasp (e.g. bread) raised
    # AttributeError at CLOSE.  Same default as the spoon recipe.
    three_finger_force_deadband_n: float = .5
    prelift_stabilization_s: float = .5
    settle_s: float = 1.
    hold_s: float = 5.
    joint_speed_rad_s: float = .45
    cartesian_speed_m_s: float = .05
    minimum_lift_m: float = .10
    maximum_slip_m: float = .005
    maximum_slip_deg: float = 5.
    contact_ticks: int = 5
    # Minimum antipodal-normal opposition for the 2F contact gate (ready()).
    # 1.0 = the two finger normals point exactly at each other.  The default
    # 0.5 suits chunky objects, but a very thin flat tool gripped across its
    # width contacts only a ~2-3 mm-tall edge strip, so a firm squeeze rolls the
    # blade slightly and the normals splay (opposition falls to ~0.1) even while
    # both pads still clamp it with several N over a wide span.  A thin tool
    # lowers this so the real, holding grip is not rejected as CONTACT_LOST.
    minimum_normal_opposition: float = .5
    # Optional (sliding, torsional, rolling) pad friction applied to the finger
    # contact geoms with MuJoCo's 6-D contact model before CLOSE.  Default None
    # leaves the model's pad friction untouched (existing recipes unchanged); a
    # very thin flat tool sets a high value so the light blade rides up with the
    # pad on lift instead of sliding off the ~2.5 mm edge.
    fingerpad_friction: tuple | None = None
    minimum_vacuum_contact_count: int = 3
    maximum_vacuum_attach_penetration_m: float = .002
    maximum_support_separation_penetration_m: float = .002
    maximum_joint_limit_error_rad: float = .01
    suction_command: float = 1.
    # Thin utensil handles: accept thumb+index/pinky pinch instead of full 3F.
    thin_handle_pinch: bool = False
    hold_finger_positions: bool = False
    # After a failed thin-handle CLOSE, retry GRASP+CLOSE at offset_m.x ± k*step.
    thin_handle_close_retries: int = 0
    thin_handle_lateral_retry_m: float = 0.002
    # Minimum simultaneous cup-object contacts for the vacuum contact gate.
    # A multi-finger grasp naturally makes several contacts, but a single
    # suction cup pressed flat on a flat surface makes only ONE MuJoCo contact
    # point, so the historical hard-coded >=3 could never arm for a flat disc
    # (plate: contact_count=1 with suction_alignment=0.999 and 40 N of force).
    # Default stays 3 so existing recipes are unchanged; a flat single-cup
    # target sets this to 1.
    vacuum_min_contacts: int = 3
    finger_attachment_policy: str = 'FREE_HOLD_THEN_ATTACH'
    open_hand_clearance_axis: tuple | None = None
    contact_region_endpoint_policy: str | None = None

    @property
    def model_class(self):
        return {'2F':'Robotiq85Gripper','3F':'JacoThreeFingerDexterousGripper','vac':'VacuumGripper'}[self.ee_id]

    @property
    def recipe_id(self):
        version='v2_attach' if self.ee_id=='vac' else 'v1'
        if self.finger_attachment_policy == 'STABLE_CONTACT':version='v2_contact_attach'
        return f'{self.object_id}_{self.ee_id.lower()}_center_{version}'

    def __post_init__(self):
        if self.contact_region_endpoint_policy not in {None, 'MIN_LONG_AXIS', 'MAX_LONG_AXIS'}:
            raise ValueError('Invalid contact region endpoint policy')
        if self.contact_region_endpoint_policy is not None and self.open_hand_clearance_axis is not None:
            raise ValueError('Choose one grasp clearance proposal policy')
        if self.open_hand_clearance_axis is not None:
            axis = np.asarray(self.open_hand_clearance_axis, dtype=float)
            if (self.ee_id == 'vac' or axis.shape != (3,) or not np.isfinite(axis).all()
                    or not np.isclose(np.linalg.norm(axis), 1., atol=1e-9, rtol=0.)):
                raise ValueError('Open hand clearance requires a unit body axis and finger EE')
        if self.finger_attachment_policy not in {'FREE_HOLD_THEN_ATTACH', 'STABLE_CONTACT'}:
            raise ValueError('Invalid finger attachment policy')
        if self.finger_attachment_policy == 'STABLE_CONTACT' and self.ee_id != '2F':
            raise ValueError('Stable bilateral contact attachment requires 2F')
        if self.ee_id not in {'2F','3F','vac'}: raise ValueError('UNSUPPORTED_EE')
        if self.task_id not in {'c1_1','c1_2','c2_1','c2_2','c3_1','c3_2','c4_2'}: raise ValueError('UNSUPPORTED_TASK')
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
        if self.thin_handle_pinch and self.ee_id!='3F':
            raise ValueError('thin_handle_pinch requires 3F')
        if self.hold_finger_positions and self.ee_id!='3F':
            raise ValueError('hold_finger_positions requires 3F')
        if self.ee_id=='3F':
            targets=np.asarray(self.three_finger_force_targets_n,dtype=float)
            if targets.shape!=(3,) or np.any(targets<=0) or not np.isfinite(targets).all():
                raise ValueError('Invalid three-finger force targets')
            if not 0<=self.three_finger_force_deadband_n<float(np.min(targets)):
                raise ValueError('Invalid three-finger force deadband')
        if self.physics_timestep_s not in (.0005,.001,.002): raise ValueError('Invalid timestep')
        if self.physics_integrator!='implicitfast': raise ValueError('Invalid integrator')
        if not isinstance(self.contact_ticks,int) or self.contact_ticks<1: raise ValueError('Invalid contact ticks')
        if not isinstance(self.thin_handle_pinch,bool) or not isinstance(self.hold_finger_positions,bool):
            raise ValueError('Invalid boolean recipe flag')
        if not isinstance(self.thin_handle_close_retries,int) or self.thin_handle_close_retries<0:
            raise ValueError('Invalid thin_handle_close_retries')
        if self.thin_handle_close_retries and not self.thin_handle_pinch:
            raise ValueError('thin_handle_close_retries requires thin_handle_pinch')
        if not (self.thin_handle_lateral_retry_m>0 and np.isfinite(self.thin_handle_lateral_retry_m)):
            raise ValueError('Invalid thin_handle_lateral_retry_m')
        if self.thin_handle_lateral_retry_m>.01:
            raise ValueError('thin_handle_lateral_retry_m out of range')
        if self.vacuum_cup_margin_m is not None:
            if self.ee_id!='vac': raise ValueError('Cup margin requires the vacuum EE')
            if not 0<self.vacuum_cup_margin_m<.01: raise ValueError('Invalid cup margin')
        if self.thin_contact_solimp is not None:
            if len(self.thin_contact_solimp)!=3: raise ValueError('Invalid contact solimp')
            if not all(0<v<1 for v in self.thin_contact_solimp[:2]): raise ValueError('Invalid contact solimp')
            if self.thin_contact_solimp[2]<=0: raise ValueError('Invalid contact solimp')
        if not isinstance(self.linear_lift_start,bool): raise ValueError('Invalid lift easing policy')
        if self.physics_timestep_s not in (.0005,.001,.002): raise ValueError('Invalid timestep')
        if self.physics_integrator!='implicitfast': raise ValueError('Invalid integrator')
        if not isinstance(self.contact_ticks,int) or self.contact_ticks<1: raise ValueError('Invalid contact ticks')
        if not isinstance(self.minimum_vacuum_contact_count,int) or self.minimum_vacuum_contact_count<1:
            raise ValueError('Invalid vacuum contact count')
        if not 0 < self.maximum_vacuum_attach_penetration_m <= .004:
            raise ValueError('Invalid vacuum attachment penetration')
        if not 0 < self.maximum_support_separation_penetration_m <= .006:
            raise ValueError('Invalid support separation penetration')

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
    if recipe.contact_region_endpoint_policy is not None:
        if not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError('Invalid endpoint grasp geometry')
        axis = int(np.argmax(size))
        region = (recipe.contact_region_min if recipe.contact_region_endpoint_policy == 'MIN_LONG_AXIS'
                  else recipe.contact_region_max)
        T_CG[axis, 3] = size[axis] * region[axis]
    grasp=T_WC@T_CG
    pre=grasp.copy();pre[:3,3]-=grasp[:3,2]*recipe.approach_distance_m
    lift=grasp.copy();lift[2,3]+=recipe.lift_distance_m
    return {'T_BC':T_BC,'T_WC':T_WC,'T_CG':T_CG,'PRE_GRASP':pre,'GRASP':grasp,'LIFT':lift}


def build_collision_surface_vacuum_targets(T_WB, center_in_body_m,
        local_size_m, recipe, object_record):
    """Retarget a catalog vacuum grasp to the cup-footprint collision crown.

    Reuses the generic M5 suction-surface solver.  No seating offset is added:
    the returned GRASP lies on the approach-facing target collision surface.
    """
    targets = build_catalog_targets(
        T_WB, center_in_body_m, local_size_m, recipe)
    if recipe.ee_id != 'vac':
        return targets
    from tuj.m5_motion.grasp_geometry import (
        _approach_facing_collision_surface_offset_from_center_m,
    )
    center = np.asarray(center_in_body_m, dtype=float)
    nominal = center + np.asarray(targets['T_CG'][:3, 3], dtype=float)
    approach = np.array([0., 0., 1.])
    surface = _approach_facing_collision_surface_offset_from_center_m(
        object_record, approach, nominal)
    if surface is None:
        return targets
    candidate = np.asarray(surface.grasp_local_m, dtype=float)
    lateral = candidate - center
    lateral -= approach * float(lateral @ approach)
    contact_relative = lateral + approach * float(
        surface.surface_offset_from_center_m)
    targets['T_CG'] = transform(contact_relative, rotation=targets['T_CG'][:3, :3])
    targets['GRASP'] = targets['T_WC'] @ targets['T_CG']
    targets['PRE_GRASP'] = targets['GRASP'].copy()
    targets['PRE_GRASP'][:3, 3] -= (
        targets['GRASP'][:3, 2] * recipe.approach_distance_m)
    targets['LIFT'] = targets['GRASP'].copy()
    targets['LIFT'][2, 3] += recipe.lift_distance_m
    targets['contact_surface_source'] = surface.source
    return targets


def _instance_matches_type(object_id,expected_id):
    """Exact type id, or ``type_a`` / ``type_b`` multi-instance scene ids."""
    if object_id==expected_id:
        return True
    return len(object_id)==len(expected_id)+2 and object_id.startswith(expected_id+'_') and object_id[-1] in 'ab' and object_id[-2]=='_'


def dispatch_grasp(context,object_id,recipe,expected_id):
    if recipe.object_id!=expected_id or not _instance_matches_type(object_id,expected_id):
        raise ValueError('WRONG_TARGET_OBJECT')
    return context.execute_object(recipe)
