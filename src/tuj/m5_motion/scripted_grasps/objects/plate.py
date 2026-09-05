"""Plate rim recipe. No learned model, mass, material or friction input."""
from dataclasses import asdict, dataclass
import numpy as np
from tuj.m5_motion.scripted_grasps.frames import transform


@dataclass(frozen=True)
class PlateRecipe:
    recipe_id: str = "plate_2f_calibrated_center_v5"
    ee_id: str = "2F"
    model_class: str = "Robotiq85Gripper"
    rim_sector: int = 0
    # Frozen at the successful M1 calibration capture. A changed visible bbox
    # of this same asset must not move the stored object-relative grasp point.
    calibrated_local_bbox_size_m: tuple = (0.1771523276287003, 0.18040843091814424, 0.009812861302886552)
    asset_dimensions_m: tuple = (0.1823341623518599, 0.18183352035820532, 0.011091216733562866)
    # M1 sees a partial bbox. This calibration restores the reference rim
    # anchor (14 mm inset from the complete asset bbox) for this plate.
    radial_inset_m: float = 0.0114
    vertical_offset_m: float = 0.0015
    approach_elevation_deg: float = 45.0
    approach_distance_m: float = 0.08
    lift_distance_m: float = 0.25
    close_hold_s: float = 1.5
    final_hold_s: float = 5.0
    post_lift_settle_s: float = 3.0
    minimum_lift_m: float = 0.075
    max_slip_m: float = 0.005
    max_slip_deg: float = 5.0
    preshape_clearance_m: float = 0.02
    arm_kp: float = 50.0
    closure_kp: float = 50.0
    maximum_closure_kp: float = 70.0
    contact_force_target_n: float = 40.17
    force_feedback_gain: float = 0.2
    maximum_actual_table_penetration_m: float = 0.002
    execution_collision_check_stride: int = 25
    prelift_contact_ticks: int = 5
    prelift_contact_span_m: float = .0035
    prelift_max_wait_s: float = 2.

    def __post_init__(self):
        if not isinstance(self.prelift_contact_ticks,int) or self.prelift_contact_ticks<1:
            raise ValueError('prelift_contact_ticks must be a positive integer')
        if not 0<self.prelift_contact_span_m<.085 or not 0<self.prelift_max_wait_s<=5:
            raise ValueError('Invalid pre-lift contact gate')
        if self.ee_id != "2F" or self.model_class != "Robotiq85Gripper":
            raise ValueError("UNSUPPORTED_RECIPE: expected Robotiq85 2F")
        if self.rim_sector not in range(4):
            raise ValueError("rim_sector must be 0..3")
        for dimensions in (self.calibrated_local_bbox_size_m, self.asset_dimensions_m):
            size = np.asarray(dimensions, dtype=float)
            if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
                raise ValueError('invalid recipe calibration dimensions')
        if not 0 < self.approach_elevation_deg < 90:
            raise ValueError("approach elevation must be between 0 and 90 degrees")
        for name in ['approach_distance_m','lift_distance_m','close_hold_s','final_hold_s',
                     'minimum_lift_m','max_slip_m','max_slip_deg',
                     'preshape_clearance_m','arm_kp','closure_kp',
                     'maximum_closure_kp','contact_force_target_n']:
            value=getattr(self,name)
            if not np.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0<=self.radial_inset_m<0.06 or not 0<=self.vertical_offset_m<=0.04:
            raise ValueError('rim offset outside calibrated geometry bounds')
        if not 0<=self.force_feedback_gain<=1 or self.maximum_closure_kp<self.closure_kp:
            raise ValueError('invalid clamp controller gains')
        if not 0<=self.maximum_actual_table_penetration_m<=0.002:
            raise ValueError('actual table penetration exceeds experiment bound')
        if not isinstance(self.execution_collision_check_stride,int) or not 1<=self.execution_collision_check_stride<=50:
            raise ValueError('execution collision check stride must be 1..50')
        if not np.isfinite(self.post_lift_settle_s) or self.post_lift_settle_s<0:
            raise ValueError('post-lift settling time must be finite and non-negative')

    def to_dict(self):
        return asdict(self)


def build_plate_targets(T_WB, center_in_body_m, local_bbox_size_m, recipe=None):
    """Freeze a centre frame, then retarget complete relative poses to the world.

    Size is an object-local geometric size, never the rotated world AABB size.
    Centre correction is calibrated separately from the model body origin.
    """
    recipe = recipe or PlateRecipe()
    size = np.asarray(local_bbox_size_m, dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Invalid local bbox size")
    if not (0.12 <= min(size[:2]) <= max(size[:2]) <= 0.25 and size[2] < 0.04):
        raise ValueError("UNSUPPORTED_RECIPE: plate dimensions outside initial recipe scope")
    azimuth = [0, 180, 90, -90][recipe.rim_sector]
    angle, elevation = np.deg2rad([azimuth, recipe.approach_elevation_deg])
    radial = np.array([np.cos(angle), np.sin(angle), 0.0])
    approach = np.cos(elevation) * radial + np.array([0, 0, np.sin(elevation)])
    z = -approach
    # Gripper local x is the opposed-finger closing axis in the vertical/radial plane.
    x = np.array([0., 0., 1.]) - z * z[2]
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.column_stack([x, y, z])
    half_extent = size[0 if recipe.rim_sector < 2 else 1] / 2
    contact = radial * (half_extent - recipe.radial_inset_m)
    contact[2] = recipe.vertical_offset_m
    T_BC = transform(center_in_body_m, rotation=np.eye(3))
    T_WC = T_WB @ T_BC
    T_CG = transform(contact, rotation=rotation)
    pre = transform(contact + approach * recipe.approach_distance_m, rotation=rotation)
    grasp = T_WC @ T_CG
    lift = grasp.copy()
    lift[:3, 3] += np.array([0, 0, recipe.lift_distance_m])
    return {"T_BC": T_BC, "T_WC": T_WC, "T_CG": T_CG,
            "PRE_GRASP": T_WC @ pre, "GRASP": grasp, "LIFT": lift}


def grasp_plate(context, object_id="plate", recipe=None):
    if object_id != "plate":
        raise ValueError("Initial recipe supports c1_1 plate only")
    return context.execute_plate(recipe or PlateRecipe())


def build_calibrated_plate_targets(T_WB, center_in_body_m, observed_bbox_size_m, recipe=None):
    """Retarget a stored same-asset pose; use current M1 size as an observation check.

    Partial visibility changes are not physical changes in asset size. A new
    asset/size requires its own calibration, rather than rescaling this grasp.
    """
    recipe = recipe or PlateRecipe()
    observed = np.asarray(observed_bbox_size_m, dtype=float)
    calibrated = np.asarray(recipe.calibrated_local_bbox_size_m)
    tolerance = np.array([.05 * calibrated[0], .05 * calibrated[1], .002])
    if (observed.shape != (3,) or not np.isfinite(observed).all()
            or np.any(observed <= 0) or np.any(np.abs(observed-calibrated) > tolerance)):
        raise ValueError('M1_BBOX_OUTSIDE_CALIBRATED_OBSERVATION_RANGE')
    targets = build_plate_targets(T_WB, center_in_body_m, calibrated, recipe)
    import json
    from pathlib import Path
    calibration = json.loads((Path(__file__).parents[1] / 'calibrations/plate_pose.json').read_text())
    targets['T_CG'] = targets['T_CG'] @ np.asarray(calibration['correction_from_recipe_nominal'])
    targets['GRASP'] = targets['T_WC'] @ targets['T_CG']
    targets['PRE_GRASP'] = targets['GRASP'].copy()
    targets['PRE_GRASP'][:3, 3] -= targets['GRASP'][:3, 2] * recipe.approach_distance_m
    targets['LIFT'] = targets['GRASP'].copy()
    targets['LIFT'][2, 3] += recipe.lift_distance_m
    targets['calibration'] = {'pose_policy': 'FIXED_OBJECT_CENTER_POSE',
        'source': 'outputs/plate-native-adapter-04',
        'calibrated_local_bbox_size_m': calibrated,
        'current_observed_local_bbox_size_m': observed}
    targets['calibration']['recorded_pose_source'] = calibration['source']
    targets['calibration']['recorded_pose_kind'] = calibration['kind']
    return targets
