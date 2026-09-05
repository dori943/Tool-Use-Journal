"""C4_2 rolling pin: 2F across the central cylinder."""
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe,build_catalog_targets,dispatch_grasp

def rolling_pin_recipe():
    return CatalogRecipe('rolling_pin','c4_2','2F',(.043739,.300000,.043739),
        offset_m=(0.,0.,.010),preshape_aperture_m=.050,preshape_closure_command=0.,
        contact_region_min=(-.6,-.20,-.6),contact_region_max=(.6,.20,.6),
        two_finger_force_target_n=8.,lift_distance_m=.21)
def build_rolling_pin_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe or rolling_pin_recipe())
def grasp_rolling_pin(context,object_id='rolling_pin',recipe=None):
    return dispatch_grasp(context,object_id,recipe or rolling_pin_recipe(),expected_id='rolling_pin')
