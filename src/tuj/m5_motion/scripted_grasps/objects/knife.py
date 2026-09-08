"""C2_2 knife: 2F handle pinch; blade contacts do not qualify."""
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe,build_catalog_targets,dispatch_grasp

def knife_recipe():
    return CatalogRecipe('knife','c2_2','2F',(.030185,.206132,.012947),
        offset_fraction=(0.,-.30,0.),offset_m=(.006,0.,.014),
        contact_region_min=(-.6,-.5,-.6),contact_region_max=(.6,-.1,.6),
        preshape_aperture_m=.028,preshape_closure_command=.45,lift_distance_m=.23)
def build_knife_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe or knife_recipe())
def grasp_knife(context,object_id='knife',recipe=None):
    return dispatch_grasp(context,object_id,recipe or knife_recipe(),expected_id='knife')
