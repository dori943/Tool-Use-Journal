"""C4_2 milk carton: validated 3F enclosure below the folded top."""
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe,build_catalog_targets,dispatch_grasp

def milk_recipe():
    return CatalogRecipe('milk','c4_2','3F',(.045662,.045662,.144000),
        offset_fraction=(0.,0.,.22),preshape_aperture_m=.058,preshape_closure_command=-.20,
        two_finger_force_target_n=8.,two_finger_parallel_linkage=False)
def build_milk_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe or milk_recipe())
def grasp_milk(context,object_id='milk',recipe=None):
    return dispatch_grasp(context,object_id,recipe or milk_recipe(),expected_id='milk')
