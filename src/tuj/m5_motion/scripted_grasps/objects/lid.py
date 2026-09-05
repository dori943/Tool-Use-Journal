"""C4_2 lid: cup contact followed by the existing rigid vacuum attachment."""
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe,build_catalog_targets,dispatch_grasp

def lid_recipe():
    return CatalogRecipe('lid','c4_2','vac',(.210000,.270000,.012000),
        offset_fraction=(0.,0.,.5),offset_m=(0.,0.,-.0005),two_finger_parallel_linkage=False,
        post_grasp_arm_kp=300.)
def build_lid_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe or lid_recipe())
def grasp_lid(context,object_id='lid',recipe=None):
    return dispatch_grasp(context,object_id,recipe or lid_recipe(),expected_id='lid')
