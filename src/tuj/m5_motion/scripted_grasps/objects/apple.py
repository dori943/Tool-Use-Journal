"""C2_1 apple: 3F enclosure around the broad upper body."""
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe,build_catalog_targets,dispatch_grasp

def apple_recipe():
    return CatalogRecipe('apple','c2_1','3F',(.075783,.075419,.075519),
        offset_fraction=(0.,0.,.15),offset_m=(-.002,0.,0.),two_finger_parallel_linkage=False)
def build_apple_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe or apple_recipe())
def grasp_apple(context,object_id='apple',recipe=None):
    return dispatch_grasp(context,object_id,recipe or apple_recipe(),expected_id='apple')
