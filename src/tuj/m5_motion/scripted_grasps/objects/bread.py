"""C2_1 loaf: 3F enclosure across its short horizontal axis."""
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe,build_catalog_targets,dispatch_grasp

def bread_recipe():
    return CatalogRecipe('bread','c2_1','3F',(.083472,.129975,.069446),
        offset_fraction=(0.,0.,.12),offset_m=(-.002,0.,0.),two_finger_parallel_linkage=False,
        post_grasp_arm_kp=300.)
def build_bread_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe or bread_recipe())
def grasp_bread(context,object_id='bread',recipe=None):
    return dispatch_grasp(context,object_id,recipe or bread_recipe(),expected_id='bread')
