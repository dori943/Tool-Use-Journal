"""C2_1 loaf: 3F enclosure across its short horizontal axis."""
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe,build_catalog_targets,dispatch_grasp

def bread_recipe():
    return CatalogRecipe('bread','c2_1','3F',(.083472,.129975,.069446),
        offset_fraction=(0.,0.,.12),offset_m=(-.002,0.,0.),two_finger_parallel_linkage=False,
        post_grasp_arm_kp=300.)

def bread_2f_c3_2_recipe():
    """Experimental C3_2 2F enclosure using the C2_1 bread pattern."""
    from tuj.m5_motion.scripted_grasps.objects.bread_vac import BREAD_EXPECTED_SIZE_M
    return CatalogRecipe('bread','c3_2','2F',BREAD_EXPECTED_SIZE_M,
        offset_fraction=(0.,0.,.12),offset_m=(-.002,0.,0.),
        preshape_aperture_m=.070,preshape_closure_command=-.45,
        two_finger_parallel_linkage=True,post_grasp_arm_kp=300.)
def build_bread_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe or bread_recipe())
def grasp_bread(context,object_id='bread',recipe=None):
    object_id = getattr(context, 'object_id', object_id)
    recipe = recipe or getattr(context, 'recipe', None) or bread_recipe()
    return dispatch_grasp(context,object_id,recipe,expected_id='bread')
