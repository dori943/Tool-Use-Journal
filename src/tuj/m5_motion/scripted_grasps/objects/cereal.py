"""C4_2 cereal box: 2F across the thin side walls near its top."""
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe,build_catalog_targets,dispatch_grasp

def cereal_recipe():
    return CatalogRecipe('cereal','c4_2','2F',(.038963,.123253,.150000),
        offset_fraction=(0.,0.,.36),preshape_aperture_m=.050,preshape_closure_command=0.,two_finger_force_target_n=8.)
def build_cereal_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe or cereal_recipe())
def grasp_cereal(context,object_id='cereal',recipe=None):
    return dispatch_grasp(context,object_id,recipe or cereal_recipe(),expected_id='cereal')
