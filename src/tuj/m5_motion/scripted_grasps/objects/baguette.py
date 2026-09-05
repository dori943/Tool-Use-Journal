"""C4_2 baguette: 2F across its central short axis."""
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe,build_catalog_targets,dispatch_grasp

def baguette_recipe():
    return CatalogRecipe('baguette','c4_2','2F',(.059554,.261350,.042218),
        offset_m=(0.,0.,.010),preshape_aperture_m=.070,preshape_closure_command=-.45,
        contact_region_min=(-.6,-.20,-.6),contact_region_max=(.6,.20,.6))
def build_baguette_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe or baguette_recipe())
def grasp_baguette(context,object_id='baguette',recipe=None):
    return dispatch_grasp(context,object_id,recipe or baguette_recipe(),expected_id='baguette')
