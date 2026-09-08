"""C4_2 whisk: 2F around the solid handle, excluding the wire head."""
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe,build_catalog_targets,dispatch_grasp

def whisk_recipe():
    return CatalogRecipe('whisk','c4_2','2F',(.067597,.280000,.067728),
        offset_fraction=(0.,-.28,0.),offset_m=(0.,0.,.006),
        contact_region_min=(-.6,-.5,-.6),contact_region_max=(.6,-.12,.6),
        preshape_aperture_m=.040,preshape_closure_command=.20)
def build_whisk_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe or whisk_recipe())
def grasp_whisk(context,object_id='whisk',recipe=None):
    return dispatch_grasp(context,object_id,recipe or whisk_recipe(),expected_id='whisk')
