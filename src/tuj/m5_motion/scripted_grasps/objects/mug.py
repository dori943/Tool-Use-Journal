"""C2_1 mug: validated 3F enclosure around the body, away from the handle."""
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe,build_catalog_targets,dispatch_grasp

def mug_recipe():
    # The grasp target is already centered on the cup body: build_catalog_targets
    # places T_WC at center_in_body, which the M5 record puts at the body wall
    # centroid (center_in_body_y=-0.0069, pulled off-origin BY the handle on +y).
    # An extra offset_fraction_y=+0.1 (=+0.0102 m) then shoved the whole grip back
    # toward the handle: thumb+index closed on the body across x (5.1 N / 4 N), but
    # the lone +y finger (pinky) reached past the wall into the handle gap and stayed
    # at ~0 N the entire CLOSE, so ready() (all three fingers) never armed
    # (GRASP_CONTACT_NOT_STABLE).  Drop the y offset so the enclosure sits on the
    # body center and all three fingers land on the cylindrical wall.
    return CatalogRecipe('mug','c2_1','3F',(.070630,.101936,.080475),
        offset_fraction=(0.,0.,.1),offset_m=(0.,0.,0.),two_finger_parallel_linkage=False,
        contact_region_min=(-.6,-.25,-.45),contact_region_max=(.6,.45,.5),
        preshape_aperture_m=.081,preshape_closure_command=-.82,two_finger_force_target_n=8.,
        lift_distance_m=.14)
def build_mug_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe or mug_recipe())
def grasp_mug(context,object_id='mug',recipe=None):
    return dispatch_grasp(context,object_id,recipe or mug_recipe(),expected_id='mug')
