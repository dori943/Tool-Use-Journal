"""Bread enclosure grasps: validated C2_1 3F, plus C3_2 size-gated variants."""
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe,build_catalog_targets,dispatch_grasp

def bread_recipe():
    return CatalogRecipe('bread','c2_1','3F',(.083472,.129975,.069446),
        offset_fraction=(0.,0.,.12),offset_m=(-.002,0.,0.),two_finger_parallel_linkage=False,
        post_grasp_arm_kp=300.)

def bread_3f_c3_2_recipe():
    """C3_2 loaf: C2_1 enclosure offsets with breakfast AABB + tray-clear yaw.

    Loaves sit between tray_a and tray_b. Catalog yaw=0 / 180 puts a distal pad
    into one tray or the other (bread_a pinky↔tray_b, bread_b thumb↔tray_a).
    yaw=90 (same station family as c3_2 fork/spoon) clears both instances in
    MuJoCo probes of the failing live worlds. Near-open + short fruit standoff
    match other crowded c3_2 3F approaches.
    """
    from tuj.m5_motion.scripted_grasps.objects.bread_vac import BREAD_EXPECTED_SIZE_M
    return CatalogRecipe('bread','c3_2','3F',BREAD_EXPECTED_SIZE_M,
        offset_fraction=(0.,0.,.12),offset_m=(-.002,0.,0.),
        rotation_xyz_deg=(0.,0.,90.),
        two_finger_parallel_linkage=False,post_grasp_arm_kp=300.,
        # Near-open: catalog_runtime treats closure < 0.1 as OPEN tips.
        preshape_closure_command=.05,
        approach_distance_m=.07,
        lift_distance_m=.12,
        minimum_lift_m=.08,
    )

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
