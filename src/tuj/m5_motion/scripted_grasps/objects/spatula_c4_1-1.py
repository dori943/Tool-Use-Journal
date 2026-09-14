"""C4_1 (IntervalFitExtraction) tool pick: 2F grip on the flat spatula.

C4_1 is a RoboCasa kitchen: the task picks a thin spatula off the island top to
reach a card wedged in the gap between two appliances.  The tool lies flat and
flush on the counter (tool_2_spatula_a bbox ~= 134 x 31.7 x 1.4 mm), so the
generic M5 pick put the 2F fingertips into the island surface and every
generated grasp was collision-filtered (gripper0_right_*_fingertip_collision
<-> island_island_group_top_2, clearance below the 5 mm swept margin).

Routing the pick through this scripted catalog recipe uses the grasp path that
treats the island top as the support surface (CatalogContext.find_support_geom
recognises 'island_island_group_top_2'), so fingertip proximity to the counter
is handled the same way a tabletop pick handles the table, instead of being
hard-filtered.

First cut for the pick only (SG1_d1 PICK_TOOL); the downstream tool_act:extract
motion is a separate, still-unimplemented step.  Grip choices:
  * rotation_xyz_deg z=90  -> the 2F jaw closes across the 31.7 mm WIDTH, not
    the 134 mm length (which will not fit the 85 mm opening).
  * offset_fraction=0      -> grip at the body centre for a stable lift (a
    handle-end grip matters for the extraction, not the pick).
task_id stays 'c4_2' -- the CatalogRecipe whitelist has no c4_1 and dispatch
keys on object_id, so this is inert (same trick as plate_vac / apple_2f).
"""
from tuj.m5_motion.scripted_grasps.catalog_types import (
    CatalogRecipe, build_catalog_targets, dispatch_grasp,
)


def spatula_c4_1_recipe():
    return CatalogRecipe(
        'tool_2_spatula_a', 'c4_2', '2F',
        (.134, .0317, .0014),
        offset_fraction=(0., 0., 0.),
        offset_m=(0., 0., 0.),
        rotation_xyz_deg=(0., 0., 90.),
        preshape_aperture_m=.05,
        preshape_closure_command=0.,
        two_finger_force_target_n=6.,
    )


def build_spatula_c4_1_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    return build_catalog_targets(T_WB, center_in_body_m, local_size_m,
                                 recipe or spatula_c4_1_recipe())


def grasp_spatula_c4_1(context, object_id='tool_2_spatula_a', recipe=None):
    return dispatch_grasp(context, object_id, recipe or spatula_c4_1_recipe(),
                          expected_id='tool_2_spatula_a')
