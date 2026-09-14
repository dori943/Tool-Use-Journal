"""C4_1 (IntervalFitExtraction) tool pick: 2F grip on the flat spatula_a.

C4_1 is a RoboCasa kitchen: the task picks a thin tool off the island top to
reach a card wedged in the appliance gap.  The tool lies flat and flush on the
counter, so the generic M5 pick drove the 2F fingertips into the island surface
and every generated grasp was collision-filtered.  Routing the pick through this
scripted catalog recipe uses the grasp path that treats the island top as the
support surface (CatalogContext.find_support_geom recognises
'island_island_group_top_2'), so fingertip proximity to the counter is handled
like a tabletop pick instead of being hard-filtered.

Geometry note: the M5 object record dims are ordered (x=width, y=length,
z=thickness) -- e.g. spatula_a = (0.036, 0.140, 0.004).  The catalog 2F jaw
closes across the FIRST axis (width) by default (same layout as the C2_2 knife
recipe), so no rotation is needed; offset stays at the body centre for a stable
lift.  task_id stays 'c4_2' (the whitelist has no c4_1; dispatch keys on
object_id, so this is inert).
"""
from tuj.m5_motion.scripted_grasps.catalog_types import (
    CatalogRecipe, build_catalog_targets, dispatch_grasp,
)


def spatula_c4_1_recipe():
    return CatalogRecipe(
        'tool_2_spatula_a', 'c4_2', '2F',
        (0.036, 0.140, 0.004),
        offset_fraction=(0., 0., 0.),
        offset_m=(0., 0., 0.),
        preshape_aperture_m=.049,
        preshape_closure_command=0.,
        two_finger_force_target_n=6.,
        # Thin flat tool on the counter: close gently and relax the antipodal
        # opposition gate so a firm-squeeze roll does not reject the real grip
        # (see objects/knife_c4_1.py for the full rationale).
        close_duration_s=8.,
        minimum_normal_opposition=.12,
        fingerpad_friction=(2.0, 0.1, 0.02),
    )


def build_spatula_c4_1_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    return build_catalog_targets(T_WB, center_in_body_m, local_size_m,
                                 recipe or spatula_c4_1_recipe())


def grasp_spatula_c4_1(context, object_id='tool_2_spatula_a', recipe=None):
    return dispatch_grasp(context, object_id, recipe or spatula_c4_1_recipe(),
                          expected_id='tool_2_spatula_a')
