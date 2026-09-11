"""C2_1 plate: vacuum-cup contact on the flat top face, then rigid attachment.

The sorting plate in C2_1 is the same flat asset as the C1_1 plate, but M4
mounts the vacuum EE for it here (a flat disc has no rim a parallel jaw can
pinch from above on the tray-loading approach).  The per-run LLM vacuum grasp
kept missing the surface (BREAKABLE_WELD CONTACT_COUNT=0: the cup stopped above
the plate), so this scripted recipe reuses the validated catalog vacuum path —
descend along the tool -z onto the top face, touch, engage suction, lift — the
same mechanism the C4_2 lid uses.  offset_fraction z=+0.5 targets the top face
and offset_m z=-0.5 mm presses the cup just into the surface so contact is made.
"""
from tuj.m5_motion.scripted_grasps.catalog_types import (
    CatalogRecipe, build_catalog_targets, dispatch_grasp,
)


def plate_vac_recipe():
    return CatalogRecipe(
        'plate', 'c2_1', 'vac',
        (0.182334163, 0.181833528, 0.011091216),
        offset_fraction=(0., 0., .5),
        offset_m=(0., 0., -.0005),
        two_finger_parallel_linkage=False,
        post_grasp_arm_kp=300.,
    )


def build_plate_vac_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    return build_catalog_targets(T_WB, center_in_body_m, local_size_m,
                                 recipe or plate_vac_recipe())


def grasp_plate_vac(context, object_id='plate', recipe=None):
    return dispatch_grasp(context, object_id, recipe or plate_vac_recipe(),
                          expected_id='plate')
