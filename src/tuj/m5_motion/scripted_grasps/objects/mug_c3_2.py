"""C3_2 mug: 3F body enclosure; C2_1 mug recipe in ``mug.py`` is unchanged."""
from tuj.m5_motion.scripted_grasps.catalog_types import (
    CatalogRecipe, build_catalog_targets, dispatch_grasp,
)

# Compiled C3_2 mug AABB (mug_a / mug_b share the same asset).
MUG_EXPECTED_SIZE_M = (0.088287793, 0.127419893, 0.100593936)

# Normalized body grasp (same layout as validated c2_1 mug: away from handle).
OFFSET_FRACTION = (0.0, 0.1, 0.1)
CONTACT_REGION_MIN = (-0.6, -0.25, -0.45)
CONTACT_REGION_MAX = (0.6, 0.45, 0.5)
# Lift kept near the c2_1 mug value; physical hold may refine.
LIFT_DISTANCE_M = 0.18
MINIMUM_LIFT_M = 0.08


def mug_c3_2_recipe():
    return CatalogRecipe(
        'mug', 'c3_2', '3F', MUG_EXPECTED_SIZE_M,
        offset_fraction=OFFSET_FRACTION,
        offset_m=(0.0, 0.0, 0.0),
        two_finger_parallel_linkage=False,
        contact_region_min=CONTACT_REGION_MIN,
        contact_region_max=CONTACT_REGION_MAX,
        preshape_aperture_m=.081,
        preshape_closure_command=-.82,
        two_finger_force_target_n=8.0,
        lift_distance_m=LIFT_DISTANCE_M,
        minimum_lift_m=MINIMUM_LIFT_M,
        # Freeze close commands after acquire; force-servo unload drops index
        # during LIFT on the taller c3_2 mug mesh.
        hold_finger_positions=True,
    )


def build_mug_c3_2_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    return build_catalog_targets(
        T_WB, center_in_body_m, local_size_m, recipe or mug_c3_2_recipe())


def grasp_mug_c3_2(context, object_id=None, recipe=None):
    object_id = object_id or getattr(context, 'object_id', 'mug')
    return dispatch_grasp(
        context, object_id, recipe or mug_c3_2_recipe(), expected_id='mug')
