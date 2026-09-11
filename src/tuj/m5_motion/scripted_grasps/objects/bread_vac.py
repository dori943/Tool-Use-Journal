"""C3_2 bread vacuum: top-face cup contact (lid-style catalog vac path).

C2_1 bread remains the separate 3F catalog recipe in ``bread.py``. The breakfast
loaf has a solid upper surface, so contact is at the AABB top center rather than
a rim annulus.

Vertical seating: the VacuumGripper ``vac_cup`` contact face coincides with the
grip site. ``offset_fraction`` z=0.5 already places GRASP on the AABB top face;
a negative ``offset_m`` z would immerse the cup into the loaf and press the free
body into its support after attach (same failure mode as plate_vac). Keep
seating at the top face — do not copy lid's small immersion.
"""
from tuj.m5_motion.scripted_grasps.catalog_types import (
    CatalogRecipe, build_catalog_targets, dispatch_grasp,
)

# Compiled C3_2 bread AABB (bread_a / bread_b share the same asset).
BREAD_EXPECTED_SIZE_M = (0.064208909, 0.099980534, 0.053419888)

# Cup contact face ≡ grip site at AABB top (offset_fraction z=0.5). No negative
# seating: see module docstring / plate_vac.
SEATING_OFFSET_Z_M = 0.0


def bread_vac_recipe():
    return CatalogRecipe(
        'bread', 'c3_2', 'vac', BREAD_EXPECTED_SIZE_M,
        offset_fraction=(0.0, 0.0, 0.5),
        offset_m=(0.0, 0.0, SEATING_OFFSET_Z_M),
        two_finger_parallel_linkage=False,
        post_grasp_arm_kp=300.0,
        contact_region_min=(-0.7, -0.7, 0.0),
        contact_region_max=(0.7, 0.7, 0.6),
    )


def build_bread_vac_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    return build_catalog_targets(
        T_WB, center_in_body_m, local_size_m, recipe or bread_vac_recipe())


def grasp_bread_vac(context, object_id=None, recipe=None):
    object_id = object_id or getattr(context, 'object_id', 'bread')
    return dispatch_grasp(
        context, object_id, recipe or bread_vac_recipe(), expected_id='bread')
