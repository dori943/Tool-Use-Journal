"""C3_2 fruit: 3F enclosure following the validated C2_1 apple pattern."""
from tuj.m5_motion.scripted_grasps.catalog_types import (
    CatalogRecipe, build_catalog_targets, dispatch_grasp,
)

# Compiled C3_2 fruit AABB (fruit_a / fruit_b share the same asset).
# Slightly larger than C2_1 apple (~0.076); offsets reuse apple normalized pose.
FRUIT_EXPECTED_SIZE_M = (0.080834846, 0.080446873, 0.080553459)

# Upper-body enclosure fraction (apple uses 0.15). Physical close may refine.
UPPER_BODY_FRACTION_Z = 0.15
# Small lateral bias copied as an explicit apple-pattern parameter, not hidden.
LATERAL_OFFSET_M = -0.002


def fruit_recipe():
    return CatalogRecipe(
        'fruit', 'c3_2', '3F', FRUIT_EXPECTED_SIZE_M,
        offset_fraction=(0.0, 0.0, UPPER_BODY_FRACTION_Z),
        offset_m=(LATERAL_OFFSET_M, 0.0, 0.0),
        two_finger_parallel_linkage=False,
        # fruit_a PRE at the default 0.12 m standoff has no IK from the
        # deterministic seeds; 0.07 m is reachable for both fruit_a/fruit_b.
        approach_distance_m=0.07,
        # fruit_a also fails IK at the default 0.18 m LIFT height.
        lift_distance_m=0.12,
        minimum_lift_m=0.08,
        # Kinematic carry + post-LIFT LEVEL: live fruit_b place failed when a
        # ~6° grasp tilt skewed PRE_PLACE into an unreachable precise-IK pose.
        hold_finger_positions=True,
    )


def fruit_2f_c3_2_recipe():
    """Greedy-extra 2F pinch: fruit diameter (~81 mm) fits the 85 mm stroke."""
    return CatalogRecipe(
        'fruit', 'c3_2', '2F', FRUIT_EXPECTED_SIZE_M,
        offset_fraction=(0.0, 0.0, 0.08),
        offset_m=(LATERAL_OFFSET_M, 0.0, 0.0),
        approach_distance_m=0.07,
        lift_distance_m=0.12,
        minimum_lift_m=0.08,
        preshape_aperture_m=0.084,
        preshape_closure_command=-0.7,
        two_finger_force_target_n=8.0,
        # Smooth fruit can yaw about the pinch axis (same class as apple_2f).
        maximum_slip_deg=25.0,
    )


def build_fruit_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    return build_catalog_targets(
        T_WB, center_in_body_m, local_size_m, recipe or fruit_recipe())


def grasp_fruit(context, object_id=None, recipe=None):
    object_id = object_id or getattr(context, 'object_id', 'fruit')
    return dispatch_grasp(
        context, object_id, recipe or fruit_recipe(), expected_id='fruit')
