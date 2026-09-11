"""C3_2 fork: 3F thin-handle pinch mirroring the validated c3_2 spoon pose.

Scene layout (top-down): forks sit under the spoons with the same orientation —
tines toward +Y, dark handle toward -Y. Grasp the wider mid-handle (dark
section), not the tapered tip or silver neck.

Long axis matches spoon/knife local frames. TCP uses the spoon_3f_c3_2
yaw-90 / tilt-5 thumb+index station via ``rotation_xyz_deg``.
"""
from tuj.m5_motion.scripted_grasps.catalog_types import (
    CatalogRecipe, build_catalog_targets, dispatch_grasp,
)

# Compiled C3_2 fork AABB (fork_a / fork_b share the same asset).
FORK_EXPECTED_SIZE_M = (0.025003964, 0.153709715, 0.018469741)

# Spoon-matched yaw/tilt on the dark mid-handle (image: forks under spoons,
# handles toward -Y). Keep lateral on the spoon side so GRASP clears tray_a;
# height sits between spoon (+14 mm) and the tray-colliding +8 mm seat.
HANDLE_FRACTION_Y = -0.20
HEIGHT_OFFSET_M = 0.010
LATERAL_OFFSET_M = -0.004
# Catalog euler xyz ≡ spoon Rx(0) @ Rz(90) @ Ry(5) (see spoon_3f_c3_2).
ROTATION_XYZ_DEG = (0.0, 5.0, 90.0)


def fork_recipe():
    return CatalogRecipe(
        'fork', 'c3_2', '3F', FORK_EXPECTED_SIZE_M,
        offset_fraction=(0.0, HANDLE_FRACTION_Y, 0.0),
        offset_m=(LATERAL_OFFSET_M, 0.0, HEIGHT_OFFSET_M),
        rotation_xyz_deg=ROTATION_XYZ_DEG,
        # Credit contacts on the handle half only (not the tine tips).
        contact_region_min=(-0.6, -0.75, -0.6),
        contact_region_max=(0.6, 0.05, 0.6),
        two_finger_parallel_linkage=False,
        three_finger_force_targets_n=(3.0, 1.5, 0.5),
        lift_distance_m=0.16,
        # Near-open approach; mid-close tips strike the island/tray.
        preshape_aperture_m=0.025,
        preshape_closure_command=0.05,
        thin_handle_pinch=True,
        hold_finger_positions=True,
    )


def fork_2f_c3_2_recipe():
    """Experimental 2F handle pinch reusing the validated knife geometry."""
    return CatalogRecipe(
        'fork', 'c3_2', '2F', FORK_EXPECTED_SIZE_M,
        offset_fraction=(0.0, -0.30, 0.0),
        offset_m=(0.006, 0.0, 0.014),
        contact_region_min=(-0.6, -0.5, -0.6),
        contact_region_max=(0.6, -0.1, 0.6),
        preshape_aperture_m=.028,
        preshape_closure_command=.45,
        lift_distance_m=.23,
    )


def build_fork_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    return build_catalog_targets(
        T_WB, center_in_body_m, local_size_m, recipe or fork_recipe())


def grasp_fork(context, object_id=None, recipe=None):
    object_id = object_id or getattr(context, 'object_id', 'fork')
    recipe = recipe or getattr(context, 'recipe', None) or fork_recipe()
    return dispatch_grasp(
        context, object_id, recipe, expected_id='fork')
