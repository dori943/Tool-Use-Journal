"""Vacuum plate recipes: C2_1 flat-top seating and C3_2 rim-annulus seating.

C2_1 / C1_1 vac alternatives use the flat disc top face (lid-style catalog vac
path). C3_2 breakfast plates are shallow dishes: lateral contact is derived from
the plate bbox and an explicit cup-clearance so the TCP is not aimed at the
recessed center.
"""
from tuj.m5_motion.scripted_grasps.catalog_types import (
    CatalogRecipe, build_catalog_targets, dispatch_grasp,
)

# Compiled C3_2 plate AABB (plate_a / plate_b share the same asset).
PLATE_EXPECTED_SIZE_M = (0.157694411, 0.157261429, 0.009592404)

# VacuumGripper cup outer contact radius used to keep the seal on solid rim.
CUP_CLEARANCE_RADIUS_M = 0.030
RIM_EDGE_MARGIN_M = 0.005
# Cup contact face ≡ grip site. Do not seat below the AABB top on C3_2.
SEATING_OFFSET_Z_M = 0.0

# C2_1 sorting plate (same flat asset family as C1_1 plate).
C2_1_PLATE_EXPECTED_SIZE_M = (0.182334163, 0.181833528, 0.011091216)


def _rim_offset_fraction(size_m=PLATE_EXPECTED_SIZE_M):
    sx, sy, _ = size_m
    half_min = 0.5 * min(sx, sy)
    radial_m = half_min - CUP_CLEARANCE_RADIUS_M - RIM_EDGE_MARGIN_M
    if radial_m <= 0:
        raise ValueError('PLATE_TOO_SMALL_FOR_VACUUM_CUP')
    return (radial_m / sx, 0.0, 0.5)


def plate_vac_c2_1_recipe():
    """Flat top-face vac contact for C2_1 / C1_1 vac alternatives."""
    return CatalogRecipe(
        'plate', 'c2_1', 'vac',
        C2_1_PLATE_EXPECTED_SIZE_M,
        offset_fraction=(0., 0., .5),
        offset_m=(0., 0., -.0005),
        two_finger_parallel_linkage=False,
        post_grasp_arm_kp=300.,
    )


def plate_vac_c3_2_recipe():
    """Rim-annulus vac contact for C3_2 breakfast plates."""
    fx, fy, fz = _rim_offset_fraction()
    return CatalogRecipe(
        'plate', 'c3_2', 'vac', PLATE_EXPECTED_SIZE_M,
        offset_fraction=(fx, fy, fz),
        offset_m=(0.0, 0.0, SEATING_OFFSET_Z_M),
        two_finger_parallel_linkage=False,
        post_grasp_arm_kp=300.0,
        contact_region_min=(-0.95, -0.95, 0.0),
        contact_region_max=(0.95, 0.95, 0.6),
    )


def plate_vac_recipe(task_id=None):
    """Dispatch by task id; default remains the C3_2 rim recipe used by local tests."""
    if task_id in (None, 'c3_2'):
        return plate_vac_c3_2_recipe()
    if task_id in {'c2_1', 'c1_1'}:
        return plate_vac_c2_1_recipe()
    raise ValueError(f'UNSUPPORTED_PLATE_VAC_TASK: {task_id!r}')


def build_plate_vac_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    return build_catalog_targets(
        T_WB, center_in_body_m, local_size_m, recipe or plate_vac_recipe())


def grasp_plate_vac(context, object_id=None, recipe=None):
    object_id = object_id or getattr(context, 'object_id', 'plate')
    # Prefer the recipe bind_context already installed (C1_1/C2_1 flat vs C3_2
    # rim). Falling back to plate_vac_recipe() defaults to C3_2 and trips
    # "Context must be initialized with the same recipe" on C1_1 vac.
    active = recipe or getattr(context, 'recipe', None) or plate_vac_recipe()
    return dispatch_grasp(context, object_id, active, expected_id='plate')
