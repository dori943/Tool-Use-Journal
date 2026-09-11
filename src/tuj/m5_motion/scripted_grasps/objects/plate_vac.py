"""C3_2 plate vacuum recipe: rim-annulus cup contact (not the hollow center).

Uses the catalog vac path (lid-style PRE→GRASP→attach→LIFT). Lateral contact
is derived from the plate bbox and an explicit cup-clearance parameter so the
TCP is not aimed at the recessed dish center.

Vertical seating: the VacuumGripper ``vac_cup`` is a cylinder whose contact
face coincides with the grip site (cup center sits one half-height along the
site axis away from the object, half-height ≈ 0.012 m). Commanding the grip
site below the plate AABB top therefore immerses the cup into the plate and
presses the free plate into the island. Keep vertical seating at the top face
(no negative z), unlike lid's small immersion on a thicker free body.
"""
from dataclasses import replace

import numpy as np

from tuj.m5_motion.scripted_grasps.catalog_types import (
    CatalogRecipe, build_catalog_targets, dispatch_grasp,
)

# Nominal compiled C3_2 plate AABB at environments.PLATE_SCALE == 0.17
# (plate_a / plate_b share the asset). Runtime retunes to the measured AABB so
# scale edits do not leave the cup on the recessed dish.
PLATE_EXPECTED_SIZE_M = (0.167550312, 0.167090268, 0.010191929)

# VacuumGripper cup outer contact radius used to keep the seal on solid rim.
# Physical validation may refine this; do not hide it inside unrelated recipes.
CUP_CLEARANCE_RADIUS_M = 0.030
# Keep the cup inside the outer rim after subtracting cup radius from half-width.
RIM_EDGE_MARGIN_M = 0.005
# Cup contact face ≡ grip site (see module docstring). Do not seat below the
# AABB top: negative values push the plate into island_island_group_top_2 and
# fail LIFT with ~1 mm residual support penetration after attach.
SEATING_OFFSET_Z_M = 0.0


def _rim_offset_fraction(size_m=PLATE_EXPECTED_SIZE_M):
    sx, sy, _ = (float(v) for v in size_m)
    half_min = 0.5 * min(sx, sy)
    radial_m = half_min - CUP_CLEARANCE_RADIUS_M - RIM_EDGE_MARGIN_M
    if radial_m <= 0:
        raise ValueError('PLATE_TOO_SMALL_FOR_VACUUM_CUP')
    # Prefer +X; plate is nearly square so either lateral axis is a rim patch.
    return (radial_m / sx, 0.0, 0.5)


def tune_recipe_to_measured_size(recipe, local_size_m):
    """Recompute rim fractions for the live plate AABB.

    ``offset_fraction`` is size-relative. Keeping a stale expected size after a
    ``PLATE_SCALE`` edit places the cup inward of the solid rim; CLOSE then
    loses suction alignment as the cup walks into the dish.
    """
    size = tuple(float(v) for v in np.asarray(local_size_m, dtype=float).reshape(3))
    if any(v <= 0.0 or not np.isfinite(v) for v in size):
        raise ValueError('Invalid plate local size')
    fx, fy, fz = _rim_offset_fraction(size)
    return replace(
        recipe,
        expected_size_m=size,
        offset_fraction=(fx, fy, fz),
        offset_m=(0.0, 0.0, SEATING_OFFSET_Z_M),
    )


def plate_vac_recipe():
    fx, fy, fz = _rim_offset_fraction()
    return CatalogRecipe(
        'plate', 'c3_2', 'vac', PLATE_EXPECTED_SIZE_M,
        offset_fraction=(fx, fy, fz),
        offset_m=(0.0, 0.0, SEATING_OFFSET_Z_M),
        two_finger_parallel_linkage=False,
        post_grasp_arm_kp=300.0,
        # Restrict contact credit to the upper face (cup seal), not the underside.
        contact_region_min=(-0.95, -0.95, 0.0),
        contact_region_max=(0.95, 0.95, 0.6),
    )


def build_plate_vac_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    tuned = tune_recipe_to_measured_size(
        recipe or plate_vac_recipe(), local_size_m)
    return build_catalog_targets(
        T_WB, center_in_body_m, local_size_m, tuned)


def grasp_plate_vac(context, object_id=None, recipe=None):
    object_id = object_id or getattr(context, 'object_id', 'plate')
    return dispatch_grasp(
        context, object_id, recipe or plate_vac_recipe(), expected_id='plate')
