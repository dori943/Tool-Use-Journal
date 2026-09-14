"""Task-scoped vacuum plate recipes.

C1_1, C2_1 and C3_1 use the full-size plate. C3_2 uses the smaller
``plate_a`` / ``plate_b`` instances and seats the cup on the solid rim annulus
instead of the recessed center. Routes descend along tool -z, engage suction,
and lift with task-scoped contact calibration.
"""
from dataclasses import replace
import numpy as np
from tuj.m5_motion.scripted_grasps.catalog_types import (
    CatalogRecipe, build_catalog_targets, dispatch_grasp,
)

# Compiled C3_2 plate AABB at PLATE_SCALE=0.17 (plate_a / plate_b share asset).
PLATE_EXPECTED_SIZE_M = (0.167550312, 0.167090268, 0.010191929)

# VacuumGripper cup outer contact radius used to keep the seal on solid rim.
CUP_CLEARANCE_RADIUS_M = 0.030
RIM_EDGE_MARGIN_M = 0.005
# Cup contact face ≡ grip site. Do not seat below the AABB top on C3_2.
SEATING_OFFSET_Z_M = 0.0

# C2_1 sorting plate (same flat asset family as C1_1 plate).
C2_1_PLATE_EXPECTED_SIZE_M = (0.182334163, 0.181833528, 0.011091216)
_TASK_IDS = {
    'C1_1_LegoSweep': 'c1_1',
    'C2_1_ObjectSorting': 'c2_1',
    'C3_1_ObjectSorting': 'c3_1',
    'C3_2_BreakfastTrayPreparation': 'c3_2',
}
_FULL_SIZE_M = (0.182334163, 0.181833528, 0.011091216)
_BREAKFAST_SIZE_M = (0.157261405, 0.157261434, 0.009592403)


def _rim_offset_fraction(size_m=PLATE_EXPECTED_SIZE_M):
    sx, sy, _ = size_m
    half_min = 0.5 * min(sx, sy)
    radial_m = half_min - CUP_CLEARANCE_RADIUS_M - RIM_EDGE_MARGIN_M
    if radial_m <= 0:
        raise ValueError('PLATE_TOO_SMALL_FOR_VACUUM_CUP')
    return (radial_m / sx, 0.0, 0.5)


def _task_recipe(task_id, object_id, expected_size_m):
    is_breakfast = task_id == 'c3_2'
    offset_fraction = (
        _rim_offset_fraction(expected_size_m)
        if is_breakfast else (0., 0., .5)
    )
    extra = ({
        'contact_region_min': (-0.95, -0.95, 0.0),
        'contact_region_max': (0.95, 0.95, 0.6),
    } if is_breakfast else {})
    return CatalogRecipe(
        object_id, task_id, 'vac',
        expected_size_m,
        offset_fraction=offset_fraction,
        # C1_1 rests on table_collision: negative immersion pushes the plate
        # through its support during CLOSE / early LIFT.
        offset_m=(0., 0., 0. if task_id in {'c1_1', 'c3_2'} else -.0005),
        linear_lift_start=task_id == 'c1_1',
        arm_kp=300. if task_id == 'c1_1' else 150.,
        two_finger_parallel_linkage=False,
        post_grasp_arm_kp=300.,
        # Reach the suction command before the thin plate is pushed or tilted
        # by a slow four-second command ramp.
        close_duration_s=.04 if task_id == 'c1_1' else .2,
        # The plate collision mesh yields one aligned cup contact. Requiring
        # three contacts is appropriate for the lid mesh but rejects this
        # sustained, centered contact before attachment.
        minimum_vacuum_contact_count=1,
        contact_ticks=3 if task_id == 'c1_1' else 5,
        prelift_stabilization_s=.02 if task_id == 'c1_1' else .5,
        # C1_1's concave plate mesh reports deeper cup and initial table overlap
        # at an otherwise aligned top-face contact. Keep both exceptions scoped
        # to that environment and below their catalog safety caps.
        maximum_vacuum_attach_penetration_m=(
            .0035 if task_id == 'c1_1' else .002),
        maximum_support_separation_penetration_m=(
            .0045 if task_id == 'c1_1' else .002),
        **extra,
    )


def tune_recipe_to_measured_size(recipe, local_size_m):
    """Recompute C3_2 rim fraction from the live AABB (PLATE_SCALE-safe)."""
    if (
        recipe.object_id not in {'plate', 'plate_a', 'plate_b'}
        or recipe.task_id != 'c3_2'
        or recipe.ee_id != 'vac'
    ):
        return recipe
    size = tuple(float(x) for x in np.asarray(local_size_m, dtype=float).reshape(3))
    if not np.isfinite(size).all() or min(size) <= 0:
        raise ValueError('Invalid measured plate size')
    return replace(
        recipe,
        expected_size_m=size,
        offset_fraction=_rim_offset_fraction(size),
        offset_m=(0.0, 0.0, SEATING_OFFSET_Z_M),
    )


def plate_vac_c1_1_recipe():
    """Flat top-face vac contact for C1_1 LegoSweep plates.

    Do not use a negative seating offset: C1_1 plates rest on ``table_collision``,
    and immersing the cup presses the free plate into the table. CLOSE then
    reports ~1 cm depression and early-LIFT fails on residual table penetration.
    """
    return _task_recipe('c1_1', 'plate', C2_1_PLATE_EXPECTED_SIZE_M)


def plate_vac_c2_1_recipe():
    """Flat top-face vac contact for C2_1 sorting plates."""
    return _task_recipe('c2_1', 'plate', C2_1_PLATE_EXPECTED_SIZE_M)


def plate_vac_c3_2_recipe(object_id='plate'):
    """Rim-annulus vac contact for C3_2 breakfast plates."""
    if object_id not in {'plate', 'plate_a', 'plate_b'}:
        raise ValueError(f'UNSUPPORTED_PLATE_VAC_OBJECT: {object_id}')
    return _task_recipe('c3_2', object_id, PLATE_EXPECTED_SIZE_M)


def plate_vac_recipe(environment=None, object_id='plate'):
    """Build an environment-scoped recipe; no-arg calls retain C3_2 defaults."""
    if environment is None:
        if object_id != 'plate':
            raise ValueError(f'UNSUPPORTED_PLATE_VAC_OBJECT: {object_id}')
        return plate_vac_c3_2_recipe()
    try:
        task_id = _TASK_IDS[environment]
    except KeyError as error:
        raise ValueError(
            f'UNSUPPORTED_PLATE_VAC_ENVIRONMENT: {environment}'
        ) from error
    allowed_ids = {'plate_a', 'plate_b'} if task_id == 'c3_2' else {'plate'}
    if object_id not in allowed_ids:
        raise ValueError(f'UNSUPPORTED_PLATE_VAC_OBJECT: {object_id}')
    expected = PLATE_EXPECTED_SIZE_M if task_id == 'c3_2' else _FULL_SIZE_M
    return _task_recipe(task_id, object_id, expected)


def build_plate_vac_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    return build_catalog_targets(
        T_WB, center_in_body_m, local_size_m, recipe or plate_vac_recipe())


def grasp_plate_vac(context, object_id=None, recipe=None):
    target = object_id or getattr(context, 'object_id', 'plate')
    # Prefer the recipe bind_context already installed (C1_1/C2_1 flat vs C3_2
    # rim). Falling back to plate_vac_recipe() defaults to C3_2 and trips
    # "Context must be initialized with the same recipe" on C1_1 vac.
    active = recipe or getattr(context, 'recipe', None) or plate_vac_recipe()
    return dispatch_grasp(context, target, active, expected_id=active.object_id)
