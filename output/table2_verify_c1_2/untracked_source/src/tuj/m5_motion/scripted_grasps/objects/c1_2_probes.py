"""C1_2 fixed-EE coverage candidates, available only to the coverage validator.

Scene tools (Table II denominator 4): bottle, spatula, spoon, tongs. Production
routes (bottle 3F, spatula 3F, spoon 2F/3F) keep their own calibrated recipes and
are never replaced. The 7 missing pairs get one geometric candidate each with
CatalogRecipe defaults (5 N / 5 deg hold, lift 0.18 m, minimum 0.10 m, hold 5 s,
FREE_HOLD_THEN_ATTACH). Sizes are the assets' body-local ``reg_bbox`` extents.

Pose assumptions (verify with the static validator before trusting a FAIL):
  * tool long axis is body +Y, handle on the -Y side (fork/spatula convention);
    finger grasps take the handle at -0.15 of the length, yaw 90 deg so the fingers
    close across the handle width (X);
  * bottle is an upright cylinder; the 2F candidate is a side pinch at the centre,
    i.e. the same approach as the production 3F BottleRecipe but with catalog defaults;
  * vacuum tries the top surface centre (0, 0, +0.5). Tool tops are narrow
    (31-56 mm) so vacuum candidates are expected to be rejected by the seal gates.
A failed candidate shows that this pose failed in the original scene, not that the
EE cannot grasp the object.
"""
from dataclasses import replace
from ..catalog_types import CatalogRecipe, build_catalog_targets, dispatch_grasp

ENVIRONMENT = 'C1_2_DoughFlatten'
PRIMARY_PAIRS = (('bottle', '3F'), ('spatula', '3F'), ('spoon', '2F'), ('spoon', '3F'))
OBJECTS = ('bottle', 'spatula', 'spoon', 'tongs')
MISSING_TYPE_EES = {
    'bottle': ('2F', 'vac'),
    'spatula': ('2F', 'vac'),
    'spoon': ('vac',),
    'tongs': ('2F', '3F', 'vac'),
}
SIZES = {'bottle': (.0498, .05, .135),
         'spatula': (.0468, .25, .0051),
         'spoon': (.0562, .1715, .0299),
         'tongs': (.0307, .2584, .0253)}
HANDLE_FRACTION_Y = -.15
GEOMETRY_FIELDS = ('offset_fraction', 'offset_m', 'rotation_xyz_deg',
                   'preshape_aperture_m', 'preshape_closure_command')


def probe_recipe(object_id, ee, environment=ENVIRONMENT):
    if environment != ENVIRONMENT or ee not in MISSING_TYPE_EES.get(object_id, ()):
        raise ValueError('UNSUPPORTED_C1_2_PROBE')
    if (object_id, ee) in PRIMARY_PAIRS:
        raise ValueError('PRIMARY_RECIPE_MUST_NOT_BE_REPLACED')
    base = CatalogRecipe(object_id, 'c1_2', ee, SIZES[object_id],
                         two_finger_parallel_linkage=ee == '2F')
    if ee == 'vac':
        # Spoon +Y is its bowl (C3_1 convention); aim the cup at the handle for the spoon.
        y = -.25 if object_id == 'spoon' else 0.
        return replace(base, offset_fraction=(0., y, .5), preshape_closure_command=0.)
    if object_id == 'bottle':
        return replace(base, offset_fraction=(0., 0., 0.),
                       rotation_xyz_deg=(0., 90., 0.), preshape_closure_command=0.)
    # spatula 2F, tongs 2F/3F: top-down handle grasp across the handle width.
    return replace(base, offset_fraction=(0., HANDLE_FRACTION_Y, 0.),
                   rotation_xyz_deg=(0., 0., 90.), preshape_closure_command=0.)


def probe_2f_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, '2F', environment)


def probe_3f_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, '3F', environment)


def probe_vac_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, 'vac', environment)


def build_c1_2_probes_targets(T_WB, center_in_body_m, local_size_m, recipe):
    return build_catalog_targets(T_WB, center_in_body_m, local_size_m, recipe)


def grasp_c1_2_probes(context):
    return dispatch_grasp(context, context.object_id, context.recipe,
                          expected_id=context.recipe.object_id)
