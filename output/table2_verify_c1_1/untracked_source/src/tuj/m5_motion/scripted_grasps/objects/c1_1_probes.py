"""C1_1 fixed-EE coverage candidates, available only to the coverage validator.

Scene targets (Table II denominator 5): plate, ladle, fork, scissors, bottle_distractor.
Two production routes exist (plate 2F rim, plate vac) and are left untouched. The
other 13 object/EE pairs get one geometric candidate each: pose/opening only, with
CatalogRecipe defaults (5 N / 5 deg hold, lift 0.18 m, minimum 0.10 m, hold 5 s,
FREE_HOLD_THEN_ATTACH). Sizes are the assets' body-local ``reg_bbox`` extents.

Pose assumptions (verify with the static validator before trusting a FAIL):
  * tool long axis is body +Y, handle on the -Y side (same convention as the fork
    and spatula recipes); finger grasps take the handle at -0.15 of the length and
    yaw 90 deg so the fingers close across the handle width (X), as fork_recipe does;
  * bottle is an upright cylinder; fingers use a side pinch (rotation about Y);
  * vacuum tries the top surface centre (0, 0, +0.5).
A failed candidate shows that this pose failed in the original scene, not that the
EE cannot grasp the object.
"""
from dataclasses import replace
from ..catalog_types import CatalogRecipe, build_catalog_targets, dispatch_grasp

ENVIRONMENT = 'C1_1_LegoSweep'
PRIMARY_PAIRS = (('plate', '2F'), ('plate', 'vac'))
OBJECTS = ('plate', 'ladle', 'fork', 'scissors', 'bottle_distractor')
MISSING_TYPE_EES = {
    'plate': ('3F',),
    'ladle': ('2F', '3F', 'vac'),
    'fork': ('2F', '3F', 'vac'),
    'scissors': ('2F', '3F', 'vac'),
    'bottle_distractor': ('2F', '3F', 'vac'),
}
# Body-local reg_bbox full extents of the C1_1 assets (environments/assets/objects).
SIZES = {'plate': (.1818, .1818, .0111),
         'ladle': (.0635, .2081, .1102),
         'fork': (.0357, .2195, .0264),
         'scissors': (.0811, .1792, .0118),
         'bottle_distractor': (.0498, .05, .135)}
HANDLE_FRACTION_Y = -.15
GEOMETRY_FIELDS = ('offset_fraction', 'offset_m', 'rotation_xyz_deg',
                   'preshape_aperture_m', 'preshape_closure_command')


def probe_recipe(object_id, ee, environment=ENVIRONMENT):
    if environment != ENVIRONMENT or ee not in MISSING_TYPE_EES.get(object_id, ()):
        raise ValueError('UNSUPPORTED_C1_1_PROBE')
    if (object_id, ee) in PRIMARY_PAIRS:
        raise ValueError('PRIMARY_RECIPE_MUST_NOT_BE_REPLACED')
    base = CatalogRecipe(object_id, 'c1_1', ee, SIZES[object_id],
                         two_finger_parallel_linkage=ee == '2F')
    if ee == 'vac':
        return replace(base, offset_fraction=(0., 0., .5), preshape_closure_command=0.)
    if object_id == 'plate':
        # Diameter exceeds the 3F opening; attempt a side rim pinch (as C2_1/C3_1 probes).
        return replace(base, offset_fraction=(.46, 0., 0.),
                       rotation_xyz_deg=(0., 90., 0.), preshape_closure_command=0.)
    if object_id == 'bottle_distractor':
        # Upright cylinder: side pinch at body centre, closing across the 50 mm diameter.
        return replace(base, offset_fraction=(0., 0., 0.),
                       rotation_xyz_deg=(0., 90., 0.), preshape_closure_command=0.)
    # Long tools: top-down handle grasp, fingers across the handle width.
    return replace(base, offset_fraction=(0., HANDLE_FRACTION_Y, 0.),
                   rotation_xyz_deg=(0., 0., 90.), preshape_closure_command=0.)


def probe_2f_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, '2F', environment)


def probe_3f_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, '3F', environment)


def probe_vac_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, 'vac', environment)


def build_c1_1_probes_targets(T_WB, center_in_body_m, local_size_m, recipe):
    return build_catalog_targets(T_WB, center_in_body_m, local_size_m, recipe)


def grasp_c1_1_probes(context):
    return dispatch_grasp(context, context.object_id, context.recipe,
                          expected_id=context.recipe.object_id)
