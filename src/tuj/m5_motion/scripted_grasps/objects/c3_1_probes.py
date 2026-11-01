"""Missing C3_1 EE candidates, available only to the coverage validator.

Only grasp pose/opening are proposed. Default force, collision/attachment gates,
lift and hold criteria stay intact. The nine legacy recipes remain separate.
"""
from dataclasses import replace
from ..catalog_types import CatalogRecipe, build_catalog_targets, dispatch_grasp

ENVIRONMENT = 'C3_1_ObjectSorting'
MISSING_TYPE_EES = {'apple': ('vac',), 'bread': ('vac',), 'mug': ('vac',),
                    'plate': ('2F', '3F'), 'spoon': ('vac',)}
# Measured local collision bounds of the current C3_1 assets (seed-0 preflight).
SIZES = {'apple': (.075783, .075419, .075519),
         'bread': (.083472, .129975, .069446),
         'mug': (.070630234, .101935913, .080475149),
         'plate': (.182334163, .181833528, .011091216),
         'spoon': (.056231103, .171516523, .029876016)}


def probe_recipe(object_id, ee, environment=ENVIRONMENT):
    if environment != ENVIRONMENT or ee not in MISSING_TYPE_EES.get(object_id, ()):
        raise ValueError('UNSUPPORTED_C3_1_PROBE')
    base = CatalogRecipe(object_id, 'c3_1', ee, SIZES[object_id],
                         two_finger_parallel_linkage=ee == '2F')
    if object_id == 'plate':
        # Diameter exceeds either hand opening; attempt a side rim pinch.
        # This pose still undergoes the normal IK/support/collision checks.
        return replace(base, offset_fraction=(.46, 0., 0.),
                       rotation_xyz_deg=(0., 90., 0.), preshape_closure_command=0.)
    # Spoon +Y is its bowl. Mug +Y is its handle side; propose the upper rim.
    y = .28 if object_id == 'spoon' else .1 if object_id == 'mug' else 0.
    return replace(base, offset_fraction=(0., y, .5), preshape_closure_command=0.)


def probe_2f_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, '2F', environment)


def probe_3f_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, '3F', environment)


def probe_vac_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, 'vac', environment)


def build_c3_1_probes_targets(T_WB, center_in_body_m, local_size_m, recipe):
    return build_catalog_targets(T_WB, center_in_body_m, local_size_m, recipe)


def grasp_c3_1_probes(context):
    return dispatch_grasp(context, context.object_id, context.recipe, expected_id=context.recipe.object_id)
