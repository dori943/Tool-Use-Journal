"""C1_2 fixed-EE coverage candidates, available only to the coverage validator.

Scene tools (Table II denominator 4): bottle, spatula, spoon, tongs. Production
routes (bottle 3F, spatula 3F, spoon 2F/3F) are never replaced.

Pattern: the C4_1 probes — inherit a validated recipe of the same shape class
unchanged and swap only identity, declared size, EE identity and pose/opening:

  flat thin tools, 2F   <- spatula_c4_1_recipe  (C4_1 2F flat tool on a support, validated)
  handled tool,   3F    <- fork_recipe          (C3_2 3F thin-handle pinch, validated)
  bottle 2F, all vac    <- CatalogRecipe defaults (5 N / 5 deg, lift 0.18 m, hold 5 s)

The first probe set (2026-11-05 a) closed the 2F jaw along the tool length and
was rejected for fingertip/island collisions; those FAILs measure the probe.
"""
from dataclasses import replace
from ..catalog_types import CatalogRecipe, build_catalog_targets, dispatch_grasp
from .fork import fork_recipe
from .spatula_c4_1 import spatula_c4_1_recipe

ENVIRONMENT = 'C1_2_DoughFlatten'
OBJECTS = ('bottle', 'spatula', 'spoon', 'tongs')
PRIMARY_PAIRS = (('bottle', '3F'), ('spatula', '3F'), ('spoon', '2F'), ('spoon', '3F'))
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
ALLOWED = {'task_id', 'ee_id', 'two_finger_parallel_linkage', 'finger_attachment_policy',
           'preshape_closure_command', 'preshape_aperture_m', 'rotation_xyz_deg',
           'offset_fraction', 'offset_m'}


def source_recipe(object_id, ee):
    size = SIZES[object_id]
    if ee == '2F' and object_id in ('spatula', 'tongs'):
        src = spatula_c4_1_recipe()
    elif ee == '3F' and object_id == 'tongs':
        src = fork_recipe()
    else:
        src = CatalogRecipe(object_id, 'c1_2', ee, size)
    return replace(src, object_id=object_id, expected_size_m=size, task_id='c1_2')


def probe_recipe(object_id, ee, environment=ENVIRONMENT):
    if environment != ENVIRONMENT or ee not in MISSING_TYPE_EES.get(object_id, ()):
        raise ValueError('UNSUPPORTED_C1_2_PROBE')
    if (object_id, ee) in PRIMARY_PAIRS:
        raise ValueError('PRIMARY_RECIPE_MUST_NOT_BE_REPLACED')
    src = source_recipe(object_id, ee)
    changes = dict(ee_id=ee, two_finger_parallel_linkage=(ee == '2F'),
                   finger_attachment_policy='FREE_HOLD_THEN_ATTACH')
    if ee == 'vac':
        # Spoon +Y is its bowl; aim the cup at the handle for the spoon.
        y = -.25 if object_id == 'spoon' else 0.
        changes.update(offset_fraction=(0., y, .5), offset_m=(0., 0., 0.),
                       rotation_xyz_deg=(0., 0., 0.), preshape_closure_command=0.)
    elif object_id == 'bottle':
        # Upright cylinder: top-down across the 50 mm diameter, upper third.
        changes.update(offset_fraction=(0., 0., .35), offset_m=(0., 0., 0.),
                       rotation_xyz_deg=(0., 0., 0.), preshape_aperture_m=.07,
                       preshape_closure_command=0.)
    # spatula/tongs 2F and tongs 3F keep the source pose unchanged.
    return replace(src, **changes)


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
