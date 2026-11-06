"""C1_1 fixed-EE coverage candidates, available only to the coverage validator.

Scene targets (Table II denominator 5): plate, ladle, fork, scissors, bottle_distractor.
Production routes (plate 2F rim, plate vac) are never replaced.

Pattern: the C4_1 probes. Each candidate inherits a *validated* recipe of the same
shape class unchanged (force, contact gates, lift/hold criteria) and only swaps the
object identity, declared size, EE identity and pose/opening:

  long handled tools, 3F  <- fork_recipe          (C3_2 3F thin-handle pinch, validated)
  long handled tools, 2F  <- fork_2f_c3_2_recipe  (C3_2 2F handle pinch, validated 2F core)
  flat thin tool,   2F    <- spatula_c4_1_recipe  (C4_1 2F flat tool on a support, validated)
  everything without a same-class validated recipe (plate 3F, bottle, all vac)
                          <- CatalogRecipe defaults (5 N / 5 deg, lift 0.18 m, hold 5 s)

The first probe set (2026-11-05 a) used guessed poses on catalog defaults; its 2F
candidates closed along the tool length (yaw 90 is a 3F convention), so those
FAILs measure the probe, not the hand. Keep that run only as a superseded record.
"""
from dataclasses import replace
from ..catalog_types import CatalogRecipe, build_catalog_targets, dispatch_grasp
from .fork import fork_2f_c3_2_recipe, fork_recipe
from .spatula_c4_1 import spatula_c4_1_recipe

ENVIRONMENT = 'C1_1_LegoSweep'
OBJECTS = ('plate', 'ladle', 'fork', 'scissors', 'bottle_distractor')
PRIMARY_PAIRS = (('plate', '2F'), ('plate', 'vac'))
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
HANDLED = ('ladle', 'fork', 'scissors')
# Fields a probe may change relative to its source recipe (C4_1 contract).
ALLOWED = {'task_id', 'ee_id', 'two_finger_parallel_linkage', 'finger_attachment_policy',
           'preshape_closure_command', 'preshape_aperture_m', 'rotation_xyz_deg',
           'offset_fraction', 'offset_m'}


def source_recipe(object_id, ee):
    """Validated recipe this candidate inherits, re-bound to the C1_1 object."""
    size = SIZES[object_id]
    if ee == '3F' and object_id in HANDLED:
        src = fork_recipe()
    elif ee == '2F' and object_id in ('ladle', 'fork'):
        src = fork_2f_c3_2_recipe()
    elif ee == '2F' and object_id == 'scissors':
        src = spatula_c4_1_recipe()
    else:
        src = CatalogRecipe(object_id, 'c1_1', ee, size)
    return replace(src, object_id=object_id, expected_size_m=size, task_id='c1_1')


def probe_recipe(object_id, ee, environment=ENVIRONMENT):
    if environment != ENVIRONMENT or ee not in MISSING_TYPE_EES.get(object_id, ()):
        raise ValueError('UNSUPPORTED_C1_1_PROBE')
    if (object_id, ee) in PRIMARY_PAIRS:
        raise ValueError('PRIMARY_RECIPE_MUST_NOT_BE_REPLACED')
    src = source_recipe(object_id, ee)
    changes = dict(ee_id=ee, two_finger_parallel_linkage=(ee == '2F'),
                   finger_attachment_policy='FREE_HOLD_THEN_ATTACH')
    if ee == 'vac':
        changes.update(offset_fraction=(0., 0., .5), offset_m=(0., 0., 0.),
                       rotation_xyz_deg=(0., 0., 0.), preshape_closure_command=0.)
    elif object_id == 'plate':
        # Diameter exceeds the 3F opening; side rim pinch (as the C2_1/C3_1 plate probes).
        changes.update(offset_fraction=(.46, 0., 0.), offset_m=(0., 0., 0.),
                       rotation_xyz_deg=(0., 90., 0.), preshape_closure_command=0.)
    elif object_id == 'bottle_distractor':
        # Upright cylinder: top-down across the 50 mm diameter, upper third of the body.
        changes.update(offset_fraction=(0., 0., .35), offset_m=(0., 0., 0.),
                       rotation_xyz_deg=(0., 0., 0.), preshape_aperture_m=.07,
                       preshape_closure_command=0.)
    elif ee == '2F' and object_id == 'scissors':
        # 81 mm wide: the jaw cannot span the width; pinch the blade edge across
        # its 12 mm thickness (C4_1 wide-tool 2F probe pose).
        changes.update(offset_fraction=(.46, 0., 0.), rotation_xyz_deg=(0., 90., 0.))
    # ladle/fork 2F and ladle/fork/scissors 3F keep the source pose unchanged.
    return replace(src, **changes)


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
