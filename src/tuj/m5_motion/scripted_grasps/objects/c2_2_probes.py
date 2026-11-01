"""First C2_2 finger candidates; validator-only, with unchanged physical gates.

The five existing vacuum recipes are not replaced. New recipes change only pose
and opening relative to CatalogRecipe defaults; an unsuccessful candidate does
not establish impossibility for every grasp pose or strategy.
"""
from dataclasses import replace
from ..catalog_types import CatalogRecipe, build_catalog_targets, dispatch_grasp

ENVIRONMENT = 'C2_2_SandwichAssembly'
SIZES = {'bread_a': (.113114620, .113348011, .022920362),
         'bread_b': (.113114620, .113348011, .022920362),
         'turkey_1': (.079998568, .050000000, .002515832),
         'cheese_1': (.063230542, .061322246, .004000000),
         'tomato_slice': (.061140000, .060672120, .004736000)}
OBJECTS = tuple(SIZES)


def probe_recipe(object_id, ee, environment=ENVIRONMENT):
    if environment != ENVIRONMENT or object_id not in SIZES or ee not in ('2F','3F'):
        raise ValueError('UNSUPPORTED_C2_2_PROBE')
    base = CatalogRecipe(object_id, 'c2_2', ee, SIZES[object_id],
                         two_finger_parallel_linkage=ee == '2F')
    if ee == '2F' and object_id.startswith('bread_'):
        # A 113 mm loaf exceeds the 85 mm stroke. Propose a side edge pinch
        # across its 23 mm thickness, subject to normal reach/support checks.
        return replace(base, offset_fraction=(.46,0.,0.),
                       rotation_xyz_deg=(0.,90.,0.), preshape_closure_command=0.)
    if ee == '2F':
        # Close across each slice's shorter Y extent, with room to straddle it.
        # No support penetration exemption or contact-gate relaxation is used.
        return replace(base, offset_fraction=(0.,0.,.5),
                       rotation_xyz_deg=(0.,0.,90.),
                       preshape_aperture_m=SIZES[object_id][1]+.012,
                       preshape_closure_command=-.55)
    # Open 3F enclosure, centered at the upper surface of each slab.
    return replace(base, offset_fraction=(0.,0.,.5), preshape_closure_command=0.)


def probe_2f_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id,'2F',environment)


def probe_3f_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id,'3F',environment)


def build_c2_2_probes_targets(T_WB, center_in_body_m, local_size_m, recipe):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe)


def grasp_c2_2_probes(context):
    return dispatch_grasp(context,context.object_id,context.recipe,expected_id=context.recipe.object_id)
