"""C2_1 missing-EE candidates; geometry reuse without legacy gate relaxation.

The six pre-existing C2_1 routes keep their own calibrated recipes. New 2F
apple/bread/mug candidates borrow only pose/opening from C3_1; they retain
CatalogRecipe defaults (including 5 N, 5 degrees and no post-grasp gain change).
"""
from dataclasses import replace
from ..catalog_types import CatalogRecipe,build_catalog_targets,dispatch_grasp
from .c3_1_probes import SIZES
from .apple_2f import apple_2f_recipe
from .bread_2f import bread_2f_recipe
from .mug_2f import mug_2f_recipe

ENVIRONMENT='C2_1_ObjectSorting'
MISSING_TYPE_EES={'apple':('2F','vac'),'bread':('2F','vac'),'mug':('2F','vac'),
                  'plate':('2F','3F'),'spoon':('vac',)}
GEOMETRY_FIELDS=('offset_fraction','offset_m','rotation_xyz_deg',
                 'preshape_aperture_m','preshape_closure_command')
TWO_FINGER_SOURCES={'apple':apple_2f_recipe,'bread':bread_2f_recipe,'mug':mug_2f_recipe}


def probe_recipe(object_id,ee,environment=ENVIRONMENT):
    if environment!=ENVIRONMENT or ee not in MISSING_TYPE_EES.get(object_id,()):
        raise ValueError('UNSUPPORTED_C2_1_PROBE')
    base=CatalogRecipe(object_id,'c2_1',ee,SIZES[object_id],two_finger_parallel_linkage=ee=='2F')
    if ee=='2F' and object_id in TWO_FINGER_SOURCES:
        source=TWO_FINGER_SOURCES[object_id](environment=environment)
        return replace(base,**{key:getattr(source,key) for key in GEOMETRY_FIELDS})
    if object_id=='plate':
        return replace(base,offset_fraction=(.46,0.,0.),rotation_xyz_deg=(0.,90.,0.),preshape_closure_command=0.)
    y=.28 if object_id=='spoon' else .1 if object_id=='mug' else 0.
    return replace(base,offset_fraction=(0.,y,.5),preshape_closure_command=0.)


def probe_2f_recipe(object_id,environment=ENVIRONMENT):
    return probe_recipe(object_id,'2F',environment)


def probe_3f_recipe(object_id,environment=ENVIRONMENT):
    return probe_recipe(object_id,'3F',environment)


def probe_vac_recipe(object_id,environment=ENVIRONMENT):
    return probe_recipe(object_id,'vac',environment)


def build_c2_1_probes_targets(T_WB,center_in_body_m,local_size_m,recipe):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe)


def grasp_c2_1_probes(context):
    return dispatch_grasp(context,context.object_id,context.recipe,expected_id=context.recipe.object_id)
