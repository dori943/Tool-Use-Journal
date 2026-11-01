"""First missing-EE C3_2 candidates; validator-only, no production promotion.

New candidates share unchanged CatalogRecipe defaults for force, contact gates,
physics and lift/hold criteria. Existing twenty routes keep their own calibrated
recipes; this matrix is candidate coverage, not a controlled EE comparison.
Fruit 2F borrows only the greedy candidate pose/opening, not its 8 N force or
25 degree slip tolerance. Every finger candidate must hold freely before attach.
"""
from dataclasses import replace
from ..catalog_types import CatalogRecipe, build_catalog_targets, dispatch_grasp
from .plate_vac import PLATE_EXPECTED_SIZE_M
from .mug_c3_2 import MUG_EXPECTED_SIZE_M
from .fork import FORK_EXPECTED_SIZE_M
from .spoon import C3_2_SPOON_EXPECTED_SIZE_M
from .fruit import FRUIT_EXPECTED_SIZE_M

ENVIRONMENT = "C3_2_BreakfastTrayPreparation"
MISSING_TYPE_EES = {"plate": ("2F", "3F"), "mug": ("2F", "vac"),
                    "fork": ("vac",), "spoon": ("vac",), "fruit": ("2F", "vac")}
SIZES = {"plate": PLATE_EXPECTED_SIZE_M, "mug": MUG_EXPECTED_SIZE_M,
         "fork": FORK_EXPECTED_SIZE_M, "spoon": C3_2_SPOON_EXPECTED_SIZE_M,
         "fruit": FRUIT_EXPECTED_SIZE_M}


def probe_recipe(object_id, ee, environment=ENVIRONMENT):
    kind = object_id[:-2] if object_id.endswith(("_a", "_b")) else object_id
    if environment != ENVIRONMENT or ee not in MISSING_TYPE_EES.get(kind, ()):
        raise ValueError("UNSUPPORTED_C3_2_PROBE")
    base = CatalogRecipe(object_id, "c3_2", ee, SIZES[kind],
                         two_finger_parallel_linkage=(ee == "2F"))
    changes = {}
    if kind == "plate":
        # The full diameter exceeds either hand opening; propose a side rim
        # pinch across the thickness with all support/collision gates retained.
        changes.update(offset_fraction=(.46, 0., 0.), rotation_xyz_deg=(0., 90., 0.),
                       preshape_closure_command=0.)
    elif kind == "mug" and ee == "2F":
        # Near-maximum opening at the upper body. A collision/closure failure
        # is evidence about this candidate, not proof that handle grasps fail.
        changes.update(offset_fraction=(0., .1, .1), preshape_aperture_m=.084,
                       preshape_closure_command=-.7)
    elif kind == "fruit" and ee == "2F":
        changes.update(offset_fraction=(0., 0., .08), offset_m=(-.002, 0., 0.),
                       preshape_aperture_m=.084, preshape_closure_command=-.7)
    elif ee == "vac":
        y = -.15 if kind == "fork" else .28 if kind == "spoon" else .1 if kind == "mug" else 0.
        changes.update(offset_fraction=(0., y, .5), preshape_closure_command=0.)
    return replace(base, **changes)


def probe_2f_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, "2F", environment)


def probe_3f_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, "3F", environment)


def probe_vac_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, "vac", environment)


def build_c3_2_probes_targets(T_WB, center_in_body_m, local_size_m, recipe):
    return build_catalog_targets(T_WB, center_in_body_m, local_size_m, recipe)


def grasp_c3_2_probes(context):
    return dispatch_grasp(context, context.object_id, context.recipe, expected_id=context.recipe.object_id)
