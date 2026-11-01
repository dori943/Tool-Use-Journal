"""First C4_1 fixed-EE candidates, isolated from production planning.

Existing tool recipes supply unchanged physical limits and lift/hold criteria.
The added tools inherit spatula_a's limits with their declared scene dimensions.
Card uses unmodified CatalogRecipe defaults. Candidate differences are geometry,
EE identity, opening and truthful task metadata only. A failure establishes only
that this candidate failed in the original scene (the card remains in the gap).
"""
from dataclasses import replace
from ..catalog_types import CatalogRecipe, build_catalog_targets, dispatch_grasp
from .spatula_c4_1 import spatula_c4_1_recipe
from .spatula_b_c4_1 import spatula_b_c4_1_recipe

ENVIRONMENT = "C4_1_IntervalFitExtraction"
OBJECTS = ("tool_2_spatula_a", "tool_3_spatula_b", "tool_4_spatula_c",
           "tool_5_cutting_board", "card")
PRIMARY_PAIRS = (("tool_2_spatula_a", "2F"), ("tool_3_spatula_b", "2F"))


def source_recipe(object_id):
    if object_id == "tool_2_spatula_a":
        return spatula_c4_1_recipe()
    if object_id == "tool_3_spatula_b":
        return spatula_b_c4_1_recipe()
    if object_id in ("tool_4_spatula_c", "tool_5_cutting_board"):
        size = (.056, .220, .011) if object_id == "tool_4_spatula_c" else (.072, .220, .016)
        return replace(spatula_c4_1_recipe(), object_id=object_id, expected_size_m=size, task_id="c4_1")
    if object_id == "card":
        return CatalogRecipe("card", "c4_1", "2F", (.054, .032, .0012))
    raise ValueError("UNSUPPORTED_C4_1_OBJECT")


def probe_recipe(object_id, ee, environment=ENVIRONMENT):
    if environment != ENVIRONMENT or ee not in ("2F", "3F", "vac"):
        raise ValueError("UNSUPPORTED_C4_1_PROBE")
    if (object_id, ee) in PRIMARY_PAIRS:
        raise ValueError("PRIMARY_RECIPE_MUST_NOT_BE_REPLACED")
    source = source_recipe(object_id)
    changes = dict(task_id="c4_1", ee_id=ee, two_finger_parallel_linkage=(ee == "2F"),
                   finger_attachment_policy="FREE_HOLD_THEN_ATTACH", preshape_closure_command=0.)
    if ee == "vac":
        changes.update(offset_fraction=(0., 0., .5), offset_m=(0., 0., 0.),
                       rotation_xyz_deg=(0., 0., 0.))
    elif object_id == "card":
        # Close across the shorter, 32 mm side; do not move the card out of its gap.
        changes.update(rotation_xyz_deg=(0., 0., 90.), preshape_aperture_m=.049)
    elif ee == "2F":
        # Added tools exceed the installed 2F opening. Propose a side edge pinch
        # across their thickness; retain the full approach/collision checks.
        changes.update(offset_fraction=(.46, 0., 0.), rotation_xyz_deg=(0., 90., 0.))
    return replace(source, **changes)


def probe_2f_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, "2F", environment)


def probe_3f_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, "3F", environment)


def probe_vac_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, "vac", environment)


def build_c4_1_probes_targets(T_WB, center_in_body_m, local_size_m, recipe):
    return build_catalog_targets(T_WB, center_in_body_m, local_size_m, recipe)


def grasp_c4_1_probes(context):
    return dispatch_grasp(context, context.object_id, context.recipe, expected_id=context.recipe.object_id)
