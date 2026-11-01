"""C4_2 alternative grasp recipes and four remaining validator-only probes.

The registry promotes only candidates backed by saved controller trials. Recipe
geometry is identical to the first-candidate experiment. Preserve the source
object's lift/hold/slip thresholds and material/force settings. A failed
candidate does not prove physical impossibility. Finger alternatives use
unassisted acquisition; vacuum uses the existing contact-gated suction model.
No changes are made to M1/M4 inference.
"""
from dataclasses import replace
from importlib import import_module

from ..catalog_types import build_catalog_targets, dispatch_grasp

ENVIRONMENT = "C4_2_DiagonalFitPacking"
PRIMARY_EES = {"rolling_pin": "2F", "baguette": "2F", "whisk": "2F",
               "cereal": "2F", "milk": "3F", "lid": "vac"}


def probe_recipe(object_id, ee, environment=ENVIRONMENT):
    if environment != ENVIRONMENT or object_id not in PRIMARY_EES:
        raise ValueError("UNSUPPORTED_C4_2_PROBE")
    if ee == PRIMARY_EES[object_id]:
        raise ValueError("PRIMARY_RECIPE_MUST_NOT_BE_REPLACED")
    module = import_module(f"{__package__}.{object_id}")
    source = getattr(module, f"{object_id}_recipe")()
    changes = dict(ee_id=ee, finger_attachment_policy="FREE_HOLD_THEN_ATTACH",
                   two_finger_parallel_linkage=(ee == "2F"),
                   contact_region_endpoint_policy=None, open_hand_clearance_axis=None)
    if ee == "vac":
        # Keep the whisk's handle region; all other cups target the top center.
        y = -.28 if object_id == "whisk" else 0.
        changes.update(offset_fraction=(0., y, .5), offset_m=(0., 0., 0.),
                       rotation_xyz_deg=(0., 0., 0.))
    elif object_id == "lid":
        # Enclosure of the entire lid exceeds both hands' opening. Propose a
        # side pinch of the thin +X rim, with the closing direction vertical.
        changes.update(offset_fraction=(.46, 0., 0.), offset_m=(0., 0., 0.),
                       rotation_xyz_deg=(0., 90., 0.), preshape_aperture_m=.030,
                       preshape_closure_command=0.)
    elif ee == "3F":
        changes.update(preshape_closure_command=0.)
    elif ee == "2F" and object_id == "milk":
        changes.update(preshape_aperture_m=.058, preshape_closure_command=0.)
    return replace(source, **changes)


def probe_2f_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, "2F", environment)


def probe_3f_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, "3F", environment)


def probe_vac_recipe(object_id, environment=ENVIRONMENT):
    return probe_recipe(object_id, "vac", environment)


def build_c4_2_probes_targets(T_WB, center_in_body_m, local_size_m, recipe):
    return build_catalog_targets(T_WB, center_in_body_m, local_size_m, recipe)


def grasp_c4_2_probes(context):
    return dispatch_grasp(context, context.object_id, context.recipe,
                          expected_id=context.recipe.object_id)
