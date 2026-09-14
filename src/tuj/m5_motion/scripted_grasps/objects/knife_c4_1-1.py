"""C4_1 (IntervalFitExtraction) tool pick: 2F grip on the flat knife.

Same situation and rationale as objects/spatula_c4_1.py: the knife lies flush on
the kitchen island, so the generic M5 pick collides the 2F fingertips with the
island top.  This scripted catalog recipe routes the pick through the path that
treats the island top as the support surface.

M5 record dims (x=width, y=length, z=thickness) = (0.0162, 0.2212, 0.0025): a
long, narrow, thin blade.  The 2F jaw closes across the 16 mm width (first axis,
default), body-centre grip for a stable lift.  task_id 'c4_2' is inert (dispatch
keys on object_id).
"""
from tuj.m5_motion.scripted_grasps.catalog_types import (
    CatalogRecipe, build_catalog_targets, dispatch_grasp,
)


def knife_c4_1_recipe():
    return CatalogRecipe(
        'tool_1_knife', 'c4_2', '2F',
        (0.0162, 0.2212, 0.0025),
        offset_fraction=(0., 0., 0.),
        offset_m=(0., 0., 0.),
        # Pre-shape aperture must sit inside this gripper's reachable window:
        # too tight (.035) drove the aperture-feedback loop into the closed joint
        # limit (GRIPPER_JOINT_LIMIT_EXCEEDED); too wide (.055) saturated at the
        # gripper's max fingerpad separation (~47.5 mm) and never settled
        # (PRESHAPE_APERTURE_NOT_SETTLED). .045 is comfortably reachable and well
        # clear of the 16 mm blade.
        preshape_aperture_m=.045,
        preshape_closure_command=0.,
        two_finger_force_target_n=6.,
    )


def build_knife_c4_1_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    return build_catalog_targets(T_WB, center_in_body_m, local_size_m,
                                 recipe or knife_c4_1_recipe())


def grasp_knife_c4_1(context, object_id='tool_1_knife', recipe=None):
    return dispatch_grasp(context, object_id, recipe or knife_c4_1_recipe(),
                          expected_id='tool_1_knife')
