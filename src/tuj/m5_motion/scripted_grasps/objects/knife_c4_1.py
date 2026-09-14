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
        # This kitchen gripper's max fingerpad separation is ~47.5 mm and its
        # pre-shape gains are fixed (closure_kp is applied only AFTER pre-shape).
        # A target below the max makes the aperture-feedback loop CLOSE toward it
        # and overshoot the closed joint limit (GRIPPER_JOINT_LIMIT_EXCEEDED at
        # .035/.045); a target far above the max never settles (.055 -> 7.5 mm
        # off).  Target just ABOVE the max so the loop only ever OPENS (no closed
        # overshoot) and saturates at ~47.5 mm, within the 3 mm settle band of 49.
        preshape_aperture_m=.049,
        preshape_closure_command=0.,
        two_finger_force_target_n=6.,
        # A 2.5 mm blade gripped across its width contacts only a thin edge
        # strip; a firm squeeze rolls it and the antipodal-normal opposition
        # collapses to ~0.1 even though both pads still clamp it (5-8 N, ~1.1 cm
        # span) through the pre-lift hold.  Close more gently so the blade stays
        # flatter, and relax the opposition gate so this real grip is not
        # rejected as CONTACT_LOST_BEFORE_LIFT.
        close_duration_s=8.,
        minimum_normal_opposition=.12,
        # High pad friction (sliding, torsional, rolling) with the 6-D contact
        # model so the light 2.5 mm blade rides up with the pads on lift instead
        # of sliding off the thin edge (CONTACT_LOST_DURING_LIFT).
        fingerpad_friction=(2.0, 0.1, 0.02),
    )


def build_knife_c4_1_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    return build_catalog_targets(T_WB, center_in_body_m, local_size_m,
                                 recipe or knife_c4_1_recipe())


def grasp_knife_c4_1(context, object_id='tool_1_knife', recipe=None):
    return dispatch_grasp(context, object_id, recipe or knife_c4_1_recipe(),
                          expected_id='tool_1_knife')
