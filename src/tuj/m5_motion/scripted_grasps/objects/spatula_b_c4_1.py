"""C4_1 (IntervalFitExtraction) tool pick: 2F grip on the flat spatula_b.

Same situation and rationale as objects/spatula_c4_1.py.  M5 record dims
(x=width, y=length, z=thickness) = (0.040, 0.220, 0.007).  The 2F jaw closes
across the 40 mm width (first axis, default), body-centre grip for a stable
lift.  task_id 'c4_2' is inert (dispatch keys on object_id).
"""
from tuj.m5_motion.scripted_grasps.catalog_types import (
    CatalogRecipe, build_catalog_targets, dispatch_grasp,
)


def spatula_b_c4_1_recipe():
    return CatalogRecipe(
        'tool_3_spatula_b', 'c4_2', '2F',
        (0.040, 0.220, 0.007),
        # Grip near the BACK end (toward -length) rather than the centre.  For
        # extraction the tool tip (+length) is inserted into the deep gap; a
        # centre grip only reaches ~half the tool ahead of the hand, so the eef
        # ends up inside the entrance and the forearm/wrist collide with the
        # island top and appliances.  Holding near the back extends the full
        # ~200 mm blade forward, keeping the eef outside the entrance where the
        # arm has room.  (Extraction anchors the +length end as the tip; this
        # grip is on the opposite end.)
        #
        # -0.35 (77 mm behind centre -> grip-to-tip 187 mm) still left the eef,
        # and the roughly colinear wrist chain behind it, sitting right at the
        # channel mouth (eef x~2.571 vs entrance x=2.57): the tool tip reaches
        # the card at ~183 mm depth exactly when the hand is at the entrance, so
        # wrist3/wrist2 grazed the appliances (~11 mm) as the insert IK pushed
        # them into the 52 mm channel.  Grip further back (-0.44 -> 97 mm behind
        # centre, ~13 mm from the back end -> grip-to-tip ~207 mm) so the tip
        # still reaches the card while the eef and the whole wrist chain sit
        # ~20 mm OUT in front of the mouth, clearing the appliances without
        # allowing any arm<->appliance contact.
        offset_fraction=(0., -0.44, 0.),
        offset_m=(0., 0., 0.),
        preshape_aperture_m=.049,
        preshape_closure_command=0.,
        # HOLD force only: the end grip sits ~200 mm behind the tip, so the
        # extraction hook/drag load levers a finger off the thin blade
        # (SCRIPTED_GRASP_CONTACT_LOST).  two_finger_force_target_n is the
        # RETENTION hold target (retention.before_tick), NOT the CLOSE gate, so
        # raising it firms the hold DURING the drag without touching the delicate
        # CLOSE (a prior attempt that also raised fingerpad_friction broke CLOSE
        # with GRASP_CONTACT_NOT_STABLE -- friction is left at the validated
        # value; only the hold force is raised).
        two_finger_force_target_n=18.,
        # Thin flat tool on the counter: close gently and relax the antipodal
        # opposition gate so a firm-squeeze roll does not reject the real grip
        # (see objects/knife_c4_1.py for the full rationale).
        close_duration_s=8.,
        minimum_normal_opposition=.12,
        fingerpad_friction=(2.0, 0.1, 0.02),
    )


def build_spatula_b_c4_1_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    return build_catalog_targets(T_WB, center_in_body_m, local_size_m,
                                 recipe or spatula_b_c4_1_recipe())


def grasp_spatula_b_c4_1(context, object_id='tool_3_spatula_b', recipe=None):
    return dispatch_grasp(context, object_id, recipe or spatula_b_c4_1_recipe(),
                          expected_id='tool_3_spatula_b')
