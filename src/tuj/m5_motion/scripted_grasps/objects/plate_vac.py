"""Task-scoped plate vacuum grasp on the flat top face.

C1_1, C2_1 and C3_1 share the full-size plate geometry. C3_2 uses two smaller
instances, ``plate_a`` and ``plate_b``. All routes descend along the tool -z,
touch the top face, engage suction and lift. ``offset_fraction`` z=+0.5 targets
the top face; each environment keeps its calibrated press depth so the contact
gate does not stop with ``CONTACT_COUNT=0``.
"""
from tuj.m5_motion.scripted_grasps.catalog_types import (
    CatalogRecipe, build_catalog_targets, dispatch_grasp,
)


_TASK_IDS = {
    'C1_1_LegoSweep': 'c1_1',
    'C2_1_ObjectSorting': 'c2_1',
    'C3_1_ObjectSorting': 'c3_1',
    'C3_2_BreakfastTrayPreparation': 'c3_2',
}
_FULL_SIZE_M = (0.182334163, 0.181833528, 0.011091216)
_BREAKFAST_SIZE_M = (0.157261405, 0.157261434, 0.009592403)


def plate_vac_recipe(environment='C2_1_ObjectSorting', object_id='plate'):
    try:
        task_id = _TASK_IDS[environment]
    except KeyError as error:
        raise ValueError(f'UNSUPPORTED_PLATE_VAC_ENVIRONMENT: {environment}') from error
    allowed_ids = {'plate_a', 'plate_b'} if task_id == 'c3_2' else {'plate'}
    if object_id not in allowed_ids:
        raise ValueError(f'UNSUPPORTED_PLATE_VAC_OBJECT: {object_id}')
    return CatalogRecipe(
        object_id, task_id, 'vac',
        _BREAKFAST_SIZE_M if task_id == 'c3_2' else _FULL_SIZE_M,
        offset_fraction=(0., 0., .5),
        offset_m=(0., 0., -.0015 if task_id == 'c1_1' else -.0005),
        linear_lift_start=task_id == 'c1_1',
        arm_kp=300. if task_id == 'c1_1' else 150.,
        two_finger_parallel_linkage=False,
        post_grasp_arm_kp=300.,
        # Reach the suction command before the thin plate is pushed or tilted
        # by a slow four-second command ramp.
        close_duration_s=.04 if task_id == 'c1_1' else .2,
        # The plate collision mesh yields one aligned cup contact. Requiring
        # three contacts is appropriate for the lid mesh but rejects this
        # sustained, centered contact before attachment.
        minimum_vacuum_contact_count=1,
        contact_ticks=3 if task_id == 'c1_1' else 5,
        prelift_stabilization_s=.02 if task_id == 'c1_1' else .5,
        # C1_1's concave plate mesh reports deeper cup and initial table overlap
        # at an otherwise aligned top-face contact. Keep both exceptions scoped
        # to that environment and below their catalog safety caps.
        maximum_vacuum_attach_penetration_m=(
            .0035 if task_id == 'c1_1' else .002),
        maximum_support_separation_penetration_m=(
            .0045 if task_id == 'c1_1' else .002),
    )


def build_plate_vac_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    return build_catalog_targets(T_WB, center_in_body_m, local_size_m,
                                 recipe or plate_vac_recipe())


def grasp_plate_vac(context, object_id=None, recipe=None):
    target = object_id or getattr(context, 'object_id', 'plate')
    selected = recipe or getattr(context, 'recipe', None) or plate_vac_recipe(object_id=target)
    return dispatch_grasp(context, target, selected, expected_id=target)
