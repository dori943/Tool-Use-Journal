"""C1_1 plate suction; contact-gated rigid attach, not suction physics.

Validated in the isolated M5 continuation probe; selected explicitly because
the existing default plate dispatch still requests 2F.
"""
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe, dispatch_grasp


def plate_vacuum_recipe():
    return CatalogRecipe(
        'plate', 'c1_1', 'vac',
        (.1823341623518599, .18183352035820532, .011091216733562866),
        # The central surface is below the highest point of the plate rim.
        offset_fraction=(0., 0., .5), offset_m=(0., 0., -.0045),
        two_finger_parallel_linkage=False, post_grasp_arm_kp=300.,
        close_duration_s=.5, prelift_stabilization_s=0.,
    )


def grasp_plate_vacuum(context, object_id='plate', recipe=None):
    return dispatch_grasp(context, object_id, recipe or plate_vacuum_recipe(), expected_id='plate')
