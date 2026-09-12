"""C3_1 mug: 2F pinch across the body diameter, away from the handle.

Registered so M4 can choose between 2F and 3F. The body is 70.6 mm across x,
comfortably inside the 85 mm stroke, and the fingers close along x while the
handle sits on +y, so the handle is not between the pads. The 3F recipe had to
fight the handle because its third finger reached toward +y (see objects/mug.py);
a two finger pinch across x does not have that problem.
"""
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe,build_catalog_targets,dispatch_grasp

_TASK_IDS = {'C2_1_ObjectSorting': 'c2_1', 'C3_1_ObjectSorting': 'c3_1'}


def _task(environment):
    try:
        return _TASK_IDS[environment]
    except KeyError as error:
        raise ValueError(f'UNSUPPORTED_ENVIRONMENT: {environment}') from error


def mug_2f_recipe(environment='C3_1_ObjectSorting'):
    return CatalogRecipe('mug',_task(environment),'2F',(.070630,.101936,.080475),
        # Grip the upper body wall, the same height the 3F enclosure uses.
        offset_fraction=(0.,0.,.1),offset_m=(0.,0.,0.),
        # Bias the contact band away from +y so neither pad lands on the handle.
        contact_region_min=(-.6,-.35,-.45),contact_region_max=(.6,.15,.5),
        preshape_aperture_m=.078,preshape_closure_command=-.60,
        two_finger_force_target_n=8.,lift_distance_m=.14)
def build_mug_2f_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe or mug_2f_recipe())
def grasp_mug_2f(context,object_id='mug',recipe=None):
    return dispatch_grasp(context,object_id,recipe or mug_2f_recipe(),expected_id='mug')
