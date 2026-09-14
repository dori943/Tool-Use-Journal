"""C3_1 loaf: 2F pinch across its short horizontal axis.

The 3F enclosure (objects/bread.py) is the validated grasp for this loaf. This
2F variant exists so M4 can actually choose between the two hands: with only a
3F entry registered, constrain_task_request narrows feasible_ee to ['3F'] and
the planner has no decision left to make, which makes an EE-selection score
meaningless (c2_1 measured 3F for bread/apple/mug with exactly one candidate).

The pinch is tight by construction: the loaf is 83.5 mm across its short axis
and the Robotiq 85 stroke is 85 mm, so the preshape leaves about 0.75 mm per
finger. That narrow margin is the physical reason 3F is the better hand here,
and it is what this recipe is meant to expose rather than hide.
"""
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe,build_catalog_targets,dispatch_grasp

_TASK_IDS = {'C2_1_ObjectSorting': 'c2_1', 'C3_1_ObjectSorting': 'c3_1'}


def _task(environment):
    try:
        return _TASK_IDS[environment]
    except KeyError as error:
        raise ValueError(f'UNSUPPORTED_ENVIRONMENT: {environment}') from error


def bread_2f_recipe(environment='C3_1_ObjectSorting'):
    return CatalogRecipe('bread',_task(environment),'2F',(.083472,.129975,.069446),
        # Grip the upper body, same height the 3F enclosure uses.
        offset_fraction=(0.,0.,.12),offset_m=(0.,0.,0.),
        # Only the middle of the loaf counts: a finger that catches an end cap
        # is not an opposed pinch.
        contact_region_min=(-.6,-.25,-.6),contact_region_max=(.6,.25,.6),
        preshape_aperture_m=.0845,preshape_closure_command=-.74,
        # 0912: 파지 후 팔 게인. 기본값 150 이면 이 로프를 든 자세에서 팔이
        # 지령 관절각에 5.4 mrad 못 미친 채로 멈춘다 (LIFT 가 0.18 목표에서
        # 0.136 까지만 올라가고 5초 HOLD 동안 0.167 까지 기어올라감). 그
        # 잔차가 EEF 에서 6.3 mm 로 나와 2 cm 이송 목표 허용치 5 mm 를
        # 넘긴다. 같은 로프의 3F 레시피(objects/bread.py)가 이미 같은 이유로
        # 300 을 쓰고 있어 값을 맞춘다.
        post_grasp_arm_kp=300.,
        two_finger_force_target_n=5.,lift_distance_m=.18)
def build_bread_2f_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe or bread_2f_recipe())
def grasp_bread_2f(context,object_id='bread',recipe=None):
    return dispatch_grasp(context,object_id,recipe or bread_2f_recipe(),expected_id='bread')
