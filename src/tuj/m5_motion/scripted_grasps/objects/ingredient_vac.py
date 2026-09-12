"""C2_2 sandwich ingredients: vacuum pick straight down onto the top face.

Every ingredient in this task is a flat slab wider than the 30 mm suction seal,
so one recipe covers all five; only the measured size changes per object.

Without a scripted route these picks fell to the generic M5 path, which
generated side approaches (strat_cheese_side_x_left and friends) at the
countertop height.  A suction cup cannot seal on the 4 mm edge of a cheese
slice, and reaching sideways at z = 0.922 drove the arm through the island:
forearm 137 mm inside island_island_group_top_2, wrist 46 mm inside
tomato_plate.  Fixing the approach to the top face is what removes that whole
class of strategy, the same way objects/plate_vac.py does for the plate.

The press depth, contact count and attachment tolerances are taken from the
validated plate recipe.  A single flat cup on a flat face makes exactly one
MuJoCo contact, so the vacuum gate stays at one contact rather than the
three a lid-shaped mesh produces.
"""
from tuj.m5_motion.scripted_grasps.catalog_types import (
    CatalogRecipe, build_catalog_targets, dispatch_grasp,
)


_TASK_IDS = {'C2_2_SandwichAssembly': 'c2_2'}

# 실측 (output/c2_2/m5/initial_world.json).
_SIZES_M = {
    'turkey_1': (0.080000, 0.050000, 0.002520),
    'cheese_1': (0.063200, 0.061300, 0.004000),
    'tomato_slice': (0.061100, 0.060700, 0.004740),
    'bread_a': (0.113100, 0.113300, 0.022920),
    'bread_b': (0.113100, 0.113300, 0.022920),
}


def ingredient_vac_recipe(environment='C2_2_SandwichAssembly', object_id='turkey_1'):
    try:
        task_id = _TASK_IDS[environment]
    except KeyError as error:
        raise ValueError(f'UNSUPPORTED_INGREDIENT_ENVIRONMENT: {environment}') from error
    try:
        size = _SIZES_M[object_id]
    except KeyError as error:
        raise ValueError(f'UNSUPPORTED_INGREDIENT_OBJECT: {object_id}') from error
    return CatalogRecipe(
        object_id, task_id, 'vac', size,
        # 윗면을 겨눈다. 도구 -z 로 내려와 접촉 후 흡착한다.
        offset_fraction=(0., 0., .5),
        offset_m=(0., 0., -.0005),
        two_finger_parallel_linkage=False,
        post_grasp_arm_kp=300.,
        # 얇은 슬라이스를 4초 램프로 밀지 않도록 흡착 명령을 빨리 세운다.
        close_duration_s=.2,
        # 평평한 면에 평평한 컵 하나는 MuJoCo 접촉점이 하나다.
        minimum_vacuum_contact_count=1,
        contact_ticks=5,
        prelift_stabilization_s=.5,
        maximum_vacuum_attach_penetration_m=.002,
        maximum_support_separation_penetration_m=.002,
    )


def build_ingredient_vac_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    return build_catalog_targets(T_WB, center_in_body_m, local_size_m,
                                 recipe or ingredient_vac_recipe())


def grasp_ingredient_vac(context, object_id=None, recipe=None):
    # entry.function() 은 컨텍스트만 넘기므로 대상은 컨텍스트에서 읽는다.
    # 한 모듈이 다섯 재료를 담당해서 고정 기본값을 두면 컨텍스트가 초기화된
    # 레시피와 어긋난다 (plate_vac 이 plate/plate_a/plate_b 를 다루는 방식).
    target = object_id or getattr(context, 'object_id', 'turkey_1')
    selected = (recipe or getattr(context, 'recipe', None)
                or ingredient_vac_recipe(object_id=target))
    return dispatch_grasp(context, target, selected, expected_id=target)
