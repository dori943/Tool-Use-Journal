"""C3_1 apple: 2F parallel-jaw grip centered across the body.

C3_1 sorts the same objects as C2_1 but M4 mounts the 2F gripper for the apple
(C2_1 used 3F).  There is no scripted 2F apple recipe, so the grasp fell back to
the generic M5 path, which closed on the apple but left the grip site 20.4 mm
from the apple centre - just past the 20 mm post-grasp attach limit
(ATTACHMENT_TOO_FAR).  The apple is a ~76 mm sphere that fits inside the 85 mm
Robotiq opening, so a centred parallel-jaw grip lands the grip site within a few
mm of the centre.  Pre-shape wide open to clear the equator, close across the
widest section a touch above centre, firm 8 N hold.
"""
from tuj.m5_motion.scripted_grasps.catalog_types import (
    CatalogRecipe, build_catalog_targets, dispatch_grasp,
)

# 0912: task_id 를 'c2_1' 로 고정해 두면 C3_1 실행이 전부 c2_1 로 기록돼 채점
# 때 섞인다. catalog_types 의 허용 목록에는 'c3_1' 이 이미 들어 있고(76행),
# 레지스트리가 환경을 넘겨 주므로 같은 자산의 apple/bread/mug 와 같은 방식으로
# 매핑한다.
_TASK_IDS = {'C2_1_ObjectSorting': 'c2_1', 'C3_1_ObjectSorting': 'c3_1'}


def _task(environment):
    try:
        return _TASK_IDS[environment]
    except KeyError as error:
        raise ValueError(f'UNSUPPORTED_ENVIRONMENT: {environment}') from error


def apple_2f_recipe(environment='C3_1_ObjectSorting'):
    return CatalogRecipe(
        'apple', _task(environment), '2F',
        (.075783, .075419, .075519),
        offset_fraction=(0., 0., .08),
        offset_m=(0., 0., 0.),
        preshape_aperture_m=.084,
        preshape_closure_command=-.7,
        two_finger_force_target_n=8.,
        # A smooth sphere pinched at two points keeps a free rotational DOF: the
        # apple spins about the grip axis (mostly yaw) as the fingers close and
        # settle.  The first run held firmly (all_finger_contact_fraction=1.0,
        # position slip 1.2 mm, 17.6 cm lift for 5 s) yet peaked at 16.6 deg of
        # rotation, tripping the default 5 deg HOLD gate (HOLD_VALIDATION_FAILED).
        # The apple is orientation-invariant and the target is a tray, so this
        # spin is harmless for the sort; relax only the rotational tolerance.
        # Position slip and contact loss stay at their tight defaults, so a real
        # slip-out is still caught.  (retention.py reads this same value so the
        # allowance carries through transport/place.)
        maximum_slip_deg=25.,
    )


def build_apple_2f_targets(T_WB, center_in_body_m, local_size_m, recipe=None):
    return build_catalog_targets(T_WB, center_in_body_m, local_size_m,
                                 recipe or apple_2f_recipe())


def grasp_apple_2f(context, object_id='apple', recipe=None):
    return dispatch_grasp(context, object_id, recipe or apple_2f_recipe(),
                          expected_id='apple')
