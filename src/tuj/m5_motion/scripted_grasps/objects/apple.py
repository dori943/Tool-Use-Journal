"""C2_1 apple: 3F enclosure around the broad upper body."""
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe,build_catalog_targets,dispatch_grasp

# 0912: 같은 자산이 C2_1 과 C3_1 (환경 파일이 클래스명과 render_camera 두 줄만
# 다른 같은 씬) 양쪽에 등록된다. task_id 를 'c2_1' 로 고정해 두면 C3_1 실행이
# 전부 c2_1 로 기록돼 채점 때 섞인다. 레지스트리가 환경을 넘겨 주므로 매핑한다.
_TASK_IDS = {'C2_1_ObjectSorting': 'c2_1', 'C3_1_ObjectSorting': 'c3_1'}


def _task(environment):
    try:
        return _TASK_IDS[environment]
    except KeyError as error:
        raise ValueError(f'UNSUPPORTED_ENVIRONMENT: {environment}') from error


def apple_recipe(environment='C2_1_ObjectSorting'):
    return CatalogRecipe('apple',_task(environment),'3F',(.075783,.075419,.075519),
        offset_fraction=(0.,0.,.15),offset_m=(-.002,0.,0.),two_finger_parallel_linkage=False)
def build_apple_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe or apple_recipe())
def grasp_apple(context,object_id='apple',recipe=None):
    return dispatch_grasp(context,object_id,recipe or apple_recipe(),expected_id='apple')
