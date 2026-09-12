"""C2_1 loaf: 3F enclosure across its short horizontal axis."""
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


def bread_recipe(environment='C2_1_ObjectSorting'):
    return CatalogRecipe('bread',_task(environment),'3F',(.083472,.129975,.069446),
        offset_fraction=(0.,0.,.12),offset_m=(-.002,0.,0.),two_finger_parallel_linkage=False,
        post_grasp_arm_kp=300.)
def build_bread_targets(T_WB,center_in_body_m,local_size_m,recipe=None):
    return build_catalog_targets(T_WB,center_in_body_m,local_size_m,recipe or bread_recipe())
def grasp_bread(context,object_id='bread',recipe=None):
    return dispatch_grasp(context,object_id,recipe or bread_recipe(),expected_id='bread')
