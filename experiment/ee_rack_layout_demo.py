"""
EE Rack Layout Demo  (robosuite 1.5.2 / mujoco 3.3.0)
============================================================================
로봇(UR5e) 주변에 3개의 엔드이펙터(3F/Soft, Vacuum, 2F)와 곡선형 Rack 을
배치하는 "장면 구성" 데모. 실제 교체/체결/제어/경로생성은 하지 않는다.

설계 원칙 (요구사항 2):
  - 기존 Python / XML / asset / registry 파일은 하나도 수정하지 않는다.
  - 기존 모델(로봇 측: UR5e 내장, EE: 내장 Jaco / 내장 Robotiq85 / 저장소 커스텀
    VacuumGripper)은 import 하여 재사용한다.
  - 이번 장면에 필요한 것은 모두 새 파일(이 파일 + experiment/assets/*.xml)로만 작성.

핵심 배치 개념:
  - UR5e 손목에는 Quick Changer 로봇 측 Master(플레이스홀더) 만 장착.
  - 3개 EE 는 로봇에 장착하지 않고, 독립 MJCF 모델로 장면에 merge 하여
    로봇 베이스 중심의 동일 반경 원호 위 슬롯에 배치.
  - 세 EE 의 coupling face(결합면) 는 모두 같은 세계 Z 높이.
  - EE 길이 차이는 받침대(support) 높이로 보정(짧은 EE→높은 housing, 긴 EE→낮은 plate).

실행:
  python experiment/ee_rack_layout_demo.py
  python experiment/ee_rack_layout_demo.py --show-sites --camera ee_rack_overview
  python experiment/ee_rack_layout_demo.py --render                 # 헤드리스 GIF/PNG
  python experiment/ee_rack_layout_demo.py --rack-radius 0.5 --slot-spacing 0.24
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# 경로 처리 (실행 위치와 무관하게 동작) : 이 파일 = <repo>/experiment/...        #
# 기존 커스텀 VacuumGripper 는 <repo>/scripts/_vacuum.py 에 있으므로 import 위해 #
# scripts 디렉터리를 sys.path 에 추가한다(파일 수정 아님).                      #
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
QC_MASTER_XML = HERE / "assets" / "quick_changer_master.xml"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import robosuite as suite  # noqa: E402
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv  # noqa: E402
from robosuite.models.arenas import TableArena  # noqa: E402
from robosuite.models.grippers import GRIPPER_MAPPING, register_gripper  # noqa: E402
from robosuite.models.grippers.gripper_model import GripperModel  # noqa: E402
from robosuite.models.tasks import ManipulationTask  # noqa: E402
from robosuite.models.world import MujocoWorldBase  # noqa: E402
from robosuite.utils.mjcf_utils import (  # noqa: E402
    array_to_string,
    new_element,
    new_site,
    string_to_array,
)


# =========================================================================== #
# 1. 상수 / 설정 (모든 길이 = m, 모든 각도 = rad)                              #
# =========================================================================== #

# 곡선 위 슬롯 순서(상단/시계방향 기준). 상수 하나로 쉽게 변경 가능.
EE_ORDER = ["3f_soft", "vacuum", "2f"]

RACK_RADIUS = 0.55           # 허용 0.45~0.60 : 베이스 중심 XY → 슬롯 coupling 중심 XY 거리
SLOT_SPACING = 0.24          # 허용 0.23~0.25 : 이웃 슬롯 coupling 중심 간 (현) 거리
ARC_CENTER_ANGLE = math.radians(60.0)   # 로봇 전방축에서 RACK_SIDE 방향으로 잰 중앙 슬롯 각도
RACK_SIDE = "right"          # "right" 또는 "left" (로봇 기준). 좌우가 반대면 뒤집는다.

# 정적 배치가 목적이므로, 독립 배치되는 EE 의 내부 관절/액추에이터/텐던을 제거해
# "완전 강체"로 만든다. => (1) 중력에 의한 손가락 진동/발산 원천 차단,
#                         (2) detached actuator 가 애초에 존재하지 않아 로봇 action space
#                             오염 불가. 향후 실제 pick-up 을 붙일 때 False 로 되돌리면
#                             원본 관절 그대로 살아난다(배치 코드와 분리).
RIGIDIZE_DETACHED = True

# 공통 결합면 높이. None 이면 (테이블 상판 + 가장 긴 EE + 최소 support) 로 자동 산출.
COMMON_COUPLING_Z = None
COUPLING_HEIGHT_TOLERANCE = 0.005

PRE_DOCK_OFFSET = 0.04       # 허용 0.03~0.05
APPROACH_OFFSET = 0.12       # 허용 0.10~0.15

MIN_SUPPORT_HEIGHT = 0.03    # 가장 긴 EE 아래 최소 plate 두께
SUPPORT_GAP = 0.002          # EE 최하단과 support 상단 사이 미세 간격(초기 관통 방지)
SUPPORT_XY_MARGIN = 0.7      # support 반경 = EE 수평반경 * 이 값 (이웃/작업영역 침범 방지)
SUPPORT_HALF_XY_MAX = 0.09

# 작업 공간(테이블). rack 반경 0.55 를 담기 위해 기본 Lift(0.8) 보다 크게.
TABLE_FULL_SIZE = (1.4, 1.4, 0.05)
TABLE_OFFSET_Z = 0.8         # 상판 표면 세계 Z

# UR5e 베이스 XY(세계). set_base_xpos((x, y, 0)) 로 넣으면 마운트 높이가 자동 적용되어
# 베이스가 테이블 위(z≈0.912)에 선다. 슬롯 원호 계산에도 이 XY 를 사용하고,
# reset 후 실제 sim 값과 일치하는지 검증한다(요구사항: 추측 금지).
BASE_XY = (-0.45, 0.0)

# 전방 중앙 작업 영역(비워 둘 영역) 중심과 반경(검증용).
WORKSPACE_FORWARD = 0.45     # 베이스에서 전방으로 이만큼 떨어진 지점을 작업영역 중심으로 본다
WORKSPACE_CLEAR_RADIUS = 0.16   # 전방중앙에 비워 둘 반경(≈0.32 m 지름의 pick&place 영역)

_EPS = 1e-6                  # 경계 수치 비교용 미세 허용오차(asin/sin 왕복 오차 흡수)

OVERVIEW_CAMERA = "ee_rack_overview"
SIDE_CAMERA = "ee_rack_dock_side"


# =========================================================================== #
# 2. EE 설정 : 모델별 예외를 여기 한 곳에 모은다 (요구사항 7/14)              #
# =========================================================================== #
#   resolver           : ("builtin", <GRIPPER_MAPPING key>) 또는 ("custom_vacuum", None)
#   coupling_local_z   : EE root 로컬 프레임에서 "결합면 바깥 법선(마스터 쪽)" 방향.
#                        robosuite 그리퍼는 root origin(z=0) 이 손목 결합면이고 +z 가
#                        핑거/툴 방향이므로, 바깥 법선 = 로컬 -z.
#   coupling_offset    : root origin → 실제 결합면까지 로컬 z 오프셋(없으면 0).
#   yaw_offset         : 커넥터 키 방향이 다르면 슬롯 yaw 에 더할 보정각.
EE_CONFIGS = {
    "3f_soft": dict(
        resolver=("builtin", "JacoThreeFingerGripper"),
        idn="rack_3f_soft",
        coupling_local_z=(0.0, 0.0, -1.0),
        coupling_offset=0.0,
        yaw_offset=0.0,
        label="3-Finger / Soft EE  (repo 3-finger = robosuite JacoThreeFingerGripper)",
    ),
    "vacuum": dict(
        resolver=("custom_vacuum", None),
        idn="rack_vacuum",
        coupling_local_z=(0.0, 0.0, -1.0),
        coupling_offset=0.0,
        yaw_offset=0.0,
        label="Vacuum EE  (repo custom VacuumGripper: scripts/assets/vacuum_gripper.xml)",
    ),
    "2f": dict(
        resolver=("builtin", "Robotiq85Gripper"),
        idn="rack_2f",
        coupling_local_z=(0.0, 0.0, -1.0),
        coupling_offset=0.0,
        yaw_offset=0.0,
        label="2-Finger EE  (robosuite Robotiq85Gripper = UR5e default 2F gripper)",
    ),
}

QC_MASTER_GRIPPER_NAME = "QuickChangerMaster"


# =========================================================================== #
# 3. 작은 수학/기하 헬퍼                                                       #
# =========================================================================== #
def _rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# 로컬 +z 를 세계 -z 로 뒤집는(결합면을 위로) 180° 회전 (x축 기준).
_ROT_FLIP_X = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def mat_to_quat_wxyz(R):
    """3x3 회전행렬 -> MuJoCo 쿼터니언 (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def look_at_quat_wxyz(cam_pos, target, world_up=(0.0, 0.0, 1.0)):
    """카메라 위치/타깃 -> MuJoCo 카메라 쿼터니언(w,x,y,z).

    MuJoCo 카메라는 -z 축을 바라보고 +y 가 위. look-at 프레임을 회전행렬로 만들어
    변환한다(고정 quat 을 감으로 쓰지 않음)."""
    cam_pos = np.asarray(cam_pos, float)
    target = np.asarray(target, float)
    fwd = target - cam_pos
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        raise ValueError("look_at: 카메라와 타깃이 같은 위치")
    fwd /= n
    up = np.asarray(world_up, float)
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:      # fwd 가 up 과 평행하면 다른 up 사용
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    # 카메라 축(세계 표현): x=right, y=true_up, z=-fwd
    R = np.column_stack([right, true_up, -fwd])
    return mat_to_quat_wxyz(R)


# =========================================================================== #
# 4. 기존 모델 해석 + 형상 측정 (요구사항 3)                                   #
# =========================================================================== #
def _resolve_ee_class(resolver):
    """EE_CONFIGS 의 resolver -> GripperModel 서브클래스. 실패 시 명확한 오류."""
    kind, key = resolver
    if kind == "builtin":
        if key not in GRIPPER_MAPPING:
            raise KeyError(
                f"[해석 실패] 내장 그리퍼 '{key}' 를 GRIPPER_MAPPING 에서 찾지 못했습니다. "
                f"사용 가능: {sorted(k for k in GRIPPER_MAPPING if k)}"
            )
        return GRIPPER_MAPPING[key]
    if kind == "custom_vacuum":
        try:
            from _vacuum import register_vacuum  # <repo>/scripts/_vacuum.py
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                f"[해석 실패] 저장소 커스텀 VacuumGripper 를 import 하지 못했습니다 "
                f"({SCRIPTS_DIR}/_vacuum.py). 원인: {e}"
            ) from e
        return register_vacuum()
    raise ValueError(f"[해석 실패] 알 수 없는 resolver 종류: {kind!r}")


def register_quick_changer_master():
    """로봇 측 Quick Changer Master(플레이스홀더)를 robosuite 에 등록하고 반환.

    ※ 저장소에 실제 Quick Changer Master 모델이 없어, 기존 파일을 건드리지 않고
       새 XML(experiment/assets/quick_changer_master.xml) 을 로봇 그리퍼로 등록한다.
       actuator/joint 가 없어 dof=0 → 로봇 action space 에 그리퍼 채널이 없다."""
    if QC_MASTER_GRIPPER_NAME in GRIPPER_MAPPING:
        return GRIPPER_MAPPING[QC_MASTER_GRIPPER_NAME]
    if not QC_MASTER_XML.exists():
        raise FileNotFoundError(f"[해석 실패] QC Master XML 없음: {QC_MASTER_XML}")
    xml_path = str(QC_MASTER_XML)

    @register_gripper
    class QuickChangerMaster(GripperModel):
        """로봇 측 커플러 마스터(기하 플레이스홀더, dof=0)."""

        def __init__(self, idn=0):
            super().__init__(xml_path, idn=idn)

        def format_action(self, action):
            return action

        @property
        def init_qpos(self):
            return None

        @property
        def dof(self):
            return 0

        @property
        def _important_geoms(self):
            return {}

    # 클래스명이 QuickChangerMaster 이므로 register_gripper 가 그 이름으로 매핑.
    return GRIPPER_MAPPING[QC_MASTER_GRIPPER_NAME]


def _rigidize_gripper(g):
    """detached EE 를 완전 강체로: 내부 <joint>/<actuator>/<tendon>/<equality> 제거.

    ※ robosuite 로 만든 *이 인스턴스*의 XML 트리만 수정한다(원본 파일은 불변).
    """
    # worldbody 내부 모든 joint 제거 (free joint 없음 → 강체로 world 고정)
    def _strip_joints(parent):
        for child in list(parent):
            if child.tag == "joint":
                parent.remove(child)
            else:
                _strip_joints(child)
    _strip_joints(g.worldbody)
    for container in (g.actuator, g.tendon, g.equality):
        for child in list(container):
            container.remove(child)


def _hide_gripper_viz_sites(g):
    """detached EE 의 그리퍼 자체 viz site(grip_site, 10m grip_site_cylinder, ee_*,
    ft_frame)를 숨긴다(alpha=0). 장면에는 우리가 만든 rack_* dock site 만 보이게."""
    def _walk(parent):
        for child in parent:
            if child.tag == "site":
                rgba = string_to_array(child.get("rgba", "0.5 0.5 0.5 1"))
                rgba[3] = 0.0
                child.set("rgba", array_to_string(rgba))
            _walk(child)
    _walk(g.worldbody)


def _make_ee_model(name, place_pos=(0, 0, 0), place_quat_wxyz=(1, 0, 0, 0),
                   rigidize=RIGIDIZE_DETACHED):
    """EE 이름 -> 배치된 GripperModel 인스턴스(로컬 root pos/quat 설정)."""
    cfg = EE_CONFIGS[name]
    cls = _resolve_ee_class(cfg["resolver"])
    g = cls(idn=cfg["idn"])
    if rigidize:
        _rigidize_gripper(g)
    _hide_gripper_viz_sites(g)
    root = g.worldbody[0]
    root.set("pos", array_to_string(np.asarray(place_pos, float)))
    root.set("quat", array_to_string(np.asarray(place_quat_wxyz, float)))
    return g


def measure_ee_extent(name):
    """EE 를 단독 world 에 merge(→에셋 중복 제거)해 compile 후 형상 측정.

    각 geom 의 실제 AABB(geom_aabb, 로컬 타이트 박스)를 root 프레임으로 투영해
    측정한다. 바운딩 스피어(rbound)는 납작한 실린더(진공 컵)를 과대평가해 support
    를 낮게 잡아 EE 가 떠 보이게 하므로 사용하지 않는다.

    반환: dict(L, below, horiz_r, root_body, dof, actuators, joints, sites)
      L      = root origin(결합면) 에서 +z(툴 끝) 방향 최대 길이 (결합면→최하단, 타이트)
      below  = root origin 뒤쪽(-z)으로 튀어나온 길이
      horiz_r= root 축 기준 최대 수평 반경
    """
    import mujoco

    g = _make_ee_model(name)  # 기본 pose(원점) 에서 측정
    world = MujocoWorldBase()
    world.merge(g)
    m = mujoco.MjModel.from_xml_string(world.get_xml())
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rbid = m.body(g.root_body).id
    root_xy = np.array(d.xpos[rbid][:2])
    aabb = np.asarray(m.geom_aabb).reshape(-1, 6)   # [cx,cy,cz, hx,hy,hz] (geom 로컬)
    zmax = -1e9
    zmin = 1e9
    hr = 0.0
    for gi in range(m.ngeom):
        if int(m.geom_bodyid[gi]) == 0:             # world(바닥 등) geom 제외
            continue
        R = np.array(d.geom_xmat[gi]).reshape(3, 3)
        p = np.array(d.geom_xpos[gi])
        center_local = aabb[gi, :3]
        half = np.abs(aabb[gi, 3:])
        wc = p + R @ center_local                   # AABB 중심(세계)
        habs = np.abs(R) @ half                     # 회전된 AABB 반크기(세계 축별)
        zmax = max(zmax, wc[2] + habs[2])
        zmin = min(zmin, wc[2] - habs[2])
        hr = max(hr, float(np.linalg.norm(wc[:2] - root_xy)) + float(math.hypot(habs[0], habs[1])))
    return dict(
        L=float(zmax), below=float(-zmin), horiz_r=float(hr),
        root_body=g.root_body, dof=g.dof,
        actuators=list(g.actuators), joints=list(g.joints), sites=list(g.sites),
    )


def inspect_or_resolve_existing_models(verbose=True):
    """모든 사용 모델(로봇/QC Master/3 EE)을 해석·측정하고 요약 반환.

    찾지 못한 모델이 있으면 임의 이름을 만들지 않고 예외로 중단한다."""
    register_quick_changer_master()      # 등록(존재 검증 포함)
    info = {"ee": {}}
    for name in EE_ORDER:
        if name not in EE_CONFIGS:
            raise KeyError(f"[설정 오류] EE_ORDER 의 '{name}' 이 EE_CONFIGS 에 없음")
        info["ee"][name] = measure_ee_extent(name)
    if verbose:
        print("\n[resolve] 사용 모델 해석 결과")
        print(f"  robot          : UR5e (robosuite 내장)")
        print(f"  wrist master   : {QC_MASTER_GRIPPER_NAME} (신규 플레이스홀더 XML)")
        for name in EE_ORDER:
            e = info["ee"][name]
            print(f"  EE[{name:8s}] {EE_CONFIGS[name]['label']}")
            print(f"      root_body={e['root_body']}  dof={e['dof']}  "
                  f"L(결합면→끝)={e['L']:.3f}  horiz_r={e['horiz_r']:.3f}")
    return info


# =========================================================================== #
# 5. 원호 슬롯/도킹 pose 계산 (요구사항 6/10)                                   #
# =========================================================================== #
def compute_arc_slot_poses(base_xy, forward_xy, right_xy, radius, spacing,
                           center_angle, ee_order):
    """원호 위 각 슬롯의 XY, yaw(로봇을 향함), 각도를 계산.

    delta_angle = 2*asin(spacing / (2*radius)) (현 길이 → 중심각)."""
    if radius <= 0:
        raise ValueError("RACK_RADIUS 는 양수여야 합니다")
    ratio = spacing / (2.0 * radius)
    if not (-1.0 <= ratio <= 1.0):
        raise ValueError(f"SLOT_SPACING({spacing}) 이 반경({radius})에 비해 너무 큽니다")
    delta = 2.0 * math.asin(ratio)
    n = len(ee_order)
    # 중앙 정렬된 각도들 (n 개)
    offsets = [(i - (n - 1) / 2.0) for i in range(n)]
    base_xy = np.asarray(base_xy, float)
    forward_xy = np.asarray(forward_xy, float)
    right_xy = np.asarray(right_xy, float)
    slots = []
    for name, k in zip(ee_order, offsets):
        angle = center_angle + k * delta
        direction = math.cos(angle) * forward_xy + math.sin(angle) * right_xy
        xy = base_xy + radius * direction
        yaw = math.atan2(base_xy[1] - xy[1], base_xy[0] - xy[0]) + EE_CONFIGS[name]["yaw_offset"]
        slots.append(dict(name=name, angle=angle, xy=xy, yaw=yaw))
    return dict(delta_angle=delta, slots=slots)


def compute_ee_placement(name, slot_xy, slot_yaw, coupling_z, ee_meas):
    """결합면 pose 를 먼저 정하고 EE root pose 를 역산 (요구사항 7).

    반환: dict(root_pos, root_quat_wxyz, coupling_pos, docking_axis, R_root,
              support_top_z, support_height)
    """
    cfg = EE_CONFIGS[name]
    # EE root 세계 회전 = yaw(z) ∘ flip(x180)  → 로컬 +z 가 세계 -z(툴 끝이 아래),
    #                                            로컬 -z(결합면 바깥법선) 가 세계 +z.
    R_root = _rot_z(slot_yaw) @ _ROT_FLIP_X
    # 결합면 바깥 법선(도킹 축) 을 회전행렬에서 유도(세계 +Z 를 감으로 쓰지 않음).
    local_normal = np.asarray(cfg["coupling_local_z"], float)
    docking_axis = R_root @ local_normal
    docking_axis /= np.linalg.norm(docking_axis)
    # 결합면 위치(=dock). root origin 에서 로컬 +z 로 coupling_offset 만큼 떨어진 곳.
    # 여기서는 coupling_offset=0 이라 root origin = 결합면.
    coupling_pos = np.array([slot_xy[0], slot_xy[1], coupling_z], float)
    root_pos = coupling_pos - R_root @ np.array([0, 0, cfg["coupling_offset"]], float)
    # support: EE 최하단(root +z 로 L) 세계 z.
    tip_world_z = root_pos[2] + (R_root @ np.array([0, 0, ee_meas["L"]]))[2]
    support_top_z = tip_world_z - SUPPORT_GAP
    return dict(
        root_pos=root_pos,
        root_quat_wxyz=mat_to_quat_wxyz(R_root),
        coupling_pos=coupling_pos,
        docking_axis=docking_axis,
        R_root=R_root,
        tip_world_z=tip_world_z,
        support_top_z=support_top_z,
    )


def compute_docking_poses(coupling_pos, docking_axis):
    """dock / pre-dock / approach 위치 (docking_axis 바깥 방향으로 offset)."""
    coupling_pos = np.asarray(coupling_pos, float)
    axis = np.asarray(docking_axis, float)
    return dict(
        dock=coupling_pos.copy(),
        pre_dock=coupling_pos + axis * PRE_DOCK_OFFSET,
        approach=coupling_pos + axis * APPROACH_OFFSET,
    )


# =========================================================================== #
# 6. 전체 레이아웃 플랜 (env 와 검증이 공유)                                    #
# =========================================================================== #
def build_layout_plan(models_info,
                      rack_radius=RACK_RADIUS, slot_spacing=SLOT_SPACING,
                      arc_center_angle=ARC_CENTER_ANGLE, rack_side=RACK_SIDE,
                      common_coupling_z=COMMON_COUPLING_Z):
    """모든 배치 수치를 계산해 dict 로 반환. sim 없이 순수 계산."""
    table_top = TABLE_OFFSET_Z + TABLE_FULL_SIZE[2] / 2.0
    base_xy = np.array(BASE_XY, float)
    # 로봇 전방/우측 단위벡터 (베이스 orientation = identity → forward=+x).
    forward_xy = np.array([1.0, 0.0])
    right_sign = -1.0 if rack_side == "right" else 1.0   # forward=+x, up=+z → 로봇 우측 = -y
    right_xy = np.array([0.0, right_sign])

    # 공통 결합면 높이 자동 산출: 가장 긴 EE 아래에도 최소 support 두께가 남도록.
    max_L = max(models_info["ee"][n]["L"] for n in EE_ORDER)
    if common_coupling_z is None:
        common_coupling_z = table_top + max_L + MIN_SUPPORT_HEIGHT

    arc = compute_arc_slot_poses(base_xy, forward_xy, right_xy, rack_radius,
                                 slot_spacing, arc_center_angle, EE_ORDER)

    entries = {}
    for slot in arc["slots"]:
        name = slot["name"]
        meas = models_info["ee"][name]
        place = compute_ee_placement(name, slot["xy"], slot["yaw"], common_coupling_z, meas)
        docks = compute_docking_poses(place["coupling_pos"], place["docking_axis"])
        support_height = place["support_top_z"] - table_top
        half_xy = float(min(max(meas["horiz_r"] * SUPPORT_XY_MARGIN, 0.04), SUPPORT_HALF_XY_MAX))
        support_type = "housing (tall)" if support_height >= 0.06 else "plate (low)"
        entries[name] = dict(
            name=name, angle=slot["angle"], slot_xy=slot["xy"], yaw=slot["yaw"],
            meas=meas, **place, **docks,
            support_height=float(support_height), support_half_xy=half_xy,
            support_type=support_type,
        )

    return dict(
        table_top=table_top, base_xy=base_xy, forward_xy=forward_xy, right_xy=right_xy,
        common_coupling_z=common_coupling_z, delta_angle=arc["delta_angle"],
        rack_radius=rack_radius, slot_spacing=slot_spacing,
        arc_center_angle=arc_center_angle, rack_side=rack_side,
        workspace_center=base_xy + forward_xy * WORKSPACE_FORWARD,
        entries=entries,
    )


# =========================================================================== #
# 7. 받침대 / Rack 베이스 / 디버그 site MJCF 빌더 (요구사항 8/10)              #
# =========================================================================== #
def build_support_for_ee(name, entry, table_top):
    """EE 받침대 body(ET.Element). 시각 geom + 충돌 geom 포함, 월드 고정."""
    cx, cy = float(entry["slot_xy"][0]), float(entry["slot_xy"][1])
    h = entry["support_height"]
    half_z = max(h / 2.0, 0.005)
    cz = table_top + half_z
    half_xy = entry["support_half_xy"]
    body = new_element(tag="body", name=f"rack_{name}_support", pos=array_to_string([cx, cy, cz]))
    size = array_to_string([half_xy, half_xy, half_z])
    body.append(new_element(tag="geom", name=f"rack_{name}_support_col", type="box",
                            size=size, group="0", contype="1", conaffinity="1",
                            rgba="0.25 0.27 0.30 1"))
    body.append(new_element(tag="geom", name=f"rack_{name}_support_vis", type="box",
                            size=size, group="1", contype="0", conaffinity="0",
                            rgba="0.32 0.36 0.42 1"))
    # 상단 테두리(작은 링 느낌의 얇은 판) - 시각용, EE 안착면 표시.
    body.append(new_element(tag="geom", name=f"rack_{name}_support_top_vis", type="box",
                            size=array_to_string([half_xy, half_xy, 0.004]),
                            pos=array_to_string([0, 0, half_z]),
                            group="1", contype="0", conaffinity="0",
                            rgba="0.55 0.6 0.66 1"))
    return body


def build_rack_base(plan):
    """슬롯을 잇는 곡선형 Rack 베이스 레일(여러 box segment) body 반환."""
    table_top = plan["table_top"]
    base_xy = plan["base_xy"]
    fwd, right = plan["forward_xy"], plan["right_xy"]
    R = plan["rack_radius"]
    angles = [plan["entries"][n]["angle"] for n in EE_ORDER]
    a0, a1 = min(angles), max(angles)
    pad = plan["delta_angle"] * 0.6
    a0 -= pad
    a1 += pad
    rail_h = 0.018
    cz = table_top + rail_h
    body = new_element(tag="body", name="rack_base", pos="0 0 0")
    nseg = 12
    for i in range(nseg):
        a = a0 + (a1 - a0) * (i + 0.5) / nseg
        direction = math.cos(a) * fwd + math.sin(a) * right
        p = base_xy + R * direction
        # 세그먼트 길이 = 호 길이 / nseg (+겹침 여유)
        seg_len = (a1 - a0) * R / nseg * 0.62
        tangent_yaw = a + math.pi / 2.0 if plan["rack_side"] == "left" else a - math.pi / 2.0
        quat = mat_to_quat_wxyz(_rot_z(tangent_yaw))
        body.append(new_element(
            tag="geom", name=f"rack_base_seg{i}_col", type="box",
            size=array_to_string([seg_len, 0.035, rail_h]),
            pos=array_to_string([p[0], p[1], cz]), quat=array_to_string(quat),
            group="0", contype="1", conaffinity="1", rgba="0.18 0.2 0.23 1"))
        body.append(new_element(
            tag="geom", name=f"rack_base_seg{i}_vis", type="box",
            size=array_to_string([seg_len, 0.035, rail_h]),
            pos=array_to_string([p[0], p[1], cz]), quat=array_to_string(quat),
            group="1", contype="0", conaffinity="0", rgba="0.22 0.25 0.29 1"))
    return body


def add_debug_sites(worldbody, plan, show_sites):
    """dock/pre-dock/approach site 추가. show_sites 일 때만 보이도록 alpha 설정."""
    a = 1.0 if show_sites else 0.0
    colors = dict(dock=(0.1, 0.9, 0.2, a), pre_dock=(0.95, 0.85, 0.1, a),
                  approach=(0.1, 0.7, 0.95, a))
    sizes = dict(dock=0.012, pre_dock=0.010, approach=0.010)
    for name in EE_ORDER:
        e = plan["entries"][name]
        for key in ("dock", "pre_dock", "approach"):
            worldbody.append(new_site(
                name=f"rack_{name}_{key}", pos=np.asarray(e[key], float),
                size=[sizes[key]], rgba=list(colors[key]), type="sphere", group="1"))
        # 도킹 축(결합면 바깥 법선) 시각화용 얇은 실린더 (dock→approach).
        mid = (np.asarray(e["dock"], float) + np.asarray(e["approach"], float)) / 2.0
        axis = np.asarray(e["docking_axis"], float)
        # +z 를 axis 로 돌리는 최소 회전
        zc = np.array([0, 0, 1.0])
        v = np.cross(zc, axis)
        c = float(np.dot(zc, axis))
        if np.linalg.norm(v) < 1e-8:
            Rax = np.eye(3) if c > 0 else _ROT_FLIP_X
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            Rax = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
        worldbody.append(new_site(
            name=f"rack_{name}_dockaxis", pos=mid, quat=mat_to_quat_wxyz(Rax),
            size=[0.003, APPROACH_OFFSET / 2.0], rgba=[0.6, 0.6, 0.6, a * 0.6],
            type="cylinder", group="1"))


# =========================================================================== #
# 8. 환경 클래스 (요구사항 14)                                                 #
# =========================================================================== #
class EERackLayoutEnv(ManipulationEnv):
    """UR5e + 손목 QC Master + 곡선 Rack 위 독립 EE 3종 배치 장면.

    실제 조작/보상 없음(placement 확인용). reward=0, success=False.
    """

    def __init__(self, models_info=None, show_sites=False,
                 rack_radius=RACK_RADIUS, slot_spacing=SLOT_SPACING,
                 arc_center_angle=ARC_CENTER_ANGLE, rack_side=RACK_SIDE,
                 common_coupling_z=COMMON_COUPLING_Z,
                 has_renderer=False, has_offscreen_renderer=False,
                 render_camera=OVERVIEW_CAMERA, control_freq=20, **kwargs):
        register_quick_changer_master()
        self._models_info = models_info or inspect_or_resolve_existing_models(verbose=False)
        self._show_sites = show_sites
        # 플랜은 _load_model 전에 필요 → 여기서 계산.
        self.plan = build_layout_plan(
            self._models_info, rack_radius=rack_radius, slot_spacing=slot_spacing,
            arc_center_angle=arc_center_angle, rack_side=rack_side,
            common_coupling_z=common_coupling_z)
        self.ee_body_ids = {}
        super().__init__(
            robots="UR5e",
            gripper_types=QC_MASTER_GRIPPER_NAME,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            use_camera_obs=False,
            control_freq=control_freq,
            horizon=10_000,
            ignore_done=True,
            **kwargs,
        )

    # --- 필수 오버라이드 --------------------------------------------------- #
    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False

    def _load_model(self):
        super()._load_model()           # RobotEnv: 로봇 로드
        plan = self.plan

        # 로봇 베이스를 원하는 XY 에 배치(z=0 → 마운트가 테이블 위로 올려줌).
        self.robots[0].robot_model.set_base_xpos(np.array([BASE_XY[0], BASE_XY[1], 0.0]))

        # 큰 테이블 arena.
        arena = TableArena(table_full_size=TABLE_FULL_SIZE,
                           table_offset=(0.0, 0.0, TABLE_OFFSET_Z),
                           has_legs=True)
        arena.set_origin([0.0, 0.0, 0.0])
        self._add_cameras(arena, plan)

        # objects 없이 task 구성(작업영역 비움).
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[self.robots[0].robot_model],
            mujoco_objects=[],
        )

        # --- 독립 EE 3종을 static body 로 merge (로봇 손목에 붙이지 않음) ---- #
        for name in EE_ORDER:
            e = plan["entries"][name]
            g = _make_ee_model(name, place_pos=e["root_pos"],
                               place_quat_wxyz=e["root_quat_wxyz"])
            self.model.merge(g)

        # --- 받침대 + Rack 베이스 레일 + 디버그 site --------------------- #
        wb = self.model.worldbody
        wb.append(build_rack_base(plan))
        for name in EE_ORDER:
            wb.append(build_support_for_ee(name, plan["entries"][name], plan["table_top"]))
        add_debug_sites(wb, plan, self._show_sites)

    def _add_cameras(self, arena, plan):
        """overview(오블리크 탑뷰) + dock_side(측면) 카메라 추가."""
        entries = plan["entries"]
        base_xy = plan["base_xy"]
        # 장면 핵심점들의 AABB 중심을 look-at 타깃으로.
        pts = [np.array([base_xy[0], base_xy[1], 0.95])]
        for n in EE_ORDER:
            pts.append(np.asarray(entries[n]["coupling_pos"], float))
        pts.append(np.array([plan["workspace_center"][0], plan["workspace_center"][1], plan["table_top"]]))
        pts = np.array(pts)
        center = (pts.min(0) + pts.max(0)) / 2.0

        side_sign = -1.0 if plan["rack_side"] == "right" else 1.0
        # 오블리크 탑뷰: 전방(+x) & rack 반대편에서 위로 내려다봄.
        overview_pos = center + np.array([1.35, -side_sign * 1.35, 1.5])
        arena.set_camera(OVERVIEW_CAMERA, pos=overview_pos,
                         quat=look_at_quat_wxyz(overview_pos, center),
                         camera_attribs={"fovy": "55"})
        # 측면 도킹 확인: rack 중앙 슬롯을 옆에서 수평에 가깝게.
        mid = np.asarray(entries[EE_ORDER[len(EE_ORDER) // 2]]["coupling_pos"], float)
        side_pos = mid + np.array([0.85, side_sign * 0.15, 0.28])
        arena.set_camera(SIDE_CAMERA, pos=side_pos,
                         quat=look_at_quat_wxyz(side_pos, mid),
                         camera_attribs={"fovy": "50"})

        # 인터랙티브(free) 뷰어가 동일한 시점에서 시작하도록 카메라 셋업 저장.
        # (fixed 카메라는 마우스로 못 움직이므로 --view 는 free 카메라를 쓴다.)
        self._scene_center = center
        self._cam_setups = {OVERVIEW_CAMERA: (overview_pos, center),
                            SIDE_CAMERA: (side_pos, mid)}

    def _setup_references(self):
        super()._setup_references()
        self.ee_body_ids = {}
        for name in EE_ORDER:
            rb = self._models_info["ee"][name]["root_body"]
            self.ee_body_ids[name] = self.sim.model.body_name2id(rb)


# =========================================================================== #
# 9. 자동 검증 (요구사항 16)                                                   #
# =========================================================================== #
def _body_world_aabb(sim, root_body_id):
    """root_body 서브트리 geom 들의 세계 AABB (min, max)."""
    m = sim.model._model
    d = sim.data._data
    # 서브트리 body 집합
    subtree = set()
    stack = [root_body_id]
    while stack:
        b = stack.pop()
        subtree.add(b)
        for c in range(m.nbody):
            if m.body_parentid[c] == b and c != b:
                stack.append(c)
    lo = np.array([np.inf] * 3)
    hi = np.array([-np.inf] * 3)
    for gi in range(m.ngeom):
        if m.geom_bodyid[gi] in subtree:
            c = np.array(d.geom_xpos[gi])
            r = float(m.geom_rbound[gi])
            lo = np.minimum(lo, c - r)
            hi = np.maximum(hi, c + r)
    return lo, hi


def _aabb_overlap(a, b, margin=0.0):
    (alo, ahi), (blo, bhi) = a, b
    return all(ahi[i] + margin > blo[i] and bhi[i] + margin > alo[i] for i in range(3))


def validate_layout(env, settle_seconds=2.0, verbose=True):
    """reset 이후 배치 검증. 표 출력 + assert. 통과하면 True."""
    import mujoco

    sim = env.sim
    m = sim.model
    plan = env.plan
    entries = plan["entries"]
    results = {"ok": True, "warnings": [], "errors": []}

    def check(cond, msg):
        if not cond:
            results["ok"] = False
            results["errors"].append(msg)
        return cond

    # --- 표 출력 ---------------------------------------------------------- #
    base_xy = plan["base_xy"]
    if verbose:
        print("\n" + "=" * 108)
        print("EE RACK LAYOUT — 검증 표 (단위 m)")
        print("=" * 108)
        hdr = (f"{'EE':9s}{'slot_xyz':>26s}{'d_base':>8s}{'d_prev':>8s}"
               f"{'coupZ':>8s}{'supH':>7s} {'support':>15s}")
        print(hdr)
        print("-" * 108)
    prev_xy = None
    coupling_zs = []
    dists = []
    for name in EE_ORDER:
        e = entries[name]
        sx, sy = float(e["slot_xy"][0]), float(e["slot_xy"][1])
        cz = float(e["coupling_pos"][2])
        coupling_zs.append(cz)
        d_base = math.hypot(sx - base_xy[0], sy - base_xy[1])
        dists.append(d_base)
        d_prev = (math.hypot(sx - prev_xy[0], sy - prev_xy[1]) if prev_xy is not None else 0.0)
        prev_xy = (sx, sy)
        if verbose:
            print(f"{name:9s}({sx:7.3f},{sy:7.3f},{cz:6.3f})"
                  f"{d_base:8.3f}{d_prev:8.3f}{cz:8.3f}{e['support_height']:7.3f} {e['support_type']:>15s}")
    if verbose:
        print("-" * 108)
        for name in EE_ORDER:
            e = entries[name]
            print(f"  [{name}] dock={np.round(e['dock'],3)}  "
                  f"pre_dock={np.round(e['pre_dock'],3)}  approach={np.round(e['approach'],3)}")

    # --- 수치 assert ------------------------------------------------------ #
    for name in EE_ORDER:
        r = math.hypot(entries[name]["slot_xy"][0] - base_xy[0],
                       entries[name]["slot_xy"][1] - base_xy[1])
        check(0.45 - _EPS <= r <= 0.60 + _EPS, f"[반경] {name} radius={r:.3f} 가 0.45~0.60 밖")
    for i in range(1, len(EE_ORDER)):
        a = entries[EE_ORDER[i - 1]]["slot_xy"]
        b = entries[EE_ORDER[i]]["slot_xy"]
        dd = math.hypot(a[0] - b[0], a[1] - b[1])
        check(0.23 - _EPS <= dd <= 0.25 + _EPS,
              f"[슬롯간격] {EE_ORDER[i-1]}~{EE_ORDER[i]} dist={dd:.3f} 가 0.23~0.25 밖")
    for i in range(len(coupling_zs)):
        for j in range(i + 1, len(coupling_zs)):
            check(abs(coupling_zs[i] - coupling_zs[j]) <= COUPLING_HEIGHT_TOLERANCE,
                  f"[결합면높이] {EE_ORDER[i]}~{EE_ORDER[j]} Δz={abs(coupling_zs[i]-coupling_zs[j]):.4f}")
    check(0.03 <= PRE_DOCK_OFFSET <= 0.05, f"[pre-dock] {PRE_DOCK_OFFSET} 범위 밖")
    check(0.10 <= APPROACH_OFFSET <= 0.15, f"[approach] {APPROACH_OFFSET} 범위 밖")
    for name in EE_ORDER:
        check(entries[name]["support_height"] >= -1e-9,
              f"[support] {name} 받침 높이 음수 {entries[name]['support_height']:.3f}")

    # --- 구조 검증 : EE 가 로봇 손목 자식이 아님 -------------------------- #
    robot_body_ids = set()
    for bi in range(m.nbody):
        nm = m.body_id2name(bi)
        if nm and (nm.startswith("robot0") or nm.startswith("gripper0")):
            robot_body_ids.add(bi)
    for name in EE_ORDER:
        rbid = env.ee_body_ids[name]
        parent = int(m._model.body_parentid[rbid])
        check(parent == 0, f"[부착] {name} root parent={m.body_id2name(parent)} (world=0 이어야)")
        check(rbid not in robot_body_ids, f"[부착] {name} 가 로봇 body 로 분류됨")

    # --- QC Master 가 손목에 장착됐고 EE 는 손목에 없음 ------------------ #
    robot_root = env.robots[0].robot_model.root_body
    robot_root_id = m.body_name2id(robot_root)

    def _in_robot_subtree(bid):
        cur = bid
        for _ in range(m._model.nbody + 1):
            if cur == robot_root_id:
                return True
            if cur == 0:
                return False
            cur = int(m._model.body_parentid[cur])
        return False

    qc_ids = [bi for bi in range(m._model.nbody)
              if (m.body_id2name(bi) or "").find("qc_master") >= 0]
    check(len(qc_ids) > 0, "[QC마스터] qc_master body 를 찾지 못함(손목 장착 안됨)")
    check(all(_in_robot_subtree(bi) for bi in qc_ids),
          "[QC마스터] QC Master 가 로봇 손목 하위에 있지 않음")
    for name in EE_ORDER:
        check(not _in_robot_subtree(env.ee_body_ids[name]),
              f"[부착] {name} EE 가 로봇 손목 하위에 있음(장착되면 안 됨)")
    if verbose and qc_ids:
        print(f"  [wrist] QC Master body='{m.body_id2name(qc_ids[0])}' → 로봇 손목 하위 OK, "
              f"EE 3종은 손목 비하위 OK")

    # --- 결합면 방향 동일성 --------------------------------------------- #
    d = sim.data
    normals = []
    for name in EE_ORDER:
        rbid = env.ee_body_ids[name]
        R = np.array(d._data.xmat[rbid]).reshape(3, 3)
        local_n = np.asarray(EE_CONFIGS[name]["coupling_local_z"], float)
        normals.append(R @ local_n)
    for i in range(1, len(normals)):
        cosang = float(np.clip(np.dot(normals[0], normals[i]), -1, 1))
        check(cosang > 0.999, f"[결합면방향] {EE_ORDER[i]} 법선이 {EE_ORDER[0]} 와 다름 (cos={cosang:.4f})")
    if verbose:
        print(f"\n  결합면 바깥 법선(세계) 예시: {np.round(normals[0],4)}  (수직 도킹 → +Z 근처)")

    # --- 베이스 pose 가정 검증(추측 금지) ------------------------------- #
    bid = m.body_name2id(env.robots[0].robot_model.root_body)
    base_pos = np.array(d._data.xpos[bid])
    base_mat = np.array(d._data.xmat[bid]).reshape(3, 3)
    check(np.allclose(base_pos[:2], BASE_XY, atol=1e-3),
          f"[베이스] 실제 XY {np.round(base_pos[:2],3)} ≠ 가정 {BASE_XY}")
    check(base_mat[0, 0] > 0.999,
          f"[베이스] forward 가 +x 가 아님 (xmat[:,0]={np.round(base_mat[:,0],3)})")

    # --- 이웃 EE AABB 비겹침 -------------------------------------------- #
    aabbs = {n: _body_world_aabb(sim, env.ee_body_ids[n]) for n in EE_ORDER}
    for i in range(1, len(EE_ORDER)):
        a, b = aabbs[EE_ORDER[i - 1]], aabbs[EE_ORDER[i]]
        check(not _aabb_overlap(a, b), f"[AABB] {EE_ORDER[i-1]} 와 {EE_ORDER[i]} EE 겹침")

    # --- 작업영역 침범 검사 --------------------------------------------- #
    wc = plan["workspace_center"]
    for name in EE_ORDER:
        lo, hi = aabbs[name]
        # 작업영역 원기둥(중심 wc, 반경 R)과 EE AABB(xy) 최소거리
        cx = min(max(wc[0], lo[0]), hi[0])
        cy = min(max(wc[1], lo[1]), hi[1])
        dxy = math.hypot(cx - wc[0], cy - wc[1])
        check(dxy >= WORKSPACE_CLEAR_RADIUS - 1e-6,
              f"[작업영역] {name} 가 전방중앙 작업영역 침범 (거리 {dxy:.3f} < {WORKSPACE_CLEAR_RADIUS})")

    # --- 이름 고유성 (compile 성공 = 고유. 추가 확인) -------------------- #
    for kind, n in (("body", m._model.nbody), ("geom", m._model.ngeom),
                    ("site", m._model.nsite), ("joint", m._model.njnt),
                    ("actuator", m._model.nu)):
        names = []
        for i in range(n):
            if kind == "body":
                names.append(m.body_id2name(i))
            elif kind == "geom":
                names.append(m.geom_id2name(i))
            elif kind == "site":
                names.append(m.site_id2name(i))
            elif kind == "joint":
                names.append(mujoco.mj_id2name(m._model, mujoco.mjtObj.mjOBJ_JOINT, i))
            else:
                names.append(mujoco.mj_id2name(m._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
        named = [x for x in names if x]
        check(len(named) == len(set(named)), f"[이름중복] {kind} 에 중복 존재")

    # --- 카메라 존재 ----------------------------------------------------- #
    for cam in (OVERVIEW_CAMERA, SIDE_CAMERA):
        cid = mujoco.mj_name2id(m._model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        check(cid >= 0, f"[카메라] '{cam}' 가 model 에 없음")

    # --- 초기 contact 분류 ---------------------------------------------- #
    _classify_contacts(env, verbose)

    # --- 안정성: settle_seconds 진행 후 발산/이탈 없는지 ----------------- #
    n_act = env.action_dim
    pre = {n: np.array(d._data.xpos[env.ee_body_ids[n]]) for n in EE_ORDER}
    steps = int(settle_seconds * env.control_freq)
    max_qvel = 0.0
    for _ in range(steps):
        env.step(np.zeros(n_act))
        max_qvel = max(max_qvel, float(np.max(np.abs(sim.data._data.qvel))))
    for name in EE_ORDER:
        post = np.array(sim.data._data.xpos[env.ee_body_ids[name]])
        drift = float(np.linalg.norm(post - pre[name]))
        check(drift < 1e-4, f"[안정성] {name} EE root 이동 {drift:.5f} m (>0.1mm)")
    check(np.all(np.isfinite(sim.data._data.qpos)), "[안정성] qpos 발산(NaN/Inf)")
    if verbose:
        print(f"\n  [안정성] {settle_seconds:.0f}s({steps} step) 진행: EE root 이동 ~0, "
              f"max|qvel|={max_qvel:.3f}  action_dim={n_act}")

    if verbose:
        print("\n" + ("[검증 통과] 모든 조건 OK ✅" if results["ok"]
              else "[검증 실패] 아래 문제 확인 ❌"))
        for e in results["errors"]:
            print("   -", e)
        print("=" * 108)
    return results


def _classify_contacts(env, verbose=True):
    """초기 contact 를 support/static(정상) vs 로봇/기타(주의) 로 분류 출력."""
    sim = env.sim
    m = sim.model._model
    d = sim.data._data
    import mujoco
    mujoco.mj_forward(m, d)
    static_kw = ("rack_", "table", "support")
    normal, other = [], []
    for i in range(d.ncon):
        con = d.contact[i]
        g1, g2 = con.geom1, con.geom2
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom{g1}"
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom{g2}"
        pen = float(con.dist)
        is_static = (any(k in n1 for k in static_kw) and any(k in n2 for k in static_kw))
        (normal if is_static else other).append((n1, n2, pen))
    if verbose:
        print(f"\n  [contact] 초기 접촉 {d.ncon} 개  (정상 지지/static {len(normal)}, 기타 {len(other)})")
        for n1, n2, pen in other[:12]:
            print(f"      · {n1} ↔ {n2}  dist={pen:+.4f}")
    return normal, other


# =========================================================================== #
# 10. 실행 (요구사항 15)                                                       #
# =========================================================================== #
def _run_viewer(env, camera, seconds=None):
    """mujoco 인터랙티브 창(launch_passive). 지정 카메라로 시작."""
    import time
    import mujoco
    import mujoco.viewer as mjv

    m = env.sim.model._model
    d = env.sim.data._data
    try:
        viewer = mjv.launch_passive(m, d, show_left_ui=False, show_right_ui=False)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 인터랙티브 창 생성 실패(디스플레이 없음?): {e}")
        print("       GUI 데스크톱에서 실행하세요. 헤드리스면 --render 로 GIF 저장.")
        return
    # free 카메라로 시작(마우스 회전/줌/이동 가능). 시작 시점만 선택 카메라와 일치.
    setups = getattr(env, "_cam_setups", {})
    center = getattr(env, "_scene_center", np.array([0.0, 0.0, 1.0]))
    if camera in setups:
        cam_pos, target = setups[camera]
        offset = np.asarray(cam_pos, float) - np.asarray(target, float)
        dist = float(np.linalg.norm(offset))
        fwd = -offset / dist                       # 카메라가 바라보는 방향
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.asarray(target, float)
        viewer.cam.distance = dist
        viewer.cam.azimuth = math.degrees(math.atan2(fwd[1], fwd[0]))
        viewer.cam.elevation = math.degrees(math.asin(float(np.clip(fwd[2], -1, 1))))
    else:                                          # frontview/agentview 등: 일반 오블리크
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = center
        viewer.cam.distance = 2.4
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0
    viewer.opt.geomgroup[0] = 0     # 충돌(group0) 숨기고 시각 메시(group1) 만
    n = env.action_dim
    print(f"[view] free 카메라(마우스 조작 가능), 시작 시점='{camera}'. 창을 닫으면 종료. action_dim={n}")
    t0 = time.time()
    try:
        while viewer.is_running():
            env.step(np.zeros(n))
            viewer.sync()
            time.sleep(env.control_timestep)
            if seconds is not None and time.time() - t0 > seconds:
                break
    finally:
        viewer.close()


def _render_image(env, camera, out_stem, width=1280, height=800):
    """오프스크린 렌더 → PNG(헤드리스 OK)."""
    import imageio.v2 as imageio

    frame = env.sim.render(width=width, height=height, camera_name=camera)[::-1]
    out_dir = REPO_ROOT / "renders"
    out_dir.mkdir(exist_ok=True)
    png = out_dir / f"{out_stem}.png"
    imageio.imwrite(str(png), frame)
    print(f"[OK] png = {png}")
    return png


def parse_args():
    p = argparse.ArgumentParser(description="EE Rack Layout Demo (robosuite 1.5)")
    p.add_argument("--camera", default=OVERVIEW_CAMERA,
                   choices=[OVERVIEW_CAMERA, SIDE_CAMERA, "frontview", "agentview"],
                   help="렌더/뷰어 기본 카메라")
    p.add_argument("--show-sites", action="store_true", help="dock/pre-dock/approach site 표시")
    p.add_argument("--rack-radius", type=float, default=RACK_RADIUS)
    p.add_argument("--slot-spacing", type=float, default=SLOT_SPACING)
    p.add_argument("--arc-center-deg", type=float, default=math.degrees(ARC_CENTER_ANGLE))
    p.add_argument("--rack-side", default=RACK_SIDE, choices=["right", "left"])
    p.add_argument("--coupling-z", type=float, default=None, help="공통 결합면 Z(미지정=자동)")
    p.add_argument("--view", action="store_true", help="인터랙티브 창(기본은 검증만)")
    p.add_argument("--render", action="store_true", help="헤드리스 오프스크린 PNG 저장")
    p.add_argument("--seconds", type=float, default=None, help="--view 자동 종료 시간(초)")
    p.add_argument("--no-validate", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    try:                                  # Windows 콘솔에서 한글 깨짐 방지(선택)
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print("=" * 70)
    print(" EE RACK LAYOUT DEMO  (robosuite %s)" % suite.__version__)
    print("=" * 70)

    models_info = inspect_or_resolve_existing_models(verbose=True)

    env = EERackLayoutEnv(
        models_info=models_info,
        show_sites=args.show_sites,
        rack_radius=args.rack_radius,
        slot_spacing=args.slot_spacing,
        arc_center_angle=math.radians(args.arc_center_deg),
        rack_side=args.rack_side,
        common_coupling_z=args.coupling_z,
        has_renderer=False,
        has_offscreen_renderer=args.render,
        render_camera=args.camera,
    )
    env.reset()

    if not args.no_validate:
        res = validate_layout(env, settle_seconds=2.0, verbose=True)
    else:
        res = {"ok": True}

    if args.render:
        env.reset()
        _render_image(env, args.camera, args.camera)
        if args.camera != SIDE_CAMERA:
            _render_image(env, SIDE_CAMERA, SIDE_CAMERA)

    if args.view:
        _run_viewer(env, args.camera, seconds=args.seconds)

    env.close()
    print("\n[done] layout ok =", res["ok"])
    return 0 if res.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
