"""Vacuum 흡착 그리퍼. robosuite 등록용 공유 모듈.

숫자로 시작하는 스크립트는 import할 수 없어 여기에 두고
`from _vacuum import register_vacuum`으로 재사용한다.
모델은 scripts/assets/vacuum_gripper.xml 이다.
"""

import os

GRIPPER_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "assets", "vacuum_gripper.xml")

_registered = None


def register_vacuum():
    """VacuumGripper를 GRIPPER_MAPPING에 등록하고 클래스를 반환한다. 중복 등록은 하지 않는다."""
    global _registered
    if _registered is not None:
        return _registered

    from robosuite.models.grippers import register_gripper
    from robosuite.models.grippers.gripper_model import GripperModel

    @register_gripper
    class VacuumGripper(GripperModel):
        """흡착 EE. 액션 1차원(-1=off ~ +1=흡착max)을 adhesion ctrl로 매핑한다."""

        def __init__(self, idn=0):
            super().__init__(GRIPPER_XML, idn=idn)

        def format_action(self, action):
            return action  # 컨트롤러가 [-1,1] → ctrlrange(0~1)로 스케일

        @property
        def init_qpos(self):
            return None  # 관절 없음

        @property
        def dof(self):
            return 1

        @property
        def _important_geoms(self):
            return {}

    _registered = VacuumGripper
    return VacuumGripper
