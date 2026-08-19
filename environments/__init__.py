"""프로젝트 커스텀 robosuite 환경.

``robosuite.make`` 호출 전에 import 해야 환경이 등록된다.
"""

from environments.c1_1_lego_sweep import C1_1_LegoSweep
from environments.c2_1_object_sorting import C2_1_ObjectSorting

__all__ = [
    "C1_1_LegoSweep",
    "C2_1_ObjectSorting",
]
