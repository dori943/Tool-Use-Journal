"""프로젝트 커스텀 robosuite 환경.

``robosuite.make`` 호출 전에 import 해야 환경이 등록된다.
"""

from environments.c1_1_lego_sweep import C1_1_LegoSweep
from environments.c2_1_object_sorting import C2_1_ObjectSorting

__all__ = [
    "C1_1_LegoSweep",
    "C2_1_ObjectSorting",
]

# RoboCasa는 별도 환경에 설치될 수 있으므로 soft-import 한다.
try:
    from environments.kitchen_base import KitchenBase
except ImportError:
    KitchenBase = None  # type: ignore[misc, assignment]
else:
    __all__.append("KitchenBase")

try:
    from environments.c1_2_dough_flatten import C1_2_DoughFlatten
except ImportError:
    C1_2_DoughFlatten = None  # type: ignore[misc, assignment]
else:
    __all__.append("C1_2_DoughFlatten")

try:
    from environments.c2_2_sandwich_assembly import C2_2_SandwichAssembly
except ImportError:
    C2_2_SandwichAssembly = None  # type: ignore[misc, assignment]
else:
    __all__.append("C2_2_SandwichAssembly")
