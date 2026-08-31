"""프로젝트 커스텀 robosuite 환경.

``robosuite.make`` 호출 전에 import 해야 환경이 등록된다.

등록 목록은 task_registry.TASKS(단일 출처)를 따른다. 태스크를 늘릴 때 여기를
고칠 필요가 없다 — task_registry 에 한 줄만 추가하면 된다. RoboCasa(KitchenBase)
의존 환경(robocasa=True)은 별도 설치될 수 있으므로 soft-import 하고, 미설치 시
해당 클래스는 None 이 되어 __all__ 에서 빠진다.
"""

import importlib

from task_registry import TASKS as _TASKS

__all__ = []

for _tid, _t in _TASKS.items():
    try:
        _module = importlib.import_module(f"environments.{_t.module}")
        _cls = getattr(_module, _t.cls)
    except ImportError:
        # RoboCasa 미설치 등으로 로드 불가. 비의존 환경이 실패하면 진짜 오류이므로 전파.
        if not _t.robocasa:
            raise
        globals()[_t.cls] = None  # type: ignore[assignment]
    else:
        globals()[_t.cls] = _cls
        __all__.append(_t.cls)

# RoboCasa 베이스 클래스도 soft-import (환경들이 상속).
try:
    from environments.kitchen_base import KitchenBase
except ImportError:
    KitchenBase = None  # type: ignore[misc, assignment]
else:
    __all__.append("KitchenBase")
