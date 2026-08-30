"""설치 없이 M4를 import 할 수 있게 경로를 잡는다 — scripts/ 실행기와 같은 규약."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
_REPOSITORY = _SRC.parent
_TASK_PLANNER_SOURCES = (
    _REPOSITORY.parent / "dain-m3" / "src",
    _REPOSITORY.parent / "tuj-m3" / "src",
)

for path in reversed((_SRC, *_TASK_PLANNER_SOURCES, _REPOSITORY)):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
