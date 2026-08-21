"""Make the package and its sibling Task_Planner importable without installation."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TASK_PLANNER = _ROOT.parent / "Task_Planner"

for path in (_ROOT, _TASK_PLANNER):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
