"""Run generic M5 planning for any compatible M4 SelectedPlan input."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    REPOSITORY / "src",
    REPOSITORY.parent / "dain-m3" / "src",
    REPOSITORY.parent / "tuj-m3" / "src",
)


def main(argv: Sequence[str] | None = None) -> int:
    for source_root in reversed(SOURCE_ROOTS):
        if source_root.is_dir() and str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
    from tuj.m5_motion.generic_runner import main as generic_main

    return generic_main(argv, repository=REPOSITORY)


if __name__ == "__main__":
    raise SystemExit(main())
