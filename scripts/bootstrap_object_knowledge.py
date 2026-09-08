"""Bootstrap M0 Object Knowledge from an existing production m3.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tuj.m3_grounding import PropertyMemory  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--m3", default=None)
    parser.add_argument("--output", default=str(ROOT / "output" / "memory.json"))
    args = parser.parse_args()
    source = Path(args.m3) if args.m3 else ROOT / "output" / args.task / "m3.json"
    manager = PropertyMemory(args.output, task_id=args.task)
    result = manager.update_from_m3(source, args.task)
    print(f"[M0 Object Knowledge] source={source} output={args.output}")
    print(f"updated={len(result['updated'])}: {result['updated']}")
    print(f"skipped={len(result['skipped'])}: {result['skipped']}")


if __name__ == "__main__":
    main()
