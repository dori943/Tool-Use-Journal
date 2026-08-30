"""Run the M4 Task Planner with repository-local C1_1 defaults.

Run from any working directory:

    python scripts/run_m4_task_planner.py

The selected tool is read from ``gk_bundle.json`` and is never supplied by
this runner.  Explicit paths override every default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY / "src"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gk",
        type=Path,
        default=REPOSITORY / "output" / "c1_1" / "gk_bundle.json",
        help="GK bundle containing the upstream-selected tool",
    )
    parser.add_argument(
        "--m2",
        type=Path,
        default=REPOSITORY / "output" / "c1_1" / "m2.json",
        help="M2 scene graph and subgoal input",
    )
    parser.add_argument(
        "--robot-spec",
        type=Path,
        default=REPOSITORY / "configs" / "robot_spec.json",
        help="robot and end-effector specification",
    )
    parser.add_argument("--m1", type=Path, help="optional separate M1 scene graph")
    parser.add_argument(
        "--initial-state",
        type=Path,
        help="optional normalized initial robot state",
    )
    parser.add_argument("--id-aliases", type=Path)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Task Planner result consumed by M5; defaults to "
            "<GK input folder>/task_planner.json"
        ),
    )
    return parser


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _append_optional(command: list[str], flag: str, path: Path | None) -> None:
    if path is not None:
        command.extend((flag, str(_resolved(path))))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    required_inputs = {
        "--gk": _resolved(args.gk),
        "--m2": _resolved(args.m2),
        "--robot-spec": _resolved(args.robot_spec),
    }
    optional_inputs = {
        "--m1": args.m1,
        "--initial-state": args.initial_state,
        "--id-aliases": args.id_aliases,
        "--resources": args.resources,
        "--candidates": args.candidates,
        "--policy": args.policy,
    }
    missing = [
        f"{flag}={path}"
        for flag, path in required_inputs.items()
        if not path.is_file()
    ]
    missing.extend(
        f"{flag}={_resolved(path)}"
        for flag, path in optional_inputs.items()
        if path is not None and not _resolved(path).is_file()
    )
    if missing:
        parser.error("input file not found: " + ", ".join(missing))

    output = (
        _resolved(args.output)
        if args.output is not None
        else required_inputs["--gk"].parent / "task_planner.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    try:
        from tuj.m4_taskplanner.cli import main as task_planner_main
    except ModuleNotFoundError as exc:
        parser.error(
            f"missing dependency {exc.name!r}; run "
            "`python -m pip install -r requirements.txt`"
        )

    command = [
        "plan",
        "--gk",
        str(required_inputs["--gk"]),
        "--m2",
        str(required_inputs["--m2"]),
        "--robot-spec",
        str(required_inputs["--robot-spec"]),
        "--output",
        str(output),
    ]
    for flag, path in optional_inputs.items():
        _append_optional(command, flag, path)

    print(f"[M4] Task Planner input: {required_inputs['--gk']}")
    exit_code = task_planner_main(command)
    if exit_code != 0:
        return exit_code

    result = json.loads(output.read_text(encoding="utf-8"))
    assignments = (result.get("selected_plan") or {}).get(
        "candidate_assignments", []
    )
    selected_pairs = list(
        dict.fromkeys(
            (item.get("ee"), item.get("tool"))
            for item in assignments
            if item.get("ee") is not None
        )
    )
    selection = ", ".join(
        f"EE={ee}, tool={tool or 'none'}" for ee, tool in selected_pairs
    )
    print(f"[M4] SUCCESS: {selection or 'no assignments'}")
    print(f"[M4] output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
