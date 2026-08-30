"""Run the C1_1-specific physical M5 grasp and sweep workflow.

The selected end-effector, tool, targets, and goal are read only from the M4
Task Planner result. Unknown options are forwarded to the physical C1_1 runner.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY / "src"
MOTION_RUNNER = (
    SOURCE_ROOT
    / "tuj"
    / "m5_motion"
    / "examples"
    / "c1_1_openai_motion_run.py"
)
DEFAULT_MOTION_PROFILE = (
    SOURCE_ROOT
    / "tuj"
    / "m5_motion"
    / "examples"
    / "c1_1_physical_grasp_profile.json"
)


def _source_roots() -> list[Path]:
    return [
        SOURCE_ROOT,
        REPOSITORY.parent / "dain-m3" / "src",
        REPOSITORY.parent / "tuj-m3" / "src",
    ]


def _default_task_planner_result() -> Path:
    candidates = [
        REPOSITORY / "output" / "c1_1" / "task_planner.json",
        REPOSITORY.parent
        / "dain-m3"
        / "output"
        / "c1_1"
        / "task_planner.json",
        REPOSITORY.parent
        / "tuj-m3"
        / "output"
        / "c1_1"
        / "task_planner.json",
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=REPOSITORY,
        help="Tool-Use-Journal checkout containing the runtime assets",
    )
    parser.add_argument(
        "--task-planner",
        type=Path,
        default=_default_task_planner_result(),
        help="M4 Task Planner result JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY / "artifacts" / "c1_1",
        help="directory for M5 plans, simulation records, and summaries",
    )
    parser.add_argument(
        "--pick-keyframes",
        type=Path,
        help="reuse an existing relative PICK keyframe artifact",
    )
    parser.add_argument(
        "--motion-profile",
        type=Path,
        default=DEFAULT_MOTION_PROFILE,
        help="validated C1_1 physical grasp, contact, and recovery settings",
    )
    parser.add_argument(
        "--sweep-provider",
        choices=("task-geometry", "openai"),
        default="task-geometry",
    )
    parser.add_argument(
        "--validate-input-only",
        action="store_true",
        help="validate the M4-to-M5 contract without OpenAI or MuJoCo execution",
    )
    parser.add_argument(
        "--stop-after-pick",
        action="store_true",
        help="stop after planning and physically validating the selected-tool pick",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved command without executing it",
    )
    return parser


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _pythonpath_environment() -> dict[str, str]:
    environment = os.environ.copy()
    roots = [path for path in _source_roots() if path.is_dir()]
    existing = environment.get("PYTHONPATH")
    if existing:
        roots.append(Path(existing))
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in roots)
    return environment


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args, forwarded = parser.parse_known_args(argv)

    repository = _resolved(args.repository)
    task_planner = _resolved(args.task_planner)
    output_dir = _resolved(args.output_dir)
    pick_keyframes = (
        _resolved(args.pick_keyframes) if args.pick_keyframes is not None else None
    )
    motion_profile = _resolved(args.motion_profile)

    if not repository.is_dir():
        parser.error(f"repository not found: {repository}")
    if not MOTION_RUNNER.is_file():
        parser.error(f"Motion Planner runner not found: {MOTION_RUNNER}")
    if not task_planner.is_file():
        parser.error(
            "M4 Task Planner result not found: "
            f"{task_planner}. Run `scripts/run_m4_task_planner.py` in the "
            "Task Planner checkout first or pass --task-planner explicitly."
        )
    if pick_keyframes is not None and not pick_keyframes.is_file():
        parser.error(f"PICK keyframe artifact not found: {pick_keyframes}")
    if not motion_profile.is_file():
        parser.error(f"motion profile not found: {motion_profile}")

    needs_openai = not args.dry_run and not args.validate_input_only and (
        pick_keyframes is None or args.sweep_provider == "openai"
    )
    if needs_openai and not os.environ.get("OPENAI_API_KEY"):
        parser.error(
            "OPENAI_API_KEY is required for keyframe generation. Set the "
            "environment variable, pass --pick-keyframes with the default "
            "task-geometry sweep, or use --validate-input-only."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(MOTION_RUNNER),
        str(repository),
        "--task-planner",
        str(task_planner),
        "--output-dir",
        str(output_dir),
        "--sweep-provider",
        args.sweep_provider,
        "--motion-profile",
        str(motion_profile),
    ]
    if pick_keyframes is not None:
        command.extend(("--pick-keyframes", str(pick_keyframes)))
    if args.validate_input_only:
        command.append("--validate-input-only")
    if args.stop_after_pick:
        command.append("--stop-after-pick")
    command.extend(forwarded)

    print("[M5:C1_1] " + shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    try:
        completed = subprocess.run(
            command,
            cwd=repository,
            env=_pythonpath_environment(),
            check=False,
        )
    except KeyboardInterrupt:
        return 130
    if completed.returncode == 0:
        print(f"[M5:C1_1] output: {output_dir}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
