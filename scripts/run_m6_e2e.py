"""Common M6 end-to-end smoke runner (artifacts → Recovery, no re-execution).

Examples (PowerShell):

  # Mock backends (default; no API cost)
  python scripts/run_m6_e2e.py --task c1_1 --subgoal SG1_s1_d1

  # OpenAI diagnosis + recovery
  python scripts/run_m6_e2e.py `
    --task c1_1 `
    --subgoal SG1_s1_d1 `
    --diagnoser-backend openai `
    --recovery-backend openai

  # List tasks with/without M5 failure artifacts (no API calls)
  python scripts/run_m6_e2e.py --list-tasks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tuj.m6_diagnosis.e2e_runner import (  # noqa: E402
    M6E2EError,
    list_failure_task_availability,
    run_m6_e2e,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run M6 end-to-end from output/<task>/ M1–M5 artifacts through "
            "Recovery + Routing. Does not re-execute modules or update memory."
        )
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task id under output/ (e.g. c1_1, c2_1)",
    )
    parser.add_argument(
        "--subgoal",
        default=None,
        help=(
            "Failed subgoal/detail id. If omitted, auto-detect from M5 failure "
            "artifact when unambiguous."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root containing task dirs (default: <repo>/output)",
    )
    parser.add_argument(
        "--memory",
        type=Path,
        default=None,
        help="Path to memory.json (default: output/memory.json)",
    )
    parser.add_argument(
        "--diagnoser-backend",
        choices=("mock", "openai"),
        default=None,
        help="Override M6_DIAGNOSER_BACKEND (default: mock via env/policy)",
    )
    parser.add_argument(
        "--recovery-backend",
        choices=("mock", "openai"),
        default=None,
        help="Override M6_RECOVERY_ROUTER_BACKEND (default: mock via env/policy)",
    )
    parser.add_argument(
        "--diagnosis-model",
        default=None,
        help="Override M6_DIAGNOSIS_MODEL",
    )
    parser.add_argument(
        "--recovery-model",
        default=None,
        help="Override M6_RECOVERY_MODEL",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Retrieval top-k (default: 3)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write output/<task>/m6 artifacts",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List task dirs and whether M5 failure artifacts exist (no API calls)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.list_tasks:
        rows = list_failure_task_availability(output_root=args.output_root)
        if not rows:
            print("No task directories with m2.json found under output/")
            return 0
        print("Available failure tasks:")
        for row in rows:
            marker = "failure artifact found" if row.has_failure_artifact else "no failure artifact"
            print(f"- {row.task_id} : {marker}")
        return 0

    if not args.task:
        print("error: --task is required unless --list-tasks is set", file=sys.stderr)
        return 2

    try:
        result = run_m6_e2e(
            task_id=args.task,
            subgoal_id=args.subgoal,
            output_root=args.output_root,
            memory_path=args.memory,
            diagnoser_backend=args.diagnoser_backend,
            recovery_backend=args.recovery_backend,
            diagnosis_model=args.diagnosis_model,
            recovery_model=args.recovery_model,
            top_k=args.top_k,
            save=not args.no_save,
            print_summary=True,
        )
    except M6E2EError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface validation/API failures loudly
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary = result["summary"]
    print()
    print(f"Saved under: {result['m6_dir']}")
    print(
        "memory unchanged: "
        f"{summary['memory_sha256_before'] == summary['memory_sha256_after']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
