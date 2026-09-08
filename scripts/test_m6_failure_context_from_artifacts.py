"""Build Failure Context from task output artifacts (no diagnosis/recovery).

Examples (PowerShell):

  python scripts/test_m6_failure_context_from_artifacts.py `
    --output-dir output/c1_1 `
    --subgoal-id SG1_s1_d1

  python scripts/test_m6_failure_context_from_artifacts.py `
    --output-dir output/c1_1 `
    --subgoal-id SG1_s1_d1 `
    --save
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tuj.m6_diagnosis.artifact_adapter import (  # noqa: E402
    ArtifactAdapterError,
    FailureContextArtifactAdapter,
)
from tuj.m6_diagnosis.failure_context import FailureContextBuilder  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load M1–M5 artifacts into canonical M6 Failure Context. "
            "Does not run retrieval, diagnosis, or recovery."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Task output directory (e.g. output/c1_1)",
    )
    parser.add_argument(
        "--subgoal-id",
        required=True,
        help="Failed subgoal or detail id present in m2.json",
    )
    parser.add_argument(
        "--failure-id",
        default=None,
        help="Optional deterministic failure_id override",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write debug artifact to <output-dir>/m6/failure_context.json",
    )
    return parser


def _summarize(context: dict) -> None:
    scene = context.get("scene") or {}
    grounding = context.get("grounding") or {}
    task_plan = context.get("task_plan") or {}
    motion = context.get("motion_plan") or {}
    execution = context.get("execution") or {}
    verification = context.get("verification") or {}
    subgoal = context.get("subgoal") or {}
    task = context.get("task") or {}

    print("===== Failure Context Summary =====")
    print(f"failure_id: {context.get('failure_id')}")
    print(f"task.task_id: {task.get('task_id')}")
    print(f"task.instruction: {task.get('instruction')}")
    print(f"subgoal.subgoal_id: {subgoal.get('subgoal_id')}")
    print(f"subgoal.description: {subgoal.get('description')}")
    print(f"subgoal.action_type: {subgoal.get('action_type')}")
    print(f"scene.nodes: {len(scene.get('nodes') or [])}")
    print(f"scene.relations: {len(scene.get('relations') or [])}")
    print(
        "grounding keys: "
        f"physical={len(grounding.get('physical_properties') or {})}, "
        f"geometry={len(grounding.get('geometry') or {})}, "
        f"ee={len(grounding.get('ee_feasibility') or {})}, "
        f"confidence={len(grounding.get('confidence') or {})}"
    )
    print(f"task_plan.selected_ee: {task_plan.get('selected_ee')}")
    print(f"task_plan.selected_tool: {task_plan.get('selected_tool')}")
    print(f"motion_plan.planning_status: {motion.get('planning_status')}")
    error = motion.get("planning_error")
    if isinstance(error, list) and error:
        first = error[0] if isinstance(error[0], dict) else {}
        print(
            "motion_plan.planning_error[0].failure_code: "
            f"{first.get('failure_code')}"
        )
    else:
        print(f"motion_plan.planning_error: {error}")
    print(f"execution.controller_status: {execution.get('controller_status')}")
    print(f"execution.executed_actions: {len(execution.get('executed_actions') or [])}")
    print(f"verification.result: {verification.get('result')}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()

    memory_path = ROOT / "output" / "memory.json"
    memory_before = None
    if memory_path.is_file():
        memory_before = hashlib.sha256(memory_path.read_bytes()).hexdigest()

    adapter = FailureContextArtifactAdapter(output_dir)
    try:
        pipeline_state = adapter.build_pipeline_state(
            failed_subgoal_id=args.subgoal_id,
            failure_id=args.failure_id,
        )
    except ArtifactAdapterError as error:
        print(f"[err] {error}", file=sys.stderr)
        return 2

    context = FailureContextBuilder().build(pipeline_state)
    _summarize(context)
    print("===== Failure Context JSON =====")
    print(json.dumps(context, indent=2, ensure_ascii=False))

    if args.save:
        m6_dir = output_dir / "m6"
        m6_dir.mkdir(parents=True, exist_ok=True)
        out_path = m6_dir / "failure_context.json"
        out_path.write_text(
            json.dumps(context, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\n[saved] {out_path}")

    if memory_before is not None:
        memory_after = hashlib.sha256(memory_path.read_bytes()).hexdigest()
        if memory_before != memory_after:
            print("[err] output/memory.json changed", file=sys.stderr)
            return 3
        print("\n[ok] output/memory.json unchanged")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
