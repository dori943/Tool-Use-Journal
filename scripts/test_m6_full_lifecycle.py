"""M6 full-lifecycle smoke test (Phase A real + mock re-exec + Phase B real).

PowerShell examples:

  # PASS recovery (default: copy of production memory under output/m6_smoke/)
  python scripts/test_m6_full_lifecycle.py c1_1 --outcome success

  # FAIL recovery
  python scripts/test_m6_full_lifecycle.py c1_1 --outcome fail

  # Explicit mock backends
  python scripts/test_m6_full_lifecycle.py c1_1 --outcome success --backend mock

  # OpenAI backends (requires OPENAI_API_KEY; no silent mock fallback)
  python scripts/test_m6_full_lifecycle.py c1_1 --outcome success --backend openai

  # Append to a chosen memory file (must already exist)
  python scripts/test_m6_full_lifecycle.py c1_1 --outcome success --memory output/m6_smoke/custom.json

  # Intentionally append to production output/memory.json
  python scripts/test_m6_full_lifecycle.py c1_1 --outcome success --use-production-memory

REAL: Failure Context → Retrieval → Diagnosis → Recovery → Dispatch
MOCK: target/downstream module re-execution → recovery M5 result
REAL: process_recovery_outcome → Runtime Experience → M0 append
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tuj.m0_memory.memory_store import UnifiedMemoryStore  # noqa: E402
from tuj.m6_diagnosis import (  # noqa: E402
    M6E2EError,
    process_recovery_outcome,
    run_m6_e2e,
)
from tuj.m6_diagnosis.memory_adapter import DEFAULT_MEMORY_PATH  # noqa: E402
from tuj.m6_diagnosis.recovery_outcome import RecoveryResultError  # noqa: E402

VALID_OUTCOMES = frozenset({"success", "fail"})


def build_mock_recovery_result(*, subgoal_id: str, outcome: str) -> dict[str, Any]:
    """Build an M5 subgoal_result-shaped mock recovery result.

    FAIL uses ``failure_code: null`` (plus detail) so the placeholder is not
    mistaken for an M6 canonical diagnosis taxonomy label.
    """
    if outcome not in VALID_OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(VALID_OUTCOMES)}")
    if not isinstance(subgoal_id, str) or not subgoal_id.strip():
        raise ValueError("subgoal_id is required for mock recovery_result")
    subgoal_id = subgoal_id.strip()

    if outcome == "success":
        return {
            "subgoal_id": subgoal_id,
            "status": "SUCCESS",
            "phase": "execution",
            "failure_code": None,
            "detail": None,
        }
    return {
        "subgoal_id": subgoal_id,
        "status": "FAIL",
        "phase": "execution",
        "failure_code": None,
        "detail": "Mock recovery failure for M6 full-lifecycle smoke test",
    }


def count_experiences(memory_path: Path) -> int:
    store = UnifiedMemoryStore(memory_path)
    experiences = (
        store.document.get("failure_recovery_experience", {}).get("experiences") or []
    )
    if not isinstance(experiences, list):
        return 0
    return len(experiences)


def prepare_smoke_memory(
    *,
    production_memory: Path,
    smoke_memory: Path,
) -> Path:
    """Copy production memory into a smoke working copy (fresh each run)."""
    if not production_memory.is_file():
        raise FileNotFoundError(f"production memory not found: {production_memory}")
    smoke_memory.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(production_memory, smoke_memory)
    return smoke_memory


def _banner(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def _section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def _pp(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _print_initial_m5(failure_context: dict[str, Any]) -> None:
    _section("[1] Initial M5 Failure Result")
    m5_result = failure_context.get("m5_result")
    if m5_result is None:
        print("(failure_context.m5_result is null — adapter may lack subgoal_result.json)")
    else:
        _pp(m5_result)


def _print_failure_context(failure_context: dict[str, Any]) -> None:
    _section("[2] Failure Context (compact)")
    subgoal = failure_context.get("subgoal") or {}
    task = failure_context.get("task") or {}
    task_plan = failure_context.get("task_plan") or {}
    motion_plan = failure_context.get("motion_plan") or {}
    execution = failure_context.get("execution") or {}
    compact = {
        "failure_id": failure_context.get("failure_id"),
        "task": {"task_id": task.get("task_id"), "instruction": task.get("instruction")},
        "subgoal": {
            "subgoal_id": subgoal.get("subgoal_id"),
            "description": subgoal.get("description"),
            "action_type": subgoal.get("action_type"),
            "selected_object_id": subgoal.get("selected_object_id"),
            "selected_object_class": subgoal.get("selected_object_class"),
            "target_object_ids": subgoal.get("target_object_ids"),
        },
        "task_plan": {
            "selected_ee": task_plan.get("selected_ee"),
            "selected_tool": task_plan.get("selected_tool"),
        },
        "motion_plan": {
            "planning_status": motion_plan.get("planning_status"),
            "planning_error": motion_plan.get("planning_error"),
        },
        "execution": {
            "controller_status": execution.get("controller_status"),
            "timeout": execution.get("timeout"),
            "error": execution.get("error"),
        },
        "m5_result": failure_context.get("m5_result"),
        "has_verification_block": "verification" in failure_context,
        "scene_node_count": len((failure_context.get("scene") or {}).get("nodes") or []),
        "grounding_geometry_keys": len(
            (failure_context.get("grounding") or {}).get("geometry") or {}
        ),
    }
    _pp(compact)


def _print_retrieval(retrieved: list[Any]) -> None:
    _section("[3] Retrieved Experiences")
    items = retrieved if isinstance(retrieved, list) else []
    print(f"count = {len(items)}")
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            print(f"{index}. (non-dict entry)")
            continue
        experience = item.get("experience") or {}
        exp_id = experience.get("experience_id") if isinstance(experience, dict) else None
        print(f"{index}. experience_id = {exp_id}")
        if "context_similarity" in item:
            print(f"   similarity    = {item.get('context_similarity')}")
        if "comparison_coverage" in item:
            print(f"   coverage      = {item.get('comparison_coverage')}")


def _print_diagnosis(diagnosis: dict[str, Any]) -> None:
    _section("[4] Diagnosis")
    cause = diagnosis.get("failure_cause") or {}
    print(f"failure_type    : {diagnosis.get('failure_type')}")
    print(f"failure_cause   : {cause.get('code')}")
    print(f"description     : {cause.get('description')}")
    print(f"affected_module : {diagnosis.get('affected_module')}")
    conf = diagnosis.get("confidence")
    if isinstance(conf, (int, float)):
        print(f"confidence      : {conf:.2f}")
    else:
        print(f"confidence      : {conf}")


def _print_recovery(recovery: dict[str, Any]) -> None:
    _section("[5] Recovery")
    action = recovery.get("action") or {}
    routing = recovery.get("routing") or {}
    print(f"decision_mode     : {recovery.get('decision_mode')}")
    print(f"recovery_category : {recovery.get('recovery_category')}")
    print(f"recovery_type     : {action.get('recovery_type')}")
    print(f"target_module     : {action.get('target_module')}")
    print(f"restart_from      : {routing.get('restart_from')}")
    print(f"rerun_modules     : {routing.get('rerun_modules')}")
    print(f"invalidate        : {routing.get('invalidate')}")


def _print_dispatch(dispatch: dict[str, Any]) -> None:
    _section("[6] Recovery Dispatch")
    print(f"target_module   : {dispatch.get('target_module')}")
    print(f"recovery_type   : {dispatch.get('recovery_type')}")
    print(f"request_path    : {dispatch.get('request_path')}")
    print(f"module_executed : {dispatch.get('module_executed')}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "M6 full-lifecycle smoke: real Phase A (through Dispatch), "
            "mock module re-execution, real Phase B (Runtime Experience + M0 append)."
        )
    )
    parser.add_argument(
        "task_id",
        help="Task id under output/ (e.g. c1_1)",
    )
    parser.add_argument(
        "--outcome",
        choices=sorted(VALID_OUTCOMES),
        required=True,
        help="Mock recovery result: success → SUCCESS/PASS, fail → FAIL/FAIL",
    )
    parser.add_argument(
        "--backend",
        choices=("mock", "openai"),
        default=None,
        help=(
            "Set both diagnosis and recovery backends "
            "(overrides env; openai requires OPENAI_API_KEY; no mock fallback)"
        ),
    )
    parser.add_argument(
        "--diagnoser-backend",
        choices=("mock", "openai"),
        default=None,
        help="Override diagnosis backend only (same as run_m6_e2e.py)",
    )
    parser.add_argument(
        "--recovery-backend",
        choices=("mock", "openai"),
        default=None,
        help="Override recovery backend only (same as run_m6_e2e.py)",
    )
    parser.add_argument(
        "--subgoal",
        default=None,
        help="Failed subgoal/detail id (optional; auto-detect when unambiguous)",
    )
    parser.add_argument(
        "--memory",
        type=Path,
        default=None,
        help=(
            "Existing memory.json used for Phase A retrieval and Phase B append. "
            "Default: fresh copy of production memory under output/m6_smoke/."
        ),
    )
    parser.add_argument(
        "--use-production-memory",
        action="store_true",
        help=(
            "Append Runtime Experience to production output/memory.json "
            "(mutually exclusive with --memory)"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root containing task dirs (default: <repo>/output)",
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
        help="Do not write output/<task>/m6 Phase A artifacts",
    )
    return parser


def resolve_memory_path(args: argparse.Namespace) -> tuple[Path, str]:
    """Return (memory_path, mode_description)."""
    if args.use_production_memory and args.memory is not None:
        raise SystemExit("error: --memory and --use-production-memory are mutually exclusive")

    production = DEFAULT_MEMORY_PATH
    if args.use_production_memory:
        if not production.is_file():
            raise SystemExit(f"error: production memory not found: {production}")
        return production, "production (WARNING: will append to output/memory.json)"

    if args.memory is not None:
        path = Path(args.memory).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(
                f"error: --memory file does not exist: {path}\n"
                "Provide an existing memory.json (or omit --memory to auto-copy production)."
            )
        return path, f"explicit ({path})"

    smoke_dir = ROOT / "output" / "m6_smoke"
    smoke_path = smoke_dir / f"{args.task_id}_memory_smoke.json"
    prepare_smoke_memory(production_memory=production, smoke_memory=smoke_path)
    return smoke_path.resolve(), f"smoke copy ({smoke_path})"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    diagnoser_backend = args.diagnoser_backend or args.backend
    recovery_backend = args.recovery_backend or args.backend

    try:
        memory_path, memory_mode = resolve_memory_path(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    mock_label = "SUCCESS" if args.outcome == "success" else "FAIL"

    _banner("M6 FULL LIFECYCLE SMOKE TEST")
    print(f"Task: {args.task_id}")
    print(f"Mock recovery outcome: {mock_label}")
    print(f"Diagnoser backend: {diagnoser_backend or '(env/default)'}")
    print(f"Recovery backend : {recovery_backend or '(env/default)'}")
    print(f"Memory mode      : {memory_mode}")
    print()
    print("WARNING:")
    print("  Phase B appends one Runtime Experience to the memory file above.")
    print("  Default mode uses a fresh copy under output/m6_smoke/ (production intact).")
    print("  --use-production-memory appends to output/memory.json.")
    print()

    # ------------------------------------------------------------------ #
    # Phase A — real M6 through Dispatch
    # ------------------------------------------------------------------ #
    try:
        phase_a = run_m6_e2e(
            task_id=args.task_id,
            subgoal_id=args.subgoal,
            output_root=args.output_root,
            memory_path=memory_path,
            diagnoser_backend=diagnoser_backend,
            recovery_backend=recovery_backend,
            top_k=args.top_k,
            save=not args.no_save,
            print_summary=False,
        )
    except M6E2EError as exc:
        print(f"error (Phase A): {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface loudly for smoke debugging
        print(f"error (Phase A): {exc}", file=sys.stderr)
        return 1

    failure_context = phase_a["failure_context"]
    diagnosis = phase_a["diagnosis"]
    recovery = phase_a["recovery"]
    dispatch = phase_a["dispatch"]
    retrieved = phase_a.get("retrieved_experiences") or []
    failed_subgoal_id = phase_a["subgoal_id"]

    _print_initial_m5(failure_context)
    _print_failure_context(failure_context)
    _print_retrieval(retrieved)
    _print_diagnosis(diagnosis)
    _print_recovery(recovery)
    _print_dispatch(dispatch if isinstance(dispatch, dict) else {})

    # ------------------------------------------------------------------ #
    # MOCK BOUNDARY — no real module re-execution
    # ------------------------------------------------------------------ #
    _section("[MOCK BOUNDARY]")
    print("Actual module re-execution is NOT performed.")
    print("A mock M5 recovery result will be injected.")
    print("Do not treat this as evidence that target modules re-ran.")

    mock_recovery_result = build_mock_recovery_result(
        subgoal_id=failed_subgoal_id,
        outcome=args.outcome,
    )
    _section("[7] Mock Recovery Result")
    _pp(mock_recovery_result)

    experiences_before = count_experiences(memory_path)

    # ------------------------------------------------------------------ #
    # Phase B — real Runtime Experience + M0 append
    # ------------------------------------------------------------------ #
    try:
        phase_b = process_recovery_outcome(
            failure_context=failure_context,
            diagnosis=diagnosis,
            recovery=recovery,
            recovery_result=mock_recovery_result,
            memory_path=memory_path,
        )
    except RecoveryResultError as exc:
        print(f"error (Phase B validation): {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error (Phase B): {exc}", file=sys.stderr)
        return 1

    experience = phase_b.get("experience") or {}
    outcome = phase_b.get("outcome") or {}
    experiences_after = count_experiences(memory_path)
    delta = experiences_after - experiences_before

    _section("[8] Runtime Experience")
    _pp(experience)

    _section("[9] M0 Update")
    print(f"memory_path        : {memory_path}")
    print(f"experiences before : {experiences_before}")
    print(f"experiences after  : {experiences_after}")
    print(f"delta              : {delta:+d}")

    recovery_outcome = outcome.get("status")
    memory_updated = bool(phase_b.get("appended")) and delta == 1

    _section("[10] Final Result")
    print(f"Recovery Outcome : {recovery_outcome}")
    print(f"Memory Updated   : {'YES' if memory_updated else 'NO'}")
    print(f"Experience ID    : {phase_b.get('experience_id')}")
    print(f"Phase A subgoal  : {failed_subgoal_id}")
    print(f"Dispatch target  : {(dispatch or {}).get('target_module')}")
    print(f"rerun_modules    : {((recovery or {}).get('routing') or {}).get('rerun_modules')}")
    print()
    print("Note: Recovery Outcome FAIL still means a successful smoke lifecycle")
    print("      if Runtime Experience was built and appended (+1).")

    _banner("SMOKE TEST COMPLETE")

    # Smoke script success ≠ recovery PASS. Lifecycle OK when append succeeded.
    if not memory_updated:
        print("error: expected experiences delta +1 after Phase B", file=sys.stderr)
        return 1
    if recovery_outcome not in {"PASS", "FAIL"}:
        print(f"error: unexpected recovery outcome {recovery_outcome!r}", file=sys.stderr)
        return 1
    if args.outcome == "success" and recovery_outcome != "PASS":
        print("error: --outcome success expected Runtime outcome PASS", file=sys.stderr)
        return 1
    if args.outcome == "fail" and recovery_outcome != "FAIL":
        print("error: --outcome fail expected Runtime outcome FAIL", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
