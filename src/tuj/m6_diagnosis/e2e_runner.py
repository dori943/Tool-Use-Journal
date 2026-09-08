"""Task-agnostic M6 end-to-end runner (artifacts → Recovery, no re-execution)."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_adapter import ArtifactAdapterError, FailureContextArtifactAdapter
from .diagnosis import apply_diagnosis_output
from .diagnosis_aware_selection import DiagnosisAwareExperienceSelector
from .diagnosis_config import create_failure_diagnoser, get_diagnoser_backend
from .evidence import prepare_diagnosis_evidence, prepare_recovery_evidence
from .memory_adapter import DEFAULT_MEMORY_PATH, MemoryAdapter
from .recovery_config import create_recovery_router, get_recovery_router_backend
from .recovery_dispatcher import RecoveryDispatchError, dispatch_recovery
from .recovery_router import apply_recovery_output
from .retrieval_query import build_retrieval_query
from .schemas import empty_diagnosis, empty_recovery


class M6E2EError(RuntimeError):
    """Raised for clear, user-facing E2E orchestration failures."""


@dataclass(frozen=True)
class FailureTaskAvailability:
    task_id: str
    output_dir: Path
    has_failure_artifact: bool
    reason: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_task_output_dir(
    task_id: str,
    *,
    output_root: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    if not isinstance(task_id, str) or not task_id.strip():
        raise M6E2EError("task_id is required")
    task_id = task_id.strip()
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[3]
    base = Path(output_root) if output_root is not None else root / "output"
    output_dir = (base / task_id).expanduser().resolve()
    if not output_dir.is_dir():
        raise M6E2EError(f"task output directory does not exist: {output_dir}")
    return output_dir


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _known_subgoal_ids(m2: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    # Prefer canonical M2 key; tolerate a legacy alias if present.
    subgoals = m2.get("m2_subgoals")
    if not isinstance(subgoals, list):
        subgoals = m2.get("subgoals") or []
    for subgoal in subgoals:
        if not isinstance(subgoal, Mapping):
            continue
        sid = subgoal.get("subgoal_id")
        if isinstance(sid, str) and sid.strip():
            ids.append(sid.strip())
        for detail in subgoal.get("details") or []:
            if not isinstance(detail, Mapping):
                continue
            did = detail.get("detail_id")
            if isinstance(did, str) and did.strip():
                ids.append(did.strip())
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def collect_failure_strategy_ids(output_dir: Path) -> list[str]:
    failures = _read_json(Path(output_dir) / "m5" / "m5_failure.json")
    if not isinstance(failures, list):
        return []
    strategy_ids: list[str] = []
    for item in failures:
        if not isinstance(item, Mapping):
            continue
        strategy_id = item.get("strategy_id")
        if isinstance(strategy_id, str) and strategy_id.strip():
            strategy_ids.append(strategy_id.strip())
    return strategy_ids


def has_m5_failure_artifact(output_dir: Path) -> bool:
    """True when on-disk M5 failure evidence is present for M6 diagnosis.

    Official trigger contract is ``m5/subgoal_result.json`` with ``status == FAIL``.
    Legacy ``m5_failure.json`` / failed ``m5_summary.json`` remain accepted so
    existing E2E smoke fixtures keep working; ``scripts/run.py`` itself must not
    treat those legacy files as the M6 trigger.
    """
    output_dir = Path(output_dir)
    subgoal_result = _read_json(output_dir / "m5" / "subgoal_result.json")
    if isinstance(subgoal_result, Mapping):
        status = subgoal_result.get("status")
        if isinstance(status, str) and status.strip() == "FAIL":
            return True

    failures = _read_json(output_dir / "m5" / "m5_failure.json")
    if isinstance(failures, list) and any(isinstance(item, Mapping) for item in failures):
        return True

    summary = _read_json(output_dir / "m5" / "m5_summary.json")
    if isinstance(summary, Mapping):
        status = summary.get("planning_status")
        if not isinstance(status, str):
            status = summary.get("status")
        if isinstance(status, str) and status.upper() == "FAILED":
            return True
    return False


def match_subgoal_from_strategy_id(strategy_id: str, known_ids: Sequence[str]) -> str | None:
    """Map an M5 strategy_id to an explicit m2 subgoal/detail id when unambiguous."""
    if not isinstance(strategy_id, str) or not strategy_id.strip():
        return None
    strategy_id = strategy_id.strip()
    known = [item for item in known_ids if isinstance(item, str) and item.strip()]
    if not known:
        return None

    prefix = strategy_id.split(":", 1)[0].strip()
    if prefix in known:
        return prefix

    contained = [item for item in known if item in strategy_id]
    if not contained:
        return None
    # Prefer the longest explicit id to avoid parent/child ambiguity (SG1 vs SG1_s1_d1).
    best_len = max(len(item) for item in contained)
    best = [item for item in contained if len(item) == best_len]
    if len(best) != 1:
        return None
    return best[0]


def discover_failed_subgoal_candidates(output_dir: Path) -> list[str]:
    output_dir = Path(output_dir)
    m2 = _read_json(output_dir / "m2.json")
    if not isinstance(m2, Mapping):
        raise M6E2EError(f"required artifact missing or invalid: {output_dir / 'm2.json'}")

    known_ids = _known_subgoal_ids(m2)
    candidates: list[str] = []
    seen: set[str] = set()

    # Prefer the official M5 subgoal_result contract when present.
    subgoal_result = _read_json(output_dir / "m5" / "subgoal_result.json")
    if isinstance(subgoal_result, Mapping):
        status = subgoal_result.get("status")
        result_id = subgoal_result.get("subgoal_id")
        if (
            isinstance(status, str)
            and status.strip() == "FAIL"
            and isinstance(result_id, str)
            and result_id.strip()
        ):
            rid = result_id.strip()
            if rid in known_ids and rid not in seen:
                seen.add(rid)
                candidates.append(rid)
            elif rid not in known_ids:
                # Still surface the contract id so callers can fail loudly later.
                if rid not in seen:
                    seen.add(rid)
                    candidates.append(rid)

    for strategy_id in collect_failure_strategy_ids(output_dir):
        matched = match_subgoal_from_strategy_id(strategy_id, known_ids)
        if matched is None or matched in seen:
            continue
        seen.add(matched)
        candidates.append(matched)
    return candidates


def resolve_failed_subgoal_id(output_dir: Path, subgoal_id: str | None = None) -> str:
    output_dir = Path(output_dir)
    if isinstance(subgoal_id, str) and subgoal_id.strip():
        return subgoal_id.strip()

    if not has_m5_failure_artifact(output_dir):
        raise M6E2EError("No failure artifact available for M6 failure diagnosis")

    candidates = discover_failed_subgoal_candidates(output_dir)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise M6E2EError(
            "Could not auto-detect failed subgoal from M5 failure artifact. "
            "Pass --subgoal explicitly."
        )
    joined = ", ".join(candidates)
    raise M6E2EError(
        "Ambiguous failed subgoal candidates from M5 failure artifact: "
        f"{joined}. Pass --subgoal to select one."
    )


def list_failure_task_availability(
    *,
    output_root: str | Path | None = None,
    project_root: str | Path | None = None,
) -> list[FailureTaskAvailability]:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[3]
    base = Path(output_root) if output_root is not None else root / "output"
    if not base.is_dir():
        return []

    rows: list[FailureTaskAvailability] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "m2.json").is_file():
            continue
        if has_m5_failure_artifact(child):
            rows.append(
                FailureTaskAvailability(
                    task_id=child.name,
                    output_dir=child,
                    has_failure_artifact=True,
                    reason="failure artifact found",
                )
            )
        else:
            rows.append(
                FailureTaskAvailability(
                    task_id=child.name,
                    output_dir=child,
                    has_failure_artifact=False,
                    reason="no failure artifact",
                )
            )
    return rows


def _require_openai_api_key(backend_name: str) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise M6E2EError(
            f"OPENAI_API_KEY is not set but {backend_name} backend='openai' was requested"
        )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _compact_retrieved(retrieved: list[dict]) -> list[dict]:
    compact: list[dict] = []
    for item in retrieved:
        experience = item.get("experience") or {}
        compact.append(
            {
                "experience_id": experience.get("experience_id"),
                "context_similarity": item.get("context_similarity"),
                "comparison_coverage": item.get("comparison_coverage"),
                "candidate_valid": item.get("candidate_valid"),
                "candidate_validity": item.get("candidate_validity"),
                "source": (experience.get("metadata") or {}).get("source"),
            }
        )
    return compact


def build_e2e_summary(
    *,
    task_id: str,
    subgoal_id: str,
    retrieval_query: dict,
    retrieved: list[dict],
    diagnosis: dict,
    selection_result: dict,
    recovery: dict,
    memory_before: str,
    memory_after: str,
    diagnoser_backend: str,
    recovery_backend: str,
    dispatch: dict | None = None,
) -> dict[str, Any]:
    cause = ((diagnosis.get("failure_cause") or {}).get("code"))
    action = recovery.get("action") or {}
    routing = recovery.get("routing") or {}
    summary = {
        "task_id": task_id,
        "subgoal_id": subgoal_id,
        "backends": {
            "diagnoser": diagnoser_backend,
            "recovery": recovery_backend,
        },
        "retrieval": {
            "candidate_count": len(retrieved),
            "experience_ids": [
                (item.get("experience") or {}).get("experience_id") for item in retrieved
            ],
            "query": {
                "action_type": retrieval_query.get("action_type"),
                "target": deepcopy(retrieval_query.get("target") or {}),
                "selected_ee": retrieval_query.get("selected_ee"),
                "selected_tool": retrieval_query.get("selected_tool"),
                "execution_signature": deepcopy(
                    retrieval_query.get("execution_signature") or {}
                ),
            },
        },
        "diagnosis": {
            "failure_type": diagnosis.get("failure_type"),
            "failure_cause": cause,
            "affected_module": diagnosis.get("affected_module"),
            "confidence": diagnosis.get("confidence"),
        },
        "selection": {
            "selected_count": selection_result.get("selection_count", 0),
            "selected_experience_ids": list(
                selection_result.get("selected_experience_ids") or []
            ),
        },
        "decision_mode": recovery.get("decision_mode"),
        "recovery": {
            "recovery_category": recovery.get("recovery_category"),
            "recovery_type": action.get("recovery_type"),
            "target_module": action.get("target_module"),
            "rerun_modules": list(routing.get("rerun_modules") or []),
        },
        "routing": {
            "restart_from": routing.get("restart_from"),
            "rerun_modules": list(routing.get("rerun_modules") or []),
            "invalidate": list(routing.get("invalidate") or []),
        },
        "selective_reexecution_performed": False,
        "memory_updated": False,
        "memory_sha256_before": memory_before,
        "memory_sha256_after": memory_after,
    }
    if dispatch is not None:
        summary["dispatch"] = {
            "target_module": dispatch.get("target_module"),
            "recovery_type": dispatch.get("recovery_type"),
            "request_path": dispatch.get("request_path"),
        }
    return summary


def print_stage_summary(
    *,
    task_id: str,
    subgoal_id: str,
    failure_context: dict,
    retrieval_query: dict,
    retrieved: list[dict],
    diagnosis: dict,
    selection_result: dict,
    recovery: dict,
    dispatch: dict | None = None,
) -> None:
    subgoal = failure_context.get("subgoal") or {}
    motion = failure_context.get("motion_plan") or {}
    target = retrieval_query.get("target") or {}
    cause = (diagnosis.get("failure_cause") or {}).get("code")
    action = recovery.get("action") or {}
    routing = recovery.get("routing") or {}

    print(f"[M6 E2E] Task: {task_id}")
    print(f"[M6 E2E] Subgoal: {subgoal_id}")
    print()
    print("[1/8] Failure Context")
    print(f"  action_type: {subgoal.get('action_type')}")
    print(f"  selected_object: {subgoal.get('selected_object_id')}")
    print(f"  planning_status: {motion.get('planning_status')}")
    print()
    print("[2/8] Context Retrieval")
    print(f"  candidates: {len(retrieved)}")
    if retrieved:
        ids = [(item.get("experience") or {}).get("experience_id") for item in retrieved]
        print(f"  experience_ids: {ids}")
    print(f"  query.target: {target}")
    print()
    print("[3/8] Diagnosis")
    print(f"  failure_type: {diagnosis.get('failure_type')}")
    print(f"  cause: {cause}")
    print(f"  affected_module: {diagnosis.get('affected_module')}")
    print(f"  confidence: {diagnosis.get('confidence')}")
    print()
    print("[4/8] Diagnosis-aware Selection")
    print(f"  selected: {selection_result.get('selection_count', 0)}")
    print(
        "  experience_ids: "
        f"{list(selection_result.get('selected_experience_ids') or [])}"
    )
    print()
    print("[5/8] Decision Mode")
    print(f"  {recovery.get('decision_mode')}")
    print()
    print("[6/8] Recovery")
    print(f"  category: {recovery.get('recovery_category')}")
    print(f"  recovery_type: {action.get('recovery_type')}")
    print(f"  target_module: {action.get('target_module')}")
    print()
    print("[7/8] Routing")
    print(f"  restart_from: {routing.get('restart_from')}")
    print(f"  rerun_modules: {routing.get('rerun_modules')}")
    print(f"  invalidate: {routing.get('invalidate')}")
    print()
    print("[8/8] Recovery Dispatch")
    if dispatch is None:
        print("  (skipped)")
    else:
        print(f"  target_module: {dispatch.get('target_module')}")
        print(f"  recovery_type: {dispatch.get('recovery_type')}")
        print(f"  request_path: {dispatch.get('request_path')}")
    print()
    print("STOP: target module execution is disabled.")


def run_m6_e2e(
    *,
    task_id: str,
    subgoal_id: str | None = None,
    output_root: str | Path | None = None,
    project_root: str | Path | None = None,
    memory_path: str | Path | None = None,
    diagnoser_backend: str | None = None,
    recovery_backend: str | None = None,
    diagnosis_model: str | None = None,
    recovery_model: str | None = None,
    top_k: int = 3,
    save: bool = True,
    print_summary: bool = True,
) -> dict[str, Any]:
    """Run M6 through Recovery Request dispatch; never re-executes modules or updates memory."""
    output_dir = resolve_task_output_dir(
        task_id,
        output_root=output_root,
        project_root=project_root,
    )
    resolved_memory = Path(memory_path) if memory_path is not None else DEFAULT_MEMORY_PATH
    if not resolved_memory.is_file():
        raise M6E2EError(f"memory.json not found: {resolved_memory}")

    if not (output_dir / "m2.json").is_file():
        raise M6E2EError(f"required artifact missing: {output_dir / 'm2.json'}")
    if not (output_dir / "m1.json").is_file():
        # M1 is optional for adapter mapping, but E2E smoke expects scene evidence.
        # Keep soft: only require m2 as adapter does; surface missing m1 in logs later.
        pass

    if not has_m5_failure_artifact(output_dir):
        raise M6E2EError("No failure artifact available for M6 failure diagnosis")

    resolved_subgoal = resolve_failed_subgoal_id(output_dir, subgoal_id)

    diagnoser_name = get_diagnoser_backend(diagnoser_backend)
    recovery_name = get_recovery_router_backend(recovery_backend)
    if diagnoser_name == "openai":
        _require_openai_api_key("diagnoser")
    if recovery_name == "openai":
        _require_openai_api_key("recovery")

    memory_before = sha256_file(resolved_memory)

    try:
        artifact_adapter = FailureContextArtifactAdapter(output_dir)
        failure_context = artifact_adapter.build_failure_context(
            failed_subgoal_id=resolved_subgoal
        )
    except ArtifactAdapterError as exc:
        raise M6E2EError(str(exc)) from exc

    retrieval_query = build_retrieval_query(failure_context)

    memory_adapter = MemoryAdapter(memory_path=resolved_memory)
    retrieved = memory_adapter.retrieve_experiences(failure_context, top_k=top_k)
    if not isinstance(retrieved, list):
        retrieved = []

    diagnosis_evidence = prepare_diagnosis_evidence(retrieved)
    recovery_evidence = prepare_recovery_evidence(retrieved)

    diagnoser = create_failure_diagnoser(
        diagnoser_name,
        model=diagnosis_model,
    )
    recovery_router = create_recovery_router(
        recovery_name,
        model=recovery_model,
    )
    selector = DiagnosisAwareExperienceSelector()

    diagnosis = empty_diagnosis()
    diagnosis["memory_context"]["retrieved_experiences"] = retrieved
    diagnosis["memory_context"]["diagnosis_evidence"] = diagnosis_evidence

    diagnosis_output = diagnoser.diagnose(failure_context, diagnosis_evidence)
    apply_diagnosis_output(diagnosis, diagnosis_output)

    selection_result = selector.select(diagnosis, retrieved, recovery_evidence)
    selected_recovery_evidence = selection_result.get("selected_recovery_evidence") or []

    recovery = empty_recovery()
    recovery["decision_mode"] = (
        "EXPERIENCE_GUIDED"
        if selection_result["selection_count"] > 0
        else "DIAGNOSIS_GUIDED"
    )
    recovery["guidance"]["experience_ids"] = list(selection_result["selected_experience_ids"])
    recovery["guidance"]["recovery_evidence"] = selected_recovery_evidence
    recovery["guidance"]["selection"] = {
        "selected_experience_ids": list(selection_result["selected_experience_ids"]),
        "selection_count": selection_result["selection_count"],
        "selection_audit": list(selection_result["selection_audit"]),
    }

    recovery_output = recovery_router.route(
        failure_context,
        diagnosis,
        recovery["decision_mode"],
        selected_recovery_evidence,
    )
    apply_recovery_output(recovery, recovery_output)

    if "action_type" in (recovery.get("action") or {}):
        raise M6E2EError(
            "recovery.action.action_type is forbidden; expected recovery_type only"
        )

    try:
        dispatch = dispatch_recovery(recovery, output_dir=output_dir, save=save)
    except RecoveryDispatchError as exc:
        raise M6E2EError(str(exc)) from exc

    memory_after = sha256_file(resolved_memory)
    if memory_before != memory_after:
        raise M6E2EError(
            "output/memory.json changed during M6 E2E; Runtime Experience updates are forbidden"
        )

    m6_dir = output_dir / "m6"
    summary = build_e2e_summary(
        task_id=task_id.strip(),
        subgoal_id=resolved_subgoal,
        retrieval_query=retrieval_query,
        retrieved=retrieved,
        diagnosis=diagnosis,
        selection_result=selection_result,
        recovery=recovery,
        memory_before=memory_before,
        memory_after=memory_after,
        diagnoser_backend=diagnoser_name,
        recovery_backend=recovery_name,
        dispatch=dispatch,
    )

    if save:
        _write_json(m6_dir / "failure_context.json", failure_context)
        _write_json(m6_dir / "retrieval_query.json", retrieval_query)
        _write_json(m6_dir / "retrieved_experiences.json", _compact_retrieved(retrieved))
        _write_json(m6_dir / "diagnosis.json", diagnosis)
        _write_json(
            m6_dir / "diagnosis_aware_selection.json",
            {
                "selected_experience_ids": list(
                    selection_result.get("selected_experience_ids") or []
                ),
                "selection_count": selection_result.get("selection_count", 0),
                "selection_audit": list(selection_result.get("selection_audit") or []),
                "decision_mode": recovery.get("decision_mode"),
            },
        )
        _write_json(m6_dir / "recovery.json", recovery)
        _write_json(m6_dir / "e2e_summary.json", summary)

    if print_summary:
        print_stage_summary(
            task_id=task_id.strip(),
            subgoal_id=resolved_subgoal,
            failure_context=failure_context,
            retrieval_query=retrieval_query,
            retrieved=retrieved,
            diagnosis=diagnosis,
            selection_result=selection_result,
            recovery=recovery,
            dispatch=dispatch,
        )

    return {
        "task_id": task_id.strip(),
        "subgoal_id": resolved_subgoal,
        "output_dir": str(output_dir),
        "m6_dir": str(m6_dir),
        "failure_context": failure_context,
        "retrieval_query": retrieval_query,
        "retrieved_experiences": retrieved,
        "diagnosis": diagnosis,
        "selection": selection_result,
        "recovery": recovery,
        "dispatch": dispatch,
        "summary": summary,
        "selective_reexecution_performed": False,
        "memory_updated": False,
    }
