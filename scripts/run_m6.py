"""Run standalone M6 retrieval demos for similar and dissimilar failure contexts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tuj.m6.evidence import prepare_recovery_evidence
from tuj.m6 import DiagnoseRouter, create_failure_diagnoser, get_diagnoser_backend
from tuj.m6.recovery_config import create_recovery_router, get_recovery_router_backend
from tuj.m6.context_similarity import evaluate_all_candidates
from tuj.m6.retrieval_config import DEFAULT_SIMILARITY_THRESHOLD, RetrievalConfig
from tuj.m6.retrieval_query import build_retrieval_query
from tuj.m6.memory_adapter import DEFAULT_MEMORY_PATH, MemoryAdapter
from tuj.m6.prompts import build_failure_diagnosis_payload


M6_DEBUG_CONTEXT_ENV = "M6_DEBUG_CONTEXT"
_IMAGE_DATA_OMITTED = "<image data omitted>"
_SENSITIVE_KEY_FRAGMENTS = ("api_key", "authorization", "secret", "token", "password")


# Test A uses semantically similar description + explicit object_class.
MOCK_SIMILAR_PIPELINE_STATE = {
    "failure_id": "failure-test-a-001",
    "task": {"task_id": "task-spatula-acquire", "instruction": "Acquire the wooden spatula."},
    "subgoal": {
        "subgoal_id": "sg-spatula-acquire",
        "description": "Approach the wooden spatula and get ready to grasp it.",
        "action_type": "acquire",
        "target_object_ids": ["spatula_03"],
        "selected_object_id": "spatula_03",
        "selected_object_class": "spatula",
        "postconditions": ["holding(spatula)"],
    },
    "verification": {
        "result": "FAIL",
        "expected_state": ["aligned(spatula)"],
        "observed_state": ["misaligned(spatula)"],
        "violated_predicates": [],
    },
    "task_plan": {"selected_ee": "2F", "selected_tool": None},
    "motion_plan": {"planning_status": None},
    "execution": {"controller_status": None},
}

MOCK_DISSIMILAR_PIPELINE_STATE = {
    "failure_id": "failure-test-b-001",
    "task": {"task_id": "task-card-extract", "instruction": "Extract the card from the slot."},
    "subgoal": {
        "subgoal_id": "sg-card-extract",
        "description": "Extract the loyalty card from the dispenser slot using the tool.",
        "action_type": "tool_act:extract",
        "target_object_ids": ["card"],
        "selected_object_id": "card",
        "selected_object_class": "card",
        "postconditions": ["holding(card)"],
    },
    "verification": {
        "result": "FAIL",
        "expected_state": ["holding(card)"],
        "observed_state": ["not_holding(card)"],
        "violated_predicates": ["holding(card)"],
    },
    "motion_plan": {
        "planning_status": "SUCCESS",
        "planning_error": None,
    },
    "execution": {
        "executed_actions": ["approach_target", "close_gripper"],
        "controller_status": "SUCCESS",
        "gripper": {
            "command": "close",
            "position": 0.0,
            "contact_detected": False,
            "force": 0.0,
        },
        "timeout": False,
        "error": None,
    },
}


def _format_context_summary(failure_context: dict, retrieval_query: dict) -> str:
    subgoal = failure_context.get("subgoal") or {}
    task_plan = failure_context.get("task_plan") or {}
    verification = failure_context.get("verification") or {}
    motion_plan = failure_context.get("motion_plan") or {}
    execution = failure_context.get("execution") or {}
    target = retrieval_query.get("target") or {}
    lines = [
        f"description : {subgoal.get('description')}",
        f"action_type : {subgoal.get('action_type')}",
        f"target.object_id   : {target.get('object_id')}",
        f"target.object_class: {target.get('object_class')}",
        f"violated    : {verification.get('violated_predicates')}",
        f"selected_ee : {task_plan.get('selected_ee')}",
        f"selected_tool : {task_plan.get('selected_tool')}",
        f"plan_status : {motion_plan.get('planning_status')}",
        f"ctrl_status : {execution.get('controller_status')}",
    ]
    return "\n".join(lines)


def _is_debug_context_enabled() -> bool:
    return os.environ.get(M6_DEBUG_CONTEXT_ENV, "0").strip() == "1"


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in _SENSITIVE_KEY_FRAGMENTS)


def _sanitize_debug_value(value):
    if isinstance(value, dict):
        return {
            key: (
                _IMAGE_DATA_OMITTED
                if _is_sensitive_key(key)
                else _sanitize_debug_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_debug_value(item) for item in value]
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value:
            return _IMAGE_DATA_OMITTED
    return value


def _print_json_debug_block(title: str, payload) -> None:
    sanitized = _sanitize_debug_value(payload)
    print(f"===== {title} =====")
    print(json.dumps(sanitized, indent=2, ensure_ascii=False))
    print()


def _print_debug_context_views(failure_context: dict, diagnosis_evidence: list[dict]) -> None:
    _print_json_debug_block("FULL CURRENT FAILURE CONTEXT", failure_context)
    diagnosis_payload = build_failure_diagnosis_payload(failure_context, diagnosis_evidence)
    _print_json_debug_block("OPENAI DIAGNOSIS PAYLOAD", diagnosis_payload)


def _load_memory_experiences() -> list[dict]:
    with DEFAULT_MEMORY_PATH.open(encoding="utf-8") as handle:
        memory = json.load(handle)
    return memory.get("failure_recovery_experience", {}).get("experiences") or []


def _print_candidate_audit(
    failure_context: dict,
    *,
    config: RetrievalConfig,
    focus_ids: list[str] | None = None,
) -> None:
    candidates = evaluate_all_candidates(
        failure_context,
        _load_memory_experiences(),
        config=config,
    )
    if focus_ids is not None:
        focus = set(focus_ids)
        candidates = [
            item
            for item in candidates
            if item["experience"].get("experience_id") in focus
        ]

    print("Candidate Audit:")
    if not candidates:
        print("None")
        return

    for item in candidates:
        experience_id = item["experience"].get("experience_id")
        validity = item["candidate_validity"]
        print(f"- {experience_id}")
        print(f"  context_similarity: {item['context_similarity']:.4f}")
        print(f"  comparison_coverage: {item['comparison_coverage']:.4f}")
        print(f"  passes_threshold: {item['passes_threshold']}")
        print(f"  candidate_valid: {item['candidate_valid']}")
        print(f"  candidate_validity: {validity}")
        print(f"  field_scores: {item['similarity_breakdown'].get('field_scores')}")


def _print_retrieval_result(
    test_name: str,
    pipeline_state: dict,
    *,
    config: RetrievalConfig,
    candidate_focus_ids: list[str] | None = None,
) -> None:
    adapter = MemoryAdapter(config=config)
    failure_diagnoser = create_failure_diagnoser()
    recovery_router = create_recovery_router()
    router = DiagnoseRouter(
        memory_adapter=adapter,
        failure_diagnoser=failure_diagnoser,
        recovery_router=recovery_router,
    )
    result = router.run(pipeline_state, top_k=config.top_k)
    failure_context = result["failure_context"]
    retrieval_query = build_retrieval_query(failure_context)
    retrieved = result["diagnosis"]["memory_context"]["retrieved_experiences"]
    diagnosis_evidence = result["diagnosis"]["memory_context"]["diagnosis_evidence"]
    all_recovery_evidence = prepare_recovery_evidence(retrieved)
    filtered_recovery_evidence = result["recovery"]["guidance"]["recovery_evidence"]
    diagnosis = result["diagnosis"]

    print("=" * 50)
    print(test_name)
    print("=" * 50)
    print()
    if _is_debug_context_enabled():
        _print_debug_context_views(failure_context, diagnosis_evidence)
    print("Current Context")
    print(_format_context_summary(failure_context, retrieval_query))
    print()
    print(f"Threshold: {config.similarity_threshold}")
    print(f"Memory Path: {DEFAULT_MEMORY_PATH}")
    print(f"retrieved_count: {len(retrieved)}")
    print()
    
    if retrieved:
        print("Retrieved:")
        for index, item in enumerate(retrieved, start=1):
            experience = item["experience"]
            breakdown = item["similarity_breakdown"]
            print(f"{index}. {experience.get('experience_id')}")
            print(f"   context_similarity: {item['context_similarity']:.4f}")
            print(f"   comparison_coverage: {item['comparison_coverage']:.4f}")
            print(f"   compared_field_count: {item['compared_field_count']}")
            print(f"   candidate_valid: {item['candidate_valid']}")
            print(f"   candidate_validity: {item['candidate_validity']}")
            print(f"   field_scores: {breakdown.get('field_scores')}")
            print(f"   matched: {breakdown.get('matched_fields')}")
            print(f"   compared: {breakdown.get('compared_fields')}")
            print(f"   ignored: {breakdown.get('ignored_fields')}")
        print()
        print("final Top-K experience_id:")
        print([item["experience"].get("experience_id") for item in retrieved])
    else:
        print("Retrieved:")
        print("None")

    print()
    _print_candidate_audit(
        failure_context,
        config=config,
        focus_ids=candidate_focus_ids,
    )
    print()
    print("diagnosis_evidence experience_id:")
    print([item.get("experience_id") for item in diagnosis_evidence] or "None")
    print("recovery_evidence experience_id:")
    print([item.get("experience_id") for item in all_recovery_evidence] or "None")
    print()
    print("Diagnosis:")
    print(f"failure_type      : {diagnosis.get('failure_type')}")
    print(f"failure_cause.code: {diagnosis.get('failure_cause', {}).get('code')}")
    print(f"affected_module   : {diagnosis.get('affected_module')}")
    print(f"evidence          : {diagnosis.get('evidence')}")
    print(f"confidence        : {diagnosis.get('confidence')}")
    print()
    selection = result["recovery"]["guidance"]["selection"]
    print("Diagnosis-aware Selection:")
    print(f"selected_experience_ids: {selection.get('selected_experience_ids') or []}")
    print(f"selection_count        : {selection.get('selection_count')}")
    print("Selection Audit:")
    audit = selection.get("selection_audit") or []
    if not audit:
        print("None")
    else:
        for entry in audit:
            print(f"- {entry.get('experience_id')}")
            print(f"  failure_type_match     : {entry.get('failure_type_match')}")
            print(f"  failure_cause_match    : {entry.get('failure_cause_match')}")
            print(f"  affected_module_match  : {entry.get('affected_module_match')}")
            print(f"  matched_field_count    : {entry.get('matched_field_count')}")
            print(f"  candidate_relevant     : {entry.get('candidate_relevant')}")
            print(f"  relevance_reason       : {entry.get('relevance_reason')}")
    print()
    print("Filtered Recovery Evidence IDs:")
    print([item.get("experience_id") for item in filtered_recovery_evidence] or "None")
    print()
    print("Decision Mode:")
    print(result["recovery"]["decision_mode"])
    print()
    recovery = result["recovery"]
    action = recovery.get("action") or {}
    routing = recovery.get("routing") or {}
    guidance = recovery.get("guidance") or {}
    print("Recovery:")
    print(f"decision_mode: {recovery.get('decision_mode')}")
    print(f"guidance.experience_ids: {guidance.get('experience_ids') or []}")
    print(f"recovery_category: {recovery.get('recovery_category')}")
    print(f"action.action_type: {action.get('action_type')}")
    print(f"action.target_module: {action.get('target_module')}")
    print(f"routing.restart_from: {routing.get('restart_from')}")
    print(f"routing.rerun_modules: {routing.get('rerun_modules') or []}")
    print(f"routing.invalidate: {routing.get('invalidate') or []}")
    print()


def main() -> None:
    config = RetrievalConfig.create_default(
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        top_k=3,
    )
    print(f"Diagnoser backend: {get_diagnoser_backend()}")
    print(f"Recovery Router backend: {get_recovery_router_backend()}")
    print()
    _print_retrieval_result(
        "TEST A - Similar Experience",
        MOCK_SIMILAR_PIPELINE_STATE,
        config=config,
        candidate_focus_ids=["VF-SPAT-007", "VF-SPAT-015", "VF-SPOON-007", "RLBF-DOOR-032-TR"],
    )
    _print_retrieval_result(
        "TEST B - No Similar Experience",
        MOCK_DISSIMILAR_PIPELINE_STATE,
        config=config,
    )


if __name__ == "__main__":
    main()
