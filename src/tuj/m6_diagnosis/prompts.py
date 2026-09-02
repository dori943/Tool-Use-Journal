"""Prompt construction for OpenAI VLM failure diagnosis."""

from __future__ import annotations

import json
from typing import Any

from .diagnosis import DIAGNOSIS_EVIDENCE_ALLOWED_KEYS
from .taxonomy import (
    FAILURE_MODULES,
    FAILURE_VOCABULARY,
    RECOVERY_ACTION_MODULES,
    RECOVERY_CATEGORY_MODULES,
    RECOVERY_ROUTING_PROFILES,
    RECOVERY_VOCABULARY,
    VALID_ROUTING_MODULES,
)

RECOVERY_EVIDENCE_ALLOWED_KEYS = frozenset(
    {
        "experience_id",
        "retrieval",
        "past_diagnosis",
        "past_recovery",
        "outcome",
        "source",
    }
)


def build_failure_diagnosis_instructions() -> str:
    return (
        "You are the failure diagnosis component of a robotic manipulation system.\n"
        "Analyze why the current subgoal failed and return one canonical diagnosis.\n"
        "\n"
        "Important rules:\n"
        "1. Prioritize actual evidence from the current failure context.\n"
        "2. Past experiences are supplementary priors only, never ground truth.\n"
        "3. Do not copy a past diagnosis as the current diagnosis.\n"
        "4. Do not decide recovery actions or routing.\n"
        "5. Use only labels from the provided fixed failure vocabulary.\n"
        "6. Select one primary failure cause.\n"
        "7. Put concrete observations or states supporting the diagnosis in evidence.\n"
        "8. Even with incomplete information, choose the best-supported canonical diagnosis.\n"
        "9. Return confidence as a numeric value between 0 and 1.\n"
        "10. failure_type, failure_cause.code, and affected_module must all be consistent "
        "with the fixed vocabulary mapping."
    )


def _failure_context_for_prompt(failure_context: dict) -> dict[str, Any]:
    observation = failure_context.get("observation") or {}
    return {
        "failure_id": failure_context.get("failure_id"),
        "task": failure_context.get("task"),
        "subgoal": failure_context.get("subgoal"),
        "verification": failure_context.get("verification"),
        "scene": failure_context.get("scene"),
        "grounding": failure_context.get("grounding"),
        "task_plan": failure_context.get("task_plan"),
        "motion_plan": failure_context.get("motion_plan"),
        "execution": failure_context.get("execution"),
        "history": failure_context.get("history"),
        "observation": {
            "before_image": observation.get("before_image"),
            "after_image": observation.get("after_image"),
            "before_scene": observation.get("before_scene"),
            "after_scene": observation.get("after_scene"),
        },
    }


def _sanitize_diagnosis_evidence(diagnosis_evidence: list[dict]) -> list[dict]:
    sanitized: list[dict] = []
    for item in diagnosis_evidence:
        sanitized.append(
            {key: item.get(key) for key in DIAGNOSIS_EVIDENCE_ALLOWED_KEYS if key in item}
        )
    return sanitized


def build_failure_diagnosis_payload(
    failure_context: dict,
    diagnosis_evidence: list[dict],
) -> dict[str, Any]:
    return {
        "current_failure_context": _failure_context_for_prompt(failure_context),
        "diagnosis_side_past_experience_evidence": _sanitize_diagnosis_evidence(
            diagnosis_evidence
        ),
        "fixed_failure_vocabulary": {
            "failure_types_and_causes": {
                failure_type: list(causes)
                for failure_type, causes in FAILURE_VOCABULARY.items()
            },
            "failure_type_to_affected_module": dict(FAILURE_MODULES),
        },
    }


def build_failure_diagnosis_text(failure_context: dict, diagnosis_evidence: list[dict]) -> str:
    payload = build_failure_diagnosis_payload(failure_context, diagnosis_evidence)
    return (
        "Diagnose the current subgoal failure using the JSON payload below. "
        "Return structured output only.\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def build_recovery_router_instructions() -> str:
    return (
        "You are the recovery routing component of a robotic manipulation system.\n"
        "Generate one canonical recovery decision and routing plan for the current failure.\n"
        "\n"
        "Important rules:\n"
        "1. Use the current diagnosis as the primary evidence for recovery selection.\n"
        "2. Inspect raw evidence from the current failure context.\n"
        "3. In EXPERIENCE_GUIDED mode, past recovery experiences are supplementary only.\n"
        "4. Do not treat offline NOT_EXECUTED recoveries as verified successes.\n"
        "5. Runtime PASS recoveries may be positive evidence in similar contexts.\n"
        "6. Runtime FAIL recoveries may be negative evidence against repeating the same recovery.\n"
        "7. Use only labels from the provided fixed recovery vocabulary.\n"
        "8. Keep recovery target/routing logically consistent with the current diagnosis.\n"
        "9. Rerun only the minimum necessary modules; avoid unnecessary full pipeline restarts.\n"
        "10. Do not execute recovery; only generate recovery decision and routing.\n"
        "11. Do not generate outcome fields; recovery has not been executed yet.\n"
        "12. Select one primary recovery action.\n"
        "13. Do not copy a past recovery automatically; decide from current evidence first.\n"
        "14. decision_mode and guidance are already fixed by the pipeline; do not change them.\n"
        "15. Return only recovery_category, action, and routing."
    )


def _diagnosis_for_prompt(diagnosis: dict) -> dict[str, Any]:
    failure_cause = diagnosis.get("failure_cause") or {}
    return {
        "failure_type": diagnosis.get("failure_type"),
        "failure_cause": {
            "code": failure_cause.get("code"),
            "description": failure_cause.get("description"),
        },
        "affected_module": diagnosis.get("affected_module"),
        "evidence": list(diagnosis.get("evidence") or []),
        "confidence": diagnosis.get("confidence"),
    }


def _sanitize_recovery_evidence(recovery_evidence: list[dict]) -> list[dict]:
    sanitized: list[dict] = []
    for item in recovery_evidence:
        sanitized.append(
            {key: item.get(key) for key in RECOVERY_EVIDENCE_ALLOWED_KEYS if key in item}
        )
    return sanitized


def _routing_constraints_for_prompt() -> dict[str, Any]:
    return {
        "recovery_action_to_target_module": dict(RECOVERY_ACTION_MODULES),
        "recovery_category_to_default_module": dict(RECOVERY_CATEGORY_MODULES),
        "special_routing_profiles": {
            action: {
                "restart_from": profile["restart_from"],
                "rerun_modules": list(profile["rerun_modules"]),
                "invalidate": list(profile.get("invalidate") or []),
            }
            for action, profile in RECOVERY_ROUTING_PROFILES.items()
        },
        "valid_routing_modules": sorted(VALID_ROUTING_MODULES),
    }


def build_recovery_router_payload(
    failure_context: dict,
    diagnosis: dict,
    decision_mode: str,
    recovery_evidence: list[dict],
) -> dict[str, Any]:
    return {
        "current_failure_context": _failure_context_for_prompt(failure_context),
        "current_diagnosis": _diagnosis_for_prompt(diagnosis),
        "decision_mode": decision_mode,
        "filtered_recovery_evidence": _sanitize_recovery_evidence(recovery_evidence),
        "fixed_recovery_vocabulary": {
            "recovery_categories_and_actions": {
                category: list(actions)
                for category, actions in RECOVERY_VOCABULARY.items()
            }
        },
        "canonical_routing_constraints": _routing_constraints_for_prompt(),
        "deterministic_pipeline_state": {
            "decision_mode_is_fixed": True,
            "guidance_is_fixed": True,
            "do_not_generate_outcome": True,
        },
    }


def build_recovery_router_text(
    failure_context: dict,
    diagnosis: dict,
    decision_mode: str,
    recovery_evidence: list[dict],
) -> str:
    payload = build_recovery_router_payload(
        failure_context,
        diagnosis,
        decision_mode,
        recovery_evidence,
    )
    return (
        "Generate a canonical recovery decision using the JSON payload below. "
        "Return structured output only.\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )
