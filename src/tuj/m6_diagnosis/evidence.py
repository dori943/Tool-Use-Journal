"""Prepare failure-side and recovery-side evidence from retrieval results."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _retrieval_meta(item: dict) -> dict:
    return {
        "context_similarity": item.get("context_similarity"),
        "comparison_coverage": item.get("comparison_coverage"),
        "compared_field_count": item.get("compared_field_count"),
        "possible_field_count": item.get("possible_field_count"),
    }


def prepare_diagnosis_evidence(retrieved_results: list[dict]) -> list[dict]:
    """Build diagnosis-only views without recovery leakage."""
    evidence: list[dict] = []
    for item in retrieved_results:
        experience = item.get("experience") or {}
        context_signature = experience.get("context_signature") or {}
        diagnosis_summary = experience.get("diagnosis_summary") or {}
        failure_cause = diagnosis_summary.get("failure_cause") or {}
        metadata = experience.get("metadata") or {}

        evidence.append(
            {
                "experience_id": experience.get("experience_id"),
                "retrieval": _retrieval_meta(item),
                "past_context": {
                    "subgoal_description": context_signature.get("subgoal_description"),
                    "action_type": context_signature.get("action_type"),
                    "target": dict(context_signature.get("target") or {}),
                    "violated_predicates": list(context_signature.get("violated_predicates") or []),
                    "selected_ee": context_signature.get("selected_ee"),
                    "selected_tool": context_signature.get("selected_tool"),
                    "execution_signature": dict(
                        context_signature.get("execution_signature") or {}
                    ),
                },
                "past_diagnosis": {
                    "failure_type": diagnosis_summary.get("failure_type"),
                    "failure_cause": {
                        "code": failure_cause.get("code"),
                        "description": failure_cause.get("description"),
                    },
                    "affected_module": diagnosis_summary.get("affected_module"),
                    "confidence": diagnosis_summary.get("confidence"),
                },
                "source": metadata.get("source"),
            }
        )

    logger.debug(
        "prepared diagnosis evidence ids=%s",
        [entry.get("experience_id") for entry in evidence],
    )
    return evidence


def prepare_recovery_evidence(retrieved_results: list[dict]) -> list[dict]:
    """Build recovery-router views preserving source and outcome metadata."""
    evidence: list[dict] = []
    for item in retrieved_results:
        experience = item.get("experience") or {}
        diagnosis_summary = experience.get("diagnosis_summary") or {}
        failure_cause = diagnosis_summary.get("failure_cause") or {}
        recovery_summary = experience.get("recovery_summary") or {}
        metadata = experience.get("metadata") or {}
        outcome = recovery_summary.get("outcome") or {}

        evidence.append(
            {
                "experience_id": experience.get("experience_id"),
                "retrieval": _retrieval_meta(item),
                "past_diagnosis": {
                    "failure_type": diagnosis_summary.get("failure_type"),
                    "failure_cause": {"code": failure_cause.get("code")},
                    "affected_module": diagnosis_summary.get("affected_module"),
                },
                "past_recovery": {
                    "recovery_category": recovery_summary.get("recovery_category"),
                    "action": recovery_summary.get("action"),
                    "changes": list(recovery_summary.get("changes") or []),
                    "routing": dict(recovery_summary.get("routing") or {}),
                },
                "outcome": {
                    "status": outcome.get("status"),
                    "verification_result": outcome.get("verification_result"),
                },
                "source": metadata.get("source"),
            }
        )

    logger.debug(
        "prepared recovery evidence ids=%s",
        [entry.get("experience_id") for entry in evidence],
    )
    return evidence
