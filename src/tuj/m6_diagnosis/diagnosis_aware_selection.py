"""Diagnosis-aware experience selection for M6 recovery guidance."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _current_diagnosis_fields(current_diagnosis: dict) -> dict[str, str | None]:
    failure_cause = current_diagnosis.get("failure_cause") or {}
    return {
        "failure_type": current_diagnosis.get("failure_type"),
        "failure_cause_code": failure_cause.get("code"),
        "affected_module": current_diagnosis.get("affected_module"),
    }


def _past_diagnosis_fields(retrieved_item: dict) -> dict[str, str | None]:
    experience = retrieved_item.get("experience") or {}
    diagnosis_summary = experience.get("diagnosis_summary") or {}
    failure_cause = diagnosis_summary.get("failure_cause") or {}
    return {
        "failure_type": diagnosis_summary.get("failure_type"),
        "failure_cause_code": failure_cause.get("code"),
        "affected_module": diagnosis_summary.get("affected_module"),
    }


def _audit_entry(
    experience_id: str | None,
    *,
    failure_type_match: bool,
    failure_cause_match: bool,
    affected_module_match: bool,
    matched_field_count: int,
    candidate_relevant: bool,
    relevance_reason: str,
) -> dict:
    return {
        "experience_id": experience_id,
        "failure_type_match": failure_type_match,
        "failure_cause_match": failure_cause_match,
        "affected_module_match": affected_module_match,
        "matched_field_count": matched_field_count,
        "candidate_relevant": candidate_relevant,
        "relevance_reason": relevance_reason,
    }


def filter_recovery_evidence_by_ids(
    recovery_evidence: list[dict],
    selected_experience_ids: list[str],
) -> list[dict]:
    selected_ids = set(selected_experience_ids)
    return [
        item
        for item in recovery_evidence
        if item.get("experience_id") in selected_ids
    ]


class DiagnosisAwareExperienceSelector:
    """Select recovery-relevant experiences using exact diagnosis consistency."""

    def select(
        self,
        current_diagnosis: dict,
        retrieved_experiences: list[dict],
        recovery_evidence: list[dict] | None = None,
    ) -> dict:
        current = _current_diagnosis_fields(current_diagnosis)
        if not all(_is_present(value) for value in current.values()):
            logger.debug("current diagnosis incomplete; no experience can be selected")
            audit = [
                _audit_entry(
                    (item.get("experience") or {}).get("experience_id"),
                    failure_type_match=False,
                    failure_cause_match=False,
                    affected_module_match=False,
                    matched_field_count=0,
                    candidate_relevant=False,
                    relevance_reason="insufficient_diagnosis_information",
                )
                for item in retrieved_experiences
            ]
            return {
                "selected_experiences": [],
                "selected_experience_ids": [],
                "selection_count": 0,
                "selection_audit": audit,
            }

        selected_experiences: list[dict] = []
        selection_audit: list[dict] = []

        for retrieved_item in retrieved_experiences:
            experience = retrieved_item.get("experience") or {}
            experience_id = experience.get("experience_id")
            past = _past_diagnosis_fields(retrieved_item)

            if not all(_is_present(value) for value in past.values()):
                selection_audit.append(
                    _audit_entry(
                        experience_id,
                        failure_type_match=False,
                        failure_cause_match=False,
                        affected_module_match=False,
                        matched_field_count=0,
                        candidate_relevant=False,
                        relevance_reason="insufficient_diagnosis_information",
                    )
                )
                continue

            failure_cause_match = past["failure_cause_code"] == current["failure_cause_code"]
            failure_type_match = past["failure_type"] == current["failure_type"]
            affected_module_match = past["affected_module"] == current["affected_module"]
            matched_field_count = sum(
                [failure_cause_match, failure_type_match, affected_module_match]
            )

            if matched_field_count == 3:
                candidate_relevant = True
                relevance_reason = "exact_diagnosis_match"
                selected_experiences.append(retrieved_item)
            elif matched_field_count == 0:
                candidate_relevant = False
                relevance_reason = "diagnosis_mismatch"
            else:
                candidate_relevant = False
                relevance_reason = "partial_diagnosis_match"

            selection_audit.append(
                _audit_entry(
                    experience_id,
                    failure_type_match=failure_type_match,
                    failure_cause_match=failure_cause_match,
                    affected_module_match=affected_module_match,
                    matched_field_count=matched_field_count,
                    candidate_relevant=candidate_relevant,
                    relevance_reason=relevance_reason,
                )
            )

        selected_experience_ids = [
            (item.get("experience") or {}).get("experience_id")
            for item in selected_experiences
            if (item.get("experience") or {}).get("experience_id") is not None
        ]

        result = {
            "selected_experiences": selected_experiences,
            "selected_experience_ids": selected_experience_ids,
            "selection_count": len(selected_experiences),
            "selection_audit": selection_audit,
        }
        if recovery_evidence is not None:
            result["selected_recovery_evidence"] = filter_recovery_evidence_by_ids(
                recovery_evidence,
                selected_experience_ids,
            )

        logger.debug(
            "diagnosis-aware selection selected_ids=%s audit_count=%s",
            selected_experience_ids,
            len(selection_audit),
        )
        return result
