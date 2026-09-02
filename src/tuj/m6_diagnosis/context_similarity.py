"""Field-aware context similarity for M0 failure-recovery experience retrieval."""

from __future__ import annotations

import logging
from typing import Any

from .retrieval_config import POSSIBLE_FIELD_COUNT, RetrievalConfig
from .retrieval_query import build_retrieval_query

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.5

CONTEXT_SPECIFIC_FIELDS = frozenset(
    {
        "target.object_id",
        "target.object_class",
        "verification.violated_predicates",
        "task_plan.selected_tool",
        "task_plan.selected_ee",
        "motion_plan.planning_status",
        "execution.controller_status",
    }
)

TARGET_IDENTITY_FIELDS = frozenset(
    {
        "target.object_id",
        "target.object_class",
    }
)


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def _normalize_string(value: Any) -> str:
    return str(value).strip().lower()


def _normalize_predicate(value: Any) -> str:
    return _normalize_string(value)


def _score_exact_match(current_value: Any, past_value: Any) -> float | None:
    if not _is_present(current_value) or not _is_present(past_value):
        return None
    return 1.0 if _normalize_string(current_value) == _normalize_string(past_value) else 0.0


def _score_violated_predicates(current_values: list[Any], past_values: list[Any]) -> float | None:
    if not _is_present(current_values) or not _is_present(past_values):
        return None
    current_set = {_normalize_predicate(item) for item in current_values if _is_present(item)}
    past_set = {_normalize_predicate(item) for item in past_values if _is_present(item)}
    if not current_set or not past_set:
        return None
    intersection = current_set & past_set
    union = current_set | past_set
    return len(intersection) / len(union)


def evaluate_candidate_validity(
    field_scores: dict[str, float],
    compared_fields: list[str],
) -> tuple[bool, dict]:
    """Return whether a candidate has enough context-specific evidence."""
    for field_name in TARGET_IDENTITY_FIELDS:
        if field_name in field_scores and field_scores[field_name] == 0.0:
            return False, {
                "has_context_specific_evidence": False,
                "reason": f"{field_name}_mismatch",
            }

    has_context_specific_evidence = any(
        field_name in CONTEXT_SPECIFIC_FIELDS and field_scores[field_name] > 0.0
        for field_name in compared_fields
    )
    if not has_context_specific_evidence:
        return False, {
            "has_context_specific_evidence": False,
            "reason": "action_type_only_match",
        }

    return True, {
        "has_context_specific_evidence": True,
        "reason": "context_specific_evidence_present",
    }


def compute_context_similarity(
    retrieval_query: dict,
    context_signature: dict,
    *,
    possible_field_count: int = POSSIBLE_FIELD_COUNT,
) -> dict:
    """Compare a retrieval query against a stored context_signature."""
    query_target = retrieval_query.get("target") or {}
    past_target = context_signature.get("target") or {}
    query_exec = retrieval_query.get("execution_signature") or {}
    past_exec = context_signature.get("execution_signature") or {}

    comparisons: list[tuple[str, float | None]] = [
        (
            "subgoal.action_type",
            _score_exact_match(
                retrieval_query.get("action_type"),
                context_signature.get("action_type"),
            ),
        ),
        (
            "target.object_id",
            _score_exact_match(query_target.get("object_id"), past_target.get("object_id")),
        ),
        (
            "target.object_class",
            _score_exact_match(query_target.get("object_class"), past_target.get("object_class")),
        ),
        (
            "verification.violated_predicates",
            _score_violated_predicates(
                retrieval_query.get("violated_predicates"),
                context_signature.get("violated_predicates"),
            ),
        ),
        (
            "task_plan.selected_ee",
            _score_exact_match(
                retrieval_query.get("selected_ee"),
                context_signature.get("selected_ee"),
            ),
        ),
        (
            "task_plan.selected_tool",
            _score_exact_match(
                retrieval_query.get("selected_tool"),
                context_signature.get("selected_tool"),
            ),
        ),
        (
            "motion_plan.planning_status",
            _score_exact_match(
                query_exec.get("motion_planning_status"),
                past_exec.get("motion_planning_status"),
            ),
        ),
        (
            "execution.controller_status",
            _score_exact_match(
                query_exec.get("controller_status"),
                past_exec.get("controller_status"),
            ),
        ),
    ]

    compared_fields: list[str] = []
    matched_fields: list[str] = []
    ignored_fields: list[str] = []
    field_scores: dict[str, float] = {}
    score_values: list[float] = []

    for field_name, score in comparisons:
        if score is None:
            ignored_fields.append(field_name)
            continue
        compared_fields.append(field_name)
        field_scores[field_name] = score
        score_values.append(score)
        if score == 1.0:
            matched_fields.append(field_name)

    compared_field_count = len(compared_fields)
    if score_values:
        context_similarity = sum(score_values) / len(score_values)
    else:
        context_similarity = 0.0

    comparison_coverage = (
        compared_field_count / possible_field_count if possible_field_count else 0.0
    )
    candidate_valid, candidate_validity = evaluate_candidate_validity(field_scores, compared_fields)

    result = {
        "context_similarity": context_similarity,
        "compared_field_count": compared_field_count,
        "possible_field_count": possible_field_count,
        "comparison_coverage": comparison_coverage,
        "candidate_valid": candidate_valid,
        "candidate_validity": candidate_validity,
        "similarity_breakdown": {
            "matched_fields": matched_fields,
            "compared_fields": compared_fields,
            "ignored_fields": ignored_fields,
            "field_scores": field_scores,
        },
    }
    logger.debug(
        "context similarity computed: compared=%s matched=%s ignored=%s score=%.4f coverage=%.4f valid=%s validity=%s field_scores=%s",
        compared_fields,
        matched_fields,
        ignored_fields,
        context_similarity,
        comparison_coverage,
        candidate_valid,
        candidate_validity,
        field_scores,
    )
    return result


def _build_candidate_result(experience: dict, similarity: dict, *, passes_threshold: bool) -> dict:
    return {
        "experience": experience,
        "context_similarity": similarity["context_similarity"],
        "comparison_coverage": similarity["comparison_coverage"],
        "compared_field_count": similarity["compared_field_count"],
        "possible_field_count": similarity["possible_field_count"],
        "candidate_valid": similarity["candidate_valid"],
        "candidate_validity": similarity["candidate_validity"],
        "passes_threshold": passes_threshold,
        "similarity_breakdown": similarity["similarity_breakdown"],
    }


def evaluate_all_candidates(
    failure_context: dict,
    experiences: list[dict],
    *,
    config: RetrievalConfig | None = None,
    similarity_threshold: float | None = None,
) -> list[dict]:
    """Score every experience, including invalid or below-threshold candidates."""
    cfg = config or RetrievalConfig()
    effective_threshold = (
        cfg.similarity_threshold if similarity_threshold is None else similarity_threshold
    )
    retrieval_query = build_retrieval_query(failure_context)

    candidates: list[dict] = []
    for experience in experiences:
        context_signature = experience.get("context_signature") or {}
        similarity = compute_context_similarity(
            retrieval_query,
            context_signature,
            possible_field_count=cfg.possible_field_count,
        )
        compared_fields = similarity["similarity_breakdown"]["compared_fields"]
        if not compared_fields:
            continue

        passes_threshold = similarity["context_similarity"] >= effective_threshold
        candidates.append(
            _build_candidate_result(
                experience,
                similarity,
                passes_threshold=passes_threshold,
            )
        )
    return candidates


def rank_experiences(
    failure_context: dict,
    experiences: list[dict],
    *,
    config: RetrievalConfig | None = None,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> list[dict]:
    """Rank valid experiences by context similarity and return Top-K above threshold."""
    cfg = config or RetrievalConfig()
    effective_top_k = cfg.top_k if top_k is None else top_k
    effective_threshold = (
        cfg.similarity_threshold if similarity_threshold is None else similarity_threshold
    )

    retrieval_query = build_retrieval_query(failure_context)
    logger.debug("retrieval query=%s", retrieval_query)

    ranked: list[dict] = []
    for experience in experiences:
        experience_id = experience.get("experience_id")
        context_signature = experience.get("context_signature") or {}
        similarity = compute_context_similarity(
            retrieval_query,
            context_signature,
            possible_field_count=cfg.possible_field_count,
        )
        compared_fields = similarity["similarity_breakdown"]["compared_fields"]
        if not compared_fields:
            logger.debug("candidate %s skipped: no comparable fields", experience_id)
            continue

        context_similarity = similarity["context_similarity"]
        passes_threshold = context_similarity >= effective_threshold
        candidate_valid = similarity["candidate_valid"]
        logger.debug(
            "candidate %s similarity=%.4f coverage=%.4f threshold=%.4f pass=%s valid=%s validity=%s breakdown=%s",
            experience_id,
            context_similarity,
            similarity["comparison_coverage"],
            effective_threshold,
            passes_threshold,
            candidate_valid,
            similarity["candidate_validity"],
            similarity["similarity_breakdown"],
        )
        if not passes_threshold or not candidate_valid:
            continue

        ranked.append(_build_candidate_result(experience, similarity, passes_threshold=True))

    ranked.sort(
        key=lambda item: (
            item["context_similarity"],
            item["comparison_coverage"],
        ),
        reverse=True,
    )
    top_results = ranked[:effective_top_k]
    logger.debug(
        "retrieval top-k=%s threshold=%.4f results=%s",
        effective_top_k,
        effective_threshold,
        [item["experience"].get("experience_id") for item in top_results],
    )
    return top_results
