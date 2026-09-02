"""Configuration for M0 failure-recovery experience retrieval."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_TOP_K = 3
POSSIBLE_FIELD_COUNT = 8


def build_default_retrieval_config(
    *,
    top_k: int = DEFAULT_TOP_K,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    possible_field_count: int = POSSIBLE_FIELD_COUNT,
) -> "RetrievalConfig":
    """Build a default retrieval config."""
    return RetrievalConfig(
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        possible_field_count=possible_field_count,
    )


@dataclass
class RetrievalConfig:
    top_k: int = DEFAULT_TOP_K
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    possible_field_count: int = POSSIBLE_FIELD_COUNT

    @classmethod
    def create_default(
        cls,
        *,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> "RetrievalConfig":
        return build_default_retrieval_config(
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
