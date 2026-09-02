"""Read-only M0 failure-recovery experience retrieval for M6."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .context_similarity import rank_experiences
from .retrieval_config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    RetrievalConfig,
    build_default_retrieval_config,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MEMORY_PATH = _PROJECT_ROOT / "output" / "memory.json"


class MemoryAdapter:
    def __init__(
        self,
        memory_path: str | Path | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        config: RetrievalConfig | None = None,
    ):
        self.memory_path = Path(memory_path) if memory_path is not None else DEFAULT_MEMORY_PATH
        if config is None:
            config = build_default_retrieval_config(
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
        self.config = config
        self._experiences: list[dict] | None = None

    def _load_experiences(self) -> list[dict]:
        if self._experiences is None:
            logger.debug("loading failure-recovery experiences from %s", self.memory_path)
            with self.memory_path.open(encoding="utf-8") as handle:
                memory = json.load(handle)
            self._experiences = (
                memory.get("failure_recovery_experience", {}).get("experiences") or []
            )
        return self._experiences

    def retrieve_experiences(
        self,
        failure_context,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list:
        experiences = self._load_experiences()
        return rank_experiences(
            failure_context,
            experiences,
            config=self.config,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )


def retrieve_experiences(
    failure_context,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> list:
    return MemoryAdapter().retrieve_experiences(
        failure_context,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )
