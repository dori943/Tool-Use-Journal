"""M0 failure-recovery experience retrieval and runtime append for M6."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from tuj.m0_memory.memory_store import UnifiedMemoryStore

from .context_similarity import rank_experiences
from .recovery_router import normalize_recovery_action
from .retrieval_config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    RetrievalConfig,
    build_default_retrieval_config,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MEMORY_PATH = _PROJECT_ROOT / "output" / "memory.json"


class MemoryAppendError(ValueError):
    """Raised when a runtime experience cannot be appended safely."""


def _normalize_loaded_experience(experience: dict) -> dict:
    """Normalize legacy fields when reading memory (never invent missing facts)."""
    if not isinstance(experience, dict):
        return experience
    normalized = deepcopy(experience)
    context = normalized.get("context_signature")
    if isinstance(context, dict) and "detail_id" not in context:
        # Schema compatibility: older experiences omit detail_id.
        context["detail_id"] = None
    recovery = normalized.get("recovery_summary")
    if not isinstance(recovery, dict):
        return normalized
    action = recovery.get("action")
    if not isinstance(action, dict):
        return normalized
    recovery["action"] = normalize_recovery_action(action)
    return normalized


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

    def invalidate_cache(self) -> None:
        self._experiences = None

    def _load_experiences(self) -> list[dict]:
        if self._experiences is None:
            logger.debug("loading failure-recovery experiences from %s", self.memory_path)
            with self.memory_path.open(encoding="utf-8") as handle:
                memory = json.load(handle)
            raw = (
                memory.get("failure_recovery_experience", {}).get("experiences") or []
            )
            self._experiences = [_normalize_loaded_experience(item) for item in raw]
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

    def append_experience(self, experience: dict[str, Any]) -> dict[str, Any]:
        """Append one experience to the shared M0 experiences[] array."""
        stored = append_experience(experience, memory_path=self.memory_path)
        self.invalidate_cache()
        return stored


def append_experience(
    experience: dict[str, Any],
    *,
    memory_path: str | Path | None = None,
) -> dict[str, Any]:
    """Append ``experience`` to ``failure_recovery_experience.experiences``.

    Uses ``UnifiedMemoryStore`` so ``object_knowledge``, ``schema_version``, and
    offline seeds are preserved. Rejects duplicate ``experience_id`` values.
    """
    if not isinstance(experience, dict):
        raise MemoryAppendError("experience must be a dict")
    exp_id = experience.get("experience_id")
    if not isinstance(exp_id, str) or not exp_id.strip():
        raise MemoryAppendError("experience.experience_id is required")
    exp_id = exp_id.strip()

    path = Path(memory_path) if memory_path is not None else DEFAULT_MEMORY_PATH
    store = UnifiedMemoryStore(path)
    fre = store.document.setdefault("failure_recovery_experience", {"experiences": []})
    if not isinstance(fre, dict):
        raise MemoryAppendError("failure_recovery_experience must be an object")
    experiences = fre.setdefault("experiences", [])
    if not isinstance(experiences, list):
        raise MemoryAppendError("failure_recovery_experience.experiences must be a list")

    existing_ids = {
        item.get("experience_id")
        for item in experiences
        if isinstance(item, dict)
    }
    if exp_id in existing_ids:
        raise MemoryAppendError(
            f"experience_id already exists in memory; refusing to replace: {exp_id}"
        )

    stored = deepcopy(experience)
    stored["experience_id"] = exp_id
    experiences.append(stored)
    store.save()
    logger.debug("appended experience_id=%s to %s", exp_id, path)
    return deepcopy(stored)


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
