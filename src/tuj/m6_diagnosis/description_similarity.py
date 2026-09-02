"""Optional semantic backend for subgoal description similarity."""

from __future__ import annotations

import importlib.util
import logging
from typing import Protocol

logger = logging.getLogger(__name__)

_DEFAULT_BACKEND: "DescriptionSimilarityBackend | None" = None


def _normalize_string(value: str) -> str:
    return value.strip().lower()


class DescriptionSimilarityBackend(Protocol):
    @property
    def name(self) -> str:
        """Human-readable backend identifier."""

    def score(self, text_a: str, text_b: str) -> float:
        """Return a similarity score in [0.0, 1.0]."""


class ExactDescriptionSimilarityBackend:
    """Fallback using normalized exact string match."""

    @property
    def name(self) -> str:
        return "exact"

    def score(self, text_a: str, text_b: str) -> float:
        return 1.0 if _normalize_string(text_a) == _normalize_string(text_b) else 0.0


class TfidfDescriptionSimilarityBackend:
    """Local TF-IDF cosine similarity using scikit-learn."""

    @property
    def name(self) -> str:
        return "tfidf_cosine"

    def score(self, text_a: str, text_b: str) -> float:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text_a, text_b])
        if vectors.shape[1] == 0:
            return 0.0
        similarity = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
        return max(0.0, min(1.0, similarity))


class SentenceTransformerDescriptionSimilarityBackend:
    """Optional sentence-transformer backend when locally available."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    @property
    def name(self) -> str:
        return f"sentence_transformer:{self._model_name}"

    def score(self, text_a: str, text_b: str) -> float:
        embeddings = self._model.encode([text_a, text_b], normalize_embeddings=True)
        similarity = float(embeddings[0] @ embeddings[1])
        return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


def _try_sentence_transformer_backend() -> DescriptionSimilarityBackend | None:
    if importlib.util.find_spec("sentence_transformers") is None:
        logger.info("description backend: sentence-transformers unavailable")
        return None
    try:
        backend = SentenceTransformerDescriptionSimilarityBackend()
    except Exception as exc:
        logger.info(
            "description backend: sentence-transformers unavailable (%s)",
            exc,
        )
        return None
    logger.info("description backend: selected %s", backend.name)
    return backend


def _try_tfidf_backend() -> DescriptionSimilarityBackend | None:
    if importlib.util.find_spec("sklearn") is None:
        logger.info("description backend: scikit-learn unavailable (module not found)")
        return None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
        from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401
    except Exception as exc:
        logger.info(
            "description backend: scikit-learn unavailable (%s)",
            exc,
        )
        return None

    backend = TfidfDescriptionSimilarityBackend()
    logger.info("description backend: selected %s", backend.name)
    return backend


def create_default_description_backend(*, reset_cache: bool = False) -> DescriptionSimilarityBackend:
    """Select the best locally available backend without downloads or new deps."""
    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is not None and not reset_cache:
        return _DEFAULT_BACKEND

    backend = _try_sentence_transformer_backend()
    if backend is not None:
        _DEFAULT_BACKEND = backend
        return backend

    backend = _try_tfidf_backend()
    if backend is not None:
        _DEFAULT_BACKEND = backend
        return backend

    backend = ExactDescriptionSimilarityBackend()
    logger.info("description backend: selected %s (fallback)", backend.name)
    _DEFAULT_BACKEND = backend
    return backend
