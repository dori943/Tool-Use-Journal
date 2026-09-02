"""Configuration and factory helpers for M6 failure diagnosers."""

from __future__ import annotations

import os

DEFAULT_M6_DIAGNOSIS_MODEL = "gpt-4.1-mini"
DEFAULT_M6_DIAGNOSER_BACKEND = "mock"
M6_DIAGNOSIS_MODEL_ENV = "M6_DIAGNOSIS_MODEL"
M6_DIAGNOSER_BACKEND_ENV = "M6_DIAGNOSER_BACKEND"
VALID_M6_DIAGNOSER_BACKENDS = frozenset({"mock", "openai"})


def get_diagnosis_model(model: str | None = None) -> str:
    if model is not None:
        return model
    return os.environ.get(M6_DIAGNOSIS_MODEL_ENV, DEFAULT_M6_DIAGNOSIS_MODEL)


def get_diagnoser_backend(backend: str | None = None) -> str:
    resolved = backend or os.environ.get(
        M6_DIAGNOSER_BACKEND_ENV,
        DEFAULT_M6_DIAGNOSER_BACKEND,
    )
    if resolved not in VALID_M6_DIAGNOSER_BACKENDS:
        raise ValueError(
            f"invalid M6 diagnoser backend {resolved!r}; "
            f"expected one of {sorted(VALID_M6_DIAGNOSER_BACKENDS)}"
        )
    return resolved


def create_failure_diagnoser(
    backend: str | None = None,
    *,
    model: str | None = None,
    client=None,
):
    """Create a failure diagnoser for the requested backend."""
    from .diagnosis import MockFailureDiagnoser
    from .openai_vlm_diagnoser import OpenAIVLMFailureDiagnoser

    resolved_backend = get_diagnoser_backend(backend)
    if resolved_backend == "mock":
        return MockFailureDiagnoser()
    return OpenAIVLMFailureDiagnoser(model=model, client=client)
