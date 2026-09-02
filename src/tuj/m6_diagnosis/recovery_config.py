"""Configuration and factory helpers for M6 recovery routers."""

from __future__ import annotations

import os

DEFAULT_M6_RECOVERY_MODEL = "gpt-4.1-mini"
DEFAULT_M6_RECOVERY_ROUTER_BACKEND = "mock"
M6_RECOVERY_MODEL_ENV = "M6_RECOVERY_MODEL"
M6_RECOVERY_ROUTER_BACKEND_ENV = "M6_RECOVERY_ROUTER_BACKEND"
VALID_M6_RECOVERY_ROUTER_BACKENDS = frozenset({"mock", "openai"})


def get_recovery_model(model: str | None = None) -> str:
    if model is not None:
        return model
    return os.environ.get(M6_RECOVERY_MODEL_ENV, DEFAULT_M6_RECOVERY_MODEL)


def get_recovery_router_backend(backend: str | None = None) -> str:
    resolved = backend or os.environ.get(
        M6_RECOVERY_ROUTER_BACKEND_ENV,
        DEFAULT_M6_RECOVERY_ROUTER_BACKEND,
    )
    if resolved not in VALID_M6_RECOVERY_ROUTER_BACKENDS:
        raise ValueError(
            f"invalid M6 recovery router backend {resolved!r}; "
            f"expected one of {sorted(VALID_M6_RECOVERY_ROUTER_BACKENDS)}"
        )
    return resolved


def create_recovery_router(
    backend: str | None = None,
    *,
    model: str | None = None,
    client=None,
):
    """Create a recovery router for the requested backend."""
    from .openai_recovery_router import OpenAIRecoveryRouter
    from .recovery_router import MockRecoveryRouter

    resolved_backend = get_recovery_router_backend(backend)
    if resolved_backend == "mock":
        return MockRecoveryRouter()
    return OpenAIRecoveryRouter(model=model, client=client)
