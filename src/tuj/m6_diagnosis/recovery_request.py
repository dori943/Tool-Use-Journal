"""Module-facing Recovery Request contract for M6 dispatch.

M6 decides *what* recovery is needed and *which* module owns it.
Target modules later consume ``recovery_request.json`` and perform the
concrete recovery internally (LLM / planner / controller logic).

Contract consumed by future module entry points:
1. read recovery_request.json
2. inspect recovery_type
3. inspect target
4. inspect parameters (default {})
5. execute module-local recovery

``target_module`` is intentionally omitted from this request body; it is
routing metadata owned by the dispatcher.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .recovery_router import normalize_recovery_action


class RecoveryRequestError(ValueError):
    """Raised when a module-facing Recovery Request cannot be built."""


class RecoveryRequestTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subgoal_id: str | None = None
    object_id: str | None = None
    property: str | None = None
    relation: str | None = None
    ee_id: str | None = None
    tool_id: str | None = None


class RecoveryRequest(BaseModel):
    """Minimal recovery payload delivered to a target module."""

    model_config = ConfigDict(extra="forbid")

    recovery_type: str = Field(min_length=1)
    target: RecoveryRequestTarget
    parameters: dict[str, Any] = Field(default_factory=dict)


def build_recovery_request(recovery: dict) -> dict:
    """Build a module-facing Recovery Request from an M6 recovery object.

    Does not mutate ``recovery``.
    """
    if not isinstance(recovery, dict):
        raise RecoveryRequestError("recovery must be a dict")

    action = normalize_recovery_action(recovery.get("action"))
    recovery_type = action.get("recovery_type")
    if not isinstance(recovery_type, str) or not recovery_type.strip():
        raise RecoveryRequestError("recovery.action.recovery_type is required")

    raw_target = action.get("target")
    if raw_target is None:
        raw_target = {}
    if not isinstance(raw_target, dict):
        raise RecoveryRequestError("recovery.action.target must be an object")

    parameters = action.get("parameters")
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        raise RecoveryRequestError("recovery.action.parameters must be an object")

    try:
        request = RecoveryRequest.model_validate(
            {
                "recovery_type": recovery_type.strip(),
                "target": deepcopy(raw_target),
                "parameters": deepcopy(parameters),
            }
        )
    except ValidationError as error:
        raise RecoveryRequestError(str(error)) from error

    return request.model_dump()
