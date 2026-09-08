"""Recovery Dispatcher: create Recovery Request artifacts without re-executing modules.

Dispatch stops after:
- validating recovery.action.target_module vs recovery_type
- building module-facing Recovery Request
- writing output/<task>/m6/recovery_request.json
- returning dispatch metadata

Actual M1–M5 recovery execution is intentionally out of scope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .recovery_request import RecoveryRequestError, build_recovery_request
from .recovery_router import normalize_recovery_action
from .taxonomy import RECOVERY_ACTION_MODULES, VALID_ROUTING_MODULES


class RecoveryDispatchError(ValueError):
    """Raised when recovery cannot be dispatched to a target module."""


def validate_module_recovery_type(target_module: str, recovery_type: str) -> None:
    """Ensure recovery_type is owned by the declared target_module."""
    if not isinstance(target_module, str) or not target_module.strip():
        raise RecoveryDispatchError("target_module is required")
    if not isinstance(recovery_type, str) or not recovery_type.strip():
        raise RecoveryDispatchError("recovery_type is required")

    target_module = target_module.strip()
    recovery_type = recovery_type.strip()

    if target_module not in VALID_ROUTING_MODULES or target_module in {"M0", "M6"}:
        raise RecoveryDispatchError(
            f"unsupported dispatch target_module: {target_module!r}"
        )

    expected = RECOVERY_ACTION_MODULES.get(recovery_type)
    if expected is None:
        raise RecoveryDispatchError(f"unknown recovery_type: {recovery_type!r}")
    if expected != target_module:
        raise RecoveryDispatchError(
            f"recovery_type {recovery_type!r} belongs to {expected!r}, "
            f"not target_module {target_module!r}"
        )


def dispatch_recovery(
    recovery: dict,
    *,
    output_dir: str | Path,
    save: bool = True,
) -> dict[str, Any]:
    """Create recovery_request.json and return dispatch metadata.

    Does not mutate ``recovery`` and does not execute target modules.
    """
    if not isinstance(recovery, dict):
        raise RecoveryDispatchError("recovery must be a dict")

    action = normalize_recovery_action(recovery.get("action"))
    target_module = action.get("target_module")
    recovery_type = action.get("recovery_type")

    if not isinstance(target_module, str) or not target_module.strip():
        raise RecoveryDispatchError("recovery.action.target_module is required")
    if not isinstance(recovery_type, str) or not recovery_type.strip():
        raise RecoveryDispatchError("recovery.action.recovery_type is required")

    validate_module_recovery_type(target_module, recovery_type)

    try:
        request = build_recovery_request(recovery)
    except RecoveryRequestError as error:
        raise RecoveryDispatchError(str(error)) from error

    output_dir = Path(output_dir)
    m6_dir = output_dir / "m6"
    request_path = m6_dir / "recovery_request.json"

    if save:
        m6_dir.mkdir(parents=True, exist_ok=True)
        request_path.write_text(
            json.dumps(request, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "target_module": target_module.strip(),
        "recovery_type": recovery_type.strip(),
        "request_path": str(request_path),
        "request": request,
        "module_executed": False,
    }
