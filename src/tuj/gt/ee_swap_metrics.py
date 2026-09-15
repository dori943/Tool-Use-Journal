"""Count EE attach / exchange events from live execution manifests.

Aligns with M4 ``CostVector.ee_switches``:

- successful ``EE_EXCHANGE`` → one switch (and one detach + one attach)
- successful ``EE_ATTACH`` / ``INITIAL_ATTACH_EE`` → initial mount only (not a switch)
- ``EE_EXCHANGE_ENTRY`` → rack approach, not a switch
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_ATTACH_ACTIONS = frozenset({"EE_ATTACH", "INITIAL_ATTACH_EE"})
_EXCHANGE_ACTIONS = frozenset({"EE_EXCHANGE"})


def _action_type(step: Mapping[str, Any]) -> str:
    raw = step.get("action_type") or step.get("operation") or step.get("action") or ""
    return str(raw).strip().upper()


def _step_succeeded(step: Mapping[str, Any]) -> bool:
    status = str(step.get("status") or "").upper()
    if status in {"FAILED", "ERROR", "RUNNING"}:
        return False
    if status in {"SUCCESS", "OK", "SUCCEEDED"}:
        return True
    # Older records may only expose the simulation player status.
    execution = str(step.get("execution_status") or "").upper()
    return execution in {"SUCCESS", "OK", "SUCCEEDED", ""}


def count_executed_ee_metrics(steps: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    """Aggregate executed EE metrics from live step records."""

    attaches = 0
    detaches = 0
    switches = 0
    exchange_entry = 0
    other = 0
    considered: list[dict[str, Any]] = []

    for step in steps or ():
        if not isinstance(step, Mapping):
            continue
        action = _action_type(step)
        if action not in _ATTACH_ACTIONS | _EXCHANGE_ACTIONS | {"EE_EXCHANGE_ENTRY"}:
            if action:
                other += 1
            continue
        ok = _step_succeeded(step)
        entry = {
            "index": step.get("index"),
            "subgoal_id": step.get("subgoal_id"),
            "action_type": action,
            "status": step.get("status"),
            "counted": ok,
        }
        considered.append(entry)
        if not ok:
            continue
        if action in _ATTACH_ACTIONS:
            attaches += 1
        elif action in _EXCHANGE_ACTIONS:
            switches += 1
            attaches += 1
            detaches += 1
        elif action == "EE_EXCHANGE_ENTRY":
            exchange_entry += 1

    return {
        "executed_ee_switches": switches,
        "executed_n_ee_attaches": attaches,
        "executed_n_ee_detaches": detaches,
        "executed_ee_exchange_entry": exchange_entry,
        "executed_ee_other_actions": other,
        "executed_ee_events": considered,
        "definition": (
            "executed_ee_switches counts successful EE_EXCHANGE only "
            "(matches M4 cost_vector.ee_switches; initial EE_ATTACH excluded)"
        ),
    }


def metrics_from_manifest_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return count_executed_ee_metrics([])
    steps = payload.get("steps") or payload.get("results") or payload.get("runs") or []
    if not isinstance(steps, list):
        steps = []
    return count_executed_ee_metrics(steps)


def load_executed_ee_metrics(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, Mapping) and isinstance(payload.get("executed_ee_metrics"), Mapping):
        return dict(payload["executed_ee_metrics"])
    return metrics_from_manifest_payload(payload if isinstance(payload, Mapping) else None)


def find_latest_live_manifest(task_m5_dir: str | Path) -> Path | None:
    """Prefer ``live/live-execution-manifest.json``, else newest run manifest."""

    root = Path(task_m5_dir)
    latest = root / "live" / "live-execution-manifest.json"
    if latest.is_file():
        return latest
    live_root = root / "live"
    if not live_root.is_dir():
        # Some runners write the manifest directly under m5/.
        direct = root / "live-execution-manifest.json"
        return direct if direct.is_file() else None
    candidates = sorted(
        live_root.glob("run-*/live-execution-manifest.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None
