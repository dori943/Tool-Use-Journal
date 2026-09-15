"""Shared timing / token summary helpers for ablation runners."""

from __future__ import annotations

from typing import Any


def llm_usage_rows(usage: dict | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, entry in (usage or {}).items():
        if not isinstance(entry, dict):
            continue
        rows.append(
            {
                "call": name,
                "prompt_tokens": int(entry.get("prompt_tokens") or 0),
                "completion_tokens": int(entry.get("completion_tokens") or 0),
                "total_tokens": int(
                    entry.get("tokens")
                    or entry.get("total_tokens")
                    or 0
                ),
                "seconds": float(entry.get("seconds") or 0.0),
                "calls": int(entry.get("calls") or 0),
            }
        )
    return rows


def sum_llm_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
        "total_tokens": sum(row["total_tokens"] for row in rows),
        "seconds": round(sum(row["seconds"] for row in rows), 2),
        "calls": sum(row["calls"] for row in rows),
    }


def build_timing_summary(
    *,
    stage_order: tuple[str, ...],
    stage_seconds: dict[str, float | None],
    stage_status: dict[str, str],
    llm_usage: dict | None,
    wall_seconds: float | None,
    combined_stage_keys: tuple[str, ...] = (),
    combined_label: str = "combined_seconds",
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a timing/token report for ablation runners."""

    llm_calls = llm_usage_rows(llm_usage)
    stages: dict[str, dict[str, Any]] = {}
    for name in stage_order:
        seconds = stage_seconds.get(name)
        stages[name] = {
            "seconds": None if seconds is None else round(float(seconds), 2),
            "status": stage_status.get(name, "not_run"),
        }
    measured = [
        stages[name]["seconds"]
        for name in stage_order
        if stages[name]["seconds"] is not None
    ]
    combined = None
    if combined_stage_keys and any(
        stages[name]["seconds"] is not None
        for name in combined_stage_keys
        if name in stages
    ):
        combined = round(
            sum(stages[name]["seconds"] or 0.0 for name in combined_stage_keys),
            2,
        )
    summary: dict[str, Any] = {
        "llm_calls": llm_calls,
        "llm_total": sum_llm_rows(llm_calls),
        "stages": stages,
        combined_label: combined,
        "total_seconds": (
            None if wall_seconds is None else round(float(wall_seconds), 2)
        ),
        "measured_stage_sum_seconds": (
            None if not measured else round(sum(measured), 2)
        ),
    }
    if extras:
        summary.update(extras)
    return summary


def merge_stage_seconds(
    previous: dict[str, Any] | None,
    current: dict[str, float | None],
) -> dict[str, float | None]:
    """Keep prior stage wall times when a resume does not re-run that stage."""

    merged = dict(current)
    if not previous:
        return merged
    prior_stages = previous.get("stages") or {}
    for name, entry in prior_stages.items():
        if not isinstance(entry, dict):
            continue
        if merged.get(name) is None and entry.get("seconds") is not None:
            merged[name] = float(entry["seconds"])
    return merged


def merge_stage_status(
    previous: dict[str, Any] | None,
    current: dict[str, str],
) -> dict[str, str]:
    merged = dict(current)
    if not previous:
        return merged
    prior_stages = previous.get("stages") or {}
    for name, entry in prior_stages.items():
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        if merged.get(name, "not_run") in {"not_run", "resumed"} and status:
            if merged.get(name) == "resumed" or merged.get(name) == "not_run":
                # Prefer a concrete prior status over a bare resumed/not_run label
                # when this session did not re-measure the stage.
                if merged.get(name) == "not_run" and status not in {None, "not_run"}:
                    merged[name] = str(status)
                elif merged.get(name) == "resumed" and status not in {
                    None,
                    "not_run",
                    "resumed",
                }:
                    merged[name] = f"resumed:{status}"
    return merged


def print_timing_summary(
    summary: dict[str, Any],
    *,
    stage_order: tuple[str, ...],
    combined_label: str | None = None,
) -> None:
    llm = summary.get("llm_total") or {}
    print(
        "[timing] LLM "
        f"prompt={llm.get('prompt_tokens', 0)} "
        f"completion={llm.get('completion_tokens', 0)} "
        f"total={llm.get('total_tokens', 0)} "
        f"({llm.get('seconds', 0):.2f}s)"
    )
    for row in summary.get("llm_calls") or []:
        print(
            f"  [{row['call']}] "
            f"{row['prompt_tokens']}+{row['completion_tokens']}="
            f"{row['total_tokens']}tok {row['seconds']:.2f}s"
        )
    stages = summary.get("stages") or {}
    for name in stage_order:
        entry = stages.get(name) or {}
        seconds = entry.get("seconds")
        label = "n/a" if seconds is None else f"{seconds:.2f}s"
        print(f"  [{name}] {label} ({entry.get('status', 'not_run')})")
    if combined_label and summary.get(combined_label) is not None:
        print(f"  [{combined_label}] {summary[combined_label]:.2f}s")
    total = summary.get("total_seconds")
    if total is not None:
        print(f"  [total] {total:.2f}s")
