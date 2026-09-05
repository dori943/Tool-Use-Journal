"""Write Experiment 1 CSV / JSON outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


RAW_FIELDS = [
    "object",
    "condition",
    "condition_name",
    "input_factors",
    "property",
    "gt",
    "prediction",
    "error",
    "evaluated",
    "evaluation_status",
    "gt_source",
    "provider",
    "model",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "model_call_count",
    "image_used_in_inference",
    "object_name_in_prompt",
    "skipped",
    "skip_reason",
    "error_message",
    "failure_reason",
]

SUMMARY_FIELDS = [
    "condition",
    "condition_name",
    "material_accuracy",
    "density_error",
    "mass_error",
    "mu_error",
    "youngs_error",
    "overall_error",
    "overall_error_definition",
    "n_evaluated_properties",
    "n_prediction_failures",
    "n_gt_unavailable",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_total_tokens",
    "total_model_calls",
    "n_units",
]


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return str(value).lower()
    return value


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in fields})


def _normalize_csv_row(row: dict) -> dict:
    """Restore types after DictReader (bools/empty cells become strings)."""
    out = dict(row)
    for key in ("evaluated", "image_used_in_inference", "object_name_in_prompt", "skipped"):
        if key not in out:
            continue
        val = out[key]
        if isinstance(val, str):
            low = val.strip().lower()
            if low in ("", "none", "null"):
                out[key] = False
            else:
                out[key] = low in ("true", "1", "yes")
    for key in ("error", "prediction", "skip_reason", "error_message", "failure_reason", "gt"):
        if key in out and out[key] == "":
            out[key] = None
    for key in ("input_tokens", "output_tokens", "total_tokens", "model_call_count"):
        if key not in out:
            continue
        val = out[key]
        if val in ("", None, "unavailable"):
            continue
        try:
            out[key] = int(float(val))
        except (TypeError, ValueError):
            pass
    return out


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [_normalize_csv_row(dict(r)) for r in reader]


def merge_raw_results(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    """Merge by Object×Condition unit: replace matching units, keep others.

    Same (object, condition) re-run replaces all property rows for that unit
    (no duplicate append). Other units are preserved.
    """
    if not new_rows:
        return list(existing)
    replace_keys = {(str(r["object"]), str(r["condition"])) for r in new_rows}
    kept = [
        r
        for r in existing
        if (str(r["object"]), str(r["condition"])) not in replace_keys
    ]
    merged = kept + list(new_rows)
    prop_order = {
        "material": 0,
        "density_kgm3": 1,
        "mass_kg": 2,
        "mu": 3,
        "youngs_gpa": 4,
    }
    merged.sort(
        key=lambda r: (
            str(r.get("object", "")),
            str(r.get("condition", "")),
            prop_order.get(str(r.get("property", "")), 99),
        )
    )
    return merged


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_object_table(object_key: str, rows: list[dict]) -> None:
    selected = [r for r in rows if r["object"] == object_key]
    print(f"\n===== {object_key.upper()} RAW RESULT =====")
    header = (
        f"{'Cond':4s} {'Property':14s} {'GT':18s} {'Pred':18s} "
        f"{'Error':10s} {'Eval':5s} {'InTok':6s} {'OutTok':6s} {'TotTok':6s} {'Calls':5s}"
    )
    print(header)
    for r in selected:
        print(
            f"{r['condition']:4s} {r['property']:14s} "
            f"{str(r.get('gt'))[:18]:18s} {str(r.get('prediction'))[:18]:18s} "
            f"{str(r.get('error'))[:10]:10s} {str(r.get('evaluated')):5s} "
            f"{str(r.get('input_tokens')):6s} {str(r.get('output_tokens')):6s} "
            f"{str(r.get('total_tokens')):6s} {str(r.get('model_call_count')):5s}"
        )
