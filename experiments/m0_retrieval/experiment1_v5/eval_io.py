"""v5 output / evaluation I/O ??same semantics as Experiment 1 v3.

Reuses (import only; do not edit):
  - experiments/m0_retrieval/evaluator.py
  - experiments/m0_retrieval/result_writer.py

Writes under output/m0_retrieval/experiment1_v5/ (units/, raw_results*.csv,
condition_summary.csv). Does not read live_runs/ into aggregates.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXP_V5 = Path(__file__).resolve().parent
PARENT_EXP = EXP_V5.parent
ROOT = EXP_V5.parents[2]

for p in (ROOT, ROOT / "src", PARENT_EXP, EXP_V5):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluator import aggregate_condition, evaluate_inference  # noqa: E402
from result_writer import (  # noqa: E402
    RAW_FIELDS as V3_RAW_FIELDS,
    SUMMARY_FIELDS,
    merge_raw_results,
    print_object_table,
    read_csv,
    write_csv,
    write_json,
)
from v5_conditions import CONDITIONS, PROPERTIES  # noqa: E402

OUT_DIR = ROOT / "output" / "m0_retrieval" / "experiment1_v5"
UNITS_DIR = OUT_DIR / "units"

# Experiment 1 matrix used in reports (matches v3 primary set).
CORE_OBJECT_KEYS = ("bottle", "spoon", "ladle", "plate", "mug")

# v3 columns + v5 stage token/call breakdown (common aliases kept).
V5_EXTRA_FIELDS = [
    "siphy_input_tokens",
    "siphy_output_tokens",
    "siphy_total_tokens",
    "gemini_input_tokens",
    "gemini_output_tokens",
    "gemini_total_tokens",
    "total_input_tokens",
    "total_output_tokens",
    "siphy_model_call_count",
    "gemini_model_call_count",
    "total_llm_call_count",
]
RAW_FIELDS = list(V3_RAW_FIELDS) + [f for f in V5_EXTRA_FIELDS if f not in V3_RAW_FIELDS]


def _tok(value: Any) -> Any:
    if value is None:
        return "unavailable"
    return value


def unit_is_batch_eligible(unit: dict[str, Any]) -> bool:
    """Include in final aggregates when the unit is a completed live inference.

    Excludes dry-run / required-input skips and legacy placeholders so partial
    failures are not mixed into the rolling raw_results as if they were scores.
    API/JSON failures with empty preds are still included (v3 prediction_missing).
    """
    if unit.get("dry_run") or unit.get("dry_run_placeholder"):
        return False
    # Must be a real object×condition unit (ignore stray files).
    if not unit.get("object") or not unit.get("condition"):
        return False
    if str(unit.get("condition")) not in CONDITIONS:
        return False
    skip = bool(unit.get("skipped"))
    reason = str(unit.get("skip_reason") or "")
    if skip and "required input unavailable" in reason:
        return False
    if skip and "dry-run" in reason.lower():
        return False
    return True


def build_unit_payload(
    *,
    object_key: str,
    condition,
    unit,
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Serialize a live UnitResult (+ evals) to the v3-style unit JSON schema."""
    return {
        "experiment_version": "experiment1_v5",
        "object": object_key,
        "condition": condition.id,
        "condition_name": condition.name,
        "input_factors": condition.input_factors_label,
        "prediction": unit.prediction,
        "parsed_prediction": unit.prediction,
        "evaluations": evaluations,
        "siphy_cache_hit": False,
        "siphy_call_executed": unit.siphy_call_executed,
        "siphy_model_call_count": unit.siphy_model_call_count,
        "gemini_model_call_count": unit.gemini_model_call_count,
        "total_llm_call_count": unit.total_llm_call_count,
        "siphy_input_tokens": unit.siphy_input_tokens,
        "siphy_output_tokens": unit.siphy_output_tokens,
        "siphy_total_tokens": unit.siphy_total_tokens,
        "gemini_input_tokens": unit.gemini_input_tokens,
        "gemini_output_tokens": unit.gemini_output_tokens,
        "gemini_total_tokens": unit.gemini_total_tokens,
        "total_input_tokens": unit.total_input_tokens,
        "total_output_tokens": unit.total_output_tokens,
        "total_tokens": unit.total_tokens,
        "total_tokens_combined": unit.total_tokens_combined,
        "input_tokens": unit.input_tokens,
        "output_tokens": unit.output_tokens,
        "model_call_count": unit.model_call_count,
        "provider": unit.provider,
        "model": unit.model,
        "siphy_material": unit.siphy_material,
        "siphy_density_kgm3": unit.siphy_density_kgm3,
        "bbox_mm": unit.bbox_mm,
        "crop_image_path": unit.crop_image_path,
        "fixed_cues": unit.stage2.fixed_cues_applied,
        "prompt_user": unit.stage2.prompt_user,
        "raw_response": unit.stage2.raw_response,
        "stage1_raw": unit.stage1.to_json() if unit.stage1 else None,
        "image_used": True,
        "object_name_in_prompt": unit.stage2.object_name_in_prompt,
        "error": unit.stage2.error,
        "failure_reason": unit.stage2.failure_reason,
        "skipped": unit.stage2.skipped,
        "skip_reason": unit.stage2.skip_reason,
    }


def rows_from_unit_dict(unit: dict[str, Any]) -> list[dict[str, Any]]:
    """Build property-level raw rows from a saved unit JSON (v3 semantics)."""
    cid = str(unit["condition"])
    cond = CONDITIONS[cid]
    obj_key = str(unit["object"])
    gt = unit.get("_gt")  # optional inject for tests
    if gt is None:
        from gt_loader import load_all_gt

        gt = load_all_gt().get(obj_key) or {}

    pred = unit.get("prediction") or unit.get("parsed_prediction") or {}
    evals = evaluate_inference(cond, pred, gt)

    unit_failed = bool(unit.get("error")) or bool(
        unit.get("failure_reason")
        and str(unit.get("failure_reason")).startswith(("api_", "json_"))
    )

    rows: list[dict[str, Any]] = []
    for ev in evals:
        status = ev.get("evaluation_status")
        pred_out = ev["prediction"]
        err = ev["error"]
        evaluated = ev["evaluated"]
        if unit_failed:
            pred_out = "unavailable"
            err = "failed"
            evaluated = False
            status = "prediction_missing"
        rows.append(
            {
                "object": obj_key,
                "condition": cid,
                "condition_name": unit.get("condition_name") or cond.name,
                "input_factors": unit.get("input_factors") or cond.input_factors_label,
                "property": ev["property"],
                "gt": ev["gt"],
                "prediction": pred_out,
                "error": err,
                "evaluated": evaluated,
                "evaluation_status": status,
                "gt_source": ev["gt_source"],
                "provider": unit.get("provider"),
                "model": unit.get("model"),
                "input_tokens": _tok(unit.get("input_tokens")),
                "output_tokens": _tok(unit.get("output_tokens")),
                "total_tokens": _tok(unit.get("total_tokens")),
                "model_call_count": unit.get("model_call_count")
                if unit.get("model_call_count") is not None
                else unit.get("total_llm_call_count") or 0,
                "image_used_in_inference": unit.get("image_used", True),
                "object_name_in_prompt": unit.get("object_name_in_prompt", False),
                "skipped": unit.get("skipped", False),
                "skip_reason": unit.get("skip_reason"),
                "error_message": unit.get("error"),
                "failure_reason": unit.get("failure_reason")
                or (
                    f"prediction_missing:{ev['property']}"
                    if status == "prediction_missing"
                    else None
                ),
                "siphy_input_tokens": _tok(unit.get("siphy_input_tokens")),
                "siphy_output_tokens": _tok(unit.get("siphy_output_tokens")),
                "siphy_total_tokens": _tok(unit.get("siphy_total_tokens")),
                "gemini_input_tokens": _tok(unit.get("gemini_input_tokens")),
                "gemini_output_tokens": _tok(unit.get("gemini_output_tokens")),
                "gemini_total_tokens": _tok(unit.get("gemini_total_tokens")),
                "total_input_tokens": _tok(unit.get("total_input_tokens")),
                "total_output_tokens": _tok(unit.get("total_output_tokens")),
                "siphy_model_call_count": unit.get("siphy_model_call_count"),
                "gemini_model_call_count": unit.get("gemini_model_call_count"),
                "total_llm_call_count": unit.get("total_llm_call_count"),
            }
        )
    return rows


def rows_from_live_unit(
    *,
    object_key: str,
    condition,
    unit,
    gt: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate a live UnitResult and return (raw_rows, unit_payload)."""
    evals = evaluate_inference(condition, unit.prediction, gt)
    payload = build_unit_payload(
        object_key=object_key,
        condition=condition,
        unit=unit,
        evaluations=evals,
    )
    # Attach gt for rows_from_unit_dict without reloading
    payload_for_rows = dict(payload)
    payload_for_rows["_gt"] = gt
    rows = rows_from_unit_dict(payload_for_rows)
    return rows, payload


def load_eligible_units(
    units_dir: Path | None = None,
    *,
    successful_only: bool = True,
) -> list[dict[str, Any]]:
    """Load unit JSONs from experiment root units/ (never live_runs/)."""
    units_dir = units_dir or UNITS_DIR
    if not units_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(units_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if successful_only and not unit_is_batch_eligible(data):
            continue
        out.append(data)
    return out


def write_raw_and_summary(
    raw_rows: list[dict[str, Any]],
    *,
    out_dir: Path | None = None,
    merge_rolling: bool = True,
    write_timestamp_snapshot: bool = True,
) -> dict[str, Path]:
    """Write raw_results.csv (+ timestamp snapshot) and condition_summary.csv."""
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if write_timestamp_snapshot and raw_rows:
        snap = out_dir / f"raw_results_{stamp}.csv"
        write_csv(snap, raw_rows, RAW_FIELDS)
        paths["snapshot"] = snap

    rolling = out_dir / "raw_results.csv"
    if merge_rolling and rolling.exists():
        existing = read_csv(rolling)
        # Drop v4-incompatible-only handling: merge by object×condition like v3.
        merged = merge_raw_results(existing, raw_rows)
    else:
        merged = list(raw_rows)
    write_csv(rolling, merged, RAW_FIELDS)
    paths["rolling"] = rolling

    summary_rows: list[dict[str, Any]] = []
    for cid in CONDITIONS:
        cond_rows = [r for r in merged if str(r.get("condition")) == cid]
        if cond_rows:
            summary_rows.append(aggregate_condition(cond_rows))
    summary_path = out_dir / "condition_summary.csv"
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    paths["summary"] = summary_path
    return paths


def recompute_from_units(
    *,
    successful_only: bool = True,
    out_dir: Path | None = None,
) -> int:
    """Rebuild raw_results + condition_summary from units/ (v3 --recompute-from-units)."""
    out_dir = out_dir or OUT_DIR
    units = load_eligible_units(out_dir / "units", successful_only=successful_only)
    if not units:
        print("[v5] No eligible units/ found to recompute (live_runs/ is ignored).")
        return 1

    from gt_loader import load_all_gt

    gts = load_all_gt()
    write_json(out_dir / "gt_manifest.json", gts)

    all_rows: list[dict[str, Any]] = []
    for unit in units:
        u = dict(unit)
        u["_gt"] = gts.get(str(unit["object"])) or {}
        all_rows.extend(rows_from_unit_dict(u))

    # Full rebuild (no merge with prior CSV ??units/ is source of truth).
    paths = write_raw_and_summary(
        all_rows,
        out_dir=out_dir,
        merge_rolling=False,
        write_timestamp_snapshot=True,
    )
    write_json(
        out_dir / "run_metadata.json",
        {
            "experiment": "experiment1_v5",
            "mode": "recompute_from_units",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_units": len(units),
            "n_raw_rows": len(all_rows),
            "successful_only": successful_only,
            "paths": {k: str(v) for k, v in paths.items()},
            "note": "Aggregated from experiment1_v5/units/ only; live_runs/ ignored.",
        },
    )
    print(f"[v5] recomputed {len(units)} units ??{len(all_rows)} raw rows")
    print(f"[CSV] {paths['rolling']}")
    print(f"[CSV] {paths['summary']}")
    if "snapshot" in paths:
        print(f"[CSV] {paths['snapshot']}")
    # Spot-check first object if present
    objs = sorted({str(u["object"]) for u in units})
    if objs:
        print_object_table(objs[0], all_rows)
    return 0


def static_eval_pipeline_test() -> int:
    """No API: synthesize 5×7 units, verify 175 raw rows + summary (in smoke dir)."""
    from gt_loader import load_all_gt
    from objects import OBJECTS

    gts = load_all_gt()
    smoke = OUT_DIR / "_static_eval_smoke"
    units_dir = smoke / "units"
    if units_dir.exists():
        for p in units_dir.glob("*.json"):
            p.unlink()
    units_dir.mkdir(parents=True, exist_ok=True)

    object_keys = [k for k in CORE_OBJECT_KEYS if k in OBJECTS]
    n_units = 0
    for ok in object_keys:
        gt = gts[ok]
        for cid, cond in CONDITIONS.items():
            # Deterministic dummy preds (valid for scoring paths).
            pred = {
                "material": "plastic",
                "density_kgm3": 1000.0,
                "mass_kg": 0.1,
                "mu": 0.5,
                "youngs_gpa": 1.0,
            }
            unit = {
                "experiment_version": "experiment1_v5",
                "object": ok,
                "condition": cid,
                "condition_name": cond.name,
                "input_factors": cond.input_factors_label,
                "prediction": pred,
                "provider": "siphy(auto)+gemini" if cond.uses_siphy else "gemini",
                "model": "siphy=gemini-3.6-flash;gemini=gemini-3.6-flash",
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "model_call_count": 2 if cond.uses_siphy else 1,
                "siphy_input_tokens": 40 if cond.uses_siphy else None,
                "siphy_output_tokens": 20 if cond.uses_siphy else None,
                "siphy_total_tokens": 60 if cond.uses_siphy else None,
                "gemini_input_tokens": 60,
                "gemini_output_tokens": 30,
                "gemini_total_tokens": 90,
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "siphy_model_call_count": 1 if cond.uses_siphy else 0,
                "gemini_model_call_count": 1,
                "total_llm_call_count": 2 if cond.uses_siphy else 1,
                "image_used": True,
                "object_name_in_prompt": False,
                "skipped": False,
                "skip_reason": None,
                "error": None,
                "failure_reason": None,
                "bbox_mm": [1.0, 2.0, 3.0] if cond.uses_bbox else None,
            }
            evals = evaluate_inference(cond, pred, gt)
            unit["evaluations"] = evals
            write_json(units_dir / f"{ok}_{cid}.json", unit)
            n_units += 1

    # Also drop an ineligible partial that must NOT appear in aggregate.
    write_json(
        units_dir / "_partial_skip_bottle_C1.json",
        {
            "experiment_version": "experiment1_v5",
            "object": "bottle",
            "condition": "C1",
            "prediction": {k: None for k in PROPERTIES},
            "skipped": True,
            "skip_reason": "required input unavailable: ['bbox(m1)']",
        },
    )

    code = recompute_from_units(successful_only=True, out_dir=smoke)
    if code != 0:
        return code

    rows = read_csv(smoke / "raw_results.csv")
    expected = len(object_keys) * len(CONDITIONS) * len(PROPERTIES)
    errors: list[str] = []
    if len(rows) != expected:
        errors.append(f"raw rows {len(rows)} != {expected}")
    if n_units != len(object_keys) * len(CONDITIONS):
        errors.append(f"units written {n_units}")
    with (smoke / "raw_results.csv").open(encoding="utf-8-sig") as f:
        header = f.readline().strip().split(",")
    for col in RAW_FIELDS:
        if col not in header:
            errors.append(f"missing column {col}")
    summary = read_csv(smoke / "condition_summary.csv")
    if len(summary) != len(CONDITIONS):
        errors.append(f"summary conditions {len(summary)} != {len(CONDITIONS)}")

    bottle_c1 = [r for r in rows if r["object"] == "bottle" and r["condition"] == "C1"]
    if len(bottle_c1) != len(PROPERTIES):
        errors.append(f"bottle C1 rows={len(bottle_c1)} (partial leak?)")

    print("=" * 66)
    print("[v5] STATIC EVAL PIPELINE TEST")
    print(f"  objects: {object_keys}")
    print(f"  units: {n_units}")
    print(f"  raw_results rows: {len(rows)} (expected {expected})")
    print(f"  condition_summary rows: {len(summary)}")
    print(f"  RAW_FIELDS ({len(RAW_FIELDS)}): {RAW_FIELDS}")
    print(f"  smoke dir: {smoke}")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("OK: schema + 175-row aggregate + partial skip excluded")
    print("OK: live_runs/ not used")
    print("=" * 66)
    return 0
