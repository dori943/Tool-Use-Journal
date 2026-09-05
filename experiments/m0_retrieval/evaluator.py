"""Evaluation helpers for Experiment 1."""
from __future__ import annotations

import math
import re
from typing import Any

from conditions import Condition, PROPERTIES
from gt_loader import gt_available, gt_value

# Relative-error contribution when a prediction target is missing but GT exists.
PREDICTION_FAILURE_ERROR = 1.0


def normalize_material(label: Any) -> str | None:
    if label is None:
        return None
    text = str(label).strip().lower()
    if text in ("", "null", "none", "nan", "unavailable"):
        return None
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text or None


def prediction_present(property_name: str, prediction: Any) -> bool:
    """True if the model returned a usable value for this property."""
    if property_name == "material":
        return normalize_material(prediction) is not None
    if prediction is None:
        return False
    if isinstance(prediction, str):
        text = prediction.strip().lower()
        if text in ("", "null", "none", "nan", "unavailable", "failed"):
            return False
        try:
            prediction = float(text)
        except ValueError:
            return False
    try:
        val = float(prediction)
    except (TypeError, ValueError):
        return False
    return math.isfinite(val)


def relative_error(prediction: Any, gt: Any) -> float | None:
    if prediction is None or gt is None:
        return None
    try:
        p = float(prediction)
        g = float(gt)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p) or not math.isfinite(g):
        return None
    if abs(g) < 1e-12:
        return None if abs(p) < 1e-12 else float("inf")
    return abs(p - g) / abs(g)


def evaluate_property(
    condition: Condition,
    property_name: str,
    prediction: Any,
    gt: dict[str, Any],
) -> dict[str, Any]:
    # Experiment 1 v2: every property is always a prediction target.
    is_target = True
    has_gt = gt_available(gt, property_name)
    gt_val = gt_value(gt, property_name) if has_gt else None
    source_key = {
        "material": "material",
        "density_kgm3": "density_kgm3",
        "mass_kg": "mass_kg",
        "mu": "mu",
        "youngs_gpa": "youngs_gpa",
    }[property_name]
    gt_source = gt[source_key].get("source") if has_gt else None

    has_pred = prediction_present(property_name, prediction)

    # Status taxonomy (v2 — no not_target):
    # - gt_unavailable: no simulator/asset GT → exclude from scores
    # - prediction_missing: GT available + model returned null/missing → failure (penalty 1.0)
    # - evaluated: GT + prediction present → scored
    if not has_gt:
        status = "gt_unavailable"
        evaluated = False
        error = None
        pred_out = prediction if has_pred else None
    elif not has_pred:
        status = "prediction_missing"
        evaluated = False
        error = "failed"
        pred_out = "unavailable"
    else:
        status = "evaluated"
        evaluated = True
        pred_out = prediction
        if property_name == "material":
            pred_n = normalize_material(prediction)
            gt_n = normalize_material(gt_val)
            error = 0.0 if (pred_n is not None and pred_n == gt_n) else 1.0
        else:
            error = relative_error(prediction, gt_val)

    return {
        "property": property_name,
        "gt": gt_val,
        "prediction": pred_out,
        "error": error,
        "evaluated": evaluated,
        "evaluation_status": status,
        "is_prediction_target": is_target,
        "gt_available": has_gt,
        "prediction_available": has_pred,
        "gt_source": gt_source,
    }


def evaluate_inference(condition: Condition, prediction: dict[str, Any], gt: dict[str, Any]) -> list[dict]:
    return [evaluate_property(condition, prop, prediction.get(prop), gt) for prop in PROPERTIES]


def _row_status(row: dict[str, Any]) -> str:
    status = row.get("evaluation_status")
    if status:
        return str(status)
    # Legacy CSV rows (pre-status field): infer
    if not row.get("is_prediction_target") and row.get("evaluated") in (False, "false", ""):
        # may still be target with missing pred recorded as evaluated true historically
        pass
    evaluated = row.get("evaluated")
    if isinstance(evaluated, str):
        evaluated = evaluated.strip().lower() in ("true", "1", "yes")
    pred = row.get("prediction")
    err = row.get("error")
    prop = str(row.get("property", ""))
    if evaluated and err not in (None, "", "failed") and pred not in (None, "", "unavailable"):
        return "evaluated"
    if pred in (None, "", "unavailable") or err in ("failed", "nan", "NaN"):
        # If GT present and looks like a target row with empty pred
        if row.get("gt") not in (None, "") and prop:
            return "prediction_missing"
    if row.get("gt") in (None, "") and prop == "youngs_gpa":
        return "gt_unavailable"
    if evaluated:
        return "evaluated"
    return "not_target"


def aggregate_condition(rows: list[dict]) -> dict[str, Any]:
    """Aggregate property-level raw rows for one condition."""
    by_prop: dict[str, list[dict]] = {p: [] for p in PROPERTIES}
    for row in rows:
        by_prop[row["property"]].append(row)

    n_pred_fail = 0
    n_gt_unavailable = 0
    n_evaluated = 0

    def collect_errors(prop: str) -> list[float]:
        nonlocal n_pred_fail, n_gt_unavailable, n_evaluated
        vals: list[float] = []
        for r in by_prop[prop]:
            status = _row_status(r)
            if status == "prediction_missing":
                n_pred_fail += 1
                vals.append(PREDICTION_FAILURE_ERROR)
            elif status == "gt_unavailable":
                n_gt_unavailable += 1
            elif status == "evaluated":
                err = r.get("error")
                if err is None or err == "":
                    continue
                try:
                    fe = float(err)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fe):
                    n_evaluated += 1
                    vals.append(fe)
        return vals

    material_vals = collect_errors("material")
    # material_accuracy only over successfully evaluated material rows (not failures)
    material_eval_only = [
        float(r["error"])
        for r in by_prop["material"]
        if _row_status(r) == "evaluated"
        and r.get("error") not in (None, "")
        and math.isfinite(float(r["error"]))
    ]
    material_acc = (
        (1.0 - (sum(material_eval_only) / len(material_eval_only)))
        if material_eval_only
        else None
    )
    # For overall, material failures count as error 1.0 via material_vals
    material_error_for_overall = (
        sum(material_vals) / len(material_vals) if material_vals else None
    )

    dens_vals = collect_errors("density_kgm3")
    mass_vals = collect_errors("mass_kg")
    mu_vals = collect_errors("mu")
    young_vals = collect_errors("youngs_gpa")

    def mean_or_none(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    overall_parts: list[float] = []
    if material_error_for_overall is not None:
        overall_parts.append(material_error_for_overall)
    for vals in (dens_vals, mass_vals, mu_vals, young_vals):
        m = mean_or_none(vals)
        if m is not None:
            overall_parts.append(m)
    overall = sum(overall_parts) / len(overall_parts) if overall_parts else None

    unit_keys = sorted({(r["object"], r["condition"]) for r in rows})
    token_rows = []
    for obj, cid in unit_keys:
        sample = next(r for r in rows if r["object"] == obj and r["condition"] == cid)
        token_rows.append(sample)

    def mean_token(field: str) -> float | None:
        vals = [float(r[field]) for r in token_rows if r.get(field) not in (None, "", "unavailable")]
        return sum(vals) / len(vals) if vals else None

    total_calls = sum(int(r.get("model_call_count") or 0) for r in token_rows)

    return {
        "condition": rows[0]["condition"] if rows else None,
        "condition_name": rows[0]["condition_name"] if rows else None,
        "material_accuracy": material_acc,
        "density_error": mean_or_none(dens_vals),
        "mass_error": mean_or_none(mass_vals),
        "mu_error": mean_or_none(mu_vals),
        "youngs_error": mean_or_none(young_vals),
        "overall_error": overall,
        "overall_error_definition": (
            "mean of property mean-errors; "
            "material uses match error (0/1); numeric properties use relative error; "
            "prediction_missing (target+GT, null pred) contributes error=1.0; "
            "gt_unavailable (e.g. Young's) is excluded"
        ),
        "n_evaluated_properties": n_evaluated,
        "n_prediction_failures": n_pred_fail,
        "n_gt_unavailable": n_gt_unavailable,
        "avg_input_tokens": mean_token("input_tokens"),
        "avg_output_tokens": mean_token("output_tokens"),
        "avg_total_tokens": mean_token("total_tokens"),
        "total_model_calls": total_calls,
        "n_units": len(unit_keys),
    }
