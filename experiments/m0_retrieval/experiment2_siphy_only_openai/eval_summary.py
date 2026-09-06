"""GT evaluation for Experiment 2 OpenAI (same formulas as Exp1 / Exp2 intent).

Uses shared ``gt_loader`` + ``evaluator`` helpers. Does not modify production.
Errors are stored as percentages (0-100) per user Experiment2 OpenAI schema.
Young's GT unavailable rows are excluded from overall.
"""
from __future__ import annotations

from typing import Any

from evaluator import normalize_material, prediction_present, relative_error
from gt_loader import gt_available, gt_value, load_all_gt


PROPERTIES = ("material", "density_kgm3", "mass_kg", "mu", "youngs_gpa")
OVERALL_PROPS = ("material", "density_kgm3", "mass_kg", "mu")  # Young's excluded


def evaluate_prediction(prediction: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    """Return per-property GT/pred/error and overall_error (Young's excluded)."""
    out: dict[str, Any] = {}
    errors_for_overall: list[float] = []

    for prop in PROPERTIES:
        has_gt = gt_available(gt, prop)
        gt_val = gt_value(gt, prop) if has_gt else None
        pred_val = prediction.get(prop)
        has_pred = prediction_present(prop, pred_val)

        row: dict[str, Any] = {
            "gt": gt_val,
            "prediction": pred_val if has_pred else None,
            "evaluated": False,
            "error": None,
            "gt_available": has_gt,
        }

        if not has_gt:
            row["status"] = "gt_unavailable"
        elif not has_pred:
            row["status"] = "prediction_missing"
            row["error"] = 100.0  # full penalty when GT exists but pred missing
            if prop in OVERALL_PROPS:
                errors_for_overall.append(100.0)
        else:
            row["status"] = "evaluated"
            row["evaluated"] = True
            if prop == "material":
                match = normalize_material(pred_val) == normalize_material(gt_val)
                row["material_match"] = bool(match)
                row["error"] = 0.0 if match else 100.0
            else:
                rel = relative_error(pred_val, gt_val)
                row["error"] = None if rel is None else float(rel) * 100.0
            if prop in OVERALL_PROPS and row["error"] is not None:
                errors_for_overall.append(float(row["error"]))

        out[prop] = row

    overall = (
        sum(errors_for_overall) / len(errors_for_overall) if errors_for_overall else None
    )
    return {
        "by_property": out,
        "overall_error": overall,
        "overall_error_definition": (
            "mean of Material/Density/Mass/Mu percent errors; "
            "Material match=0 mismatch=100; numeric=abs(pred-gt)/abs(gt)*100; "
            "Young's excluded; prediction_missing with GT contributes 100"
        ),
        "material_match": out["material"].get("material_match"),
    }


def summary_row_from_unit(result: dict[str, Any], gt: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build one summary.csv row including GT + token fields."""
    pred = result.get("prediction") or {}
    tok = result.get("token_usage") or {}
    ev = result.get("evaluation")
    if ev is None:
        if gt is None:
            gts = load_all_gt()
            gt = gts.get(str(result.get("object"))) or {}
        ev = evaluate_prediction(pred, gt)

    bp = ev["by_property"]

    def _g(prop: str, key: str) -> Any:
        return (bp.get(prop) or {}).get(key)

    return {
        "object": result.get("object"),
        "material_gt": _g("material", "gt"),
        "material_pred": pred.get("material"),
        "material_match": ev.get("material_match"),
        "density_gt": _g("density_kgm3", "gt"),
        "density_pred": pred.get("density_kgm3"),
        "density_error": _g("density_kgm3", "error"),
        "mass_gt": _g("mass_kg", "gt"),
        "mass_pred": pred.get("mass_kg"),
        "mass_error": _g("mass_kg", "error"),
        "mu_gt": _g("mu", "gt"),
        "mu_pred": pred.get("mu"),
        "mu_error": _g("mu", "error"),
        "youngs_gt": _g("youngs_gpa", "gt"),
        "youngs_pred": pred.get("youngs_gpa"),
        "overall_error": ev.get("overall_error"),
        "input_tokens": tok.get("input_tokens"),
        "output_tokens": tok.get("output_tokens"),
        "total_tokens": tok.get("total_tokens"),
        "llm_call_count": tok.get("llm_call_count"),
        "api_attempt_count": tok.get("api_attempt_count"),
        "failure_reason": result.get("failure_reason"),
    }


def aggregate_batch(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean metrics across objects (skip None)."""

    def _mean(key: str) -> float | None:
        vals = []
        for r in summary_rows:
            v = r.get(key)
            if v is None or v == "":
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        return sum(vals) / len(vals) if vals else None

    matches = [r for r in summary_rows if r.get("material_match") is not None]
    n_match = sum(1 for r in matches if r.get("material_match") in (True, "true", "True"))
    mat_acc = (n_match / len(matches)) if matches else None

    return {
        "n_objects": len(summary_rows),
        "material_accuracy": mat_acc,
        "avg_density_error": _mean("density_error"),
        "avg_mass_error": _mean("mass_error"),
        "avg_mu_error": _mean("mu_error"),
        "avg_overall_error": _mean("overall_error"),
        "avg_total_tokens": _mean("total_tokens"),
        "avg_input_tokens": _mean("input_tokens"),
        "avg_output_tokens": _mean("output_tokens"),
        "sum_total_tokens": (
            sum(float(r["total_tokens"]) for r in summary_rows if r.get("total_tokens") not in (None, ""))
            if any(r.get("total_tokens") not in (None, "") for r in summary_rows)
            else None
        ),
    }
