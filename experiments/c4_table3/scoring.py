"""Table 3 evaluator. Only this module may load numeric ground truth.

No prediction is generated here. The pure scorers take a frozen expected set,
so missing/failed predictions never shrink a denominator after inference.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


REAL5_METRICS = {"Mass_MnRE", "Friction_MAE"}
PREDICTION_UNITS = {"Mass_MnRE": "kg", "Friction_MAE": "coefficient",
                    "Clearance_RelErr": "mm"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_real5_gt(gt_path: Path, expected_sha256: str) -> dict[str, dict]:
    """Load Table 5 numerics inside the evaluator, after input scheduling."""
    if sha256(gt_path) != expected_sha256:
        raise ValueError("GT_SHA_MISMATCH")
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    source = gt.get("source", {})
    if (source.get("dataset") != "EV-RealPhys" or
            source.get("gt_table") != "Table 5: real-world object properties" or
            source.get("friction_target") != "combined_object_table"):
        raise ValueError("GT_SOURCE_OR_TARGET_MISMATCH")
    rows = gt.get("objects", [])
    by_id = {r["object_id"]: r for r in rows}
    if len(rows) != 5 or len(by_id) != 5 or any("tuna" in oid.lower() for oid in by_id):
        raise ValueError("GT_OBJECT_SET_MISMATCH")
    for row in rows:
        if row["source_trial_count"] != 10:
            raise ValueError("GT_SOURCE_TRIAL_COUNT_MISMATCH")
        for key in ("mass_gt_kg", "friction_gt"):
            value = row[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"INVALID_GT_{key}")
    return by_id


def snapshot_gt(gt_path: Path, expected_sha256: str, target: Path) -> None:
    load_real5_gt(gt_path, expected_sha256)
    with target.open("xb") as out:
        out.write(gt_path.read_bytes())


def snapshot_d_gt(gt_path: Path, expected_sha256: str, target: Path) -> None:
    """Evaluator-only D GT validation; the empty panel remains blocked."""
    if sha256(gt_path) != expected_sha256:
        raise ValueError("D_GT_SHA_MISMATCH")
    panel = yaml.safe_load(gt_path.read_text(encoding="utf-8"))
    if panel.get("status") != "BLOCKED" or panel.get("samples"):
        raise ValueError("D_GT_CHANGED_WITHOUT_NEW_EVALUATOR")
    with target.open("xb") as out:
        out.write(gt_path.read_bytes())


def safe_input(row: dict, metric: str) -> dict:
    """Positive allowlist: no scene_gt, measured mass/friction or oracle value."""
    if metric == "Mass_MnRE":
        allowed = row["mass"]
        return {"input_id": row["sequence_id"], "object_id": row["object_id"],
                "frame": row["fixed_mass_frame"], "rgb": allowed["rgb"],
                "depth": allowed["depth"], "crop": allowed["crop"],
                "crop_sha256": allowed["crop_sha256"]}
    if metric == "Friction_MAE":
        return {"input_id": row["sequence_id"], "object_id": row["object_id"],
                "dataset_split": row["dataset_split"],
                "sequence_id": row["sequence_id"], "reference_frame": row["fixed_mass_frame"]}
    raise ValueError(f"no R input adapter for {metric}")


def prediction_key(row: dict) -> tuple[str, str, str, int]:
    return (row["condition_id"], row["metric"], row["input_id"], int(row["repeat_id"]))


def validate_prediction_envelope(row: dict, expected_unit: str | None = None) -> None:
    required = ("condition_id", "metric", "input_id", "object_id", "repeat_id",
                "requested_seed", "seed_sent_to_api", "model_version", "prompt_sha256",
                "temperature", "started_at_utc", "finished_at_utc", "elapsed_s",
                "raw_response", "value", "unit", "parse_status", "failure_reason",
                "input_sha256")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"PREDICTION_SCHEMA_MISSING:{','.join(missing)}")
    if row["parse_status"] not in ("OK", "FAILED"):
        raise ValueError("INVALID_PARSE_STATUS")
    if expected_unit is not None and row["unit"] != expected_unit:
        raise ValueError("PREDICTION_UNIT_MISMATCH")
    if row["parse_status"] == "OK":
        value = row["value"]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("NONFINITE_PREDICTION")
        if row["failure_reason"] is not None:
            raise ValueError("SUCCESS_WITH_FAILURE_REASON")
    elif not row["failure_reason"]:
        raise ValueError("FAILED_WITHOUT_REASON")
    if row["elapsed_s"] < 0:
        raise ValueError("NEGATIVE_ELAPSED_TIME")


def _unique_predictions(rows: list[dict], key_field: str = "input_id") -> dict[str, dict]:
    indexed = {}
    for row in rows:
        key = row[key_field]
        if key in indexed:
            raise ValueError(f"DUPLICATE_PREDICTION:{key}")
        indexed[key] = row
    return indexed


def _valid_numeric(row: dict | None, *, positive: bool = False) -> float | None:
    if row is None or row.get("parse_status") != "OK":
        return None
    value = row.get("value")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    if positive and value <= 0:
        return None
    return float(value)


def _sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _object_macro(results: list[dict]) -> tuple[float | None, float | None]:
    values = [r["score"] for r in results if r["score"] is not None]
    if len(values) != len(results) or not values:
        return None, None
    return statistics.mean(values), _sample_std(values)


def score_real5(metric: str, expected: list[dict], predictions: list[dict],
                gt_by_id: dict[str, dict], *, object_aggregate: str = "median") -> dict:
    if metric not in REAL5_METRICS or object_aggregate not in ("median", "mean"):
        raise ValueError("INVALID_REAL5_METRIC_OR_AGGREGATE")
    if object_aggregate == "mean" and metric in REAL5_METRICS:
        aggregation_name = "sensitivity_object_mean_prediction"
    else:
        aggregation_name = "primary_object_median_prediction"
    expected_map = {r["input_id"]: r["object_id"] for r in expected}
    if len(expected_map) != len(expected) or set(expected_map.values()) != set(gt_by_id):
        raise ValueError("EXPECTED_REAL5_SET_INVALID")
    indexed = _unique_predictions(predictions)
    if any(sid not in expected_map or row["object_id"] != expected_map[sid]
           for sid, row in indexed.items()):
        raise ValueError("PREDICTION_OUTSIDE_FROZEN_SET")
    field = "mass_gt_kg" if metric == "Mass_MnRE" else "friction_gt"
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in expected:
        grouped[row["object_id"]].append(row)
    object_rows = []
    missing = []
    for oid in gt_by_id:
        valid_values = []
        expected_rows = grouped[oid]
        for item in expected_rows:
            sid = item["input_id"]
            pred = indexed.get(sid)
            value = _valid_numeric(pred, positive=True)
            if value is None:
                missing.append(sid)
            else:
                valid_values.append(value)
        aggregate = (statistics.median(valid_values) if object_aggregate == "median"
                     else statistics.mean(valid_values)) if valid_values else None
        gt_value = float(gt_by_id[oid][field])
        error = None if aggregate is None else abs(aggregate - gt_value)
        if error is not None and metric == "Mass_MnRE":
            error /= gt_value
        object_rows.append({"object_id": oid, "expected_sequences": len(expected_rows),
                            "valid_sequences": len(valid_values), "prediction_aggregate": aggregate,
                            "sequence_prediction_std_ddof1": _sample_std(valid_values),
                            "score": error})
    # A continuous metric is withheld if any locked input fails, even when
    # every object has at least one successful sequence.
    complete = not missing and all(row["valid_sequences"] for row in object_rows)
    macro, object_std = _object_macro(object_rows) if complete else (None, None)
    return {"metric": metric, "status": "COMPLETE" if complete else "INCOMPLETE",
            "aggregation": aggregation_name, "score": macro,
            "object_error_std_ddof1": object_std,
            "expected_sequences": len(expected), "valid_sequences": len(expected)-len(missing),
            "failure_count": len(missing), "coverage": (len(expected)-len(missing))/len(expected),
            "missing_input_ids": missing,
            "missing_objects": [r["object_id"] for r in object_rows if r["valid_sequences"] == 0],
            "per_object": object_rows}


def score_binary_macro(expected: list[dict], predictions: list[dict]) -> dict:
    """GT binary labels must come from independent D measurements upstream."""
    indexed = _unique_predictions(predictions)
    if len({r["unit_id"] for r in expected}) != len(expected):
        raise ValueError("DUPLICATE_EXPECTED_UNIT")
    valid_ids = {r["unit_id"] for r in expected}
    if set(indexed) - valid_ids:
        raise ValueError("PREDICTION_OUTSIDE_FROZEN_SET")
    grouped = defaultdict(list)
    valid_predictions = 0
    for row in expected:
        gt = row["gt_label"]
        if type(gt) is not bool:
            raise ValueError("MISSING_OR_NONINDEPENDENT_BINARY_GT")
        pred = indexed.get(row["unit_id"])
        predicted = pred.get("value") if pred and pred.get("parse_status") == "OK" else None
        valid_predictions += type(predicted) is bool
        correct = int(type(predicted) is bool and predicted == gt)
        grouped[row["object_id"]].append(correct)
    per_object = [{"object_id": oid, "denominator": len(scores),
                   "correct": sum(scores), "score": statistics.mean(scores)}
                  for oid, scores in grouped.items()]
    macro, between = _object_macro(per_object) if per_object else (None, None)
    return {"status": "COMPLETE" if per_object else "BLOCKED", "score": macro,
            "object_accuracy_std_ddof1": between, "denominator": len(expected),
            "correct": sum(r["correct"] for r in per_object),
            "valid_predictions": valid_predictions,
            "failed_prediction_count": len(expected)-valid_predictions,
            "coverage": valid_predictions/len(expected) if expected else 0,
            "per_object": per_object}


def score_mass_acc(samples: list[dict], predictions: list[dict],
                   payload_kg: dict[str, float]) -> dict:
    """Each independently weighed sample contributes all three EE decisions."""
    if set(payload_kg) != {"2F", "3F", "vac"}:
        raise ValueError("EE_PAYLOAD_SET_MISMATCH")
    indexed = _unique_predictions(predictions)
    if set(indexed) - {s["sample_id"] for s in samples}:
        raise ValueError("PREDICTION_OUTSIDE_FROZEN_SET")
    units, decisions = [], []
    for sample in samples:
        gt = sample.get("mass_gt_kg")
        if (not isinstance(gt, (int, float)) or not math.isfinite(gt) or gt <= 0 or
                sample.get("payload_applicability_verified") is not True or
                sample.get("gt_source") != "independent_scale"):
            raise ValueError("MISSING_INDEPENDENT_MASS_OR_PAYLOAD_APPLICABILITY")
        predicted_mass = _valid_numeric(indexed.get(sample["sample_id"]), positive=True)
        for ee_id, payload in payload_kg.items():
            if not isinstance(payload, (int, float)) or payload <= 0:
                raise ValueError("INVALID_EE_PAYLOAD")
            unit = f"{sample['sample_id']}::{ee_id}"
            units.append({"unit_id": unit, "object_id": sample["object_id"],
                          "gt_label": bool(gt < payload)})
            decisions.append({"input_id": unit,
                              "parse_status": "OK" if predicted_mass is not None else "FAILED",
                              "value": bool(predicted_mass < payload) if predicted_mass is not None else None})
    return score_binary_macro(units, decisions)


def score_independent_trials(expected: list[dict], predictions: list[dict]) -> dict:
    """Suction/Feasibility: five physical trials, >=4 successes, fixed units."""
    units = []
    for row in expected:
        trials = row.get("trial_success")
        if (row.get("gt_source") != "independent_physical_trials" or
                not isinstance(trials, list) or len(trials) != 5 or
                any(type(t) is not bool for t in trials)):
            raise ValueError("MISSING_INDEPENDENT_FIVE_TRIAL_GT")
        units.append({"unit_id": row["unit_id"], "object_id": row["object_id"],
                      "gt_label": sum(trials) >= 4})
    return score_binary_macro(units, predictions)


def score_suction_pf(pairs: list[dict], predictions: list[dict]) -> dict:
    indexed = _unique_predictions(predictions)
    grouped = defaultdict(list)
    required_pose_ids = set()
    valid_pose_predictions = 0
    for pair in pairs:
        poses = pair["poses"]
        if len(poses) != 2 or {p["pose_id"] for p in poses} & required_pose_ids:
            raise ValueError("INVALID_OR_DUPLICATE_POSE_PAIR")
        required_pose_ids.update(p["pose_id"] for p in poses)
        if any(type(p["gt_label"]) is not bool for p in poses):
            raise ValueError("MISSING_POSE_GT")
        if {p["gt_label"] for p in poses} != {True, False}:
            raise ValueError("PF_PAIR_MUST_CONTRAST_FEASIBLE_AND_INFEASIBLE")
        if any(p.get("gt_source") != "independent_physical_trials" or
               not isinstance(p.get("trial_success"), list) or
               len(p["trial_success"]) != 5 or
               any(type(t) is not bool for t in p["trial_success"]) or
               (sum(p["trial_success"]) >= 4) != p["gt_label"] for p in poses):
            raise ValueError("PF_GT_TRIAL_EVIDENCE_MISSING")
        both = True
        for pose in poses:
            pred = indexed.get(pose["pose_id"])
            value = pred.get("value") if pred and pred.get("parse_status") == "OK" else None
            valid_pose_predictions += type(value) is bool
            both &= type(value) is bool and value == pose["gt_label"]
        grouped[pair["object_id"]].append(int(both))
    if set(indexed) - required_pose_ids:
        raise ValueError("PREDICTION_OUTSIDE_FROZEN_SET")
    per_object = [{"object_id": oid, "pairs": len(scores), "correct_pairs": sum(scores),
                   "score": statistics.mean(scores)} for oid, scores in grouped.items()]
    macro, between = _object_macro(per_object) if per_object else (None, None)
    return {"status": "COMPLETE" if per_object else "BLOCKED", "score": macro,
            "object_accuracy_std_ddof1": between, "denominator_pairs": len(pairs),
            "correct_pairs": sum(r["correct_pairs"] for r in per_object),
            "valid_pose_predictions": valid_pose_predictions,
            "pose_coverage": valid_pose_predictions/(2*len(pairs)) if pairs else 0,
            "per_object": per_object}


def score_clearance(expected: list[dict], predictions: list[dict],
                    *, denominator_floor_mm: float = 5.0) -> dict:
    indexed = _unique_predictions(predictions)
    if set(indexed) - {r["unit_id"] for r in expected} or denominator_floor_mm != 5.0:
        raise ValueError("CLEARANCE_SET_OR_PROTOCOL_FLOOR_MISMATCH")
    grouped = defaultdict(list)
    missing = []
    for row in expected:
        gt = row["gt_margin_mm"]
        if (not isinstance(gt, (int, float)) or not math.isfinite(gt) or
                row.get("gt_source") != "independent_caliper_or_3d" or
                not row.get("reference_frame")):
            raise ValueError("MISSING_INDEPENDENT_CLEARANCE_GT")
        pred = _valid_numeric(indexed.get(row["unit_id"]))
        if pred is None:
            missing.append(row["unit_id"])
        else:
            grouped[row["object_id"]].append(abs(pred - gt)/max(abs(gt), denominator_floor_mm))
    object_ids = {r["object_id"] for r in expected}
    per_object = [{"object_id": oid, "valid_samples": len(grouped[oid]),
                   "score": statistics.median(grouped[oid]) if grouped[oid] else None}
                  for oid in sorted(object_ids)]
    complete = bool(expected) and not missing
    macro, between = _object_macro(per_object) if complete else (None, None)
    return {"status": "COMPLETE" if complete else "INCOMPLETE", "score": macro,
            "object_error_std_ddof1": between, "denominator": len(expected),
            "coverage": (len(expected)-len(missing))/len(expected) if expected else 0,
            "missing_input_ids": missing, "per_object": per_object}


def score_da(expected: list[dict], predictions: list[dict]) -> dict:
    units = []
    for row in expected:
        allowed = row["allowed_ee_ids"]
        if (not isinstance(allowed, list) or any(x not in ("2F", "3F", "vac") for x in allowed) or
                not row.get("independent_trial_and_constraint_evidence_ids") or
                row.get("objective_frozen") is not True):
            raise ValueError("MISSING_INDEPENDENT_ALLOWED_SET")
        pred = next((p for p in predictions if p["input_id"] == row["unit_id"]), None)
        value = pred.get("value") if pred and pred.get("parse_status") == "OK" else None
        correct = (value in allowed) if allowed else (value == "NO_FEASIBLE_EE")
        units.append({"unit_id": row["unit_id"], "object_id": row["object_id"],
                      "gt_label": True, "selected_correct": bool(correct)})
    _unique_predictions(predictions)
    if set(p["input_id"] for p in predictions) - {r["unit_id"] for r in expected}:
        raise ValueError("PREDICTION_OUTSIDE_FROZEN_SET")
    return score_binary_macro(units, [{"input_id": r["unit_id"],
                                       "parse_status": "OK", "value": r["selected_correct"]}
                                      for r in units])


def score_critical_mass(expected: list[dict], predictions: list[dict],
                        *, threshold_kg: float = 0.5, half_width_kg: float = 0.05) -> dict:
    if threshold_kg != 0.5 or half_width_kg != 0.05:
        raise ValueError("CRIT_PROTOCOL_MISMATCH")
    subset = []
    for row in expected:
        mass = row["mass_gt_kg"]
        if (not isinstance(mass, (int, float)) or not math.isfinite(mass) or mass <= 0 or
                row.get("gt_source") != "independent_scale"):
            raise ValueError("MISSING_INDEPENDENT_MASS_GT")
        if abs(mass - threshold_kg) <= half_width_kg:
            subset.append({"unit_id": row["unit_id"], "object_id": row["object_id"],
                           "gt_label": mass < threshold_kg})
    if not subset:
        return {"status": "BLOCKED", "score": None, "denominator": 0, "reason": "NO_CRITICAL_SAMPLES"}
    pred_index = _unique_predictions(predictions)
    converted = []
    for row in subset:
        raw = pred_index.get(row["unit_id"])
        mass_est = _valid_numeric(raw, positive=True)
        converted.append({"input_id": row["unit_id"], "parse_status": "OK" if mass_est is not None else "FAILED",
                          "value": mass_est < threshold_kg if mass_est is not None else None})
    return score_binary_macro(subset, converted)


def score_repeats(repeat_results: list[dict], expected_runs: int,
                  *, deterministic: bool = False) -> dict:
    expected = 1 if deterministic else expected_runs
    ids = [row["repeat_id"] for row in repeat_results]
    if sorted(ids) != list(range(expected)):
        return {"status": "INCOMPLETE", "score_mean": None, "repeat_std_ddof1": None,
                "repeat_count": len(repeat_results), "expected_repeat_count": expected}
    scores = [row.get("score") if row.get("status") == "COMPLETE" else None
              for row in repeat_results]
    if any(s is None or not math.isfinite(s) for s in scores):
        return {"status": "INCOMPLETE", "score_mean": None, "repeat_std_ddof1": None,
                "repeat_count": len(scores), "expected_repeat_count": expected}
    return {"status": "COMPLETE", "score_mean": statistics.mean(scores),
            "repeat_std_ddof1": _sample_std(scores),
            "repeat_std_display": "N/A" if deterministic else _sample_std(scores),
            "repeat_count": len(scores), "expected_repeat_count": expected,
            "std_meaning": "whole-evaluation run macro scores, ddof=1; not independent objects"}
