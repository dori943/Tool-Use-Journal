"""Read-only, GT-free integrity audit for a completed Static Real-5 inference run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from run_locked_inference import (EXPECTED_CONFIG_SHA256, EXPECTED_MANIFEST_SHA256,
                                  METRICS, REPEAT_IDS, prediction_key, preflight, sha)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    run = args.run_dir.resolve()
    rows, cfg, hashes = preflight(args.manifest.resolve(), args.config.resolve())
    expected_files = ("config_snapshot.yaml", "manifest_snapshot.csv",
                      "condition_matrix_snapshot.csv", "manifest_hash.txt", "prompt_hashes.json",
                      "raw_predictions.jsonl", "parsed_predictions.csv", "shared_predictions.csv",
                      "failures.csv", "request_metadata.jsonl", "checkpoint.jsonl", "run.log",
                      "reproduce_inference.bat", "run_identity.json", "inference_summary.json")
    if any(not (run / name).is_file() for name in expected_files):
        raise ValueError("INFERENCE_ARTIFACT_MISSING")
    if (sha(run / "config_snapshot.yaml") != EXPECTED_CONFIG_SHA256 or
            sha(run / "manifest_snapshot.csv") != EXPECTED_MANIFEST_SHA256 or
            (run / "manifest_hash.txt").read_text(encoding="utf-8").strip() != EXPECTED_MANIFEST_SHA256):
        raise ValueError("SNAPSHOT_HASH_MISMATCH")
    prompt_hashes = json.loads((run / "prompt_hashes.json").read_text(encoding="utf-8"))
    if prompt_hashes != {"Mass_MnRE": cfg["mass_prompt_sha256"],
                          "StaticVisualFriction_MAE": cfg["friction_prompt_sha256"]}:
        raise ValueError("PROMPT_HASH_MISMATCH")
    raw = [json.loads(line) for line in (run / "raw_predictions.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if len(raw) != 500:
        raise ValueError("NOT_500_PREDICTIONS")
    by_key = {}
    manifest_by_id = {r["input_id"]: r for r in rows}
    per_repeat = Counter()
    retry_statuses = Counter()
    attempt_count = 0
    for r in raw:
        key = prediction_key(r["condition"], r["metric"], int(r["repeat_id"]), r["input_id"],
                             r["manifest_hash"], r["prompt_hash"], r["model_version"])
        if key in by_key:
            raise ValueError("DUPLICATE_PREDICTION_KEY")
        by_key[key] = r
        source = manifest_by_id.get(r["input_id"])
        if (source is None or r["object_id"] != source["object_id"] or
                r["sequence_id"] != source["sequence_id"] or
                r["frame_id"] != int(source["frame_id"]) or
                r["input_sha256"] != source["input_sha256"] or
                r["manifest_hash"] != hashes["manifest_sha256"] or
                r["config_hash"] != hashes["config_sha256"]):
            raise ValueError("PREDICTION_INPUT_PROVENANCE_MISMATCH")
        if (r["prediction_id"] != hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest()
                or r["repeat_id"] not in REPEAT_IDS or r["requested_seed"] != r["repeat_id"]-1
                or r["provider_seed_supported"] is not False or
                r["provider_seed_sent"] is not False or
                r["model_version"] != cfg["model"] or
                r["response_model_version"] != cfg["model"] or
                r["temperature"] != cfg["temperature"]):
            raise ValueError("PREDICTION_MODEL_SEED_OR_ID_MISMATCH")
        if r["metric"] == "Mass_MnRE":
            if (r["condition"] != "siphy_adopted" or r["unit"] != "kg" or
                    r["input_modality"] != "static_rgbd" or
                    r["prompt_hash"] != cfg["mass_prompt_sha256"]):
                raise ValueError("MASS_ADAPTER_MISMATCH")
        elif r["metric"] == "StaticVisualFriction_MAE":
            if (r["condition"] != "ours_full" or r["unit"] != "dimensionless" or
                    r["input_modality"] != "static_rgb" or
                    r["estimation_method"] != "visual_property_estimation" or
                    r["prompt_hash"] != cfg["friction_prompt_sha256"]):
                raise ValueError("VISUAL_FRICTION_ADAPTER_MISMATCH")
        else:
            raise ValueError("UNSUPPORTED_METRIC_PREDICTION")
        if (r["parsing_status"] != "PARSED" or r["failure_reason"] is not None or
                type(r["parsed_prediction"]) not in (int, float) or
                not math.isfinite(r["parsed_prediction"]) or
                (r["metric"] == "Mass_MnRE" and r["parsed_prediction"] <= 0) or
                (r["metric"] == "StaticVisualFriction_MAE" and r["parsed_prediction"] < 0) or
                not r["raw_response"] or not r["api_request_id"]):
            raise ValueError("PARSER_OR_RESPONSE_FAILURE")
        if (r["metric"] == "Mass_MnRE" and r["mass_confidence"] is not None and
                not 0 <= r["mass_confidence"] <= 1):
            raise ValueError("MASS_CONFIDENCE_RANGE")
        if (r["metric"] == "StaticVisualFriction_MAE" and r["friction_confidence"] is not None and
                not 0 <= r["friction_confidence"] <= 1):
            raise ValueError("FRICTION_CONFIDENCE_RANGE")
        per_repeat[(r["repeat_id"], r["metric"])] += 1
        attempt_count += len(r["provider_attempts"])
        retry_statuses.update(a["error_status"] for a in r["provider_attempts"] if a["error_status"])
    if (len(by_key) != 500 or any(per_repeat[(i, m)] != 50 for i in REPEAT_IDS for m in METRICS)
            or attempt_count != 501 or retry_statuses != {"INVALID_FORMAT": 1}):
        raise ValueError("CALL_COUNT_OR_RETRY_POLICY_MISMATCH")
    parsed = read_csv(run / "parsed_predictions.csv")
    if len(parsed) != 500 or {r["prediction_id"] for r in parsed} != {r["prediction_id"] for r in raw}:
        raise ValueError("PARSED_CSV_MISMATCH")
    shared = read_csv(run / "shared_predictions.csv")
    mass_by_id = {r["prediction_id"]: r for r in raw if r["metric"] == "Mass_MnRE"}
    if (len(shared) != 500 or Counter(r["condition"] for r in shared) !=
            {"geometric_grounding": 250, "ours_full": 250}):
        raise ValueError("SHARED_MASS_COUNT_MISMATCH")
    alias_keys = set()
    for r in shared:
        source = mass_by_id.get(r["source_prediction_id"])
        key = (r["condition"], r["repeat_id"], r["input_id"])
        if (source is None or r["shared_prediction"] != "True" or
                r["shared_prediction_source"] != "SiPhy" or
                r["independent_model_call"] != "False" or
                r["source_parsing_status"] != source["parsing_status"] or
                r["input_id"] != source["input_id"] or
                int(r["repeat_id"]) != source["repeat_id"] or key in alias_keys):
            raise ValueError("SHARED_MASS_PROVENANCE_MISMATCH")
        alias_keys.add(key)
    requests = [json.loads(line) for line in (run / "request_metadata.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if len(requests) != attempt_count or sum(q["error_status"] == "INVALID_FORMAT" for q in requests) != 1:
        raise ValueError("REQUEST_METADATA_MISMATCH")
    checkpoint = [json.loads(line) for line in (run / "checkpoint.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if len(checkpoint) != 500 or len({tuple(c["key"]) for c in checkpoint}) != 500:
        raise ValueError("CHECKPOINT_COUNT_MISMATCH")
    for c in checkpoint:
        r = by_key.get(tuple(c["key"]))
        if r is None or c["record_sha256"] != hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest():
            raise ValueError("CHECKPOINT_HASH_MISMATCH")
    if read_csv(run / "failures.csv"):
        raise ValueError("FAILURES_PRESENT")
    forbidden_names = ("evaluator_only_gt.yaml", "gt_snapshot.json", "evaluation.json",
                       "result.json", "table3.csv", "table3.md")
    if any((run / name).exists() for name in forbidden_names):
        raise ValueError("GT_OR_SCORE_FILE_IN_INFERENCE_RUN")
    for path in run.iterdir():
        if path.is_file() and path.suffix in (".yaml", ".json", ".jsonl", ".csv", ".txt", ".log", ".bat"):
            data = path.read_text(encoding="utf-8")
            if re.search(r"AIza[0-9A-Za-z_-]{20,}|sk-proj-[0-9A-Za-z_-]{20,}", data):
                raise ValueError("CREDENTIAL_LEAKAGE_IN_OUTPUT")
    summary = json.loads((run / "inference_summary.json").read_text(encoding="utf-8"))
    if (summary["status"] != "INFERENCE_COMPLETE" or summary["prediction_slots_recorded"] != 500 or
            summary["provider_request_attempts"] != 501 or summary["shared_mass_aliases"] != 500 or
            summary["gt_file_access_count"] != 0 or summary["evaluator_executed"] is not False):
        raise ValueError("SUMMARY_MISMATCH")
    audit = {"status": "PASS", "manifest_sha256": hashes["manifest_sha256"],
             "config_sha256": hashes["config_sha256"], "raw_sha256": sha(run / "raw_predictions.jsonl"),
             "checkpoint_sha256": sha(run / "checkpoint.jsonl"),
             "mass_unique_calls": 250, "friction_unique_calls": 250,
             "provider_request_attempts": 501, "retry_causes": dict(retry_statuses),
             "per_repeat_each_metric": 50, "parsed_success": 500, "failed": 0, "missing": 0,
             "shared_mass_aliases": 500, "duplicate_keys": 0,
             "smoke_imported": 0, "trajectory_imported": 0, "gt_access_count": 0,
             "credential_leakage": False, "evaluator_executed": False}
    target = run / "inference_validation.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
