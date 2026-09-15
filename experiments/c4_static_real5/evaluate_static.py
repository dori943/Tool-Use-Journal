"""Separate post-inference evaluator for locked Static Real-5 5-repeat predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/c4_table3"))
from scoring import score_real5  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(rows: list[dict]) -> None:
    expected = {"003_cracker_box", "006_mustard_bottle", "019_pitcher_base",
                "021_bleach_cleanser", "025_mug"}
    if (len(rows) != 50 or len({r["input_id"] for r in rows}) != 50
            or set(r["object_id"] for r in rows) != expected
            or set(Counter(r["object_id"] for r in rows).values()) != {10}
            or set(Counter(r["sequence_id"] for r in rows).values()) != {2}
            or any(r["mass_qc_status"] not in ("APPROVED", "WARNING") or
                   r["friction_qc_status"] not in ("APPROVED", "WARNING") for r in rows)):
        raise ValueError("STATIC_MANIFEST_NOT_LOCKED_5x5x2")


def load_gt(path: Path) -> dict:
    gt = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (gt.get("benchmark") != "C4 Static Real-5 evaluator-only"
            or gt["source"].get("gt_table") != "Table 5: real-world object properties"
            or gt["source"].get("friction_target") != "combined_object_table"):
        raise ValueError("GT_SOURCE_MISMATCH")
    by_id = {}
    for r in gt["objects"]:
        if r["source_trial_count"] != 10 or not r["mass_gt_kg"] > 0 or not r["combined_friction_gt"] > 0:
            raise ValueError("INVALID_TABLE5_GT")
        by_id[r["object_id"]] = {"mass_gt_kg": r["mass_gt_kg"],
                                  "friction_gt": r["combined_friction_gt"]}
    if len(by_id) != 5 or any("tuna" in x for x in by_id):
        raise ValueError("GT_OBJECT_SET_MISMATCH")
    return by_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep-dir", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, nargs=5, required=True,
                    help="exactly five full 50-input repeat files; smoke files are rejected")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run = args.prep_dir.resolve()
    manifest = run / "selected_static_inputs.csv"
    cfg = yaml.safe_load((run / "inference_config.yaml").read_text(encoding="utf-8"))
    if sha(manifest) != cfg["input_manifest_sha256"]:
        raise ValueError("MANIFEST_CONFIG_HASH_MISMATCH")
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    validate_manifest(rows)
    prediction_paths = [p.resolve() for p in args.predictions]
    if (len(set(prediction_paths)) != 5 or
            any("smoke_test" in p.parts or not p.is_file() for p in prediction_paths)):
        raise ValueError("FIVE_COMPLETE_NON_SMOKE_PREDICTION_FILES_REQUIRED")
    # Numeric GT is first loaded here, after all five inference files already exist.
    gt_path = run / "evaluator_only_gt.yaml"
    gt = load_gt(gt_path)
    if set(gt) != {r["object_id"] for r in rows}:
        raise ValueError("GT_MANIFEST_OBJECT_SET_MISMATCH")
    expected = [{"input_id": r["input_id"], "object_id": r["object_id"]} for r in rows]
    expected_by_id = {r["input_id"]: r for r in rows}
    scores = []
    prediction_hashes = []
    for repeat, path in enumerate(prediction_paths):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if (len(records) != 100 or len({(r["metric"], r["input_id"], r["repeat_id"])
                                        for r in records}) != 100):
            raise ValueError("REPEAT_NOT_50_IMAGES_X_2_METRICS")
        for record in records:
            source_row = expected_by_id.get(record["input_id"])
            if (source_row is None or record["object_id"] != source_row["object_id"] or
                    record["sequence_id"] != source_row["sequence_id"] or
                    record["frame_id"] != int(source_row["frame_id"]) or
                    record["input_sha256"] != source_row["input_sha256"]):
                raise ValueError("PREDICTION_INPUT_IDENTITY_OR_HASH_MISMATCH")
            if (record["repeat_id"] != repeat or
                    record["input_manifest_sha256"] != cfg["input_manifest_sha256"] or
                    record["model_version"] != cfg["model"] or
                    record["temperature"] != cfg["temperature"] or
                    record["requested_seed"] != cfg["requested_seed_base"] + repeat or
                    record["provider_seed_sent"] is not False or
                    record["metric"] not in ("Mass_MnRE", "StaticVisualFriction_MAE")):
                raise ValueError("PREDICTION_CONFIG_OR_METRIC_MISMATCH")
            expected_prompt = (cfg["mass_prompt_sha256"] if record["metric"] == "Mass_MnRE"
                               else cfg["friction_prompt_sha256"])
            if record["prompt_sha256"] != expected_prompt:
                raise ValueError("PROMPT_HASH_MISMATCH")
            if record["parsing_status"] == "PARSED" and not record["provider_attempts"]:
                raise ValueError("PARSED_WITHOUT_PROVIDER_EVIDENCE")
        per_metric = {}
        for metric in ("Mass_MnRE", "StaticVisualFriction_MAE"):
            subset = [r for r in records if r["metric"] == metric]
            if len(subset) != 50:
                raise ValueError("METRIC_MUST_HAVE_50_PREDICTIONS")
            want_unit = "kg" if metric == "Mass_MnRE" else "dimensionless_combined_object_table_visual_prior"
            want_cond = "siphy_adopted" if metric == "Mass_MnRE" else "ours_full"
            if any(r["unit"] != want_unit or r["condition_id"] != want_cond for r in subset):
                raise ValueError("UNIT_OR_CONDITION_MISMATCH")
            normalized = [{"input_id": r["input_id"], "object_id": r["object_id"],
                           "parse_status": "OK" if r["parsing_status"] == "PARSED" else "FAILED",
                           "value": r["parsed_prediction"]} for r in subset]
            score = score_real5("Mass_MnRE" if metric == "Mass_MnRE" else "Friction_MAE",
                                expected, normalized, gt)
            per_metric[metric] = score
        scores.append({"repeat_id": repeat, "metrics": per_metric})
        prediction_hashes.append({"repeat_id": repeat, "path": str(path), "sha256": sha(path)})
    overall = {}
    for metric in ("Mass_MnRE", "StaticVisualFriction_MAE"):
        values = [s["metrics"][metric]["score"] for s in scores]
        overall[metric] = {"status": "COMPLETE" if all(v is not None for v in values) else "INCOMPLETE",
                           "mean": statistics.mean(values) if all(v is not None for v in values) else None,
                           "run_std_ddof1": statistics.stdev(values) if all(v is not None for v in values) else None,
                           "std_unit": "five independent full inference repeats",
                           "object_aggregate": "median of valid static image predictions",
                           "macro_aggregate": "equal-weight mean of five object errors",
                           "per_repeat_coverage": [s["metrics"][metric]["coverage"] for s in scores]}
    result = {"benchmark": "C4 Static Real-5 visual-prior", "manifest_sha256": sha(manifest),
              "gt_sha256": sha(gt_path), "prediction_files": prediction_hashes,
              "per_repeat": scores, "overall": overall,
              "trajectory_friction_relation": "separate auxiliary result; not pooled"}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["status"] for k, v in overall.items()}))


if __name__ == "__main__":
    main()
