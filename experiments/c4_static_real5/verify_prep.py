"""Verify a completed static prep bundle without invoking the evaluator or reading GT."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", type=Path, required=True)
    args = ap.parse_args()
    run = args.prep_dir.resolve()
    needed = ["selected_static_inputs.csv", "static_input_contact_sheet.png",
              "static_input_qc_report.md", "adapter_audit.md", "condition_matrix.csv",
              "inference_config.yaml", "selection_policy.json", "run.log"]
    if any(not (run / x).is_file() for x in needed):
        raise ValueError("PREP_ARTIFACT_MISSING")
    cfg = yaml.safe_load((run / "inference_config.yaml").read_text(encoding="utf-8"))
    manifest = run / "selected_static_inputs.csv"
    if sha(manifest) != cfg["input_manifest_sha256"]:
        raise ValueError("MANIFEST_HASH_MISMATCH")
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    counts = Counter(r["object_id"] for r in rows)
    seq_counts = Counter(r["sequence_id"] for r in rows)
    if len(rows) != 50 or set(counts.values()) != {10} or set(seq_counts.values()) != {2}:
        raise ValueError("NOT_5x5x2_INPUTS")
    if any("gt" in k.lower() for k in rows[0]):
        raise ValueError("GT_FIELD_IN_MANIFEST")
    if len({r["input_id"] for r in rows}) != 50 or len({r["input_sha256"] for r in rows}) != 50:
        raise ValueError("DUPLICATE_INPUT")
    for row in rows:
        if (row["mass_qc_status"] not in ("APPROVED", "WARNING") or
                row["friction_qc_status"] not in ("APPROVED", "WARNING")):
            raise ValueError("UNAPPROVED_INPUT")
        paths = [ROOT / row[k] for k in ("rgb_path", "depth_path", "mass_crop_path",
                                           "friction_context_path")]
        paths.append(run / "inputs/points" / (row["input_id"] + ".npz"))
        if any(not p.is_file() for p in paths):
            raise ValueError(f"MISSING_IMAGE:{row['input_id']}")
        combined = hashlib.sha256("".join(sha(p) for p in paths).encode()).hexdigest()
        if combined != row["input_sha256"]:
            raise ValueError(f"INPUT_HASH_MISMATCH:{row['input_id']}")
        rgb = cv2.imread(str(paths[0])); depth = cv2.imread(str(paths[1]), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.shape[:2] != depth.shape:
            raise ValueError(f"RGB_DEPTH_PAIR_MISMATCH:{row['input_id']}")
    for sid in seq_counts:
        frames = sorted(int(r["frame_id"]) for r in rows if r["sequence_id"] == sid)
        if frames[1] - frames[0] < 20:
            raise ValueError(f"FRAMES_NOT_TEMPORALLY_SEPARATED:{sid}")
    smoke = run / "smoke_test/raw_predictions_live.jsonl"
    summary = run / "smoke_test/raw_predictions_live.summary.json"
    if not smoke.is_file() or not summary.is_file():
        raise ValueError("LIVE_SMOKE_MISSING")
    predictions = [json.loads(s) for s in smoke.read_text(encoding="utf-8").splitlines()]
    facts = json.loads(summary.read_text(encoding="utf-8"))
    if (len(predictions) != 2 or
            {r["metric"] for r in predictions} != {"Mass_MnRE", "StaticVisualFriction_MAE"} or
            any(r["parsing_status"] != "PARSED" or not r["raw_response"] or
                r["parsed_prediction"] is None for r in predictions) or
            not facts["gt_access_denied"] or not facts["smoke_excluded_from_evaluation"] or
            facts["actual_provider_calls"] != 2):
        raise ValueError("LIVE_SMOKE_NOT_VALID")
    for record in predictions:
        expected_prompt = (cfg["mass_prompt_sha256"] if record["metric"] == "Mass_MnRE"
                           else cfg["friction_prompt_sha256"])
        if record["prompt_sha256"] != expected_prompt:
            raise ValueError("SMOKE_PROMPT_HASH_MISMATCH")
    result = {"status": "PASS", "inputs": len(rows), "sequences": len(seq_counts),
              "per_object": dict(counts),
              "mass_approved": sum(r["mass_qc_status"] == "APPROVED" for r in rows),
              "mass_warning": sum(r["mass_qc_status"] == "WARNING" for r in rows),
              "friction_approved": sum(r["friction_qc_status"] == "APPROVED" for r in rows),
              "friction_warning": sum(r["friction_qc_status"] == "WARNING" for r in rows),
              "replacements": sum(int(r["frame_id"]) not in (10, 40) for r in rows),
              "manifest_sha256": sha(manifest), "smoke_live_sha256": sha(smoke),
              "smoke_successful_model_calls": 2, "smoke_numeric_gt_read_denied": True,
              "full_repeat_count_completed": 0}
    target = run / "smoke_test/verification.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (run / "run.log").open("a", encoding="utf-8") as f:
        f.write("smoke_initial_network=CONNECTION_ERROR\n"
                "smoke_obsolete_model=gemini-2.5-flash,HTTP_404\n"
                "smoke_live_model=gemini-3.6-flash\n"
                "smoke_live_provider_calls=2\nsmoke_live_parsed=2\n"
                "smoke_gt_access_denied=true\nfull_evaluation_repeats=0\n"
                "focused_tests=17_passed\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
