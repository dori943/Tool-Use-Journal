"""Legacy Table 3 runner for the pre-Static-Real-5 trajectory path.

The current matrix has four READY cells backed by the separate static runner
and evaluation bundle. This legacy runner refuses to execute those cells rather
than silently using its incompatible fixed-frame/trajectory adapter. Numeric
GT is opened only by scoring.py to create evaluator-owned snapshots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scoring import (prediction_key, sha256, snapshot_d_gt, snapshot_gt,
                     validate_prediction_envelope)  # noqa: E402

METRICS = ("Mass_MnRE", "Mass_Acc", "Suction_Acc", "Suction_PF",
           "Friction_MAE", "Clearance_RelErr", "Feasibility_Acc", "DA", "Crit")
STATUS_INPUT = {"READY", "NEEDS_DATA", "NEEDS_IMPLEMENTATION", "NOT_APPLICABLE"}
DISPLAY = {"Mass_MnRE": "Mass MnRE↓", "Mass_Acc": "Mass Acc↑",
           "Suction_Acc": "Suction Acc↑", "Suction_PF": "Suction PF↑",
           "Friction_MAE": "Friction MAE↓", "Clearance_RelErr": "Clearance RelErr↓",
           "Feasibility_Acc": "Feas. Acc↑", "DA": "DA↑", "Crit": "Crit↑"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def write_new_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def copy_new(source: Path, target: Path) -> None:
    with target.open("xb") as out:
        out.write(source.read_bytes())


def read_matrix(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6 or len({r["condition_id"] for r in rows}) != 6:
        raise ValueError("CONDITION_MATRIX_NOT_SIX_UNIQUE_ROWS")
    if any(set(METRICS) - set(row) or any(row[m] not in STATUS_INPUT for m in METRICS)
           for row in rows):
        raise ValueError("CONDITION_MATRIX_INVALID_STATUS")
    return rows


def verify_manifest_lineage(manifest: dict, manifest_path: Path, counts: dict,
                            policy_path: Path, gt_path: Path) -> None:
    if (manifest.get("schema_version") != "table3_real5_inputs_v1" or
            counts.get("manifest_path") != manifest_path.relative_to(ROOT).as_posix() or
            len(manifest.get("records", [])) != 25 or
            counts["real5_raw_sequences"] != 25 or counts["real5_raw_objects"] != 5):
        raise ValueError("INPUT_MANIFEST_OR_COUNTS_MISMATCH")
    expected = {"original_manifest": "original_manifest_sha256",
                "audit_input": "audit_sha256", "qc_policy": "qc_policy_sha256",
                "mask_generator": "mask_generator_sha256",
                "friction_tracker": "friction_tracker_sha256"}
    for path_key, hash_key in expected.items():
        source = ROOT / manifest[path_key]
        if sha256(source) != manifest[hash_key]:
            raise ValueError(f"INPUT_LINEAGE_MISMATCH:{path_key}")
    source_manifest = json.loads((ROOT / manifest["original_manifest"]).read_text(encoding="utf-8"))
    archive = ROOT / "data/external/kandukuri_ev_realphys/ev-realphys.tar.gz"
    if (source_manifest.get("archive_sha256") != manifest["archive_sha256"] or
            sha256(archive) != manifest["archive_sha256"]):
        raise ValueError("OFFICIAL_ARCHIVE_SHA_MISMATCH")
    if (sha256(gt_path) != manifest["table5_gt_sha256"] or
            sha256(policy_path) != manifest["qc_policy_sha256"]):
        raise ValueError("GT_OR_POLICY_SHA_MISMATCH")
    seen = set()
    source_mapping = {row["sequence_id"]: (row["object_id"], row["bop_object_id"])
                      for row in source_manifest["sequences"]}
    for row in manifest["records"]:
        sid = row["sequence_id"]
        if (sid in seen or row["dataset_split"] != "test_sliding" or
                "tuna" in row["object_id"].lower() or
                row["fixed_mass_frame"] != 15 or
                source_mapping.get(sid) != (row["object_id"], row["bop_object_id"])):
            raise ValueError(f"BAD_LOCKED_INPUT_ROW:{sid}")
        seen.add(sid)
        crop = row["mass"].get("crop")
        if crop and sha256(ROOT / crop) != row["mass"]["crop_sha256"]:
            raise ValueError(f"INPUT_CROP_SHA_MISMATCH:{sid}")
    if len(seen) != 25:
        raise ValueError("MISSING_LOCKED_INPUT")


def cell_state(matrix_status: str, metric_info: dict) -> tuple[str, str]:
    if matrix_status == "NOT_APPLICABLE":
        return "-", "structurally unsupported or oracle direct error is not estimation"
    if metric_info["status"] in {"BLOCKED", "TARGET_SEMANTICS_MISMATCH"}:
        return "BLOCKED", metric_info.get("missing", metric_info["status"])
    if matrix_status == "NEEDS_DATA":
        return "BLOCKED", "condition-specific independent data or train-only prior missing"
    if matrix_status == "NEEDS_IMPLEMENTATION":
        return "TBD", "condition adapter/scorer not complete"
    if matrix_status == "READY" and metric_info.get("ready_samples", metric_info.get("ready_sequences", 0)) == 0:
        return "BLOCKED", "READY matrix conflicts with zero eligible inputs"
    return "READY", ""


def build_cells(matrix: list[dict], counts: dict, registry: dict) -> list[dict]:
    cells = []
    for condition in matrix:
        for metric in METRICS:
            info = counts["metrics"][metric]
            status, reason = cell_state(condition[metric], info)
            planned = info.get("planned_sequences", info.get("planned_samples", 0))
            ready = info.get("ready_sequences", info.get("ready_samples", 0))
            cells.append({"condition_id": condition["condition_id"],
                          "condition": condition["condition"], "metric": metric,
                          "display": DISPLAY[metric], "benchmark": info["panel"],
                          "planned_n": planned, "ready_n": ready,
                          "unit": registry["metrics"][metric]["unit"],
                          "matrix_status": condition[metric], "status": status,
                          "value": None, "repeat_std_ddof1": None,
                          "object_std_ddof1": None,
                          "coverage": None if status == "-" or not planned else 0.0,
                          "reason": reason})
    return cells


def render_table(cells: list[dict], matrix: list[dict], counts: dict) -> str:
    by_key = {(c["condition_id"], c["metric"]): c for c in cells}
    heading = "| Condition | " + " | ".join(DISPLAY[m] for m in METRICS) + " |"
    divider = "| --- | " + " | ".join("---" for _ in METRICS) + " |"
    rows = ["# ICRA Table 3 — C4 평가 실행 상태", "", heading, divider]
    for condition in matrix:
        values = []
        for metric in METRICS:
            c = by_key[(condition["condition_id"], metric)]
            if c["status"] in ("-", "TBD", "BLOCKED"):
                values.append(c["status"])
            else:
                value = c["value"]
                if value is None:
                    values.append("TBD")
                else:
                    factor = 1 if metric == "Friction_MAE" else 100
                    mean_text = f"{value*factor:.2f}"
                    repeat_std = c.get("repeat_std_ddof1")
                    values.append((f"{mean_text} ± {repeat_std*factor:.2f}"
                                   if repeat_std is not None else f"{mean_text} (std N/A)"))
        rows.append("| " + condition["condition"] + " | " + " | ".join(values) + " |")
    rows += ["", "캡션: R은 공식 EV-RealPhys test_sliding의 5개 실물·25개 시퀀스 "
             f"(현재 질량 승인 {counts['metrics']['Mass_MnRE']['ready_sequences']}/25, "
             f"마찰 승인 {counts['metrics']['Friction_MAE']['ready_sequences']}/25); "
             "D는 별도 조작·기하 패널(현재 0개)이다. Mass MnRE는 무차원 상대오차를 %로, "
             "Friction MAE는 무차원 결합 마찰계수 절대오차로, Clearance RelErr는 상대오차를 %, "
             "accuracy/PF/DA/Crit은 %로 표시한다. 객체별 시퀀스 예측 중앙값 뒤 객체 오차를 "
             "동일 가중 평균한다. stochastic 조건의 표시 형식은 전체 평가 5회의 macro 점수 "
             "mean ± run 간 표본 std(ddof=1)이며, deterministic 조건은 1회·std N/A다. "
             "객체 간 std와 시퀀스 간 예측 std는 별도 필드이고 반복은 객체 수를 늘리지 않는다. "
             "‘-’는 구조적 미지원, TBD는 미실행/구현 미완료, BLOCKED는 입력·독립 GT·목표 의미 "
             "게이트 미통과다. 이 legacy 경로에서는 호환 adapter가 없어 점수를 산출하지 않았다.", ""]
    return "\n".join(rows)


def load_existing_predictions(path: Path) -> list[dict]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            validate_prediction_envelope(row)
            key = prediction_key(row)
            if key in seen:
                raise ValueError(f"DUPLICATE_CHECKPOINT_PREDICTION:{key}")
            seen.add(key)
            rows.append(row)
    return rows


def resume_check(run_dir: Path, current_config: dict, source_manifest: Path) -> dict:
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if (checkpoint["config_sha256"] != canonical_sha(current_config) or
            checkpoint["config_sha256"] != canonical_sha(saved_config) or
            checkpoint["manifest_sha256"] != sha256(source_manifest) or
            checkpoint["manifest_sha256"] != sha256(run_dir / "manifest.json")):
        raise ValueError("RESUME_CONFIG_OR_MANIFEST_HASH_MISMATCH")
    predictions = load_existing_predictions(run_dir / "raw_predictions.jsonl")
    if sorted(str(prediction_key(r)) for r in predictions) != sorted(checkpoint["completed_keys"]):
        raise ValueError("RESUME_CHECKPOINT_KEYS_MISMATCH")
    if sha256(run_dir / "raw_predictions.jsonl") != checkpoint["prediction_file_sha256"]:
        raise ValueError("RESUME_PREDICTION_SHA_MISMATCH")
    if checkpoint["state"] in ("BLOCKED", "COMPLETE"):
        return {"status": "UNCHANGED", "run_dir": str(run_dir), "prediction_count": len(predictions)}
    raise RuntimeError("resume of an active READY backend is not enabled; no predictions were reused")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output/c4_table3")
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    protocol_dir = ROOT / "experiments/table3_protocol"
    prep = protocol_dir / "data_preparation"
    matrix_path = protocol_dir / "condition_matrix.csv"
    registry_path = protocol_dir / "metric_registry.yaml"
    manifest_path = prep / "real5_input_manifest.json"
    counts_path = prep / "metric_sample_counts.json"
    policy_path = prep / "qc_policy.yaml"
    gt_path = ROOT / "configs/c4_real5_gt.json"
    d_gt_path = prep / "panel_d_manifest.yaml"
    matrix = read_matrix(matrix_path)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = json.loads(counts_path.read_text(encoding="utf-8"))
    verify_manifest_lineage(manifest, manifest_path, counts, policy_path, gt_path)
    if len(registry["metrics"]) != len(METRICS) or set(registry["metrics"]) != set(METRICS):
        raise ValueError("METRIC_REGISTRY_MISMATCH")
    config = {"schema_version": "c4_table3_execution_v1",
              "protocol_version": registry["protocol_version"],
              "repeat_override": "latest user instruction: 5 full evaluations for stochastic conditions; protocol file retains earlier 3",
              "stochastic_full_repeats": 5, "requested_seeds": [100, 101, 102, 103, 104],
              "deterministic_full_repeats": 1, "deterministic_only_if_proven": True,
              "model_requested": args.model, "temperature_requested": args.temperature,
              "seed_support": "record per API call; no call implies null; temperature 0 does not guarantee determinism",
              "matrix_sha256": sha256(matrix_path), "registry_sha256": sha256(registry_path),
              "manifest_sha256": sha256(manifest_path), "metric_counts_sha256": sha256(counts_path),
              "qc_policy_sha256": sha256(policy_path), "gt_source_sha256": sha256(gt_path),
              "d_gt_source_sha256": sha256(d_gt_path),
              "robot_spec_sha256": sha256(ROOT / "configs/robot_spec.json"),
              "runner_sha256": sha256(Path(__file__)),
              "scorer_sha256": sha256(HERE / "scoring.py")}
    if args.resume is not None:
        print(json.dumps(resume_check(args.resume.resolve(), config, manifest_path), ensure_ascii=False, indent=2))
        return 0
    cells = build_cells(matrix, counts, registry)
    ready = [c for c in cells if c["status"] == "READY"]
    if ready:
        # Existing run_inference.py couples mass and friction failures and does
        # not retain raw VLM responses. It cannot safely execute a Table 3 cell.
        raise RuntimeError("READY_ADAPTER_NOT_IMPLEMENTED: use experiments/c4_static_real5/evaluate_static.py bundle for Static Real-5; legacy trajectory adapter is incompatible")
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = next((out_root / (stamp if i == 0 else f"{stamp}_{i:03d}"))
                   for i in range(1000) if not (out_root / (stamp if i == 0 else f"{stamp}_{i:03d}")).exists())
    run_dir.mkdir(exist_ok=False)
    (run_dir / "gt_snapshot").mkdir()
    write_new_json(run_dir / "config.json", config)
    copy_new(manifest_path, run_dir / "manifest.json")
    copy_new(matrix_path, run_dir / "condition_matrix.csv")
    copy_new(registry_path, run_dir / "metric_registry.yaml")
    copy_new(policy_path, run_dir / "qc_policy.yaml")
    # Evaluator-owned GT boundary. No numeric GT enters inference scheduling.
    snapshot_gt(gt_path, manifest["table5_gt_sha256"], run_dir / "gt_snapshot/c4_real5_gt.json")
    snapshot_d_gt(d_gt_path, config["d_gt_source_sha256"], run_dir / "gt_snapshot/panel_d_manifest.yaml")
    with (run_dir / "raw_predictions.jsonl").open("x", encoding="utf-8"):
        pass
    with (run_dir / "oracle_inputs.jsonl").open("x", encoding="utf-8"):
        pass
    write_new_json(run_dir / "failures.json", [])
    write_new_json(run_dir / "per_object_summary.json", [])
    write_new_json(run_dir / "per_repeat_summary.json", [])
    write_new_json(run_dir / "prediction_schema.json", {
        "required": ["condition_id", "metric", "input_id", "object_id", "repeat_id",
                     "requested_seed", "seed_sent_to_api", "model_version", "prompt_sha256",
                     "temperature", "started_at_utc", "finished_at_utc", "elapsed_s",
                     "raw_response", "value", "unit", "parse_status", "failure_reason", "input_sha256"],
        "oracle": "GT numeric inputs, when permitted, are evaluator-owned and logged only in oracle_inputs.jsonl",
        "note": "No prediction records were generated in this blocked run."})
    write_new_json(run_dir / "blocked_cells.json", [c for c in cells if c["status"] == "BLOCKED"])
    overall = {"run_id": run_dir.name, "status": "BLOCKED_NO_READY_CELLS",
               "ready_cell_count": 0, "executed_cell_count": 0, "api_call_count": 0,
               "prediction_count": 0, "stochastic_repeat_plan": 5,
               "actual_repeat_count": 0, "actual_seed_sent_to_api": None,
               "model_version_observed": None, "prompt_sha256_observed": None,
               "repeat_std_ddof1": None, "object_std_ddof1": None,
               "std_note": "No run scores; repeat std N/A. Object/sequence std are distinct statistics, never additional independent objects.",
               "official_real5_score": "INCOMPLETE", "cells": cells,
               "excluded_or_blocked_inputs": {
                   "mass": counts["mass_input_status_counts"],
                   "friction": counts["friction_input_status_counts"]},
               "historical_c4_result_is_not_table3": True}
    write_new_json(run_dir / "overall_summary.json", overall)
    fields = ("condition_id", "condition", "metric", "display", "benchmark", "planned_n",
              "ready_n", "unit", "matrix_status", "status", "value", "repeat_std_ddof1",
              "object_std_ddof1", "coverage", "reason")
    with (run_dir / "table3.csv").open("x", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cells)
    with (run_dir / "table3.md").open("x", encoding="utf-8") as out:
        out.write(render_table(cells, matrix, counts))
    with (run_dir / "run.log").open("x", encoding="utf-8") as out:
        out.write(f"{now_utc()} cmd={' '.join(sys.argv)}\n")
        out.write(f"{now_utc()} config_sha256={canonical_sha(config)} manifest_sha256={sha256(manifest_path)}\n")
        out.write(f"{now_utc()} READY=0 API_CALLS=0 PREDICTIONS=0 STATUS=BLOCKED_NO_READY_CELLS\n")
    write_new_json(run_dir / "checkpoint.json", {
        "state": "BLOCKED", "config_sha256": canonical_sha(config),
        "manifest_sha256": sha256(manifest_path), "completed_keys": [],
        "prediction_file_sha256": sha256(run_dir / "raw_predictions.jsonl")})
    print(json.dumps({"run_dir": str(run_dir), "status": overall["status"],
                      "ready_cells": 0, "api_calls": 0,
                      "table3_md": str(run_dir / "table3.md")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
