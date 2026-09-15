"""Generate evaluator-only D_SIM oracle predictions.

This command is intentionally separate from model inference.  It reads
``sim_gt.yaml`` and writes an explicitly tagged oracle stream; the stream is
used only for downstream consistency checks and never for Mass MnRE or
Friction MAE estimator scores.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    from .adapters import OracleAdapter, load_observation
except ImportError:
    from adapters import OracleAdapter, load_observation


METRICS = ("Mass_Acc", "Suction_Acc", "Suction_PF", "Clearance_RelErr", "Feasibility_Acc", "DA", "Crit")


def run(prep: Path, output: Path) -> dict[str, int | str]:
    manifest = yaml.safe_load((prep / "sim_manifest.yaml").read_text(encoding="utf-8"))
    gt = yaml.safe_load((prep / "sim_gt.yaml").read_text(encoding="utf-8"))
    if manifest.get("panel") != "D_SIM" or gt.get("benchmark") != "SIM_PROXY":
        raise RuntimeError("ORACLE_INPUT_PANEL_OR_BENCHMARK_MISMATCH")
    gt_by_sample = {row["sample_id"]: row for row in gt["samples"]}
    oracle = OracleAdapter(oracle=True)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in manifest["samples"]:
        sample_id = sample["sample_id"]
        gt_row = gt_by_sample.get(sample_id)
        if gt_row is None:
            raise RuntimeError(f"ORACLE_GT_SAMPLE_MISSING:{sample_id}")
        obs = load_observation(prep / sample["observation_path"])
        for metric in METRICS:
            row = oracle.predict(observation=obs, gt_row=gt_row, metric=metric)
            row.update({"sample_id": sample_id, "object_id": sample["object_instance_id"],
                        "condition": "gt_numerics",
                        "repeat_id": 0, "requested_seed": None,
                        "provider_seed_supported": False, "provider_seed_sent": False,
                        "model_version": "oracle_evaluator_only", "temperature": None,
                        "raw_response": None, "request_id": None,
                        "parse_status": "OK", "failure_reason": None,
                        "oracle_channel": "evaluator_only",
                        "independent_model_call": False,
                        "started_at_utc": datetime.now(timezone.utc).isoformat()})
            rows.append(row)
    path = output / "oracle_predictions.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    (output / "oracle_run.log").write_text(
        f"benchmark=SIM_PROXY\npanel=D_SIM\nsource=sim_gt.yaml\nmodel_calls=0\nrows={len(rows)}\n",
        encoding="utf-8")
    return {"status": "COMPLETE", "rows": len(rows), "model_calls": 0, "output": str(output.resolve())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(run(args.prep, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
