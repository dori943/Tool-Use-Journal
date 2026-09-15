"""Evaluate cached D_SIM predictions per condition/repeat.

This is a simulator-only report.  It loads ``sim_gt.yaml`` here (evaluator
process), normalizes downstream outputs (including pose-pair predictions) to
the registered unit IDs, and never modifies raw predictions or the Real-5 table.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

try:
    from .score_sim_d import score
except ImportError:
    from score_sim_d import score


SUPPORTED = {
    "name_only": ["Mass_Acc", "Crit"],
    "affordance_labels": ["Mass_Acc", "Crit"],
    "siphy_adopted": ["Mass_Acc", "Crit"],
    "geometric_grounding": ["Mass_Acc", "Clearance_RelErr", "Feasibility_Acc", "DA", "Crit"],
    "ours_full": ["Mass_Acc", "Suction_Acc", "Suction_PF", "Clearance_RelErr", "Feasibility_Acc", "DA", "Crit"],
}


def _normalize(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        metric = row.get("metric")
        if metric in {"Suction_Acc", "Suction_PF"}:
            # Adapters return a pose pair keyed as pose_A/pose_B. Expand it to
            # the frozen pose IDs required by the registered scorers.
            value = row.get("value")
            if not isinstance(value, dict) or set(value) != {"pose_A", "pose_B"}:
                continue
            for suffix in ("pose_A", "pose_B"):
                out.append({**row, "input_id": f'{row["sample_id"]}_{suffix}',
                            "value": value[suffix], "pose_id": f'{row["sample_id"]}_{suffix}'})
            continue
        if metric == "Clearance_RelErr":
            out.append({**row, "input_id": f"{row['sample_id']}::clearance"})
        elif metric == "DA":
            out.append({**row, "input_id": f"{row['sample_id']}::decision"})
        elif metric == "Feasibility_Acc" and isinstance(row.get("value"), dict):
            for ee_id, value in row["value"].items():
                out.append({**row, "input_id": f"{row['sample_id']}::{ee_id}", "value": value})
        else:
            out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line]
    root = args.gt.parent
    repeats = sorted({int(r["repeat_id"]) for r in rows})
    per_repeat = []
    for condition, metrics in SUPPORTED.items():
        for repeat_id in repeats:
            selected = _normalize([r for r in rows if r.get("condition") == condition and int(r.get("repeat_id", -1)) == repeat_id])
            if not selected:
                for metric in metrics:
                    per_repeat.append({"condition": condition, "repeat_id": repeat_id,
                                       "metric": metric, "status": "INCOMPLETE", "score": None,
                                       "reason": "CONDITION_PREDICTION_MISSING"})
                continue
            result = score(root, _write_temp(args.output.parent, condition, repeat_id, selected))
            for metric in metrics:
                item = result[metric]
                per_repeat.append({"condition": condition, "repeat_id": repeat_id, "metric": metric, **item})
    args.output.write_text(json.dumps(per_repeat, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = []
    for condition in SUPPORTED:
        for metric in SUPPORTED[condition]:
            vals = [x["score"] for x in per_repeat if x["condition"] == condition and x["metric"] == metric and x.get("score") is not None]
            summary.append({"condition": condition, "metric": metric,
                            "status": "COMPLETE" if len(vals) == len(repeats) else "INCOMPLETE",
                            "mean": statistics.mean(vals) if vals and len(vals) == len(repeats) else None,
                            "std_ddof1": statistics.stdev(vals) if len(vals) > 1 and len(vals) == len(repeats) else None,
                            "n_repeats": len(vals), "expected_repeats": len(repeats)})
    (args.output.parent / "d_sim_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_temp(parent: Path, condition: str, repeat_id: int, rows: list[dict]) -> Path:
    path = parent / f".score_{condition}_{repeat_id}.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
