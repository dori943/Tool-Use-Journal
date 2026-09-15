"""Apply one GT-blind continuity rule to all Real-5 RGB-D centroid tracks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/c4_real5")]
from friction_from_rgbd import track_scene  # noqa: E402
from real_rgbd import load_frame  # noqa: E402


def select_window(positions: np.ndarray, policy: dict) -> tuple[int, int, dict]:
    xy = np.asarray(positions, dtype=float)[:, :2]
    if len(xy) < policy["minimum_intervals"] + 1 or not np.isfinite(xy).all():
        raise ValueError("too few finite positions")
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    jump_intervals = np.flatnonzero(steps > policy["max_step_mm"])
    bounds = []
    start = 0
    for jump in jump_intervals:
        bounds.append((start, int(jump)))
        start = int(jump) + 1
    bounds.append((start, len(xy) - 1))
    candidates = []
    for start, end in bounds:
        for i in range(start, end - policy["stop_consecutive_steps"] + 1):
            if np.all(steps[i:i + policy["stop_consecutive_steps"]] < policy["stop_step_mm"]):
                end = i
                break
        if end - start < policy["minimum_intervals"]:
            continue
        segment = xy[start:end + 1]
        travel = float(np.linalg.norm(segment[-1] - segment[0]))
        if travel < policy["minimum_net_travel_mm"]:
            continue
        direction = (segment[-1] - segment[0]) / travel
        along = (segment - segment[0]) @ direction
        t = np.arange(len(segment)) * policy["dt_s"]
        coefficients = np.polynomial.polynomial.polyfit(t, along, 2)
        rms = float(np.sqrt(np.mean((along -
                                     np.polynomial.polynomial.polyval(t, coefficients)) ** 2)))
        candidates.append({"start": start, "end": end, "intervals": end - start,
                           "net_travel_mm": round(travel, 3),
                           "quadratic_rms_mm": round(rms, 3),
                           "reversal_step_count": int(np.sum(np.diff(along) <= 0))})
    if not candidates:
        raise ValueError("no continuous window passes uniform length/travel constraints")
    selected = min(candidates, key=lambda c: (-c["intervals"], c["quadratic_rms_mm"], c["start"]))
    return selected["start"], selected["end"], {
        "selected": selected, "candidate_count": len(candidates),
        "jump_intervals": jump_intervals.tolist(), "all_candidates": candidates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.run_dir.resolve()
    prior_file = out / "friction_window_audit.json"
    prior = json.loads(prior_file.read_text())
    if prior["numeric_gt_used"] is not False or prior["prior_prediction_error_used"] is not False:
        raise ValueError("v2 tracks not input-only")
    policy_file = ROOT / "experiments/c4_table3/r_friction_window_policy_v3.yaml"
    policy = yaml.safe_load(policy_file.read_text())
    results = []
    for row in prior["rows"]:
        sid = row["sequence_id"]
        scene = ROOT / "data/external/kandukuri_ev_realphys/ev-realphys" / sid
        result = {"sequence_id": sid, "object_id": row["object_id"],
                  "friction_qc_status": "QC_UNREVIEWED",
                  "friction_qc_reason": "human contact/rotation/track review and table gravity level pending",
                  "semantics_status": "TARGET_SEMANTICS_UNVERIFIED",
                  "selected_friction_window": None}
        try:
            if "positions_mm" in row:
                positions = np.asarray(row["positions_mm"], dtype=float)
                result["track_source"] = "input-only v2 RGB-D centroid audit"
            else:
                positions, track_info = track_scene(scene, 15)
                result["track_source"] = "recomputed RGB-D centroid track, reference 15"
                result["track_info"] = track_info
            start, end, details = select_window(positions, policy)
            result["selected_friction_window"] = [start, end]
            result["window_details"] = details
            result["positions_mm"] = positions.tolist()
            strip = np.full((3 * 240, 424, 3), 255, np.uint8)
            for i, frame in enumerate((start, (start + end) // 2, end)):
                bgr, _, _, _ = load_frame(scene, frame)
                panel = cv2.resize(bgr, (424, 240))
                cv2.putText(panel, f"{sid[-6:]} frame {frame}", (5, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
                strip[i * 240:(i + 1) * 240] = panel
            path = out / "friction_window_frames_v3" / f"{sid[-6:]}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), strip)
            result["window_visual_path"] = str(path.relative_to(ROOT))
        except Exception as exc:
            result["friction_qc_status"] = "EXCLUDED"
            result["friction_qc_reason"] = f"v3 continuity search failed: {type(exc).__name__}: {exc}"
        results.append(result)
        print(sid, result["selected_friction_window"], flush=True)
    destination = out / "friction_window_audit_v3.json"
    with destination.open("x", encoding="utf-8") as f:
        json.dump({"policy_sha256": hashlib.sha256(policy_file.read_bytes()).hexdigest(),
                   "v2_audit_sha256": hashlib.sha256(prior_file.read_bytes()).hexdigest(),
                   "numeric_gt_used": False, "prior_prediction_error_used": False,
                   "rows": results}, f, indent=2)
        f.write("\n")
    with (out / "friction_window_audit_v3.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence_id", "object_id", "selected_friction_window",
                                                 "friction_qc_status", "friction_qc_reason", "semantics_status"])
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})


if __name__ == "__main__":
    main()
