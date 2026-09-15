"""Re-run all Real-5 RGB-D tracks and fixed, GT-blind slide-window selection."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/c4_real5")]
from friction_from_rgbd import choose_slide_window, track_scene  # noqa: E402
from real_rgbd import load_frame  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.run_dir.resolve()
    source = json.loads((ROOT / "data/external/kandukuri_ev_realphys/manifest.json").read_text())
    policy = yaml.safe_load((ROOT / "experiments/c4_table3/r_qc_policy_v2.yaml").read_text())
    results = []
    for row in source["sequences"]:
        sid, oid = row["sequence_id"], row["object_id"]
        scene = ROOT / "data/external/kandukuri_ev_realphys/ev-realphys" / sid
        result = {"sequence_id": sid, "object_id": oid,
                  "reference_frame": policy["friction"]["reference_frame"],
                  "selected_friction_window": None,
                  "friction_qc_status": "QC_UNREVIEWED",
                  "friction_qc_reason": "RGB-D contact/rotation and track QC pending",
                  "semantics_status": "TARGET_SEMANTICS_UNVERIFIED"}
        try:
            positions, track_info = track_scene(scene, result["reference_frame"])
            start, end, window_info = choose_slide_window(positions)
            xy = positions[start:end + 1, :2]
            step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            t = np.arange(len(xy)) / 30.0
            u = (xy[-1] - xy[0]) / np.linalg.norm(xy[-1] - xy[0])
            longitudinal = (xy - xy[0]) @ u
            fit = np.polynomial.polynomial.polyfit(t, longitudinal, 2)
            rms = float(np.sqrt(np.mean((longitudinal -
                                         np.polynomial.polynomial.polyval(t, fit)) ** 2)))
            lateral = (xy - xy[0]) @ np.array([-u[1], u[0]])
            result.update({"selected_friction_window": [start, end],
                           "track_info": track_info, "window_info": window_info,
                           "longitudinal_quadratic_rms_mm": round(rms, 3),
                           "lateral_rms_mm": round(float(np.sqrt(np.mean(lateral ** 2))), 3),
                           "reversal_step_count": int(np.sum(np.diff(longitudinal) <= 0)),
                           "step_min_mm": round(float(step.min()), 3),
                           "step_max_mm": round(float(step.max()), 3),
                           "positions_mm": positions.tolist()})
            montage = np.full((3 * 240, 424, 3), 255, np.uint8)
            for i, frame in enumerate((start, (start + end) // 2, end)):
                bgr, _, _, _ = load_frame(scene, frame)
                small = cv2.resize(bgr, (424, 240))
                cv2.putText(small, f"{sid[-6:]} frame {frame}", (5, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
                montage[i * 240:(i + 1) * 240] = small
            path = out / "friction_window_frames" / f"{sid[-6:]}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), montage)
            result["window_visual_path"] = str(path.relative_to(ROOT))
        except Exception as exc:
            result["friction_qc_status"] = "EXCLUDED"
            result["friction_qc_reason"] = f"RGB-D track/window failure: {type(exc).__name__}: {exc}"
        results.append(result)
        print(sid, result["selected_friction_window"], result["friction_qc_status"], flush=True)
    (out / "friction_window_audit.json").write_text(json.dumps({
        "numeric_gt_used": False, "prior_prediction_error_used": False,
        "window_rule": "friction_from_rgbd.choose_slide_window; policy v2",
        "source_manifest": "data/external/kandukuri_ev_realphys/manifest.json",
        "rows": results}, indent=2) + "\n")
    with (out / "friction_window_audit.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence_id", "object_id", "reference_frame",
                                                 "selected_friction_window", "friction_qc_status",
                                                 "friction_qc_reason", "semantics_status"])
        writer.writeheader()
        for r in results:
            writer.writerow({key: r.get(key) for key in writer.fieldnames})


if __name__ == "__main__":
    main()
