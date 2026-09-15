"""Render all policy candidate crops/overlays for a visual re-selection audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from audit_r_inputs import ROOT, candidate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--sequence", action="append", required=True,
                    help="six-digit test_sliding sequence directory")
    args = ap.parse_args()
    out = args.run_dir.resolve()
    policy = yaml.safe_load((ROOT / "experiments/c4_table3/r_qc_policy_v2.yaml").read_text())
    for sid in args.sequence:
        scene = ROOT / "data/external/kandukuri_ev_realphys/ev-realphys/test_sliding" / sid
        entries = []
        sheet = np.full((2 * 260, 5 * 420, 3), 255, np.uint8)
        for n, frame in enumerate(policy["candidate_frames"]):
            try:
                record = candidate(scene, frame, policy, out)
                crop = cv2.imread(str(ROOT / record["artifacts"]["crop"]))
                overlay = cv2.imread(str(ROOT / record["artifacts"]["overlay"]))
                row, col = divmod(n, 5)
                panel = np.full((260, 420, 3), 255, np.uint8)
                if crop is not None:
                    ch, cw = crop.shape[:2]
                    scale = min(185 / cw, 205 / ch, 2)
                    crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))))
                    panel[35:35 + crop.shape[0], 8:8 + crop.shape[1]] = crop
                if overlay is not None:
                    overlay = cv2.resize(overlay, (210, 119))
                    panel[40:159, 205:415] = overlay
                caption = f"{sid} f{frame} {'AUTO PASS' if record['passes_automatic'] else 'AUTO FAIL'}"
                cv2.putText(panel, caption, (7, 23), cv2.FONT_HERSHEY_SIMPLEX, .48,
                            (0, 0, 0), 1, cv2.LINE_AA)
                sheet[row * 260:(row + 1) * 260, col * 420:(col + 1) * 420] = panel
                entries.append(record)
            except Exception as exc:
                entries.append({"frame": frame, "error": f"{type(exc).__name__}: {exc}"})
        cv2.imwrite(str(out / f"{sid}_candidate_strip.png"), sheet)
        (out / f"{sid}_candidate_metrics.json").write_text(json.dumps(entries, indent=2) + "\n")
        print(sid, out / f"{sid}_candidate_strip.png", flush=True)


if __name__ == "__main__":
    main()
