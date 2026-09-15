"""Render start/mid/end RGB evidence for every GT-blind friction candidate window."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from audit_r_inputs import ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.run_dir.resolve()
    rows = json.loads((out / "friction_window_audit.json").read_text())["rows"]
    for stage in range(3):
        sheet = np.full((5 * 260, 5 * 424, 3), 255, np.uint8)
        for n, row in enumerate(rows):
            if not row.get("window_visual_path"):
                continue
            strip = cv2.imread(str(ROOT / row["window_visual_path"]))
            if strip is None:
                continue
            panel = strip[stage * 240:(stage + 1) * 240]
            y, x = divmod(n, 5)
            sheet[y * 260:y * 260 + 240, x * 424:(x + 1) * 424] = panel
            cv2.putText(sheet, row["sequence_id"][-6:], (x * 424 + 4, y * 260 + 255),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
        path = out / f"friction_{('start', 'middle', 'end')[stage]}_contact_sheet.png"
        if path.exists():
            raise SystemExit(f"refuse to overwrite {path}")
        cv2.imwrite(str(path), sheet)


if __name__ == "__main__":
    main()
