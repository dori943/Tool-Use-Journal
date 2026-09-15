"""Render a contact sheet of selected mask overlays from an input audit."""
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
    rows = json.loads((out / "mass_candidate_audit.json").read_text())["rows"]
    canvas = np.full((5 * 260, 5 * 424, 3), 255, np.uint8)
    for n, row in enumerate(rows):
        selected = row["selected"]
        if selected is None:
            continue
        image = cv2.imread(str(ROOT / selected["artifacts"]["overlay"]))
        image = cv2.resize(image, (424, 240))
        y, x = divmod(n, 5)
        canvas[y * 260:y * 260 + 240, x * 424:(x + 1) * 424] = image
        cv2.putText(canvas, f"{row['sequence_id'][-6:]} f{selected['frame']}",
                    (x * 424 + 5, y * 260 + 255), cv2.FONT_HERSHEY_SIMPLEX,
                    .48, (0, 0, 0), 1)
    path = out / "selected_mass_overlay_contact_sheet.png"
    if path.exists():
        raise SystemExit(f"refuse to overwrite {path}")
    cv2.imwrite(str(path), canvas)


if __name__ == "__main__":
    main()
