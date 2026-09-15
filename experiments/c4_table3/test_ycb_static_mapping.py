from __future__ import annotations

import csv
import json
from pathlib import Path

from ycb_static_mapping import build_manifest


def test_mapping_rejects_tuna_and_requires_fifty_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    fields = ["input_id", "object_id", "sequence_id", "frame_id", "mass_crop_path", "friction_context_path"]
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: "007_tuna_fish_can" if k == "object_id" else "x" for k in fields})
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"targets": {}}), encoding="utf-8")
    try:
        build_manifest(manifest, audit, tmp_path / "out")
    except ValueError as exc:
        assert "50" in str(exc)
    else:
        raise AssertionError("a non-50-row manifest must be rejected")
