from __future__ import annotations

import json
from pathlib import Path

from .generate_sim_gt import generate
from .validate_sim_gt import validate


def test_simulator_gt_is_valid_and_input_manifest_has_no_gt(tmp_path: Path) -> None:
    result = generate(tmp_path)
    assert result["sample_count"] == 5
    assert result["critical_count"] == 3
    assert validate(tmp_path) == []
    manifest_text = (tmp_path / "sim_manifest.yaml").read_text(encoding="utf-8")
    assert "mass_gt_kg" not in manifest_text
    assert "trial_success" not in manifest_text
    for path in (tmp_path / "observations").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        assert row["input_gt_excluded"]
