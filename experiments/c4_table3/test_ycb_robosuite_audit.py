from __future__ import annotations

import json
from pathlib import Path

from ycb_robosuite_audit import audit


def test_audit_requires_exact_id_and_excludes_tuna(tmp_path: Path) -> None:
    ycb = tmp_path / "ycb"
    (ycb / "003_cracker_box").mkdir(parents=True)
    (ycb / "003_cracker_box" / "textured.obj").write_text(
        "v 0 0 0\nv 1 2 3\nf 1 2 2\n", encoding="utf-8"
    )
    (ycb / "007_tuna_fish_can").mkdir()
    report = audit(tmp_path / "robosuite", ycb, tmp_path / "out")
    assert report["targets"]["003_cracker_box"]["mapping_status"] == "EXACT_ASSET_FOUND"
    assert report["targets"]["006_mustard_bottle"]["mapping_status"] == "YCB_ASSET_NOT_FOUND"
    assert "007_tuna_fish_can" in report["excluded_object_ids"]
    assert report["provenance_rules"]["real5_gt_allowed_in_simulator"] is False
    saved = json.loads((tmp_path / "out" / "ycb_robosuite_audit.json").read_text(encoding="utf-8"))
    assert "mass_gt_kg" not in json.dumps(saved)


def test_mjcf_wrapper_requires_explicit_scale(tmp_path: Path) -> None:
    ycb = tmp_path / "ycb" / "025_mug"
    ycb.mkdir(parents=True)
    (ycb / "model.obj").write_text("v 0 0 0\nv 1 1 1\n", encoding="utf-8")
    try:
        audit(tmp_path / "robosuite", tmp_path / "ycb", tmp_path / "out", emit_mjcf=True)
    except ValueError as exc:
        assert "unit-scale" in str(exc)
    else:
        raise AssertionError("MJCF emission must require an explicit unit scale")
