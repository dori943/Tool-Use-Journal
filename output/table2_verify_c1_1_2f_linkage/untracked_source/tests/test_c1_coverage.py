# -*- coding: utf-8 -*-
"""C1_1 / C1_2 coverage probes follow the C4_1 contract: validator-only routing,
production routes untouched, and each probe differs from its validated source
recipe only in identity/pose/opening fields (no limit changes to create success)."""
from dataclasses import asdict

import pytest

from tuj.m5_motion.scripted_grasps.objects import c1_1_probes, c1_2_probes
from tuj.m5_motion.scripted_grasps.registry import integration_status, resolve
from tuj.m5_motion.scripted_grasps.validate import _acquire_request, resolve_case

MODULES = {"c1_1": c1_1_probes, "c1_2": c1_2_probes}
CASES = [(task, name, ee) for task, mod in MODULES.items()
         for name in mod.OBJECTS for ee in ("2F", "3F", "vac")]


@pytest.mark.parametrize("task,name,ee", CASES)
def test_routes_resolve_and_preserve_source_limits(task, name, ee):
    mod = MODULES[task]
    trial = resolve_case(name, ee, environment=mod.ENVIRONMENT)
    assert trial["status"] == "RESOLVED", trial
    entry = trial["entry"]
    assert entry.scene_object_id == name and entry.recipe().ee_id == ee
    request = _acquire_request(mod.ENVIRONMENT, name, ee)
    request.task.metadata.pop("scripted_grasp_validator_experimental", None)
    if (name, ee) in mod.PRIMARY_PAIRS:
        assert resolve(request) is not None
        with pytest.raises(ValueError, match="PRIMARY_RECIPE"):
            mod.probe_recipe(name, ee)
    else:
        assert resolve(request) is None
        assert integration_status(entry) == "EXPERIMENTAL"
        before, after = asdict(mod.source_recipe(name, ee)), asdict(entry.recipe())
        assert {k for k in before if before[k] != after[k]} <= mod.ALLOWED
        assert entry.recipe().finger_attachment_policy == "FREE_HOLD_THEN_ATTACH"


def test_two_finger_handle_probes_do_not_close_along_the_tool_length():
    """2F jaw closes across body X by default; yaw 90 would close across the length."""
    for mod in MODULES.values():
        for name, ees in mod.MISSING_TYPE_EES.items():
            if "2F" in ees:
                r = mod.probe_recipe(name, "2F")
                assert abs(r.rotation_xyz_deg[2]) < 1e-9, (name, r.rotation_xyz_deg)


def test_sizes_match_asset_reg_bbox():
    import xml.etree.ElementTree as ET
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "environments" / "assets" / "objects"
    for mod, alias in ((c1_1_probes, {"bottle_distractor": "bottle"}), (c1_2_probes, {})):
        for name, size in mod.SIZES.items():
            bbox = ET.parse(root / alias.get(name, name) / "model.xml").getroot().find(".//geom[@name='reg_bbox']")
            half = [float(v) for v in bbox.get("size").split()]
            assert all(abs(2 * h - s) < 1e-3 for h, s in zip(half, size)), (name, size, half)


def test_probe_counts_match_table2_denominators():
    assert len(c1_1_probes.OBJECTS) == 5 and len(c1_2_probes.OBJECTS) == 4
    assert sum(map(len, c1_1_probes.MISSING_TYPE_EES.values())) + len(c1_1_probes.PRIMARY_PAIRS) == 15
    assert sum(map(len, c1_2_probes.MISSING_TYPE_EES.values())) + len(c1_2_probes.PRIMARY_PAIRS) == 12
