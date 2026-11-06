# -*- coding: utf-8 -*-
"""C1_1 / C1_2 coverage probes: bound to the validator only, defaults preserved,
production routes untouched (same contract as tests/test_c2_1_coverage.py)."""
from dataclasses import asdict

import pytest

from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe
from tuj.m5_motion.scripted_grasps.objects import c1_1_probes, c1_2_probes
from tuj.m5_motion.scripted_grasps.registry import ENABLED_ENTRIES, integration_status, resolve
from tuj.m5_motion.scripted_grasps.validate import _acquire_request, resolve_case

MODULES = {"c1_1": c1_1_probes, "c1_2": c1_2_probes}
CASES = [(task, name, ee)
         for task, mod in MODULES.items()
         for name in mod.OBJECTS
         for ee in ("2F", "3F", "vac")]


@pytest.mark.parametrize("task,name,ee", CASES)
def test_every_pair_resolves_for_the_validator(task, name, ee):
    mod = MODULES[task]
    resolved = resolve_case(name, ee, environment=mod.ENVIRONMENT)
    entry = resolved["entry"]
    assert entry is not None, resolved
    assert entry.scene_object_id == name
    assert entry.recipe().ee_id == ee


@pytest.mark.parametrize("task,name,ee", CASES)
def test_probe_pairs_are_experimental_and_keep_defaults(task, name, ee):
    mod = MODULES[task]
    entry = resolve_case(name, ee, environment=mod.ENVIRONMENT)["entry"]
    request = _acquire_request(mod.ENVIRONMENT, name, ee)
    request.task.metadata.pop("scripted_grasp_validator_experimental", None)
    production = resolve(request)
    if ee in mod.MISSING_TYPE_EES.get(name, ()):
        assert production is None, "probe pair must not be selectable in production"
        assert integration_status(entry) == "EXPERIMENTAL"
        r = entry.recipe()
        base = CatalogRecipe(name, task, ee, r.expected_size_m, two_finger_parallel_linkage=ee == "2F")
        a, b = asdict(base), asdict(r)
        assert {k for k in a if a[k] != b[k]} <= set(mod.GEOMETRY_FIELDS)
        assert r.finger_attachment_policy == "FREE_HOLD_THEN_ATTACH"
        assert (r.two_finger_force_target_n, r.maximum_slip_deg, r.hold_s) == (5., 5., 5.)
    else:
        assert production is not None, "production route must stay resolvable"
        assert production.recipe() == entry.recipe()
        assert (name, ee) in mod.PRIMARY_PAIRS


def test_primary_pairs_are_not_replaced():
    for mod in MODULES.values():
        for name, ee in mod.PRIMARY_PAIRS:
            with pytest.raises(ValueError):
                mod.probe_recipe(name, ee)
            assert any(e.environment == mod.ENVIRONMENT and e.ee == ee
                       and e.object_id in (name, name.rsplit("_", 1)[0]) for e in ENABLED_ENTRIES)


def test_sizes_match_asset_reg_bbox():
    import xml.etree.ElementTree as ET
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "environments" / "assets" / "objects"
    for mod, alias in ((c1_1_probes, {"bottle_distractor": "bottle"}), (c1_2_probes, {})):
        for name, size in mod.SIZES.items():
            asset = root / alias.get(name, name) / "model.xml"
            bbox = ET.parse(asset).getroot().find(".//geom[@name='reg_bbox']")
            assert bbox is not None, asset
            half = [float(v) for v in bbox.get("size").split()]
            assert all(abs(2 * h - s) < 1e-3 for h, s in zip(half, size)), (name, size, half)


def test_probe_counts_match_table2_denominators():
    assert len(c1_1_probes.OBJECTS) == 5 and len(c1_2_probes.OBJECTS) == 4
    assert sum(len(v) for v in c1_1_probes.MISSING_TYPE_EES.values()) + len(c1_1_probes.PRIMARY_PAIRS) == 15
    assert sum(len(v) for v in c1_2_probes.MISSING_TYPE_EES.values()) + len(c1_2_probes.PRIMARY_PAIRS) == 12
