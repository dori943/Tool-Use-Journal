"""Validate experimental routing and prevent changing limits to create success."""
from dataclasses import asdict
import pytest
from tuj.m5_motion.scripted_grasps.objects.c4_1_probes import (
    ENVIRONMENT, OBJECTS, PRIMARY_PAIRS, probe_recipe, source_recipe)
from tuj.m5_motion.scripted_grasps.validate import _acquire_request, resolve_case
from tuj.m5_motion.scripted_grasps.registry import resolve, integration_status

PAIRS = [(name, ee) for name in OBJECTS for ee in ("2F", "3F", "vac")]
ALLOWED = {"task_id", "ee_id", "two_finger_parallel_linkage", "finger_attachment_policy",
           "preshape_closure_command", "preshape_aperture_m", "rotation_xyz_deg",
           "offset_fraction", "offset_m"}

@pytest.mark.parametrize("name,ee", PAIRS)
def test_validation_routes_preserve_production_and_physical_limits(name, ee):
    trial = resolve_case(name, ee, environment=ENVIRONMENT)
    assert trial["status"] == "RESOLVED"
    entry = trial["entry"]
    request = _acquire_request(ENVIRONMENT, name, ee)
    request.task.metadata.pop("scripted_grasp_validator_experimental")
    if (name, ee) in PRIMARY_PAIRS:
        assert resolve(request).recipe() == source_recipe(name)
        with pytest.raises(ValueError, match="PRIMARY_RECIPE"):
            probe_recipe(name, ee)
    else:
        assert resolve(request) is None
        assert integration_status(entry) == "EXPERIMENTAL"
        before, after = asdict(source_recipe(name)), asdict(entry.recipe())
        assert {k for k in before if before[k] != after[k]} <= ALLOWED
        assert entry.recipe().finger_attachment_policy == "FREE_HOLD_THEN_ATTACH"


def test_current_scene_is_not_reported_as_old_four_object_table():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from audit_ee_coverage import TASK_GRASP_OBJECTS
    from verify_table2_coverage import summarize_trials
    assert tuple(TASK_GRASP_OBJECTS["c4_1"]) == OBJECTS
    rows = [dict(object=n, ee="2F", status="FAIL", inferred_feasible=None,
                 grasp_attempted=True) for n in OBJECTS]
    summary = summarize_trials("c4_1", rows, ["2F"])["2F"]
    assert summary["coverage"] == "0/5" and summary["matrix_complete"]
    assert summary["table2"] == "4/4"
    assert summary["table2_same_object_count"] is False
