"""C3_1 validator additions must preserve existing production routes and gates."""
from dataclasses import asdict
import pytest
from tuj.m5_motion.scripted_grasps.registry import resolve, integration_status
from tuj.m5_motion.scripted_grasps.validate import resolve_case, _acquire_request
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe
from tuj.m5_motion.scripted_grasps.objects.c3_1_probes import ENVIRONMENT, MISSING_TYPE_EES

@pytest.mark.parametrize('name,ee', [(name, ee) for name in MISSING_TYPE_EES for ee in ('2F','3F','vac')])
def test_validation_routes_do_not_expand_production_or_relax_gates(name, ee):
    entry = resolve_case(name, ee, environment=ENVIRONMENT)['entry']
    assert entry is not None and entry.scene_object_id == name
    assert entry.recipe().ee_id == ee
    req = _acquire_request(ENVIRONMENT, name, ee)
    req.task.metadata.pop('scripted_grasp_validator_experimental')
    prod = resolve(req)
    if ee in MISSING_TYPE_EES[name]:
        assert prod is None and integration_status(entry) == 'EXPERIMENTAL'
        recipe = entry.recipe()
        base = CatalogRecipe(name, 'c3_1', ee, recipe.expected_size_m,
                             two_finger_parallel_linkage=ee == '2F')
        a, b = asdict(base), asdict(recipe)
        assert {k for k in a if a[k] != b[k]} <= {'offset_fraction', 'rotation_xyz_deg', 'preshape_closure_command'}
        assert recipe.finger_attachment_policy == 'FREE_HOLD_THEN_ATTACH'
    else:
        assert prod is not None and prod.recipe() == entry.recipe()


def test_legacy_apple_rotation_tolerance_stays_explicit():
    recipe = resolve_case('apple','2F',environment=ENVIRONMENT)['entry'].recipe()
    assert recipe.maximum_slip_deg == 25.
    assert recipe.two_finger_force_target_n == 8.
