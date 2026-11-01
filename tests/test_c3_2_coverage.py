"""Keep validation extensions isolated and preserve multi-instance binding."""
from dataclasses import asdict
import pytest
from tuj.m5_motion.scripted_grasps.registry import resolve, integration_status
from tuj.m5_motion.scripted_grasps.validate import resolve_case, _acquire_request
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe
from tuj.m5_motion.scripted_grasps.objects.c3_2_probes import ENVIRONMENT, MISSING_TYPE_EES

PAIRS = [(f"{kind}_{suffix}", ee) for kind in ("plate", "mug", "fork", "spoon", "bread", "fruit")
         for suffix in ("a", "b") for ee in ("2F", "3F", "vac")]
GEOMETRY_FIELDS = {"offset_fraction", "offset_m", "rotation_xyz_deg", "preshape_aperture_m", "preshape_closure_command"}

@pytest.mark.parametrize("name,ee", PAIRS)
def test_all_cases_bind_correct_body_without_production_expansion(name, ee):
    entry = resolve_case(name, ee, environment=ENVIRONMENT)["entry"]
    assert entry is not None and entry.scene_object_id == name
    assert entry.recipe().ee_id == ee
    request = _acquire_request(ENVIRONMENT, name, ee)
    request.task.metadata.pop("scripted_grasp_validator_experimental")
    prod = resolve(request)
    kind = name[:-2]
    if ee in MISSING_TYPE_EES.get(kind, ()):
        assert prod is None
        assert integration_status(entry) == "EXPERIMENTAL"
        recipe = entry.recipe()
        base = CatalogRecipe(recipe.object_id, "c3_2", ee, recipe.expected_size_m,
                             two_finger_parallel_linkage=ee == "2F")
        before, after = asdict(base), asdict(recipe)
        assert {key for key in before if before[key] != after[key]} <= GEOMETRY_FIELDS
        assert recipe.finger_attachment_policy == "FREE_HOLD_THEN_ATTACH"
    elif kind in {"bread", "spoon", "fork"} and ee == "2F":
        assert prod is None
    else:
        assert prod is not None and prod.recipe() == entry.recipe()


def test_greedy_fruit_recipe_remains_separate():
    request = _acquire_request(ENVIRONMENT, "fruit_a", "2F")
    validator_recipe = resolve(request).recipe()
    request.task.metadata.pop("scripted_grasp_validator_experimental")
    request.task.metadata["scripted_grasp_greedy_extra"] = True
    greedy_recipe = resolve(request).recipe()
    assert validator_recipe.maximum_slip_deg == 5.
    assert validator_recipe.two_finger_force_target_n == 5.
    assert greedy_recipe.maximum_slip_deg == 25.
    assert greedy_recipe.two_finger_force_target_n == 8.
