"""Object wrappers must forward the environment-scoped recipe already bound."""
from types import SimpleNamespace
from dataclasses import replace
import pytest
from tuj.m5_motion.scripted_grasps.validate import resolve_case

@pytest.mark.parametrize('name', ['apple','mug'])
@pytest.mark.parametrize('environment', ['C2_1_ObjectSorting','C3_1_ObjectSorting'])
def test_wrapper_uses_bound_environment_recipe(name, environment):
    entry=resolve_case(name,'3F',environment=environment)['entry']
    # Custom geometry proves that the wrapper preserves the actual bound value,
    # not merely a newly constructed same-environment recipe.
    recipe=replace(entry.recipe(), offset_m=(.001,0.,0.))
    context=SimpleNamespace(recipe=recipe, execute_object=lambda actual: actual)
    assert entry.function()(context) is recipe

@pytest.mark.parametrize('name', ['apple','mug'])
def test_explicit_recipe_and_standalone_default_remain_supported(name):
    entry=resolve_case(name,'3F',environment='C2_1_ObjectSorting')['entry']
    context=SimpleNamespace(execute_object=lambda actual: actual)
    assert entry.function()(context)==entry.recipe()
    explicit=replace(entry.recipe(),offset_m=(.002,0.,0.))
    context.recipe=entry.recipe()
    assert entry.function()(context,recipe=explicit) is explicit
