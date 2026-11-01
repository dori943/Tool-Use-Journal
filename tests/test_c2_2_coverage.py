"""C2_2 coverage must not expand M4 choices or relax contact/physics gates."""
from dataclasses import asdict
import pytest
from tuj.m5_motion.scripted_grasps.registry import resolve, integration_status
from tuj.m5_motion.scripted_grasps.validate import resolve_case, _acquire_request
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe
from tuj.m5_motion.scripted_grasps.objects.c2_2_probes import ENVIRONMENT, OBJECTS, probe_recipe

@pytest.mark.parametrize('name,ee',[(n,e) for n in OBJECTS for e in ('2F','3F','vac')])
def test_routes_preserve_production_and_physical_limits(name,ee):
    entry=resolve_case(name,ee,environment=ENVIRONMENT)['entry']
    assert entry is not None and entry.scene_object_id==name
    r=entry.recipe();assert r.object_id==name and r.ee_id==ee and r.task_id=='c2_2'
    request=_acquire_request(ENVIRONMENT,name,ee)
    request.task.metadata.pop('scripted_grasp_validator_experimental')
    prod=resolve(request)
    if ee=='vac':
        assert prod is not None and prod.recipe()==r
    else:
        assert prod is None and integration_status(entry)=='EXPERIMENTAL'
        a=asdict(CatalogRecipe(name,'c2_2',ee,r.expected_size_m,two_finger_parallel_linkage=ee=='2F'));b=asdict(r)
        assert {k for k in a if a[k]!=b[k]} <= {'offset_fraction','rotation_xyz_deg','preshape_aperture_m','preshape_closure_command'}
        assert r.finger_attachment_policy=='FREE_HOLD_THEN_ATTACH'

@pytest.mark.parametrize('name',OBJECTS)
def test_finger_probes_cannot_replace_vacuum(name):
    with pytest.raises(ValueError,match='UNSUPPORTED_C2_2_PROBE'):probe_recipe(name,'vac')
