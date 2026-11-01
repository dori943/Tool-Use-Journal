"""C2_1 new candidates preserve defaults, legacy routes and production choices."""
from dataclasses import asdict
import pytest
from tuj.m5_motion.scripted_grasps.registry import resolve,integration_status
from tuj.m5_motion.scripted_grasps.validate import resolve_case,_acquire_request
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe
from tuj.m5_motion.scripted_grasps.objects.c2_1_probes import ENVIRONMENT,MISSING_TYPE_EES,GEOMETRY_FIELDS,TWO_FINGER_SOURCES

@pytest.mark.parametrize('name,ee',[(n,e) for n in MISSING_TYPE_EES for e in ('2F','3F','vac')])
def test_registered_candidates_are_bound_and_isolated(name,ee):
    entry=resolve_case(name,ee,environment=ENVIRONMENT)['entry'];assert entry is not None
    assert entry.scene_object_id==name and entry.recipe().ee_id==ee
    request=_acquire_request(ENVIRONMENT,name,ee);request.task.metadata.pop('scripted_grasp_validator_experimental');prod=resolve(request)
    if ee in MISSING_TYPE_EES[name]:
        assert prod is None and integration_status(entry)=='EXPERIMENTAL'
        r=entry.recipe();base=CatalogRecipe(name,'c2_1',ee,r.expected_size_m,two_finger_parallel_linkage=ee=='2F')
        a,b=asdict(base),asdict(r);assert {k for k in a if a[k]!=b[k]}<=set(GEOMETRY_FIELDS)
        assert r.finger_attachment_policy=='FREE_HOLD_THEN_ATTACH'
        if ee=='2F' and name in TWO_FINGER_SOURCES:
            source=TWO_FINGER_SOURCES[name](environment=ENVIRONMENT)
            assert all(getattr(r,k)==getattr(source,k) for k in GEOMETRY_FIELDS)
    else:assert prod is not None and prod.recipe()==entry.recipe()


def test_new_apple_probe_does_not_copy_legacy_force_or_rotation_tolerance():
    r=resolve_case('apple','2F',environment=ENVIRONMENT)['entry'].recipe()
    old=TWO_FINGER_SOURCES['apple'](environment=ENVIRONMENT)
    assert (r.two_finger_force_target_n,r.maximum_slip_deg)==(5.,5.)
    assert (old.two_finger_force_target_n,old.maximum_slip_deg)==(8.,25.)
