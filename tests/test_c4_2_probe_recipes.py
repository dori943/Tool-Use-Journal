"""Experimental EE candidates preserve existing physical limits and evaluation."""
from importlib import import_module
import pytest
from tuj.m5_motion.scripted_grasps.objects.c4_2_probes import PRIMARY_EES, probe_recipe

CASES=[(name,ee) for name,primary in PRIMARY_EES.items() for ee in ('2F','3F','vac') if ee != primary]

@pytest.mark.parametrize('name,ee', CASES)
def test_probe_preserves_source_thresholds_and_forces(name,ee):
    source=getattr(import_module('tuj.m5_motion.scripted_grasps.objects.'+name),name+'_recipe')()
    probe=probe_recipe(name,ee)
    for field in ('expected_size_m','minimum_lift_m','maximum_slip_m','maximum_slip_deg',
                  'hold_s','contact_ticks','minimum_normal_opposition','fingerpad_friction',
                  'two_finger_force_target_n','three_finger_force_targets_n',
                  'maximum_vacuum_attach_penetration_m','vacuum_min_contacts'):
        assert getattr(probe,field)==getattr(source,field), field
    assert probe.ee_id==ee
    assert probe.finger_attachment_policy=='FREE_HOLD_THEN_ATTACH'


def test_primary_recipes_cannot_be_replaced_by_probes():
    for name,ee in PRIMARY_EES.items():
        with pytest.raises(ValueError,match='PRIMARY_RECIPE'):
            probe_recipe(name,ee)


@pytest.mark.parametrize('name,ee', CASES)
def test_only_successful_candidates_resolve_in_production(name,ee):
    from tuj.m5_motion.scripted_grasps.validate import _acquire_request, resolve_case
    from tuj.m5_motion.scripted_grasps.registry import (resolve, integration_status,
        C4_2_VALIDATED_ALTERNATIVE_PAIRS)
    environment='C4_2_DiagonalFitPacking'
    trial=resolve_case(name,ee,environment=environment)
    assert trial['status']=='RESOLVED'
    request=_acquire_request(environment,name,ee)
    request.task.metadata.pop('scripted_grasp_validator_experimental')
    if (name,ee) in C4_2_VALIDATED_ALTERNATIVE_PAIRS:
        assert integration_status(trial['entry'])=='VALIDATED'
        assert resolve(request).recipe()==probe_recipe(name,ee)
    else:
        assert integration_status(trial['entry'])=='EXPERIMENTAL'
        assert resolve(request) is None
