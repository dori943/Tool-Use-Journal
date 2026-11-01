"""Follow-up diagnostics may change poses, never experimental acceptance."""
from dataclasses import asdict
import pytest
from scripts.diagnose_c4_2_grasps import CANDIDATES, ALLOWED_FIELDS, candidate_recipe
from tuj.m5_motion.scripted_grasps.objects.c4_2_probes import probe_recipe

@pytest.mark.parametrize('candidate', CANDIDATES, ids=[c['id'] for c in CANDIDATES])
def test_pose_diagnostics_preserve_every_non_pose_field(candidate):
    original = asdict(probe_recipe(candidate['object'], candidate['ee']))
    revised = asdict(candidate_recipe(candidate))
    assert {k for k in original if original[k] != revised[k]} <= ALLOWED_FIELDS


def test_pose_diagnostics_reject_gate_changes():
    candidate = {**CANDIDATES[0], 'changes': {'maximum_slip_m': .2}}
    with pytest.raises(ValueError, match='POSE_ONLY'):
        candidate_recipe(candidate)
