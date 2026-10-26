"""Rejected M2 proposals remain inspectable without becoming executable."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_rejected_packing_plan_retains_proposal_and_paid_usage(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    import run_m2 as runner

    usage = {'rough': {'calls': 2, 'tokens': 43, 'seconds': 0.2}}
    proposal = {
        'm2_subgoals': [{'subgoal_id': 'SG1', 'details': []}],
        'm2_stats': {}, 'm2_partial_order': [], 'm2_mutex': [],
    }
    rough = SimpleNamespace(_client=object(), model='mock', usage=usage)
    monkeypatch.setattr(runner, 'LLMRough', lambda: rough)
    monkeypatch.setattr(runner, 'load_scene', lambda _p: {'nodes': [], 'edges': []})
    monkeypatch.setattr(runner, 'run_m2', lambda *_a, **_k: copy.deepcopy(proposal))
    monkeypatch.setattr(runner, 'apply_grounding', lambda *_a: [])
    monkeypatch.setattr(runner, 'measurement_feedback', lambda _p: None)
    monkeypatch.setattr(runner, 'update_confidence', lambda *_a, **_k: [])
    monkeypatch.setattr(runner, 'split_by_partition', lambda _p: [])

    def reject(*_a):
        raise ValueError('packing policy: unresolved tool')

    monkeypatch.setattr(runner, 'apply_task_packing_policy', reject)
    monkeypatch.setattr(sys, 'argv', ['run_m2.py', 'c4_2', '--m1-json', 'mock.json',
                                    '--output-dir', str(tmp_path)])
    with pytest.raises(ValueError, match='unresolved tool'):
        runner.main()
    assert not (tmp_path / 'm2.json').exists()
    rejected = json.loads((tmp_path / 'm2_rejected.json').read_text())
    assert rejected['m2_subgoals'] == proposal['m2_subgoals']
    assert rejected['m2_stats']['llm_usage'] == usage
    assert rejected['m2_validation']['status'] == 'REJECTED'
    assert rejected['m2_validation']['error'] == 'packing policy: unresolved tool'
