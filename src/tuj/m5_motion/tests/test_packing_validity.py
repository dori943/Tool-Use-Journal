from types import SimpleNamespace as NS
import numpy as np
import pytest
from tuj.m5_motion.scripted_grasps.container_orientation import packing_transport_has_ik
from tuj.m5_motion.scripted_grasps.packing_validity import bind_packing_transport_filter


@pytest.mark.parametrize('valid', [(False, False), (False, True)])
def test_raw_ik_requires_a_collision_valid_branch(monkeypatch, valid):
    from tuj.m5_motion.tests.test_container_orientation import fixture
    g, _ = fixture(0.)
    g.task = NS(metadata={}); g.T_WE = g.T_WB.copy()
    solutions = [NS(qpos=(i,) * 6) for i in range(2)]
    c = NS(kinematics=NS(solve_all_ik=lambda *args, **kwargs: NS(solutions=solutions)),
           data=NS(qpos=np.zeros(6)), arm_ids=np.arange(6))
    calls = []
    def check(q):
        calls.append(q)
        return NS(valid=valid[q[0]], failure_code=None if valid[q[0]] else 'COLLISION_MARGIN_VIOLATION',
                  detail='' if valid[q[0]] else 'arm collides with wall')
    g.retention = NS(context=c, packing_transport_state_check=check)
    monkeypatch.setattr('tuj.m5_motion.scripted_grasps.transport.transport_destination_center', lambda g: np.ones(3))
    monkeypatch.setattr('tuj.m5_motion.scripted_grasps.container_release.clear_container_rim', lambda g, body, r: (body, {}))
    assert packing_transport_has_ik(g) is any(valid)
    assert calls == [solution.qpos for solution in solutions]
    audit = g.task.metadata['packing_orientation_ik_candidates'][0]
    assert audit['raw_ik_count'] == 2 and audit['valid_ik_count'] == sum(valid)
    del g.retention.packing_transport_state_check
    with pytest.raises(ValueError, match='COLLISION_BINDING_REQUIRED'):
        packing_transport_has_ik(g)


def test_official_context_binding_is_fresh_and_preserves_request_policy():
    from tuj.m5_motion.tests.test_container_stable_face import request
    from tuj.m5_motion.tool_use_journal_planning import ToolUseJournalCollisionContextFactory
    req, oid = request()
    req.task.action_type = 'transport'
    req.world.metadata['environment_name'] = 'test-scene'
    req.world.metadata['physical_active_ee'] = req.task.ee
    contexts_seen = []; policies_seen = []; checks = []
    def build(contexts, **kwargs):
        context = next(iter(contexts.values()))
        contexts_seen.append(context); policies_seen.append(kwargs)
        def check(q, *, context):
            checks.append((q, context))
            return NS(valid=True, failure_code=None, detail='')
        return NS(check=check)
    compiler = NS(environment_name='test-scene', model_version_for=lambda ee: 'v1',
                  build_collision_registry=build)
    factory = ToolUseJournalCollisionContextFactory(compiler, attachment_reference_name='grip')
    retention = NS(); planner = NS(collision_context_factory=factory)
    bind_packing_transport_filter(req, retention, planner)
    first = contexts_seen[0]
    assert oid in first.attached_object_ids
    assert first.attached_object_transforms
    assert policies_seen[0]['collision_margin_m'] == req.constraints.collision_margin_m
    assert policies_seen[0]['allowed_collision_pairs'] == req.constraints.allowed_collision_pairs
    req.world.scene.signature = 'changed-scene'
    req.constraints.collision_margin_m = .007
    bind_packing_transport_filter(req, retention, planner)
    retention.packing_transport_state_check((0.,) * 6)
    assert checks[-1][1] is contexts_seen[1] and contexts_seen[1] is not first
    assert first.scene_state_id != contexts_seen[1].scene_state_id
    assert policies_seen[1]['collision_margin_m'] == .007
