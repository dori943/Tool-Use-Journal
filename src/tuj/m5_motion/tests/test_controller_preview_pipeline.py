import pytest

from tuj.m5_motion.pipeline import MotionPlanningPipeline, MotionPlanningPipelineError
from tuj.m5_motion.plan_builder import MotionPlanBuildError
from tuj.m5_motion.schema import CollisionContext
from tuj.m5_motion.tests.test_pipeline import _FeedbackAwareProvider, _FakeKinematics, _request


def run(provider, validator):
    return MotionPlanningPipeline(provider, _FakeKinematics()).plan(
        _request(), state_validator=lambda q, keyframe: True,
        collision_contexts={'default': CollisionContext(context_id='default', collision_model_version='fixture')},
        initial_collision_context_id='default',
        final_segment_validator=lambda waypoints, context: True,
        final_plan_validator=validator,
    )


def test_rejected_controller_candidate_is_not_selected():
    calls = []
    def validate(request, plan):
        calls.append(plan)
        if len(calls) == 1:
            raise MotionPlanBuildError('CONTROLLER_PREVIEW_REJECTED grasp unavailable')
    result = run(_FeedbackAwareProvider(), validate)
    assert len(calls) == 2
    assert result.plan is calls[1]
    assert result.compilation.attempts[0].failure_code == 'FINAL_VALIDATION_FAILED'


def test_collision_repair_rechecks_controller_and_is_bounded():
    provider = _FeedbackAwareProvider(repair_on_attempt=None)
    calls = []
    def reject(request, plan):
        calls.append(plan)
        raise MotionPlanBuildError('CONTROLLER_PREVIEW_REJECTED COLLISION_MARGIN_VIOLATION: held_g <-> table_g clearance 0.004 m is below required 0.005 m')
    with pytest.raises(MotionPlanningPipelineError, match='exhausted'):
        run(provider, reject)
    assert len(provider.calls) == 3
    assert len(calls) == 6
    assert provider.calls[1]['failed_strategies'][0]['collision_observations']


def test_untrusted_preview_error_fails_closed_without_candidate_retry():
    calls = []
    def broken(request, plan):
        calls.append(plan)
        raise RuntimeError('checkpoint mismatch')
    with pytest.raises(RuntimeError, match='checkpoint mismatch'):
        run(_FeedbackAwareProvider(), broken)
    assert len(calls) == 1
