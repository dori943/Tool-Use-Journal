from types import SimpleNamespace

import pytest

from tuj.m5_motion.plan_builder import MotionPlanBuilder
from tuj.m5_motion.schema import CollisionContext, ContactManipulationSpec, RelativeKeyframeSpec, ModuleName
from tuj.m5_motion.strategy import ConnectedStrategy, EdgePlanResult
from tuj.m5_motion.tests.test_plan_builder import _request, _node
from tuj.m5_motion.tool_use_journal_runtime import (
    ToolUseJournalRuntimeError, _maybe_enable_held_tool_fill_for_plan,
    _plan_requests_tabletop_held_tool_push,
)


def _side_push_plan():
    request = _request()
    request.task.contact = ContactManipulationSpec(primitive='sweep', path_pattern='SIDE_PUSH')
    frame = RelativeKeyframeSpec(keyframe_id='endpoint', keyframe_type='TRANSFER',
        frame_ref='world', anchor='origin', approach_axis_xyz=(0.,0.,1.),
        offset_along_approach_m=.8, planner='CARTESIAN')
    connected = ConnectedStrategy(strategy_id='side', nodes=(_node(frame,(.1,.1)),),
        edges=(EdgePlanResult(valid=True,joint_path=((0.,0.),(.1,.1))),), edge_evaluations=1)
    context = CollisionContext(context_id='ctx',scene_state_id='scene',active_ee='2f',
        collision_model_version='v1')
    return MotionPlanBuilder().build(request, connected, plan_id='side-plan',
        provenance=request.provenance.model_copy(update={'artifact_type':'MotionPlan', 'produced_by':ModuleName.MOTION_PLANNER}),
        collision_contexts={'ctx': context}, initial_collision_context_id='ctx',
        final_segment_validator=lambda *args: True)


def test_side_push_mode_survives_plan_building_and_cannot_arm_target_assistance():
    plan = _side_push_plan()
    assert plan.metadata['contact_path_pattern'] == 'SIDE_PUSH'
    def forbidden(*args, **kwargs):
        pytest.fail('native plan must not arm or clear a target-moving assist')
    runtime = SimpleNamespace(enable_held_tool_collision_fill=forbidden,
        enable_held_tool_contact_drag=forbidden, disable_held_tool_collision_fill=forbidden)
    assert _plan_requests_tabletop_held_tool_push(plan) is False
    assert _maybe_enable_held_tool_fill_for_plan(runtime, plan) is False


@pytest.mark.parametrize('active_state', ['_held_tool_push_partner_ids',
    '_held_tool_fill_geom_backup','_held_tool_rim_collision_backup',
    '_held_tool_contact_drag','_held_tool_drag_offsets'])
def test_native_plan_rejects_inherited_assistance_without_moving_targets(active_state):
    runtime = SimpleNamespace(**{active_state: {'target': 1}})
    before = vars(runtime).copy()
    with pytest.raises(ToolUseJournalRuntimeError, match='NATIVE_CONTACT_REQUIRES_UNASSISTED_STATE'):
        _maybe_enable_held_tool_fill_for_plan(runtime, _side_push_plan())
    assert vars(runtime) == before


def test_generated_native_preparation_is_unassisted_after_normal_plan_building():
    from tuj.m5_motion.tests.test_native_contact_preparation import _build_preparation_plan
    plan = _build_preparation_plan()
    def forbidden(*args, **kwargs):
        pytest.fail('native preparation must not enable or clear target-moving assistance')
    runtime = SimpleNamespace(enable_held_tool_collision_fill=forbidden,
        enable_held_tool_contact_drag=forbidden, disable_held_tool_collision_fill=forbidden)
    assert _plan_requests_tabletop_held_tool_push(plan) is False
    assert _maybe_enable_held_tool_fill_for_plan(runtime, plan) is False


@pytest.mark.parametrize('active_state', ['_held_tool_push_partner_ids',
    '_held_tool_fill_geom_backup','_held_tool_rim_collision_backup',
    '_held_tool_contact_drag','_held_tool_drag_offsets'])
def test_native_preparation_rejects_inherited_assist_without_cleanup(active_state):
    from tuj.m5_motion.tests.test_native_contact_preparation import _build_preparation_plan
    runtime = SimpleNamespace(**{active_state: {'target': 1}})
    before = vars(runtime).copy()
    with pytest.raises(ToolUseJournalRuntimeError, match='NATIVE_CONTACT_REQUIRES_UNASSISTED_STATE'):
        _maybe_enable_held_tool_fill_for_plan(runtime, _build_preparation_plan())
    assert vars(runtime) == before
