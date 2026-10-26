"""Scene-tool contacts are physical candidates, never attachment exemptions."""
import copy
from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.m5_motion.geometry import RelativePoseResolver
from tuj.m5_motion.scene_tool_grasp import ANCHOR, SOURCE, bind_scene_tool_contact
from tuj.m5_motion.schema import KeyframeEventType, KeyframeType
from tuj.m5_motion.tests.test_suction_surface_contact import _request, _keyframe, _artifact


def scene_pick():
    request = _request(ee='2F', action_type='PICK_TOOL', dimensions=(.08,.18,.012), object_z=.806,
                       orientation_xyzw=tuple(Rotation.from_euler('z',-90,degrees=True).as_quat()))
    request.task.tool = 'part'
    record = request.world.objects['part']
    record.update(free_joint_name='part_joint', collision_enabled=True,
                  collision_points_m=[list(p) for p in product([-.04,.04],[-.09,.09],[-.006,.006])])
    request.world.obstacles = [{'obstacle_id':'table','aabb_min_m':[-1,-1,.7],'aabb_max_m':[1,1,.8]}]
    grasp = _keyframe(kind=KeyframeType.GRASP,anchor='bottom',offset=.05)
    grasp.events_after = [KeyframeEventType.GRIPPER_CLOSE,KeyframeEventType.ATTACH_OBJECT]
    grasp.metadata = {'event_target_id':'part','event_parameters':{'ATTACH_OBJECT':{'resource_kind':'tool'}}}
    return request,_artifact(grasp)


def test_airborne_scene_tool_grasp_is_replaced_and_close_precedes_attach():
    request,raw = scene_pick()
    original_raw = raw.model_dump(); original_object = copy.deepcopy(request.world.objects['part'])
    original_constraints = request.constraints.model_dump(); original_goal = request.task.goal.model_dump()
    result = bind_scene_tool_contact(raw,request)
    assert result is not raw
    assert raw.model_dump() == original_raw
    assert request.constraints.model_dump() == original_constraints
    assert request.task.goal.model_dump() == original_goal
    unchanged = copy.deepcopy(request.world.objects['part']);unchanged['anchors'].pop(ANCHOR)
    assert unchanged == original_object
    assert len(result.candidates) == 4
    offsets = []
    resolver = RelativePoseResolver(request.world)
    for candidate in result.candidates:
        pre,grasp,lift = candidate.keyframes
        assert [k.keyframe_type for k in candidate.keyframes] == [KeyframeType.PRE_GRASP,KeyframeType.GRASP,KeyframeType.LIFT]
        assert candidate.metadata['contact_candidate_only'] is True
        assert grasp.metadata['contact_geometry_source'] == SOURCE
        assert grasp.events_after == [KeyframeEventType.GRIPPER_CLOSE,KeyframeEventType.ATTACH_OBJECT]
        assert grasp.metadata['event_parameters'] == original_raw['candidates'][0]['keyframes'][0]['metadata']['event_parameters']
        assert grasp.metadata['event_time_offsets_s']['ATTACH_OBJECT'] >= 1.5
        assert grasp.metadata['hold_duration_after_s'] >= grasp.metadata['event_time_offsets_s']['ATTACH_OBJECT']
        assert resolver.resolve(pre).position_m[2] > resolver.resolve(grasp).position_m[2]
        assert resolver.resolve(lift).position_m[2] > resolver.resolve(grasp).position_m[2]
        world_axis = Rotation.from_quat(request.world.objects['part']['pose']['orientation_xyzw']).as_matrix() @ grasp.approach_axis_xyz
        assert world_axis == pytest.approx([0,0,1])
        offsets.append(grasp.offset_along_approach_m)
    assert offsets == pytest.approx([0,.005,.010,.015])


@pytest.mark.parametrize('kind', ['rack_key','dock_metadata','no_free_joint','non_tool','vacuum','free_space'])
def test_docking_and_other_grasp_contracts_are_unchanged(kind):
    request,raw = scene_pick()
    if kind=='rack_key':request.world.rack['part']={'dock_pose':request.world.objects['part']['pose']}
    elif kind=='dock_metadata':request.world.objects['part']['rack_slot']='tool_slot'
    elif kind=='no_free_joint':request.world.objects['part'].pop('free_joint_name')
    elif kind=='non_tool':request.task.action_type='ACQUIRE';request.task.metadata['operation']='ACQUIRE'
    elif kind=='vacuum':request.task.ee='vac'
    elif kind=='free_space':request.world.objects['part']['pose']['position_m'][2]=1.2
    before = request.model_dump()
    assert bind_scene_tool_contact(raw,request) is raw
    assert request.model_dump() == before


def test_existing_later_attachment_time_is_preserved():
    request,raw = scene_pick();g=raw.candidates[0].keyframes[0]
    g.metadata.update(event_time_offsets_s={'GRIPPER_CLOSE':.4,'ATTACH_OBJECT':3.},hold_duration_after_s=4.)
    result = bind_scene_tool_contact(raw,request)
    for c in result.candidates:
        assert c.keyframes[1].metadata['event_time_offsets_s']=={'GRIPPER_CLOSE':.4,'ATTACH_OBJECT':3.}
        assert c.keyframes[1].metadata['hold_duration_after_s']==4.


def test_built_motion_plan_holds_pose_for_real_close_before_attach():
    from tuj.m5_motion.tests.test_plan_builder import _request as builder_request, _node
    from tuj.m5_motion.plan_builder import MotionPlanBuilder
    from tuj.m5_motion.strategy import ConnectedStrategy, EdgePlanResult
    from tuj.m5_motion.schema import ArtifactProvenance, CollisionContext, ModuleName
    request,raw = scene_pick();bound=bind_scene_tool_contact(raw,request)
    grasp=bound.candidates[0].keyframes[1]
    grasp.collision_context_id='contact';grasp.collision_context_after_events_id='held'
    request.world.robot_state=builder_request().world.robot_state
    request.constraints=builder_request().constraints
    connected=ConnectedStrategy(strategy_id='scene-pick',nodes=(_node(grasp,(.2,.1)),),
                                edges=(EdgePlanResult(valid=True,joint_path=((0.,0.),(.2,.1))),),edge_evaluations=1)
    contexts={
        'contact':CollisionContext(context_id='contact',scene_state_id='free',active_ee='2F',collision_model_version='free'),
        'held':CollisionContext(context_id='held',scene_state_id='held',active_ee='2F',attached_object_ids=['part'],collision_model_version='held'),
    }
    plan=MotionPlanBuilder().build(request,connected,plan_id='scene-pick',
        provenance=ArtifactProvenance(artifact_id='plan',artifact_type='MotionPlan',produced_by=ModuleName.MOTION_PLANNER,invocation_id='test'),
        collision_contexts=contexts,initial_collision_context_id='contact',final_segment_validator=lambda waypoints,context:bool(waypoints))
    times={e.event_type.value:e.time_from_start_s for e in plan.events}
    assert times['ATTACH_OBJECT']-times['GRIPPER_CLOSE']==pytest.approx(1.5)
    assert times['GRIPPER_CLOSE']==pytest.approx(plan.segments[0].metadata['motion_end_time_s'])
    assert plan.segments[0].end_time_s==pytest.approx(times['ATTACH_OBJECT'])
    assert plan.segments[0].waypoints[-1].joint_positions_rad==plan.segments[0].waypoints[-2].joint_positions_rad
