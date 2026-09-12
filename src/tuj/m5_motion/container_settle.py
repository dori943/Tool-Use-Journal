"""Bounded post-release settling with the unchanged container goal."""
from copy import deepcopy
from types import SimpleNamespace
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .push_to_region import target_fully_inside_region


def configure_container_settle(request, plan):
    goal=request.task.goal
    region_id=goal.target_region_id
    region=request.world.objects.get(region_id, {})
    if (not request.task.metadata.get('held_place_goal')
            or region.get('packing_metadata', {}).get('kind') != 'CONTAINER'):
        return
    target=goal.target_object_id
    releases=[e for e in plan.events if e.event_type.value == 'DETACH_OBJECT'
              and e.target_id == target]
    if not releases or not plan.segments or plan.segments[-1].segment_type.value != 'RETREAT':
        raise ValueError('CONTAINER_SETTLE_RELEASE_RETREAT_REQUIRED')
    timeout=float(request.world.metadata['free_object_settle_seconds'])
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('CONTAINER_SETTLE_TIMEOUT_REQUIRED')
    targets={target}
    for name,record in request.world.objects.items():
        if (record.get('packing_metadata', {}).get('kind') == 'PACKABLE_OBJECT'
                and target_fully_inside_region(request.world,target_id=name,
                     region_id=region_id,include_vertical=True)):
            targets.add(name)
    segment=plan.segments[-1]
    tracking=dict(segment.metadata.get('tracking_settle', {}))
    tracking.setdefault('eef_tolerance_m',request.constraints.position_tolerance_m)
    tracking.setdefault('eef_orientation_tolerance_rad',request.constraints.orientation_tolerance_rad)
    tracking['max_wait_s']=timeout
    segment.metadata['tracking_settle']=tracking
    segment.metadata['container_settle']={
        'region_id':region_id,'target_ids':sorted(targets),
        'objects':{n:deepcopy(request.world.objects[n]) for n in targets|{region_id}},
        'position_tolerance_m':request.constraints.position_tolerance_m,
        'orientation_tolerance_rad':request.constraints.orientation_tolerance_rad,
    }


def evaluate_container_settle(config, poses, previous, *, attached=False, hand_contacts=0):
    objects=deepcopy(config['objects'])
    stable=previous is not None
    max_position=0.;max_angle=0.
    for name,pose in poses.items():
        objects[name]['pose']=pose
        if previous is not None and name in config['target_ids']:
            delta=float(np.linalg.norm(np.array(pose['position_m'])-previous[name]['position_m']))
            angle=float((Rotation.from_quat(previous[name]['orientation_xyzw']).inv()
                         *Rotation.from_quat(pose['orientation_xyzw'])).magnitude())
            max_position=max(max_position,delta);max_angle=max(max_angle,angle)
    stable=stable and max_position<=config['position_tolerance_m'] and max_angle<=config['orientation_tolerance_rad']
    world=SimpleNamespace(objects=objects)
    outside=[name for name in config['target_ids'] if not target_fully_inside_region(
        world,target_id=name,region_id=config['region_id'],include_vertical=True)]
    return {'succeeded':not outside and stable and not attached and hand_contacts==0,
            'outside_target_ids':outside,'stable':stable,'attached':attached,
            'hand_contact_count':hand_contacts,'max_position_delta_m':max_position,
            'max_orientation_delta_rad':max_angle}


def measured_container_settle(runtime, config, state):
    from .tool_use_journal import _raw_model_data
    model,data=_raw_model_data(runtime.env)
    poses={};target_bodies=set()
    for name in config['objects']:
        bid=int(runtime.env.obj_body_id[name])
        poses[name]={'frame_id':'world','position_m':data.xpos[bid].tolist(),
                     'orientation_xyzw':Rotation.from_matrix(data.xmat[bid].reshape(3,3)).as_quat().tolist()}
        if name in config['target_ids']:
            for child in range(model.nbody):
                ancestor=child
                while ancestor>0 and ancestor!=bid:ancestor=int(model.body_parentid[ancestor])
                if ancestor==bid:target_bodies.add(child)
    prefix=runtime.env.robots[0].gripper['right'].naming_prefix
    hand_contacts=0
    for contact in data.contact[:data.ncon]:
        if contact.dist>0:continue
        a,b=int(contact.geom1),int(contact.geom2)
        if a<0 or b<0:continue
        hand_contacts+=int((int(model.geom_bodyid[a]) in target_bodies and (model.geom(b).name or '').startswith(prefix))
                          or (int(model.geom_bodyid[b]) in target_bodies and (model.geom(a).name or '').startswith(prefix)))
    result=evaluate_container_settle(config,poses,state.get('container_previous_poses'),
        attached=runtime.attached_object_id in config['target_ids'],hand_contacts=hand_contacts)
    state['container_previous_poses']=poses
    return result
