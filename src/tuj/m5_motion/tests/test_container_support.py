from copy import deepcopy
from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
from tuj.m5_motion.scripted_grasps.placement_contacts import measure_contacts
from tuj.m5_motion.scripted_grasps.container_support import (
    world_digest, contact_support_path, evaluate_preserved_placement)


def scene():
    model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001"/>
      <worldbody><geom name="floor" type="box" size=".2 .2 .01"/>
      <body name="lower" pos="0 0 .04"><freejoint/>
        <geom name="lower_geom" type="box" size=".08 .08 .03" mass=".2"/></body>
      <body name="upper" pos="0 0 .10"><freejoint/>
        <geom name="upper_geom" type="box" size=".04 .04 .03" mass=".1"/></body>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model)
    for _ in range(1500):mujoco.mj_step(model,data)
    objects={'box':{'pose':{'position_m':[0,0,0],'orientation_xyzw':[0,0,0,1]},
        'dimensions_m':[.4,.4,.3],'geom_names':['floor'],
        'packing_metadata':{'kind':'CONTAINER','interior_center_m':[0,0,.16],
                            'interior_dimensions_m':[.4,.4,.3]}}}
    for name,size in [('lower',[.16,.16,.06]),('upper',[.08,.08,.06])]:
        objects[name]={'pose':{'position_m':data.body(name).xpos.tolist(),
            'orientation_xyzw':[0,0,0,1]},'dimensions_m':size,'geom_names':[name+'_geom'],
            'body_name':name,'packing_metadata':{'kind':'PACKABLE_OBJECT'}}
    world={'objects':objects,'robot_state':{'attached_object_id':None,'held_tool_id':None}}
    pairs=[{'target':a,'region':b} for a in ['lower','upper'] for b in ['box','lower','upper'] if a!=b]
    evidence={'world_sha256':world_digest(world),'basis':'CURRENT_RUNTIME_CONTACT_FORCES',
              'pairs':measure_contacts(model,data,objects,pairs)}
    return model,data,world,evidence


def test_actual_stacked_contacts_form_support_path_without_changing_physics():
    model,data,world,evidence=scene();qpos=data.qpos.copy();qvel=data.qvel.copy()
    passed,detail=contact_support_path(world,'upper','box',evidence)
    assert passed and detail['support_path']==['upper','lower','box']
    assert evaluate_preserved_placement(world,'upper','box',evidence)['passed'] is True
    np.testing.assert_array_equal(data.qpos,qpos);np.testing.assert_array_equal(data.qvel,qvel)


@pytest.mark.parametrize('bad', ['floating','side','penetrating','robot','cycle'])
def test_missing_upward_grounded_contact_path_is_rejected(bad):
    _,_,world,evidence=scene()
    if bad=='floating':evidence['pairs']=[]
    elif bad=='side':
        for p in evidence['pairs']:
            for c in p['contacts']:c['upward_force_n']=0.
    elif bad=='penetrating':
        for p in evidence['pairs']:
            for c in p['contacts']:c['distance_m']=-.006
    elif bad=='robot':
        for p in evidence['pairs']:
            if p['region']=='box':p['region']='robot'
    else:
        for p in evidence['pairs']:
            if p['region']=='box':p['region']='upper'
    assert contact_support_path(world,'upper','box',evidence)[0] is False


def test_missing_or_stale_evidence_is_unknown_and_held_peer_cannot_support():
    _,_,world,evidence=scene()
    assert contact_support_path(world,'upper','box',None)[0] is None
    changed=deepcopy(world);changed['objects']['upper']['pose']['position_m'][2]+=.01
    assert contact_support_path(changed,'upper','box',evidence)[0] is None
    world['robot_state']['attached_object_id']='lower'
    evidence['world_sha256']=world_digest(world)
    assert contact_support_path(world,'upper','box',evidence)[0] is False


def test_container_requires_vertical_inclusion_even_with_contact_evidence():
    _,_,world,evidence=scene()
    world['objects']['upper']['pose']['position_m'][2]=.5
    evidence['world_sha256']=world_digest(world)
    result=evaluate_preserved_placement(world,'upper','box',evidence)
    assert result['passed'] is False
    assert next(c for c in result['checks'] if c['name']=='contained')['passed'] is False


@pytest.mark.parametrize('frictional', [True, False])
def test_wall_support_requires_actual_upward_friction_force(frictional):
    from tuj.m5_motion.tests.test_placement_contacts import wall_friction_scene
    model, data, objects = wall_friction_scene(frictional=frictional)
    objects['walls'].update(pose={'position_m': [0, 0, 0], 'orientation_xyzw': [0, 0, 0, 1]},
        dimensions_m=[.2, .4, .4], packing_metadata={'kind': 'CONTAINER',
        'interior_center_m': [0, 0, .2], 'interior_dimensions_m': [.2, .4, .4]})
    objects['payload'].update(pose={'position_m': data.body('payload').xpos.tolist(),
        'orientation_xyzw': [0, 0, 0, 1]}, dimensions_m=[.1, .08, .08],
        packing_metadata={'kind': 'PACKABLE_OBJECT'})
    world={'objects': objects, 'robot_state': {'attached_object_id': None, 'held_tool_id': None}}
    evidence={'world_sha256': world_digest(world), 'basis': 'CURRENT_RUNTIME_CONTACT_FORCES',
        'pairs': measure_contacts(model, data, objects, [{'target': 'payload', 'region': 'walls'}])}
    passed, detail = contact_support_path(world, 'payload', 'walls', evidence)
    assert passed is frictional
    assert evaluate_preserved_placement(world, 'payload', 'walls', evidence)['passed'] is frictional
