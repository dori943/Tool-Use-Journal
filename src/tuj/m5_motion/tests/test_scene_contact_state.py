from dataclasses import replace
from types import SimpleNamespace as NS
import mujoco
import numpy as np
import pytest
from tuj.m5_motion.tool_use_journal_runtime import _capture_runtime_state, _restore_runtime_state
from tuj.m5_motion.scene_contact_state import capture_scene_solref, restore_scene_solref


def environment(*, reverse=False, object_type='sphere', object_body='item', hand_solref='.02 1'):
    item=f'<body name="{object_body}" pos=".4 0 .5"><freejoint name="item_free"/><geom name="item_geom" type="{object_type}" size=".03 .03 .03"/></body>'
    robot=f'<body name="robot_root"><joint name="arm"/><geom name="robot_geom" size=".03"/><body name="hand" pos="0 0 .2"><geom name="hand_geom" size=".02" solref="{hand_solref}"/></body></body>'
    bodies=robot+item if reverse else item+robot
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="support" type="plane" size="1 1 .1"/>'+bodies+'</worldbody><actuator><motor name="arm_motor" joint="arm"/></actuator></mujoco>')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    env=NS(sim=NS(model=NS(_model=m),data=NS(_data=d)),robots=[NS(robot_model=NS(root_body='robot_root'))])
    return env,m,d


def test_scene_calibration_survives_reordered_model_and_fresh_hand_stays_fresh():
    old,m,d=environment();new,n,e=environment(reverse=True,hand_solref='.006 1')
    for name in ('support','item_geom','robot_geom','hand_geom'):m.geom_solref[m.geom(name).id]=[.004,1]
    d.qpos[m.jnt_qposadr[m.joint('arm').id]]=.25
    d.qvel[:]=np.arange(m.nv)*.003;d.ctrl[:]=.12;d.time=4.75
    unrelated={name:getattr(n,name).copy() for name in ('geom_solimp','geom_friction','geom_contype','geom_conaffinity','geom_margin','geom_pos','geom_size','body_mass','actuator_gainprm')}
    source={name:getattr(d,name).copy() for name in ('qpos','qvel','ctrl')}
    state=_capture_runtime_state(old)
    assert set(state.scene_geom_solref)=={'support','item_geom'}
    assert m.geom('item_geom').id != n.geom('item_geom').id
    _restore_runtime_state(new,state);restored=_capture_runtime_state(new)
    assert restored==state
    np.testing.assert_array_equal(n.geom_solref[n.geom('hand_geom').id],[.006,1])
    np.testing.assert_array_equal(n.geom_solref[n.geom('robot_geom').id],[.02,1])
    for name,value in unrelated.items():np.testing.assert_array_equal(getattr(n,name),value)
    for name,value in source.items():np.testing.assert_array_equal(getattr(d,name),value)


@pytest.mark.parametrize('change', [{'object_type':'box'},{'object_body':'different_item'}])
def test_changed_common_identity_fails_before_mutating_new_contact_settings(change):
    old,m,_=environment();new,n,_=environment(**change)
    m.geom_solref[m.geom('support').id]=[.004,1]
    saved=capture_scene_solref(old,m);before=n.geom_solref.copy()
    with pytest.raises(ValueError,match='IDENTITY_CHANGED'):restore_scene_solref(new,n,saved)
    np.testing.assert_array_equal(n.geom_solref,before)


def test_legacy_state_only_checkpoint_does_not_invent_calibration_history():
    old,m,_=environment();new,n,_=environment()
    m.geom_solref[m.geom('item_geom').id]=[.004,1]
    state=replace(_capture_runtime_state(old),scene_geom_solref={})
    _restore_runtime_state(new,state)
    np.testing.assert_array_equal(n.geom_solref[n.geom('item_geom').id],[.02,1])


def test_missing_robot_root_does_not_copy_robot_as_scene():
    env,m,_=environment();env.robots[0].robot_model.root_body='missing'
    with pytest.raises(ValueError,match='ROBOT_ROOT_NOT_FOUND'):capture_scene_solref(env,m)


def test_nonfinite_scene_reference_fails_closed():
    env,m,_=environment();m.geom_solref[m.geom('item_geom').id,0]=float('nan')
    with pytest.raises(ValueError,match='NONFINITE_SOLREF'):capture_scene_solref(env,m)
