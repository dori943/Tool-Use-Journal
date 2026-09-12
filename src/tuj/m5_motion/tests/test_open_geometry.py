from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
from tuj.m5_motion.scripted_grasps.open_geometry import open_joint_positions


class Hand:
    dof=1
    speed=.01
    joints=['active','passive']
    def __init__(self): self.current_action=np.array([0.])
    def format_action(self, action):
        self.current_action=np.clip(self.current_action-self.speed*np.sign(action),-1,1)
        return self.current_action


def context(passive_max=1.):
    model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
    <body><joint name="active" range="0 1" limited="true"/>
      <geom type="sphere" size=".01"/><body pos=".1 0 0">
      <joint name="passive" range="0 {passive_max}" limited="true"/>
      <geom type="sphere" size=".01"/></body></body></worldbody>
    <compiler angle="radian"/><equality><joint joint1="passive" joint2="active" polycoef="0 .5 0 0 0"/></equality>
    <actuator><position joint="active" kp="10" ctrlrange="0 1"/></actuator></mujoco>''')
    data=mujoco.MjData(model);data.qpos[:]=[.2,.1]
    return SimpleNamespace(model=model,data=data,gripper=Hand(),gripper_actuator_ids=[0])


def test_public_endpoint_resolves_passive_linkage_without_live_mutation():
    c=context();before=c.data.qpos.copy();action=c.gripper.current_action.copy()
    result=open_joint_positions(c)
    assert result['active']==pytest.approx(1.)
    assert result['passive']==pytest.approx(.5,abs=1e-8)
    np.testing.assert_array_equal(c.data.qpos,before)
    np.testing.assert_array_equal(c.gripper.current_action,action)


def test_impossible_passive_range_fails_instead_of_using_reset_pose():
    with pytest.raises(ValueError,match='LINKAGE_NOT_SOLVED'):
        open_joint_positions(context(.1))


def test_unsupported_transmission_fails_explicitly():
    c=context();c.model.actuator_gear[0,0]=2.
    with pytest.raises(ValueError,match='UNSUPPORTED_POSITION_TRANSMISSION'):
        open_joint_positions(c)


@pytest.mark.parametrize('boundary', ['lower', 'upper'])
def test_open_linkage_converges_at_joint_boundary_from_grasped_state(boundary):
    c=context(.5 if boundary == 'upper' else 1.)
    c.data.qpos[:]=[.8,.4]
    if boundary == 'lower':
        c.model.actuator_ctrlrange[0]=[-1.,0.]
    before=c.data.qpos.copy()
    result=open_joint_positions(c)
    expected=0. if boundary == 'lower' else .5
    assert result['passive']==pytest.approx(expected,abs=1e-8)
    assert result['passive']==pytest.approx(.5*result['active'],abs=1e-8)
    np.testing.assert_array_equal(c.data.qpos,before)
