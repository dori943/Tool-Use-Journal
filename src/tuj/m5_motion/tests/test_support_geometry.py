from types import SimpleNamespace as NS
import numpy as np
import pytest
from tuj.m5_motion.scripted_grasps.support_geometry import measured_support_geoms
from tuj.m5_motion.scripted_grasps.catalog_runtime import CatalogContext
from tuj.m5_motion.scripted_grasps.spoon_runtime import SpoonContext


def context(contacts):
    body=np.eye(4);body[2,3]=.2
    return NS(model=NS(opt=NS(gravity=[0,0,-9.81]),geom_bodyid=[1,0,2,3],body_weldid=[0,1,2,0]),
        data=NS(contact=contacts,ncon=len(contacts)),object_geoms={0},robot_geoms=set(),gripper_geoms=set(),
        center_in_body=np.zeros(3),body_pose=lambda:body)


def contact(a,b,normal,dist=-.001):
    return NS(geom1=a,geom2=b,dist=dist,frame=np.r_[normal,[1,0,0],[0,1,0]],pos=np.array([0.,0.,0.]))


@pytest.mark.parametrize('pair,normal',[((1,0),[0,0,1]),((0,1),[0,0,-1])])
def test_actual_support_is_independent_of_pair_order(pair,normal):
    c=context([contact(*pair,normal)])
    assert measured_support_geoms(c)=={1}


def test_side_dynamic_and_noncontact_surfaces_are_not_supports():
    c=context([contact(1,0,[1,0,0]),contact(2,0,[0,0,1]),contact(3,0,[0,0,1],dist=.0001)])
    assert measured_support_geoms(c)==set()


def test_only_measured_tiles_join_support_set():
    c=context([contact(1,0,[0,0,1]),contact(3,0,[0,0,1])])
    assert measured_support_geoms(c)=={1,3}


@pytest.mark.parametrize('height,released,depth,allowed',[(0.,False,.0012,True),
    (0.,False,.0021,False),(.006,False,.0012,False),(0.,True,.0012,False)])
def test_existing_lift_bounds_and_recontact_gate_remain(monkeypatch,height,released,depth,allowed):
    contacts=[{'geoms':['actual_tile','rod'],'penetration_m':depth},
              {'geoms':['unverified_tile','rod'],'penetration_m':.0012}]
    monkeypatch.setattr(SpoonContext,'bad_contacts',lambda *args:contacts)
    c=CatalogContext();c.support_gid=1;c.support_gids={1};c.object_geoms={0};c.support_released=released
    c.model=NS(geom=lambda gid:NS(name={0:'rod',1:'actual_tile'}[gid]));c.initial_body=np.eye(4)
    body=np.eye(4);body[2,3]=height;c.body_pose=lambda *args:body
    bad=c.bad_contacts(None,'LIFT')
    assert (contacts[0] not in bad)==allowed
    assert contacts[1] in bad


def test_fixed_child_of_free_body_is_not_welded_to_world():
    import mujoco
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="moving"><freejoint/><geom type="box" size=".1 .1 .1"/><body name="child"><geom type="box" size=".01 .01 .01"/></body></body><body name="fixed"><geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>')
    assert m.body_weldid[m.body('child').id]!=0
    assert m.body_weldid[m.body('fixed').id]==0
