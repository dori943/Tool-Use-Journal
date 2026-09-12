from types import SimpleNamespace as NS
import numpy as np
import pytest
import mujoco
from scipy.spatial.transform import Rotation
from tuj.m5_motion.scripted_grasps.release_corridor import OpenHandDropCorridor


def context(rotation=None, translation=None):
    rotation=np.eye(3) if rotation is None else rotation
    translation=np.zeros(3) if translation is None else translation
    q=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
    hand=translation+rotation@np.array([0.,0.,-.06])
    model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
    <geom name="hand" type="box" size=".02 .02 .01" pos="{' '.join(map(str,hand))}" quat="{' '.join(map(str,q))}"/>
    <body name="item" pos="{' '.join(map(str,translation))}" quat="{' '.join(map(str,q))}"><freejoint/><geom name="item_geom" type="sphere" size=".01"/></body>
    </worldbody></mujoco>''')
    model.opt.gravity[:]=rotation@np.array([0.,0.,-9.81])
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    return NS(mj=mujoco,model=model,data=data,body_id=model.body('item').id,
              gripper=NS(joints=[]),gripper_actuator_ids=[],gripper_geoms={model.geom('hand').id},object_geoms={model.geom('item_geom').id})


@pytest.mark.parametrize('rotation', [np.eye(3), Rotation.from_euler('xyz',[.3,-.7,1.1]).as_matrix()])
def test_actual_narrowphase_detects_drop_obstruction_and_clear_orientation(rotation):
    c=context(rotation,np.array([1.2,-.8,2.1]));query=OpenHandDropCorridor(c,.005,.003)
    blocked=query.evaluate(rotation)
    clear=query.evaluate(rotation@Rotation.from_euler('y',np.pi/2).as_matrix())
    assert not blocked['clear'] and blocked['minimum']['distance_m'] < 0
    assert clear['clear']
    for result in (blocked,clear):
        assert result['sample_interval_m']<=.003+1e-12
        assert result['required_sample_distance_m']==pytest.approx(.005+result['sample_interval_m']/2)


def test_queries_preserve_all_live_state_and_model_collision_settings():
    c=context();c.data.qvel[:]=np.arange(c.model.nv)*.001;c.data.time=3.25
    state={k:getattr(c.data,k).copy() for k in ('qpos','qvel','ctrl')}
    settings={k:getattr(c.model,k).copy() for k in ('geom_contype','geom_conaffinity','geom_margin','geom_gap')}
    query=OpenHandDropCorridor(c,.005,.003)
    first=query.evaluate(np.eye(3));query.evaluate(Rotation.from_euler('x',.8).as_matrix())
    assert query.evaluate(np.eye(3))==first
    for k,v in state.items():np.testing.assert_array_equal(getattr(c.data,k),v)
    for k,v in settings.items():np.testing.assert_array_equal(getattr(c.model,k),v)
    assert c.data.time==3.25


@pytest.mark.parametrize('gravity', [[0,0,0],[0,0,float('nan')]])
def test_missing_gravity_does_not_invent_a_drop_direction(gravity):
    c=context();c.model.opt.gravity[:]=gravity
    with pytest.raises(ValueError,match='GRAVITY_REQUIRED'):OpenHandDropCorridor(c,.005,.005)


@pytest.mark.parametrize('margin,tolerance', [(-.001,.005),(float('nan'),.005),(.005,0),(.005,float('inf'))])
def test_invalid_clearance_rejected(margin,tolerance):
    with pytest.raises(ValueError,match='INVALID_CLEARANCE'):OpenHandDropCorridor(context(),margin,tolerance)


def test_zero_requested_margin_still_samples_and_rejects_intersection():
    result=OpenHandDropCorridor(context(),0.,.005).evaluate(np.eye(3))
    assert not result['clear'] and result['sample_interval_m']<=.005


def test_missing_geometry_fails_closed():
    c=context();c.object_geoms=set()
    with pytest.raises(ValueError,match='GEOMETRY_REQUIRED'):OpenHandDropCorridor(c,.005,.005)


def test_nonfinite_distance_fails_closed(monkeypatch):
    query=OpenHandDropCorridor(context(),.005,.005)
    monkeypatch.setattr(mujoco,'mj_geomDistance',lambda *args:float('nan'))
    with pytest.raises(ValueError,match='NONFINITE_DISTANCE'):query.evaluate(np.eye(3))


def test_invalid_destination_rotation_rejected():
    query=OpenHandDropCorridor(context(),.005,.005)
    with pytest.raises(ValueError,match='RIGID_ROTATION'):query.evaluate(np.zeros((3,3)))


def test_corridor_rejection_happens_before_ik_and_remains_explicit(monkeypatch):
    from tuj.m5_motion.tests.test_container_orientation import fixture
    from tuj.m5_motion.scripted_grasps import container_orientation as module
    g,_=fixture(.7);g.task=NS(action_type='TRANSPORT',metadata={});g.retention=NS(context=context())
    monkeypatch.setattr('tuj.m5_motion.scripted_grasps.release_corridor.OpenHandDropCorridor',
                        lambda *args:NS(evaluate=lambda rotation:{'clear':False,'minimum':{'distance_m':-.001}}))
    calls=[];monkeypatch.setattr(module,'packing_transport_has_ik',lambda *args:calls.append(args))
    with pytest.raises(ValueError,match='NO_OPEN_HAND_RELEASE_CORRIDOR'):module.configure_packing_orientation(g)
    assert not calls and g.task.metadata['packing_release_corridors']
