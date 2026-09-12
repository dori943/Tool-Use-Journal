from itertools import product
from types import SimpleNamespace as NS
import numpy as np
import pytest
import mujoco
from scipy.spatial.transform import Rotation
from tuj.m5_motion.scripted_grasps.packing_occupancy import packing_overlap_preference


def scene(monkeypatch, yaw):
    region = np.eye(4)
    region[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    region[:3, 3] = [.2, -.3, .8]
    position = region[:3, 3] + region[:3, :3] @ [.06, 0., .1]
    quat = Rotation.from_euler('z', yaw).as_quat()[[3, 0, 1, 2]]
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="target" pos="1 1 1"><freejoint/><geom type="box" size=".08 .01 .01"/></body>
      <body name="occupant" pos="{' '.join(map(str, position))}" quat="{' '.join(map(str, quat))}">
        <geom type="box" size=".02 .02 .1"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    context = NS(model=model, data=data, body_id=1, object_geoms={0},
                 env=NS(obj_body_id={'occupied': 2}), descendant=lambda child, root: child == root)
    points = np.array(list(product([-.08, .08], [-.01, .01], [-.01, .01])))
    g = NS(retention=NS(context=context), task=NS(metadata={}),
           record={'packing_metadata': {'interior_center_m': [0, 0, .1],
                                        'interior_dimensions_m': [.3, .3, .2]}},
           request=NS(constraints=NS(collision_margin_m=.005, position_tolerance_m=.005)),
           T_WR=region, destination_rotation=region[:3, :3].copy(),
           center_in_body=np.zeros(3), packing_bounds=(points.min(0), points.max(0)))
    monkeypatch.setattr('tuj.m5_motion.scripted_grasps.transport.transport_destination_center',
                        lambda g: region[:3, 3] + region[:3, :3] @ [0, 0, .3])
    return g, points


@pytest.mark.parametrize('yaw', [0., .7, -1.1])
def test_real_geometry_ranks_clear_direction_and_preserves_live_state(monkeypatch, yaw):
    g, points = scene(monkeypatch, yaw)
    c = g.retention.context
    before = {f: getattr(c.data, f).copy() for f in ('qpos', 'qvel', 'ctrl', 'act', 'qacc', 'xpos')}
    model_before = {f: getattr(c.model, f).copy() for f in ('geom_pos', 'geom_size', 'geom_contype', 'body_pos')}
    time = c.data.time
    blocked = packing_overlap_preference(g, ['occupied'])
    local = Rotation.from_euler('z', np.pi / 2).as_matrix()
    g.destination_rotation = g.T_WR[:3, :3] @ local
    rotated = points @ local.T
    g.packing_bounds = (rotated.min(0), rotated.max(0))
    clear = packing_overlap_preference(g, ['occupied'])
    assert blocked > .01
    assert clear == 0.
    for f, original in before.items():
        np.testing.assert_array_equal(getattr(c.data, f), original)
    for f, original in model_before.items():
        np.testing.assert_array_equal(getattr(c.model, f), original)
    assert c.data.time == time
    assert g.task.metadata['packing_occupancy_preferences'][0]['sample_count'] > 2


@pytest.mark.parametrize('resolution,error', [(0., 'INVALID_RESOLUTION'),
                                               (float('nan'), 'INVALID_RESOLUTION'),
                                               (1e-9, 'SAMPLE_BUDGET')])
def test_invalid_or_unbounded_sampling_fails_explicitly(monkeypatch, resolution, error):
    g, _ = scene(monkeypatch, 0.)
    g.request.constraints.position_tolerance_m = resolution
    with pytest.raises(ValueError, match=error):
        packing_overlap_preference(g, ['occupied'])


def _isolate_ranking_from_release_corridor(monkeypatch, g):
    # These ranking unit tests isolate IK and occupancy scoring. The physical
    # release gate has real MuJoCo geometry coverage in test_release_corridor.
    class ClearCorridor:
        def __init__(self, context, margin, resolution):
            pass

        def evaluate(self, rotation):
            return {"clear": True}

    g.retention = NS(context=object())
    monkeypatch.setattr(
        'tuj.m5_motion.scripted_grasps.release_corridor.OpenHandDropCorridor',
        ClearCorridor,
    )


def test_preference_cannot_select_unreachable_orientation(monkeypatch):
    from tuj.m5_motion.tests.test_container_orientation import fixture
    from tuj.m5_motion.scripted_grasps import container_orientation, packing_occupancy
    g, _ = fixture(0.)
    g.task = NS(action_type='TRANSPORT', metadata={})
    _isolate_ranking_from_release_corridor(monkeypatch, g)
    reached = []; scored = []
    monkeypatch.setattr(packing_occupancy, 'packing_occupants', lambda g: ['occupied'])
    def ik(probe):
        reached.append(probe.destination_rotation.copy())
        return len(reached) != 2
    def score(probe, occupants):
        scored.append(probe.destination_rotation.copy())
        return 1. / len(scored)
    monkeypatch.setattr(container_orientation, 'packing_transport_has_ik', ik)
    monkeypatch.setattr(packing_occupancy, 'packing_overlap_preference', score)
    container_orientation.configure_packing_orientation(g)
    assert len(reached) == 4 and len(scored) == 3
    np.testing.assert_allclose(g.destination_rotation, scored[-1])
    assert not any(np.allclose(reached[1], value) for value in scored)


def test_equal_scores_preserve_closest_orientation(monkeypatch):
    from tuj.m5_motion.tests.test_container_orientation import fixture
    from tuj.m5_motion.scripted_grasps import container_orientation, packing_occupancy
    g, _ = fixture(.3)
    g.task = NS(action_type='TRANSPORT', metadata={})
    _isolate_ranking_from_release_corridor(monkeypatch, g)
    calls = []
    monkeypatch.setattr(packing_occupancy, 'packing_occupants', lambda g: ['occupied'])
    def ik(probe):
        calls.append(probe.destination_rotation.copy())
        return True
    monkeypatch.setattr(container_orientation, 'packing_transport_has_ik', ik)
    monkeypatch.setattr(packing_occupancy, 'packing_overlap_preference', lambda *args: 0.)
    container_orientation.configure_packing_orientation(g)
    assert len(calls) == 4
    angles = [Rotation.from_matrix(g.T_WB[:3, :3].T @ rotation).magnitude()
              for rotation in calls]
    chosen_angle = Rotation.from_matrix(g.T_WB[:3, :3].T @ g.destination_rotation).magnitude()
    assert chosen_angle == pytest.approx(min(angles))
