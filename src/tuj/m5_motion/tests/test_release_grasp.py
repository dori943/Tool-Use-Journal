from itertools import product
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe, build_catalog_targets
from tuj.m5_motion.scripted_grasps.release_grasp import (
    resolve_release_clearance_targets, shift_targets_for_open_clearance,
)


def cube(half):
    return np.asarray(list(product((-1., 1.), repeat=3))) * half


@pytest.mark.parametrize('rotation', [(0., 0., 0.), (23., -51., 71.)])
def test_clearance_follows_body_frame_and_preserves_approach_lift(rotation):
    recipe = CatalogRecipe('object', 'c4_2', '2F', (.04, .12, .15))
    body = np.eye(4)
    body[:3, :3] = Rotation.from_euler('xyz', rotation, degrees=True).as_matrix()
    body[:3, 3] = [1., -2., .7]
    center = np.array([.02, -.03, .04])
    targets = build_catalog_targets(body, center, recipe.expected_size_m, recipe)
    original = {k: v.copy() for k, v in targets.items()}
    obj = cube(np.array([.02, .06, .075])) + center
    hand = cube(np.array([.02, .04, .01]))
    result, evidence = shift_targets_for_open_clearance(targets, obj, hand, (0., 0., 1.), .005)
    np.testing.assert_allclose(evidence['grasp_shift_m'], .09, atol=1e-12)
    in_body = result['T_BC'] @ result['T_CG']
    actual = hand @ in_body[:3, :3].T + in_body[:3, 3]
    assert actual[:, 2].min() - obj[:, 2].max() == pytest.approx(.005)
    for k in targets:
        np.testing.assert_array_equal(targets[k], original[k])
    for name in ('PRE_GRASP', 'LIFT'):
        np.testing.assert_allclose(result[name][:3, 3] - result['GRASP'][:3, 3],
                                   original[name][:3, 3] - original['GRASP'][:3, 3])
    np.testing.assert_allclose(result['GRASP'][:3, 3] - original['GRASP'][:3, 3],
                               body[:3, 2] * .09)


def test_already_clear_grasp_is_not_moved_and_disabled_policy_needs_no_context():
    recipe = CatalogRecipe('object', 'c4_2', '2F', (.04, .12, .15), offset_fraction=(0., 0., 1.))
    targets = build_catalog_targets(np.eye(4), np.zeros(3), recipe.expected_size_m, recipe)
    adjusted, evidence = shift_targets_for_open_clearance(targets, cube(.02), cube(.01), (0., 0., 1.), .005)
    assert evidence['grasp_shift_m'] == 0.
    np.testing.assert_array_equal(adjusted['GRASP'], targets['GRASP'])
    result, evidence = resolve_release_clearance_targets(NS(recipe=recipe), targets)
    assert result is targets and evidence is None


@pytest.mark.parametrize('axis,margin', [((0., 0., 0.), .005), ((0., 0., 2.), .005),
                                        ((0., 0., 1.), 0.), ((0., 0., 1.), float('nan'))])
def test_invalid_clearance_inputs_are_rejected(axis, margin):
    with pytest.raises(ValueError):
        shift_targets_for_open_clearance({}, cube(.02), cube(.01), axis, margin)


def test_measurement_does_not_mutate_live_physics_or_model(monkeypatch):
    from tuj.m5_motion.scripted_grasps import open_geometry
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="hand" pos="0 0 .5"><site name="grip"/>
        <body><joint name="hinge" type="hinge" axis="0 1 0"/>
          <geom name="finger" type="box" size=".01 .01 .04" mass=".1"/>
        </body>
      </body>
      <body name="object" pos=".3 0 .3"><freejoint/>
        <geom name="object_geom" type="box" size=".02 .06 .075" mass=".1"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qpos[0] = .1
    data.qvel[:] = .02
    data.time = 3.
    mujoco.mj_forward(model, data)
    bid = model.body('object').id
    body = np.eye(4)
    body[:3, 3] = data.xpos[bid]
    recipe = CatalogRecipe('object', 'c4_2', '2F', (.04, .12, .15), open_hand_clearance_axis=(0., 0., 1.))
    c = NS(recipe=recipe, mj=mujoco, model=model, data=data, site_id=model.site('grip').id,
           object_geoms={model.geom('object_geom').id}, gripper_geoms={model.geom('finger').id},
           body_pose=lambda: body.copy(), request_collision_margin_m=.008)
    monkeypatch.setattr(open_geometry, 'open_joint_positions', lambda _: {'hinge': .4})
    saved = {name: getattr(data, name).copy() for name in ('qpos', 'qvel', 'ctrl', 'act')}
    model_saved = model.geom_size.copy()
    targets = build_catalog_targets(body, np.zeros(3), recipe.expected_size_m, recipe)
    _, evidence = resolve_release_clearance_targets(c, targets)
    assert evidence['clearance_m'] == .008
    for name, value in saved.items():
        np.testing.assert_array_equal(getattr(data, name), value)
    assert data.time == 3.
    np.testing.assert_array_equal(model.geom_size, model_saved)
