from dataclasses import replace
from itertools import product
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tuj.m5_motion.scripted_grasps.catalog_types import CatalogRecipe, build_catalog_targets
from tuj.m5_motion.scripted_grasps.endpoint_grasp import lift_targets_above_support, resolve_endpoint_support_clearance


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('policy,sign', [('MIN_LONG_AXIS', -1), ('MAX_LONG_AXIS', 1)])
def test_endpoint_uses_measured_extent_and_region_with_rotated_offset_body(axis, policy, sign):
    size = np.array([.04, .04, .04]); size[axis] = .2
    recipe = CatalogRecipe('handle', 'c4_2', '2F', tuple(size), offset_m=(.003, .004, .005),
                           contact_region_min=(-.4, -.3, -.2), contact_region_max=(.2, .3, .4))
    body = np.eye(4); body[:3, :3] = Rotation.from_euler('xyz', [.2, -.4, .8]).as_matrix(); body[:3, 3] = [1, 2, 3]
    center = [.013, -.024, .006]
    old = build_catalog_targets(body, center, size, recipe)
    new = build_catalog_targets(body, center, size, replace(recipe, contact_region_endpoint_policy=policy))
    endpoint = (recipe.contact_region_min if sign < 0 else recipe.contact_region_max)[axis]
    expected = np.array(recipe.offset_m); expected[axis] = size[axis] * endpoint
    np.testing.assert_allclose(new['T_CG'][:3, 3], expected)
    delta = body[:3, :3] @ (expected - np.array(recipe.offset_m))
    for name in ('GRASP', 'PRE_GRASP', 'LIFT'):
        np.testing.assert_allclose(new[name][:3, 3] - old[name][:3, 3], delta, atol=1e-14)
        np.testing.assert_array_equal(new[name][:3, :3], old[name][:3, :3])
    np.testing.assert_allclose(new['T_WC'] @ new['T_CG'], new['GRASP'], atol=1e-14)


def test_default_target_formula_unchanged():
    r = CatalogRecipe('handle', 'c4_2', '2F', (.04, .2, .04), offset_fraction=(0, -.28, 0), offset_m=(0, 0, .006))
    t = build_catalog_targets(np.eye(4), [.01, .02, .03], r.expected_size_m, r)
    np.testing.assert_allclose(t['GRASP'][:3, 3], [.01, .02-.056, .036])
    assert r.contact_region_endpoint_policy is None


@pytest.mark.parametrize('size', [[.04, float('nan'), .04], [.04, 0., .04]])
def test_invalid_endpoint_geometry_rejected(size):
    r = CatalogRecipe('handle', 'c4_2', '2F', (.04, .2, .04), contact_region_endpoint_policy='MIN_LONG_AXIS')
    with pytest.raises(ValueError): build_catalog_targets(np.eye(4), [0, 0, 0], size, r)


@pytest.mark.parametrize('policy', ['BAD', 'min_long_axis'])
def test_invalid_policy_rejected(policy):
    with pytest.raises(ValueError): CatalogRecipe('x', 'c4_2', '2F', (.04, .2, .04), contact_region_endpoint_policy=policy)


@pytest.mark.parametrize('height', [.7, 1.1])
def test_support_lift_is_minimal_world_vertical_and_transform_coherent(height):
    r = CatalogRecipe('handle', 'c4_2', '2F', (.04, .2, .04), contact_region_endpoint_policy='MIN_LONG_AXIS')
    b = np.eye(4); b[:3, :3] = Rotation.from_euler('xyz', [.4, -.5, .2]).as_matrix(); b[:3, 3] = [2, -3, height]
    targets = build_catalog_targets(b, [.01, -.02, .03], r.expected_size_m, r)
    original = {k:v.copy() for k,v in targets.items()}
    points = np.array(list(product([-.04,.04], [-.06,.06], [-.08,.08])))
    result, evidence = lift_targets_above_support(targets, points, .8, .005)
    minimum = float((points @ result['GRASP'][:3,:3].T + result['GRASP'][:3,3])[:,2].min())
    assert minimum >= .805 - 1e-12
    if evidence['world_up_translation_m'] > 0: assert minimum == pytest.approx(.805)
    for name in ('GRASP', 'PRE_GRASP', 'LIFT'):
        np.testing.assert_allclose(result[name][:3,3]-targets[name][:3,3], [0,0,evidence['world_up_translation_m']], atol=1e-14)
    np.testing.assert_allclose(result['T_WC'] @ result['T_CG'], result['GRASP'], atol=1e-14)
    for k in original: np.testing.assert_array_equal(targets[k],original[k])


@pytest.mark.parametrize('support,margin', [(float('nan'), .005), (.8, float('nan')), (.8, 0.)])
def test_invalid_support_rejected(support, margin):
    with pytest.raises(ValueError): lift_targets_above_support({}, np.zeros((4,3)), support, margin)


def test_live_preshaped_geometry_measurement_does_not_mutate_state():
    import mujoco
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body pos="1 2 .8" euler=".2 .3 .4"><freejoint/>
    <geom type="box" size=".03 .04 .05" pos=".01 -.02 .03"/><site name="grip" pos=".02 .01 .06"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model); mujoco.mj_forward(model,data)
    grip=np.eye(4);grip[:3,:3]=data.site_xmat[0].reshape(3,3);grip[:3,3]=data.site_xpos[0]
    recipe=CatalogRecipe('handle','c4_2','2F',(.04,.2,.04),contact_region_endpoint_policy='MIN_LONG_AXIS')
    c=NS(model=model,data=data,gripper_geoms={0},grip_pose=lambda:grip,recipe=recipe,
         request_collision_margin_m=.005,support_top_z=.8,local_size=np.array(recipe.expected_size_m))
    targets=build_catalog_targets(np.eye(4),[0,0,0],recipe.expected_size_m,recipe)
    before={k:getattr(data,k).copy() for k in ('qpos','qvel','ctrl')};time=data.time
    _, evidence=resolve_endpoint_support_clearance(c,targets)
    assert evidence['long_axis']==1 and evidence['world_up_translation_m']>0
    for k,v in before.items():np.testing.assert_array_equal(getattr(data,k),v)
    assert data.time==time
