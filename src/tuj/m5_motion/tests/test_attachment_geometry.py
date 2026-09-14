from itertools import product

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.m5_motion.attachment_geometry import (
    certify_attachment_penetration, projection_penetration_certificate,
)


@pytest.mark.parametrize('rotation', [(0, 0, 0), (31, -47, 63)])
@pytest.mark.parametrize('depth,passes', [(.00005, True), (.009, True), (.01, False), (.015, False)])
def test_projection_bounds_preserve_deep_overlap_and_frames(rotation, depth, passes):
    box = np.array(list(product((-1., 1.), repeat=3))) * .02
    r = Rotation.from_euler('xyz', rotation, degrees=True).as_matrix()
    a = box @ r.T + [2., -3., 1.]
    b = (box + [0, 0, .04 - depth]) @ r.T + [2., -3., 1.]
    audit = projection_penetration_certificate(a, b, .01, 1e-6)
    assert audit['certified'] == passes
    assert audit['penetration_upper_bound_m'] == pytest.approx(depth + 1e-6)


def test_containment_uses_translation_distance_not_interval_overlap():
    outer = np.array(list(product((-1., 1.), repeat=3))) * .04
    inner = outer * .25
    audit = projection_penetration_certificate(outer, inner, .02, 1e-6)
    assert not audit['certified']
    assert audit['penetration_upper_bound_m'] == pytest.approx(.05 + 1e-6)


@pytest.mark.parametrize('invalid', [np.zeros((4, 3)), np.full((4, 3), np.nan), np.zeros((3, 3))])
def test_missing_or_degenerate_geometry_cannot_certify(invalid):
    box = np.array(list(product((-1., 1.), repeat=3)))
    assert projection_penetration_certificate(invalid, box, .01, 1e-6) is None


def contact_scene(z=.03995):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <geom name="hand" type="box" size=".02 .02 .02"/>
        <body pos="0 0 {z}"><freejoint/>
          <geom name="object" type="box" size=".02 .02 .02" mass=".1"/>
        </body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_numerical_rejection_requires_both_geometry_and_loaded_contact(monkeypatch):
    model, data = contact_scene()
    original = {k: getattr(data, k).copy() for k in ('qpos', 'qvel', 'ctrl')}
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: -.0445)
    audit = certify_attachment_penetration(model, data, {0}, {1}, .01)
    assert audit['certified']
    assert audit['physical_contacts']
    assert audit['offending_pairs'][0]['raw_distance_m'] == -.0445
    assert audit['offending_pairs'][0]['certificate']['penetration_upper_bound_m'] < .0001
    for key, value in original.items():
        np.testing.assert_array_equal(getattr(data, key), value)


def test_real_deep_physical_contact_still_rejects():
    model, data = contact_scene(.02)
    audit = certify_attachment_penetration(model, data, {0}, {1}, .01)
    assert not audit['certified']
    assert audit['reason'] in ('PHYSICAL_PENETRATION_EXCEEDS_LIMIT', 'PENETRATION_NOT_CERTIFIED')


def test_false_negative_query_cannot_prove_proximity_without_contact(monkeypatch):
    model, data = contact_scene(.20)
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: -.0445)
    assert not certify_attachment_penetration(model, data, {0}, {1}, .01)['certified']


def test_nonfinite_query_fails_closed(monkeypatch):
    model, data = contact_scene()
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: float('nan'))
    assert not certify_attachment_penetration(model, data, {0}, {1}, .01)['certified']


def test_every_offending_pair_must_have_a_certificate(monkeypatch):
    model, data = contact_scene()
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: -.0445)
    # An additional coincident pair cannot inherit another pair's certificate.
    audit = certify_attachment_penetration(model, data, {0, 1}, {1}, .01)
    assert not audit['certified']
    assert len(audit['offending_pairs']) == 2
    assert audit['offending_pairs'][0]['certificate']['certified']
    assert not audit['offending_pairs'][1]['certificate']['certified']
