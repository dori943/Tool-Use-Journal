"""Topology and native MuJoCo binding checks for deformable object geometry."""
from collections import Counter
from itertools import combinations
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from environments.deformable_material import MaterialParameters
from environments.objects.deformable_ellipsoid import DeformableEllipsoidObject, ellipsoid_mesh


def test_tetrahedral_grid_is_conforming_and_has_ellipsoid_boundary():
    points, tets = ellipsoid_mesh([.06, .08, .10])
    faces = Counter(tuple(sorted(face)) for tet in tets for face in combinations(tet, 3))
    assert set(faces.values()) == {1, 2}
    surface = np.unique([face for face, number in faces.items() if number == 1])
    np.testing.assert_allclose(np.sum((points[surface] / [.03, .04, .05]) ** 2, axis=1), 1)
    det = np.linalg.det(np.swapaxes(points[tets[:, 1:]] - points[tets[:, :1]], 1, 2))
    assert np.all(det > 0)


def test_native_flex_mass_and_named_node_positions_match_material():
    obj = DeformableEllipsoidObject('sample', [.06, .06, .061],
                                   MaterialParameters(500, 20000, 100, 1000))
    root = ET.Element('mujoco')
    world = ET.SubElement(root, 'worldbody')
    body = obj.get_obj()
    body.set('pos', '1 2 3')
    body.set('quat', '0.7071067811865476 0 0 0.7071067811865476')
    world.append(body)
    ET.SubElement(root, 'deformable').append(obj.flex)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    expected = obj.reference_positions @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) + [1, 2, 3]
    np.testing.assert_allclose(data.flexvert_xpos, expected, atol=1e-12)
    assert model.njnt == 3 * len(expected)
    assert all(model.joint(i).name for i in range(model.njnt))
    assert model.ngeom == 0  # no hidden rigid collision proxy
    np.testing.assert_allclose(model.body_mass.sum(), obj.new_material().reference_volumes.sum() * 1000)
