"""Constitutive invariants independent of any task completion threshold."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from environments.deformable_material import (
    InvalidDeformation, MaterialParameters, TetrahedralPlasticMaterial,
)


def material():
    return TetrahedralPlasticMaterial(
        np.array([[0,0,0], [.02,0,0], [0,.02,0], [0,0,.02]]),
        [[0,1,2,3]], MaterialParameters(500, 20000, 100, 1000),
    )


def test_force_is_negative_energy_gradient_and_conserves_momentum():
    m = material()
    x = m.reference @ np.array([[1.1,.1,0],[0,.95,.1],[0,0,1.05]]).T
    _, force = m.energy_and_forces(x)
    numerical = np.zeros_like(x)
    for node in range(4):
        for axis in range(3):
            plus, minus = x.copy(), x.copy()
            plus[node,axis] += 1e-7
            minus[node,axis] -= 1e-7
            numerical[node,axis] = -(m.energy_and_forces(plus)[0] - m.energy_and_forces(minus)[0]) / 2e-7
    np.testing.assert_allclose(force, numerical, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(np.cross(x,force).sum(axis=0), 0, atol=1e-12)


def test_rigid_transform_has_no_strain_or_plastic_flow():
    m = material()
    x = m.reference @ Rotation.from_rotvec([.7,.4,-.9]).as_matrix().T + [1,2,3]
    assert not m.update_plasticity(x).any()
    energy, forces = m.energy_and_forces(x)
    assert energy < 1e-20
    np.testing.assert_allclose(forces, 0, atol=1e-10)


def test_large_isochoric_compression_preserves_plastic_volume_and_state():
    m = material()
    x = m.reference @ np.diag([2,2,.25])
    assert m.update_plasticity(x).all()
    np.testing.assert_allclose(np.linalg.det(m.plastic_gradient), 1, atol=1e-12)
    np.testing.assert_allclose(m.volumes(x), m.reference_volumes, atol=1e-12)
    other = material()
    other.restore(m.state())
    np.testing.assert_allclose(other.energy_and_forces(x)[1], m.energy_and_forces(x)[1])
    # A permanent reference change must leave residual stress upon forced return.
    assert np.linalg.norm(m.energy_and_forces(m.reference)[1]) > .01


@pytest.mark.parametrize('z', [0, -1, float('nan')])
def test_invalid_elements_fail_explicitly(z):
    m = material()
    x = m.reference.copy()
    x[-1,2] *= z
    with pytest.raises(InvalidDeformation):
        m.energy_and_forces(x)
