"""Finite-strain tetrahedral material for contact-driven deformable objects.

MuJoCo owns nodal motion and collision geometry. This module only computes
internal forces and isochoric plastic state; it never edits nodal positions.
Material parameters are explicit simulation assumptions, not measured dough data.
"""
from dataclasses import dataclass

import numpy as np


class InvalidDeformation(ValueError):
    """A collapsed, inverted, or non-finite element cannot be simulated."""


@dataclass(frozen=True)
class MaterialParameters:
    shear_modulus_pa: float
    bulk_modulus_pa: float
    yield_stress_pa: float
    density_kg_m3: float

    def __post_init__(self):
        for value in (self.shear_modulus_pa, self.bulk_modulus_pa,
                      self.yield_stress_pa, self.density_kg_m3):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("material parameters must be finite and positive")


class TetrahedralPlasticMaterial:
    """Logarithmic elastic energy with an isochoric J2 radial return.

For every element F = Fe Fp. Plastic flow changes deviatoric elastic strain,
leaving det(Fp)=1. A bulk energy penalizes volume change. Volume conservation
must still be measured in the simulation; this is not a volume projection.
"""

    def __init__(self, reference_positions, tetrahedra, parameters):
        self.reference = np.array(reference_positions, dtype=float, copy=True)
        self.tetrahedra = np.array(tetrahedra, dtype=int, copy=True)
        self.parameters = parameters
        if (self.reference.ndim != 2 or self.reference.shape[1] != 3
                or not np.isfinite(self.reference).all()):
            raise ValueError("reference positions must be a finite Nx3 array")
        if (self.tetrahedra.ndim != 2 or self.tetrahedra.shape[1] != 4
                or not len(self.tetrahedra) or self.tetrahedra.min() < 0
                or self.tetrahedra.max() >= len(self.reference)):
            raise ValueError("tetrahedra must contain valid four-node indices")
        dm = self._edge_matrices(self.reference)
        determinants = np.linalg.det(dm)
        if not np.isfinite(determinants).all() or np.any(determinants <= 0):
            raise InvalidDeformation("reference tetrahedra must have positive volume")
        self.reference_volumes = determinants / 6
        self.inverse_reference = np.linalg.inv(dm)
        self.plastic_gradient = np.repeat(np.eye(3)[None], len(dm), axis=0)

    def _edge_matrices(self, positions):
        vertices = np.asarray(positions, dtype=float)[self.tetrahedra]
        return np.swapaxes(vertices[:, 1:] - vertices[:, :1], 1, 2)

    def deformation_gradient(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != self.reference.shape or not np.isfinite(positions).all():
            raise InvalidDeformation("positions must match the finite reference shape")
        gradient = self._edge_matrices(positions) @ self.inverse_reference
        determinants = np.linalg.det(gradient)
        if not np.isfinite(determinants).all() or np.any(determinants <= 0):
            raise InvalidDeformation("tetrahedron collapsed or inverted")
        return gradient

    def volumes(self, positions):
        return np.linalg.det(self.deformation_gradient(positions)) * self.reference_volumes

    @staticmethod
    def _elastic_svd(gradient, plastic):
        inverse_plastic = np.linalg.inv(plastic)
        elastic = gradient @ inverse_plastic
        u, singular, vt = np.linalg.svd(elastic)
        if (not np.isfinite(singular).all() or np.any(singular <= 0)
                or np.any(np.linalg.det(elastic) <= 0)):
            raise InvalidDeformation("elastic gradient is singular or inverted")
        return u, singular, vt, inverse_plastic

    def update_plasticity(self, positions):
        gradient = self.deformation_gradient(positions)
        _, singular, vt, _ = self._elastic_svd(gradient, self.plastic_gradient)
        strain = np.log(singular)
        mean = strain.mean(axis=1, keepdims=True)
        deviator = strain - mean
        norm = np.linalg.norm(deviator, axis=1)
        limit = np.sqrt(2 / 3) * self.parameters.yield_stress_pa / (2 * self.parameters.shear_modulus_pa)
        scale = np.ones_like(norm)
        yielded = norm > limit
        scale[yielded] = limit / norm[yielded]
        elastic_strain = mean + deviator * scale[:, None]
        v = np.swapaxes(vt, 1, 2)
        increment = (v * np.exp(strain - elastic_strain)[:, None, :]) @ vt
        updated = increment @ self.plastic_gradient
        if (not np.isfinite(updated).all()
                or not np.allclose(np.linalg.det(updated), 1, rtol=1e-9, atol=1e-10)):
            raise InvalidDeformation("plastic flow failed volume conservation")
        self.plastic_gradient = updated
        return yielded

    def energy_and_forces(self, positions):
        gradient = self.deformation_gradient(positions)
        u, singular, vt, inverse_plastic = self._elastic_svd(gradient, self.plastic_gradient)
        strain = np.log(singular)
        trace = strain.sum(axis=1, keepdims=True)
        deviator = strain - trace / 3
        mu, bulk = self.parameters.shear_modulus_pa, self.parameters.bulk_modulus_pa
        density = mu * np.sum(deviator ** 2, axis=1) + bulk * trace[:, 0] ** 2 / 2
        stress = 2 * mu * deviator + bulk * trace
        pk1 = ((u * (stress / singular)[:, None, :]) @ vt) @ np.swapaxes(inverse_plastic, 1, 2)
        h = -self.reference_volumes[:, None, None] * (pk1 @ np.swapaxes(self.inverse_reference, 1, 2))
        forces = np.zeros_like(self.reference)
        for local in range(1, 4):
            np.add.at(forces, self.tetrahedra[:, local], h[:, :, local - 1])
        np.add.at(forces, self.tetrahedra[:, 0], -h.sum(axis=2))
        if not np.isfinite(forces).all():
            raise InvalidDeformation("non-finite constitutive forces")
        return float(density @ self.reference_volumes), forces

    def state(self):
        return {"version": 1, "plastic_gradient": self.plastic_gradient.tolist()}

    def restore(self, state):
        if state.get("version") != 1:
            raise ValueError("unsupported material state version")
        gradient = np.asarray(state.get("plastic_gradient"), dtype=float)
        if (gradient.shape != self.plastic_gradient.shape or not np.isfinite(gradient).all()
                or not np.allclose(np.linalg.det(gradient), 1, rtol=1e-9, atol=1e-10)):
            raise InvalidDeformation("invalid isochoric plastic state")
        self.plastic_gradient = gradient.copy()
