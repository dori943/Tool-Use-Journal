"""Named native-flex nodes for a tetrahedral ellipsoid, without rigid proxies."""
from itertools import permutations, product
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.models.objects import MujocoObject

from environments.deformable_material import TetrahedralPlasticMaterial


def ellipsoid_mesh(full_size, count=3):
    """Map a conforming cube grid to an ellipsoid and split each cell into six tets."""
    size = np.asarray(full_size, dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("full size must contain three finite positive lengths")
    if not isinstance(count, int) or count < 3 or count % 2 != 1:
        raise ValueError("grid count must be an odd integer of at least three")
    grid = np.array(list(product(np.linspace(-1, 1, count), repeat=3)))
    points = np.empty_like(grid)
    for axis in range(3):
        a, b = [i for i in range(3) if i != axis]
        points[:, axis] = grid[:, axis] * np.sqrt(
            1 - grid[:, a] ** 2 / 2 - grid[:, b] ** 2 / 2
            + grid[:, a] ** 2 * grid[:, b] ** 2 / 3)
    points *= size / 2
    tetrahedra = []
    for cell in product(range(count - 1), repeat=3):
        for order in permutations(range(3)):
            corner = np.array(cell)
            indices = [np.ravel_multi_index(corner, (count,) * 3)]
            for axis in order:
                corner[axis] += 1
                indices.append(np.ravel_multi_index(corner, (count,) * 3))
            if np.linalg.det((points[indices[1:]] - points[indices[0]]).T) < 0:
                indices[1], indices[2] = indices[2], indices[1]
            tetrahedra.append(indices)
    return points, np.asarray(tetrahedra, dtype=int)


def _numbers(values):
    return " ".join(str(float(v)) for v in values)


class DeformableEllipsoidObject(MujocoObject):
    """Mocap root is a fixed placement frame; every material node moves freely.

    No position constraints tie nodes to the root. Each node has three named
    translational DOFs, so normal joint-state capture can preserve its motion.
    The flex must also be merged into the task's top-level deformable section.
    """

    def __init__(self, name, full_size, parameters, *, radius=0.001, count=3,
                 node_damping=0.01, rgba=(0.8, 0.6, 0.3, 1)):
        super().__init__(obj_type="all", duplicate_collision_geoms=False)
        self._name = name
        self.bbox_full_size_m = tuple(full_size)
        self.reference_positions, self.tetrahedra = ellipsoid_mesh(full_size, count)
        self.parameters = parameters
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("flex collision radius must be positive")
        self.radius = float(radius)
        material = self.new_material()
        masses = np.zeros(len(self.reference_positions))
        for local in range(4):
            np.add.at(masses, self.tetrahedra[:, local],
                      material.reference_volumes * parameters.density_kg_m3 / 4)
        self._obj = ET.Element("body", name="main", mocap="true")
        for index, (point, mass) in enumerate(zip(self.reference_positions, masses)):
            body = ET.SubElement(self._obj, "body", name=f"node_{index}", pos=_numbers(point))
            # Rotations are not DOFs; inertia only supplies a valid point-body MJCF.
            ET.SubElement(body, "inertial", pos="0 0 0", mass=str(mass),
                          diaginertia=_numbers(np.full(3, mass * radius ** 2)))
            for axis in range(3):
                ET.SubElement(body, "joint", name=f"node_{index}_{axis}", type="slide",
                              axis=_numbers(np.eye(3)[axis]), limited="false",
                              damping=str(node_damping))
        ET.SubElement(self._obj, "site", name="default_site", size="0.002", rgba="0 0 0 0")
        self._get_object_properties()
        self.flex = ET.Element("flex", name=self.naming_prefix + "material", dim="3",
                               radius=str(radius), rgba=_numbers(rgba),
                               body=" ".join(self.naming_prefix + f"node_{i}"
                                             for i in range(len(masses))),
                               vertex=" ".join(["0 0 0"] * len(masses)),
                               element=" ".join(str(i) for i in self.tetrahedra.flat))
        ET.SubElement(self.flex, "edge", stiffness="0", damping="0")
        ET.SubElement(self.flex, "contact", internal="true", selfcollide="auto",
                      friction="0.5 0.005 0.0001", solref="0.004 1")

    def new_material(self):
        return TetrahedralPlasticMaterial(self.reference_positions, self.tetrahedra, self.parameters)

    def exclude_from_prefixing(self, inp):
        return False

    def get_bounding_box_half_size(self):
        return np.asarray(self.bbox_full_size_m) / 2 + self.radius

    @property
    def size(self):
        return list(self.bbox_full_size_m)

    @property
    def bottom_offset(self):
        return np.array([0., 0., -self.get_bounding_box_half_size()[2]])

    @property
    def top_offset(self):
        return -self.bottom_offset

    @property
    def horizontal_radius(self):
        return float(max(self.get_bounding_box_half_size()[:2]))
