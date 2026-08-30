"""Tool contact-patch discovery and selection.

The core planner consumes geometric patches and never branches on concrete
tool names.  Tool-specific providers may derive patches from meshes, semantic
registries, or simple dimensions and can be registered without changing the
motion-goal schema.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

import numpy as np

from tuj.m5_motion.schema import (
    ContactManipulationSpec,
    ContactSurfaceType,
    ToolContactPatch,
    WorldSnapshot,
)


class ToolAffordanceError(ValueError):
    """A requested tool patch cannot be derived or selected safely."""


class ToolAffordanceProvider(Protocol):
    def patches(
        self,
        tool_id: str,
        world: WorldSnapshot,
    ) -> tuple[ToolContactPatch, ...]: ...


class StaticToolAffordanceProvider:
    """Return validated patches supplied by a registry or an upstream module."""

    def __init__(self, patches: Iterable[ToolContactPatch]) -> None:
        grouped: dict[str, list[ToolContactPatch]] = {}
        for patch in patches:
            grouped.setdefault(patch.tool_id, []).append(patch)
        self._patches = {
            tool_id: tuple(sorted(items, key=lambda item: item.patch_id))
            for tool_id, items in grouped.items()
        }

    def patches(
        self,
        tool_id: str,
        world: WorldSnapshot,
    ) -> tuple[ToolContactPatch, ...]:
        del world
        return self._patches.get(tool_id, ())


class CompositeToolAffordanceProvider:
    """Merge independent patch providers while rejecting conflicting IDs."""

    def __init__(self, providers: Sequence[ToolAffordanceProvider]) -> None:
        self._providers = tuple(providers)

    def patches(
        self,
        tool_id: str,
        world: WorldSnapshot,
    ) -> tuple[ToolContactPatch, ...]:
        result: dict[str, ToolContactPatch] = {}
        for provider in self._providers:
            for patch in provider.patches(tool_id, world):
                previous = result.get(patch.patch_id)
                if previous is not None and previous != patch:
                    raise ToolAffordanceError(
                        f"contact patch {patch.patch_id!r} has conflicting definitions"
                    )
                result[patch.patch_id] = patch
        return tuple(result[key] for key in sorted(result))


class CircularPlateAffordanceProvider:
    """Dimension-derived broad-face and rim patches for a circular plate.

    This provider is deliberately optional: the generic planner knows only
    ``ToolContactPatch``.  A richer mesh-backed provider can replace it without
    changing planning or execution code.
    """

    def __init__(
        self,
        *,
        supported_primitives: Sequence[str] = ("PUSH", "SWEEP"),
    ) -> None:
        self._supported = [str(value).upper() for value in supported_primitives]

    def patches(
        self,
        tool_id: str,
        world: WorldSnapshot,
    ) -> tuple[ToolContactPatch, ...]:
        record = world.objects.get(tool_id)
        if not isinstance(record, Mapping):
            raise ToolAffordanceError(f"tool {tool_id!r} is missing from the world")
        dimensions = np.asarray(record.get("dimensions_m"), dtype=float)
        if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions)):
            raise ToolAffordanceError(
                f"tool {tool_id!r} has no finite three-dimensional bounds"
            )
        if np.any(dimensions <= 0.0):
            raise ToolAffordanceError(f"tool {tool_id!r} dimensions must be positive")
        radius = float(max(dimensions[0], dimensions[1]) * 0.5)
        half_thickness = float(dimensions[2] * 0.5)
        face_extent = (float(dimensions[0]), float(dimensions[1]))
        common = {
            "tool_id": tool_id,
            "supported_primitives": list(self._supported),
            "collision_geometry_refs": list(
                record.get("collision_geometry_refs", ())
                if isinstance(record.get("collision_geometry_refs", ()), Sequence)
                and not isinstance(
                    record.get("collision_geometry_refs", ()), (str, bytes)
                )
                else ()
            ),
        }
        patches = [
            ToolContactPatch(
                patch_id=f"{tool_id}:broad-face:+z",
                surface_type=ContactSurfaceType.BROAD_FACE,
                position_in_tool_m=(0.0, 0.0, half_thickness),
                normal_in_tool_xyz=(0.0, 0.0, 1.0),
                tangent_in_tool_xyz=(1.0, 0.0, 0.0),
                extent_m=face_extent,
                metadata={"side": "+z", "shape": "circular"},
                **common,
            ),
            ToolContactPatch(
                patch_id=f"{tool_id}:broad-face:-z",
                surface_type=ContactSurfaceType.BROAD_FACE,
                position_in_tool_m=(0.0, 0.0, -half_thickness),
                normal_in_tool_xyz=(0.0, 0.0, -1.0),
                tangent_in_tool_xyz=(1.0, 0.0, 0.0),
                extent_m=face_extent,
                metadata={"side": "-z", "shape": "circular"},
                **common,
            ),
        ]
        for label, position, normal, tangent in (
            ("+x", (radius, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ("-x", (-radius, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ("+y", (0.0, radius, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
            ("-y", (0.0, -radius, 0.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
        ):
            patches.append(
                ToolContactPatch(
                    patch_id=f"{tool_id}:rim:{label}",
                    surface_type=ContactSurfaceType.RIM,
                    position_in_tool_m=position,
                    normal_in_tool_xyz=normal,
                    tangent_in_tool_xyz=tangent,
                    extent_m=(float(dimensions[2]), float(dimensions[2])),
                    curvature_radius_m=radius,
                    metadata={"side": label, "shape": "circular"},
                    **common,
                )
            )
        return tuple(patches)


def select_contact_patch(
    patches: Sequence[ToolContactPatch],
    spec: ContactManipulationSpec,
) -> ToolContactPatch:
    """Select a deterministic compatible patch or fail closed."""

    if not patches:
        raise ToolAffordanceError("tool exposes no contact patches")
    if spec.contact_patch_id is not None:
        matches = [patch for patch in patches if patch.patch_id == spec.contact_patch_id]
        if len(matches) != 1:
            raise ToolAffordanceError(
                f"requested contact patch {spec.contact_patch_id!r} is unavailable"
            )
        candidates = matches
    elif spec.contact_surface is not ContactSurfaceType.AUTO:
        candidates = [
            patch for patch in patches if patch.surface_type is spec.contact_surface
        ]
    else:
        candidates = list(patches)
    primitive = spec.primitive.strip().upper()
    candidates = [
        patch
        for patch in candidates
        if not patch.supported_primitives
        or primitive in {value.strip().upper() for value in patch.supported_primitives}
    ]
    if not candidates:
        raise ToolAffordanceError(
            f"no contact patch supports {primitive!r} with surface "
            f"{spec.contact_surface.value!r}"
        )
    preference = {
        ContactSurfaceType.BROAD_FACE: 0,
        ContactSurfaceType.RIM: 1,
        ContactSurfaceType.EDGE: 2,
        ContactSurfaceType.POINT: 3,
        ContactSurfaceType.AUTO: 4,
    }
    return min(candidates, key=lambda patch: (preference[patch.surface_type], patch.patch_id))


__all__ = [
    "CircularPlateAffordanceProvider",
    "CompositeToolAffordanceProvider",
    "StaticToolAffordanceProvider",
    "ToolAffordanceError",
    "ToolAffordanceProvider",
    "select_contact_patch",
]
