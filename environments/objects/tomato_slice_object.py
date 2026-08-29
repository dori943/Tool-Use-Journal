"""Local tomato slice MJCF wrapper."""

from __future__ import annotations

from robosuite.models.objects import MujocoXMLObject

from environments.objects.xml_asset import (
    OBJECTS_ASSET_DIR,
    make_resolved_object_xml,
    xml_bbox_full_size_m,
    xml_default_scale,
    xml_material_gt,
)

ASSET_DIR = OBJECTS_ASSET_DIR / "tomato_slice"
DEFAULT_SCALE = xml_default_scale(ASSET_DIR)


class TomatoSliceObject(MujocoXMLObject):
    def __init__(self, name: str = "tomato_slice", scale: float | None = None):
        applied = DEFAULT_SCALE if scale is None else float(scale)
        bbox = xml_bbox_full_size_m(ASSET_DIR)
        ratio = applied / DEFAULT_SCALE if DEFAULT_SCALE else 1.0
        self.applied_scale = applied
        self.bbox_full_size_m = tuple(v * ratio for v in bbox)
        super().__init__(
            fname=make_resolved_object_xml(ASSET_DIR, scale=scale),
            name=name,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def size(self):
        return list(self.bbox_full_size_m)

    @property
    def asset_id(self) -> str:
        return "tomato_slice"

    @property
    def semantic_category(self) -> str:
        return "vegetable"

    @property
    def material_gt(self) -> str:
        return xml_material_gt(ASSET_DIR)
