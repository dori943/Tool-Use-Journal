"""Objaverse 접시 MJCF 래퍼."""

from __future__ import annotations

from robosuite.models.objects import MujocoXMLObject

from environments.objects.xml_asset import (
    OBJECTS_ASSET_DIR,
    make_resolved_object_xml,
    xml_bbox_full_size_m,
    xml_default_scale,
)

PLATE_ASSET_DIR = OBJECTS_ASSET_DIR / "plate"

# model.xml 기본 scale=0.185, reg_bbox ≈ 181.8 × 181.8 × 11.1 mm
DEFAULT_PLATE_SCALE = xml_default_scale(PLATE_ASSET_DIR)


class PlateObject(MujocoXMLObject):
    """Objaverse 접시. 충돌 geom은 model.xml의 convex 32개."""

    def __init__(self, name: str = "plate", scale: float | None = None):
        applied_scale = DEFAULT_PLATE_SCALE if scale is None else float(scale)
        bbox = xml_bbox_full_size_m(PLATE_ASSET_DIR)
        scale_ratio = applied_scale / DEFAULT_PLATE_SCALE if DEFAULT_PLATE_SCALE else 1.0
        self.applied_scale = applied_scale
        self.bbox_full_size_m = tuple(v * scale_ratio for v in bbox)

        super().__init__(
            fname=make_resolved_object_xml(PLATE_ASSET_DIR, scale=scale),
            name=name,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def asset_id(self) -> str:
        return "plate"

    @property
    def semantic_category(self) -> str:
        return "dish"
