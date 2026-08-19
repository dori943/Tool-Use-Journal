"""Objaverse 빵 MJCF 래퍼."""

from __future__ import annotations

from robosuite.models.objects import MujocoXMLObject

from environments.objects.xml_asset import (
    OBJECTS_ASSET_DIR,
    make_resolved_object_xml,
    xml_bbox_full_size_m,
    xml_default_scale,
)

BREAD_ASSET_DIR = OBJECTS_ASSET_DIR / "bread"

# model.xml 기본 scale=0.13, reg_bbox ≈ 81.1 × 127.5 × 66.8 mm
DEFAULT_BREAD_SCALE = xml_default_scale(BREAD_ASSET_DIR)


class BreadObject(MujocoXMLObject):
    """Objaverse 빵. 충돌 geom은 model.xml의 convex 16개."""

    def __init__(self, name: str = "bread", scale: float | None = None):
        applied_scale = DEFAULT_BREAD_SCALE if scale is None else float(scale)
        bbox = xml_bbox_full_size_m(BREAD_ASSET_DIR)
        scale_ratio = applied_scale / DEFAULT_BREAD_SCALE if DEFAULT_BREAD_SCALE else 1.0
        self.applied_scale = applied_scale
        self.bbox_full_size_m = tuple(v * scale_ratio for v in bbox)

        super().__init__(
            fname=make_resolved_object_xml(BREAD_ASSET_DIR, scale=scale),
            name=name,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def asset_id(self) -> str:
        return "bread"

    @property
    def semantic_category(self) -> str:
        return "food"
