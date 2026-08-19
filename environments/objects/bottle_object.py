"""Objaverse 병 MJCF 래퍼."""

from __future__ import annotations

from robosuite.models.objects import MujocoXMLObject

from environments.objects.xml_asset import (
    OBJECTS_ASSET_DIR,
    make_resolved_object_xml,
    xml_bbox_full_size_m,
    xml_default_scale,
)

BOTTLE_ASSET_DIR = OBJECTS_ASSET_DIR / "bottle"

# model.xml 기본 scale=0.135, reg_bbox ≈ 49.8 × 50.0 × 135.0 mm
DEFAULT_BOTTLE_SCALE = xml_default_scale(BOTTLE_ASSET_DIR)


class BottleObject(MujocoXMLObject):
    """Objaverse 병. 충돌 geom은 model.xml의 convex 5개."""

    def __init__(self, name: str = "bottle", scale: float | None = None):
        applied_scale = DEFAULT_BOTTLE_SCALE if scale is None else float(scale)
        bbox = xml_bbox_full_size_m(BOTTLE_ASSET_DIR)
        scale_ratio = applied_scale / DEFAULT_BOTTLE_SCALE if DEFAULT_BOTTLE_SCALE else 1.0
        self.applied_scale = applied_scale
        self.bbox_full_size_m = tuple(v * scale_ratio for v in bbox)

        super().__init__(
            fname=make_resolved_object_xml(BOTTLE_ASSET_DIR, scale=scale),
            name=name,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def asset_id(self) -> str:
        return "bottle"

    @property
    def semantic_category(self) -> str:
        return "bottle"
