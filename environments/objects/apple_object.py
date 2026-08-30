"""Objaverse 사과 MJCF 래퍼."""

from __future__ import annotations

from robosuite.models.objects import MujocoXMLObject

from environments.objects.xml_asset import (
    OBJECTS_ASSET_DIR,
    make_resolved_object_xml,
    xml_bbox_full_size_m,
    xml_default_scale,
    xml_material_gt,
)

APPLE_ASSET_DIR = OBJECTS_ASSET_DIR / "apple"

# model.xml 기본 scale=0.075, reg_bbox ≈ 75.8 × 75.4 × 75.5 mm
DEFAULT_APPLE_SCALE = xml_default_scale(APPLE_ASSET_DIR)


class AppleObject(MujocoXMLObject):
    """Objaverse 사과. 충돌 geom은 model.xml의 convex 16개."""

    def __init__(self, name: str = "apple", scale: float | None = None):
        applied_scale = DEFAULT_APPLE_SCALE if scale is None else float(scale)
        bbox = xml_bbox_full_size_m(APPLE_ASSET_DIR)
        scale_ratio = applied_scale / DEFAULT_APPLE_SCALE if DEFAULT_APPLE_SCALE else 1.0
        self.applied_scale = applied_scale
        self.bbox_full_size_m = tuple(v * scale_ratio for v in bbox)

        super().__init__(
            fname=make_resolved_object_xml(APPLE_ASSET_DIR, scale=scale),
            name=name,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def asset_id(self) -> str:
        return "apple"

    @property
    def semantic_category(self) -> str:
        return "fruit"

    @property
    def material_gt(self) -> str:
        return xml_material_gt(APPLE_ASSET_DIR)
