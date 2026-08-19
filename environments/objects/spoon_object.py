"""Objaverse 숟가락 MJCF 래퍼."""

from __future__ import annotations

from robosuite.models.objects import MujocoXMLObject

from environments.objects.xml_asset import (
    OBJECTS_ASSET_DIR,
    make_resolved_object_xml,
    xml_bbox_full_size_m,
    xml_default_scale,
)

SPOON_ASSET_DIR = OBJECTS_ASSET_DIR / "spoon"

# model.xml 기본 scale=0.17, 90도 refquat. reg_bbox ≈ 56.2 × 171.5 × 29.9 mm
DEFAULT_SPOON_SCALE = xml_default_scale(SPOON_ASSET_DIR)


class SpoonObject(MujocoXMLObject):
    """Objaverse 숟가락. 충돌 geom은 model.xml의 convex 15개."""

    def __init__(self, name: str = "spoon", scale: float | None = None):
        applied_scale = DEFAULT_SPOON_SCALE if scale is None else float(scale)
        bbox = xml_bbox_full_size_m(SPOON_ASSET_DIR)
        scale_ratio = applied_scale / DEFAULT_SPOON_SCALE if DEFAULT_SPOON_SCALE else 1.0
        self.applied_scale = applied_scale
        self.bbox_full_size_m = tuple(v * scale_ratio for v in bbox)

        super().__init__(
            fname=make_resolved_object_xml(SPOON_ASSET_DIR, scale=scale),
            name=name,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def asset_id(self) -> str:
        return "spoon"

    @property
    def semantic_category(self) -> str:
        return "food_container"

    @property
    def intended_ee(self) -> str:
        return "2F"

    @property
    def feasible_ees(self) -> list[str]:
        return ["2F"]
