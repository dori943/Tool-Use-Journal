"""로컬 dough MJCF 래퍼."""

from __future__ import annotations

from robosuite.models.objects import MujocoXMLObject

from environments.objects.xml_asset import (
    OBJECTS_ASSET_DIR,
    make_resolved_object_xml,
    xml_bbox_full_size_m,
    xml_default_scale,
    xml_material_gt,
)

DOUGH_ASSET_DIR = OBJECTS_ASSET_DIR / "dough"

# model.xml에 mesh scale 없음 → 1.0. reg_bbox ≈ 60 × 60 × 61 mm
DEFAULT_DOUGH_SCALE = xml_default_scale(DOUGH_ASSET_DIR)


class DoughObject(MujocoXMLObject):
    """로컬 cookie dough ball."""

    def __init__(self, name: str = "dough", scale: float | None = None):
        applied_scale = DEFAULT_DOUGH_SCALE if scale is None else float(scale)
        bbox = xml_bbox_full_size_m(DOUGH_ASSET_DIR)
        scale_ratio = (
            applied_scale / DEFAULT_DOUGH_SCALE if DEFAULT_DOUGH_SCALE else 1.0
        )
        self.applied_scale = applied_scale
        self.bbox_full_size_m = tuple(v * scale_ratio for v in bbox)

        super().__init__(
            fname=make_resolved_object_xml(DOUGH_ASSET_DIR, scale=scale),
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
        return "dough"

    @property
    def semantic_category(self) -> str:
        return "food"

    @property
    def material_gt(self) -> str:
        return xml_material_gt(DOUGH_ASSET_DIR)
