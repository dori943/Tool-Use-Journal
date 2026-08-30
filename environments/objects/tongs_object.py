"""로컬 tongs MJCF 래퍼."""

from __future__ import annotations

from robosuite.models.objects import MujocoXMLObject

from environments.objects.xml_asset import (
    OBJECTS_ASSET_DIR,
    make_resolved_object_xml,
    xml_bbox_full_size_m,
    xml_default_scale,
    xml_material_gt,
)

TONGS_ASSET_DIR = OBJECTS_ASSET_DIR / "tongs"

# model.xml에 mesh scale 없음 → 1.0. reg_bbox ≈ 31 × 258 × 25 mm
DEFAULT_TONGS_SCALE = xml_default_scale(TONGS_ASSET_DIR)


class TongsObject(MujocoXMLObject):
    """로컬 tongs (rigid)."""

    def __init__(self, name: str = "tongs", scale: float | None = None):
        applied_scale = DEFAULT_TONGS_SCALE if scale is None else float(scale)
        bbox = xml_bbox_full_size_m(TONGS_ASSET_DIR)
        scale_ratio = (
            applied_scale / DEFAULT_TONGS_SCALE if DEFAULT_TONGS_SCALE else 1.0
        )
        self.applied_scale = applied_scale
        self.bbox_full_size_m = tuple(v * scale_ratio for v in bbox)

        super().__init__(
            fname=make_resolved_object_xml(TONGS_ASSET_DIR, scale=scale),
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
        return "tongs"

    @property
    def semantic_category(self) -> str:
        return "tool"

    @property
    def material_gt(self) -> str:
        return xml_material_gt(TONGS_ASSET_DIR)
