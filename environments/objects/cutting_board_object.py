"""로컬 cutting board MJCF 래퍼."""

from __future__ import annotations

from robosuite.models.objects import MujocoXMLObject

from environments.objects.xml_asset import (
    OBJECTS_ASSET_DIR,
    make_resolved_object_xml,
    xml_bbox_full_size_m,
    xml_default_scale,
    xml_material_gt,
)

CUTTING_BOARD_ASSET_DIR = OBJECTS_ASSET_DIR / "cutting_board"

# model.xml 기본 scale=0.25, reg_bbox ≈ 147 × 245 × 15 mm
DEFAULT_CUTTING_BOARD_SCALE = xml_default_scale(CUTTING_BOARD_ASSET_DIR)


class CuttingBoardObject(MujocoXMLObject):
    """로컬 cutting board. 충돌은 box geom."""

    def __init__(self, name: str = "cutting_board", scale: float | None = None):
        applied_scale = (
            DEFAULT_CUTTING_BOARD_SCALE if scale is None else float(scale)
        )
        bbox = xml_bbox_full_size_m(CUTTING_BOARD_ASSET_DIR)
        scale_ratio = (
            applied_scale / DEFAULT_CUTTING_BOARD_SCALE
            if DEFAULT_CUTTING_BOARD_SCALE
            else 1.0
        )
        self.applied_scale = applied_scale
        self.bbox_full_size_m = tuple(v * scale_ratio for v in bbox)

        super().__init__(
            fname=make_resolved_object_xml(CUTTING_BOARD_ASSET_DIR, scale=scale),
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
        return "cutting_board"

    @property
    def semantic_category(self) -> str:
        return "receptacle"

    @property
    def material_gt(self) -> str:
        return xml_material_gt(CUTTING_BOARD_ASSET_DIR)
