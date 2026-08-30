"""Local whisk MJCF wrapper."""

from __future__ import annotations

from robosuite.models.objects import MujocoXMLObject

from environments.objects.xml_asset import (
    OBJECTS_ASSET_DIR,
    make_resolved_object_xml,
    xml_bbox_full_size_m,
    xml_material_gt,
)


WHISK_ASSET_DIR = OBJECTS_ASSET_DIR / "whisk"


class WhiskObject(MujocoXMLObject):
    """Whisk asset whose bbox long axis is local Y."""

    def __init__(self, name: str = "whisk"):
        self.bbox_full_size_m = xml_bbox_full_size_m(WHISK_ASSET_DIR)
        super().__init__(
            fname=make_resolved_object_xml(WHISK_ASSET_DIR),
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
        return "whisk"

    @property
    def semantic_category(self) -> str:
        return "tool"

    @property
    def material_gt(self) -> str:
        return xml_material_gt(WHISK_ASSET_DIR)
