"""Minimal MJCF wrappers for C4-T2 packing assets."""

from __future__ import annotations

from robosuite.models.objects import MujocoXMLObject

from environments.objects.xml_asset import (
    OBJECTS_ASSET_DIR,
    make_resolved_object_xml,
    xml_bbox_full_size_m,
    xml_material_gt,
)


class _PackingAssetObject(MujocoXMLObject):
    ASSET_ID = ""

    def __init__(self, name: str):
        asset_dir = OBJECTS_ASSET_DIR / self.ASSET_ID
        self.bbox_full_size_m = xml_bbox_full_size_m(asset_dir)
        super().__init__(
            fname=make_resolved_object_xml(asset_dir),
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
        return self.ASSET_ID

    @property
    def material_gt(self) -> str:
        return xml_material_gt(OBJECTS_ASSET_DIR / self.ASSET_ID)


class RollingPinObject(_PackingAssetObject):
    ASSET_ID = "rolling_pin"


class BaguetteObject(_PackingAssetObject):
    ASSET_ID = "baguette"


class CerealObject(_PackingAssetObject):
    ASSET_ID = "cereal"


class MilkObject(_PackingAssetObject):
    ASSET_ID = "milk"
