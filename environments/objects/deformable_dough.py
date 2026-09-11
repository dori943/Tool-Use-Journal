"""Dough semantic asset backed by native deformable collision geometry."""
from environments.deformable_material import MaterialParameters
from environments.objects.deformable_ellipsoid import DeformableEllipsoidObject
from environments.objects.dough_object import DOUGH_ASSET_DIR
from environments.objects.xml_asset import xml_bbox_full_size_m, xml_material_gt


class DeformableDoughObject(DeformableEllipsoidObject):
    def __init__(self, name, material):
        super().__init__(name, xml_bbox_full_size_m(DOUGH_ASSET_DIR),
                         MaterialParameters(**material))

    @property
    def asset_id(self):
        return "dough"

    @property
    def semantic_category(self):
        return "food"

    @property
    def material_gt(self):
        return xml_material_gt(DOUGH_ASSET_DIR)
