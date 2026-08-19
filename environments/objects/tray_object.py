"""Objaverse 트레이 MJCF 래퍼. 분류용 고정 용기(free joint 없음)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from robosuite.models.objects import MujocoXMLObject

from environments.objects.xml_asset import (
    OBJECTS_ASSET_DIR,
    make_resolved_object_xml,
    xml_bbox_full_size_m,
    xml_default_scale,
)

TRAY_ASSET_DIR = OBJECTS_ASSET_DIR / "tray"

# model.xml 기본 scale=0.225, reg_bbox ≈ 133.4 × 221.2 × 35.7 mm
DEFAULT_TRAY_SCALE = xml_default_scale(TRAY_ASSET_DIR)


def _fmt_rgba(rgba) -> str:
    return " ".join(f"{float(v):.12g}" for v in rgba)


def _apply_visual_rgba(xml_path: str, rgba) -> str:
    """같은 메시 트레이를 색으로 구분하려고 시각 geom/material만 칠한다."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rgba_str = _fmt_rgba(rgba)

    for geom in root.findall(".//geom"):
        if geom.get("mesh") is None:
            continue
        if geom.get("conaffinity") != "0":
            continue
        geom.set("rgba", rgba_str)
        if "material" in geom.attrib:
            del geom.attrib["material"]

    for material in root.findall("./asset/material"):
        material.set("rgba", rgba_str)
        if "texture" in material.attrib:
            del material.attrib["texture"]

    tree.write(xml_path)
    return xml_path


class TrayObject(MujocoXMLObject):
    """Objaverse 트레이. 충돌 geom은 model.xml의 convex 32개."""

    def __init__(
        self,
        name: str = "tray",
        scale: float | None = None,
        rgba=None,
        joints=None,
    ):
        applied_scale = DEFAULT_TRAY_SCALE if scale is None else float(scale)
        bbox = xml_bbox_full_size_m(TRAY_ASSET_DIR)
        scale_ratio = applied_scale / DEFAULT_TRAY_SCALE if DEFAULT_TRAY_SCALE else 1.0
        self.applied_scale = applied_scale
        self.bbox_full_size_m = tuple(v * scale_ratio for v in bbox)
        self.rgba = None if rgba is None else tuple(float(v) for v in rgba)

        xml_path = make_resolved_object_xml(TRAY_ASSET_DIR, scale=scale)
        if rgba is not None:
            xml_path = _apply_visual_rgba(xml_path, rgba)

        super().__init__(
            fname=xml_path,
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def asset_id(self) -> str:
        return "tray"

    @property
    def interior_half_size_xy(self) -> tuple[float, float]:
        return (
            0.5 * self.bbox_full_size_m[0],
            0.5 * self.bbox_full_size_m[1],
        )
