"""MimicGen ShapeNet 머그 MJCF 래퍼.

MuJoCo가 병합 XML 기준으로 상대경로를 해석하지 않도록
texture / mesh 경로를 절대경로로 바꾼 뒤 로드한다.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET

from robosuite.models.objects import MujocoXMLObject


ENVIRONMENTS_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

MUG_ASSET_DIR = (
    ENVIRONMENTS_DIR
    / "assets"
    / "objects"
    / "mug_3143a4ac"
)

MUG_XML_PATH = (
    MUG_ASSET_DIR
    / "model.xml"
)


def _make_resolved_mug_xml() -> str:
    """texture / mesh 경로를 절대경로로 바꾼 임시 XML을 만든다."""

    if not MUG_XML_PATH.exists():
        raise FileNotFoundError(
            f"Mug XML not found: {MUG_XML_PATH}"
        )

    tree = ET.parse(
        MUG_XML_PATH
    )

    root = tree.getroot()

    asset = root.find("asset")

    if asset is None:
        raise RuntimeError(
            "Mug XML does not contain an <asset> section."
        )

    for texture in asset.findall("texture"):

        file_attr = texture.get("file")

        if file_attr is None:
            continue

        texture_path = (
            MUG_ASSET_DIR
            / "textures"
            / Path(file_attr).name
        ).resolve()

        if not texture_path.exists():
            raise FileNotFoundError(
                f"Mug texture not found: {texture_path}"
            )

        texture.set(
            "file",
            texture_path.as_posix(),
        )

    for mesh in asset.findall("mesh"):

        file_attr = mesh.get("file")

        if file_attr is None:
            continue

        mesh_path = (
            MUG_ASSET_DIR
            / file_attr
        ).resolve()

        if not mesh_path.exists():
            raise FileNotFoundError(
                f"Mug mesh not found: {mesh_path}"
            )

        mesh.set(
            "file",
            mesh_path.as_posix(),
        )

    temp_file = NamedTemporaryFile(
        mode="w",
        suffix=".xml",
        delete=False,
        encoding="utf-8",
    )

    tree.write(
        temp_file,
        encoding="unicode",
    )

    temp_file.close()

    return temp_file.name


class MugObject(MujocoXMLObject):

    def __init__(
        self,
        name: str = "mug",
    ):

        resolved_xml = (
            _make_resolved_mug_xml()
        )

        super().__init__(
            fname=resolved_xml,
            name=name,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
        )


    @property
    def asset_id(self) -> str:
        return "3143a4ac"


    @property
    def semantic_category(self) -> str:
        return "drink_container"


    @property
    def intended_ee(self) -> str:
        return "2F"
