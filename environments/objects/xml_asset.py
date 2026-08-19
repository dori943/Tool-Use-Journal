"""로컬 Objaverse / MimicGen MJCF 에셋 공통 처리.

mesh·texture 경로를 절대경로로 바꾸고, 필요 시 스케일·배치 사이트를 보정한
임시 XML을 만들어 MujocoXMLObject에 넘긴다.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET


ENVIRONMENTS_DIR = Path(__file__).resolve().parents[1]
OBJECTS_ASSET_DIR = ENVIRONMENTS_DIR / "assets" / "objects"


def _as_floats(text: str) -> list[float]:
    return [float(v) for v in text.replace(",", " ").split()]


def _fmt_floats(values) -> str:
    return " ".join(f"{float(v):.12g}" for v in values)


def _resolve_asset_file(asset_dir: Path, file_attr: str, kind: str) -> Path:
    """texture / mesh 파일을 오브젝트 에셋 디렉터리 기준으로 찾는다.

    순서: XML에 적힌 상대경로 → textures/ → visual/ → 파일명만.
    """
    basename = Path(file_attr).name
    candidates = [
        (asset_dir / file_attr.lstrip("/\\")).resolve(),
        (asset_dir / "textures" / basename).resolve(),
        (asset_dir / "visual" / basename).resolve(),
        (asset_dir / basename).resolve(),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"{kind} not found for '{file_attr}'. Tried:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def _current_mesh_scale(root: ET.Element) -> float:
    mesh = root.find("./asset/mesh")
    if mesh is None or mesh.get("scale") is None:
        return 1.0
    return _as_floats(mesh.get("scale"))[0]


def _apply_absolute_scale(root: ET.Element, scale: float) -> None:
    """mesh scale을 덮어쓰고, reg_bbox도 같은 비율로 맞춘다."""
    current = _current_mesh_scale(root)
    ratio = scale / current if current else 1.0
    scale_str = _fmt_floats([scale, scale, scale])
    asset = root.find("asset")
    if asset is not None:
        for mesh in asset.findall("mesh"):
            mesh.set("scale", scale_str)

    bbox = root.find(".//geom[@name='reg_bbox']")
    if bbox is not None and ratio != 1.0:
        if bbox.get("size") is not None:
            bbox.set("size", _fmt_floats(v * ratio for v in _as_floats(bbox.get("size"))))
        if bbox.get("pos") is not None:
            bbox.set("pos", _fmt_floats(v * ratio for v in _as_floats(bbox.get("pos"))))


def _ensure_placement_sites(root: ET.Element) -> None:
    """robosuite 배치용 bottom/top/horizontal_radius 사이트가 없으면 reg_bbox로 만든다."""
    parent = root.find("./worldbody/body")
    if parent is None:
        raise RuntimeError("Object XML is missing worldbody/body.")

    existing = {site.get("name") for site in parent.findall("site")}
    if {"bottom_site", "top_site", "horizontal_radius_site"} <= existing:
        return

    object_body = parent.find("./body[@name='object']")
    if object_body is None:
        raise RuntimeError("Object XML is missing body name='object'.")

    bbox = object_body.find("./geom[@name='reg_bbox']")
    if bbox is not None:
        center = _as_floats(bbox.get("pos", "0 0 0"))
        half = _as_floats(bbox.get("size", "0.05 0.05 0.05"))
    else:
        center = [0.0, 0.0, 0.0]
        half = [0.05, 0.05, 0.05]

    radius = (half[0] ** 2 + half[1] ** 2) ** 0.5
    bottom = [center[0], center[1], center[2] - half[2]]
    top = [center[0], center[1], center[2] + half[2]]
    horiz = [radius, 0.0, center[2]]

    def _add_site(name: str, pos) -> None:
        if name in existing:
            return
        site = ET.SubElement(parent, "site")
        site.set("name", name)
        site.set("pos", _fmt_floats(pos))
        site.set("size", "0.005")
        site.set("rgba", "0 0 0 0")

    _add_site("bottom_site", bottom)
    _add_site("top_site", top)
    _add_site("horizontal_radius_site", horiz)


def make_resolved_object_xml(
    asset_dir: Path,
    xml_name: str = "model.xml",
    scale: float | None = None,
) -> str:
    """로컬 model.xml을 읽어 절대경로·스케일·배치 사이트를 반영한 임시 XML 경로를 반환한다."""
    xml_path = asset_dir / xml_name
    if not xml_path.exists():
        raise FileNotFoundError(f"Object XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError(f"{xml_path} does not contain an <asset> section.")

    for texture in asset.findall("texture"):
        file_attr = texture.get("file")
        if file_attr is None:
            continue
        texture_path = _resolve_asset_file(asset_dir, file_attr, "Texture")
        texture.set("file", texture_path.as_posix())

    for mesh in asset.findall("mesh"):
        file_attr = mesh.get("file")
        if file_attr is None:
            continue
        mesh_path = _resolve_asset_file(asset_dir, file_attr, "Mesh")
        mesh.set("file", mesh_path.as_posix())

    if scale is not None:
        _apply_absolute_scale(root, scale)

    _ensure_placement_sites(root)

    temp_file = NamedTemporaryFile(
        mode="w",
        suffix=".xml",
        delete=False,
        encoding="utf-8",
    )
    tree.write(temp_file, encoding="unicode")
    temp_file.close()
    return temp_file.name


def xml_default_scale(asset_dir: Path, xml_name: str = "model.xml") -> float:
    xml_path = asset_dir / xml_name
    root = ET.parse(xml_path).getroot()
    return _current_mesh_scale(root)


def xml_bbox_full_size_m(asset_dir: Path, xml_name: str = "model.xml") -> tuple[float, float, float]:
    """reg_bbox geom의 전체 크기 (x, y, z) [m]."""
    xml_path = asset_dir / xml_name
    root = ET.parse(xml_path).getroot()
    bbox = root.find(".//geom[@name='reg_bbox']")
    if bbox is None:
        return (0.0, 0.0, 0.0)
    half = _as_floats(bbox.get("size", "0 0 0"))
    return (2.0 * half[0], 2.0 * half[1], 2.0 * half[2])
