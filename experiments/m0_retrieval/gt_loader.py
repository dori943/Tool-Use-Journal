"""Load simulator / asset Ground Truth for Experiment 1 objects.

Rules:
- Only values actually defined in MJCF / MuJoCo are used.
- No invented catalogue values, no LLM-filled GT.
- Unavailable properties are recorded as null with an explicit reason.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from objects import ExperimentObject, OBJECTS, asset_dir


def _as_floats(text: str) -> list[float]:
    return [float(v) for v in text.replace(",", " ").split()]


def _load_root(obj: ExperimentObject) -> ET.Element:
    xml_path = asset_dir(obj) / obj.xml_name
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)
    return ET.parse(xml_path).getroot()


def _material_gt(root: ET.Element, xml_path: Path) -> dict[str, Any]:
    entry = root.find("./custom/text[@name='material_gt']")
    if entry is None or not entry.get("data"):
        return {
            "value": None,
            "available": False,
            "source": None,
            "reason": f"material_gt custom text missing in {xml_path.name}",
        }
    return {
        "value": entry.get("data"),
        "available": True,
        "source": f"mjcf_custom_text:material_gt ({xml_path.as_posix()})",
        "reason": None,
    }


def _bbox_mm(root: ET.Element, xml_path: Path) -> dict[str, Any]:
    bbox = root.find(".//geom[@name='reg_bbox']")
    if bbox is not None and bbox.get("size"):
        half = _as_floats(bbox.get("size"))
        full = [round(2.0 * abs(v) * 1000.0, 1) for v in half]
        return {
            "value": full,
            "available": True,
            "source": f"mjcf_reg_bbox_size ({xml_path.as_posix()})",
            "reason": None,
        }
    body = root.find("./worldbody/body")
    if body is not None:
        sites = {s.get("name"): s for s in body.findall("site")}
        bottom = sites.get("bottom_site")
        top = sites.get("top_site")
        horiz = sites.get("horizontal_radius_site")
        if bottom is not None and top is not None and horiz is not None:
            bz = _as_floats(bottom.get("pos", "0 0 0"))
            tz = _as_floats(top.get("pos", "0 0 0"))
            hz = _as_floats(horiz.get("pos", "0 0 0"))
            height_m = abs(tz[2] - bz[2])
            radius_m = abs(hz[0])
            if height_m > 0 and radius_m > 0:
                full = [
                    round(2.0 * radius_m * 1000.0, 1),
                    round(2.0 * radius_m * 1000.0, 1),
                    round(height_m * 1000.0, 1),
                ]
                return {
                    "value": full,
                    "available": True,
                    "source": (
                        "mjcf_placement_sites "
                        f"(bottom/top/horizontal_radius; {xml_path.as_posix()})"
                    ),
                    "reason": (
                        "reg_bbox missing; bbox_mm derived from placement sites "
                        "(not an invented catalogue value)"
                    ),
                }
    return {
        "value": None,
        "available": False,
        "source": None,
        "reason": f"reg_bbox and usable placement sites missing in {xml_path.name}",
    }


def _collision_geom_attrs(root: ET.Element) -> tuple[list[float], list[list[float]]]:
    densities: list[float] = []
    frictions: list[list[float]] = []
    for geom in root.findall(".//geom"):
        name = geom.get("name") or ""
        if name == "reg_bbox" or geom.get("class") == "region":
            continue
        group = geom.get("group")
        if group == "1":
            continue
        if geom.get("density") is not None:
            densities.append(float(geom.get("density")))
        if geom.get("friction") is not None:
            frictions.append(_as_floats(geom.get("friction")))
    if not densities or not frictions:
        for geom in root.findall(".//geom"):
            name = geom.get("name") or ""
            if name == "reg_bbox" or geom.get("class") == "region":
                continue
            if geom.get("density") is not None and not densities:
                densities.append(float(geom.get("density")))
            if geom.get("friction") is not None and not frictions:
                frictions.append(_as_floats(geom.get("friction")))
    return densities, frictions


def _density_gt(root: ET.Element, xml_path: Path) -> dict[str, Any]:
    densities, _ = _collision_geom_attrs(root)
    if not densities:
        return {
            "value": None,
            "available": False,
            "source": None,
            "reason": f"no geom density attribute in {xml_path.name}",
        }
    unique = sorted(set(round(d, 6) for d in densities))
    if len(unique) != 1:
        return {
            "value": None,
            "available": False,
            "source": None,
            "reason": (
                f"inconsistent geom densities in {xml_path.name}: {unique} "
                "(no single object-level density GT)"
            ),
        }
    return {
        "value": unique[0],
        "available": True,
        "source": f"mjcf_collision_geom_density ({xml_path.as_posix()})",
        "reason": (
            "NOTE: MuJoCo geom density authored in MJCF "
            "(often a placeholder such as 100), not a calibrated material catalogue density."
        ),
    }


def _mu_gt(root: ET.Element, xml_path: Path) -> dict[str, Any]:
    _, frictions = _collision_geom_attrs(root)
    if not frictions:
        return {
            "value": None,
            "available": False,
            "source": None,
            "reason": f"no geom friction attribute in {xml_path.name}",
        }
    sliding = [round(f[0], 6) for f in frictions if f]
    unique = sorted(set(sliding))
    if len(unique) != 1:
        return {
            "value": None,
            "available": False,
            "source": None,
            "reason": f"inconsistent geom friction[0] in {xml_path.name}: {unique}",
        }
    return {
        "value": unique[0],
        "available": True,
        "source": f"mjcf_collision_geom_friction[0] ({xml_path.as_posix()})",
        "reason": None,
    }


def _mass_gt(obj: ExperimentObject) -> dict[str, Any]:
    try:
        import mujoco
        import numpy as np
        from environments.objects.xml_asset import make_resolved_object_xml
    except Exception as exc:  # noqa: BLE001
        return {
            "value": None,
            "available": False,
            "source": None,
            "reason": f"mujoco import/compile unavailable: {exc}",
        }
    resolved = None
    try:
        resolved = make_resolved_object_xml(asset_dir(obj), xml_name=obj.xml_name)
        model = mujoco.MjModel.from_xml_path(resolved)
        mass = float(np.sum(model.body_mass))
        return {
            "value": round(mass, 6),
            "available": True,
            "source": "mujoco_body_mass_sum (compiled object XML)",
            "reason": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "value": None,
            "available": False,
            "source": None,
            "reason": f"mujoco mass compile failed: {exc}",
        }
    finally:
        if resolved:
            Path(resolved).unlink(missing_ok=True)


def _youngs_gt() -> dict[str, Any]:
    return {
        "value": None,
        "available": False,
        "source": None,
        "reason": (
            "Young's modulus is not defined in MJCF/assets for these objects "
            "(no youngs/young/modulus field found under environments/assets/objects)."
        ),
    }


def load_object_gt(obj: ExperimentObject) -> dict[str, Any]:
    xml_path = asset_dir(obj) / obj.xml_name
    root = _load_root(obj)
    return {
        "object": obj.key,
        "label": obj.label,
        "asset": obj.asset,
        "xml": xml_path.as_posix(),
        "asset_exists": xml_path.exists(),
        "bbox_mm": _bbox_mm(root, xml_path),
        "material": _material_gt(root, xml_path),
        "density_kgm3": _density_gt(root, xml_path),
        "mass_kg": _mass_gt(obj),
        "mu": _mu_gt(root, xml_path),
        "youngs_gpa": _youngs_gt(),
    }


def load_all_gt() -> dict[str, dict[str, Any]]:
    return {key: load_object_gt(obj) for key, obj in OBJECTS.items()}


def gt_value(gt: dict[str, Any], property_name: str):
    key_map = {
        "bbox": "bbox_mm",
        "bbox_mm": "bbox_mm",
        "material": "material",
        "density": "density_kgm3",
        "density_kgm3": "density_kgm3",
        "mass_kg": "mass_kg",
        "mu": "mu",
        "youngs_gpa": "youngs_gpa",
    }
    entry = gt[key_map[property_name]]
    if not entry.get("available"):
        return None
    return entry.get("value")


def gt_available(gt: dict[str, Any], property_name: str) -> bool:
    key_map = {
        "bbox": "bbox_mm",
        "material": "material",
        "density": "density_kgm3",
        "density_kgm3": "density_kgm3",
        "mass_kg": "mass_kg",
        "mu": "mu",
        "youngs_gpa": "youngs_gpa",
    }
    return bool(gt[key_map[property_name]].get("available"))
