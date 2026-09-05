"""Experiment 1 single-object observation scene (experiment-only).

Builds a lightweight MuJoCo scene with:
  - floor + table
  - exactly one production object asset
  - one fixed camera

No production task environments (c1_1 / c2_1 / c2_2) are used or modified.
No robot, no distractors.
"""
from __future__ import annotations

import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Fixed scene / camera (identical for all 7 objects)
TABLE_FULL_SIZE = (0.8, 0.8, 0.05)  # L, W, H meters
TABLE_OFFSET = (0.0, 0.0, 0.80)  # table-top upper surface at z=0.80
OBJECT_XY = (0.0, 0.0)  # table-center placement
CAMERA_NAME = "exp1_cam"
CAMERA_POS = (1.05, -0.75, 1.45)  # fixed world position
CAMERA_QUAT_WXYZ: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # set below after helper
CAMERA_FOVY = 40.0
IMAGE_H = 512
IMAGE_W = 512


def _object_factory_map() -> dict[str, Callable[[], Any]]:
    """Lazy import of production object classes (same assets as task envs)."""
    from environments.objects.apple_object import AppleObject
    from environments.objects.bottle_object import BottleObject
    from environments.objects.bread_object import BreadObject
    from environments.objects.ladle_object import LadleObject
    from environments.objects.mug_object import MugObject
    from environments.objects.plate_object import PlateObject
    from environments.objects.spoon_object import SpoonObject

    return {
        "bottle": lambda: BottleObject(name="target"),
        "spoon": lambda: SpoonObject(name="target"),
        "ladle": lambda: LadleObject(name="target"),
        "plate": lambda: PlateObject(name="target"),
        "mug": lambda: MugObject(name="target"),
        "apple": lambda: AppleObject(name="target"),
        "bread": lambda: BreadObject(name="target"),
    }


def _asset_dir_for_key(key: str) -> Path:
    from environments.objects.xml_asset import OBJECTS_ASSET_DIR
    from objects import OBJECTS

    return OBJECTS_ASSET_DIR / OBJECTS[key].asset


def _as_floats(text: str) -> list[float]:
    return [float(v) for v in text.replace(",", " ").split()]


def _fmt(values) -> str:
    return " ".join(f"{float(v):.12g}" for v in values)


def _place_z(table_top_z: float, bottom_offset_z: float) -> float:
    """Body origin z so object bottom rests on table top.

    robosuite bottom_offset.z is typically negative (body origin above bottom).
    body_z = table_top_z - bottom_offset_z
    """
    return float(table_top_z - bottom_offset_z)


def _copy_defaults(src_root: ET.Element, dst_root: ET.Element) -> None:
    src_default = src_root.find("default")
    if src_default is None:
        return
    # Insert before asset/worldbody if possible
    dst_root.insert(0, src_default)


def _copy_assets(src_root: ET.Element, dst_root: ET.Element) -> None:
    src_asset = src_root.find("asset")
    if src_asset is None:
        return
    dst_asset = dst_root.find("asset")
    if dst_asset is None:
        dst_asset = ET.SubElement(dst_root, "asset")
    for child in list(src_asset):
        dst_asset.append(child)


def _find_object_body(root: ET.Element) -> ET.Element:
    body = root.find("./worldbody/body/body[@name='object']")
    if body is not None:
        return body
    body = root.find("./worldbody/body[@name='object']")
    if body is not None:
        return body
    # fallback: first nested body under worldbody/body
    outer = root.find("./worldbody/body")
    if outer is None:
        raise RuntimeError("object XML missing worldbody/body")
    inner = outer.find("./body")
    return inner if inner is not None else outer


def build_scene_xml(object_key: str, obj) -> tuple[str, dict[str, Any]]:
    """Compose a temporary MJCF path for table + one object + fixed camera."""
    table_top_z = TABLE_OFFSET[2]
    half_h = TABLE_FULL_SIZE[2] / 2.0
    table_body_z = table_top_z - half_h

    bottom = np.asarray(obj.bottom_offset, dtype=np.float64)
    place_xy = (OBJECT_XY[0] - float(bottom[0]), OBJECT_XY[1] - float(bottom[1]))
    place_z = _place_z(table_top_z, float(bottom[2]))

    # Read resolved object XML (absolute mesh/texture paths already applied by Object ctor)
    obj_xml_path = None
    for attr in ("file", "fname", "xml_path"):
        if hasattr(obj, attr):
            candidate = Path(getattr(obj, attr))
            if candidate.exists():
                obj_xml_path = candidate
                break
    if obj_xml_path is None:
        raise FileNotFoundError(f"cannot locate resolved XML for {type(obj).__name__}")
    obj_tree = ET.parse(str(obj_xml_path))
    obj_root = obj_tree.getroot()
    obj_body = _find_object_body(obj_root)

    root = ET.Element("mujoco", model=f"exp1_{object_key}")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(IMAGE_W), offheight=str(IMAGE_H))
    ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0.1 0.1 0.1")
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.9 0.9 0.95",
        rgb2="0.6 0.7 0.8",
        width="256",
        height="256",
    )
    ET.SubElement(
        asset,
        "texture",
        name="texplane",
        type="2d",
        builtin="checker",
        rgb1="0.2 0.3 0.4",
        rgb2="0.1 0.15 0.2",
        width="512",
        height="512",
    )
    ET.SubElement(asset, "material", name="matplane", texture="texplane", texrepeat="2 2", reflectance="0.0")
    ET.SubElement(asset, "material", name="table_mat", rgba="0.65 0.55 0.4 1", reflectance="0.05")

    _copy_defaults(obj_root, root)
    _copy_assets(obj_root, root)

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 2.5", dir="0 0 -1", directional="true", diffuse="0.8 0.8 0.8")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="2 2 0.1",
        material="matplane",
        contype="1",
        conaffinity="1",
    )
    table = ET.SubElement(world, "body", name="table", pos=_fmt([TABLE_OFFSET[0], TABLE_OFFSET[1], table_body_z]))
    ET.SubElement(
        table,
        "geom",
        name="table_collision",
        type="box",
        size=_fmt([TABLE_FULL_SIZE[0] / 2, TABLE_FULL_SIZE[1] / 2, half_h]),
        material="table_mat",
        contype="1",
        conaffinity="1",
        friction="1 0.005 0.0001",
    )
    ET.SubElement(
        table,
        "site",
        name="table_top",
        pos=_fmt([0, 0, half_h]),
        size="0.001",
        rgba="0 0 0 0",
    )

    # Object body: freejoint + geoms/sites from production asset (default orientation)
    target = ET.SubElement(
        world,
        "body",
        name="target_main",
        pos=_fmt([place_xy[0], place_xy[1], place_z]),
        quat="1 0 0 0",
    )
    ET.SubElement(target, "freejoint", name="target_joint")
    for child in list(obj_body):
        # Avoid nested free joints from source if any; keep geoms/sites/bodies
        if child.tag == "joint" and child.get("type") == "free":
            continue
        target.append(child)

    ET.SubElement(
        world,
        "camera",
        name=CAMERA_NAME,
        pos=_fmt(CAMERA_POS),
        quat=_fmt(CAMERA_QUAT_WXYZ),
        fovy=str(CAMERA_FOVY),
    )

    # Write temp XML
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8")
    ET.ElementTree(root).write(tmp, encoding="unicode")
    tmp.close()

    meta = {
        "object_key": object_key,
        "object_class": type(obj).__name__,
        "object_xml": str(obj_xml_path),
        "asset_dir": str(_asset_dir_for_key(object_key)),
        "bottom_offset": [float(v) for v in bottom],
        "place_pos": [place_xy[0], place_xy[1], place_z],
        "table_top_z": table_top_z,
        "table_full_size": list(TABLE_FULL_SIZE),
        "camera_name": CAMERA_NAME,
        "camera_pos": list(CAMERA_POS),
        "camera_quat_wxyz": list(CAMERA_QUAT_WXYZ),
        "camera_fovy": CAMERA_FOVY,
        "image_hw": [IMAGE_H, IMAGE_W],
        "scene_xml": tmp.name,
    }
    return tmp.name, meta


def _camera_quat_lookat(pos: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    """MuJoCo camera quat (wxyz) that looks from pos toward target with world-up."""
    forward = target - pos
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    # MuJoCo camera looks along -Z in camera frame; build rotation.
    z = -forward
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    rot = np.stack([x, y, z], axis=1)  # columns = camera axes in world
    # rotation matrix -> quat wxyz
    m = rot
    tr = float(np.trace(m))
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        xq = (m[2, 1] - m[1, 2]) / s
        yq = (m[0, 2] - m[2, 0]) / s
        zq = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s
            xq = 0.25 * s
            yq = (m[0, 1] + m[1, 0]) / s
            zq = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s
            xq = (m[0, 1] + m[1, 0]) / s
            yq = 0.25 * s
            zq = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s
            xq = (m[0, 2] + m[2, 0]) / s
            yq = (m[1, 2] + m[2, 1]) / s
            zq = 0.25 * s
    q = np.array([w, xq, yq, zq], dtype=np.float64)
    q = q / np.linalg.norm(q)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


# Fixed camera orientation (look at table center; identical for all objects).
CAMERA_QUAT_WXYZ = _camera_quat_lookat(
    np.asarray(CAMERA_POS, dtype=np.float64),
    np.asarray([TABLE_OFFSET[0], TABLE_OFFSET[1], TABLE_OFFSET[2] + 0.05], dtype=np.float64),
)


def render_single_object(object_key: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Create scene, settle, render, run production M1 bbox extraction."""
    import mujoco
    from m1_bbox_extract import extract_m1_node

    factories = _object_factory_map()
    if object_key not in factories:
        raise KeyError(object_key)

    t0 = time.perf_counter()
    print("[1] Create Experiment 1 single-object scene", flush=True)
    obj = factories[object_key]()
    print(
        f"[2] Load {object_key} asset  class={type(obj).__name__}  "
        f"asset_dir={_asset_dir_for_key(object_key)}  ({time.perf_counter()-t0:.2f}s)",
        flush=True,
    )
    scene_xml, meta = build_scene_xml(object_key, obj)
    print(
        f"[2a] GT asset match check: scene_asset={meta['asset_dir']} "
        f"object_xml={meta['object_xml']}",
        flush=True,
    )
    print(
        f"[3] Place {object_key} on table  pos={meta['place_pos']}  "
        f"bottom_offset={meta['bottom_offset']}  table_top_z={meta['table_top_z']}  "
        f"({time.perf_counter()-t0:.2f}s)",
        flush=True,
    )

    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    print(f"[4] Reset / settle... ({time.perf_counter()-t0:.2f}s)", flush=True)
    mujoco.mj_resetData(model, data)
    for _ in range(30):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id < 0:
        raise RuntimeError(f"camera {CAMERA_NAME} missing")

    print(
        f"[5] Capture camera + M1 depth/seg ({CAMERA_NAME})  "
        f"pos={CAMERA_POS} fovy={CAMERA_FOVY} res={IMAGE_W}x{IMAGE_H}  "
        f"({time.perf_counter()-t0:.2f}s)",
        flush=True,
    )
    m1 = extract_m1_node(model, data, object_key=object_key, camera_name=CAMERA_NAME)
    rgb = m1["rgb"]
    print(
        f"[5a] M1 bbox extracted  bbox_mm={m1['m1_bbox_mm']}  "
        f"center_mm={m1['m1_center_mm']}  n_points={m1['n_points']}  "
        f"via {m1['source']}",
        flush=True,
    )

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_main")
    body_pos = np.array(data.xpos[bid], dtype=np.float64)
    body_mat = np.array(data.xmat[bid], dtype=np.float64).reshape(3, 3)
    meta["body_pos"] = body_pos.tolist()
    meta["body_mat"] = body_mat.tolist()
    meta["cam_xpos"] = np.array(data.cam_xpos[cam_id], dtype=np.float64).tolist()
    meta["cam_xmat"] = np.array(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3).tolist()
    meta["m1_bbox_mm"] = m1["m1_bbox_mm"]
    meta["m1_center_mm"] = m1["m1_center_mm"]
    meta["m1_node_id"] = m1["m1_node_id"]
    meta["m1_n_points"] = m1["n_points"]
    meta["m1_source"] = m1["source"]
    meta["m1_representation"] = m1["representation"]
    meta["m1_image_xyxy_from_mask"] = m1["image_xyxy_from_mask"]
    meta["seg_mask"] = m1["seg_mask"]  # bool HxW — SiPhy-style crop source
    meta["elapsed_build_sec"] = round(time.perf_counter() - t0, 3)

    try:
        Path(scene_xml).unlink(missing_ok=True)
    except OSError:
        pass

    return np.asarray(rgb, dtype=np.uint8), meta


def project_asset_bbox(
    body_pos: np.ndarray,
    body_mat: np.ndarray,
    bbox_mm: list[float],
    model,
    data,
    camera_name: str = CAMERA_NAME,
) -> tuple[int, int, int, int] | None:
    """Project axis-aligned asset bbox (object frame extents) to image pixel AABB."""
    import mujoco

    half = np.asarray(bbox_mm, dtype=np.float64) / 2000.0  # mm full -> m half
    corners_local = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], half[1], half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], -half[2]],
            [half[0], half[1], half[2]],
        ]
    )
    corners_w = (body_mat @ corners_local.T).T + body_pos

    # Use mujoco camera projection via free camera matrices from renderer scene is complex;
    # compute with camera extrinsic from model.
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    cam_pos = np.array(data.cam_xpos[cam_id])
    cam_mat = np.array(data.cam_xmat[cam_id]).reshape(3, 3)  # camera axes in world
    # Camera looks along -Z; OpenCV-style: x right, y down, z forward
    R_w2c = cam_mat.T
    fovy = CAMERA_FOVY
    fy = 0.5 * IMAGE_H / np.tan(np.deg2rad(fovy) / 2.0)
    fx = fy
    cx, cy = IMAGE_W / 2.0, IMAGE_H / 2.0

    us, vs = [], []
    for p in corners_w:
        pc = R_w2c @ (p - cam_pos)
        # MuJoCo camera frame: +X right, +Y up, -Z look. Convert to OpenCV (+Y down, +Z forward)
        x, y, z = pc[0], -pc[1], -pc[2]
        if z <= 1e-6:
            continue
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
        us.append(u)
        vs.append(v)
    if not us:
        return None
    return (
        int(np.floor(min(us))),
        int(np.floor(min(vs))),
        int(np.ceil(max(us))),
        int(np.ceil(max(vs))),
    )
