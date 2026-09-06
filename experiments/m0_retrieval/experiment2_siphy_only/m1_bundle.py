"""Build production M1 node (+ crop) for Experiment 2 objects.

v4 observations store crop / bbox but NOT ``_points``. Production
``ground_intrinsic`` requires ``node["_points"]`` for:
  - PCA geometry / surface RMS (mu)
  - SiPhy ``shell_mass_integral`` (mass)

Re-runs the Experiment-1 single-object scene + production
``points_from_frame`` / ``build_m1`` path, then prefers the existing v4
``{object}_crop.png`` as the VLM image when present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

EXP2 = Path(__file__).resolve().parent
PARENT_EXP = EXP2.parent
ROOT = EXP2.parents[2]

for p in (ROOT, ROOT / "src", PARENT_EXP, EXP2):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from objects import OBJECTS  # noqa: E402

V4_OUT = ROOT / "output" / "m0_retrieval" / "experiment1_v4"
V4_OBS = V4_OUT / "observations"
V4_M1_MANIFEST = V4_OUT / "m1_bbox_manifest.json"
EXP2_OUT = ROOT / "output" / "m0_retrieval" / "experiment2_siphy_only"

# Primary Experiment 2 matrix (matches Exp1 report set).
CORE_OBJECT_KEYS = ("bottle", "spoon", "ladle", "plate", "mug")


def _load_v4_m1_bbox(object_key: str) -> list[float] | None:
    if not V4_M1_MANIFEST.is_file():
        return None
    data = json.loads(V4_M1_MANIFEST.read_text(encoding="utf-8"))
    entry = data.get(object_key) or {}
    bbox = entry.get("m1_bbox_mm")
    return [float(x) for x in bbox] if bbox is not None else None


def _extract_full_m1_node(object_key: str) -> dict[str, Any]:
    """Same pipeline as ``m1_bbox_extract.extract_m1_node``, but keep ``_points``."""
    import mujoco
    from m1_bbox_extract import (
        _body_geom_ids,
        camera_extrinsic_cam2world,
        camera_intrinsics,
        render_rgb_depth_seg,
    )
    from single_object_scene import CAMERA_NAME, build_scene_xml, _object_factory_map
    from tuj.m1_scene import build_m1, points_from_frame

    factories = _object_factory_map()
    if object_key not in factories:
        raise KeyError(f"unknown object_key for single-object scene: {object_key}")
    obj = factories[object_key]()
    scene_xml, scene_meta = build_scene_xml(object_key, obj)
    try:
        model = mujoco.MjModel.from_xml_path(scene_xml)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        for _ in range(30):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_main")
        if cam_id < 0 or body_id < 0:
            raise RuntimeError("camera or target_main body missing")

        rgb, depth_m, seg_geom, _seg_obj = render_rgb_depth_seg(
            model, data, CAMERA_NAME
        )
        target_geoms = _body_geom_ids(model, body_id)
        mask = np.isin(seg_geom, list(target_geoms))
        if int(mask.sum()) < 20:
            raise RuntimeError(f"segmentation mask too small ({int(mask.sum())} px)")

        depth_clean = np.array(depth_m, copy=True)
        depth_clean[~np.isfinite(depth_clean)] = 0.0
        depth_clean[depth_clean > 10.0] = 0.0
        depth_clean[depth_clean < 0.05] = 0.0

        seg = np.zeros(mask.shape, dtype=np.int32)
        seg[mask] = 1
        name_of_id = {1: ("target", object_key)}

        K = camera_intrinsics()
        T = camera_extrinsic_cam2world(data.cam_xpos[cam_id], data.cam_xmat[cam_id])
        objects = points_from_frame(
            depth_clean,
            seg,
            K,
            T,
            name_of_id,
            base_offset_mm=(0.0, 0.0, 0.0),
            min_pixels=20,
        )
        if not objects:
            raise RuntimeError("points_from_frame returned no objects")
        m1 = build_m1(objects)
        if not m1["nodes"]:
            raise RuntimeError("build_m1 returned no nodes")
        node = m1["nodes"][0]
        return {
            "node": node,
            "rgb": np.asarray(rgb, dtype=np.uint8),
            "seg_mask": mask,
            "scene_meta": scene_meta,
            "m1_source": "tuj.m1_scene.points_from_frame + tuj.m1_scene.build_m1",
        }
    finally:
        try:
            Path(scene_xml).unlink(missing_ok=True)
        except OSError:
            pass


def prepare_object_bundle(
    object_key: str,
    *,
    out_dir: Path | None = None,
    prefer_v4_crop: bool = True,
) -> dict[str, Any]:
    """Return production-ready M1 node, crop path, and provenance for one object."""
    from observation_capture import save_siphy_object_crop

    if object_key not in OBJECTS:
        raise SystemExit(f"unknown --object: {object_key}; choose from {list(OBJECTS)}")

    out_dir = out_dir or EXP2_OUT
    obs_dir = out_dir / "observations"
    obs_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[exp2] Rebuilding {object_key} M1 node "
        f"(production points_from_frame + build_m1)...",
        flush=True,
    )
    bundle = _extract_full_m1_node(object_key)
    node = bundle["node"]
    rgb = bundle["rgb"]
    mask = bundle["seg_mask"]

    v4_crop = V4_OBS / f"{object_key}_crop.png"
    local_crop = obs_dir / f"{object_key}_crop.png"
    crop_source = "generated_from_scene"
    if prefer_v4_crop and v4_crop.is_file():
        local_crop.write_bytes(v4_crop.read_bytes())
        crop_source = str(v4_crop)
        print(f"[exp2] Reusing v4 crop: {v4_crop}", flush=True)
    else:
        save_siphy_object_crop(rgb, mask, local_crop)
        print(f"[exp2] Wrote new SiPhy-style crop: {local_crop}", flush=True)

    points_path = obs_dir / f"{object_key}_m1_points.npz"
    pts = np.asarray(node["_points"], dtype=np.float64)
    np.savez_compressed(points_path, points_mm=pts)

    node_meta = {
        "id": node["id"],
        "class": node["class"],
        "bbox_mm": list(node["bbox_mm"]),
        "center_mm": list(node["center_mm"]),
        "n_points": int(len(pts)),
        "m1_source": bundle["m1_source"],
    }
    (obs_dir / f"{object_key}_m1_node.json").write_text(
        json.dumps(node_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    v4_bbox = _load_v4_m1_bbox(object_key)
    bbox_match = None
    if v4_bbox is not None:
        cur = [float(x) for x in node["bbox_mm"]]
        bbox_match = {
            "v4_m1_bbox_mm": v4_bbox,
            "exp2_m1_bbox_mm": cur,
            "abs_diff_mm": [round(a - b, 3) for a, b in zip(cur, v4_bbox)],
        }

    return {
        "object": object_key,
        "node": node,
        "crop_path": local_crop,
        "crop_source": crop_source,
        "points_path": points_path,
        "v4_bbox_comparison": bbox_match,
        "note": (
            f"VLM image prefers experiment1_v4/{object_key}_crop.png when present. "
            "_points re-extracted via same production M1 path (not stored in v4)."
        ),
    }


def prepare_bottle_bundle(
    *,
    out_dir: Path | None = None,
    prefer_v4_crop: bool = True,
) -> dict[str, Any]:
    """Backward-compatible alias."""
    return prepare_object_bundle(
        "bottle", out_dir=out_dir, prefer_v4_crop=prefer_v4_crop
    )
