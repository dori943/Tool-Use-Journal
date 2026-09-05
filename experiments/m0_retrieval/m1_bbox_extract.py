"""Extract production-M1-compatible bbox from Experiment 1 single-object scene.

Reuses (import only, no production edits):
  - tuj.m1_scene.points_from_frame
  - tuj.m1_scene.build_m1

Pipeline matches production M1:
  depth + segmentation → points_from_frame → build_m1 → node['bbox_mm'] (3D extents, mm)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from single_object_scene import CAMERA_FOVY, CAMERA_NAME, IMAGE_H, IMAGE_W


def camera_intrinsics(fovy_deg: float = CAMERA_FOVY, h: int = IMAGE_H, w: int = IMAGE_W) -> np.ndarray:
    fy = 0.5 * h / np.tan(np.deg2rad(fovy_deg) / 2.0)
    fx = fy
    return np.array([[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_extrinsic_cam2world(cam_xpos: np.ndarray, cam_xmat: np.ndarray) -> np.ndarray:
    """4x4 transform: OpenCV-style camera coords → world.

    Production ``points_from_frame`` builds OpenCV camera rays
    (+X right, +Y down, +Z forward). MuJoCo camera frame is
    (+X right, +Y up, -Z forward), so we convert:
        p_mj = (x, -y, -z);  p_world = R_mj @ p_mj + t
    """
    R_mj = np.asarray(cam_xmat, dtype=np.float64).reshape(3, 3)
    R_cv = R_mj @ np.diag([1.0, -1.0, -1.0])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cv
    T[:3, 3] = np.asarray(cam_xpos, dtype=np.float64)
    return T


def _body_geom_ids(model, body_id: int, skip_names: tuple[str, ...] = ("reg_bbox",)) -> set[int]:
    """Geom ids whose body is body_id or a descendant (skip helper geoms)."""
    bodies = {int(body_id)}
    changed = True
    while changed:
        changed = False
        for b in range(model.nbody):
            parent = int(model.body_parentid[b])
            if parent in bodies and b not in bodies:
                bodies.add(b)
                changed = True
    geoms = set()
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in bodies:
            continue
        import mujoco

        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name in skip_names:
            continue
        geoms.add(g)
    return geoms


def _scene_option(*, hide_collision_geoms: bool):
    """Build MjvOption matching robosuite render_collision_mesh=False when hiding.

    MuJoCo geom groups in our assets:
      group 0 — collision meshes (debug rgba red, half-alpha)
      group 1 — visual / textured meshes (+ invisible region helpers)
    """
    import mujoco

    opt = mujoco.MjvOption()
    # Defaults are typically [1,1,1,0,0,0]; keep group 1+ visible.
    if hide_collision_geoms:
        opt.geomgroup[0] = 0  # same as robosuite render_collision_mesh=False
    return opt


def render_rgb_depth_seg(model, data, camera_name: str = CAMERA_NAME):
    """Return rgb uint8, depth_m float32, seg_geom_id int32 (H,W).

    RGB appearance matches robosuite defaults: collision geoms (group 0) hidden.
    Depth + segmentation keep group 0 visible so M1 mask/bbox from the existing
    target-geom set (visual + collision) remains unchanged.
    """
    import mujoco

    renderer = mujoco.Renderer(model, height=IMAGE_H, width=IMAGE_W)
    try:
        # --- RGB / crop: hide collision (group 0), show visual (group 1+) ---
        rgb_opt = _scene_option(hide_collision_geoms=True)
        renderer.update_scene(data, camera=camera_name, scene_option=rgb_opt)
        rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()

        # --- Depth / seg: keep collision visible for stable M1 geometry ---
        m1_opt = _scene_option(hide_collision_geoms=False)
        renderer.update_scene(data, camera=camera_name, scene_option=m1_opt)

        renderer.enable_depth_rendering()
        depth = np.asarray(renderer.render(), dtype=np.float32).copy()
        renderer.disable_depth_rendering()

        renderer.enable_segmentation_rendering()
        seg = np.asarray(renderer.render()).copy()
        renderer.disable_segmentation_rendering()
    finally:
        renderer.close()

    # MuJoCo Python Renderer segmentation: (H,W,2) int32
    # Empirically in this env: channel0 = geom id, channel1 = object/type id.
    if seg.ndim == 3 and seg.shape[-1] >= 2:
        seg_geom = seg[:, :, 0].astype(np.int32)
        seg_obj = seg[:, :, 1].astype(np.int32)
    elif seg.ndim == 2:
        seg_geom = seg.astype(np.int32)
        seg_obj = seg_geom
    else:
        raise RuntimeError(f"unexpected segmentation shape: {seg.shape}")

    # Far-plane / sky depth is huge (~77); clamp for safety in callers.
    return rgb, depth, seg_geom, seg_obj


def extract_m1_node(
    model,
    data,
    object_key: str,
    camera_name: str = CAMERA_NAME,
) -> dict[str, Any]:
    """Run production M1 abstraction on the single-object frame."""
    import mujoco
    from tuj.m1_scene import build_m1, points_from_frame

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_main")
    if cam_id < 0 or body_id < 0:
        raise RuntimeError("camera or target_main body missing")

    rgb, depth_m, seg_geom, seg_obj = render_rgb_depth_seg(model, data, camera_name)
    target_geoms = _body_geom_ids(model, body_id)
    mask = np.isin(seg_geom, list(target_geoms))
    n_mask = int(mask.sum())
    print(
        f"[5a-seg] geom_ids_target={sorted(target_geoms)}  "
        f"mask_pixels={n_mask}  unique_geom_ids={sorted(np.unique(seg_geom).tolist())}",
        flush=True,
    )
    if n_mask < 20:
        raise RuntimeError(
            f"segmentation mask too small ({n_mask} px); "
            f"target_geoms={sorted(target_geoms)}"
        )

    # Invalid / sky depth → 0 so points_from_frame drops them (z > 1e-4 check)
    depth_clean = np.array(depth_m, copy=True)
    depth_clean[~np.isfinite(depth_clean)] = 0.0
    depth_clean[depth_clean > 10.0] = 0.0  # far plane
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

    vs, us = np.nonzero(mask)
    img_xyxy = [int(us.min()), int(vs.min()), int(us.max()), int(vs.max())]

    return {
        "rgb": rgb,
        "depth_m": depth_clean,
        "seg_mask": mask,
        "m1_bbox_mm": list(node["bbox_mm"]),
        "m1_center_mm": list(node["center_mm"]),
        "m1_node_id": node["id"],
        "m1_class": node["class"],
        "n_points": int(len(node["_points"])),
        "image_xyxy_from_mask": img_xyxy,
        "K": K.tolist(),
        "representation": (
            "M1 3D axis-aligned extents from depth+seg point cloud "
            "(mad_filter → hi-lo), field bbox_mm in millimeters"
        ),
        "source": "tuj.m1_scene.points_from_frame + tuj.m1_scene.build_m1",
    }


def project_world_aabb_to_image(
    center_mm: list[float],
    bbox_mm: list[float],
    cam_xpos: np.ndarray,
    cam_xmat: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Project world-frame AABB (center_mm, bbox_mm extents) to pixel xyxy."""
    center = np.asarray(center_mm, dtype=np.float64) / 1000.0
    half = np.asarray(bbox_mm, dtype=np.float64) / 2000.0
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(center + half * np.array([sx, sy, sz]))
    corners = np.asarray(corners)
    R_w2c = np.asarray(cam_xmat, dtype=np.float64).reshape(3, 3).T
    cam_pos = np.asarray(cam_xpos, dtype=np.float64)
    K = camera_intrinsics()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = [], []
    for p in corners:
        pc = R_w2c @ (p - cam_pos)
        x, y, z = pc[0], -pc[1], -pc[2]
        if z <= 1e-6:
            continue
        us.append(fx * (x / z) + cx)
        vs.append(fy * (y / z) + cy)
    if not us:
        return None
    return int(np.floor(min(us))), int(np.floor(min(vs))), int(np.ceil(max(us))), int(np.ceil(max(vs)))
