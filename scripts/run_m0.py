# -*- coding: utf-8 -*-
"""M0 실행기 — 예빈 태스크에서 M0 접지 산출물을 뽑는다.

사용법:
  python scripts/run_m0.py c1_1
  python scripts/run_m0.py c2_1 --view

출력: output/<task>/m0.json          — bbox 노드 + coarse 관계 (M1 전달용, VLM 0회)
      output/<task>/m0_points.npz   — 노드별 점군 (M2 접지 입력)
      output/<task>/crops/*.png     — 노드별 마스크 크롭 (SiPhy 백엔드 VLM 입력)
      output/<task>/frame*.png      — 카메라 진단용 프레임

태스크 추가: 아래 TASKS 딕셔너리에 항목 하나 등록 (env_name / class_of / bound_objects).
M2는 robosuite 없이 이 출력만으로 실행 가능 → M2 쪽 반복 실험 시 sim 재기동 불필요.
"""
from __future__ import annotations

import json
import os
<<<<<<< HEAD
=======
import re
>>>>>>> da6cdbb (fix(run_m0): c1_1 어댑터를 환경 추종 방식으로 - 접시 1개 신규 환경 대응, 인스턴스 이름 기반 클래스 자동 유추로 환경 변경 시 코드 수정 불필요)
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Windows 렌더링 백엔드 교정 — mujoco/robosuite import 전에 처리해야 유효
import platform
if platform.system() == "Windows" and os.environ.get("MUJOCO_GL", "").lower() not in ("", "wgl", "glfw"):
    print(f"[fix] MUJOCO_GL={os.environ.get('MUJOCO_GL')} -> wgl")
    os.environ["MUJOCO_GL"] = "wgl"

import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"                     # 상하반전 방지 (역투영 필수)

import environments  # noqa: F401  (suite.make 등록)
import robosuite as suite
from robosuite.utils import camera_utils as CU

from tuj.m0_scene import build_m0, points_from_frame, serialize

CAM, H, W = "agentview", 512, 512
FOVY_OVERRIDE = 60.0        # None이면 씬 기본값(45°)
AUTO_FIT = True             # True: 추적 객체 전부 프레임에 들어가게 카메라 위치 자동 조정
AUTO_FIT_MARGIN = 0.85      # 프레임 여백 (85% 안에 맞춤)


# ── 태스크별 어댑터 ─────────────────────────────────────────────

<<<<<<< HEAD
def _c1_1_class_of(inst):
    if inst.startswith("block_"):                       return "block"
    if inst in ("light_plate", "heavy_plate"):          return "plate"
    if inst == "bottle_distractor":                     return "bottle"
    if inst == "collection_zone_visual":                return "zone"
    if "rack" in inst.lower():                          return "rack"
    return None
=======
_ROBOT_MARKERS = ("ur5e", "mount", "nullgripper", "robotiq", "gripper0", "robot0")


def _generic_class_of(inst):
    """환경 인스턴스 이름에서 클래스를 유추한다 — 환경에 물체가 추가되어도 코드 수정 불필요.

    규칙: 로봇 부속 제외 / rack 포함 -> rack / zone 포함 -> zone /
          끝의 _숫자 제거(block_0 -> block) / _distractor 제거(bottle_distractor -> bottle).
    """
    low = inst.lower()
    if any(k in low for k in _ROBOT_MARKERS) and "rack" not in low:
        return None
    if "rack" in low:                                   return "rack"
    if "zone" in low:                                   return "zone"
    base = re.sub(r"_\d+$", "", low)
    base = re.sub(r"_distractor$", "", base)
    return base or None


def _generic_bound_bodies(env, class_of):
    """auto-fit 대상: 환경의 물체 목록(obj_body_id)에서 rack/zone 제외 전부."""
    ids = []
    for inst, bid in dict(env.obj_body_id).items():
        cls = class_of(inst)
        if cls and cls not in ("rack", "zone"):
            ids.append(int(bid))
    return ids
>>>>>>> da6cdbb (fix(run_m0): c1_1 어댑터를 환경 추종 방식으로 - 접시 1개 신규 환경 대응, 인스턴스 이름 기반 클래스 자동 유추로 환경 변경 시 코드 수정 불필요)


def _c2_1_class_of(inst):
    if inst == "apple":  return "apple"
    if inst == "bread":  return "bread"
    if inst == "mug":    return "mug"
    if inst == "plate":  return "plate"
    if inst == "spoon":  return "spoon"
    if inst.endswith("_tray"):                          return "tray"
    if "rack" in inst.lower():                          return "rack"
    return None


TASKS = {
    "c1_1": dict(env_name="C1_1_LegoSweep",
<<<<<<< HEAD
                 class_of=_c1_1_class_of,
                 bound_objects=lambda env: list(env.blocks) +
                     [env.light_plate, env.heavy_plate, env.bottle_distractor],
=======
                 class_of=_generic_class_of,
                 bound_objects=lambda env: _generic_bound_bodies(env, _generic_class_of),
>>>>>>> da6cdbb (fix(run_m0): c1_1 어댑터를 환경 추종 방식으로 - 접시 1개 신규 환경 대응, 인스턴스 이름 기반 클래스 자동 유추로 환경 변경 시 코드 수정 불필요)
                 extra_geom_names=["collection_zone"]),
    "c2_1": dict(env_name="C2_1_ObjectSorting",
                 class_of=_c2_1_class_of,
                 bound_objects=lambda env: env.target_objects + env.trays,
                 extra_geom_names=[]),
    # c1_2, c2_2: 씬 파일(environments/*.py) 추가 시 여기에 한 항목 등록
}


def save_crops(rgb, seg, name_of_id, node_ids, out_dir, min_box=8):
    """SiPhy mask_material_proposal.py 방식: bbox 크롭 + 마스크 밖 검정 처리."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    Hh, Ww = seg.shape
    for sid, (inst, cls) in name_of_id.items():
        nid = node_ids.get(inst)
        if nid is None:
            continue
        mask = seg == sid
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if x1 - x0 < min_box:
            cx = (x0 + x1) // 2
            x0 = max(0, min(cx - min_box // 2, Ww - min_box)); x1 = x0 + min_box
        if y1 - y0 < min_box:
            cy = (y0 + y1) // 2
            y0 = max(0, min(cy - min_box // 2, Hh - min_box)); y1 = y0 + min_box
        crop = np.array(rgb[y0:y1, x0:x1])
        crop[~mask[y0:y1, x0:x1]] = 0
        Image.fromarray(crop).save(out_dir / f"{nid}.png")


def object_bound_points(env, spec):
    """auto-fit 대상 월드 바운드 점 수집."""
    m, d = env.sim.model, env.sim.data
    gids = set()
    for o in spec["bound_objects"](env):
<<<<<<< HEAD
        root = m.body_name2id(o.root_body)
=======
        root = o if isinstance(o, int) else m.body_name2id(o.root_body)
>>>>>>> da6cdbb (fix(run_m0): c1_1 어댑터를 환경 추종 방식으로 - 접시 1개 신규 환경 대응, 인스턴스 이름 기반 클래스 자동 유추로 환경 변경 시 코드 수정 불필요)
        for gid in range(m.ngeom):
            bid = m.geom_bodyid[gid]
            while bid not in (0, root):
                bid = m.body_parentid[bid]
            if bid == root:
                gids.add(gid)
    for gid in range(m.ngeom):
        nm = m.geom_id2name(gid)
        if nm and any(k in nm for k in spec["extra_geom_names"]):
            gids.add(gid)
    pts = []
    for gid in gids:
        c, r = d.geom_xpos[gid], m.geom_rbound[gid]
        for k in range(3):
            e = np.zeros(3); e[k] = r
            pts += [c + e, c - e]
    return np.asarray(pts)


def fit_camera_to_points(env, cid, pts, margin=0.85, iters=12):
    """시점 방향 고정, 위치만 이동: 좌우·상하 중앙정렬 + 프레임 초과 시 후진/여유 시 전진."""
    m, d = env.sim.model, env.sim.data
    R = d.cam_xmat[cid].reshape(3, 3)
    ty = np.tan(np.radians(float(m.cam_fovy[cid])) / 2)
    tx = ty * W / H
    pos = d.cam_xpos[cid].copy()
    for _ in range(iters):
        P = (pts - pos) @ R
        z = np.maximum(-P[:, 2], 1e-3)
        u, v = P[:, 0] / z, P[:, 1] / z
        zm = float(z.mean())
        pos += R[:, 0] * (u.min() + u.max()) / 2 * zm
        pos += R[:, 1] * (v.min() + v.max()) / 2 * zm
        P = (pts - pos) @ R
        z = np.maximum(-P[:, 2], 1e-3)
        ex = max(np.abs(P[:, 0] / z).max() / (tx * margin),
                 np.abs(P[:, 1] / z).max() / (ty * margin))
        pos += R[:, 2] * float(z.mean()) * (ex - 1.0)
    m.cam_pos[cid] += pos - d.cam_xpos[cid]
    return pos


def main():
    name = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "c1_1"
    view = "--view" in sys.argv
    if name not in TASKS:
        sys.exit(f"[err] unknown task {name!r}. 등록된 태스크: {list(TASKS)}")
    spec = TASKS[name]
    OUT = ROOT / "output" / name

    env = suite.make(
        env_name=spec["env_name"], robots="UR5e",
        use_camera_obs=True, has_offscreen_renderer=True, has_renderer=False,
        camera_names=CAM, camera_heights=H, camera_widths=W,
        camera_depths=True, camera_segmentations="instance",
        render_camera="agentview", ignore_done=True,
    )
    obs = env.reset()

    cid0 = env.sim.model.camera_name2id(CAM)
    if FOVY_OVERRIDE:
        print(f"[cam] fovy {env.sim.model.cam_fovy[cid0]:.1f}° -> {FOVY_OVERRIDE:.1f}°")
        env.sim.model.cam_fovy[cid0] = FOVY_OVERRIDE
    if AUTO_FIT:
        env.sim.forward()
        pos = fit_camera_to_points(env, cid0, object_bound_points(env, spec),
                                    margin=AUTO_FIT_MARGIN)
        print(f"[cam] auto-fit -> pos={np.round(pos, 3)} (margin={AUTO_FIT_MARGIN})")
    if FOVY_OVERRIDE or AUTO_FIT:
        env.sim.forward()
        obs = env._get_observations(force_update=True)

    K = CU.get_camera_intrinsic_matrix(env.sim, CAM, H, W)
    T = CU.get_camera_extrinsic_matrix(env.sim, CAM)
    depth_m = np.asarray(CU.get_real_depth_map(env.sim, obs[f"{CAM}_depth"])).squeeze()
    seg = np.asarray(obs[f"{CAM}_segmentation_instance"]).squeeze()
    bp = env.sim.data.get_body_xpos("robot0_base")
    base_off = (bp[0] * 1000.0, bp[1] * 1000.0, 0.0)
    print(f"[env] robot base offset (mm): {base_off[:2]}")

    cid = env.sim.model.camera_name2id(CAM)
    fovy = float(env.sim.model.cam_fovy[cid])
    fovx = float(np.degrees(2 * np.arctan(np.tan(np.radians(fovy) / 2) * W / H)))
    cam_pos = env.sim.data.cam_xpos[cid]
    print(f"[cam] {CAM}: fovy={fovy:.1f}° fovx={fovx:.1f}° (H{H}×W{W}) "
          f"pos={np.round(cam_pos, 3)} f={K[0,0]:.1f}px")

    inst_keys = list(env.model.instances_to_ids.keys())
    name_of_id = {}                                   # seg 픽셀값 = 키 순서 인덱스 + 1
    for idx, inst in enumerate(inst_keys):
        cls = spec["class_of"](inst)
        if cls:
            name_of_id[idx + 1] = (inst, cls)
    print(f"[env] instances: {inst_keys}")
    print(f"[env] tracked: {[v[0] for v in name_of_id.values()]}")

    objects = points_from_frame(depth_m, seg, K, T, name_of_id, base_offset_mm=base_off)
    print(f"[M0] detected {len(objects)}/{len(name_of_id)} tracked instances")
    for o in objects:
        print(f"     {o['name']:24s} points={len(o['points'])}")
    m0 = build_m0(objects)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m0.json").write_text(
        json.dumps(serialize(m0), ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "m0_points.npz",
                        **{n["id"]: n["_points"] for n in m0["nodes"]})

    rgb = np.asarray(obs[f"{CAM}_image"])
    from PIL import Image
    Image.fromarray(rgb).save(OUT / "frame.png")
    ov = rgb.copy()
    for sid in name_of_id:
        ov[seg == sid] = (0.5 * ov[seg == sid] + [127, 0, 0]).astype(np.uint8)
    Image.fromarray(ov).save(OUT / "frame_masks.png")
    node_ids = {n["id"].split("_", 2)[-1]: n["id"] for n in m0["nodes"]}
    save_crops(rgb, seg, name_of_id, node_ids, OUT / "crops")
    print(f"[M0] nodes={len(m0['nodes'])} edges={len(m0['edges'])} "
          f"crops={len(list((OUT / 'crops').glob('*.png')))}")
    for e in m0["edges"]:
        print(f"     {e['type']:9s} {e['from']} -> {e['to']}")
    print(f"[{name}] -> {OUT}/m0.json, {OUT}/m0_points.npz")

    if view:
        import time
        import mujoco
        import mujoco.viewer
        print("[view] agentview 고정 시점 뷰어 (창 닫으면 종료)")
        with mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            v.cam.fixedcamid = cid0
            while v.is_running():
                v.sync()
                time.sleep(0.02)
    env.close()


if __name__ == "__main__":
    main()
