"""C1-1 씬에서 M0만 실행 → m0.json + m0_points.npz 산출.

배치: Tool-Use-Journal/scripts/run_m0_c1_1.py
실행: python scripts/run_m0_c1_1.py            (레포 루트에서)
출력: output/c1_1/m0.json          — bbox 노드 + coarse 관계 (M1 전달용, VLM 0회)
      output/c1_1/m0_points.npz   — 노드별 점군 (M2 접지 입력, run_m2가 로드)
      output/c1_1/crops/*.png     — 노드별 마스크 크롭 (SiPhy 백엔드 VLM 입력)

M2는 robosuite 없이 이 출력만으로 실행 가능 → M2 쪽 반복 실험 시 sim 재기동 불필요.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Windows 렌더링 백엔드 교정 — mujoco/robosuite import 전에 처리해야 유효
import os, platform
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
FOVY_OVERRIDE = 60.0        # None이면 씬 기본값(45°) 사용.
AUTO_FIT = True             # True: 추적 객체(블록·접시·병·존) 전부 들어오게 카메라 위치 자동 조정
AUTO_FIT_MARGIN = 0.85      # 화면 가장자리 여백 (0.85 = 프레임의 85% 안에 맞춤)
CAM_SHIFT_RIGHT_M = 0.0     # AUTO_FIT=False일 때만 쓰는 수동 이동 [m]
TASK = "c1_1"
OUT = ROOT / "output" / TASK   # 모듈 공용 출력 (m0.json, gk_*.json 등 전부 여기)


# 인스턴스명 → 클래스 (M0 노드 class / M2 backend 키)
def class_of(inst: str) -> str | None:
    if inst.startswith("block_"):
        return "block"
    if inst in ("light_plate", "heavy_plate"):
        return "plate"
    if inst == "bottle_distractor":
        return "bottle"
    if inst == "collection_zone_visual":
        return "zone"
    if "rack" in inst.lower():
        return "rack"                                 # tool_rest 별칭 해결용 (fits/clear 대상)
    return None                                       # 로봇·암 등은 제외


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
        if x1 - x0 < min_box:                          # 극소 박스 확장 (원본 MIN_BOX=8)
            cx = (x0 + x1) // 2
            x0 = max(0, min(cx - min_box // 2, Ww - min_box)); x1 = x0 + min_box
        if y1 - y0 < min_box:
            cy = (y0 + y1) // 2
            y0 = max(0, min(cy - min_box // 2, Hh - min_box)); y1 = y0 + min_box
        crop = np.array(rgb[y0:y1, x0:x1])
        crop[~mask[y0:y1, x0:x1]] = 0
        Image.fromarray(crop).save(out_dir / f"{nid}.png")


def object_bound_points(env):
    """추적 대상 객체들의 월드 바운드 점 (EE 랙·로봇 제외). geom rbound 구 근사."""
    m, d = env.sim.model, env.sim.data
    gids = set()
    for o in list(env.blocks) + [env.light_plate, env.heavy_plate, env.bottle_distractor]:
        root = m.body_name2id(o.root_body)
        for gid in range(m.ngeom):
            bid = m.geom_bodyid[gid]
            while bid not in (0, root):
                bid = m.body_parentid[bid]
            if bid == root:
                gids.add(gid)
    for gid in range(m.ngeom):                        # 수집존은 geom 이름으로 (visual-only)
        nm = m.geom_id2name(gid)
        if nm and "collection_zone" in nm:
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
    R = d.cam_xmat[cid].reshape(3, 3)                 # 열 = 카메라 right/up/backward (월드)
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
    m.cam_pos[cid] += pos - d.cam_xpos[cid]           # 부모 프레임 보정 (worldbody면 그대로)
    return pos


def main():
    env = suite.make(
        env_name="C1_1_LegoSweep", robots="UR5e",
        use_camera_obs=True, has_offscreen_renderer=True, has_renderer=False,
        camera_names=CAM, camera_heights=H, camera_widths=W,
        camera_depths=True, camera_segmentations="instance",
        render_camera="agentview",   # env 기본값 ee_rack_sideview는 이 씬 모델에 없음
        ignore_done=True,
    )
    obs = env.reset()

    # 카메라 조정 (렌더 전 적용 → 이후 K/T도 조정값 기준으로 계산돼 역투영 정합 유지)
    cid0 = env.sim.model.camera_name2id(CAM)
    if FOVY_OVERRIDE:
        print(f"[cam] fovy {env.sim.model.cam_fovy[cid0]:.1f}° -> {FOVY_OVERRIDE:.1f}°")
        env.sim.model.cam_fovy[cid0] = FOVY_OVERRIDE
    if AUTO_FIT:
        env.sim.forward()                              # fovy 반영 후 현재 포즈 기준으로 fit
        pos = fit_camera_to_points(env, cid0, object_bound_points(env), margin=AUTO_FIT_MARGIN)
        print(f"[cam] auto-fit -> pos={np.round(pos, 3)} (margin={AUTO_FIT_MARGIN})")
    elif CAM_SHIFT_RIGHT_M:
        right = env.sim.data.cam_xmat[cid0].reshape(3, 3)[:, 0]   # 카메라 right 축 (월드)
        env.sim.model.cam_pos[cid0] += right * CAM_SHIFT_RIGHT_M
        print(f"[cam] pos shift right {CAM_SHIFT_RIGHT_M:+.2f}m -> {np.round(env.sim.model.cam_pos[cid0], 3)}")
    if FOVY_OVERRIDE or AUTO_FIT or CAM_SHIFT_RIGHT_M:
        env.sim.forward()
        obs = env._get_observations(force_update=True)

    # ── 프레임 구성 (베이스 프레임 좌표계) ──
    K = CU.get_camera_intrinsic_matrix(env.sim, CAM, H, W)
    T = CU.get_camera_extrinsic_matrix(env.sim, CAM)
    depth_m = np.asarray(CU.get_real_depth_map(env.sim, obs[f"{CAM}_depth"])).squeeze()
    seg = np.asarray(obs[f"{CAM}_segmentation_instance"]).squeeze()
    bp = env.sim.data.get_body_xpos("robot0_base")
    base_off = (bp[0] * 1000.0, bp[1] * 1000.0, 0.0)
    print(f"[env] robot base offset (mm): {base_off[:2]}")

    # ── 카메라 FOV/포즈 진단 ──
    cid = env.sim.model.camera_name2id(CAM)
    fovy = float(env.sim.model.cam_fovy[cid])
    fovx = float(np.degrees(2 * np.arctan(np.tan(np.radians(fovy) / 2) * W / H)))
    cam_pos = env.sim.data.cam_xpos[cid]
    print(f"[cam] {CAM}: fovy={fovy:.1f}° fovx={fovx:.1f}° (H{H}×W{W}) "
          f"pos={np.round(cam_pos, 3)} f={K[0,0]:.1f}px")
    # 테이블(1.30×1.60m) 커버리지: 카메라 높이에서 테이블면까지 거리 d일 때 시야폭 ≈ 2d·tan(fov/2)
    d = float(cam_pos[2] - 0.85)                       # 테이블 상판 z≈0.85
    print(f"[cam] 테이블면까지 수직거리≈{d:.2f}m → 시야폭≈{2*d*np.tan(np.radians(fovy)/2):.2f}m "
          f"(비스듬한 시점이면 실제 커버리지는 frame.png로 확인)")

    inst_keys = list(env.model.instances_to_ids.keys())
    name_of_id = {}                                   # seg 픽셀값 = 키 순서 인덱스 + 1
    for idx, inst in enumerate(inst_keys):
        cls = class_of(inst)
        if cls:
            name_of_id[idx + 1] = (inst, cls)
    print(f"[env] instances: {inst_keys}")
    print(f"[env] tracked: {[v[0] for v in name_of_id.values()]}")

    # ── M0 ──
    objects = points_from_frame(depth_m, seg, K, T, name_of_id, base_offset_mm=base_off)
    print(f"[M0] detected {len(objects)}/{len(name_of_id)} tracked instances "
          f"(min_pixels 미달은 탈락 — 원거리 소형 블록 확인)")
    for o in objects:
        print(f"     {o['name']:24s} points={len(o['points'])}")
    m0 = build_m0(objects)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m0.json").write_text(
        json.dumps(serialize(m0), ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "m0_points.npz",
                        **{n["id"]: n["_points"] for n in m0["nodes"]})

    # 노드별 마스크 크롭 저장 (M2 SiPhy 백엔드 입력) + 전체 프레임 (FOV/커버리지 확인용)
    rgb = np.asarray(obs[f"{CAM}_image"])
    from PIL import Image
    Image.fromarray(rgb).save(OUT / "frame.png")
    ov = rgb.copy()                                    # 추적 인스턴스 마스크 오버레이
    for sid in name_of_id:
        ov[seg == sid] = (0.5 * ov[seg == sid] + [127, 0, 0]).astype(np.uint8)
    Image.fromarray(ov).save(OUT / "frame_masks.png")
    node_ids = {n["id"].split("_", 2)[-1]: n["id"] for n in m0["nodes"]}  # inst명 → node id
    save_crops(rgb, seg, name_of_id, node_ids, OUT / "crops")
    print(f"[M0] nodes={len(m0['nodes'])} edges={len(m0['edges'])} "
          f"crops={len(list((OUT / 'crops').glob('*.png')))}")
    for e in m0["edges"]:
        print(f"     {e['type']:9s} {e['from']} -> {e['to']}")
    print(f"[DONE] -> {OUT / 'm0.json'}, {OUT / 'm0_points.npz'}")

    # --view: 조정된 agentview 시점 그대로 실시간 뷰어 (창 닫으면 종료)
    if "--view" in sys.argv:
        import time
        import mujoco
        import mujoco.viewer
        print("[view] agentview 고정 시점 뷰어 — 마우스로 자유 시점 전환 가능, 창 닫으면 종료")
        with mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            v.cam.fixedcamid = cid0
            while v.is_running():
                v.sync()
                time.sleep(0.02)
    env.close()


if __name__ == "__main__":
    main()
