# -*- coding: utf-8 -*-
"""예빈 env 실제 렌더 → M0 그래프 JSON (M1 입력) 브리지.

파이프라인: robosuite 오프스크린 렌더(depth + element seg)
  → 객체별 점군 복원(도희 M0 perception과 같은 방식)
  → build_m0() → serialize() → outputs/m0/<env>.m0.json

사용법:
  python scripts/run_m0_from_env.py c1_1
  python scripts/run_m0_from_env.py c2_1
  → python scripts/run_m1.py c1_1 --m0-json outputs/m0/c1_1.m0.json

주의(실행 환경에서 확인이 필요한 2곳 — 아래 ★):
  ★1 카메라 좌표 보정: MuJoCo 카메라는 -Z를 보므로 CV 핀홀(z-forward)로
     diag(1,-1,-1) 보정을 넣었다. 점군이 뒤집혀 보이면 이 부호를 확인.
  ★2 robosuite 이미지는 상하 반전으로 나온다 → flipud 적용.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

CAM = "agentview"          # --cam frontview|birdview 로 교체 가능
H, W = 720, 1280

# env별: (env_name, 객체 나열 함수, name→cls 규칙)
def _c1_objects(env):
    objs = list(env.blocks) + [env.light_plate, env.heavy_plate, env.bottle_distractor,
                               env.collection_zone_visual]
    def cls_of(name):
        if name.startswith("block"): return "block"
        if "plate" in name: return "plate"
        if "bottle" in name: return "bottle"
        if "collection" in name: return "collection_zone"
        return "object"
    return objs, cls_of

def _c2_objects(env):
    objs = list(env.target_objects) + [env.green_tray, env.blue_tray, env.red_tray]
    def cls_of(name):
        if "tray" in name: return "tray"
        return name.split("_")[0]          # apple / bread / mug / plate / spoon
    return objs, cls_of

ENVS = {"c1_1": ("C1_1_LegoSweep", _c1_objects),
        "c2_1": ("C2_1_ObjectSorting", _c2_objects)}


def geom_ids_of(sim, root_body_id):
    """루트 바디 아래(자손 포함) 모든 geom id."""
    nbody = sim.model.nbody
    descend = set()
    for b in range(nbody):
        cur = b
        while cur != 0:
            if cur == root_body_id:
                descend.add(b)
                break
            cur = sim.model.body_parentid[cur]
    return [g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] in descend]


def main():
    global CAM
    name = sys.argv[1] if len(sys.argv) > 1 else "c1_1"
    if "--cam" in sys.argv:
        CAM = sys.argv[sys.argv.index("--cam") + 1]
    env_name, get_objs = ENVS[name]

    import environments  # noqa: F401  (커스텀 env 등록)
    import robosuite as suite
    from robosuite.utils import camera_utils as CU

    env = suite.make(env_name=env_name, robots="UR5e",
                     has_renderer=False, has_offscreen_renderer=True,
                     use_camera_obs=True, camera_names=CAM,
                     camera_depths=True, camera_segmentations="element",
                     camera_heights=H, camera_widths=W, ignore_done=True,
                     render_camera=CAM)   # env 기본값(ee_rack_sideview)이 씬에 없어 교체
    obs = env.reset()

    # ── 1. depth·seg 취득 (★2 상하 반전 보정) + 카메라 RGB 저장 ──
    depth = np.flipud(CU.get_real_depth_map(env.sim, obs[f"{CAM}_depth"]).squeeze())
    seg = np.flipud(obs[f"{CAM}_segmentation_element"].squeeze())
    os.makedirs("outputs/m0", exist_ok=True)
    try:
        from PIL import Image
        rgb = np.flipud(obs[f"{CAM}_image"])
        Image.fromarray(rgb.astype(np.uint8)).save(f"outputs/m0/{name}.rgb.png")
        print(f"[카메라] outputs/m0/{name}.rgb.png 저장 ({CAM} 카메라 장면)")
    except Exception as e:                      # 이미지 저장 실패는 파이프라인을 막지 않음
        print(f"[카메라] RGB 저장 실패: {e}")

    # ── 2. element(geom id) seg → 객체 단위 라벨로 재부호화 ──
    objs, cls_of = get_objs(env)
    seg_obj = np.zeros_like(seg, dtype=np.int32)
    name_of_id = {}
    for i, o in enumerate(objs, start=1):
        rb = env.sim.model.body_name2id(o.root_body)
        for g in geom_ids_of(env.sim, rb):
            seg_obj[seg == g] = i
        name_of_id[i] = (o.name, cls_of(o.name))

    # ── 3. 카메라 내·외부 파라미터 (★1 좌표 보정) ──
    K = CU.get_camera_intrinsic_matrix(env.sim, CAM, H, W)
    T = CU.get_camera_extrinsic_matrix(env.sim, CAM)          # cam→world (MuJoCo 관례)
    T_cv = T @ np.diag([1.0, -1.0, -1.0, 1.0])                # CV 핀홀 관례로

    base = env.sim.data.get_body_xpos(env.robots[0].robot_model.root_body)
    base_offset_mm = np.asarray(base) * 1000.0

    # ── 4. 점군 복원 → M0 ──
    from tuj.m0_scene.perception import points_from_frame
    from tuj.m0_scene.abstraction import build_m0, serialize

    frame_objs = points_from_frame(depth, seg_obj, K, T_cv, name_of_id,
                                   base_offset_mm=base_offset_mm)

    # seg에 안 잡힌 물체 경고 (픽셀 부족·가림 진단용 — RGB PNG와 대조할 것)
    got_names = {o["name"] for o in frame_objs}
    for i, (nm, cls) in name_of_id.items():
        if nm not in got_names:
            px = int((seg_obj == i).sum())
            print(f"[경고] '{nm}'({cls})가 M0에 없음 — seg 픽셀 {px}개 "
                  f"(최소 20 필요). 가려짐/너무 얇음/카메라 밖 여부를 PNG로 확인")

    # 시각 전용(알파 0) collection zone은 seg에 안 잡힐 수 있다 → env 스펙에서 주입
    got = {o["name"] for o in frame_objs}
    if name == "c1_1" and "collection_zone_visual" not in got:
        c = env.sim.data.get_body_xpos(
            env.sim.model.body_name2id(env.collection_zone_visual.root_body) if False else 0)
        # 위 접근이 버전에 따라 다르면: env 배치 스펙 사용
        zc = np.asarray(getattr(env, "collection_zone_center", (0.65, 0.0, 0.802))) * 1000.0
        zs = np.asarray(getattr(env, "collection_zone_size", (0.25, 0.18))) * 1000.0
        pts = zc - base_offset_mm + (np.random.default_rng(0).random((200, 3)) - 0.5) \
              * np.array([zs[0], zs[1], 4.0])
        frame_objs.append({"name": "collection_zone_visual", "cls": "collection_zone",
                           "points": pts})

    m0 = build_m0(frame_objs)
    os.makedirs("outputs/m0", exist_ok=True)
    out = f"outputs/m0/{name}.m0.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(serialize(m0), f, ensure_ascii=False, indent=2)
    print(f"[{name}] 노드 {len(m0['nodes'])} / 엣지 {len(m0['edges'])}  ->  {out}")
    for n in m0["nodes"]:
        print(f"  {n['id']:45s} bbox {n['bbox_mm']}")


if __name__ == "__main__":
    main()
