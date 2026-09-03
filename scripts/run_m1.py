# -*- coding: utf-8 -*-
"""M1 실행기 — 예빈 태스크에서 M1 접지 산출물을 뽑는다.

사용법:
  python scripts/run_m1.py c1_1
  python scripts/run_m1.py c2_1 --view

출력: output/<task>/m1.json          — bbox 노드 + coarse 관계 (M2 전달용, VLM 0회)
      output/<task>/m1_points.npz   — 노드별 점군 (M3 접지 입력)
      output/<task>/crops/*.png     — 노드별 마스크 크롭 (SiPhy 백엔드 VLM 입력)
      output/<task>/frame*.png      — 카메라 진단용 프레임

태스크 추가: task_registry.TASKS 에 한 줄. 이름 규칙 깨진 env 만 _OVERRIDES 에 등록.
M3는 robosuite 없이 이 출력만으로 실행 가능 → M3 쪽 반복 실험 시 sim 재기동 불필요.
"""
from __future__ import annotations

import json
import os
import re
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
from task_registry import TASK_ENVS, TASKS
import robosuite as suite
from robosuite.utils import camera_utils as CU

from tuj.m1_scene import build_m1, points_from_frame, serialize

_DEFAULT_CAM = "agentview"
_ROBOCASA_CAM_PREFERENCES = ("robot0_robotview", "robot0_eye_in_hand")

# RoboCasa 공통 third-person scene camera
# 모든 RoboCasa 태스크에서 동일한 방향을 사용하고,
# lookat / distance만 장면 범위에 맞춰 자동 조정한다.
_ROBOCASA_SCENE_CAM_AZIMUTH = 180.0
_ROBOCASA_SCENE_CAM_ELEVATION = -55.0
_ROBOCASA_SCENE_CAM_MIN_DIST = 0.9
H, W = 512, 512
FOVY_OVERRIDE = 60.0        # None이면 씬 기본값(45°)
AUTO_FIT = True             # True: 추적 객체 전부 프레임에 들어가게 카메라 위치 자동 조정
AUTO_FIT_MARGIN = 0.85      # 프레임 여백 (85% 안에 맞춤)
# RoboCasa는 환경에 미리 정의된 고정 observation camera를 그대로 사용한다.
# agentview가 없는 RoboCasa에서는 robot0_robotview를 우선 사용한다.


# ── 태스크별 어댑터 ─────────────────────────────────────────────

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


def _robocasa_task_object_names(env):
    """RoboCasa Kitchen 조작 대상 — fixture 제외, env가 등록한 task object만."""
    objects = getattr(env, "objects", None) or {}
    return tuple(sorted(objects.keys()))


# env.objects 에 없는 고정 바디(worldbody 직속)도 M1이 추적해야 하는 경우.
# c4_2 의 packing_box 는 fixture 처럼 worldbody 에 직접 붙어 env.objects 에 없어서
# M1 이 뚜껑만 보고 상자를 몰랐다 (0903 1차 실행: LLM 이 뚜껑을 용기로 삼음).
# 이름 -> 클래스. 바디 subtree 의 group=1 geom 을 그 물체의 visual 로 쓴다.
_EXTRA_ROBOCASA_BODIES = {
    "c4_2": {"packing_box": "box"},
}


def _robocasa_tracked_models(env, spec):
    """(이름, 모델) 목록 — env.objects 와 spec['extra_bodies'] 고정 바디를 합친다.

    고정 바디는 root_body 와 빈 visual_geoms 만 가진 대역 객체로 넘겨,
    _object_visual_geom_ids 가 subtree group=1 geom 으로 떨어지게 한다.
    """
    from types import SimpleNamespace
    objects = getattr(env, "objects", None) or {}
    items = [(name, objects[name]) for name in _robocasa_task_object_names(env)]
    m = env.sim.model
    for body, _cls in (spec.get("extra_bodies") or {}).items():
        try:
            m.body_name2id(body)
        except Exception:
            print(f"[env] 경고: 고정 바디 {body!r} 가 모델에 없어 추적하지 않습니다.")
            continue
        items.append((body, SimpleNamespace(root_body=body, visual_geoms=[])))
    return items


def _subtree_geom_ids_by_group(m, body_id: int, group: int) -> list[int]:
    """body subtree에서 geom_group==group 인 geom ID 수집."""
    geoms = []
    start = m.body_geomadr[body_id]
    end = start + m.body_geomnum[body_id]
    geoms.extend(gid for gid in range(start, end) if m.geom_group[gid] == group)
    for child in range(m.nbody):
        if m.body_parentid[child] == body_id:
            geoms.extend(_subtree_geom_ids_by_group(m, child, group))
    return geoms


def _object_visual_geom_ids(env, model) -> tuple[int, ...]:
    """Task object visual geom ID — contact geom 제외, 관측용 visual 우선."""
    from robosuite.utils.mjcf_utils import get_ids
    sim = env.sim
    visual_names = list(getattr(model, "visual_geoms", None) or [])
    if visual_names:
        ids = [int(g) for g in get_ids(sim=sim, elements=visual_names, element_type="geom")]
        if ids:
            return tuple(ids)
    m = sim.model
    body_id = m.body_name2id(model.root_body)
    return tuple(_subtree_geom_ids_by_group(m, body_id, group=1))











def robocasa_task_segmentation(env, spec, cam: str | None, h: int, w: int,
                               geom_seg: np.ndarray | None = None):
    """RoboCasa: env.objects visual geom → synthetic 1-based instance segmentation.

    instances_to_ids를 쓰지 않는다. fixture/robot/cabinet 등은 seg==0(배경).
    반환: (seg[H,W] int32, name_of_id {sid: (inst_name, cls)})
    """
    if geom_seg is None:
        geom_seg = _raw_geom_segmentation(env, cam, h, w)
    seg = np.zeros((h, w), dtype=np.int32)
    name_of_id = {}
    extra_cls = spec.get("extra_bodies") or {}
    for sid, (name, model) in enumerate(_robocasa_tracked_models(env, spec), start=1):
        geom_ids = _object_visual_geom_ids(env, model)
        cls = extra_cls.get(name) or spec["class_of"](name)
        if not cls:
            continue
        name_of_id[sid] = (name, cls)
        if geom_ids:
            seg[np.isin(geom_seg, geom_ids)] = sid
    return seg, name_of_id


def _track_class(inst, spec):
    """인스턴스를 M1에서 추적할지 판정. None이면 제외 (non-RoboCasa 전용)."""
    return spec["class_of"](inst)


def _c2_1_class_of(inst):
    if inst == "apple":  return "apple"
    if inst == "bread":  return "bread"
    if inst == "mug":    return "mug"
    if inst == "plate":  return "plate"
    if inst == "spoon":  return "spoon"
    if inst.endswith("_tray"):                          return "tray"
    if "rack" in inst.lower():                          return "rack"
    return None


# 이름 규칙이 깨진 env 만 M1 인식 오버라이드. 나머지는 범용 어댑터로 자동 처리한다.
_OVERRIDES = {
    "c2_1": dict(class_of=_c2_1_class_of,
                 bound_objects=lambda env: env.target_objects + env.trays),
}
# 카메라 auto-fit 에 추가로 포함할 지오메트리(바디 아님) 이름.
_EXTRA_GEOMS = {
    "c1_1": ["collection_zone"],
}


def task_spec(name):
    """task_registry + 오버라이드로 M1 씬 스펙을 만든다.

    env_name 은 단일 출처(task_registry)에서 가져오고, class_of/bound_objects 는
    기본이 범용 어댑터다. 이름 규칙이 깨진 env 만 _OVERRIDES 로 대체한다.
    """
    ov = _OVERRIDES.get(name, {})
    class_of = ov.get("class_of", _generic_class_of)
    bound = ov.get("bound_objects", lambda env: _generic_bound_bodies(env, class_of))
    return dict(env_name=TASK_ENVS[name],
                class_of=class_of,
                bound_objects=bound,
                extra_geom_names=_EXTRA_GEOMS.get(name, []),
                extra_bodies=_EXTRA_ROBOCASA_BODIES.get(name, {}),
                robocasa=TASKS[name].robocasa)


def _available_camera_names(env):
    """MuJoCo model에 등록된 camera 이름 목록."""
    m = env.sim.model
    names = []
    for cid in range(m.ncam):
        name = m.camera_id2name(cid)
        if name:
            names.append(name)
    return tuple(names)


def _initial_camera_name(spec):
    """suite.make()에 넘길 camera 이름 — RoboCasa는 robot0_robotview 우선."""
    if spec["robocasa"]:
        return _ROBOCASA_CAM_PREFERENCES[0]
    return _DEFAULT_CAM


def _resolve_observation_camera(env, preferred, *, robocasa):
    """실제 sim에 존재하는 observation camera를 선택 (RGB/depth/seg/auto-fit 공통)."""
    available = set(_available_camera_names(env))
    if preferred in available:
        return preferred
    if robocasa:
        for candidate in _ROBOCASA_CAM_PREFERENCES:
            if candidate in available:
                print(f"[cam] {preferred!r} unavailable; using {candidate!r} "
                      f"(available: {sorted(available)})")
                return candidate
    elif _DEFAULT_CAM in available:
        return _DEFAULT_CAM
    raise ValueError(
        f"No observation camera resolved (preferred={preferred!r}, robocasa={robocasa}). "
        f"Available: {sorted(available)}"
    )


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
        root = o if isinstance(o, int) else m.body_name2id(o.root_body)
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



def _robocasa_task_bound_points(env, spec) -> np.ndarray:
    """RoboCasa task object 전체의 월드 바운드 점을 수집."""
    m, d = env.sim.model, env.sim.data
    pts = []
    for name, model in _robocasa_tracked_models(env, spec):
        root = m.body_name2id(model.root_body)
        for gid in range(m.ngeom):
            bid = m.geom_bodyid[gid]
            while bid not in (0, root):
                bid = m.body_parentid[bid]
            if bid == root:
                c, r = d.geom_xpos[gid], m.geom_rbound[gid]
                for k in range(3):
                    e = np.zeros(3)
                    e[k] = r
                    pts += [c + e, c - e]
    return np.asarray(pts) if pts else np.zeros((0, 3))


def _robocasa_task_bound_center(env, spec) -> np.ndarray:
    """RoboCasa task object 전체를 바라보는 공통 lookat 중심."""
    pts = _robocasa_task_bound_points(env, spec)
    if len(pts) == 0:
        return np.array([0.0, 0.0, 0.8])
    return pts.mean(axis=0)


def _free_camera_extrinsic_from_scene(ctx) -> np.ndarray:
    """현재 MuJoCo free camera의 cam2world extrinsic (OpenCV convention)."""
    import mujoco
    from robosuite.utils import transform_utils as T

    headpos = np.zeros((3, 1))
    forward = np.zeros((3, 1))
    up = np.zeros((3, 1))
    mujoco.mjv_cameraInModel(headpos, forward, up, ctx.scn)

    hp = headpos.ravel()
    fw = forward.ravel() / np.linalg.norm(forward)
    uu = up.ravel() / np.linalg.norm(up)

    z = -fw
    x = np.cross(uu, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    pose = T.make_pose(hp, np.column_stack([x, y, z]))
    corr = np.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, -1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    )
    return pose @ corr


def _free_camera_intrinsic(h: int, w: int, fovy: float) -> np.ndarray:
    """Free camera intrinsic matrix."""
    f = 0.5 * h / np.tan(np.radians(fovy) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])


def _robocasa_scene_camera_distance(
    env, spec, center: np.ndarray, fovy: float, h: int, w: int,
    azimuth: float, elevation: float, margin: float = AUTO_FIT_MARGIN
) -> float:
    """공통 방향은 유지하면서 모든 task object가 프레임에 들어오도록 distance만 계산."""
    import mujoco
    from robosuite.utils import transform_utils as T

    ctx = env.sim._render_context_offscreen
    cam = ctx.cam
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = center
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    env.sim.model.vis.global_.fovy = float(fovy)

    pts = _robocasa_task_bound_points(env, spec)
    if len(pts) == 0:
        return _ROBOCASA_SCENE_CAM_MIN_DIST

    ty = np.tan(np.radians(fovy) / 2)
    tx = ty * w / h
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])

    def fits(dist: float) -> bool:
        cam.distance = float(dist)
        env.sim.forward()
        mujoco.mjv_updateScene(
            env.sim.model._model,
            env.sim.data._data,
            ctx.vopt,
            ctx.pert,
            cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            ctx.scn,
        )
        w2c = T.pose_inv(_free_camera_extrinsic_from_scene(ctx))
        P = (w2c @ pts_h.T).T[:, :3]
        z = P[:, 2]
        if np.any(z <= 1e-3):
            return False

        u = P[:, 0] / z
        v = P[:, 1] / z
        return (
            np.abs(u).max() <= tx * margin
            and np.abs(v).max() <= ty * margin
        )

    lo, hi = _ROBOCASA_SCENE_CAM_MIN_DIST, 6.0
    if not fits(hi):
        return hi

    for _ in range(24):
        mid = (lo + hi) / 2
        if fits(mid):
            hi = mid
        else:
            lo = mid

    return max(hi, _ROBOCASA_SCENE_CAM_MIN_DIST)


def _setup_robocasa_scene_camera(env, *, lookat, distance, fovy):
    """모든 RoboCasa 태스크에 동일한 정면-상단 사선 방향을 적용."""
    import mujoco

    ctx = env.sim._render_context_offscreen
    cam = ctx.cam
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = float(distance)
    cam.azimuth = _ROBOCASA_SCENE_CAM_AZIMUTH
    cam.elevation = _ROBOCASA_SCENE_CAM_ELEVATION
    env.sim.model.vis.global_.fovy = float(fovy)


def _robocasa_scene_camera_render(env, h: int, w: int):
    """공통 RoboCasa scene camera에서 RGB / depth / geom segmentation 획득."""
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

    convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

    rgb, depth_raw = env.sim.render(
        camera_name=None, width=w, height=h, depth=True, segmentation=False
    )
    rgb = np.asarray(rgb)[::convention]
    depth_raw = np.asarray(depth_raw)[::convention]
    depth_m = CU.get_real_depth_map(env.sim, depth_raw)

    seg_raw = env.sim.render(
        camera_name=None, width=w, height=h, depth=False, segmentation=True
    )
    geom_seg = np.asarray(seg_raw)[::convention, :, 1]
    return rgb, depth_m, geom_seg



def _raw_geom_segmentation(env, cam: str | None, h: int, w: int) -> np.ndarray:
    """MuJoCo geom-ID segmentation render (group channel). cam=None → free camera."""
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
    raw = env.sim.render(
        camera_name=cam, width=w, height=h, depth=False, segmentation=True)
    convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    return np.asarray(raw)[::convention, :, 1]


def instance_segmentation(env, obs: dict, cam: str, h: int, w: int) -> np.ndarray:
    """robosuite 호환 1-based instance segmentation (non-RoboCasa).

    obs 센서가 있으면 그대로 사용(c1_1/c2_1). 없으면 geom seg를
    env.model.instances_to_ids 순서로 매핑한다.
    """
    key = f"{cam}_segmentation_instance"
    if key in obs:
        return np.asarray(obs[key]).squeeze()
    geom_seg = _raw_geom_segmentation(env, cam, h, w)
    instance_names = list(env.model.instances_to_ids.keys())
    name_to_id = {name: index for index, name in enumerate(instance_names)}
    geom_to_id = {
        geom_id: name_to_id[name]
        for geom_id, name in env.model.geom_ids_to_instances.items()
    }
    return (np.fromiter(
        (geom_to_id.get(int(x), -1) for x in geom_seg.flat),
        dtype=np.int32, count=geom_seg.size,
    ).reshape(h, w) + 1)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "c1_1"
    view = "--view" in sys.argv
    if name not in TASK_ENVS:
        sys.exit(f"[err] unknown task {name!r}. 등록된 태스크: {list(TASK_ENVS)}")
    spec = task_spec(name)
    OUT = ROOT / "output" / name
    cam = _initial_camera_name(spec)

    make_kwargs = dict(
        env_name=spec["env_name"], robots="UR5e",
        use_camera_obs=True, has_offscreen_renderer=True, has_renderer=False,
        camera_names=cam, camera_heights=H, camera_widths=W,
        camera_depths=True,
        render_camera=cam, ignore_done=True,
    )
    if spec["robocasa"]:
        print("[env] RoboCasa Kitchen: camera_segmentations unsupported; "
              "using task-object visual geom segmentation")
    else:
        make_kwargs["camera_segmentations"] = "instance"
    env = suite.make(**make_kwargs)
    obs = env.reset()
    cam = _resolve_observation_camera(env, cam, robocasa=spec["robocasa"])

    if spec["robocasa"]:
        print(f"[env] robocasa task objects: {list(_robocasa_task_object_names(env))}")
        env.sim.forward()

        fovy = FOVY_OVERRIDE or float(env.sim.model.vis.global_.fovy)
        center = _robocasa_task_bound_center(env, spec)
        dist = _robocasa_scene_camera_distance(
            env,
            spec,
            center,
            fovy,
            H,
            W,
            azimuth=_ROBOCASA_SCENE_CAM_AZIMUTH,
            elevation=_ROBOCASA_SCENE_CAM_ELEVATION,
            margin=AUTO_FIT_MARGIN,
        )
        _setup_robocasa_scene_camera(
            env,
            lookat=center,
            distance=dist,
            fovy=fovy,
        )

        print(
            f"[cam] RoboCasa common scene camera: "
            f"lookat={np.round(center, 3)} dist={dist:.2f} "
            f"az={_ROBOCASA_SCENE_CAM_AZIMUTH:.0f}° "
            f"el={_ROBOCASA_SCENE_CAM_ELEVATION:.0f}° "
            f"fovy={fovy:.1f}°"
        )

        rgb, depth_m, geom_seg = _robocasa_scene_camera_render(env, H, W)

        ctx = env.sim._render_context_offscreen
        K = _free_camera_intrinsic(H, W, fovy)
        T = _free_camera_extrinsic_from_scene(ctx)

        seg, name_of_id = robocasa_task_segmentation(
            env, spec, None, H, W, geom_seg=geom_seg
        )
        print(f"[env] tracked: {[v[0] for v in name_of_id.values()]}")
    else:
        cid0 = env.sim.model.camera_name2id(cam)
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

        K = CU.get_camera_intrinsic_matrix(env.sim, cam, H, W)
        T = CU.get_camera_extrinsic_matrix(env.sim, cam)
        depth_m = np.asarray(CU.get_real_depth_map(env.sim, obs[f"{cam}_depth"])).squeeze()
        rgb = None
        seg = instance_segmentation(env, obs, cam, H, W)
        inst_keys = list(env.model.instances_to_ids.keys())
        name_of_id = {}                               # seg 픽셀값 = 키 순서 인덱스 + 1
        for idx, inst in enumerate(inst_keys):
            cls = _track_class(inst, spec)
            if cls:
                name_of_id[idx + 1] = (inst, cls)
        print(f"[env] instances: {inst_keys}")
        print(f"[env] tracked: {[v[0] for v in name_of_id.values()]}")

    bp = env.sim.data.get_body_xpos("robot0_base")
    base_off = (bp[0] * 1000.0, bp[1] * 1000.0, 0.0)
    print(f"[env] robot base offset (mm): {base_off[:2]}")

    if spec["robocasa"]:
        fovy = FOVY_OVERRIDE or float(env.sim.model.vis.global_.fovy)
        fovx = float(np.degrees(
            2 * np.arctan(np.tan(np.radians(fovy) / 2) * W / H)
        ))
        import mujoco
        headpos = np.zeros((3, 1))
        forward = np.zeros((3, 1))
        up = np.zeros((3, 1))
        mujoco.mjv_cameraInModel(headpos, forward, up, ctx.scn)
        cam_pos = headpos.ravel()
        print(
            f"[cam] common scene: fovy={fovy:.1f}° fovx={fovx:.1f}° "
            f"(H{H}×W{W}) pos={np.round(cam_pos, 3)} f={K[0,0]:.1f}px"
        )
    else:
        cid = env.sim.model.camera_name2id(cam)
        fovy = float(env.sim.model.cam_fovy[cid])
        fovx = float(np.degrees(
            2 * np.arctan(np.tan(np.radians(fovy) / 2) * W / H)
        ))
        cam_pos = env.sim.data.cam_xpos[cid]
        print(
            f"[cam] {cam}: fovy={fovy:.1f}° fovx={fovx:.1f}° "
            f"(H{H}×W{W}) pos={np.round(cam_pos, 3)} f={K[0,0]:.1f}px"
        )

    objects = points_from_frame(depth_m, seg, K, T, name_of_id, base_offset_mm=base_off)
    print(f"[M1] detected {len(objects)}/{len(name_of_id)} tracked instances")
    for o in objects:
        print(f"     {o['name']:24s} points={len(o['points'])}")
    m1 = build_m1(objects)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m1.json").write_text(
        json.dumps(serialize(m1), ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "m1_points.npz",
                        **{n["id"]: n["_points"] for n in m1["nodes"]})

    if rgb is None:
        rgb = np.asarray(obs[f"{cam}_image"])
    from PIL import Image
    Image.fromarray(rgb).save(OUT / "frame.png")
    ov = rgb.copy()
    for sid in name_of_id:
        ov[seg == sid] = (0.5 * ov[seg == sid] + [127, 0, 0]).astype(np.uint8)
    Image.fromarray(ov).save(OUT / "frame_masks.png")
    # 크롭 파일명 = 노드 id. inst 이름을 그대로 키로 (multi-underscore 이름도 안전)
    node_ids = {o["name"]: f"obj_{o['cls']}_{o['name']}" for o in objects}
    save_crops(rgb, seg, name_of_id, node_ids, OUT / "crops")
    print(f"[M1] nodes={len(m1['nodes'])} edges={len(m1['edges'])} "
          f"crops={len(list((OUT / 'crops').glob('*.png')))}")
    for e in m1["edges"]:
        print(f"     {e['type']:9s} {e['from']} -> {e['to']}")
    print(f"[{name}] -> {OUT}/m1.json, {OUT}/m1_points.npz")

    if view:
        import time
        import mujoco
        import mujoco.viewer
        label = "RoboCasa common scene camera" if spec["robocasa"] else cam
        print(f"[view] {label} 시점 뷰어 (창 닫으면 종료)")
        with mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data) as v:
            if spec["robocasa"]:
                v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                v.cam.lookat[:] = ctx.cam.lookat
                v.cam.distance = ctx.cam.distance
                v.cam.azimuth = ctx.cam.azimuth
                v.cam.elevation = ctx.cam.elevation
            else:
                v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                v.cam.fixedcamid = env.sim.model.camera_name2id(cam)
            while v.is_running():
                v.sync()
                time.sleep(0.02)
    env.close()


if __name__ == "__main__":
    main()