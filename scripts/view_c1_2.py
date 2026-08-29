"""C1-T2 Dough Flatten 환경 인터랙티브 뷰어."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fmt_vec(values) -> str:
    return "[" + ", ".join(f"{float(v):7.3f}" for v in values) + "]"


def _print_poses(env) -> None:
    print()
    print("=" * 78)
    print("C1-2 OBJECT / EE / PEDESTAL POSES (world)")
    print("=" * 78)
    poses = env.get_object_poses()
    order = [
        "cutting_board",
        "dough",
        "bottle",
        "spatula",
        "spoon",
        "tongs",
        "ee_rack",
        "2F",
        "3F",
        "Vacuum",
        "robot_pedestal",
    ]
    for name in order:
        if name not in poses:
            continue
        pose = poses[name]
        print(f"  {name:14s}")
        print(f"    position : {_fmt_vec(pose['position'])}")
        print(f"    quat_wxyz: {_fmt_vec(pose['quaternion_wxyz'])}")

    base_id = env.sim.model.body_name2id("robot0_base")
    base = np.array(env.sim.data.body_xpos[base_id], dtype=float)
    print(f"  {'robot0_base':14s}")
    print(f"    position : {_fmt_vec(base)}")

    report = env.get_reachability_report()
    print()
    print("-" * 78)
    print(f"Island: {env.island.name}  surface_z={env._island_surface_z:.3f}")
    if env._island_bounds:
        b = env._island_bounds
        print(
            f"Island XY bounds: x[{b['xmin']:.3f},{b['xmax']:.3f}] "
            f"y[{b['ymin']:.3f},{b['ymax']:.3f}]"
        )
    print(f"UR5e reach limit (XY): {report['reach_limit_m']:.3f} m")
    print(f"C1-T1 EE radius: {report['ee_radius_c1']:.6f} m")
    for label, dist in report["ee_xy_distances"].items():
        print(f"  robot_base → {label:7s} XY dist = {dist:.6f} m")
    print(f"  EE XY max delta = {report['ee_xy_max_delta']:.6e} m")
    print("Reachability (XY vs UR5e 0.85m):")
    for name, row in report["targets"].items():
        flag = "OK" if row["within_reach_xy"] else "OUT"
        print(
            f"  {name:8s} xy={row['distance_xy']:.3f}  "
            f"3d={row['distance_3d']:.3f}  [{flag}]"
        )
    print("=" * 78)
    print()


def _apply_work_area_camera(env) -> None:
    """Island 실험 영역(도구/보드/EE/robot/pedestal)이 한 화면에 보이게."""
    mujoco_viewer = env.viewer.viewer
    if mujoco_viewer is None:
        return

    poses = env.get_object_poses()
    pts = [poses[n]["position"] for n in poses]
    base_id = env.sim.model.body_name2id("robot0_base")
    pts.append(np.array(env.sim.data.body_xpos[base_id], dtype=float))
    center = np.mean(np.stack(pts, axis=0), axis=0)

    cam = mujoco_viewer.cam
    cam.lookat[:] = center
    cam.distance = 2.8
    # robot(-x) 쪽에서 island(+x)를 바라봄 → tool row가 좌→우로 보임
    cam.azimuth = 90.0
    cam.elevation = -28.0


def main() -> int:
    try:
        from environments.c1_2_dough_flatten import C1_2_DoughFlatten  # noqa: F401
    except ImportError as exc:
        print(
            "ERROR: C1_2_DoughFlatten를 import할 수 없습니다.\n"
            "conda robocasa 환경에서 Tool-Use-Journal 루트로 실행하세요.\n"
            f"원인: {exc}",
            file=sys.stderr,
        )
        return 1

    import robosuite as suite

    print("Loading C1_2_DoughFlatten (UR5e + pedestal, Island, Layout004/Style002)...")

    env = suite.make(
        env_name="C1_2_DoughFlatten",
        robots="UR5e",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera=None,
        ignore_done=True,
        renderer="mjviewer",
        seed=0,
    )

    env.reset()
    _print_poses(env)

    print("Scene ready. Opening MuJoCo viewer...")
    print(
        "Inspect: Island, Cutting Board, Dough, Tools, EE Rack, "
        "Robot Pedestal, UR5e"
    )
    print("Close the viewer window (or press Ctrl+C) to exit.")

    env.viewer.update()
    mujoco_viewer = env.viewer.viewer
    if mujoco_viewer is None:
        print("ERROR: Failed to launch viewer.", file=sys.stderr)
        env.close()
        return 1

    _apply_work_area_camera(env)
    env.viewer.update()

    try:
        while mujoco_viewer.is_running():
            env.viewer.update()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        env.close()

    print("Viewer closed. Exiting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
