"""KitchenBase (Layout004 + Style002) 인터랙티브 뷰어."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _apply_layout_camera(env) -> None:
    """Layout에 맞는 free-camera 기본 각도를 viewer에 적용한다."""

    try:
        from robocasa.utils.camera_utils import (
            DEFAULT_LAYOUT_CAM,
            LAYOUT_CAMS,
        )
    except ImportError:
        return

    layout_id = int(getattr(env, "layout_id", 4))
    cam_cfg = LAYOUT_CAMS.get(layout_id, DEFAULT_LAYOUT_CAM)

    mujoco_viewer = env.viewer.viewer
    if mujoco_viewer is None:
        return

    cam = mujoco_viewer.cam
    cam.lookat[:] = cam_cfg["lookat"]
    cam.distance = cam_cfg["distance"]
    cam.azimuth = cam_cfg["azimuth"]
    cam.elevation = cam_cfg["elevation"]


def main() -> int:
    # 커스텀 환경을 먼저 import해야 KitchenBase가 등록된다.
    # (RoboCasa 미설치 시 kitchen_base soft-import가 실패할 수 있어 직접 import)
    try:
        from environments.kitchen_base import (
            DEFAULT_KITCHEN_LAYOUT_ID,
            DEFAULT_KITCHEN_STYLE_ID,
        )
    except ImportError as exc:
        print(
            "ERROR: RoboCasa KitchenBase를 import할 수 없습니다.\n"
            "conda robocasa 환경에서 실행하세요.\n"
            f"원인: {exc}",
            file=sys.stderr,
        )
        return 1

    import robosuite as suite

    print(
        "Loading KitchenBase "
        f"(layout={DEFAULT_KITCHEN_LAYOUT_ID.name}, "
        f"style={DEFAULT_KITCHEN_STYLE_ID.name}, robot=PandaOmron)..."
    )

    env = suite.make(
        env_name="KitchenBase",
        robots="PandaOmron",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera=None,
        ignore_done=True,
        renderer="mjviewer",
    )

    env.reset()

    print("Scene ready. Opening MuJoCo viewer...")
    print(
        "Inspect: RoboCasa Kitchen "
        f"{DEFAULT_KITCHEN_LAYOUT_ID.name} / "
        f"{DEFAULT_KITCHEN_STYLE_ID.name}"
    )
    print("Use the MuJoCo viewer mouse controls to rotate / pan / zoom.")
    print("Close the viewer window (or press Ctrl+C) to exit.")

    env.viewer.update()

    mujoco_viewer = env.viewer.viewer
    if mujoco_viewer is None:
        print("ERROR: Failed to launch viewer.", file=sys.stderr)
        env.close()
        return 1

    _apply_layout_camera(env)
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
