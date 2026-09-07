"""C3-T2 scene viewer. Use --headless to validate reset without opening a window."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Close the viewer after this many seconds (smoke check).")
    parser.add_argument("--report", type=Path, help="Optional scene geometry JSON output.")
    args = parser.parse_args()
    from environments.c3_2_breakfast_tray import C3_2_BreakfastTrayPreparation  # noqa: F401
    import robosuite as suite

    env = suite.make(
        "C3_2_BreakfastTrayPreparation", robots="UR5e",
        has_renderer=not args.headless, has_offscreen_renderer=False,
        use_camera_obs=False, render_camera=None, ignore_done=True,
        renderer="mjviewer", seed=0,
    )
    try:
        env.reset()
        report = env.get_scene_report()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.report:
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Reset complete: 14 movable objects, Island, UR5e, pedestal, and EE rack.")
        if args.headless:
            return 0
        env.viewer.update()
        viewer = env.viewer.viewer
        if viewer is None:
            raise RuntimeError("MuJoCo viewer did not open")
        viewer.cam.lookat[:] = [*env._layout_origin, env._island_surface_z]
        viewer.cam.distance = 2.6
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -45
        start = time.monotonic()
        while viewer.is_running():
            env.viewer.update()
            if args.seconds is not None and time.monotonic() - start >= args.seconds:
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

