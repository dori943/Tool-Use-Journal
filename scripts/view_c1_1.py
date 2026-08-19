"""C1-1 레고 스윕 환경 인터랙티브 뷰어."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from view_c2_1 import (
    get_object_world_aabb,
    get_total_body_mass,
)


def print_object_physical_specs(env):
    print()
    print("=" * 72)
    print("C1-1 OBJECT PHYSICAL SPECS")
    print("=" * 72)

    spec_objects = list(env.blocks) + [
        env.light_plate,
        env.heavy_plate,
        env.bottle_distractor,
    ]

    plate_sizes_mm = {}

    for obj in spec_objects:
        root_body_id = env.sim.model.body_name2id(obj.root_body)
        mass_kg = get_total_body_mass(env.sim, root_body_id)
        mass_g = mass_kg * 1000.0
        aabb = get_object_world_aabb(env.sim, root_body_id)

        print()
        print(f"[OBJECT] {obj.name}")
        print(f"  class : {obj.__class__.__name__}")
        print(f"  mass  : {mass_kg:.6f} kg ({mass_g:.2f} g)")

        if aabb is not None:
            size_mm = aabb["size_xyz_m"] * 1000.0
            print(
                "  size  : "
                f"{size_mm[0]:.2f} × "
                f"{size_mm[1]:.2f} × "
                f"{size_mm[2]:.2f} mm"
            )
            print("          X × Y × Z")
            if obj.name in ("light_plate", "heavy_plate"):
                plate_sizes_mm[obj.name] = size_mm
        else:
            print("  size  : could not calculate")

    print()
    print("=" * 72)

    if (
        "light_plate" in plate_sizes_mm
        and "heavy_plate" in plate_sizes_mm
    ):
        light_size = plate_sizes_mm["light_plate"]
        heavy_size = plate_sizes_mm["heavy_plate"]
        same_size = np.allclose(light_size, heavy_size, atol=1e-3)
        print()
        print("Light / Heavy plate size match :", same_size)
        print(
            "  light_plate : "
            f"{light_size[0]:.2f} × {light_size[1]:.2f} × {light_size[2]:.2f} mm"
        )
        print(
            "  heavy_plate : "
            f"{heavy_size[0]:.2f} × {heavy_size[1]:.2f} × {heavy_size[2]:.2f} mm"
        )
        print()
        print("=" * 72)

    print()


def main() -> int:

    import environments  # noqa: F401
    import robosuite as suite

    print("Loading C1_1_LegoSweep (UR5e)...")

    env = suite.make(
        env_name="C1_1_LegoSweep",
        robots="UR5e",

        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,

        render_camera="frontview",

        ignore_done=True,
    )

    env.reset()

    print_object_physical_specs(env)

    print("Scene ready. Opening MuJoCo viewer...")
    print(
        "Inspect: UR5e, EE rack, tools, "
        "12 Lego blocks, and Collection Zone."
    )

    print(
        "Use the MuJoCo viewer mouse controls "
        "to rotate / pan / zoom."
    )

    print(
        "Close the viewer window "
        "(or press Ctrl+C) to exit."
    )

    env.viewer.update()

    mujoco_viewer = env.viewer.viewer

    if mujoco_viewer is None:

        print(
            "ERROR: Failed to launch viewer.",
            file=sys.stderr,
        )

        env.close()

        return 1

    try:

        while mujoco_viewer.is_running():

            env.viewer.update()

            time.sleep(
                0.01
            )

    except KeyboardInterrupt:

        print(
            "\nInterrupted."
        )

    finally:

        env.close()

    print(
        "Viewer closed. Exiting."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )