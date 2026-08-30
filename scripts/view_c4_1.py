"""C4-T1 Interval-Fit Extraction interactive viewer."""

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


# ============================================================
# Scene objects
# ============================================================

SCENE_OBJECTS = [
    "appliance_left",
    "appliance_right",
    "card",
    "tool_1_knife",
    "tool_2_spatula_a",
    "tool_3_spatula_b",
    "tool_4_spatula_c",
    "tool_5_cutting_board",
]


# ============================================================
# Formatting
# ============================================================

def _fmt_vec(values) -> str:
    return "[" + ", ".join(
        f"{float(v):7.3f}"
        for v in values
    ) + "]"


# ============================================================
# Object physical specs
# ============================================================

def print_object_physical_specs(env):
    """
    C4-T1 객체의 실제 MuJoCo 크기와 질량을 출력한다.

    - mass:
      root body와 그 하위 body들의 실제 MuJoCo mass 합산

    - size:
      현재 simulation geometry를 기준으로 계산한
      world-space AABB 크기
    """

    print()
    print("=" * 78)
    print("C4-T1 OBJECT PHYSICAL SPECS")
    print("=" * 78)

    for name in SCENE_OBJECTS:

        if name not in env.objects:
            continue

        obj = env.objects[name]

        try:
            root_body_id = env.sim.model.body_name2id(
                obj.root_body
            )

        except Exception as exc:
            print()
            print(
                f"[OBJECT] {name}"
            )
            print(
                f"  could not find body: {exc}"
            )
            continue

        # ----------------------------------------------------
        # Mass
        # ----------------------------------------------------

        mass_kg = get_total_body_mass(
            env.sim,
            root_body_id,
        )

        mass_g = (
            mass_kg
            * 1000.0
        )

        # ----------------------------------------------------
        # Actual world AABB
        # ----------------------------------------------------

        aabb = get_object_world_aabb(
            env.sim,
            root_body_id,
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print()
        print(
            f"[OBJECT] {name}"
        )

        print(
            f"  class : "
            f"{obj.__class__.__name__}"
        )

        print(
            f"  mass  : "
            f"{mass_kg:.6f} kg "
            f"({mass_g:.2f} g)"
        )

        if aabb is not None:

            size_mm = (
                aabb["size_xyz_m"]
                * 1000.0
            )

            print(
                "  size  : "
                f"{size_mm[0]:.2f} × "
                f"{size_mm[1]:.2f} × "
                f"{size_mm[2]:.2f} mm"
            )

            print(
                "          X × Y × Z "
                "(world AABB)"
            )

        else:
            print(
                "  size  : "
                "could not calculate"
            )

    print()
    print("=" * 78)
    print()


# ============================================================
# Scene information
# ============================================================

def _print_scene_info(env) -> None:

    print()
    print("=" * 78)
    print(
        "C4-T1 INTERVAL-FIT EXTRACTION"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # Task constraints
    # --------------------------------------------------------

    print(
        f"Entrance gap g       : "
        f"{env._layout_meta['entrance_gap_m'] * 1000:.1f} mm"
    )

    print(
        f"Internal width W     : "
        f"{env._layout_meta['internal_width_m'] * 1000:.1f} mm"
    )

    print(
        f"Required width w_req : "
        f"{env._layout_meta['width_requirement_m'] * 1000:.1f} mm"
    )

    print(
        f"Required reach       : "
        f"{env._layout_meta['required_reach_m'] * 1000:.1f} mm"
    )

    print()

    # --------------------------------------------------------
    # Tool geometry
    # --------------------------------------------------------

    print("-" * 78)
    print(
        "TOOL TARGET GEOMETRY"
    )
    print("-" * 78)

    tool_meta = (
        env._layout_meta.get(
            "tool_geometry_m",
            {},
        )
    )

    for name, spec in tool_meta.items():

        thickness = (
            spec["thickness_m"]
            * 1000.0
        )

        width = (
            spec["width_m"]
            * 1000.0
        )

        length = (
            spec["length_m"]
            * 1000.0
        )

        print(
            f"{name:28s} "
            f"t={thickness:6.1f} mm  "
            f"w={width:6.1f} mm  "
            f"l={length:6.1f} mm"
        )

    print()

    # --------------------------------------------------------
    # Object poses
    # --------------------------------------------------------

    print("-" * 78)
    print(
        "OBJECT POSES"
    )
    print("-" * 78)

    for name in SCENE_OBJECTS:

        if name not in env.objects:
            continue

        obj = env.objects[name]

        try:

            body_id = (
                env.sim.model.body_name2id(
                    obj.root_body
                )
            )

        except Exception:
            continue

        pos = np.asarray(
            env.sim.data.body_xpos[
                body_id
            ],
            dtype=float,
        )

        quat = np.asarray(
            env.sim.data.body_xquat[
                body_id
            ],
            dtype=float,
        )

        print()
        print(
            f"  {name}"
        )

        print(
            f"    position : "
            f"{_fmt_vec(pos)}"
        )

        print(
            f"    quat_wxyz: "
            f"{_fmt_vec(quat)}"
        )

    # --------------------------------------------------------
    # Robot base
    # --------------------------------------------------------

    base_id = (
        env.sim.model.body_name2id(
            "robot0_base"
        )
    )

    base_pos = np.asarray(
        env.sim.data.body_xpos[
            base_id
        ],
        dtype=float,
    )

    print()
    print(
        "  robot0_base"
    )

    print(
        f"    position : "
        f"{_fmt_vec(base_pos)}"
    )

    # --------------------------------------------------------
    # EE rack
    # --------------------------------------------------------

    print()
    print("-" * 78)
    print(
        "EE RACK"
    )
    print("-" * 78)

    for ee_id, info in (
        env.ee_rack_info.items()
    ):

        body_name = (
            info.get(
                "rack_body"
            )
        )

        if not body_name:
            continue

        try:

            body_id = (
                env.sim.model.body_name2id(
                    body_name
                )
            )

        except Exception:
            continue

        pos = np.asarray(
            env.sim.data.body_xpos[
                body_id
            ],
            dtype=float,
        )

        label = {
            "2F": "2F",
            "3F": "3F",
            "vac": "Vacuum",
        }.get(
            ee_id,
            ee_id,
        )

        print(
            f"  {label:8s} "
            f"{_fmt_vec(pos)}"
        )

    print()
    print("=" * 78)
    print()


# ============================================================
# Camera
# ============================================================

def _apply_camera(env) -> None:
    """
    다른 task viewer와 동일하게
    free MuJoCo viewer camera를 사용한다.
    """

    mujoco_viewer = (
        env.viewer.viewer
    )

    if mujoco_viewer is None:
        return

    points = []

    # --------------------------------------------------------
    # Scene object positions
    # --------------------------------------------------------

    for name in SCENE_OBJECTS:

        if name not in env.objects:
            continue

        obj = env.objects[name]

        try:

            body_id = (
                env.sim.model.body_name2id(
                    obj.root_body
                )
            )

            pos = np.asarray(
                env.sim.data.body_xpos[
                    body_id
                ],
                dtype=float,
            )

            points.append(
                pos
            )

        except Exception:
            continue

    # --------------------------------------------------------
    # EE rack positions
    # --------------------------------------------------------

    for _, info in (
        env.ee_rack_info.items()
    ):

        body_name = (
            info.get(
                "rack_body"
            )
        )

        if not body_name:
            continue

        try:

            body_id = (
                env.sim.model.body_name2id(
                    body_name
                )
            )

            pos = np.asarray(
                env.sim.data.body_xpos[
                    body_id
                ],
                dtype=float,
            )

            points.append(
                pos
            )

        except Exception:
            continue

    # --------------------------------------------------------
    # Robot base
    # --------------------------------------------------------

    base_id = (
        env.sim.model.body_name2id(
            "robot0_base"
        )
    )

    base_pos = np.asarray(
        env.sim.data.body_xpos[
            base_id
        ],
        dtype=float,
    )

    points.append(
        base_pos
    )

    # --------------------------------------------------------
    # Scene center
    # --------------------------------------------------------

    if points:

        center = np.mean(
            np.stack(
                points,
                axis=0,
            ),
            axis=0,
        )

    else:

        center = np.array(
            [
                0.0,
                0.0,
                0.8,
            ],
            dtype=float,
        )

    # --------------------------------------------------------
    # Camera pose
    # --------------------------------------------------------

    cam = (
        mujoco_viewer.cam
    )

    cam.lookat[:] = (
        center
    )

    # 전체 scene이 한 화면에 보이도록
    cam.distance = 3.0

    # C2-T2와 비슷한 사선 시점
    cam.azimuth = 90.0
    cam.elevation = -30.0


# ============================================================
# Main
# ============================================================

def main() -> int:

    # --------------------------------------------------------
    # Import environment
    # --------------------------------------------------------

    try:

        from environments.c4_1_interval_fit_extraction import (
            C4_1_IntervalFitExtraction,  # noqa: F401
        )

    except ImportError as exc:

        print(
            "ERROR: "
            "C4_1_IntervalFitExtraction "
            "import failed.\n"
            "Run from Tool-Use-Journal root "
            "with the robocasa environment.\n"
            f"Cause: {exc}",
            file=sys.stderr,
        )

        return 1

    import robosuite as suite

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    print(
        "Loading "
        "C4_1_IntervalFitExtraction "
        "(UR5e + pedestal, Island)..."
    )

    env = suite.make(
        env_name=(
            "C4_1_IntervalFitExtraction"
        ),
        robots="UR5e",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,

        # 자유 MuJoCo viewer 사용
        render_camera=None,

        ignore_done=True,
        renderer="mjviewer",
        seed=0,
    )

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    env.reset()

    # --------------------------------------------------------
    # Physical specs
    # --------------------------------------------------------

    print_object_physical_specs(
        env
    )

    # --------------------------------------------------------
    # Scene info
    # --------------------------------------------------------

    _print_scene_info(
        env
    )

    # --------------------------------------------------------
    # Viewer
    # --------------------------------------------------------

    print(
        "Scene ready. "
        "Opening MuJoCo viewer..."
    )

    print(
        "Check:"
    )

    print(
        "  - UR5e + pedestal"
    )

    print(
        "  - 2F / 3F / Vacuum EE rack"
    )

    print(
        "  - appliance gap"
    )

    print(
        "  - card"
    )

    print(
        "  - five tools"
    )

    print(
        "Close the viewer window "
        "(or Ctrl+C) to exit."
    )

    env.viewer.update()

    mujoco_viewer = (
        env.viewer.viewer
    )

    if mujoco_viewer is None:

        print(
            "ERROR: "
            "Failed to launch viewer.",
            file=sys.stderr,
        )

        env.close()

        return 1

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    _apply_camera(
        env
    )

    env.viewer.update()

    # --------------------------------------------------------
    # Viewer loop
    # --------------------------------------------------------

    try:

        while (
            mujoco_viewer.is_running()
        ):

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
        "Viewer closed."
    )

    return 0


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )