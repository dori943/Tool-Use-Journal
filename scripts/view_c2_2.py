"""C2-T2 Sandwich Assembly interactive viewer."""

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


# ============================================================
# Pose print order
# ============================================================

POSE_ORDER = [
    "Bread Plate",
    "Bread A",
    "Bread B",

    "Turkey Plate",
    "Turkey Slice 1",
    "Turkey Slice 2",
    "Turkey Slice 3",

    "Cheese Plate",
    "Cheese 1",
    "Cheese 2",
    "Cheese 3",

    "Tomato Plate",
    "Tomato Slice",

    "Serving Plate",

    "Knife",
    "Spatula",
    "Spoon",
    "Ladle",

    "EE Rack",
    "2F",
    "3F",
    "Vacuum",
    "Robot Pedestal",
]


# ============================================================
# Pose printing
# ============================================================

def _print_poses(env) -> None:

    print()
    print("=" * 78)
    print("C2-2 OBJECT / EE / PEDESTAL POSES (world)")
    print("=" * 78)

    if getattr(env, "_cheese_asset_note", None):
        print("NOTE:", env._cheese_asset_note)
        print("-" * 78)

    poses = env.get_object_poses()

    for name in POSE_ORDER:

        if name not in poses:
            continue

        pose = poses[name]

        print(f"  {name}")
        print(
            f"    position : "
            f"{_fmt_vec(pose['position'])}"
        )
        print(
            f"    quat_wxyz: "
            f"{_fmt_vec(pose['quaternion_wxyz'])}"
        )

    # Robot base
    base_id = env.sim.model.body_name2id(
        "robot0_base"
    )

    base = np.array(
        env.sim.data.body_xpos[base_id],
        dtype=float,
    )

    print("  robot0_base")
    print(
        f"    position : "
        f"{_fmt_vec(base)}"
    )

    # Reachability report
    report = env.get_reachability_report()

    print()
    print("-" * 78)

    print(
        f"Island: {env.island.name}  "
        f"surface_z={env._island_surface_z:.3f}"
    )

    print(
        f"UR5e reach limit (XY): "
        f"{report['reach_limit_m']:.3f} m"
    )

    print(
        f"C1-T2 EE radius reference: "
        f"{report['ee_radius_c1']:.6f} m"
    )

    for label, dist in report[
        "ee_xy_distances"
    ].items():

        delta = abs(
            dist - report["ee_radius_c1"]
        )

        print(
            f"  robot_base → {label:7s} "
            f"XY={dist:.6f} m  "
            f"|Δ vs C1|={delta:.3e} m"
        )

    print(
        f"  EE XY max delta = "
        f"{report['ee_xy_max_delta']:.6e} m"
    )

    print("Reachability (XY vs 0.85 m):")

    for name, row in report[
        "targets"
    ].items():

        flag = (
            "OK"
            if row["within_reach_xy"]
            else "OUT"
        )

        print(
            f"  {name:14s} "
            f"xy={row['distance_xy']:.3f}  "
            f"[{flag}]"
        )

    print("=" * 78)
    print()


# ============================================================
# Object size / mass inspection
# ============================================================

OBJECT_SPEC_NAMES = [
    "bread_a",
    "bread_b",
    "bread_plate",

    "turkey_plate",
    "turkey_1",
    "turkey_2",
    "turkey_3",

    "cheese_plate",
    "cheese_1",
    "cheese_2",
    "cheese_3",

    "tomato_plate",
    "tomato_slice",

    "serving_plate",

    "knife",
    "spatula",
    "spoon",
    "ladle",
]


def print_object_specs(env) -> None:
    print()
    print("=" * 105)
    print("C2-T2 OBJECT SPECS")
    print("=" * 105)

    print(
        f"{'Object':20s} | "
        f"{'Size (X × Y × Z)':35s} | "
        f"{'Mass':15s}"
    )

    print("-" * 105)

    for name in OBJECT_SPEC_NAMES:

        # ====================================================
        # Object model
        # ====================================================

        obj = getattr(
            env,
            name,
            None,
        )

        # ====================================================
        # Size
        # ====================================================

        if (
            obj is not None
            and hasattr(obj, "bbox_full_size_m")
        ):
            try:
                size_m = np.asarray(
                    obj.bbox_full_size_m,
                    dtype=float,
                )

                size_mm = (
                    size_m * 1000.0
                )

                size_text = (
                    f"{size_mm[0]:.2f} × "
                    f"{size_mm[1]:.2f} × "
                    f"{size_mm[2]:.2f} mm"
                )

            except Exception as exc:
                size_text = (
                    f"N/A ({exc})"
                )

        else:
            size_text = "N/A"

        # ====================================================
        # Mass
        # ====================================================

        try:
            body_id = (
                env.obj_body_id[name]
            )

            mass_kg = float(
                env.sim.model.body_mass[
                    body_id
                ]
            )

            mass_g = (
                mass_kg * 1000.0
            )

            mass_text = (
                f"{mass_g:.2f} g"
            )

        except Exception as exc:
            mass_text = (
                f"N/A ({exc})"
            )

        # ====================================================
        # Output
        # ====================================================

        print(
            f"{name:20s} | "
            f"{size_text:35s} | "
            f"{mass_text:15s}"
        )

    print("=" * 105)
    print()
# ============================================================
# Compare with C1-T2
# ============================================================

def _compare_with_c1(env_c2) -> None:
    """
    Load C1-T2 briefly and compare
    robot / EE / pedestal layout.
    """

    import robosuite as suite

    import environments.c1_2_dough_flatten  # noqa: F401

    print(
        "Comparing C2-T2 robot/EE layout "
        "against C1-T2..."
    )

    env_c1 = suite.make(
        env_name="C1_2_DoughFlatten",
        robots="UR5e",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        seed=0,
        render_camera=None,
    )

    env_c1.reset()

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def base_xy(env):

        bid = (
            env.sim.model.body_name2id(
                "robot0_base"
            )
        )

        return np.array(
            env.sim.data.body_xpos[
                bid
            ][:2],
            dtype=float,
        )

    def ee_xy(env, label):

        return (
            env.get_object_poses()[
                label
            ]["position"][:2]
        )

    def ped_xy(env):

        poses = (
            env.get_object_poses()
        )

        key = (
            "robot_pedestal"
            if "robot_pedestal" in poses
            else "Robot Pedestal"
        )

        return (
            poses[key]["position"][:2]
        )

    # --------------------------------------------------------
    # Robot
    # --------------------------------------------------------

    c1_base = base_xy(env_c1)
    c2_base = base_xy(env_c2)

    print(
        f"  robot XY "
        f"C1={c1_base} "
        f"C2={c2_base} "
        f"Δ={np.linalg.norm(c1_base - c2_base):.3e}"
    )

    # --------------------------------------------------------
    # Pedestal
    # --------------------------------------------------------

    c1_ped = ped_xy(env_c1)
    c2_ped = ped_xy(env_c2)

    print(
        f"  pedestal XY "
        f"Δ={np.linalg.norm(c1_ped - c2_ped):.3e}"
    )

    print(
        f"  pedestal top_z "
        f"C1={env_c1.pedestal_info['top_z']} "
        f"C2={env_c2.pedestal_info['top_z']}"
    )

    # --------------------------------------------------------
    # EE
    # --------------------------------------------------------

    for label in (
        "2F",
        "3F",
        "Vacuum",
    ):

        d = np.linalg.norm(
            ee_xy(env_c1, label)
            - ee_xy(env_c2, label)
        )

        print(
            f"  {label} XY Δ={d:.3e}"
        )

    # --------------------------------------------------------
    # EE radius
    # --------------------------------------------------------

    r1 = (
        env_c1.get_reachability_report()
    )

    r2 = (
        env_c2.get_reachability_report()
    )

    for label in (
        "2F",
        "3F",
        "Vacuum",
    ):

        d1 = (
            r1["ee_xy_distances"][
                label
            ]
        )

        d2 = (
            r2["ee_xy_distances"][
                label
            ]
        )

        print(
            f"  radius {label}: "
            f"C1={d1:.6f} "
            f"C2={d2:.6f} "
            f"Δ={abs(d1 - d2):.3e}"
        )

    env_c1.close()

    print(
        "C1-T2 comparison done.\n"
    )


# ============================================================
# Camera
# ============================================================

def _apply_camera(env) -> None:

    mujoco_viewer = (
        env.viewer.viewer
    )

    if mujoco_viewer is None:
        return

    poses = (
        env.get_object_poses()
    )

    pts = [
        poses[n]["position"]
        for n in poses
    ]

    base_id = (
        env.sim.model.body_name2id(
            "robot0_base"
        )
    )

    pts.append(
        np.array(
            env.sim.data.body_xpos[
                base_id
            ],
            dtype=float,
        )
    )

    center = np.mean(
        np.stack(
            pts,
            axis=0,
        ),
        axis=0,
    )

    cam = mujoco_viewer.cam

    cam.lookat[:] = center
    cam.distance = 3.0
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

        from environments.c2_2_sandwich_assembly import (
            C2_2_SandwichAssembly,  # noqa: F401
        )

    except ImportError as exc:

        print(
            "ERROR: C2_2_SandwichAssembly "
            "import failed.\n"
            "Use conda env robocasa from "
            "Tool-Use-Journal root.\n"
            f"Cause: {exc}",
            file=sys.stderr,
        )

        return 1

    import robosuite as suite

    # --------------------------------------------------------
    # Create C2-T2
    # --------------------------------------------------------

    print(
        "Loading C2_2_SandwichAssembly "
        "(UR5e + pedestal, Island)..."
    )

    env = suite.make(
        env_name="C2_2_SandwichAssembly",
        robots="UR5e",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
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
    # Print poses
    # --------------------------------------------------------

    _print_poses(env)

    # --------------------------------------------------------
    # Print actual object size / mass
    # --------------------------------------------------------

    print_object_specs(env)

    # --------------------------------------------------------
    # Compare C1-T2
    # --------------------------------------------------------

    _compare_with_c1(env)

    # --------------------------------------------------------
    # Viewer
    # --------------------------------------------------------

    print(
        "Scene ready. "
        "Opening MuJoCo viewer..."
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
            "ERROR: Failed to launch viewer.",
            file=sys.stderr,
        )

        env.close()

        return 1

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    _apply_camera(env)

    env.viewer.update()

    # --------------------------------------------------------
    # Viewer loop
    # --------------------------------------------------------

    try:

        while (
            mujoco_viewer.is_running()
        ):

            env.viewer.update()

            time.sleep(0.01)

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