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


# ============================================================
# UR5e reach
# ============================================================

UR5E_REACH_M = 0.85


# ============================================================
# Formatting
# ============================================================

def _fmt_vec(values) -> str:
    return "[" + ", ".join(f"{float(v):7.3f}" for v in values) + "]"


# ============================================================
# Object physical specs
# ============================================================

def print_object_physical_specs(env):
    """객체의 실제 크기와 질량 출력."""

    print()
    print("=" * 72)
    print("C1-1 OBJECT PHYSICAL SPECS")
    print("=" * 72)

    spec_objects = list(env.blocks) + [
        env.plate,
        env.ladle,
        env.scissors,
        env.fork,
        env.bottle_distractor,
    ]

    for obj in spec_objects:

        root_body_id = env.sim.model.body_name2id(
            obj.root_body
        )

        # ----------------------------------------------------
        # Mass
        # ----------------------------------------------------

        mass_kg = get_total_body_mass(
            env.sim,
            root_body_id,
        )

        mass_g = mass_kg * 1000.0

        # ----------------------------------------------------
        # AABB size
        # ----------------------------------------------------

        aabb = get_object_world_aabb(
            env.sim,
            root_body_id,
        )

        print()
        print(f"[OBJECT] {obj.name}")
        print(f"  class : {obj.__class__.__name__}")

        print(
            f"  mass  : {mass_kg:.6f} kg "
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
                "          X × Y × Z"
            )

        else:

            print(
                "  size  : could not calculate"
            )

    print()
    print("=" * 72)
    print()


# ============================================================
# Position + Reachability
# ============================================================

def print_object_positions_and_reachability(env):
    """
    C1-T1의 주요 객체 / EE 위치를 출력하고,
    robot0_base 기준 XY / 3D 거리를 계산한다.
    """

    print()
    print("=" * 78)
    print("C1-1 OBJECT / EE POSITIONS & REACHABILITY")
    print("=" * 78)

    # --------------------------------------------------------
    # Robot base
    # --------------------------------------------------------

    base_id = env.sim.model.body_name2id("robot0_base")

    robot_base = np.array(
        env.sim.data.body_xpos[base_id],
        dtype=float,
    )

    print()
    print("robot0_base")
    print(
        f"  position : {_fmt_vec(robot_base)}"
    )

    print()
    print("-" * 78)
    print("OBJECT POSITIONS")
    print("-" * 78)

    # --------------------------------------------------------
    # Objects
    # --------------------------------------------------------

    spec_objects = list(env.blocks) + [
        env.plate,
        env.ladle,
        env.scissors,
        env.fork,
        env.bottle_distractor,
    ]

    object_positions = {}

    for obj in spec_objects:

        try:

            body_id = env.sim.model.body_name2id(
                obj.root_body
            )

            pos = np.array(
                env.sim.data.body_xpos[body_id],
                dtype=float,
            )

            quat = np.array(
                env.sim.data.body_xquat[body_id],
                dtype=float,
            )

            object_positions[obj.name] = pos

            print()
            print(obj.name)

            print(
                f"  position : {_fmt_vec(pos)}"
            )

            print(
                f"  quat_wxyz: {_fmt_vec(quat)}"
            )

        except Exception as exc:

            print()
            print(
                f"{obj.name}: "
                f"could not get position ({exc})"
            )

    # --------------------------------------------------------
    # EE rack / EE
    # --------------------------------------------------------

    ee_positions = {}

    print()
    print("-" * 78)
    print("EE POSITIONS")
    print("-" * 78)

    # EE rack 자체
    try:

        rack_id = env.sim.model.body_name2id(
            "ee_rack"
        )

        rack_pos = np.array(
            env.sim.data.body_xpos[rack_id],
            dtype=float,
        )

        rack_quat = np.array(
            env.sim.data.body_xquat[rack_id],
            dtype=float,
        )

        print()
        print("ee_rack")
        print(
            f"  position : {_fmt_vec(rack_pos)}"
        )
        print(
            f"  quat_wxyz: {_fmt_vec(rack_quat)}"
        )

    except Exception:

        pass

    # --------------------------------------------------------
    # ee_rack_info가 환경에 있으면 이를 이용
    # --------------------------------------------------------

    if hasattr(env, "ee_rack_info") and env.ee_rack_info:

        for ee_id, info in env.ee_rack_info.items():

            body_name = info.get("rack_body")

            if not body_name:
                continue

            try:

                body_id = env.sim.model.body_name2id(
                    body_name
                )

                pos = np.array(
                    env.sim.data.body_xpos[body_id],
                    dtype=float,
                )

                quat = np.array(
                    env.sim.data.body_xquat[body_id],
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

                ee_positions[label] = pos

                print()
                print(label)

                print(
                    f"  position : {_fmt_vec(pos)}"
                )

                print(
                    f"  quat_wxyz: {_fmt_vec(quat)}"
                )

            except Exception as exc:

                print(
                    f"{ee_id}: "
                    f"could not get EE position ({exc})"
                )

    # --------------------------------------------------------
    # Reachability
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(
        f"REACHABILITY FROM robot0_base "
        f"(UR5e XY limit = {UR5E_REACH_M:.2f} m)"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # Object distances
    # --------------------------------------------------------

    for name, pos in object_positions.items():

        delta = pos - robot_base

        distance_xy = float(
            np.linalg.norm(delta[:2])
        )

        distance_3d = float(
            np.linalg.norm(delta)
        )

        within_reach = (
            distance_xy <= UR5E_REACH_M
        )

        flag = (
            "OK"
            if within_reach
            else "OUT"
        )

        margin = (
            UR5E_REACH_M
            - distance_xy
        )

        print(
            f"{name:20s} "
            f"xy={distance_xy:.3f} m  "
            f"3d={distance_3d:.3f} m  "
            f"margin={margin:+.3f} m  "
            f"[{flag}]"
        )

    # --------------------------------------------------------
    # EE distances
    # --------------------------------------------------------

    if ee_positions:

        print()
        print("-" * 78)
        print("EE DISTANCES")
        print("-" * 78)

        ee_distances = {}

        for name, pos in ee_positions.items():

            delta = pos - robot_base

            distance_xy = float(
                np.linalg.norm(delta[:2])
            )

            distance_3d = float(
                np.linalg.norm(delta)
            )

            ee_distances[name] = distance_xy

            within_reach = (
                distance_xy <= UR5E_REACH_M
            )

            flag = (
                "OK"
                if within_reach
                else "OUT"
            )

            margin = (
                UR5E_REACH_M
                - distance_xy
            )

            print(
                f"{name:20s} "
                f"xy={distance_xy:.6f} m  "
                f"3d={distance_3d:.3f} m  "
                f"margin={margin:+.3f} m  "
                f"[{flag}]"
            )

        # ----------------------------------------------------
        # EE들이 동일 반경인지 확인
        # ----------------------------------------------------

        if len(ee_distances) >= 2:

            values = list(
                ee_distances.values()
            )

            max_delta = (
                max(values)
                - min(values)
            )

            print()
            print(
                "EE XY max delta = "
                f"{max_delta:.6e} m"
            )

    print()
    print("=" * 78)
    print()


# ============================================================
# Main
# ============================================================

def main() -> int:

    import environments  # noqa: F401
    import robosuite as suite

    print(
        "Loading C1_1_LegoSweep (UR5e)..."
    )

    env = suite.make(
        env_name="C1_1_LegoSweep",
        robots="UR5e",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera="frontview",
        ignore_done=True,
    )

    # ========================================================
    # Reset
    # ========================================================

    env.reset()

    # ========================================================
    # Physical specs
    # ========================================================

    print_object_physical_specs(
        env
    )

    # ========================================================
    # Position + Reachability
    # ========================================================

    print_object_positions_and_reachability(
        env
    )

    # ========================================================
    # Viewer
    # ========================================================

    print(
        "Scene ready. Opening MuJoCo viewer..."
    )

    print(
        "Inspect: UR5e, EE rack, tools, "
        "12 Lego blocks, and Collection Zone."
    )

    print(
        "Front view enabled."
    )

    print(
        "Close the viewer window "
        "(or press Ctrl+C) to exit."
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

    # ========================================================
    # Viewer loop
    # ========================================================

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