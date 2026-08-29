"""Interactive viewer for C4-T2 Diagonal-Fit Packing."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# Constants
# ============================================================

UR5E_REACH_M = 0.85

OBJECT_NAMES = (
    "rolling_pin",
    "baguette",
    "whisk",
    "cereal",
    "milk",
)


# ============================================================
# Formatting
# ============================================================

def _fmt_vec(values) -> str:
    return "[" + ", ".join(
        f"{float(v):7.3f}"
        for v in values
    ) + "]"


# ============================================================
# MuJoCo physical helpers
# ============================================================

def get_total_body_mass(
    sim,
    root_body_id,
):
    """
    root body와 모든 하위 body의 실제 MuJoCo mass를 합산한다.
    """

    total = 0.0

    for body_id in range(
        sim.model.nbody
    ):
        current = body_id

        while current != 0:

            if current == root_body_id:

                total += float(
                    sim.model.body_mass[
                        body_id
                    ]
                )

                break

            current = int(
                sim.model.body_parentid[
                    current
                ]
            )

    return total


def get_object_world_aabb(
    sim,
    root_body_id,
):
    """
    root body 아래의 box / mesh geom들을 이용해
    world-space AABB를 계산한다.
    """

    points = []

    for geom_id in range(
        sim.model.ngeom
    ):

        current = int(
            sim.model.geom_bodyid[
                geom_id
            ]
        )

        belongs = False

        while current != 0:

            if current == root_body_id:
                belongs = True
                break

            current = int(
                sim.model.body_parentid[
                    current
                ]
            )

        if not belongs:
            continue

        pos = np.asarray(
            sim.data.geom_xpos[
                geom_id
            ],
            dtype=float,
        )

        mat = np.asarray(
            sim.data.geom_xmat[
                geom_id
            ],
            dtype=float,
        ).reshape(
            3,
            3,
        )

        geom_type = int(
            sim.model.geom_type[
                geom_id
            ]
        )

        # ----------------------------------------------------
        # Box geom
        # ----------------------------------------------------

        if (
            geom_type
            == mujoco.mjtGeom.mjGEOM_BOX
        ):

            half = np.asarray(
                sim.model.geom_size[
                    geom_id,
                    :3,
                ],
                dtype=float,
            )

            local = np.asarray(
                [
                    [x, y, z]
                    for x in (
                        -half[0],
                        half[0],
                    )
                    for y in (
                        -half[1],
                        half[1],
                    )
                    for z in (
                        -half[2],
                        half[2],
                    )
                ]
            )

            points.append(
                local @ mat.T
                + pos
            )

        # ----------------------------------------------------
        # Mesh geom
        # ----------------------------------------------------

        elif (
            geom_type
            == mujoco.mjtGeom.mjGEOM_MESH
        ):

            mesh_id = int(
                sim.model.geom_dataid[
                    geom_id
                ]
            )

            if mesh_id < 0:
                continue

            start = int(
                sim.model.mesh_vertadr[
                    mesh_id
                ]
            )

            count = int(
                sim.model.mesh_vertnum[
                    mesh_id
                ]
            )

            vertices = np.asarray(
                sim.model.mesh_vert[
                    start:start + count
                ],
                dtype=float,
            )

            if len(vertices):

                points.append(
                    vertices @ mat.T
                    + pos
                )

    if not points:
        return None

    vertices = np.concatenate(
        points,
        axis=0,
    )

    minimum = np.min(
        vertices,
        axis=0,
    )

    maximum = np.max(
        vertices,
        axis=0,
    )

    return {
        "min": minimum,
        "max": maximum,
        "size": maximum - minimum,
    }


# ============================================================
# Report
# ============================================================

def print_report(env):

    from environments.c4_2_diagonal_fit_packing import (
        BOX_DIAGONAL,
        BOX_INNER_D,
        BOX_INNER_H,
        BOX_INNER_W,
        OBJECT_LENGTHS,
    )

    axis_limit = max(
        BOX_INNER_W,
        BOX_INNER_D,
        BOX_INNER_H,
    )

    # ========================================================
    # Box geometry
    # ========================================================

    print()
    print("=" * 78)
    print(
        "C4-T2 DIAGONAL-FIT PACKING"
    )
    print("=" * 78)

    print()
    print("-" * 78)
    print("BOX GEOMETRY")
    print("-" * 78)

    print(
        f"Inner width      : "
        f"{BOX_INNER_W:.3f} m "
        f"({BOX_INNER_W * 1000:.1f} mm)"
    )

    print(
        f"Inner depth      : "
        f"{BOX_INNER_D:.3f} m "
        f"({BOX_INNER_D * 1000:.1f} mm)"
    )

    print(
        f"Inner height     : "
        f"{BOX_INNER_H:.3f} m "
        f"({BOX_INNER_H * 1000:.1f} mm)"
    )

    print(
        f"Max axis length  : "
        f"{axis_limit:.3f} m "
        f"({axis_limit * 1000:.1f} mm)"
    )

    print(
        f"3D diagonal      : "
        f"{BOX_DIAGONAL:.6f} m "
        f"({BOX_DIAGONAL * 1000:.1f} mm)"
    )

    # ========================================================
    # Physical specs
    # ========================================================

    print()
    print("-" * 78)
    print(
        "OBJECT PHYSICAL SPECS"
    )
    print("-" * 78)

    for name in OBJECT_NAMES:

        if name not in env.objects:
            continue

        obj = env.objects[
            name
        ]

        try:

            body_id = (
                env.sim.model.body_name2id(
                    obj.root_body
                )
            )

        except Exception as exc:

            print()
            print(
                f"{name}: "
                f"could not find body "
                f"({exc})"
            )

            continue

        mass = (
            get_total_body_mass(
                env.sim,
                body_id,
            )
        )

        aabb = (
            get_object_world_aabb(
                env.sim,
                body_id,
            )
        )

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
            f"{mass:.6f} kg "
            f"({mass * 1000:.2f} g)"
        )

        if aabb is not None:

            mm = (
                aabb["size"]
                * 1000.0
            )

            print(
                "  AABB  : "
                f"{mm[0]:.2f} × "
                f"{mm[1]:.2f} × "
                f"{mm[2]:.2f} mm"
            )

            print(
                "          X × Y × Z "
                "(world AABB)"
            )

        else:

            print(
                "  AABB  : "
                "could not calculate"
            )

    # ========================================================
    # Geometric fit
    # ========================================================

    print()
    print("-" * 78)
    print(
        "GEOMETRIC FIT"
    )
    print("-" * 78)

    for name, length in (
        OBJECT_LENGTHS.items()
    ):

        axis_fit = (
            length
            <= axis_limit
        )

        diagonal_fit = (
            length
            <= BOX_DIAGONAL
        )

        print(
            f"{name:14s} "
            f"length="
            f"{length * 1000:6.1f} mm  "
            f"axis_fit="
            f"{str(axis_fit):5s}  "
            f"diagonal_bound="
            f"{diagonal_fit}"
        )

    # ========================================================
    # Robot base
    # ========================================================

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

    targets = {}

    # ========================================================
    # Object poses
    # ========================================================

    print()
    print("-" * 78)
    print(
        "OBJECT POSES"
    )
    print("-" * 78)

    print()
    print(
        "robot0_base"
    )

    print(
        f"  position : "
        f"{_fmt_vec(base_pos)}"
    )

    for name in (
        *OBJECT_NAMES,
        "lid",
    ):

        if name not in env.objects:
            continue

        obj = (
            env.objects[
                name
            ]
        )

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

        quat = np.asarray(
            env.sim.data.body_xquat[
                body_id
            ],
            dtype=float,
        )

        targets[
            name
        ] = pos

        print()
        print(
            name
        )

        print(
            f"  position : "
            f"{_fmt_vec(pos)}"
        )

        print(
            f"  quat_wxyz: "
            f"{_fmt_vec(quat)}"
        )

    # ========================================================
    # Packing box
    # ========================================================

    try:

        box_body_id = (
            env.sim.model.body_name2id(
                "packing_box"
            )
        )

        box_body_pos = np.asarray(
            env.sim.data.body_xpos[
                box_body_id
            ],
            dtype=float,
        )

        box_center = np.asarray(
            env._packing_box_center,
            dtype=float,
        )

        targets[
            "packing_box"
        ] = box_center

        print()
        print(
            "packing_box"
        )

        print(
            f"  body position : "
            f"{_fmt_vec(box_body_pos)}"
        )

        print(
            f"  inner center  : "
            f"{_fmt_vec(box_center)}"
        )

    except Exception as exc:

        print()
        print(
            "packing_box: "
            f"could not get position "
            f"({exc})"
        )

    # ========================================================
    # EE rack
    # ========================================================

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

        targets[
            label
        ] = pos

        print(
            f"{label:8s} "
            f"{_fmt_vec(pos)}"
        )

    # ========================================================
    # Reachability
    # ========================================================

    print()
    print("-" * 78)
    print(
        "REACHABILITY"
    )
    print("-" * 78)

    print(
        f"UR5e nominal XY reach = "
        f"{UR5E_REACH_M:.2f} m"
    )

    print()

    for name, pos in (
        targets.items()
    ):

        delta = (
            pos
            - base_pos
        )

        dist_xy = float(
            np.linalg.norm(
                delta[:2]
            )
        )

        dist_3d = float(
            np.linalg.norm(
                delta
            )
        )

        margin = (
            UR5E_REACH_M
            - dist_xy
        )

        flag = (
            "OK"
            if dist_xy
            <= UR5E_REACH_M
            else "OUT"
        )

        print(
            f"{name:14s} "
            f"xy={dist_xy:.3f} m  "
            f"3d={dist_3d:.3f} m  "
            f"margin={margin:+.3f} m  "
            f"[{flag}]"
        )

    print()
    print("=" * 78)
    print()


# ============================================================
# Free camera
# ============================================================

def _apply_camera(env) -> None:
    """
    C4-T1 viewer와 동일하게
    free MuJoCo viewer camera를 사용한다.

    초기 camera 위치만 지정하고,
    이후에는 viewer에서 마우스로 자유롭게
    rotate / pan / zoom 할 수 있다.
    """

    viewer = (
        env.viewer.viewer
    )

    if viewer is None:
        return

    points = []

    # --------------------------------------------------------
    # Packing objects + lid
    # --------------------------------------------------------

    for name in (
        *OBJECT_NAMES,
        "lid",
    ):

        if name not in env.objects:
            continue

        obj = env.objects[
            name
        ]

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
    # Packing box
    # --------------------------------------------------------

    if hasattr(
        env,
        "_packing_box_center",
    ):

        points.append(
            np.asarray(
                env._packing_box_center,
                dtype=float,
            )
        )

    # --------------------------------------------------------
    # EE rack
    # --------------------------------------------------------

    if hasattr(
        env,
        "ee_rack_info",
    ):

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

    try:

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

    except Exception:
        pass

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

        center = np.asarray(
            [
                0.0,
                0.0,
                0.8,
            ],
            dtype=float,
        )

    # --------------------------------------------------------
    # Initial camera
    # --------------------------------------------------------

    cam = (
        viewer.cam
    )

    cam.lookat[:] = (
        center
    )

    cam.distance = 2.6

    # C4-T1처럼 사선으로 전체 scene 확인
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

        from environments.c4_2_diagonal_fit_packing import (
            C4_2_DiagonalFitPacking,  # noqa: F401
        )

    except ImportError as exc:

        print(
            "ERROR: "
            "C4_2_DiagonalFitPacking "
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
        "C4_2_DiagonalFitPacking "
        "(UR5e + pedestal, Island)..."
    )

    env = suite.make(
        env_name=(
            "C4_2_DiagonalFitPacking"
        ),
        robots="UR5e",

        has_renderer=True,

        has_offscreen_renderer=False,

        use_camera_obs=False,

        # 중요:
        # 특정 robot camera가 아니라
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
    print("\n[PACKING BOX DEBUG]")

    try:
        box_id = env.sim.model.body_name2id("packing_box")

        print("packing_box exists: True")
        print("body id:", box_id)
        print(
            "body position:",
            env.sim.data.body_xpos[box_id]
        )

        print("\npacking box geoms:")

        for geom_id in range(env.sim.model.ngeom):

            body_id = int(
                env.sim.model.geom_bodyid[geom_id]
            )

            if body_id != box_id:
                continue

            geom_name = env.sim.model.geom_id2name(
                geom_id
            )

            print(
                geom_name,
                "pos=",
                env.sim.data.geom_xpos[geom_id],
                "size=",
                env.sim.model.geom_size[geom_id],
                "rgba=",
                env.sim.model.geom_rgba[geom_id],
            )

    except Exception as e:
        print("packing_box exists: False")
        print("error:", e)
    try:
        box_id = env.sim.model.body_name2id("packing_box")
        print("[PACKING BOX CHECK]")
        print("packing_box exists: True")
        print("body id:", box_id)
        print("position:", env.sim.data.body_xpos[box_id])
    except Exception:
        print("[PACKING BOX CHECK]")
        print("packing_box exists: False")

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_report(
        env
    )

    # --------------------------------------------------------
    # Open viewer
    # --------------------------------------------------------

    print(
        "Scene ready. "
        "Opening free MuJoCo viewer..."
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
        "  - Packing box"
    )

    print(
        "  - Lid beside box"
    )

    print(
        "  - Rolling pin / Baguette / "
        "Whisk / Cereal / Milk"
    )

    print()
    print(
        "You can freely rotate / pan / "
        "zoom the MuJoCo viewer."
    )

    print(
        "Close the viewer window "
        "(or Ctrl+C) to exit."
    )

    env.viewer.update()

    viewer = (
        env.viewer.viewer
    )

    if viewer is None:

        print(
            "ERROR: "
            "Failed to launch viewer.",
            file=sys.stderr,
        )

        env.close()

        return 1

    # --------------------------------------------------------
    # Initial free camera
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
            viewer.is_running()
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
        "Viewer closed. Exiting."
    )

    return 0


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
