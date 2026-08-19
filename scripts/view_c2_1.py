"""C2-1 오브젝트 분류 환경 인터랙티브 뷰어."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import mujoco


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



def get_total_body_mass(
    sim,
    root_body_id,
):
    """body 서브트리 질량 합 [kg]. 자식 body가 있으면 root mass만 보면 빠진다."""

    model = sim.model

    total_mass = 0.0

    for body_id in range(
        model.nbody
    ):

        current = body_id

        while current != 0:

            if current == root_body_id:

                total_mass += float(
                    model.body_mass[
                        body_id
                    ]
                )

                break

            current = int(
                model.body_parentid[
                    current
                ]
            )

    return total_mass



def get_object_geom_ids(
    sim,
    root_body_id,
):
    """오브젝트 body 서브트리에 속한 geom id 목록."""

    model = sim.model

    geom_ids = []

    for geom_id in range(
        model.ngeom
    ):

        body_id = int(
            model.geom_bodyid[
                geom_id
            ]
        )

        current = body_id

        while current != 0:

            if current == root_body_id:

                geom_ids.append(
                    geom_id
                )

                break

            current = int(
                model.body_parentid[
                    current
                ]
            )

    return geom_ids



def get_object_world_aabb(
    sim,
    root_body_id,
):
    """실제 geom으로 월드 AABB를 계산한다. sphere/box/cylinder/ellipsoid/capsule/mesh."""

    model = sim.model
    data = sim.data

    geom_ids = (
        get_object_geom_ids(
            sim,
            root_body_id,
        )
    )

    all_points = []


    for geom_id in geom_ids:

        geom_type = int(
            model.geom_type[
                geom_id
            ]
        )


        geom_pos = np.asarray(
            data.geom_xpos[
                geom_id
            ],
            dtype=float,
        )

        geom_mat = np.asarray(
            data.geom_xmat[
                geom_id
            ],
            dtype=float,
        ).reshape(
            3,
            3,
        )


        geom_size = np.asarray(
            model.geom_size[
                geom_id
            ],
            dtype=float,
        )


        if (
            geom_type
            == mujoco.mjtGeom.mjGEOM_MESH
        ):

            mesh_id = int(
                model.geom_dataid[
                    geom_id
                ]
            )

            if mesh_id < 0:
                continue


            vert_start = int(
                model.mesh_vertadr[
                    mesh_id
                ]
            )

            vert_num = int(
                model.mesh_vertnum[
                    mesh_id
                ]
            )


            vertices = np.asarray(
                model.mesh_vert[
                    vert_start:
                    vert_start + vert_num
                ],
                dtype=float,
            )


            if len(vertices) == 0:
                continue


            world_vertices = (
                vertices
                @ geom_mat.T
                + geom_pos
            )


            all_points.append(
                world_vertices
            )


        elif (
            geom_type
            == mujoco.mjtGeom.mjGEOM_SPHERE
        ):

            r = float(
                geom_size[
                    0
                ]
            )


            local_points = np.array(
                [
                    [-r, -r, -r],
                    [-r, -r,  r],
                    [-r,  r, -r],
                    [-r,  r,  r],
                    [ r, -r, -r],
                    [ r, -r,  r],
                    [ r,  r, -r],
                    [ r,  r,  r],
                ],
                dtype=float,
            )


            world_points = (
                local_points
                @ geom_mat.T
                + geom_pos
            )


            all_points.append(
                world_points
            )


        elif (
            geom_type
            == mujoco.mjtGeom.mjGEOM_BOX
        ):

            hx = float(
                geom_size[0]
            )

            hy = float(
                geom_size[1]
            )

            hz = float(
                geom_size[2]
            )


            local_points = np.array(
                [
                    [-hx, -hy, -hz],
                    [-hx, -hy,  hz],
                    [-hx,  hy, -hz],
                    [-hx,  hy,  hz],
                    [ hx, -hy, -hz],
                    [ hx, -hy,  hz],
                    [ hx,  hy, -hz],
                    [ hx,  hy,  hz],
                ],
                dtype=float,
            )


            world_points = (
                local_points
                @ geom_mat.T
                + geom_pos
            )


            all_points.append(
                world_points
            )


        elif (
            geom_type
            == mujoco.mjtGeom.mjGEOM_CYLINDER
        ):

            radius = float(
                geom_size[
                    0
                ]
            )

            half_height = float(
                geom_size[
                    1
                ]
            )


            angles = np.linspace(
                0.0,
                2.0 * np.pi,
                128,
                endpoint=False,
            )


            points = []


            for z in (
                -half_height,
                half_height,
            ):

                for angle in angles:

                    points.append(
                        [
                            radius
                            * np.cos(
                                angle
                            ),

                            radius
                            * np.sin(
                                angle
                            ),

                            z,
                        ]
                    )


            local_points = np.asarray(
                points,
                dtype=float,
            )


            world_points = (
                local_points
                @ geom_mat.T
                + geom_pos
            )


            all_points.append(
                world_points
            )


        elif (
            geom_type
            == mujoco.mjtGeom.mjGEOM_ELLIPSOID
        ):

            rx = float(
                geom_size[0]
            )

            ry = float(
                geom_size[1]
            )

            rz = float(
                geom_size[2]
            )


            theta_values = np.linspace(
                0.0,
                2.0 * np.pi,
                64,
                endpoint=False,
            )

            phi_values = np.linspace(
                -np.pi / 2.0,
                np.pi / 2.0,
                32,
            )


            points = []


            for phi in phi_values:

                for theta in theta_values:

                    x = (
                        rx
                        * np.cos(phi)
                        * np.cos(theta)
                    )

                    y = (
                        ry
                        * np.cos(phi)
                        * np.sin(theta)
                    )

                    z = (
                        rz
                        * np.sin(phi)
                    )


                    points.append(
                        [
                            x,
                            y,
                            z,
                        ]
                    )


            local_points = np.asarray(
                points,
                dtype=float,
            )


            world_points = (
                local_points
                @ geom_mat.T
                + geom_pos
            )


            all_points.append(
                world_points
            )


        elif (
            geom_type
            == mujoco.mjtGeom.mjGEOM_CAPSULE
        ):

            radius = float(
                geom_size[
                    0
                ]
            )

            half_length = float(
                geom_size[
                    1
                ]
            )


            extent = np.array(
                [
                    radius,
                    radius,
                    half_length
                    + radius,
                ],
                dtype=float,
            )


            local_points = np.array(
                [
                    [
                        -extent[0],
                        -extent[1],
                        -extent[2],
                    ],

                    [
                        -extent[0],
                        -extent[1],
                        extent[2],
                    ],

                    [
                        -extent[0],
                        extent[1],
                        -extent[2],
                    ],

                    [
                        -extent[0],
                        extent[1],
                        extent[2],
                    ],

                    [
                        extent[0],
                        -extent[1],
                        -extent[2],
                    ],

                    [
                        extent[0],
                        -extent[1],
                        extent[2],
                    ],

                    [
                        extent[0],
                        extent[1],
                        -extent[2],
                    ],

                    [
                        extent[0],
                        extent[1],
                        extent[2],
                    ],
                ],
                dtype=float,
            )


            world_points = (
                local_points
                @ geom_mat.T
                + geom_pos
            )


            all_points.append(
                world_points
            )


    if not all_points:

        return None


    points = np.concatenate(
        all_points,
        axis=0,
    )


    min_xyz = np.min(
        points,
        axis=0,
    )

    max_xyz = np.max(
        points,
        axis=0,
    )

    size_xyz = (
        max_xyz
        - min_xyz
    )


    return {
        "min_xyz": min_xyz,
        "max_xyz": max_xyz,
        "size_xyz_m": size_xyz,
    }



def print_object_physical_specs(
    env,
):
    """대상 물체의 실제 질량(서브트리)과 크기(월드 AABB)를 출력한다."""

    print()

    print(
        "=" * 72
    )

    print(
        "C2-1 OBJECT PHYSICAL SPECS"
    )

    print(
        "=" * 72
    )


    for obj in (
        env.target_objects
    ):

        root_body_id = (
            env.sim.model.body_name2id(
                obj.root_body
            )
        )


        mass_kg = (
            get_total_body_mass(
                env.sim,
                root_body_id,
            )
        )


        mass_g = (
            mass_kg
            * 1000.0
        )


        aabb = (
            get_object_world_aabb(
                env.sim,
                root_body_id,
            )
        )


        print()

        print(
            f"[OBJECT] {obj.name}"
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
                aabb[
                    "size_xyz_m"
                ]
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
                "  size  : "
                "could not calculate"
            )


    print()

    print(
        "=" * 72
    )

    print()

    print("\n" + "=" * 72)
    print("C2-1 TRAY PHYSICAL SPECS")
    print("=" * 72)

    for tray in env.trays:
        body_id = env.obj_body_id[tray.name]

        # 이 tray body에 속한 geom만 모은다.
        geom_ids = [
            geom_id
            for geom_id in range(env.sim.model.ngeom)
            if env.sim.model.geom_bodyid[geom_id] == body_id
        ]

        if not geom_ids:
            print(f"\n[TRAY] {tray.name}")
            print("  size : could not determine")
            continue

        min_xyz = np.array([np.inf, np.inf, np.inf])
        max_xyz = np.array([-np.inf, -np.inf, -np.inf])

        for geom_id in geom_ids:
            geom_pos = env.sim.data.geom_xpos[geom_id]
            geom_mat = env.sim.data.geom_xmat[geom_id].reshape(3, 3)

            geom_type = env.sim.model.geom_type[geom_id]
            geom_size = env.sim.model.geom_size[geom_id]

            # MuJoCo box (type 6)
            if geom_type == 6:
                half_size = geom_size[:3]

                # 회전을 반영한 월드 AABB half extent
                world_half = np.abs(geom_mat) @ half_size

                geom_min = geom_pos - world_half
                geom_max = geom_pos + world_half

                min_xyz = np.minimum(min_xyz, geom_min)
                max_xyz = np.maximum(max_xyz, geom_max)

            # mesh는 이 방식으로 정확한 크기를 구할 수 없다.

        if np.any(np.isinf(min_xyz)):
            print(f"\n[TRAY] {tray.name}")
            print("  size : mesh geometry - additional calculation required")
            continue

        size_m = max_xyz - min_xyz
        size_mm = size_m * 1000.0

        print(f"\n[TRAY] {tray.name}")
        print(f"  class : {tray.__class__.__name__}")
        print(
            f"  size  : "
            f"{size_mm[0]:.2f} × "
            f"{size_mm[1]:.2f} × "
            f"{size_mm[2]:.2f} mm"
        )
        print("          X × Y × Z")

    print("\n" + "=" * 72)

def main() -> int:

    # 커스텀 환경을 먼저 import해야 C2_1_ObjectSorting이 등록된다.
    import environments  # noqa: F401

    import robosuite as suite


    print(
        "Loading C2_1_ObjectSorting (UR5e)..."
    )


    env = suite.make(
        env_name="C2_1_ObjectSorting",

        robots="UR5e",

        has_renderer=True,

        has_offscreen_renderer=False,

        use_camera_obs=False,

        ignore_done=True,
    )


    env.reset()


    print_object_physical_specs(
        env
    )


    print(
        "Scene ready. Opening MuJoCo viewer..."
    )


    print(
        "Inspect:"
    )

    print(
        "  - UR5e"
    )

    print(
        "  - 3F / Vacuum / 2F EE rack"
    )

    print(
        "  - Green / Blue / Red trays"
    )

    print(
        "  - Apple / Bread / Mug / Plate"
    )

    print(
        "Close the viewer window "
        "or press Ctrl+C to exit."
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


    cam = (
        mujoco_viewer.cam
    )

    cam.lookat[:] = [
        -0.05,
        0.0,
        0.85,
    ]

    cam.distance = (
        2.0
    )

    cam.azimuth = (
        180.0
    )

    cam.elevation = (
        -30.0
    )

    env.viewer.update()


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
        "Viewer closed. Exiting."
    )


    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )