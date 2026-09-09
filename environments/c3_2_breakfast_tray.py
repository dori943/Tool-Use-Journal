"""C3-T2 Breakfast Tray Preparation: deterministic, scene-only Kitchen Island.

Import this module before robosuite.make("C3_2_BreakfastTrayPreparation").
All 14 objects have free joints. No EE labels, ordering, or success evaluator.

Only the two C3-T2 trays are enlarged through TrayObject(scale=...).
Original tray asset / TrayObject implementation is not modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from robocasa.models.fixtures import FixtureType
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.objects import MujocoXMLObject

from environments.ee_rack import add_ee_rack
from environments.kitchen_base import KitchenBase
from environments.task_camera import add_standard_task_camera
from environments.objects import (
    AppleObject,
    BreadObject,
    PlateObject,
    TrayObject,
)
from environments.objects.xml_asset import (
    OBJECTS_ASSET_DIR,
    make_resolved_object_xml,
)
from environments.robot_pedestal import (
    add_robot_pedestal,
    remove_robot_pedestal,
)


ROBOT_SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "robot_spec.json"
)


ROBOT_BASE_X = -0.68

PEDESTAL_TOP_Z = 0.8

PEDESTAL_HALF_XY = (
    0.25,
    0.25,
)


EE_RACK_LAYOUT = {
    "3F": (-0.330000, -0.606218),
    "vac": (-0.185025, -0.494975),
    "2F": (-0.073782, -0.350000),
}


INSTRUCTION = (
    "두 사람을 위한 아침 식사 트레이를 준비하라. "
    "각 트레이에 접시, 머그컵, 포크, 숟가락을 담고, 빵과 과일은 접시 위에 올려라."
)


# ============================================================
# C3-T2 ONLY object scales
# ============================================================
# 각 값은 C3-T2 환경에서만 적용된다.
# 원본 object class / XML asset은 수정하지 않는다.
#
# 원하는 크기는 여기 숫자만 조절하면 된다.
#
TRAY_SCALE = 0.37
PLATE_SCALE = 0.16
MUG_SCALE = 1.00
FORK_SCALE = 0.7
SPOON_SCALE = 0.8
BREAD_SCALE = 0.1
FRUIT_SCALE = 0.08


# ============================================================
# Initial deterministic layout
# ============================================================
#
# x : Island near edge 기준
# y : Island center 기준
# yaw : radians
#
# 동일 category 객체끼리 나란히 배치.
#
INITIAL_LAYOUT = {
    # Bread pair
    "bread_a": (
        0.26,
        0.13,
        0.0,
    ),
    "bread_b": (
        0.26,
        0.0,
        0.0,
    ),

    # Mug pair
    "mug_a": (
        0.26,
        0.47,
        0.0,
    ),
    "mug_b": (
        0.26,
        0.31,
        0.0,
    ),

    # Tray pair
    #
    # Tray가 커졌으므로 서로 약간 여유를 두고 배치
    "tray_a": (
        0.08,
        -0.08,
        np.pi,
    ),
    "tray_b": (
        0.08,
        0.32,
        np.pi,
    ),

    # Plate pair
    "plate_a": (
        0.46,
        0.0,
        0.0,
    ),
    "plate_b": (
        0.46,
        0.19,
        0.0,
    ),

    # Fruit / Apple pair
    "fruit_a": (
        0.39,
        0.42,
        0.0,
    ),
    "fruit_b": (
        0.39,
        0.31,
        0.0,
    ),

    # Fork pair
    "fork_a": (
        0.295,
        -0.13,
        -np.pi / 2,
    ),
    "fork_b": (
        0.295,
        -0.21,
        -np.pi / 2,
    ),

    # Spoon pair
    "spoon_a": (
        0.47,
        -0.13,
        -np.pi / 2,
    ),
    "spoon_b": (
        0.47,
        -0.21,
        -np.pi / 2,
    ),
}


def subtree_body_ids(model, root_id):
    """Return every MuJoCo body id contained under root_id."""
    ids = {int(root_id)}

    for bid in range(int(root_id) + 1, model.nbody):
        if int(model.body_parentid[bid]) in ids:
            ids.add(bid)

    return ids


def body_aabb(sim, root_id, collision_only=False):
    """Actual compiled geometry AABB, excluding nonphysical region markers."""
    model = sim.model
    data = sim.data

    ids = subtree_body_ids(model, root_id)

    lows = []
    highs = []

    for gid in range(model.ngeom):

        if int(model.geom_bodyid[gid]) not in ids:
            continue

        if int(model.geom_group[gid]) not in (0, 1):
            continue

        if (
            collision_only
            and not (
                model.geom_contype[gid]
                or model.geom_conaffinity[gid]
            )
        ):
            continue

        rotation = np.asarray(
            data.geom_xmat[gid]
        ).reshape(3, 3)

        center = np.asarray(
            data.geom_xpos[gid]
        )

        kind = int(
            model.geom_type[gid]
        )

        if kind == int(
            mujoco.mjtGeom.mjGEOM_MESH
        ):
            mid = int(
                model.geom_dataid[gid]
            )

            start = int(
                model.mesh_vertadr[mid]
            )

            points = np.asarray(
                model.mesh_vert[
                    start:
                    start + int(model.mesh_vertnum[mid])
                ]
            )

            points = (
                points @ rotation.T
                + center
            )

            lo = points.min(axis=0)
            hi = points.max(axis=0)

        else:
            size = np.asarray(
                model.geom_size[gid]
            ).copy()

            if kind == int(
                mujoco.mjtGeom.mjGEOM_SPHERE
            ):
                extent = np.full(
                    3,
                    size[0],
                )

            elif kind in (
                int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            ):
                axis = rotation[:, 2]

                extent = (
                    size[0]
                    * np.sqrt(
                        np.maximum(
                            0,
                            1 - axis**2,
                        )
                    )
                    + size[1]
                    * np.abs(axis)
                )

                if kind == int(
                    mujoco.mjtGeom.mjGEOM_CAPSULE
                ):
                    extent = (
                        size[0]
                        + size[1]
                        * np.abs(axis)
                    )

            elif kind == int(
                mujoco.mjtGeom.mjGEOM_BOX
            ):
                extent = (
                    np.abs(rotation)
                    @ size
                )

            else:
                raise ValueError(
                    f"Unsupported geom type {kind} for AABB"
                )

            lo = center - extent
            hi = center + extent

        lows.append(lo)
        highs.append(hi)

    if not lows:
        raise ValueError(
            f"No physical geometry for body {root_id}"
        )

    return (
        np.min(lows, axis=0),
        np.max(highs, axis=0),
    )


class C3_2_BreakfastTrayPreparation(KitchenBase):
    """Independent KitchenBase environment."""

    def __init__(
        self,
        robots="UR5e",
        gripper_types=None,
        base_types="NullMount",
        initialization_noise=None,
        seed=0,
        **kwargs,
    ):

        kwargs.setdefault(
            "use_distractors",
            False,
        )

        kwargs.setdefault(
            "use_object_obs",
            True,
        )

        for key in (
            "robot_spawn_deviation_pos_x",
            "robot_spawn_deviation_pos_y",
            "robot_spawn_deviation_rot",
        ):
            kwargs.setdefault(
                key,
                0.0,
            )

        with ROBOT_SPEC_PATH.open(
            encoding="utf-8"
        ) as stream:
            self.robot_spec = json.load(
                stream
            )

        self.ee_rack_info = {}
        self.pedestal_info = {}

        super().__init__(
            robots=robots,
            gripper_types=gripper_types,
            base_types=base_types,
            initialization_noise=initialization_noise,
            seed=seed,
            **kwargs,
        )


    def _setup_kitchen_references(self):

        super()._setup_kitchen_references()

        self.island = (
            self.register_fixture_ref(
                "island",
                dict(
                    id=FixtureType.ISLAND,
                    size=(0.6, 0.6),
                    full_depth_region=True,
                ),
            )
        )

        self.counter = self.island
        self.init_robot_base_ref = None


    def _get_obj_cfgs(self):

        return [
            dict(
                name=name,
                placement=dict(
                    fixture=self.island,
                    size=(0.0, 0.0),
                    pos=(0.0, 0.0),
                    rotation=0.0,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    sample_region_kwargs=dict(
                        full_depth_region=True
                    ),
                ),
            )
            for name in INITIAL_LAYOUT
        ]


    def _create_objects(self):

        self.objects = {}

        self.object_cfgs = (
            self._get_obj_cfgs()
        )

        tray_dir = (
            OBJECTS_ASSET_DIR
            / "tray"
        )

        # All scale controls are local to C3-T2.
        builders = {
            "tray": lambda name: TrayObject(
                name=name,
                scale=TRAY_SCALE,
                joints="default",
            ),
            "plate": lambda name: PlateObject(
                name=name,
                scale=PLATE_SCALE,
            ),
            "bread": lambda name: BreadObject(
                name=name,
                scale=BREAD_SCALE,
            ),
            "fruit": lambda name: AppleObject(
                name=name,
                scale=FRUIT_SCALE,
            ),
        }

        # mug / fork / spoon are loaded directly instead of using wrappers
        # so intended_ee metadata is not exposed. Their scale is still
        # controlled locally in this C3-T2 environment.
        direct_asset_scales = {
            "mug": MUG_SCALE,
            "fork": FORK_SCALE,
            "spoon": SPOON_SCALE,
        }

        self.asset_paths = {}

        for cfg in self.object_cfgs:

            name = cfg["name"]

            category = (
                name.rsplit("_", 1)[0]
            )

            if category in builders:

                obj = builders[
                    category
                ](
                    name=name
                )

                asset_dir = (
                    tray_dir
                    if category == "tray"
                    else OBJECTS_ASSET_DIR
                    / (
                        "apple"
                        if category == "fruit"
                        else category
                    )
                )

            else:

                asset_dir = (
                    OBJECTS_ASSET_DIR
                    / (
                        "mug_3143a4ac"
                        if category == "mug"
                        else category
                    )
                )

                # Reuse original asset directly so
                # intended_ee metadata is not exposed.
                obj = MujocoXMLObject(
                    fname=make_resolved_object_xml(
                        asset_dir
                    ),
                    name=name,
                    joints="default",
                    obj_type="all",
                    duplicate_collision_geoms=False,
                    scale=direct_asset_scales[category],
                )

            self.asset_paths[name] = str(
                asset_dir
                / "model.xml"
            )

            cfg.update(
                type="object",
                info={
                    "groups_containing_sampled_obj": [
                        "all",
                        category,
                    ],
                    "groups": [
                        category
                    ],
                    "cat": category,
                    "mjcf_path": self.asset_paths[
                        name
                    ],
                },
            )

            self.objects[name] = obj

            setattr(
                self,
                name,
                obj,
            )

            self.model.merge_objects(
                [obj]
            )


    def _load_model(
        self,
        attempt_num=1,
    ):

        super()._load_model(
            attempt_num=attempt_num
        )

        region = (
            self.island.sample_reset_region(
                env=self,
                full_depth_region=True,
            )
        )

        self._island_surface_z = float(
            self.island.pos[2]
            + region["offset"][2]
        )

        points = np.asarray(
            self.island.get_bbox_points()
        )

        self._island_bounds = (
            points.min(axis=0),
            points.max(axis=0),
        )

        lo, hi = (
            self._island_bounds
        )

        self._layout_origin = np.array(
            [
                lo[0],
                (
                    lo[1]
                    + hi[1]
                )
                / 2,
            ]
        )

        work = (
            self._layout_origin
            + [0.40, 0.0]
        )

        self._robot_base_xy = (
            work
            + [ROBOT_BASE_X, 0.0]
        )

        remove_robot_pedestal(
            self.model
        )

        self.pedestal_info = (
            add_robot_pedestal(
                self.model,
                center_xy=self._robot_base_xy,
                top_z=PEDESTAL_TOP_Z,
                half_size_xy=PEDESTAL_HALF_XY,
            )
        )

        robot = (
            self.robots[0]
            .robot_model
        )

        robot.set_base_xpos(
            [
                *self._robot_base_xy,
                PEDESTAL_TOP_Z,
            ]
        )

        base_orientation = np.zeros(3)
        robot.set_base_ori(base_orientation)

        self._standard_camera = add_standard_task_camera(
            self.model,
            robot_base_xy=self._robot_base_xy,
            robot_base_yaw_rad=float(base_orientation[2]),
            surface_z=self._island_surface_z,
        )

        self._remove_existing_ee_rack()

        self.ee_rack_info = (
            add_ee_rack(
                arena=self.model,
                table_offset=np.array(
                    [
                        0.0,
                        0.0,
                        self._island_surface_z,
                    ]
                ),
                rack_layout={
                    key: tuple(
                        work + xy
                    )
                    for key, xy
                    in EE_RACK_LAYOUT.items()
                },
                ee_pool=self.robot_spec[
                    "ee_pool"
                ],
            )
        )

        self.object_placements = {}

        for (
            name,
            (
                x,
                y,
                yaw,
            ),
        ) in INITIAL_LAYOUT.items():

            obj = self.objects[
                name
            ]

            pos = (
                *(
                    self._layout_origin
                    + [x, y]
                ),
                self._island_surface_z
                - float(
                    obj.bottom_offset[2]
                )
                + 0.001,
            )

            quat = np.array(
                [
                    np.cos(
                        yaw / 2
                    ),
                    0.0,
                    0.0,
                    np.sin(
                        yaw / 2
                    ),
                ]
            )

            self.object_placements[
                name
            ] = (
                pos,
                quat,
                obj,
            )


    def _reset_internal(self):

        ManipulationEnv._reset_internal(
            self
        )

        self._setup_scene()

        self.init_robot_base_pos = np.array(
            [
                *self._robot_base_xy,
                PEDESTAL_TOP_Z,
            ]
        )

        self.init_robot_base_ori = (
            np.zeros(3)
        )

        if not self.deterministic_reset:

            for (
                pos,
                quat,
                obj,
            ) in self.object_placements.values():

                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.r_[
                        pos,
                        quat,
                    ],
                )

            self.sim.forward()

            # Align actual mesh bottoms with Island surface.
            for (
                name,
                (
                    _,
                    _,
                    obj,
                ),
            ) in self.object_placements.items():

                bid = (
                    self.sim.model
                    .body_name2id(
                        obj.root_body
                    )
                )

                lo, _ = body_aabb(
                    self.sim,
                    bid,
                )

                qpos = (
                    self.sim.data
                    .get_joint_qpos(
                        obj.joints[0]
                    )
                    .copy()
                )

                qpos[2] += (
                    self._island_surface_z
                    + 0.001
                    - lo[2]
                )

                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    qpos,
                )

                self.sim.data.set_joint_qvel(
                    obj.joints[0],
                    np.zeros(6),
                )

            self.sim.forward()


    def _remove_existing_ee_rack(
        self,
    ) -> None:

        worldbody = (
            self.model.worldbody
        )

        for body in list(
            worldbody.findall(
                "body"
            )
        ):

            name = (
                body.get("name")
                or ""
            )

            if (
                name == "ee_rack"
                or name.startswith(
                    "robot0_rack_"
                )
            ):
                worldbody.remove(
                    body
                )


    def get_ep_meta(self):

        meta = (
            super().get_ep_meta()
        )

        meta["lang"] = (
            INSTRUCTION
        )

        return meta


    def _check_success(self):

        return False


    def reward(
        self,
        action=None,
    ):

        return 0.0


    def _xy_size(
        self,
        name,
    ):

        bid = (
            self.sim.model
            .body_name2id(
                self.objects[
                    name
                ].root_body
            )
        )

        lo, hi = body_aabb(
            self.sim,
            bid,
        )

        return np.asarray(
            hi[:2]
            - lo[:2],
            dtype=float,
        )


    def packing_feasibility(self):
        """AABB footprint check for goal packing."""

        tray = self._xy_size(
            "tray_a"
        )

        plate = self._xy_size(
            "plate_a"
        )

        mug = self._xy_size(
            "mug_a"
        )

        fork = self._xy_size(
            "fork_a"
        )

        spoon = self._xy_size(
            "spoon_a"
        )

        bread = self._xy_size(
            "bread_a"
        )

        fruit = self._xy_size(
            "fruit_a"
        )

        fork_short, fork_long = (
            sorted(fork)
        )

        spoon_short, spoon_long = (
            sorted(spoon)
        )

        tray_short, tray_long = (
            sorted(tray)
        )

        plate_side = float(
            np.max(plate)
        )

        mug_side = float(
            np.max(mug)
        )

        strip = float(
            tray_short
            - plate_side
        )

        length_left = float(
            tray_long
            - plate_side
            - mug_side
        )

        tray_ok = (
            plate_side
            <= tray_long
            + 1e-6

            and plate_side
            <= tray_short
            + 1e-6

            and mug_side
            <= tray_short
            + 1e-6

            and (
                plate_side
                + mug_side
            )
            <= tray_long
            + 1e-6

            and fork_short
            <= max(
                strip,
                length_left,
            )
            + 1e-6

            and fork_long
            <= max(
                tray_long,
                tray_short,
            )
            + 1e-6

            and spoon_short
            <= max(
                strip,
                length_left,
            )
            + 1e-6

            and spoon_long
            <= max(
                tray_long,
                tray_short,
            )
            + 1e-6
        )

        bread_short, bread_long = (
            sorted(bread)
        )

        fruit_short, fruit_long = (
            sorted(fruit)
        )

        plate_ok = (
            (
                bread_short
                + fruit_short
            )
            <= float(
                np.min(plate)
            )
            + 1e-6

            and max(
                bread_long,
                fruit_long,
            )
            <= float(
                np.max(plate)
            )
            + 1e-6
        )

        return {
            "tray_scale": TRAY_SCALE,
            "plate_scale": PLATE_SCALE,
            "mug_scale": MUG_SCALE,
            "fork_scale": FORK_SCALE,
            "spoon_scale": SPOON_SCALE,
            "bread_scale": BREAD_SCALE,
            "fruit_scale": FRUIT_SCALE,

            "tray_xy_m": (
                tray.tolist()
            ),

            "plate_xy_m": (
                plate.tolist()
            ),

            "mug_xy_m": (
                mug.tolist()
            ),

            "fork_xy_m": (
                fork.tolist()
            ),

            "spoon_xy_m": (
                spoon.tolist()
            ),

            "bread_xy_m": (
                bread.tolist()
            ),

            "fruit_xy_m": (
                fruit.tolist()
            ),

            "tray_holds_plate_mug_fork_spoon": bool(
                tray_ok
            ),

            "plate_holds_bread_fruit": bool(
                plate_ok
            ),
        }


    def get_scene_report(self):
        """Geometry diagnostics only."""

        report = {}

        model = (
            self.sim.model
        )

        base = (
            self.sim.data.body_xpos[
                model.body_name2id(
                    "robot0_base"
                )
            ]
        )

        for (
            name,
            obj,
        ) in self.objects.items():

            bid = (
                model.body_name2id(
                    obj.root_body
                )
            )

            lo, hi = body_aabb(
                self.sim,
                bid,
            )

            pos = np.asarray(
                self.sim.data.body_xpos[
                    bid
                ]
            )

            ids = list(
                subtree_body_ids(
                    model,
                    bid,
                )
            )

            report[name] = dict(
                asset=self.asset_paths[
                    name
                ],

                object_class=type(
                    obj
                ).__name__,

                position_m=(
                    pos.tolist()
                ),

                quaternion_wxyz=(
                    self.sim.data
                    .body_xquat[
                        bid
                    ]
                    .tolist()
                ),

                aabb_min_m=(
                    lo.tolist()
                ),

                aabb_max_m=(
                    hi.tolist()
                ),

                size_m=(
                    hi - lo
                ).tolist(),

                mass_kg=float(
                    np.asarray(
                        model.body_mass
                    )[ids].sum()
                ),

                distance_xy_m=float(
                    np.linalg.norm(
                        pos[:2]
                        - base[:2]
                    )
                ),

                distance_3d_m=float(
                    np.linalg.norm(
                        pos
                        - base
                    )
                ),
            )

        report["_packing"] = (
            self.packing_feasibility()
        )

        return report
