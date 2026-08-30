"""C2-T2 Sandwich Assembly scene.

Kitchen Island layout + UR5e / EE rack / pedestal are identical to C1-T2.

Target layout (top view):

[Ham] [Cheese] [Tomato]

[Bread] [Knife] [Spatula] [Spoon] [Ladle] [Serving Plate]

Bread A / B are placed on Bread Plate.
Turkey, cheese, tomato are placed on their own plates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import robosuite.utils.transform_utils as T
from robocasa.models.fixtures import FixtureType
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

from environments.c1_1_lego_sweep import (
    EE_RACK_LAYOUT,
    ROBOT_BASE_X,
)
from environments.c1_2_dough_flatten import (
    EE_DIST_TOL,
    EE_RACK_RADIUS,
    PEDESTAL_HALF_XY,
    PEDESTAL_TOP_Z,
    UR5E_REACH_M,
)
from environments.ee_rack import add_ee_rack
from environments.kitchen_base import KitchenBase
from environments.objects import (
    CheeseObject,
    KnifeObject,
    LadleObject,
    PlateObject,
    SandwichBreadAObject,
    SandwichBreadBObject,
    SpatulaObject,
    SpoonObject,
    TomatoSliceObject,
    TurkeySliceObject,
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


# ============================================================
# Layout parameters
# ============================================================

# 재료끼리 거의 붙도록 하는 수직 간격
_STACK_CLEARANCE = 0.001  # 1 mm
_SLICE_UNDERSIDE_CLEARANCE = 0.00  # 3.5 mm
_PLATE_FOOD_SURFACE_OFFSET = -0.01

_SPATULA_THICKNESS = 0.0018
_HAM_THICKNESS = 0.005
_CHEESE_THICKNESS = 0.004


# ------------------------------------------------------------
# Object layout
#
# C1-T2와 동일:
# +x = robot이 바라보는 방향
# +y = robot 기준 왼쪽
#
# 목표:
#
# TOP:
# Ham | Cheese | Tomato
#
# BOTTOM:
# Bread | Knife | Spatula | Spoon | Ladle | Serving
# ------------------------------------------------------------

_XY_OFFSETS = {
    "turkey_plate": (0.0, 0.32),
    "cheese_plate": (0.0, 0.10),
    "tomato_plate": (0.0, -0.12),

    "bread_plate": (-0.20, 0.32),
    "serving_plate": (-0.20, 0.54),
}


# ------------------------------------------------------------
# Tool row
# Bread Plate와 Serving Plate 사이
# ------------------------------------------------------------

_TOOL_ORDER = (
    "knife",
    "spatula",
    "spoon",
    "ladle",
)

# ingredient row보다 robot 쪽
_TOOL_ROW_FORWARD = -0.23

_TOOL_SPACING = 0.105

# Bread → knife → spatula → spoon → ladle → serving plate
_TOOL_Y_FROM_CENTER = (
    1.35,
    0.45,
    -0.45,
    -1.35,
)

# 도구를 이미지처럼 세로 방향으로 정렬
_UTENSIL_YAW = -np.pi / 2

_TOOL_ROTATIONS = {
    "knife": _UTENSIL_YAW,
    "spatula": _UTENSIL_YAW,
    "spoon": _UTENSIL_YAW,
    "ladle": _UTENSIL_YAW,
}


class C2_2_SandwichAssembly(KitchenBase):
    """Sandwich assembly on Kitchen Island.

    Robot / pedestal / EE rack layout matches C1-T2.
    """

    def __init__(
        self,
        robots="UR5e",
        gripper_types=None,
        base_types="NullMount",
        initialization_noise="default",
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

        kwargs.setdefault(
            "robot_spawn_deviation_pos_x",
            0.0,
        )
        kwargs.setdefault(
            "robot_spawn_deviation_pos_y",
            0.0,
        )
        kwargs.setdefault(
            "robot_spawn_deviation_rot",
            0.0,
        )

        with ROBOT_SPEC_PATH.open(
            encoding="utf-8"
        ) as spec_file:
            self.robot_spec = json.load(
                spec_file
            )

        self.ee_catalog = {
            entry["ee_id"]: dict(entry)
            for entry in self.robot_spec["ee_pool"]
        }

        self.ee_rack_info = {}
        self.pedestal_info = {}

        self._work_origin_xy = None
        self._robot_base_xy = None
        self._robot_base_yaw = 0.0

        self._island_surface_z = None
        self._island_bounds = None

        self._layout_meta = {}
        self._cheese_asset_note = None

        super().__init__(
            robots=robots,
            gripper_types=gripper_types,
            base_types=base_types,
            initialization_noise=initialization_noise,
            seed=seed,
            **kwargs,
        )

    # ========================================================
    # Kitchen
    # ========================================================

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.island = self.register_fixture_ref(
            "island",
            dict(
                id=FixtureType.ISLAND,
                size=(0.6, 0.6),
                full_depth_region=True,
            ),
        )

        self.counter = self.island
        self.init_robot_base_ref = None

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["lang"] = (
            "C2-T2 sandwich assembly: Island, "
            "UR5e on pedestal (C1-T2 layout), "
            "ingredient plates, bread plate, "
            "tools, serving plate."
        )

        if self._cheese_asset_note:
            ep_meta["cheese_asset_note"] = (
                self._cheese_asset_note
            )

        return ep_meta

    # ========================================================
    # Island geometry
    # ========================================================

    def _island_surface_and_bounds(self):
        island = self.island

        region = island.sample_reset_region(
            env=self,
            full_depth_region=True,
        )

        surface_z = float(
            island.pos[2]
            + region["offset"][2]
        )

        pts = np.asarray(
            island.get_bbox_points(),
            dtype=float,
        )

        bounds = {
            "xmin": float(
                pts[:, 0].min()
            ),
            "xmax": float(
                pts[:, 0].max()
            ),
            "ymin": float(
                pts[:, 1].min()
            ),
            "ymax": float(
                pts[:, 1].max()
            ),
            "zmin": float(
                pts[:, 2].min()
            ),
            "zmax": float(
                pts[:, 2].max()
            ),
            "center_xy": np.array(
                [
                    0.5
                    * (
                        pts[:, 0].min()
                        + pts[:, 0].max()
                    ),
                    0.5
                    * (
                        pts[:, 1].min()
                        + pts[:, 1].max()
                    ),
                ],
                dtype=float,
            ),
        }

        return (
            surface_z,
            bounds,
        )

    # ========================================================
    # Model loading
    # ========================================================

    def _load_model(
        self,
        attempt_num=1,
    ):
        super()._load_model(
            attempt_num=attempt_num
        )

        self._apply_island_layout()

    # ========================================================
    # Objects
    # ========================================================

    def _create_objects(self):
        self.objects = {}

        self.object_cfgs = (
            self._get_obj_cfgs()
        )

        builders = {
            # Bread
            "bread_a":
                SandwichBreadAObject,

            "bread_b":
                SandwichBreadBObject,

            "bread_plate":
                PlateObject,

            # Turkey
            "turkey_plate":
                PlateObject,

            "turkey_1":
                TurkeySliceObject,

            "turkey_2":
                TurkeySliceObject,

            "turkey_3":
                TurkeySliceObject,

            # Cheese
            "cheese_plate":
                PlateObject,

            "cheese_1":
                CheeseObject,

            "cheese_2":
                CheeseObject,

            "cheese_3":
                CheeseObject,

            # Tomato
            "tomato_plate":
                PlateObject,

            "tomato_slice":
                TomatoSliceObject,

            # Final plate
            "serving_plate":
                PlateObject,

            # Tools
            "knife":
                KnifeObject,

            "spatula":
                SpatulaObject,

            "spoon":
                SpoonObject,

            "ladle":
                LadleObject,
        }

        for cfg in self.object_cfgs:
            cfg["type"] = "object"

            name = cfg["name"]

            model = builders[name](
                name=name
            )

            target_thickness = {
                "spatula": _SPATULA_THICKNESS,
                "turkey_1": _HAM_THICKNESS,
                "turkey_2": _HAM_THICKNESS,
                "turkey_3": _HAM_THICKNESS,
                "cheese_1": _CHEESE_THICKNESS,
                "cheese_2": _CHEESE_THICKNESS,
                "cheese_3": _CHEESE_THICKNESS,
            }.get(name)

            if target_thickness is not None:
                self._set_object_z_thickness(
                    model,
                    target_thickness,
                    mesh_axis=2,
                )

            cfg["info"] = {
                "groups_containing_sampled_obj": [
                    "all",
                    name,
                ],
                "groups": [
                    name
                ],
                "cat":
                    name,
                "mjcf_path":
                    "",
            }

            self.objects[
                model.name
            ] = model

            self.model.merge_objects(
                [
                    model
                ]
            )

            setattr(
                self,
                name,
                model,
            )

        self._cheese_asset_note = (
            f"cheese mesh flattened to {_CHEESE_THICKNESS * 1000:.1f} mm "
            "along object Z for this environment."
        )

    def _set_object_z_thickness(self, obj, target_thickness, mesh_axis=2):
        """Flatten one C2-T2 object, including visual and collision meshes."""

        current_thickness = float(obj.bbox_full_size_m[2])
        ratio = float(target_thickness) / current_thickness

        for mesh in obj.asset.findall("mesh"):
            scale = [float(v) for v in mesh.get("scale", "1 1 1").split()]
            scale[mesh_axis] *= ratio
            mesh.set("scale", " ".join(f"{v:.12g}" for v in scale))

        for geom in obj.worldbody.findall(".//geom"):
            if (geom.get("name") or "").endswith("reg_bbox"):
                size = [float(v) for v in geom.get("size").split()]
                pos = [float(v) for v in geom.get("pos", "0 0 0").split()]
                size[2] *= ratio
                pos[2] *= ratio
                geom.set("size", " ".join(f"{v:.12g}" for v in size))
                geom.set("pos", " ".join(f"{v:.12g}" for v in pos))

        for site in obj.worldbody.findall(".//site"):
            if (site.get("name") or "").endswith(("bottom_site", "top_site")):
                pos = [float(v) for v in site.get("pos").split()]
                pos[2] *= ratio
                site.set("pos", " ".join(f"{v:.12g}" for v in pos))

        bbox = list(obj.bbox_full_size_m)
        bbox[2] = float(target_thickness)
        obj.bbox_full_size_m = tuple(bbox)

    # ========================================================
    # Placeholder placement
    # ========================================================

    def _fixed_island_placement(
        self,
        pos,
        rotation=0.0,
    ):
        return dict(
            fixture=self.island,
            size=(0.0, 0.0),
            pos=pos,
            rotation=float(
                rotation
            ),
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
            sample_region_kwargs=dict(
                full_depth_region=True
            ),
        )

    def _get_obj_cfgs(self):

        names = [
            # Bread
            "bread_a",
            "bread_b",
            "bread_plate",

            # Turkey
            "turkey_plate",
            "turkey_1",
            "turkey_2",
            "turkey_3",

            # Cheese
            "cheese_plate",
            "cheese_1",
            "cheese_2",
            "cheese_3",

            # Tomato
            "tomato_plate",
            "tomato_slice",

            # Serving
            "serving_plate",

            # Tools
            "knife",
            "spatula",
            "spoon",
            "ladle",
        ]

        return [
            dict(
                name=n,
                placement=(
                    self._fixed_island_placement(
                        (
                            0.0,
                            0.0,
                        )
                    )
                ),
            )
            for n in names
        ]

    # ========================================================
    # EE rack
    # ========================================================

    def _remove_existing_ee_rack(
        self,
    ):
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

    # ========================================================
    # Placement utilities
    # ========================================================

    def _object_place_z(
        self,
        obj,
        support_top_z,
    ):
        bottom = float(
            np.asarray(
                obj.bottom_offset
            )[-1]
        )

        return float(
            support_top_z
            - bottom
        )

    def _yaw_quat_wxyz(
        self,
        yaw,
    ):
        quat_xyzw = T.mat2quat(
            T.euler2mat(
                np.array(
                    [
                        0.0,
                        0.0,
                        float(yaw),
                    ]
                )
            )
        )

        return np.array(
            T.convert_quat(
                quat_xyzw,
                to="wxyz",
            ),
            dtype=float,
        )

    def _set_placement(
        self,
        name,
        pos,
        quat,
    ):
        self.object_placements[
            name
        ] = (
            tuple(
                np.asarray(
                    pos,
                    dtype=float,
                ).tolist()
            ),
            np.asarray(
                quat,
                dtype=float,
            ),
            self.objects[
                name
            ],
        )

    # ========================================================
    # C1-T2 robot position
    # ========================================================

    def _compute_c1_work_origin(
        self,
        bounds,
    ):
        """Exactly the same robot/work-origin computation used in C1-T2."""

        work_y = float(
            bounds[
                "center_xy"
            ][1]
        )

        min_layout_x = min(
            float(
                xy[0]
            )
            for xy
            in EE_RACK_LAYOUT.values()
        )

        edge = 0.04
        gap = 0.05

        work_x_lo = float(
            bounds["xmin"]
            + edge
            - min_layout_x
        )

        work_x_hi = float(
            bounds["xmin"]
            - gap
            - PEDESTAL_HALF_XY[0]
            - ROBOT_BASE_X
        )

        if (
            work_x_lo
            <= work_x_hi
        ):
            work_x = (
                0.5
                * (
                    work_x_lo
                    + work_x_hi
                )
            )
        else:
            work_x = (
                work_x_lo
            )

        work_origin = np.array(
            [
                work_x,
                work_y,
            ],
            dtype=float,
        )

        robot_xy = (
            work_origin
            + np.array(
                [
                    ROBOT_BASE_X,
                    0.0,
                ],
                dtype=float,
            )
        )

        robot_yaw = 0.0

        return (
            work_origin,
            robot_xy,
            robot_yaw,
        )

    # ========================================================
    # Stack objects
    # ========================================================

    def _stack_items(
        self,
        support_xy,
        support_top_z,
        item_names,
        clearance=_STACK_CLEARANCE,
        underside_clearance=0.0,
    ):
        """Stack objects using their real top/bottom offsets."""

        top_z = float(support_top_z + underside_clearance)

        for name in item_names:
            obj = self.objects[
                name
            ]

            z = self._object_place_z(
                obj,
                top_z,
            )

            pos = np.array(
                [
                    support_xy[0],
                    support_xy[1],
                    z,
                ],
                dtype=float,
            )

            self._set_placement(
                name,
                pos,
                self._yaw_quat_wxyz(
                    0.0
                ),
            )

            top_z = (
                z
                + float(
                    np.asarray(
                        obj.top_offset
                    )[-1]
                )
                + float(
                    clearance
                )
            )

        return top_z

    # ========================================================
    # Main layout
    # ========================================================

    def _apply_island_layout(
        self,
    ):
        if not getattr(
            self,
            "object_placements",
            None,
        ):
            return

        if (
            "bread_a"
            not in self.object_placements
        ):
            return

        (
            surface_z,
            bounds,
        ) = (
            self._island_surface_and_bounds()
        )

        self._island_surface_z = (
            surface_z
        )

        self._island_bounds = (
            bounds
        )

        (
            work_origin,
            robot_xy,
            robot_yaw,
        ) = (
            self._compute_c1_work_origin(
                bounds
            )
        )

        # ====================================================
        # EE rack
        # ====================================================

        rack_layout = {
            ee_id: (
                float(
                    work_origin[0]
                    + xy[0]
                ),
                float(
                    work_origin[1]
                    + xy[1]
                ),
            )
            for ee_id, xy
            in EE_RACK_LAYOUT.items()
        }

        # ====================================================
        # Plates
        # ====================================================

        for name, (
            dx,
            dy,
        ) in _XY_OFFSETS.items():

            obj = self.objects[
                name
            ]

            pos = np.array(
                [
                    work_origin[0]
                    + dx,

                    work_origin[1]
                    + dy,

                    self._object_place_z(
                        obj,
                        surface_z,
                    ),
                ],
                dtype=float,
            )

            self._set_placement(
                name,
                pos,
                self._yaw_quat_wxyz(
                    0.0
                ),
            )

        # ====================================================
        # Tools
        # ====================================================

        for index, name in enumerate(
            _TOOL_ORDER
        ):
            obj = self.objects[
                name
            ]

            pos = np.array(
                [
                    work_origin[0]
                    + _TOOL_ROW_FORWARD,

                    work_origin[1]
                    + float(
                        _TOOL_Y_FROM_CENTER[
                            index
                        ]
                        * _TOOL_SPACING
                    ),

                    self._object_place_z(
                        obj,
                        surface_z,
                    ),
                ],
                dtype=float,
            )

            self._set_placement(
                name,
                pos,
                self._yaw_quat_wxyz(
                    float(
                        _TOOL_ROTATIONS[
                            name
                        ]
                    )
                ),
            )

        # ====================================================
        # Support utility
        # ====================================================

        def support_top(
            name,
        ):
            (
                pos,
                _,
                obj,
            ) = (
                self.object_placements[
                    name
                ]
            )

            xy = np.array(
                pos[:2],
                dtype=float,
            )

            top = (
                float(
                    pos[2]
                )
                + float(
                    np.asarray(
                        obj.top_offset
                    )[-1]
                )
                + _PLATE_FOOD_SURFACE_OFFSET
            )

            return (
                xy,
                top,
            )

        # ====================================================
        # Bread A / B → Bread Plate
        # ====================================================

        (
            bread_xy,
            bread_top,
        ) = support_top(
            "bread_plate"
        )

        self._stack_items(
            bread_xy,
            bread_top,
            [
                "bread_a",
                "bread_b",
            ],
            clearance=-0.005,
        )

        # ====================================================
        # Turkey → Turkey Plate
        # ====================================================

        (
            turkey_xy,
            turkey_top,
        ) = support_top(
            "turkey_plate"
        )

        self._stack_items(
            turkey_xy,
            turkey_top,
            [
                "turkey_1",
                "turkey_2",
                "turkey_3",
            ],
            clearance=0.0002,
            underside_clearance=_SLICE_UNDERSIDE_CLEARANCE,
        )

        # ====================================================
        # Cheese → Cheese Plate
        # ====================================================

        (
            cheese_xy,
            cheese_top,
        ) = support_top(
            "cheese_plate"
        )

        self._stack_items(
            cheese_xy,
            cheese_top,
            [
                "cheese_1",
                "cheese_2",
                "cheese_3",
            ],
            clearance=0.0002,
            underside_clearance=_SLICE_UNDERSIDE_CLEARANCE,
        )

        # ====================================================
        # Tomato → Tomato Plate
        # ====================================================

        (
            tomato_xy,
            tomato_top,
        ) = support_top(
            "tomato_plate"
        )

        self._stack_items(
            tomato_xy,
            tomato_top,
            [
                "tomato_slice",
            ],
            clearance=0.002,
        )

        # ====================================================
        # Pedestal + Robot
        # ====================================================

        remove_robot_pedestal(
            self.model
        )

        self.pedestal_info = (
            add_robot_pedestal(
                self.model,
                center_xy=robot_xy,
                top_z=PEDESTAL_TOP_Z,
                half_size_xy=(
                    PEDESTAL_HALF_XY
                ),
            )
        )

        robot_model = (
            self.robots[
                0
            ].robot_model
        )

        robot_model.set_base_xpos(
            [
                float(
                    robot_xy[0]
                ),
                float(
                    robot_xy[1]
                ),
                PEDESTAL_TOP_Z,
            ]
        )

        robot_model.set_base_ori(
            [
                0.0,
                0.0,
                float(
                    robot_yaw
                ),
            ]
        )

        self._work_origin_xy = (
            work_origin
        )

        self._robot_base_xy = (
            robot_xy
        )

        self._robot_base_yaw = (
            robot_yaw
        )

        # ====================================================
        # EE rack
        # ====================================================

        self._remove_existing_ee_rack()

        rack_info = add_ee_rack(
            arena=self.model,

            table_offset=np.array(
                [
                    0.0,
                    0.0,
                    surface_z,
                ],
                dtype=float,
            ),

            rack_layout=rack_layout,

            ee_pool=(
                self.robot_spec[
                    "ee_pool"
                ]
            ),
        )

        for (
            ee_id,
            info,
        ) in rack_info.items():
            self.ee_catalog[
                ee_id
            ].update(
                info
            )

        self.ee_rack_info = (
            rack_info
        )

        # ====================================================
        # Metadata
        # ====================================================

        self._layout_meta = {
            "island_name":
                self.island.name,

            "surface_z":
                surface_z,

            "bounds":
                bounds,

            "work_origin_xy":
                work_origin,

            "robot_xy":
                robot_xy,

            "robot_yaw":
                robot_yaw,

            "rack_layout":
                rack_layout,

            "ee_radius":
                EE_RACK_RADIUS,

            "pedestal_top_z":
                PEDESTAL_TOP_Z,

            "cheese_asset_note":
                self._cheese_asset_note,
        }

    # ========================================================
    # References
    # ========================================================

    def _setup_references(
        self,
    ):
        super()._setup_references()

        if self.ee_rack_info:
            self.ee_rack_body_id = (
                self.sim.model.body_name2id(
                    "ee_rack"
                )
            )

    # ========================================================
    # Reset
    # ========================================================

    def _reapply_object_poses(
        self,
    ):
        if not self.object_placements:
            return

        for (
            obj_pos,
            obj_quat,
            obj,
        ) in self.object_placements.values():

            self.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate(
                    [
                        np.array(
                            obj_pos
                        ),
                        np.array(
                            obj_quat
                        ),
                    ]
                ),
            )

        self.sim.forward()

    def _reset_internal(
        self,
    ):
        ManipulationEnv._reset_internal(
            self
        )

        self._setup_scene()

        if (
            not self.deterministic_reset
            and self.placement_initializer
            is not None
        ):
            object_placements = (
                self.object_placements
            )

            self._update_sliding_fxtr_obj_placement()

            for (
                obj_pos,
                obj_quat,
                obj,
            ) in object_placements.values():

                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate(
                        [
                            np.array(
                                obj_pos
                            ),
                            np.array(
                                obj_quat
                            ),
                        ]
                    ),
                )

        if (
            self._robot_base_xy
            is not None
        ):
            self.init_robot_base_pos = (
                np.array(
                    [
                        self._robot_base_xy[
                            0
                        ],
                        self._robot_base_xy[
                            1
                        ],
                        PEDESTAL_TOP_Z,
                    ],
                    dtype=float,
                )
            )

            self.init_robot_base_ori = (
                np.array(
                    [
                        0.0,
                        0.0,
                        float(
                            self._robot_base_yaw
                        ),
                    ],
                    dtype=float,
                )
            )

        action = np.zeros(
            self.action_spec[
                0
            ].shape
        )

        policy_step = True

        for _ in range(
            8
            * int(
                self.control_timestep
                / self.model_timestep
            )
        ):
            self.sim.step1()

            self._pre_action(
                action,
                policy_step,
            )

            self.sim.step2()

            policy_step = False

        self._reapply_object_poses()

    # ========================================================
    # Success
    # ========================================================

    def _check_success(
        self,
    ):
        return False

    def get_evaluation_material_gt(self):
        """Evaluation-only GT; never used by observation, M3, or tool metadata."""
        names = (
            "bread_a",
            "bread_b",
            "turkey_1",
            "turkey_2",
            "turkey_3",
            "cheese_1",
            "cheese_2",
            "cheese_3",
            "tomato_slice",
            "knife",
            "spatula",
            "spoon",
            "ladle",
        )
        return {name: self.objects[name].material_gt for name in names}

    # ========================================================
    # Pose reporting
    # ========================================================

    def get_object_poses(
        self,
    ):
        label_map = {
            "bread_plate":
                "Bread Plate",

            "bread_a":
                "Bread A",

            "bread_b":
                "Bread B",

            "turkey_plate":
                "Turkey Plate",

            "turkey_1":
                "Turkey Slice 1",

            "turkey_2":
                "Turkey Slice 2",

            "turkey_3":
                "Turkey Slice 3",

            "cheese_plate":
                "Cheese Plate",

            "cheese_1":
                "Cheese 1",

            "cheese_2":
                "Cheese 2",

            "cheese_3":
                "Cheese 3",

            "tomato_plate":
                "Tomato Plate",

            "tomato_slice":
                "Tomato Slice",

            "serving_plate":
                "Serving Plate",

            "knife":
                "Knife",

            "spatula":
                "Spatula",

            "spoon":
                "Spoon",

            "ladle":
                "Ladle",
        }

        poses = {}

        for (
            name,
            label,
        ) in label_map.items():

            body_id = (
                self.obj_body_id[
                    name
                ]
            )

            poses[
                label
            ] = {
                "position":
                    np.array(
                        self.sim.data.body_xpos[
                            body_id
                        ],
                        dtype=float,
                    ),

                "quaternion_wxyz":
                    np.array(
                        self.sim.data.body_xquat[
                            body_id
                        ],
                        dtype=float,
                    ),
            }

        # EE rack
        if self.ee_rack_info:

            rack_body_id = (
                self.sim.model.body_name2id(
                    "ee_rack"
                )
            )

            poses[
                "EE Rack"
            ] = {
                "position":
                    np.array(
                        self.sim.data.body_xpos[
                            rack_body_id
                        ],
                        dtype=float,
                    ),

                "quaternion_wxyz":
                    np.array(
                        self.sim.data.body_xquat[
                            rack_body_id
                        ],
                        dtype=float,
                    ),
            }

            for (
                ee_id,
                info,
            ) in self.ee_rack_info.items():

                body_name = (
                    info.get(
                        "rack_body"
                    )
                )

                if not body_name:
                    continue

                body_id = (
                    self.sim.model.body_name2id(
                        body_name
                    )
                )

                label = {
                    "3F":
                        "3F",

                    "vac":
                        "Vacuum",

                    "2F":
                        "2F",
                }[
                    ee_id
                ]

                poses[
                    label
                ] = {
                    "position":
                        np.array(
                            self.sim.data.body_xpos[
                                body_id
                            ],
                            dtype=float,
                        ),

                    "quaternion_wxyz":
                        np.array(
                            self.sim.data.body_xquat[
                                body_id
                            ],
                            dtype=float,
                        ),
                }

        # Robot pedestal
        if self.pedestal_info:
            try:
                ped_id = (
                    self.sim.model.body_name2id(
                        "robot_pedestal"
                    )
                )

                poses[
                    "Robot Pedestal"
                ] = {
                    "position":
                        np.array(
                            self.sim.data.body_xpos[
                                ped_id
                            ],
                            dtype=float,
                        ),

                    "quaternion_wxyz":
                        np.array(
                            self.sim.data.body_xquat[
                                ped_id
                            ],
                            dtype=float,
                        ),
                }

            except Exception:
                pass

        return poses

    # ========================================================
    # Reachability
    # ========================================================

    def get_reachability_report(
        self,
    ):
        base_id = (
            self.sim.model.body_name2id(
                "robot0_base"
            )
        )

        base = np.array(
            self.sim.data.body_xpos[
                base_id
            ],
            dtype=float,
        )

        poses = (
            self.get_object_poses()
        )

        targets = [
            "Bread Plate",
            "Bread A",
            "Bread B",

            "Turkey Plate",
            "Cheese Plate",
            "Tomato Plate",

            "Serving Plate",

            "Knife",
            "Spatula",
            "Spoon",
            "Ladle",

            "2F",
            "3F",
            "Vacuum",
        ]

        rows = {}

        for name in targets:

            if name not in poses:
                continue

            pos = poses[
                name
            ][
                "position"
            ]

            dist_xy = float(
                np.linalg.norm(
                    pos[:2]
                    - base[:2]
                )
            )

            dist_3d = float(
                np.linalg.norm(
                    pos
                    - base
                )
            )

            rows[
                name
            ] = {
                "distance_xy":
                    dist_xy,

                "distance_3d":
                    dist_3d,

                "within_reach_xy":
                    dist_xy
                    <= UR5E_REACH_M,
            }

        ee_xy = {
            label:
                rows[
                    label
                ][
                    "distance_xy"
                ]

            for label in (
                "2F",
                "3F",
                "Vacuum",
            )

            if label
            in rows
        }

        ee_vals = list(
            ee_xy.values()
        )

        return {
            "robot_base":
                base,

            "reach_limit_m":
                UR5E_REACH_M,

            "ee_radius_c1":
                EE_RACK_RADIUS,

            "ee_dist_tol":
                EE_DIST_TOL,

            "targets":
                rows,

            "ee_xy_distances":
                ee_xy,

            "ee_xy_max_delta":
                (
                    float(
                        max(
                            ee_vals
                        )
                        - min(
                            ee_vals
                        )
                    )
                    if ee_vals
                    else None
                ),

            "all_within_reach_xy":
                all(
                    row[
                        "within_reach_xy"
                    ]
                    for row
                    in rows.values()
                ),
        }
