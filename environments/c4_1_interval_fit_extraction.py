"""C4-T1: extract a card dropped into a narrow appliance gap.

Robot, pedestal, and 2F / 3F / Vacuum rack construction uses the shared
Kitchen helpers. The appliance mock-up deliberately separates the narrow
entrance gap `g` from the wider internal channel `W`.
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import robosuite.utils.transform_utils as T
from robocasa.models.fixtures import FixtureType
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import new_body, new_geom

from environments.c1_1_lego_sweep import EE_RACK_LAYOUT, ROBOT_BASE_X
from environments.c1_2_dough_flatten import PEDESTAL_HALF_XY, PEDESTAL_TOP_Z
from environments.ee_rack import add_ee_rack
from environments.kitchen_base import KitchenBase
from environments.objects import CuttingBoardObject, KnifeObject, SpatulaObject
from environments.robot_pedestal import add_robot_pedestal, remove_robot_pedestal


ROBOT_SPEC_PATH = Path(__file__).resolve().parents[1] / "configs" / "robot_spec.json"


# ---------------------------------------------------------------------
# Gap geometry (metres)
# ---------------------------------------------------------------------

ENTRANCE_GAP = 0.009
INTERNAL_WIDTH = 0.052
SIDE_MARGIN = 0.008
WIDTH_REQUIREMENT = 0.034
REQUIRED_REACH = 0.21

_APPLIANCE_DEPTH = 0.25
_APPLIANCE_WIDTH = 0.20
_APPLIANCE_HEIGHT = 0.22

_CLEARANCE_ROOF_DEPTH = 0.035
_CLEARANCE_ROOF_THICKNESS = 0.012


# ---------------------------------------------------------------------
# Tool geometry
#
# Asset-local X / Y / Z are respectively:
#   X = contact width
#   Y = total reach length
#   Z = thickness
#
# Values are:
#   (thickness t, contact width w, reach length l)
# ---------------------------------------------------------------------

_TOOL_GEOMETRY = {
    "tool_1_knife": (0.002, 0.015, 0.220),
    "tool_2_spatula_a": (0.004, 0.036, 0.140),
    "tool_3_spatula_b": (0.007, 0.040, 0.220),
    "tool_4_spatula_c": (0.011, 0.056, 0.220),
    "tool_5_cutting_board": (0.016, 0.072, 0.220),
}

_TOOL_ORDER = tuple(_TOOL_GEOMETRY)
def _tool_geometry_meta():
    return {
        name: {
            "thickness_m": thickness,
            "width_m": width,
            "length_m": length,
        }
        for name, (thickness, width, length) in _TOOL_GEOMETRY.items()
    }
_TOOL_ROW_X = -0.21
_TOOL_SPACING = 0.105
_TOOL_Y_OFFSETS = (0.21, 0.105, 0.0, -0.105, -0.21)
_TOOL_YAW = -np.pi / 2


# ---------------------------------------------------------------------
# Material appearance
# ---------------------------------------------------------------------

_METAL_SPECULAR = 0.90
_METAL_SHININESS = 0.85

_PLASTIC_SPECULAR = 0.18
_PLASTIC_SHININESS = 0.12


# Procedural BoxObjects do not have their own top-level <mujoco><custom>.
# Their GT is therefore injected into the final environment MJCF only.
_PROCEDURAL_MATERIAL_GT = {
    "appliance_left": "metal",
    "appliance_right": "metal",
    "card": "plastic",
}


class _MaterialBoxObject(BoxObject):
    """Procedural BoxObject with evaluation-only Material GT."""

    def __init__(
        self,
        *,
        material_gt,
        specular,
        shininess,
        **kwargs,
    ):
        # material_gt는 BoxObject.__init__으로 전달하지 않는다.
        super().__init__(**kwargs)

        self._pending_material_gt = str(material_gt)

        material_name = f"{self.name}_visual_material"

        ET.SubElement(
            self.asset,
            "material",
            name=material_name,
            specular=f"{float(specular):.12g}",
            shininess=f"{float(shininess):.12g}",
        )

        obj_root = self.get_obj()

        for geom in obj_root.findall(".//geom"):
            if geom.get("group") == "1":
                geom.set("material", material_name)

        self._material_gt_xml_root = None
        self._material_gt_text_name = None

    def bind_material_gt(self, mjcf_root):
        """Store evaluation GT in final MJCF <custom>."""

        text_name = f"material_gt__{self.name}"

        custom = mjcf_root.find("custom")
        if custom is None:
            custom = ET.SubElement(mjcf_root, "custom")

        entry = custom.find(f"./text[@name='{text_name}']")

        if entry is None:
            ET.SubElement(
                custom,
                "text",
                name=text_name,
                data=self._pending_material_gt,
            )
        else:
            entry.set("data", self._pending_material_gt)

        self._material_gt_xml_root = mjcf_root
        self._material_gt_text_name = text_name

    @property
    def material_gt(self) -> str:
        if self._material_gt_xml_root is None:
            raise ValueError(
                f"material_gt XML root is not bound for {self.name}"
            )

        entry = self._material_gt_xml_root.find(
            f"./custom/text[@name='{self._material_gt_text_name}']"
        )

        if entry is None or not entry.get("data"):
            raise ValueError(
                f"material_gt not found for {self.name}"
            )

        return entry.get("data")


class C4_1_IntervalFitExtraction(KitchenBase):
    """Scene-only C4-T1 environment."""

    def __init__(
        self,
        robots="UR5e",
        gripper_types=None,
        base_types="NullMount",
        initialization_noise="default",
        seed=0,
        **kwargs,
    ):
        kwargs.setdefault("use_distractors", False)
        kwargs.setdefault("use_object_obs", True)
        kwargs.setdefault("robot_spawn_deviation_pos_x", 0.0)
        kwargs.setdefault("robot_spawn_deviation_pos_y", 0.0)
        kwargs.setdefault("robot_spawn_deviation_rot", 0.0)

        with ROBOT_SPEC_PATH.open(encoding="utf-8") as spec_file:
            self.robot_spec = json.load(spec_file)

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

        super().__init__(
            robots=robots,
            gripper_types=gripper_types,
            base_types=base_types,
            initialization_noise=initialization_noise,
            seed=seed,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Kitchen references
    # ------------------------------------------------------------------

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

    def _get_kitchen_ep_meta(self):
        return super().get_ep_meta()

    def get_ep_meta(self):
        ep_meta = self._get_kitchen_ep_meta()

        ep_meta["lang"] = "가전 사이 틈에 떨어진 카드를 꺼내라."

        ep_meta["entrance_gap_m"] = ENTRANCE_GAP
        ep_meta["internal_width_m"] = INTERNAL_WIDTH
        ep_meta["width_requirement_m"] = WIDTH_REQUIREMENT
        ep_meta["required_reach_m"] = REQUIRED_REACH

        ep_meta["g"] = ENTRANCE_GAP
        ep_meta["W"] = INTERNAL_WIDTH
        ep_meta["w_req"] = WIDTH_REQUIREMENT
        ep_meta["required_reach"] = REQUIRED_REACH

        ep_meta["tool_geometry_m"] = _tool_geometry_meta()

        return ep_meta

    # ------------------------------------------------------------------
    # Island geometry
    # ------------------------------------------------------------------

    def _island_surface_and_bounds(self):
        region = self.island.sample_reset_region(
            env=self,
            full_depth_region=True,
        )

        surface_z = float(
            self.island.pos[2] + region["offset"][2]
        )

        pts = np.asarray(
            self.island.get_bbox_points(),
            dtype=float,
        )

        bounds = {
            "xmin": float(pts[:, 0].min()),
            "xmax": float(pts[:, 0].max()),
            "ymin": float(pts[:, 1].min()),
            "ymax": float(pts[:, 1].max()),
            "zmin": float(pts[:, 2].min()),
            "zmax": float(pts[:, 2].max()),
            "center_xy": np.array(
                [
                    0.5 * (
                        pts[:, 0].min()
                        + pts[:, 0].max()
                    ),
                    0.5 * (
                        pts[:, 1].min()
                        + pts[:, 1].max()
                    ),
                ],
                dtype=float,
            ),
        }

        return surface_z, bounds

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _load_model(self, attempt_num=1):
        super()._load_model(attempt_num=attempt_num)

        # At this point KitchenBase has already created and merged objects.
        # Bind generated BoxObject GT to the final environment MJCF.
        self._bind_procedural_material_gt()

        self._apply_island_layout()

    def _bind_procedural_material_gt(self):
        """Bind all procedural BoxObject GT values to final MJCF <custom>."""

        for obj in self.objects.values():
            if isinstance(obj, _MaterialBoxObject):
                obj.bind_material_gt(self.model.root)

    def _fixed_island_placement(
        self,
        pos,
        rotation=0.0,
    ):
        return dict(
            fixture=self.island,
            size=(0.0, 0.0),
            pos=pos,
            rotation=float(rotation),
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
            sample_region_kwargs=dict(
                full_depth_region=True
            ),
        )

    def _remove_existing_ee_rack(self):
        for body in list(
            self.model.worldbody.findall("body")
        ):
            name = body.get("name") or ""

            if (
                name == "ee_rack"
                or name.startswith("robot0_rack_")
            ):
                self.model.worldbody.remove(body)

    @staticmethod
    def _object_place_z(
        obj,
        support_top_z,
    ):
        return float(
            support_top_z
            - float(
                np.asarray(obj.bottom_offset)[-1]
            )
        )

    @staticmethod
    def _yaw_quat_wxyz(yaw):
        quat_xyzw = T.mat2quat(
            T.euler2mat(
                np.array(
                    [0.0, 0.0, float(yaw)]
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
        self.object_placements[name] = (
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
            self.objects[name],
        )

    @staticmethod
    def _compute_c1_work_origin(bounds):
        work_y = float(
            bounds["center_xy"][1]
        )

        min_layout_x = min(
            float(xy[0])
            for xy in EE_RACK_LAYOUT.values()
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

        work_x = (
            0.5 * (
                work_x_lo
                + work_x_hi
            )
            if work_x_lo <= work_x_hi
            else work_x_lo
        )

        work_origin = np.array(
            [work_x, work_y],
            dtype=float,
        )

        robot_xy = (
            work_origin
            + np.array(
                [ROBOT_BASE_X, 0.0],
                dtype=float,
            )
        )

        return (
            work_origin,
            robot_xy,
            0.0,
        )

    # ------------------------------------------------------------------
    # Objects
    # ------------------------------------------------------------------

    def _make_box(
    self,
    name,
    full_size,
    rgba,
    *,
    material_gt,
    specular,
    shininess,
):
        return _MaterialBoxObject(
        name=name,
        size=np.asarray(full_size, dtype=float) / 2.0,
        rgba=rgba,
        joints="default",
        obj_type="all",
        material_gt=material_gt,
        specular=specular,
        shininess=shininess,
    )

    def _create_objects(self):
        self.objects = {}
        self.object_cfgs = self._get_obj_cfgs()

        appliance_size = (
            _APPLIANCE_DEPTH,
            _APPLIANCE_WIDTH,
            _APPLIANCE_HEIGHT,
        )

        builders = {
            "appliance_left": lambda n: self._make_box(
                n,
                appliance_size,
                (0.30, 0.32, 0.35, 1.0),
                material_gt="metal",
                specular=_METAL_SPECULAR,
                shininess=_METAL_SHININESS,
            ),
            "appliance_right": lambda n: self._make_box(
                n,
                appliance_size,
                (0.30, 0.32, 0.35, 1.0),
                material_gt="metal",
                specular=_METAL_SPECULAR,
                shininess=_METAL_SHININESS,
            ),
            "card": lambda n: self._make_box(
                n,
                (0.054, 0.032, 0.0012),
                (0.12, 0.34, 0.82, 1.0),
                material_gt="plastic",
                specular=_PLASTIC_SPECULAR,
                shininess=_PLASTIC_SHININESS,
            ),
            "tool_1_knife": lambda n: KnifeObject(
                name=n
            ),
            "tool_2_spatula_a": lambda n: SpatulaObject(
                name=n
            ),
            "tool_3_spatula_b": lambda n: SpatulaObject(
                name=n
            ),
            "tool_4_spatula_c": lambda n: SpatulaObject(
                name=n
            ),
            "tool_5_cutting_board": lambda n: CuttingBoardObject(
                name=n
            ),
        }

        for cfg in self.object_cfgs:
            cfg["type"] = "object"

            name = cfg["name"]

            model = builders[name](name)

            if name in _TOOL_GEOMETRY:
                (
                    thickness,
                    width,
                    length,
                ) = _TOOL_GEOMETRY[name]

                self._set_tool_geometry(
                    model,
                    thickness=thickness,
                    width=width,
                    length=length,
                )

            cfg["info"] = {
                "groups_containing_sampled_obj": [
                    "all",
                    name,
                ],
                "groups": [name],
                "cat": name,
                "mjcf_path": "",
            }

            self.objects[model.name] = model

            self.model.merge_objects(
                [model]
            )

            setattr(
                self,
                name,
                model,
            )

    @staticmethod
    def _set_tool_geometry(
        obj,
        thickness,
        width,
        length,
    ):
        """Apply target X-width, Y-length, and Z-thickness to all geometry."""

        bbox = np.asarray(
            obj.bbox_full_size_m,
            dtype=float,
        )

        factors = np.array(
            [
                float(width) / bbox[0],
                float(length) / bbox[1],
                float(thickness) / bbox[2],
            ]
        )

        # Mesh assets are shared by visual and mesh collision geoms.
        for mesh in obj.asset.findall("mesh"):
            scale = np.array(
                [
                    float(v)
                    for v in mesh.get(
                        "scale",
                        "1 1 1",
                    ).split()
                ],
                dtype=float,
            )

            mesh.set(
                "scale",
                " ".join(
                    f"{v:.12g}"
                    for v in scale * factors
                ),
            )

        # Scale explicit box geometry too
        # (cutting-board collision and bbox).
        for geom in obj.worldbody.findall(
            ".//geom"
        ):
            if (
                geom.get("type") == "box"
                and geom.get("size")
            ):
                size = np.array(
                    [
                        float(v)
                        for v in geom.get(
                            "size"
                        ).split()
                    ]
                )

                geom.set(
                    "size",
                    " ".join(
                        f"{v:.12g}"
                        for v in size * factors
                    ),
                )

            if geom.get("pos"):
                pos = np.array(
                    [
                        float(v)
                        for v in geom.get(
                            "pos"
                        ).split()
                    ]
                )

                geom.set(
                    "pos",
                    " ".join(
                        f"{v:.12g}"
                        for v in pos * factors
                    ),
                )

        for site in obj.worldbody.findall(
            ".//site"
        ):
            if site.get("pos"):
                pos = np.array(
                    [
                        float(v)
                        for v in site.get(
                            "pos"
                        ).split()
                    ]
                )

                site.set(
                    "pos",
                    " ".join(
                        f"{v:.12g}"
                        for v in pos * factors
                    ),
                )

        obj.bbox_full_size_m = (
            float(width),
            float(length),
            float(thickness),
        )

    def _get_obj_cfgs(self):
        names = (
            "appliance_left",
            "appliance_right",
            "card",
            *_TOOL_ORDER,
        )

        return [
            dict(
                name=name,
                placement=self._fixed_island_placement(
                    (0.0, 0.0)
                ),
            )
            for name in names
        ]

    # ------------------------------------------------------------------
    # Scene layout
    # ------------------------------------------------------------------

    def _apply_island_layout(self):
        if not getattr(
            self,
            "object_placements",
            None,
        ):
            return

        if "card" not in self.object_placements:
            return

        (
            surface_z,
            bounds,
        ) = self._island_surface_and_bounds()

        self._island_surface_z = surface_z
        self._island_bounds = bounds

        (
            work_origin,
            robot_xy,
            robot_yaw,
        ) = self._compute_c1_work_origin(
            bounds
        )

        # --------------------------------------------------------------
        # Narrow channel
        #
        # Internal channel runs in X.
        # Robot-facing entrance is at -X.
        #
        # W = clear Y distance between the two appliance inner faces.
        #
        # Tool width constraint:
        #     w <= W - SIDE_MARGIN
        #     w <= 52 mm - 8 mm
        #     w <= 44 mm
        # --------------------------------------------------------------

        channel_center_x = float(
            work_origin[0] + 0.17
        )

        channel_entrance_x = (
            channel_center_x
            - 0.5 * _APPLIANCE_DEPTH
        )

        center_y = float(
            work_origin[1]
        )

        appliance_y = 0.5 * (
            INTERNAL_WIDTH
            + _APPLIANCE_WIDTH
        )

        appliance_poses = {
            "appliance_left": (
                channel_center_x,
                center_y + appliance_y,
            ),
            "appliance_right": (
                channel_center_x,
                center_y - appliance_y,
            ),
        }

        for name, xy in appliance_poses.items():
            self._set_placement(
                name,
                (
                    xy[0],
                    xy[1],
                    self._object_place_z(
                        self.objects[name],
                        surface_z,
                    ),
                ),
                self._yaw_quat_wxyz(
                    0.0
                ),
            )

        # --------------------------------------------------------------
        # Thickness clearance
        #
        # g = vertical clearance between the island surface and the
        # underside of this static roof.
        #
        # Tool thickness constraint:
        #     t < g
        #     t < 9 mm
        # --------------------------------------------------------------

        clearance_x = (
            channel_center_x
            - 0.035
        )

        clearance_body = new_body(
            name="tool_thickness_clearance",
            pos=[
                clearance_x,
                center_y,
                (
                    surface_z
                    + ENTRANCE_GAP
                    + 0.5
                    * _CLEARANCE_ROOF_THICKNESS
                ),
            ],
        )

        clearance_body.append(
            new_geom(
                name="tool_thickness_clearance_geom",
                type="box",
                size=[
                    0.5
                    * _CLEARANCE_ROOF_DEPTH,
                    0.5
                    * INTERNAL_WIDTH,
                    0.5
                    * _CLEARANCE_ROOF_THICKNESS,
                ],
                pos=[
                    0.0,
                    0.0,
                    0.0,
                ],
                group=0,
                rgba=[
                    0.20,
                    0.22,
                    0.25,
                    0.75,
                ],
                contype="1",
                conaffinity="1",
            )
        )

        self.model.worldbody.append(
            clearance_body
        )

        # --------------------------------------------------------------
        # Card
        #
        # Its far (+X) edge is placed exactly REQUIRED_REACH from
        # the channel entrance.
        #
        # T2 Spatula A = 140 mm -> cannot reach
        # T3 Spatula B = 220 mm -> can reach
        # --------------------------------------------------------------

        card_half_length = (
            0.5 * 0.054
        )

        card_center_x = (
            channel_entrance_x
            + REQUIRED_REACH
            - card_half_length
        )

        self._set_placement(
            "card",
            (
                card_center_x,
                center_y,
                self._object_place_z(
                    self.objects["card"],
                    surface_z,
                ),
            ),
            self._yaw_quat_wxyz(
                0.0
            ),
        )

        # --------------------------------------------------------------
        # Tools
        # --------------------------------------------------------------

        for name, y_offset in zip(
            _TOOL_ORDER,
            _TOOL_Y_OFFSETS,
        ):
            obj = self.objects[name]

            self._set_placement(
                name,
                (
                    float(
                        work_origin[0]
                        + _TOOL_ROW_X
                    ),
                    float(
                        work_origin[1]
                        + y_offset
                    ),
                    self._object_place_z(
                        obj,
                        surface_z,
                    ),
                ),
                self._yaw_quat_wxyz(
                    _TOOL_YAW
                ),
            )

        # --------------------------------------------------------------
        # Robot + EE rack
        # --------------------------------------------------------------

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
            in self._rack_layout().items()
        }

        self._install_robot_and_rack(
            surface_z,
            work_origin,
            robot_xy,
            robot_yaw,
            rack_layout,
        )

        self._layout_meta = {
            "island_name": self.island.name,
            "surface_z": surface_z,
            "bounds": bounds,
            "work_origin_xy": work_origin,
            "robot_xy": robot_xy,
            "robot_yaw": robot_yaw,
            "rack_layout": rack_layout,

            "entrance_gap_m": ENTRANCE_GAP,
            "internal_width_m": INTERNAL_WIDTH,
            "side_margin_m": SIDE_MARGIN,
            "width_requirement_m": WIDTH_REQUIREMENT,
            "required_reach_m": REQUIRED_REACH,

            "g": ENTRANCE_GAP,
            "W": INTERNAL_WIDTH,
            "w_req": WIDTH_REQUIREMENT,
            "required_reach": REQUIRED_REACH,

            "tool_geometry_m": _tool_geometry_meta(),
        }

    # ------------------------------------------------------------------
    # EE rack / robot
    # ------------------------------------------------------------------

    @staticmethod
    def _rack_layout():
        return EE_RACK_LAYOUT

    def _install_robot_and_rack(
        self,
        surface_z,
        work_origin,
        robot_xy,
        robot_yaw,
        rack_layout,
    ):
        """Install shared pedestal, robot base, and EE rack."""

        remove_robot_pedestal(
            self.model
        )

        self.pedestal_info = (
            add_robot_pedestal(
                self.model,
                center_xy=robot_xy,
                top_z=PEDESTAL_TOP_Z,
                half_size_xy=PEDESTAL_HALF_XY,
            )
        )

        robot_model = (
            self.robots[0].robot_model
        )

        robot_model.set_base_xpos(
            [
                float(robot_xy[0]),
                float(robot_xy[1]),
                PEDESTAL_TOP_Z,
            ]
        )

        robot_model.set_base_ori(
            [
                0.0,
                0.0,
                float(robot_yaw),
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
            ee_pool=self.robot_spec[
                "ee_pool"
            ],
        )

        for ee_id, info in rack_info.items():
            self.ee_catalog[
                ee_id
            ].update(info)

        self.ee_rack_info = rack_info

    # ------------------------------------------------------------------
    # Simulation references / reset
    # ------------------------------------------------------------------

    def _setup_references(self):
        super()._setup_references()

        if self.ee_rack_info:
            self.ee_rack_body_id = (
                self.sim.model.body_name2id(
                    "ee_rack"
                )
            )

    def _reapply_object_poses(self):
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
                        np.asarray(obj_pos),
                        np.asarray(obj_quat),
                    ]
                ),
            )

        self.sim.forward()

    def _reset_internal(self):
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
                            np.asarray(
                                obj_pos
                            ),
                            np.asarray(
                                obj_quat
                            ),
                        ]
                    ),
                )

        if self._robot_base_xy is not None:
            self.init_robot_base_pos = (
                np.array(
                    [
                        self._robot_base_xy[0],
                        self._robot_base_xy[1],
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
            self.action_spec[0].shape
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

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------

    def _check_success(self):
        return False

    # ------------------------------------------------------------------
    # Evaluation-only Material GT
    # ------------------------------------------------------------------

    def get_evaluation_material_gt(self):
        """Evaluation-only GT; never exposed via observations or metadata."""

        names = (
            "appliance_left",
            "appliance_right",
            "card",
            "tool_1_knife",
            "tool_2_spatula_a",
            "tool_3_spatula_b",
            "tool_4_spatula_c",
            "tool_5_cutting_board",
        )

        return {
            name: self.objects[
                name
            ].material_gt
            for name in names
        }