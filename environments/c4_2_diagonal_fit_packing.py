"""C4-T2: Diagonal-Fit Packing scene.

Independent KitchenBase environment.

Task:
    긴 물건들을 상자에 담고 뚜껑을 덮어라.

C4-T2 does NOT inherit from C4-T1.
Robot, pedestal, Island, and EE-rack construction are implemented
directly in this environment using the shared project helpers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import robosuite.utils.transform_utils as T

from robocasa.models.fixtures import FixtureType
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import new_body, new_geom

from environments.c1_1_lego_sweep import (
    EE_RACK_LAYOUT,
    ROBOT_BASE_X,
)
from environments.c1_2_dough_flatten import (
    PEDESTAL_HALF_XY,
    PEDESTAL_TOP_Z,
)
from environments.ee_rack import add_ee_rack
from environments.kitchen_base import KitchenBase
from environments.robot_pedestal import (
    add_robot_pedestal,
    remove_robot_pedestal,
)

from environments.objects.c4_2_packing_objects import (
    BaguetteObject,
    CerealObject,
    MilkObject,
    RollingPinObject,
)
from environments.objects.whisk_object import WhiskObject


# =====================================================================
# Paths
# =====================================================================

ROBOT_SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "robot_spec.json"
)


# =====================================================================
# Box geometry
# =====================================================================

BOX_INNER_W = 0.240
BOX_INNER_D = 0.180
BOX_INNER_H = 0.200

BOX_WALL_THICKNESS = 0.010
BOX_FLOOR_THICKNESS = 0.010

BOX_DIAGONAL = math.sqrt(
    BOX_INNER_W ** 2
    + BOX_INNER_D ** 2
    + BOX_INNER_H ** 2
)


# =====================================================================
# Benchmark object lengths
# =====================================================================

OBJECT_LENGTHS = {
    "rolling_pin": 0.300,
    "baguette": 0.260,
    "whisk": 0.280,
    "cereal": 0.150,
    "milk": 0.144,
}

_PACKING_OBJECTS = tuple(OBJECT_LENGTHS)


# =====================================================================
# Initial object row
# =====================================================================

a = -0.1

_ROW_Y_OFFSETS = (
    0.48 + a,
    0.33 + a,
    0.18 + a,
    0.03 + a,
    -0.12 + a,
)

_ROW_X_OFFSET = -0.25

_LONG_OBJECTS = {
    "rolling_pin",
    "baguette",
    "whisk",
}

_LONG_AXIS = {
    "rolling_pin": 1,
    "baguette": 1,
    "whisk": 1,
    "cereal": 2,
    "milk": 2,
}


_BOX_BODY_NAME = "packing_box"


# =====================================================================
# Material BoxObject
# =====================================================================


class _MaterialBoxObject(BoxObject):
    """Procedural BoxObject with evaluation-only Material GT.

    BoxObject itself has no top-level MJCF ``root`` attribute.
    Therefore Material GT is bound to the final environment MJCF
    <custom> section after object creation.
    """

    def __init__(
        self,
        *,
        material_gt,
        specular,
        shininess,
        **kwargs,
    ):
        # material_gt must NOT be forwarded to BoxObject.
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

        # Generated BoxObject body subtree.
        obj_root = self.get_obj()

        for geom in obj_root.findall(".//geom"):
            if geom.get("group") == "1":
                geom.set(
                    "material",
                    material_name,
                )

        self._material_gt_xml_root = None
        self._material_gt_text_name = None

    def bind_material_gt(self, mjcf_root):
        """Store evaluation GT in final MJCF <custom>."""

        text_name = f"material_gt__{self.name}"

        custom = mjcf_root.find("custom")

        if custom is None:
            custom = ET.SubElement(
                mjcf_root,
                "custom",
            )

        entry = custom.find(
            f"./text[@name='{text_name}']"
        )

        if entry is None:
            ET.SubElement(
                custom,
                "text",
                name=text_name,
                data=self._pending_material_gt,
            )
        else:
            entry.set(
                "data",
                self._pending_material_gt,
            )

        self._material_gt_xml_root = mjcf_root
        self._material_gt_text_name = text_name

    @property
    def material_gt(self) -> str:
        """Return Material GT from final MJCF."""

        if self._material_gt_xml_root is None:
            raise ValueError(
                f"material_gt XML root is not bound for {self.name}"
            )

        if not self._material_gt_text_name:
            raise ValueError(
                f"material_gt text name is not bound for {self.name}"
            )

        entry = self._material_gt_xml_root.find(
            f"./custom/text[@name='{self._material_gt_text_name}']"
        )

        if entry is None or not entry.get("data"):
            raise ValueError(
                f"material_gt not found for {self.name}"
            )

        return entry.get("data")


# =====================================================================
# Fit metadata
# =====================================================================


def _fit_metadata():
    axis_limit = max(
        BOX_INNER_W,
        BOX_INNER_D,
        BOX_INNER_H,
    )

    return {
        name: {
            "length_m": length,
            "fits_axis_aligned": bool(
                length <= axis_limit
            ),
            "fits_diagonal_bound": bool(
                length <= BOX_DIAGONAL
            ),
        }
        for name, length in OBJECT_LENGTHS.items()
    }


# =====================================================================
# C4-T2 Environment
# =====================================================================


class C4_2_DiagonalFitPacking(KitchenBase):
    """Scene-only C4-T2 packing environment.

    This environment directly inherits KitchenBase and has no dependency
    on C4-T1.
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

        self._packing_box_center = None
        self._pending_packing_box_spec = None

        super().__init__(
            robots=robots,
            gripper_types=gripper_types,
            base_types=base_types,
            initialization_noise=initialization_noise,
            seed=seed,
            **kwargs,
        )

    # =================================================================
    # Kitchen references
    # =================================================================

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

        # Robot base is handled manually.
        self.init_robot_base_ref = None

    # =================================================================
    # Episode metadata
    # =================================================================

    def _get_kitchen_ep_meta(self):
        return super().get_ep_meta()

    def get_ep_meta(self):
        ep_meta = self._get_kitchen_ep_meta()

        ep_meta.update(
            {
                "lang": (
                    "긴 물건들을 상자에 담고 "
                    "뚜껑을 덮어라."
                ),
                "box_inner_w_m": BOX_INNER_W,
                "box_inner_d_m": BOX_INNER_D,
                "box_inner_h_m": BOX_INNER_H,
                "box_diagonal_m": BOX_DIAGONAL,
                "object_fit": _fit_metadata(),
            }
        )

        return ep_meta

    # =================================================================
    # Island helpers
    # =================================================================

    def _island_surface_and_bounds(self):
        region = self.island.sample_reset_region(
            env=self,
            full_depth_region=True,
        )

        surface_z = float(
            self.island.pos[2]
            + region["offset"][2]
        )

        pts = np.asarray(
            self.island.get_bbox_points(),
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

        return surface_z, bounds

    @staticmethod
    def _compute_c1_work_origin(bounds):
        """Compute the same work origin used by the existing C4 scene."""

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

        if work_x_lo <= work_x_hi:
            work_x = 0.5 * (
                work_x_lo
                + work_x_hi
            )
        else:
            work_x = work_x_lo

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

        return (
            work_origin,
            robot_xy,
            0.0,
        )

    # =================================================================
    # Generic placement helpers
    # =================================================================

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

    @staticmethod
    def _object_place_z(
        obj,
        support_top_z,
    ):
        return float(
            support_top_z
            - float(
                np.asarray(
                    obj.bottom_offset
                )[-1]
            )
        )

    @staticmethod
    def _yaw_quat_wxyz(yaw):
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

    # =================================================================
    # Model construction
    # =================================================================

    def _load_model(self, attempt_num=1):
        """Build the independent C4-T2 scene before MuJoCo compilation."""

        self._pending_packing_box_spec = None

        # Kitchen / Island / robot / objects 기본 모델 생성
        super()._load_model(attempt_num=attempt_num)

        # C4-T1을 더 이상 상속하지 않으므로
        # C4-T2의 layout을 여기서 직접 적용해야 한다.
        self._apply_island_layout()

        # Procedural lid의 evaluation-only material GT를
        # 최종 environment MJCF에 연결한다.
        self._bind_procedural_material_gt()

        # _apply_island_layout()에서 계산된 packing-box 위치 확인
        box_spec = self._pending_packing_box_spec

        if box_spec is None:
            raise RuntimeError(
                "Packing-box layout was not computed before compile."
            )

        box_center, surface_z = box_spec

        # 재시도 시 기존 static box가 남지 않도록 제거
        self._remove_fixed_packing_box()

        # MuJoCo compile 전에 static packing box 추가
        self._add_fixed_packing_box(
            box_center,
            surface_z,
        )

    def _bind_procedural_material_gt(self):
        """Bind procedural BoxObject GT to final MJCF."""

        for obj in self.objects.values():
            if isinstance(
                obj,
                _MaterialBoxObject,
            ):
                obj.bind_material_gt(
                    self.model.root
                )

    # =================================================================
    # Objects
    # =================================================================

    def _create_objects(self):
        self.objects = {}
        self.object_cfgs = (
            self._get_obj_cfgs()
        )

        builders = {
            "rolling_pin": RollingPinObject,
            "baguette": BaguetteObject,
            "whisk": WhiskObject,
            "cereal": CerealObject,
            "milk": MilkObject,

            "lid": lambda name: _MaterialBoxObject(
                name=name,
                size=(
                    0.5
                    * (
                        BOX_INNER_D
                        + 2
                        * BOX_WALL_THICKNESS
                        + 0.010
                    ),
                    0.5
                    * (
                        BOX_INNER_W
                        + 2
                        * BOX_WALL_THICKNESS
                        + 0.010
                    ),
                    0.006,
                ),
                rgba=(
                    0.52,
                    0.34,
                    0.18,
                    1.0,
                ),
                joints="default",
                obj_type="all",
                material_gt="wood",
                specular=0.5,
                shininess=0.25,
            ),
        }

        for cfg in self.object_cfgs:
            cfg["type"] = "object"

            name = cfg["name"]

            model = builders[name](
                name
            )

            if name in OBJECT_LENGTHS:
                self._scale_to_benchmark_length(
                    model,
                    target_length=OBJECT_LENGTHS[
                        name
                    ],
                    long_axis=_LONG_AXIS[
                        name
                    ],
                )

            cfg["info"] = {
                "groups_containing_sampled_obj": [
                    "all",
                    name,
                ],
                "groups": [
                    name
                ],
                "cat": name,
                "mjcf_path": "",
            }

            self.objects[
                model.name
            ] = model

            self.model.merge_objects(
                [model]
            )

            setattr(
                self,
                name,
                model,
            )

    @staticmethod
    def _scale_to_benchmark_length(
        obj,
        target_length,
        long_axis,
    ):
        """Uniformly scale visual, collision, bbox and sites."""

        bbox = np.asarray(
            obj.bbox_full_size_m,
            dtype=float,
        )

        ratio = (
            float(target_length)
            / float(bbox[long_axis])
        )

        # -------------------------------------------------------------
        # Mesh geometry
        # -------------------------------------------------------------

        for mesh in obj.asset.findall(
            "mesh"
        ):
            scale = np.asarray(
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
                    for v in (
                        scale * ratio
                    )
                ),
            )

        # -------------------------------------------------------------
        # Explicit geometry
        # -------------------------------------------------------------

        for geom in obj.worldbody.findall(
            ".//geom"
        ):
            if (
                geom.get("type") == "box"
                and geom.get("size")
            ):
                size = np.asarray(
                    [
                        float(v)
                        for v
                        in geom.get(
                            "size"
                        ).split()
                    ]
                )

                geom.set(
                    "size",
                    " ".join(
                        f"{v:.12g}"
                        for v
                        in (
                            size
                            * ratio
                        )
                    ),
                )

            if geom.get("pos"):
                pos = np.asarray(
                    [
                        float(v)
                        for v
                        in geom.get(
                            "pos"
                        ).split()
                    ]
                )

                geom.set(
                    "pos",
                    " ".join(
                        f"{v:.12g}"
                        for v
                        in (
                            pos
                            * ratio
                        )
                    ),
                )

        # -------------------------------------------------------------
        # Sites
        # -------------------------------------------------------------

        for site in obj.worldbody.findall(
            ".//site"
        ):
            if site.get("pos"):
                pos = np.asarray(
                    [
                        float(v)
                        for v
                        in site.get(
                            "pos"
                        ).split()
                    ]
                )

                site.set(
                    "pos",
                    " ".join(
                        f"{v:.12g}"
                        for v
                        in (
                            pos
                            * ratio
                        )
                    ),
                )

        obj.bbox_full_size_m = tuple(
            float(v)
            for v in (
                bbox * ratio
            )
        )

    def _get_obj_cfgs(self):
        return [
            dict(
                name=name,
                placement=(
                    self._fixed_island_placement(
                        (0.0, 0.0)
                    )
                ),
            )
            for name in (
                *_PACKING_OBJECTS,
                "lid",
            )
        ]

    # =================================================================
    # Packing box
    # =================================================================

    def _add_fixed_packing_box(
        self,
        center_xy,
        surface_z,
    ):
        """Add fixed floor + four walls.

        BOX_INNER_W / D / H describe the clear interior.
        """

        body = new_body(
            name=_BOX_BODY_NAME,
            pos=[
                center_xy[0],
                center_xy[1],
                surface_z,
            ],
        )

        # Existing debug color retained.
        color = [
            1.0,
            0.0,
            0.0,
            1.0,
        ]

        def add_part(
            name,
            size,
            pos,
        ):
            body.append(
                new_geom(
                    name=name,
                    type="box",
                    size=[
                        0.5 * v
                        for v in size
                    ],
                    pos=pos,
                    group=1,
                    rgba=color,
                    contype="1",
                    conaffinity="1",
                    friction=(
                        "0.95 0.3 0.1"
                    ),
                )
            )

        outer_d = (
            BOX_INNER_D
            + 2
            * BOX_WALL_THICKNESS
        )

        outer_w = (
            BOX_INNER_W
            + 2
            * BOX_WALL_THICKNESS
        )

        # -------------------------------------------------------------
        # Floor
        # -------------------------------------------------------------

        add_part(
            "packing_box_floor",
            (
                outer_d,
                outer_w,
                BOX_FLOOR_THICKNESS,
            ),
            [
                0.0,
                0.0,
                0.5
                * BOX_FLOOR_THICKNESS,
            ],
        )

        wall_z = (
            BOX_FLOOR_THICKNESS
            + 0.5
            * BOX_INNER_H
        )

        wall_x = 0.5 * (
            BOX_INNER_D
            + BOX_WALL_THICKNESS
        )

        wall_y = 0.5 * (
            BOX_INNER_W
            + BOX_WALL_THICKNESS
        )

        # -------------------------------------------------------------
        # Front / back walls
        # -------------------------------------------------------------

        add_part(
            "packing_box_front_wall",
            (
                BOX_WALL_THICKNESS,
                outer_w,
                BOX_INNER_H,
            ),
            [
                -wall_x,
                0.0,
                wall_z,
            ],
        )

        add_part(
            "packing_box_back_wall",
            (
                BOX_WALL_THICKNESS,
                outer_w,
                BOX_INNER_H,
            ),
            [
                wall_x,
                0.0,
                wall_z,
            ],
        )

        # -------------------------------------------------------------
        # Left / right walls
        # -------------------------------------------------------------

        add_part(
            "packing_box_left_wall",
            (
                BOX_INNER_D,
                BOX_WALL_THICKNESS,
                BOX_INNER_H,
            ),
            [
                0.0,
                wall_y,
                wall_z,
            ],
        )

        add_part(
            "packing_box_right_wall",
            (
                BOX_INNER_D,
                BOX_WALL_THICKNESS,
                BOX_INNER_H,
            ),
            [
                0.0,
                -wall_y,
                wall_z,
            ],
        )

        self.model.worldbody.append(
            body
        )

    def _remove_fixed_packing_box(
        self
    ):
        """Remove stale packing box on model retry."""

        for body in list(
            self.model.worldbody.findall(
                "body"
            )
        ):
            if (
                body.get("name") or ""
            ) == _BOX_BODY_NAME:
                self.model.worldbody.remove(
                    body
                )

    # =================================================================
    # Scene layout
    # =================================================================

    def _apply_island_layout(self):
        if not getattr(
            self,
            "object_placements",
            None,
        ):
            return

        if (
            "rolling_pin"
            not in self.object_placements
        ):
            return

        (
            surface_z,
            bounds,
        ) = self._island_surface_and_bounds()

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
        ) = self._compute_c1_work_origin(
            bounds
        )

        # -------------------------------------------------------------
        # Packing box
        # -------------------------------------------------------------

        box_center = np.array(
            [
                work_origin[0]
                + 0.03,
                work_origin[1]
                + 0.2,
            ],
            dtype=float,
        )

        # -------------------------------------------------------------
        # Lid
        # -------------------------------------------------------------

        lid_center = np.array(
            [
                work_origin[0]
                + 0.03,
                work_origin[1]
                - 0.1,
            ],
            dtype=float,
        )

        # Static packing box is appended in _load_model().
        self._pending_packing_box_spec = (
            box_center.copy(),
            float(surface_z),
        )

        # -------------------------------------------------------------
        # Packing objects
        # -------------------------------------------------------------

        for (
            name,
            y_offset,
        ) in zip(
            _PACKING_OBJECTS,
            _ROW_Y_OFFSETS,
        ):
            obj = self.objects[name]

            if name == "whisk":
                yaw = -np.pi / 2      # 기존 +90도에서 180도 회전 → -90도
            elif name in _LONG_OBJECTS:
                yaw = np.pi / 2
            else:
                yaw = 0.0
            self._set_placement(
                name,
                (
                    float(
                        work_origin[0]
                        + _ROW_X_OFFSET
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
                    yaw
                ),
            )

        # -------------------------------------------------------------
        # Lid placement
        # -------------------------------------------------------------

        lid = self.objects["lid"]

        self._set_placement(
            "lid",
            (
                float(
                    lid_center[0]
                ),
                float(
                    lid_center[1]
                ),
                self._object_place_z(
                    lid,
                    surface_z,
                ),
            ),
            self._yaw_quat_wxyz(
                0.0
            ),
        )

        # -------------------------------------------------------------
        # EE rack
        # -------------------------------------------------------------

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
            for (
                ee_id,
                xy,
            ) in self._rack_layout().items()
        }

        # -------------------------------------------------------------
        # Robot + pedestal + rack
        # -------------------------------------------------------------

        self._install_robot_and_rack(
            surface_z,
            work_origin,
            robot_xy,
            robot_yaw,
            rack_layout,
        )

        # Interior bottom reference.
        self._packing_box_center = (
            np.array(
                [
                    box_center[0],
                    box_center[1],
                    (
                        surface_z
                        + BOX_FLOOR_THICKNESS
                    ),
                ],
                dtype=float,
            )
        )

        self._layout_meta = {
            "island_name": (
                self.island.name
            ),
            "surface_z": surface_z,
            "bounds": bounds,
            "work_origin_xy": (
                work_origin
            ),
            "robot_xy": robot_xy,
            "robot_yaw": robot_yaw,
            "rack_layout": (
                rack_layout
            ),
            "packing_box_center": (
                self._packing_box_center
            ),
            "box_inner_size_m": (
                BOX_INNER_W,
                BOX_INNER_D,
                BOX_INNER_H,
            ),
            "box_diagonal_m": (
                BOX_DIAGONAL
            ),
            "object_fit": (
                _fit_metadata()
            ),
            "object_lengths_m": dict(
                OBJECT_LENGTHS
            ),
        }

    # =================================================================
    # Robot + EE rack
    # =================================================================

    @staticmethod
    def _rack_layout():
        return EE_RACK_LAYOUT

    def _remove_existing_ee_rack(
        self
    ):
        for body in list(
            self.model.worldbody.findall(
                "body"
            )
        ):
            name = (
                body.get("name") or ""
            )

            if (
                name == "ee_rack"
                or name.startswith(
                    "robot0_rack_"
                )
            ):
                self.model.worldbody.remove(
                    body
                )

    def _install_robot_and_rack(
        self,
        surface_z,
        work_origin,
        robot_xy,
        robot_yaw,
        rack_layout,
    ):
        """Install pedestal, robot base, and EE rack."""

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

        for (
            ee_id,
            info,
        ) in rack_info.items():
            self.ee_catalog[
                ee_id
            ].update(info)

        self.ee_rack_info = (
            rack_info
        )

    # =================================================================
    # Simulation references
    # =================================================================

    def _setup_references(self):
        super()._setup_references()

        if self.ee_rack_info:
            self.ee_rack_body_id = (
                self.sim.model.body_name2id(
                    "ee_rack"
                )
            )

    # =================================================================
    # Reset
    # =================================================================

    def _reapply_object_poses(
        self
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
                        np.asarray(
                            obj_pos
                        ),
                        np.asarray(
                            obj_quat
                        ),
                    ]
                ),
            )

        self.sim.forward()

    def _reset_internal(self):
        # Same reset strategy used by the existing C4 environment,
        # now implemented independently.
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

    # =================================================================
    # Task success
    # =================================================================

    def _check_success(self):
        # Scene-only benchmark for now.
        return False

    # =================================================================
    # Evaluation-only Material GT
    # =================================================================

    def get_evaluation_material_gt(self):
        """Evaluation-only GT; never used by observations or metadata."""

        names = (
            *_PACKING_OBJECTS,
            "lid",
        )

        return {
            name: self.objects[
                name
            ].material_gt
            for name in names
        }