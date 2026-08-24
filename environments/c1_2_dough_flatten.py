"""C1-T2 Dough Flatten scene.

Kitchen Island (Layout004 / Style002) work area + C1-T1 UR5e / EE rack
equal-radius layout + separate robot pedestal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import robosuite.utils.transform_utils as T
from robocasa.models.fixtures import FixtureType
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

from environments.c1_1_lego_sweep import EE_RACK_LAYOUT, ROBOT_BASE_X, TABLE_OFFSET
from environments.ee_rack import add_ee_rack
from environments.kitchen_base import KitchenBase
from environments.objects import (
    BottleObject,
    CuttingBoardObject,
    DoughObject,
    SpatulaObject,
    SpoonObject,
    TongsObject,
)
from environments.robot_pedestal import add_robot_pedestal, remove_robot_pedestal


ROBOT_SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "robot_spec.json"
)

# C1-T1: EE는 robot base 기준 동일 반경(0.70m) 호 위에 배치
EE_RACK_OFFSETS_FROM_ROBOT = {
    ee_id: (float(xy[0] - ROBOT_BASE_X), float(xy[1] - 0.0))
    for ee_id, xy in EE_RACK_LAYOUT.items()
}
EE_RACK_RADIUS = float(
    np.mean(
        [
            np.linalg.norm(np.array(offset, dtype=float))
            for offset in EE_RACK_OFFSETS_FROM_ROBOT.values()
        ]
    )
)
EE_DIST_TOL = 1e-3

# UR5e manufacturer reach (horizontal workspace radius from base)
UR5E_REACH_M = 0.85

# C1-T1 table height → pedestal top (arm mount height reference)
PEDESTAL_TOP_Z = float(TABLE_OFFSET[2])  # 0.8
PEDESTAL_HALF_XY = (0.25, 0.25)
_BOARD_FORWARD_OFFSET = -0.26
# Island work layout (world, robot faces +x like C1-T1)
# work_origin = cutting board XY; robot at work_origin + (ROBOT_BASE_X, 0)
_TOOL_ORDER = ("bottle", "spatula", "spoon", "tongs")
_TOOL_ROW_FORWARD = 0.0 # dough/board보다 +x (viewer에서 위쪽 한 줄)
_TOOL_SPACING = 0.12
# Bottle(+y) … Tongs(-y): robot이 +x를 볼 때 좌→우
_TOOL_Y_FROM_CENTER = (1.5, 0.5, -0.5, -1.5)

_UTENSIL_YAW = -np.pi / 2
_TOOL_ROTATIONS = {
    "bottle": 0.0,
    "spatula": _UTENSIL_YAW,
    "spoon": _UTENSIL_YAW,
    "tongs": _UTENSIL_YAW,
}


class C1_2_DoughFlatten(KitchenBase):
    """Dough flatten on Kitchen Island. UR5e on pedestal + C1-T1 EE arc."""

    def __init__(
        self,
        robots="UR5e",
        gripper_types=None,
        base_types="NullMount",
        initialization_noise="default",
        seed=0,
        **kwargs,
    ):
        # NullMount: C1-T1 EE/gripper 구조는 유지하고, 별도 pedestal 위에 base mount
        kwargs.setdefault("use_distractors", False)
        kwargs.setdefault("use_object_obs", True)
        kwargs.setdefault("robot_spawn_deviation_pos_x", 0.0)
        kwargs.setdefault("robot_spawn_deviation_pos_y", 0.0)
        kwargs.setdefault("robot_spawn_deviation_rot", 0.0)

        with ROBOT_SPEC_PATH.open(encoding="utf-8") as spec_file:
            self.robot_spec = json.load(spec_file)

        self.ee_catalog = {
            entry["ee_id"]: dict(entry) for entry in self.robot_spec["ee_pool"]
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
        # wall counter는 사용하지 않음 (호환용 alias)
        self.counter = self.island
        self.init_robot_base_ref = None

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "C1-T2 dough flatten: Kitchen Island, UR5e on pedestal, "
            "C1-T1 EE rack equal-radius layout."
        )
        return ep_meta

    def _island_surface_and_bounds(self):
        island = self.island
        region = island.sample_reset_region(env=self, full_depth_region=True)
        surface_z = float(island.pos[2] + region["offset"][2])
        pts = np.asarray(island.get_bbox_points(), dtype=float)
        bounds = {
            "xmin": float(pts[:, 0].min()),
            "xmax": float(pts[:, 0].max()),
            "ymin": float(pts[:, 1].min()),
            "ymax": float(pts[:, 1].max()),
            "zmin": float(pts[:, 2].min()),
            "zmax": float(pts[:, 2].max()),
            "center_xy": np.array(
                [0.5 * (pts[:, 0].min() + pts[:, 0].max()),
                 0.5 * (pts[:, 1].min() + pts[:, 1].max())],
                dtype=float,
            ),
        }
        return surface_z, bounds

    def _load_model(self, attempt_num=1):
        super()._load_model(attempt_num=attempt_num)
        self._apply_island_layout()

    def _create_objects(self):
        self.objects = {}
        self.object_cfgs = self._get_obj_cfgs()

        builders = {
            "cutting_board": CuttingBoardObject,
            "dough": DoughObject,
            "bottle": BottleObject,
            "spatula": SpatulaObject,
            "spoon": SpoonObject,
            "tongs": TongsObject,
        }

        for cfg in self.object_cfgs:
            cfg["type"] = "object"
            name = cfg["name"]
            model = builders[name](name=name)
            cfg["info"] = {
                "groups_containing_sampled_obj": ["all", name],
                "groups": [name],
                "cat": name,
                "mjcf_path": "",
            }
            self.objects[model.name] = model
            self.model.merge_objects([model])
            setattr(self, name, model)

    def _fixed_island_placement(self, pos, rotation=0.0):
        return dict(
            fixture=self.island,
            size=(0.0, 0.0),
            pos=pos,
            rotation=float(rotation),
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
            sample_region_kwargs=dict(full_depth_region=True),
        )

    def _get_obj_cfgs(self):
        # 초기 sampler용 placeholder — 최종 좌표는 _apply_island_layout에서 island 기준으로 재계산
        cfgs = [
            dict(
                name="cutting_board",
                placement=self._fixed_island_placement((0.0, 0.0)),
            ),
            dict(
                name="dough",
                placement=dict(
                    size=(0.0, 0.0),
                    rotation=0.0,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    sample_args=dict(reference="cutting_board"),
                ),
            ),
        ]
        for index, tool_name in enumerate(_TOOL_ORDER):
            placement = self._fixed_island_placement(
                (0.0, 0.0),
                rotation=_TOOL_ROTATIONS[tool_name],
            )
            placement["reuse_region_from"] = "cutting_board"
            placement["offset"] = (
                _TOOL_ROW_FORWARD,
                float(_TOOL_Y_FROM_CENTER[index] * _TOOL_SPACING),
            )
            cfgs.append(dict(name=tool_name, placement=placement))
        return cfgs

    def _remove_existing_ee_rack(self) -> None:
        worldbody = self.model.worldbody
        for body in list(worldbody.findall("body")):
            name = body.get("name") or ""
            if name == "ee_rack" or name.startswith("robot0_rack_"):
                worldbody.remove(body)

    def _object_place_z(self, obj, support_top_z: float) -> float:
        """bottom_offset을 반영해 support 위에 안착하는 joint z."""
        bottom = float(np.asarray(obj.bottom_offset)[-1])
        return float(support_top_z - bottom)

    def _apply_island_layout(self) -> None:
        if not getattr(self, "object_placements", None):
            return
        if "cutting_board" not in self.object_placements:
            return

        surface_z, bounds = self._island_surface_and_bounds()
        self._island_surface_z = surface_z
        self._island_bounds = bounds

        # C1-T1 상대배치를 island에 이식:
        #   robot = work + (ROBOT_BASE_X, 0)
        #   EE    = work + EE_RACK_LAYOUT
        # 동시에 robot/pedestal은 island 밖, EE는 island 위.
        work_y = float(bounds["center_xy"][1])
        min_layout_x = min(float(xy[0]) for xy in EE_RACK_LAYOUT.values())  # -0.33
        # EE가 island 안: work_x + min_layout_x >= xmin + edge
        # robot+pedestal이 island 밖: work_x + ROBOT_BASE_X + ped_half <= xmin - gap
        edge = 0.04
        gap = 0.05
        work_x_lo = float(bounds["xmin"] + edge - min_layout_x)
        work_x_hi = float(
            bounds["xmin"] - gap - PEDESTAL_HALF_XY[0] - ROBOT_BASE_X
        )
        if work_x_lo <= work_x_hi:
            work_x = 0.5 * (work_x_lo + work_x_hi)
        else:
            # 여유가 없으면 EE-on-island를 우선하고 pedestal half를 줄인 효과로 타협
            work_x = work_x_lo

        work_origin = np.array([work_x, work_y], dtype=float)
        robot_xy = work_origin + np.array([ROBOT_BASE_X, 0.0], dtype=float)
        robot_yaw = 0.0  # +x → island (C1-T1과 동일)

        # EE rack: C1-T1 EE_RACK_LAYOUT을 work_origin 기준으로 평행이동 (동일 반경 유지)
        rack_layout = {
            ee_id: (
                float(work_origin[0] + xy[0]),
                float(work_origin[1] + xy[1]),
            )
            for ee_id, xy in EE_RACK_LAYOUT.items()
        }

        # 객체 world 좌표 (island surface + bottom offset)
        board = self.objects["cutting_board"]
        dough = self.objects["dough"]
        board_z = self._object_place_z(board, surface_z)
        board_top_z = board_z + float(np.asarray(board.top_offset)[-1])
        dough_z = self._object_place_z(dough, board_top_z)

        board_pos = np.array([work_origin[0]+ _BOARD_FORWARD_OFFSET, work_origin[1], board_z], dtype=float)
        dough_pos = np.array([work_origin[0]+ _BOARD_FORWARD_OFFSET, work_origin[1], dough_z], dtype=float)
        board_quat = self._yaw_quat_wxyz(0.0)
        dough_quat = self._yaw_quat_wxyz(0.0)

        tool_positions = {}
        for index, name in enumerate(_TOOL_ORDER):
            obj = self.objects[name]
            pos = np.array(
                [
                    work_origin[0] + _TOOL_ROW_FORWARD,
                    work_origin[1]
                    + float(_TOOL_Y_FROM_CENTER[index] * _TOOL_SPACING),
                    self._object_place_z(obj, surface_z),
                ],
                dtype=float,
            )
            quat = self._yaw_quat_wxyz(float(_TOOL_ROTATIONS[name]))
            tool_positions[name] = (pos, quat)

        # object_placements 갱신 (reset에서 사용)
        self.object_placements["cutting_board"] = (
            tuple(board_pos.tolist()),
            board_quat,
            board,
        )
        self.object_placements["dough"] = (
            tuple(dough_pos.tolist()),
            dough_quat,
            dough,
        )
        for name, (pos, quat) in tool_positions.items():
            self.object_placements[name] = (
                tuple(pos.tolist()),
                quat,
                self.objects[name],
            )

        # Pedestal + robot (island 밖)
        remove_robot_pedestal(self.model)
        self.pedestal_info = add_robot_pedestal(
            self.model,
            center_xy=robot_xy,
            top_z=PEDESTAL_TOP_Z,
            half_size_xy=PEDESTAL_HALF_XY,
        )

        robot_model = self.robots[0].robot_model
        robot_model.set_base_xpos(
            [float(robot_xy[0]), float(robot_xy[1]), PEDESTAL_TOP_Z]
        )
        robot_model.set_base_ori([0.0, 0.0, float(robot_yaw)])

        self._work_origin_xy = work_origin
        self._robot_base_xy = robot_xy
        self._robot_base_yaw = robot_yaw

        # EE rack on island surface
        self._remove_existing_ee_rack()
        rack_info = add_ee_rack(
            arena=self.model,
            table_offset=np.array([0.0, 0.0, surface_z], dtype=float),
            rack_layout=rack_layout,
            ee_pool=self.robot_spec["ee_pool"],
        )
        for ee_id, info in rack_info.items():
            self.ee_catalog[ee_id].update(info)
        self.ee_rack_info = rack_info

        self._layout_meta = {
            "island_name": self.island.name,
            "island_type": type(self.island).__name__,
            "surface_z": surface_z,
            "bounds": bounds,
            "work_origin_xy": work_origin,
            "robot_xy": robot_xy,
            "robot_yaw": robot_yaw,
            "rack_layout": rack_layout,
            "ee_radius": EE_RACK_RADIUS,
            "pedestal_top_z": PEDESTAL_TOP_Z,
        }

    def _setup_references(self):
        super()._setup_references()
        if self.ee_rack_info:
            self.ee_rack_body_id = self.sim.model.body_name2id("ee_rack")

    def _yaw_quat_wxyz(self, yaw: float) -> np.ndarray:
        quat_xyzw = T.mat2quat(T.euler2mat(np.array([0.0, 0.0, float(yaw)])))
        return np.array(T.convert_quat(quat_xyzw, to="wxyz"), dtype=float)

    def _align_tools_on_row(self) -> None:
        """settle 후 tool row / orientation / 높이 재고정."""
        if self._work_origin_xy is None or self._island_surface_z is None:
            return
        work = self._work_origin_xy
        surface_z = self._island_surface_z
        for index, name in enumerate(_TOOL_ORDER):
            obj = self.objects[name]
            pos = np.array(
                [
                    work[0] + _TOOL_ROW_FORWARD,
                    work[1] + float(_TOOL_Y_FROM_CENTER[index] * _TOOL_SPACING),
                    self._object_place_z(obj, surface_z),
                ],
                dtype=float,
            )
            quat = self._yaw_quat_wxyz(float(_TOOL_ROTATIONS[name]))
            self.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate([pos, quat]),
            )
        # board / dough도 재고정
        board = self.objects["cutting_board"]
        dough = self.objects["dough"]
        board_z = self._object_place_z(board, surface_z)
        board_top = board_z + float(np.asarray(board.top_offset)[-1])
        self.sim.data.set_joint_qpos(
            board.joints[0],
            np.concatenate(
                [
                    np.array([work[0]+ _BOARD_FORWARD_OFFSET, work[1], board_z], dtype=float),
                    self._yaw_quat_wxyz(0.0),
                ]
            ),
        )
        self.sim.data.set_joint_qpos(
            dough.joints[0],
            np.concatenate(
                [
                    np.array(
                        [work[0]+ _BOARD_FORWARD_OFFSET, work[1], self._object_place_z(dough, board_top)],
                        dtype=float,
                    ),
                    self._yaw_quat_wxyz(0.0),
                ]
            ),
        )
        self.sim.forward()

    def _reset_internal(self):
        """Kitchen mobile-base spawn 대신 fixed UR5e on pedestal."""
        ManipulationEnv._reset_internal(self)
        self._setup_scene()

        if not self.deterministic_reset and self.placement_initializer is not None:
            object_placements = self.object_placements
            self._update_sliding_fxtr_obj_placement()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

        if self._robot_base_xy is not None:
            self.init_robot_base_pos = np.array(
                [
                    self._robot_base_xy[0],
                    self._robot_base_xy[1],
                    PEDESTAL_TOP_Z,
                ],
                dtype=float,
            )
            self.init_robot_base_ori = np.array(
                [0.0, 0.0, float(self._robot_base_yaw)],
                dtype=float,
            )

        action = np.zeros(self.action_spec[0].shape)
        policy_step = True
        for _ in range(10 * int(self.control_timestep / self.model_timestep)):
            self.sim.step1()
            self._pre_action(action, policy_step)
            self.sim.step2()
            policy_step = False

        self._align_tools_on_row()

    def _check_success(self):
        return False

    def get_evaluation_material_gt(self):
        """평가 전용 GT. 관측 및 LLM/M2 입력 경로에서는 호출하지 않는다."""
        names = ("dough", "bottle", "spatula", "spoon", "tongs")
        return {name: self.objects[name].material_gt for name in names}

    def get_object_poses(self):
        poses = {}
        for name in (
            "cutting_board",
            "dough",
            "bottle",
            "spatula",
            "spoon",
            "tongs",
        ):
            body_id = self.obj_body_id[name]
            poses[name] = {
                "position": np.array(self.sim.data.body_xpos[body_id], dtype=float),
                "quaternion_wxyz": np.array(
                    self.sim.data.body_xquat[body_id], dtype=float
                ),
            }

        if self.ee_rack_info:
            rack_body_id = self.sim.model.body_name2id("ee_rack")
            poses["ee_rack"] = {
                "position": np.array(
                    self.sim.data.body_xpos[rack_body_id], dtype=float
                ),
                "quaternion_wxyz": np.array(
                    self.sim.data.body_xquat[rack_body_id], dtype=float
                ),
            }
            for ee_id, info in self.ee_rack_info.items():
                body_name = info.get("rack_body")
                if not body_name:
                    continue
                body_id = self.sim.model.body_name2id(body_name)
                label = {"3F": "3F", "vac": "Vacuum", "2F": "2F"}[ee_id]
                poses[label] = {
                    "position": np.array(
                        self.sim.data.body_xpos[body_id], dtype=float
                    ),
                    "quaternion_wxyz": np.array(
                        self.sim.data.body_xquat[body_id], dtype=float
                    ),
                }

        if self.pedestal_info:
            try:
                ped_id = self.sim.model.body_name2id("robot_pedestal")
                poses["robot_pedestal"] = {
                    "position": np.array(
                        self.sim.data.body_xpos[ped_id], dtype=float
                    ),
                    "quaternion_wxyz": np.array(
                        self.sim.data.body_xquat[ped_id], dtype=float
                    ),
                }
            except Exception:
                pass
        return poses

    def get_object_positions(self):
        return {
            name: pose["position"] for name, pose in self.get_object_poses().items()
        }

    def get_reachability_report(self) -> dict:
        """robot base → targets 거리 vs UR5e reach."""
        base_id = self.sim.model.body_name2id("robot0_base")
        base = np.array(self.sim.data.body_xpos[base_id], dtype=float)
        poses = self.get_object_poses()
        targets = {
            "dough": poses["dough"]["position"],
            "bottle": poses["bottle"]["position"],
            "spatula": poses["spatula"]["position"],
            "spoon": poses["spoon"]["position"],
            "tongs": poses["tongs"]["position"],
            "2F": poses["2F"]["position"],
            "3F": poses["3F"]["position"],
            "Vacuum": poses["Vacuum"]["position"],
        }
        rows = {}
        for name, pos in targets.items():
            dist = float(np.linalg.norm(pos - base))
            # horizontal reach이 주 제약; 수직 포함 Euclidean과 XY 모두 기록
            dist_xy = float(np.linalg.norm(pos[:2] - base[:2]))
            rows[name] = {
                "distance_3d": dist,
                "distance_xy": dist_xy,
                "within_reach_xy": dist_xy <= UR5E_REACH_M,
                "within_reach_3d": dist <= UR5E_REACH_M,
            }

        ee_xy = {
            label: float(np.linalg.norm(poses[label]["position"][:2] - base[:2]))
            for label in ("2F", "3F", "Vacuum")
        }
        ee_vals = list(ee_xy.values())
        return {
            "robot_base": base,
            "reach_limit_m": UR5E_REACH_M,
            "ee_radius_c1": EE_RACK_RADIUS,
            "ee_dist_tol": EE_DIST_TOL,
            "targets": rows,
            "ee_xy_distances": ee_xy,
            "ee_xy_max_delta": float(max(ee_vals) - min(ee_vals)),
            "all_within_reach_xy": all(r["within_reach_xy"] for r in rows.values()),
        }
