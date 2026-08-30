"""C1-1 레고 스윕 환경.

도구를 골라 흩어진 레고형 블록 12개를 수집 구역으로 쓸어 넣는다.

씬:
- UR5e
- EE 랙 (3F / Vacuum / 2F)
- 레고 블록 12개
- Plate
- 국자 / 포크 / 가위
- 병
- 수집 구역

EE 랙과 도구 영역은 떨어져 있고,
수집 구역은 로봇과 블록 사이에 있다.

Plate 질량은 0.20 kg이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robosuite.environments.manipulation.manipulation_env import (
    ManipulationEnv,
)
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)

from environments.ee_rack import add_ee_rack
from environments.objects import (
    BottleObject,
    ForkObject,
    LadleObject,
    PlateObject,
    ScissorsObject,
)


# ============================================================
# 기본 설정
# ============================================================

NUM_BLOCKS = 12


# BoxObject는 half extent.
# 전체 크기:
# 20 × 20 × 12 mm
#
# 밀도:
# 500 kg/m³
#
# 질량:
# 약 0.0024 kg
BLOCK_HALF_SIZE = [
    0.01,
    0.01,
    0.006,
]

BLOCK_DENSITY = 500.0


# ============================================================
# 레고 랜덤 배치 영역
# ============================================================
#
# 기존:
# X = 0.12 ~ 0.16
# Y = -0.10 ~ 0.10
#
# 12개를 non-overlap으로 놓기에는 너무 좁아서
# RandomizationError가 발생할 수 있으므로 약간 확대.
#
# 여전히 로봇에서 먼 쪽의 제한된 영역에만 생성된다.
# ============================================================

BLOCK_SPAWN_X_RANGE = (
    0.12,
    0.22,
)

BLOCK_SPAWN_Y_RANGE = (
    -0.15,
    0.15,
)


# ============================================================
# Plate 물리 속성
# ============================================================

PLATE_MASS = 0.20


# MuJoCo friction:
# (sliding, torsional, rolling)
PLATE_FRICTION = (
    0.8,
    0.005,
    0.0001,
)


# ============================================================
# EE 선택용 물리 메타데이터
# ============================================================

TOOL_PHYSICAL_METADATA = {

    "plate": {
        "object_type": "sweep_tool",
        "surface": "flat",
        "flatness_rms_mm": 0.5,
        "mass_kg": PLATE_MASS,
        "friction": PLATE_FRICTION,
        "full_size_mm": [
            181.8,
            181.8,
            11.1,
        ],
    },

    "ladle": {
        "object_type": "sweep_tool",
        "full_size_mm": [
            63.5,
            208.1,
            110.2,
        ],
    },

    "scissors": {
        "object_type": "sweep_tool",
        "full_size_mm": [
            81.1,
            179.2,
            11.8,
        ],
    },

    "fork": {
        "object_type": "sweep_tool",
        "full_size_mm": [
            35.7,
            219.5,
            26.4,
        ],
    },
}


# ============================================================
# 수집 구역
# ============================================================
#
# robot base = x -0.68
# tools      = x -0.45
# zone       = x -0.25
# lego       = x +0.12 ~ +0.22
#
# 즉:
#
# Robot → Tools → Collection Zone → Lego
#
# ============================================================

COLLECTION_ZONE_SIZE = (
    0.25,
    0.18,
)

COLLECTION_ZONE_CENTER = (
    -0.15,
    0.0,
)


# ============================================================
# 테이블
# ============================================================

DEFAULT_TABLE_FULL_SIZE = (
    1.30,
    1.60,
    0.05,
)

TABLE_OFFSET = np.array(
    [
        0.0,
        0.0,
        0.8,
    ],
    dtype=float,
)


# ============================================================
# 로봇
# ============================================================

ROBOT_BASE_X = -0.68


# ============================================================
# EE Rack
# ============================================================

EE_RACK_LAYOUT = {

    "3F": (
        -0.330000,
        -0.606218,
    ),

    "vac": (
        -0.185025,
        -0.494975,
    ),

    "2F": (
        -0.073782,
        -0.350000,
    ),
}


# ============================================================
# 도구 및 distractor 배치
# ============================================================
#
# 기존 -0.30보다 robot 쪽으로 당기되,
# -0.60처럼 테이블 가장자리에 너무 붙지 않도록 -0.45 사용.
# ============================================================

TOOL_X = -0.4


# ------------------------------------------------------------
# Plate
# ------------------------------------------------------------

PLATE_Y = -0.10


# ------------------------------------------------------------
# Ladle
# ------------------------------------------------------------

LADLE_X = TOOL_X
LADLE_Y = 0.08


# ------------------------------------------------------------
# Fork
# ------------------------------------------------------------

FORK_X = TOOL_X
FORK_Y = 0.20


# ------------------------------------------------------------
# Scissors
# ------------------------------------------------------------

SCISSORS_X = TOOL_X
SCISSORS_Y = 0.30


# 기존 90도 방향에서 180도 뒤집기
# 90 + 180 = 270도
UTENSIL_ROTATION = 3 * np.pi / 2.0


# ------------------------------------------------------------
# Bottle
# ------------------------------------------------------------

BOTTLE_Y = 0.39


# ============================================================
# Robot Spec
# ============================================================

ROBOT_SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "robot_spec.json"
)


# ============================================================
# 레고 색상
# ============================================================

BLOCK_COLORS = [
    [0.85, 0.15, 0.15, 1.0],
    [0.15, 0.55, 0.85, 1.0],
    [0.20, 0.75, 0.25, 1.0],
    [0.95, 0.75, 0.10, 1.0],
    [0.55, 0.25, 0.70, 1.0],
    [0.95, 0.45, 0.10, 1.0],
]


class C1_1_LegoSweep(ManipulationEnv):
    """레고 블록 12개를 수집 구역으로 넣는 C1-1 씬."""

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types=None,
        base_types="default",
        initialization_noise="default",
        table_full_size=DEFAULT_TABLE_FULL_SIZE,
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="ee_rack_sideview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        collection_zone_center=None,
        collection_zone_size=None,
    ):

        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = TABLE_OFFSET.copy()

        # ----------------------------------------------------
        # Robot spec
        # ----------------------------------------------------

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

        self.tool_physical_metadata = {
            tool_name: dict(metadata)
            for tool_name, metadata
            in TOOL_PHYSICAL_METADATA.items()
        }

        self.use_object_obs = use_object_obs

        self.placement_initializer = (
            placement_initializer
        )

        # ----------------------------------------------------
        # Collection zone
        # ----------------------------------------------------

        zone_center = (
            collection_zone_center
            if collection_zone_center is not None
            else COLLECTION_ZONE_CENTER
        )

        zone_size = (
            collection_zone_size
            if collection_zone_size is not None
            else COLLECTION_ZONE_SIZE
        )

        self.collection_zone_center = np.array(
            zone_center,
            dtype=float,
        )

        self.collection_zone_size = np.array(
            zone_size,
            dtype=float,
        )

        # ----------------------------------------------------
        # Parent
        # ----------------------------------------------------

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    # ========================================================
    # Model
    # ========================================================

    def _load_model(
        self,
    ):

        super()._load_model()

        # ----------------------------------------------------
        # Robot 위치
        # ----------------------------------------------------

        self.robots[
            0
        ].robot_model.set_base_xpos(
            (
                ROBOT_BASE_X,
                0.0,
                0.0,
            )
        )

        # ----------------------------------------------------
        # Arena
        # ----------------------------------------------------

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        mujoco_arena.set_origin(
            [
                0,
                0,
                0,
            ]
        )

        # ----------------------------------------------------
        # EE Rack
        # ----------------------------------------------------

        rack_info = add_ee_rack(
            arena=mujoco_arena,
            table_offset=self.table_offset,
            rack_layout=EE_RACK_LAYOUT,
            ee_pool=self.robot_spec["ee_pool"],
        )

        for ee_id, info in rack_info.items():
            self.ee_catalog[
                ee_id
            ].update(
                info
            )

        self.ee_rack_info = rack_info

        # ----------------------------------------------------
        # Lego blocks
        # ----------------------------------------------------

        self.blocks = []

        for i in range(
            NUM_BLOCKS
        ):

            name = (
                f"block_{i}"
            )

            block = BoxObject(
                name=name,
                size_min=BLOCK_HALF_SIZE,
                size_max=BLOCK_HALF_SIZE,
                rgba=BLOCK_COLORS[
                    i % len(BLOCK_COLORS)
                ],
                density=BLOCK_DENSITY,
            )

            self.blocks.append(
                block
            )

        # ----------------------------------------------------
        # Plate
        # ----------------------------------------------------

        self.plate = PlateObject(
            name="plate",
        )

        # ----------------------------------------------------
        # Utensils
        # ----------------------------------------------------

        self.ladle = LadleObject(
            name="ladle",
        )

        self.fork = ForkObject(
            name="fork",
        )

        self.scissors = ScissorsObject(
            name="scissors",
        )

        # ----------------------------------------------------
        # Bottle
        # ----------------------------------------------------

        self.bottle_distractor = (
            BottleObject(
                name="bottle_distractor"
            )
        )

        # ----------------------------------------------------
        # Collection zone
        # ----------------------------------------------------

        zone_half_x = (
            self.collection_zone_size[0]
            / 2.0
        )

        zone_half_y = (
            self.collection_zone_size[1]
            / 2.0
        )

        self.collection_zone_visual = (
            BoxObject(
                name="collection_zone_visual",

                size_min=[
                    zone_half_x,
                    zone_half_y,
                    0.001,
                ],

                size_max=[
                    zone_half_x,
                    zone_half_y,
                    0.001,
                ],

                rgba=[
                    0.20,
                    0.85,
                    0.35,
                    0.30,
                ],

                joints=None,

                obj_type="visual",
            )
        )

        # ----------------------------------------------------
        # Placement initializer
        # ----------------------------------------------------

        self._setup_placement_initializer()

        # ----------------------------------------------------
        # Mujoco objects
        # ----------------------------------------------------

        collision_objects = (
            self.blocks
            + [
                self.plate,
                self.ladle,
                self.fork,
                self.scissors,
                self.bottle_distractor,
            ]
        )

        all_objects = (
            collision_objects
            + [
                self.collection_zone_visual
            ]
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,

            mujoco_robots=[
                robot.robot_model
                for robot
                in self.robots
            ],

            mujoco_objects=all_objects,
        )

    # ========================================================
    # Placement
    # ========================================================

    def _setup_placement_initializer(
        self,
    ):

        if (
            self.placement_initializer
            is None
        ):

            cx, cy = (
                self.collection_zone_center
            )

            self.placement_initializer = (
                SequentialCompositeSampler(
                    name="C1_1_PlacementInitializer"
                )
            )

            # =================================================
            # Blocks
            # =================================================

            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="BlockSampler",

                    x_range=list(
                        BLOCK_SPAWN_X_RANGE
                    ),

                    y_range=list(
                        BLOCK_SPAWN_Y_RANGE
                    ),

                    rotation=None,

                    rotation_axis="z",

                    # 레고는 서로 겹치면 안 됨
                    ensure_object_boundary_in_range=True,

                    ensure_valid_placement=True,

                    reference_pos=self.table_offset,

                    z_offset=0.01,

                    rng=self.rng,
                )
            )

            # =================================================
            # Plate
            # =================================================

            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="PlateSampler",

                    x_range=[
                        TOOL_X,
                        TOOL_X,
                    ],

                    y_range=[
                        PLATE_Y,
                        PLATE_Y,
                    ],

                    rotation=0.0,

                    rotation_axis="z",

                    ensure_object_boundary_in_range=False,

                    # 고정 배치이므로 sampler 충돌 검사 비활성화
                    ensure_valid_placement=False,

                    reference_pos=self.table_offset,

                    z_offset=0.0,

                    rng=self.rng,
                )
            )

            # =================================================
            # Ladle
            # =================================================

            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="LadleSampler",

                    x_range=[
                        LADLE_X,
                        LADLE_X,
                    ],

                    y_range=[
                        LADLE_Y,
                        LADLE_Y,
                    ],

                    rotation=UTENSIL_ROTATION,

                    rotation_axis="z",

                    ensure_object_boundary_in_range=False,

                    ensure_valid_placement=False,

                    reference_pos=self.table_offset,

                    z_offset=0.0,

                    rng=self.rng,
                )
            )

            # =================================================
            # Fork
            # =================================================

            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="ForkSampler",

                    x_range=[
                        FORK_X,
                        FORK_X,
                    ],

                    y_range=[
                        FORK_Y,
                        FORK_Y,
                    ],

                    rotation=UTENSIL_ROTATION,

                    rotation_axis="z",

                    ensure_object_boundary_in_range=False,

                    ensure_valid_placement=False,

                    reference_pos=self.table_offset,

                    z_offset=0.0,

                    rng=self.rng,
                )
            )

            # =================================================
            # Scissors
            # =================================================

            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="ScissorsSampler",

                    x_range=[
                        SCISSORS_X,
                        SCISSORS_X,
                    ],

                    y_range=[
                        SCISSORS_Y,
                        SCISSORS_Y,
                    ],

                    rotation=UTENSIL_ROTATION,

                    rotation_axis="z",

                    ensure_object_boundary_in_range=False,

                    ensure_valid_placement=False,

                    reference_pos=self.table_offset,

                    z_offset=0.0,

                    rng=self.rng,
                )
            )

            # =================================================
            # Bottle
            # =================================================

            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="BottleSampler",

                    x_range=[
                        TOOL_X,
                        TOOL_X,
                    ],

                    y_range=[
                        BOTTLE_Y,
                        BOTTLE_Y,
                    ],

                    rotation=0.0,

                    rotation_axis="z",

                    ensure_object_boundary_in_range=False,

                    ensure_valid_placement=False,

                    reference_pos=self.table_offset,

                    z_offset=0.0,

                    rng=self.rng,
                )
            )

            # =================================================
            # Collection zone
            # =================================================

            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="CollectionZoneSampler",

                    x_range=[
                        cx,
                        cx,
                    ],

                    y_range=[
                        cy,
                        cy,
                    ],

                    rotation=0.0,

                    rotation_axis="z",

                    ensure_object_boundary_in_range=False,

                    ensure_valid_placement=False,

                    reference_pos=self.table_offset,

                    z_offset=0.002,

                    rng=self.rng,
                )
            )

        # ----------------------------------------------------
        # Sampler reset
        # ----------------------------------------------------

        self.placement_initializer.reset()

        # ----------------------------------------------------
        # Object 등록
        # ----------------------------------------------------

        self.placement_initializer.add_objects_to_sampler(
            "BlockSampler",
            self.blocks,
        )

        self.placement_initializer.add_objects_to_sampler(
            "PlateSampler",
            self.plate,
        )

        self.placement_initializer.add_objects_to_sampler(
            "LadleSampler",
            self.ladle,
        )

        self.placement_initializer.add_objects_to_sampler(
            "ForkSampler",
            self.fork,
        )

        self.placement_initializer.add_objects_to_sampler(
            "ScissorsSampler",
            self.scissors,
        )

        self.placement_initializer.add_objects_to_sampler(
            "BottleSampler",
            self.bottle_distractor,
        )

        self.placement_initializer.add_objects_to_sampler(
            "CollectionZoneSampler",
            self.collection_zone_visual,
        )

    # ========================================================
    # References
    # ========================================================

    def _setup_references(
        self,
    ):

        super()._setup_references()

        self.obj_body_id = {}

        all_objects = (
            self.blocks
            + [
                self.plate,
                self.ladle,
                self.fork,
                self.scissors,
                self.bottle_distractor,
                self.collection_zone_visual,
            ]
        )

        for obj in all_objects:

            self.obj_body_id[
                obj.name
            ] = (
                self.sim.model.body_name2id(
                    obj.root_body
                )
            )

        self.ee_rack_body_id = (
            self.sim.model.body_name2id(
                "ee_rack"
            )
        )

        # Plate mass
        self._set_object_mass(
            self.plate,
            PLATE_MASS,
        )

    # ========================================================
    # Mass setting
    # ========================================================

    def _set_object_mass(
        self,
        obj,
        target_mass,
    ):
        """컴파일된 body 서브트리 질량을 target_mass에 맞춰 스케일한다."""

        root_body_id = (
            self.obj_body_id[
                obj.name
            ]
        )

        model = self.sim.model

        body_ids = []
        current_mass = 0.0

        for body_id in range(
            model.nbody
        ):

            current = body_id

            while current != 0:

                if current == root_body_id:

                    body_ids.append(
                        body_id
                    )

                    current_mass += float(
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

        if current_mass <= 0.0:

            raise RuntimeError(
                f"Cannot set mass for {obj.name}: "
                "compiled mass is 0."
            )

        scale = (
            float(target_mass)
            / current_mass
        )

        for body_id in body_ids:

            model.body_mass[
                body_id
            ] = (
                float(
                    model.body_mass[
                        body_id
                    ]
                )
                * scale
            )

            model.body_inertia[
                body_id
            ] = (
                np.asarray(
                    model.body_inertia[
                        body_id
                    ],
                    dtype=float,
                )
                * scale
            )

    # ========================================================
    # Tool metadata
    # ========================================================

    def get_tool_physical_metadata(
        self,
        tool_name,
    ):
        """스윕 도구의 물리 / 추론 메타데이터를 반환한다."""

        if (
            tool_name
            not in self.tool_physical_metadata
        ):

            raise ValueError(
                f"Unknown sweep tool: {tool_name}"
            )

        return dict(
            self.tool_physical_metadata[
                tool_name
            ]
        )

    def get_evaluation_material_gt(self):
        """평가 전용 GT. 관측 및 LLM/M3 입력 경로에서는 호출하지 않는다."""

        objects = (
            self.plate,
            self.ladle,
            self.fork,
            self.scissors,
            self.bottle_distractor,
        )

        return {
            obj.name: obj.material_gt
            for obj in objects
        }

    # ========================================================
    # Reset
    # ========================================================

    def _reset_internal(
        self,
    ):

        super()._reset_internal()

        if not self.deterministic_reset:

            object_placements = (
                self.placement_initializer.sample()
            )

            for (
                obj_pos,
                obj_quat,
                obj,
            ) in object_placements.values():

                # --------------------------------------------
                # Visual object
                # --------------------------------------------

                if "visual" in obj.name.lower():

                    body_id = (
                        self.obj_body_id[
                            obj.name
                        ]
                    )

                    self.sim.model.body_pos[
                        body_id
                    ] = obj_pos

                    self.sim.model.body_quat[
                        body_id
                    ] = obj_quat

                # --------------------------------------------
                # Physical object
                # --------------------------------------------

                else:

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