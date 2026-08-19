"""C1-1 레고 스윕 환경.

도구를 골라 흩어진 레고형 블록 12개를 수집 구역으로 쓸어 넣는다.

씬: UR5e, EE 랙(3F/Vacuum/2F), 블록 12개, 가벼운/무거운 접시, 병(distractor), 수집 구역.
EE 랙과 도구 영역은 떨어져 있고, 수집 구역은 로봇과 블록 사이에 있다.
두 접시는 형상·마찰이 같고 질량만 다르다(0.20kg / 0.80kg).

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
from environments.objects import BottleObject, PlateObject


NUM_BLOCKS = 12

# BoxObject는 half extent. 전체 크기 20 × 20 × 12 mm, 밀도 500 kg/m³ → 질량 ≈ 0.0024 kg
BLOCK_HALF_SIZE = [
    0.01,
    0.01,
    0.006,
]

BLOCK_DENSITY = 500.0

# 같은 접시 메시. 형상은 그대로 두고 컴파일된 질량만 다르게 둔다.
LIGHT_PLATE_MASS = 0.20
HEAVY_PLATE_MASS = 0.80

# MuJoCo friction: (sliding, torsional, rolling). 두 접시 동일 → 질량만 차별 요인.
PLATE_FRICTION = (
    0.8,
    0.005,
    0.0001,
)

# EE 선택용 추론 메타데이터. MuJoCo가 해석하지 않는다.
# VacuumGripper flatness_tol_rms_mm=1.5 → 두 접시 모두 평탄도 통과.
# payload: 가벼움 0.20kg 통과, 무거움 0.80kg 실패(한도 0.50kg).
TOOL_PHYSICAL_METADATA = {

    "light_plate": {
        "object_type": "sweep_tool",
        "material": "rigid_plastic",
        "surface": "flat",
        "flatness_rms_mm": 0.5,
        "mass_kg": LIGHT_PLATE_MASS,
        "friction": PLATE_FRICTION,
        "full_size_mm": [
            181.8,
            181.8,
            11.1,
        ],
    },

    "heavy_plate": {
        "object_type": "sweep_tool",
        "material": "rigid_plastic",
        "surface": "flat",
        "flatness_rms_mm": 0.5,
        "mass_kg": HEAVY_PLATE_MASS,
        "friction": PLATE_FRICTION,
        "full_size_mm": [
            181.8,
            181.8,
            11.1,
        ],
    },
}


# 수집 구역 전체 크기 250 × 180 mm. 로봇과 블록 사이.
COLLECTION_ZONE_SIZE = (
    0.25,
    0.18,
)

COLLECTION_ZONE_CENTER = (
    -0.02,
    0.0,
)

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


ROBOT_BASE_X = -0.68


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


TOOL_X = -0.3

LIGHT_PLATE_Y = -0.10

HEAVY_PLATE_Y = 0.17

BOTTLE_Y = 0.39


ROBOT_SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "robot_spec.json"
)


BLOCK_COLORS = [
    [0.85, 0.15, 0.15, 1.0],
    [0.15, 0.55, 0.85, 1.0],
    [0.20, 0.75, 0.25, 1.0],
    [0.95, 0.75, 0.10, 1.0],
    [0.55, 0.25, 0.70, 1.0],
    [0.95, 0.45, 0.10, 1.0],
]


class C1_1_LegoSweep(ManipulationEnv):
    """레고 블록 12개를 수집 구역으로 넣는 씬. 물리 속성과 도구 메타데이터만 구성한다."""

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


        with ROBOT_SPEC_PATH.open(
            encoding="utf-8"
        ) as spec_file:

            self.robot_spec = json.load(
                spec_file
            )


      

        self.ee_catalog = {
            entry["ee_id"]: dict(entry)
            for entry
            in self.robot_spec[
                "ee_pool"
            ]
        }


        self.tool_physical_metadata = {
            tool_name: dict(metadata)
            for tool_name, metadata
            in TOOL_PHYSICAL_METADATA.items()
        }


        self.use_object_obs = (
            use_object_obs
        )

        self.placement_initializer = (
            placement_initializer
        )


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



    def _load_model(
        self,
    ):

        super()._load_model()


        self.robots[
            0
        ].robot_model.set_base_xpos(
            (
                ROBOT_BASE_X,
                0.0,
                0.0,
            )
        )


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


        rack_info = add_ee_rack(
            arena=mujoco_arena,
            table_offset=self.table_offset,
            rack_layout=EE_RACK_LAYOUT,
            ee_pool=self.robot_spec[
                "ee_pool"
            ],
        )


        for ee_id, info in (
            rack_info.items()
        ):

            self.ee_catalog[
                ee_id
            ].update(
                info
            )


        self.ee_rack_info = (
            rack_info
        )


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
                    i % len(
                        BLOCK_COLORS
                    )
                ],

                density=BLOCK_DENSITY,
            )


            self.blocks.append(
                block
            )




        self.light_plate = PlateObject(
            name="light_plate",
        )


        self.heavy_plate = PlateObject(
            name="heavy_plate",
        )


        self.bottle_distractor = (
            BottleObject(
                name="bottle_distractor"
            )
        )


        zone_half_x = (
            self.collection_zone_size[
                0
            ]
            / 2.0
        )

        zone_half_y = (
            self.collection_zone_size[
                1
            ]
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


        self._setup_placement_initializer()


        collision_objects = (
            self.blocks
            + [
                self.light_plate,
                self.heavy_plate,
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
                    name=(
                        "C1_1_"
                        "PlacementInitializer"
                    )
                )
            )


            self.placement_initializer.append_sampler(

                sampler=UniformRandomSampler(

                    name="BlockSampler",

                    x_range=[
                        0.13,
                        0.28,
                    ],

                    y_range=[
                        -0.34,
                        0.34,
                    ],

                    rotation=None,

                    rotation_axis="z",

                    ensure_object_boundary_in_range=True,

                    ensure_valid_placement=True,

                    reference_pos=self.table_offset,

                    z_offset=0.01,

                    rng=self.rng,
                )
            )


            self.placement_initializer.append_sampler(

                sampler=UniformRandomSampler(

                    name="LightPlateSampler",

                    x_range=[
                        TOOL_X,
                        TOOL_X,
                    ],

                    y_range=[
                        LIGHT_PLATE_Y,
                        LIGHT_PLATE_Y,
                    ],

                    rotation=0.0,

                    rotation_axis="z",

                    ensure_object_boundary_in_range=False,

                    ensure_valid_placement=True,

                    reference_pos=self.table_offset,

                    z_offset=0.0,

                    rng=self.rng,
                )
            )


            self.placement_initializer.append_sampler(

                sampler=UniformRandomSampler(

                    name="HeavyPlateSampler",

                    x_range=[
                        TOOL_X,
                        TOOL_X,
                    ],

                    y_range=[
                        HEAVY_PLATE_Y,
                        HEAVY_PLATE_Y,
                    ],

                    rotation=0.0,

                    rotation_axis="z",

                    ensure_object_boundary_in_range=False,

                    ensure_valid_placement=True,

                    reference_pos=self.table_offset,

                    z_offset=0.0,

                    rng=self.rng,
                )
            )


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

                    ensure_valid_placement=True,

                    reference_pos=self.table_offset,

                    z_offset=0.0,

                    rng=self.rng,
                )
            )


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


        self.placement_initializer.reset()


        self.placement_initializer.add_objects_to_sampler(
            "BlockSampler",
            self.blocks,
        )


        self.placement_initializer.add_objects_to_sampler(
            "LightPlateSampler",
            self.light_plate,
        )


        self.placement_initializer.add_objects_to_sampler(
            "HeavyPlateSampler",
            self.heavy_plate,
        )


        self.placement_initializer.add_objects_to_sampler(
            "BottleSampler",
            self.bottle_distractor,
        )


        self.placement_initializer.add_objects_to_sampler(
            "CollectionZoneSampler",
            self.collection_zone_visual,
        )


    def _setup_references(
        self,
    ):

        super()._setup_references()

        self.obj_body_id = {}

        all_objects = (
            self.blocks
            + [
                self.light_plate,
                self.heavy_plate,
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

        self._set_object_mass(
            self.light_plate,
            LIGHT_PLATE_MASS,
        )

        self._set_object_mass(
            self.heavy_plate,
            HEAVY_PLATE_MASS,
        )


    def _set_object_mass(
        self,
        obj,
        target_mass,
    ):
        """컴파일된 body 서브트리 질량을 target_mass에 맞춰 스케일한다."""
        root_body_id = self.obj_body_id[obj.name]
        model = self.sim.model

        body_ids = []
        current_mass = 0.0

        for body_id in range(model.nbody):
            current = body_id
            while current != 0:
                if current == root_body_id:
                    body_ids.append(body_id)
                    current_mass += float(model.body_mass[body_id])
                    break
                current = int(model.body_parentid[current])

        if current_mass <= 0.0:
            raise RuntimeError(
                f"Cannot set mass for {obj.name}: compiled mass is 0."
            )

        scale = float(target_mass) / current_mass

        for body_id in body_ids:
            model.body_mass[body_id] = (
                float(model.body_mass[body_id]) * scale
            )
            model.body_inertia[body_id] = (
                np.asarray(model.body_inertia[body_id], dtype=float)
                * scale
            )


    def get_tool_physical_metadata(
        self,
        tool_name,
    ):
        """스윕 도구의 물리/추론 메타데이터를 반환한다."""

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


    def _reset_internal(
        self,
    ):

        super()._reset_internal()


        if not (
            self.deterministic_reset
        ):

            object_placements = (
                self.placement_initializer.sample()
            )


            for (
                obj_pos,
                obj_quat,
                obj,
            ) in (
                object_placements.values()
            ):


                # 시각 전용은 joint가 없어 body pose를 직접 쓴다.
                if (
                    "visual"
                    in obj.name.lower()
                ):

                    body_id = (
                        self.obj_body_id[
                            obj.name
                        ]
                    )


                    self.sim.model.body_pos[
                        body_id
                    ] = (
                        obj_pos
                    )


                    self.sim.model.body_quat[
                        body_id
                    ] = (
                        obj_quat
                    )


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


 