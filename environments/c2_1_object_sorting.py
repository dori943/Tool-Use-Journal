"""C2-1 다물체 분류 환경.

테이블 위 물체를 지정 트레이로 옮긴다.
음식→초록, 음료 용기→파랑, 식기류→빨강.

씬: UR5e, EE 랙(3F/Vacuum/2F), 사과/빵/머그/접시/숟가락, 고정 트레이 3개.
EE 랙 배치는 C1-1과 같고, 위치는 고정이다.

"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robosuite.environments.manipulation.manipulation_env import (
    ManipulationEnv,
)

from robosuite.models.arenas import TableArena

from robosuite.models.tasks import ManipulationTask

from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)

from environments.ee_rack import add_ee_rack
from environments.objects import (
    AppleObject,
    BreadObject,
    MugObject,
    PlateObject,
    SpoonObject,
    TrayObject,
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


# C1-1과 같은 EE 랙 배치.


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


ROBOT_SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "robot_spec.json"
)


# 로봇은 -X. 앞줄 트레이(녹/파/빨), 뒷줄 물체.
TRAY_X = -0.43

TRAY_Y_OFFSET = 0.2

TRAY_SPACING = 0.3


GREEN_TRAY_POS = (
    TRAY_X,
    -TRAY_SPACING + TRAY_Y_OFFSET,
)

BLUE_TRAY_POS = (
    TRAY_X,
    TRAY_Y_OFFSET,
)

RED_TRAY_POS = (
    TRAY_X,
    TRAY_SPACING + TRAY_Y_OFFSET,
)
OBJECT_X = -0.04
OBJECT_Y_OFFSET = 0.10

APPLE_POS = (
    OBJECT_X,
    -0.28 + OBJECT_Y_OFFSET,
)

BREAD_POS = (
    OBJECT_X,
    -0.10 + OBJECT_Y_OFFSET,
)

MUG_POS = (
    OBJECT_X,
    0.08 + OBJECT_Y_OFFSET,
)

PLATE_POS = (
    OBJECT_X,
    0.25 + OBJECT_Y_OFFSET,
)

SPOON_POS = (
    OBJECT_X,
    0.38 + OBJECT_Y_OFFSET,
)


APPLE_ROTATION = 0.0
BREAD_ROTATION = 0.0
MUG_ROTATION = 0.0
PLATE_ROTATION = 0.0
SPOON_ROTATION = -np.pi / 2.0



GREEN_TRAY_RGBA = [
    0.15,
    0.80,
    0.25,
    0.40,
]

BLUE_TRAY_RGBA = [
    0.15,
    0.40,
    0.90,
    0.40,
]

RED_TRAY_RGBA = [
    0.90,
    0.20,
    0.20,
    0.40,
]


CATEGORY_TO_TRAY = {
    "food": "green_tray",
    "drink_container": "blue_tray",
    "food_container": "red_tray",
}


OBJECT_CATEGORY = {
    "apple": "food",
    "bread": "food",
    "mug": "drink_container",
    "plate": "food_container",
    "spoon": "food_container",
}



class C2_1_ObjectSorting(ManipulationEnv):
    """사과·빵→초록, 머그→파랑, 접시·숟가락→빨강."""

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
        render_camera="frontview",
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
    ):

        self.table_full_size = (
            table_full_size
        )

        self.table_friction = (
            table_friction
        )

        self.table_offset = (
            TABLE_OFFSET.copy()
        )


        with ROBOT_SPEC_PATH.open(
            encoding="utf-8"
        ) as spec_file:

            self.robot_spec = json.load(
                spec_file
            )


        self.current_ee_id = (
            self.robot_spec[
                "current_ee"
            ]
        )


        self.ee_swap_cost_s = float(
            self.robot_spec[
                "ee_swap_cost_s"
            ]
        )


        self.ee_catalog = {
            entry["ee_id"]: dict(entry)
            for entry
            in self.robot_spec[
                "ee_pool"
            ]
        }



        self.use_object_obs = (
            use_object_obs
        )

        self.placement_initializer = (
            placement_initializer
        )


        self.category_to_tray = dict(
            CATEGORY_TO_TRAY
        )

        self.object_category = dict(
            OBJECT_CATEGORY
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
            table_full_size=(
                self.table_full_size
            ),

            table_friction=(
                self.table_friction
            ),

            table_offset=(
                self.table_offset
            ),
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

            table_offset=(
                self.table_offset
            ),

            rack_layout=(
                EE_RACK_LAYOUT
            ),

            ee_pool=(
                self.robot_spec[
                    "ee_pool"
                ]
            ),
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


        self.apple = AppleObject(
            name="apple",
        )


        self.bread = BreadObject(
            name="bread"
        )


        self.mug = MugObject(
            name="mug"
        )


        self.plate = PlateObject(
            name="plate",
            xml_name="model_c2.xml",
        )


        self.spoon = SpoonObject(
            name="spoon",
        )


        self.target_objects = [
            self.apple,
            self.bread,
            self.mug,
            self.plate,
            self.spoon,
        ]





        TRAY_SCALE = 0.4
        self.green_tray = TrayObject(
            name="green_tray",
scale=TRAY_SCALE,
            rgba=(
                GREEN_TRAY_RGBA
            ),

            joints=None,
        )


        self.blue_tray = TrayObject(
            name="blue_tray",
scale=TRAY_SCALE,
            rgba=(
                BLUE_TRAY_RGBA
            ),

            joints=None,
        )


        self.red_tray = TrayObject(
            name="red_tray",
scale=TRAY_SCALE,
            rgba=(
                RED_TRAY_RGBA
            ),

            joints=None,
        )


        self.trays = [
            self.green_tray,
            self.blue_tray,
            self.red_tray,
        ]




        self._setup_placement_initializer()


        all_objects = (
            self.target_objects
            + self.trays
        )


        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,

            mujoco_robots=[
                robot.robot_model
                for robot
                in self.robots
            ],

            mujoco_objects=(
                all_objects
            ),
        )


    def _setup_placement_initializer(
        self,
    ):

        if (
            self.placement_initializer
            is None
        ):

            self.placement_initializer = (
                SequentialCompositeSampler(
                    name=(
                        "C2_1_"
                        "PlacementInitializer"
                    )
                )
            )


            tray_configs = [

                (
                    "GreenTraySampler",
                    GREEN_TRAY_POS,
                ),

                (
                    "BlueTraySampler",
                    BLUE_TRAY_POS,
                ),

                (
                    "RedTraySampler",
                    RED_TRAY_POS,
                ),
            ]


            for (
                sampler_name,
                position,
            ) in tray_configs:

                self.placement_initializer.append_sampler(

                    sampler=UniformRandomSampler(

                        name=sampler_name,

                        x_range=[
                            position[0],
                            position[0],
                        ],

                        y_range=[
                            position[1],
                            position[1],
                        ],

                        rotation=np.pi / 2.0,

                        rotation_axis="z",

                        ensure_object_boundary_in_range=False,

                        ensure_valid_placement=False,

                        reference_pos=(
                            self.table_offset
                        ),

                        z_offset=0.001,

                        rng=self.rng,
                    )
                )


            self.placement_initializer.append_sampler(

                sampler=UniformRandomSampler(

                    name="AppleSampler",

                    x_range=[
                        APPLE_POS[0],
                        APPLE_POS[0],
                    ],

                    y_range=[
                        APPLE_POS[1],
                        APPLE_POS[1],
                    ],

                    rotation=(
                        APPLE_ROTATION
                    ),

                    rotation_axis="z",

                    ensure_object_boundary_in_range=False,

                    ensure_valid_placement=False,

                    reference_pos=(
                        self.table_offset
                    ),

                    z_offset=0.01,

                    rng=self.rng,
                )
            )


            self.placement_initializer.append_sampler(

                sampler=UniformRandomSampler(

                    name="BreadSampler",

                    x_range=[
                        BREAD_POS[0],
                        BREAD_POS[0],
                    ],

                    y_range=[
                        BREAD_POS[1],
                        BREAD_POS[1],
                    ],

                    rotation=(
                        BREAD_ROTATION
                    ),

                    rotation_axis="z",

                    ensure_object_boundary_in_range=False,

                    ensure_valid_placement=False,

                    reference_pos=(
                        self.table_offset
                    ),

                    z_offset=0.01,

                    rng=self.rng,
                )
            )


            self.placement_initializer.append_sampler(

                sampler=UniformRandomSampler(

                    name="MugSampler",

                    x_range=[
                        MUG_POS[0],
                        MUG_POS[0],
                    ],

                    y_range=[
                        MUG_POS[1],
                        MUG_POS[1],
                    ],

                    rotation=(
                        MUG_ROTATION
                    ),

                    rotation_axis="z",

                    ensure_object_boundary_in_range=False,

                    ensure_valid_placement=False,

                    reference_pos=(
                        self.table_offset
                    ),

                    z_offset=0.01,

                    rng=self.rng,
                )
            )


            self.placement_initializer.append_sampler(

                sampler=UniformRandomSampler(

                    name="PlateSampler",

                    x_range=[
                        PLATE_POS[0],
                        PLATE_POS[0],
                    ],

                    y_range=[
                        PLATE_POS[1],
                        PLATE_POS[1],
                    ],

                    rotation=(
                        PLATE_ROTATION
                    ),

                    rotation_axis="z",

                    ensure_object_boundary_in_range=False,

                    ensure_valid_placement=False,

                    reference_pos=(
                        self.table_offset
                    ),

                    z_offset=0.01,

                    rng=self.rng,
                )
            )


            self.placement_initializer.append_sampler(

                sampler=UniformRandomSampler(

                    name="SpoonSampler",

                    x_range=[
                        SPOON_POS[0],
                        SPOON_POS[0],
                    ],

                    y_range=[
                        SPOON_POS[1],
                        SPOON_POS[1],
                    ],

                    rotation=(
                        SPOON_ROTATION
                    ),

                    rotation_axis="z",

                    ensure_object_boundary_in_range=False,

                    ensure_valid_placement=False,

                    reference_pos=(
                        self.table_offset
                    ),

                    z_offset=0.01,

                    rng=self.rng,
                )
            )


        self.placement_initializer.reset()


        self.placement_initializer.add_objects_to_sampler(
            "GreenTraySampler",
            self.green_tray,
        )


        self.placement_initializer.add_objects_to_sampler(
            "BlueTraySampler",
            self.blue_tray,
        )


        self.placement_initializer.add_objects_to_sampler(
            "RedTraySampler",
            self.red_tray,
        )


        self.placement_initializer.add_objects_to_sampler(
            "AppleSampler",
            self.apple,
        )


        self.placement_initializer.add_objects_to_sampler(
            "BreadSampler",
            self.bread,
        )


        self.placement_initializer.add_objects_to_sampler(
            "MugSampler",
            self.mug,
        )


        self.placement_initializer.add_objects_to_sampler(
            "PlateSampler",
            self.plate,
        )


        self.placement_initializer.add_objects_to_sampler(
            "SpoonSampler",
            self.spoon,
        )


    def _setup_references(
        self,
    ):

        super()._setup_references()

        self.obj_body_id = {}

        for obj in self.target_objects:

            self.obj_body_id[
                obj.name
            ] = (
                self.sim.model.body_name2id(
                    obj.root_body
                )
            )

        for tray in self.trays:

            self.obj_body_id[
                tray.name
            ] = (
                self.sim.model.body_name2id(
                    tray.root_body
                )
            )

        self.ee_rack_body_id = (
            self.sim.model.body_name2id(
                "ee_rack"
            )
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


            

                # 이동 물체는 joint qpos, 고정 물체(트레이)는 body pose.
                if obj.joints:

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


                else:

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


            self.sim.forward()


    def get_target_tray_name(
        self,
        object_name,
    ):

        if (
            object_name
            not in self.object_category
        ):

            raise ValueError(
                "Unknown target object: "
                f"{object_name}"
            )


        category = (
            self.object_category[
                object_name
            ]
        )


        return (
            self.category_to_tray[
                category
            ]
        )

    def get_evaluation_material_gt(self):
        """평가 전용 GT. 관측 및 LLM/M2 입력 경로에서는 호출하지 않는다."""
        return {
            obj.name: obj.material_gt
            for obj in self.target_objects
        }


    
