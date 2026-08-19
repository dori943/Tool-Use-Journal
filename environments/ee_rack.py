"""EE 랙 구성.

3F(Jaco), vac(VacuumGripper), 2F(Robotiq85)를
고정된 랙 위치에 정적으로 배치한다.

Vacuum은 scripts/_vacuum.py 모델을 사용한다.

"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

from robosuite.models.grippers.jaco_three_finger_gripper import (
    JacoThreeFingerDexterousGripper,
)
from robosuite.models.grippers.robotiq_85_gripper import (
    Robotiq85Gripper,
)
from robosuite.utils.mjcf_utils import (
    array_to_string,
    new_body,
    new_geom,
)


# ------------------------------------------------------------------
# VacuumGripper import
# ------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SCRIPTS_DIR = (
    PROJECT_ROOT
    / "scripts"
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPTS_DIR),
    )

from _vacuum import register_vacuum


VacuumGripper = (
    register_vacuum()
)


# ------------------------------------------------------------------
# EE 구성
# ------------------------------------------------------------------

EE_GRIPPER_CLASSES = {
    "3F": JacoThreeFingerDexterousGripper,
    "vac": VacuumGripper,
    "2F": Robotiq85Gripper,
}

EE_DISPLAY_ORDER = (
    "3F",
    "vac",
    "2F",
)


# MuJoCo quaternion [w, x, y, z]
# X축 기준 180도 회전.
EE_DISPLAY_QUAT = np.array(
    [
        0.0,
        1.0,
        0.0,
        0.0,
    ],
    dtype=float,
)


# 테이블 표면 기준 EE 표시 높이.
EE_DISPLAY_HEIGHT = 0.235


# ------------------------------------------------------------------
# Rack geometry
# ------------------------------------------------------------------

# 테이블 표면 기준 지지대 상단 높이.
# EE 길이 차이를 지지대 높이로 보정한다.
EE_SUPPORT_TOP_Z = {
    "3F": 0.05,
    "vac": 0.15,
    "2F": 0.09,
}


# MuJoCo box는 half extent.
SUPPORT_HALF_SIZE_XY = {
    "3F": (
        0.070,
        0.085,
    ),
    "vac": (
        0.035,
        0.035,
    ),
    "2F": (
        0.075,
        0.075,
    ),
}


RACK_BASE_HALF_HEIGHT = 0.012

RACK_PADDING_X = 0.075
RACK_PADDING_Y = 0.075

SUPPORT_MIN_HEIGHT = 0.005


# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------

RACK_BASE_RGBA = [
    0.25,
    0.25,
    0.27,
    1.0,
]

SUPPORT_RGBA = [
    0.30,
    0.30,
    0.33,
    1.0,
]


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def freeze_gripper_for_display(
    gripper,
):
    """EE를 rack 전시용 정적 모델로 변환한다.

    joint / actuator / tendon 등을 제거하고
    collision을 비활성화한다.

    geom group은 기존 값을 유지하여
    원래 visual mesh 외형을 유지한다.
    """

    for body in gripper.worldbody.iter(
        "body"
    ):

        for joint in list(
            body.findall(
                "joint"
            )
        ):
            body.remove(
                joint
            )


    sections = (
        gripper.actuator,
        gripper.tendon,
        gripper.equality,
        gripper.sensor,
    )


    for section in sections:

        for child in list(
            section
        ):
            section.remove(
                child
            )


    for geom in gripper.worldbody.iter(
        "geom"
    ):

        geom.set(
            "contype",
            "0",
        )

        geom.set(
            "conaffinity",
            "0",
        )


def _get_ee_by_id(
    ee_pool,
):
    """robot_spec.json의 EE 정보를 id 기준으로 정리한다."""

    ee_by_id = {
        entry["ee_id"]: entry
        for entry in ee_pool
    }


    missing = [
        ee_id
        for ee_id in EE_DISPLAY_ORDER
        if ee_id not in ee_by_id
    ]


    if missing:
        raise ValueError(
            "robot_spec.json is missing EE definitions: "
            + ", ".join(
                missing
            )
        )


    return ee_by_id


def _get_ordered_slot_positions(
    rack_layout,
):
    """3F → vac → 2F 순서로 rack 위치를 반환한다."""

    return {
        ee_id: (
            float(
                rack_layout[
                    ee_id
                ][0]
            ),
            float(
                rack_layout[
                    ee_id
                ][1]
            ),
        )
        for ee_id in EE_DISPLAY_ORDER
    }


def _calculate_support_height(
    ee_id,
):
    """EE별 지지대 높이를 계산한다."""

    rack_top = (
        2.0
        * RACK_BASE_HALF_HEIGHT
    )


    support_top = float(
        EE_SUPPORT_TOP_Z[
            ee_id
        ]
    )


    support_height = (
        support_top
        - rack_top
    )


    return max(
        SUPPORT_MIN_HEIGHT,
        support_height,
    )


# ------------------------------------------------------------------
# Rack builder
# ------------------------------------------------------------------

def add_ee_rack(
    arena,
    table_offset,
    rack_layout,
    ee_pool,
):
    """TableArena에 EE 랙과 3개의 EE 모델을 추가한다.

    Parameters
    ----------
    arena
        robosuite TableArena.

    table_offset
        테이블 위치.

    rack_layout
        EE별 XY 위치.
        예:
        {
            "3F": (..., ...),
            "vac": (..., ...),
            "2F": (..., ...),
        }

    ee_pool
        robot_spec.json의 EE 정보.

    Returns
    -------
    dict
        EE별 기본 rack 메타데이터.
    """

    table_offset = np.asarray(
        table_offset,
        dtype=float,
    )


    table_z = float(
        table_offset[
            2
        ]
    )


    ee_by_id = (
        _get_ee_by_id(
            ee_pool
        )
    )


    ordered_layout = (
        _get_ordered_slot_positions(
            rack_layout
        )
    )


    # --------------------------------------------------------------
    # Rack base
    # --------------------------------------------------------------

    rack_xs = [
        xy[0]
        for xy in ordered_layout.values()
    ]


    rack_ys = [
        xy[1]
        for xy in ordered_layout.values()
    ]


    rack_center = np.array(
        [
            np.mean(
                rack_xs
            ),

            np.mean(
                rack_ys
            ),

            table_z,
        ],
        dtype=float,
    )


    span_x = max(
        0.08,

        (
            max(
                rack_xs
            )
            -
            min(
                rack_xs
            )
        )
        / 2.0
        + RACK_PADDING_X,
    )


    span_y = max(
        0.10,

        (
            max(
                rack_ys
            )
            -
            min(
                rack_ys
            )
        )
        / 2.0
        + RACK_PADDING_Y,
    )


    rack_body = new_body(
        name="ee_rack",

        pos=(
            rack_center.tolist()
        ),
    )


    rack_body.append(

        new_geom(
            name="ee_rack_base",

            type="box",

            size=[
                span_x,
                span_y,
                RACK_BASE_HALF_HEIGHT,
            ],

            pos=[
                0.0,
                0.0,
                RACK_BASE_HALF_HEIGHT,
            ],

            group=1,

            rgba=(
                RACK_BASE_RGBA
            ),

            contype="0",

            conaffinity="0",
        )
    )


    rack_info = {}


    # --------------------------------------------------------------
    # EE slots
    # --------------------------------------------------------------

    for ee_id in EE_DISPLAY_ORDER:

        spec = (
            ee_by_id[
                ee_id
            ]
        )


        slot_x, slot_y = (
            ordered_layout[
                ee_id
            ]
        )


        slot_world_origin = np.array(
            [
                slot_x,
                slot_y,
                table_z,
            ],
            dtype=float,
        )


        slot_relative_position = (
            slot_world_origin
            - rack_center
        )


        slot_body = new_body(
            name=(
                f"ee_rack_slot_{ee_id}"
            ),

            pos=(
                slot_relative_position.tolist()
            ),
        )


        # ----------------------------------------------------------
        # Support
        # ----------------------------------------------------------

        support_height = (
            _calculate_support_height(
                ee_id
            )
        )


        rack_top = (
            2.0
            * RACK_BASE_HALF_HEIGHT
        )


        support_center_z = (
            rack_top
            + support_height / 2.0
        )


        support_top_z = (
            rack_top
            + support_height
        )


        support_half_x, support_half_y = (
            SUPPORT_HALF_SIZE_XY[
                ee_id
            ]
        )


        slot_body.append(

            new_geom(
                name=(
                    f"ee_rack_support_{ee_id}"
                ),

                type="box",

                size=[
                    support_half_x,
                    support_half_y,
                    support_height / 2.0,
                ],

                pos=[
                    0.0,
                    0.0,
                    support_center_z,
                ],

                group=1,

                rgba=(
                    SUPPORT_RGBA
                ),

                contype="0",

                conaffinity="0",
            )
        )


        rack_body.append(
            slot_body
        )


        # ----------------------------------------------------------
        # EE visual model
        # ----------------------------------------------------------

        gripper_cls = (
            EE_GRIPPER_CLASSES[
                ee_id
            ]
        )


        gripper = gripper_cls(
            idn=f"rack_{ee_id}"
        )


        freeze_gripper_for_display(
            gripper
        )


        root = (
            gripper.worldbody.find(
                "body"
            )
        )


        if root is None:
            raise RuntimeError(
                "Could not find root body "
                f"for EE '{ee_id}'."
            )


        display_pos = np.array(
            [
                slot_x,
                slot_y,
                table_z
                + EE_DISPLAY_HEIGHT,
            ],
            dtype=float,
        )


        root.set(
            "pos",
            array_to_string(
                display_pos
            ),
        )


        root.set(
            "quat",
            array_to_string(
                EE_DISPLAY_QUAT
            ),
        )


        arena.merge_assets(
            gripper
        )


        frozen_root = deepcopy(
            root
        )


        arena.worldbody.append(
            frozen_root
        )


        # ----------------------------------------------------------
        # Minimal metadata
        # ----------------------------------------------------------

        rack_info[
            ee_id
        ] = {

            **spec,

            "gripper_class": (
                gripper_cls.__name__
            ),

            "rack_body": (
                frozen_root.get(
                    "name"
                )
            ),

            "rack_slot": (
                f"ee_rack_slot_{ee_id}"
            ),

            "rack_position": (
                display_pos.copy()
            ),

            "rack_orientation": (
                EE_DISPLAY_QUAT.copy()
            ),

            "rack_support_top_z": (
                table_z
                + support_top_z
            ),

            "rack_support_height": (
                support_height
            ),
        }


    arena.worldbody.append(
        rack_body
    )


    return rack_info