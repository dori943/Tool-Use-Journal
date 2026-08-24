"""C1-T2 robot pedestal (static box stand).

Kitchen Island 밖에 UR5e를 올리기 위한 고정 pedestal.
C1-T1 / ee_rack.py 동작에는 영향 없음.
"""

from __future__ import annotations

import numpy as np
from robosuite.utils.mjcf_utils import new_body, new_geom


PEDESTAL_BODY_NAME = "robot_pedestal"
PEDESTAL_GEOM_NAME = "robot_pedestal_geom"
PEDESTAL_RGBA = [0.35, 0.35, 0.38, 1.0]


def remove_robot_pedestal(arena) -> None:
    worldbody = arena.worldbody
    for body in list(worldbody.findall("body")):
        if (body.get("name") or "") == PEDESTAL_BODY_NAME:
            worldbody.remove(body)


def add_robot_pedestal(
    arena,
    *,
    center_xy,
    top_z: float,
    half_size_xy=(0.28, 0.28),
    rgba=None,
) -> dict:
    """Floor(z=0)부터 ``top_z``까지 올라오는 회색 box pedestal을 추가한다.

    Robot base는 MJCF에서 fixed mount이므로 pedestal은 시각/구조용 static
    body로 두고 collision은 비활성화한다 (물리 불안정 방지).
    """
    remove_robot_pedestal(arena)

    cx, cy = float(center_xy[0]), float(center_xy[1])
    top_z = float(top_z)
    hx, hy = float(half_size_xy[0]), float(half_size_xy[1])
    half_h = max(top_z / 2.0, 0.01)
    center_z = half_h

    body = new_body(
        name=PEDESTAL_BODY_NAME,
        pos=[cx, cy, center_z],
    )
    body.append(
        new_geom(
            name=PEDESTAL_GEOM_NAME,
            type="box",
            size=[hx, hy, half_h],
            pos=[0.0, 0.0, 0.0],
            group=1,
            rgba=list(rgba or PEDESTAL_RGBA),
            contype="0",
            conaffinity="0",
        )
    )
    arena.worldbody.append(body)

    return {
        "name": PEDESTAL_BODY_NAME,
        "center_xy": np.array([cx, cy], dtype=float),
        "top_z": top_z,
        "half_size_xy": np.array([hx, hy], dtype=float),
        "full_size_xyz": np.array([2.0 * hx, 2.0 * hy, top_z], dtype=float),
        "center_xyz": np.array([cx, cy, center_z], dtype=float),
    }
