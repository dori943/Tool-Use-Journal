"""Shared fixed recording camera for task environments."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np


STANDARD_TASK_CAMERA_NAME = "agentview"
STANDARD_TASK_CAMERA_FOVY_DEG = 70.0 #값 ↑ → 더 넓게 (줌아웃)
STANDARD_TASK_VIDEO_WIDTH = 960
STANDARD_TASK_VIDEO_HEIGHT = 540

# Relative to the robot base and its forward (+x) direction. These values
# reproduce the approved front-on, symmetric, slightly elevated C4-2 view.
_LOOK_FORWARD_M = 0.34
_LOOK_ABOVE_SURFACE_M = 0.27
_EYE_FORWARD_FROM_LOOK_M = 1.00
_EYE_ABOVE_LOOK_M = 0.72


def _remove_named_camera(worldbody: ET.Element, name: str) -> None:
    """Remove every camera with *name*, including cameras nested in a body."""

    for parent in worldbody.iter():
        for child in list(parent):
            if child.tag == "camera" and child.get("name") == name:
                parent.remove(child)


def add_standard_task_camera(
    model,
    *,
    robot_base_xy,
    robot_base_yaw_rad: float,
    surface_z: float,
    name: str = STANDARD_TASK_CAMERA_NAME,
) -> dict[str, object]:
    """Install the same robot-relative fixed camera in any task MJCF model."""

    base_xy = np.asarray(robot_base_xy, dtype=float)
    if base_xy.shape != (2,) or not np.all(np.isfinite(base_xy)):
        raise ValueError("robot_base_xy must contain two finite values")
    yaw = float(robot_base_yaw_rad)
    support_z = float(surface_z)
    if not math.isfinite(yaw) or not math.isfinite(support_z):
        raise ValueError("camera yaw and surface height must be finite")

    robot_forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
    look_at = np.array(
        [
            *(base_xy + _LOOK_FORWARD_M * robot_forward),
            support_z + _LOOK_ABOVE_SURFACE_M,
        ],
        dtype=float,
    )
    eye = look_at + np.array(
        [
            *(_EYE_FORWARD_FROM_LOOK_M * robot_forward),
            _EYE_ABOVE_LOOK_M,
        ],
        dtype=float,
    )

    view_forward = look_at - eye
    view_forward /= np.linalg.norm(view_forward)
    right = np.cross(view_forward, np.array([0.0, 0.0, 1.0], dtype=float))
    right /= np.linalg.norm(right)
    up = np.cross(right, view_forward)

    worldbody = model.worldbody
    _remove_named_camera(worldbody, name)
    ET.SubElement(
        worldbody,
        "camera",
        name=name,
        mode="fixed",
        pos=" ".join(f"{value:.12g}" for value in eye),
        xyaxes=" ".join(
            f"{value:.12g}" for value in np.concatenate((right, up))
        ),
        fovy=f"{STANDARD_TASK_CAMERA_FOVY_DEG:.12g}",
    )
    return {
        "name": name,
        "eye_m": tuple(float(value) for value in eye),
        "look_at_m": tuple(float(value) for value in look_at),
        "fovy_deg": STANDARD_TASK_CAMERA_FOVY_DEG,
        "width": STANDARD_TASK_VIDEO_WIDTH,
        "height": STANDARD_TASK_VIDEO_HEIGHT,
    }


__all__ = [
    "STANDARD_TASK_CAMERA_FOVY_DEG",
    "STANDARD_TASK_CAMERA_NAME",
    "STANDARD_TASK_VIDEO_HEIGHT",
    "STANDARD_TASK_VIDEO_WIDTH",
    "add_standard_task_camera",
]
