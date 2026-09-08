from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

from environments.task_camera import add_standard_task_camera


class _Model:
    def __init__(self):
        self.worldbody = ET.Element("worldbody")


def _values(element, attribute):
    return tuple(float(value) for value in element.get(attribute).split())


def test_standard_camera_replaces_nested_agentview_and_uses_robot_frame():
    model = _Model()
    body = ET.SubElement(model.worldbody, "body", name="fixture")
    ET.SubElement(body, "camera", name="agentview")

    config = add_standard_task_camera(
        model,
        robot_base_xy=(2.0, -3.0),
        robot_base_yaw_rad=0.0,
        surface_z=0.8,
    )

    cameras = [item for item in model.worldbody.iter("camera")]
    assert len(cameras) == 1
    assert cameras[0].get("name") == "agentview"
    assert _values(cameras[0], "pos") == pytest.approx((3.34, -3.0, 1.79))
    assert float(cameras[0].get("fovy")) == pytest.approx(60.0)
    assert config["width"] == 960
    assert config["height"] == 540


def test_standard_camera_rotates_with_robot_yaw():
    model = _Model()

    add_standard_task_camera(
        model,
        robot_base_xy=(2.0, -3.0),
        robot_base_yaw_rad=math.pi / 2.0,
        surface_z=0.8,
    )

    camera = next(model.worldbody.iter("camera"))
    assert _values(camera, "pos") == pytest.approx((2.0, -1.66, 1.79))
