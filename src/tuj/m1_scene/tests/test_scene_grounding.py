import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tuj.m1_scene import (
    MockBackend,
    PropertyMemory,
    build_m1,
    ground_scene,
    serialize,
)


def _points(center):
    x, y, z = np.meshgrid(
        np.linspace(center[0] - 10, center[0] + 10, 6),
        np.linspace(center[1] - 10, center[1] + 10, 6),
        np.linspace(center[2] - 3, center[2] + 3, 3),
    )
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def _scene(x_offset=0):
    return build_m1([
        {"name": "spatula", "cls": "spatula", "points": _points((x_offset, 0, 10))},
        {"name": "spoon", "cls": "spoon", "points": _points((60 + x_offset, 0, 10))},
    ])


def test_ground_scene_populates_all_scene_nodes_and_reuses_memory(tmp_path):
    memory_path = tmp_path / "memory.json"
    ee_pool = [{"ee_id": "2F", "type": "parallel_2f", "stroke_mm": 85,
                "payload_kg": 5.0, "grip_force_n": 235.0}]

    first = _scene()
    memory = PropertyMemory(memory_path, task_id="test-task")
    stats = ground_scene(first, backend=MockBackend(),
                         memory=memory,
                         ee_pool=ee_pool, reach_mm=850)
    assert stats["grounded"] == 2
    assert stats["memory_update"]["new"] == 2
    for node in first["nodes"]:
        assert node["grounding_source"] == "backend"
        assert node["mass_kg"] is not None
        assert "2F" in node["ee"]
        assert node["geometry"]["center"] == node["center_mm"]
        assert node["geometry"]["aabb_size"] == node["bbox_mm"]
        assert "footprint_mm" in node["geometry"]

    serialized = serialize(first)
    assert all("center" in node["geometry"] for node in serialized["nodes"])
    assert all("aabb_size" in node["geometry"] for node in serialized["nodes"])
    assert all("_points" not in node for node in serialized["nodes"])
    cached = memory.lookup("obj_spatula_spatula")
    assert cached is not None
    assert "center" not in cached["geometry"]
    assert "aabb_size" not in cached["geometry"]

    second = _scene(x_offset=25)
    stats = ground_scene(second, backend=MockBackend(),
                         memory=PropertyMemory(memory_path, task_id="test-task"),
                         ee_pool=ee_pool, reach_mm=850)
    assert stats["memory_hits"] == 2
    assert stats["grounded"] == 0
    assert all(node["grounding_source"] == "memory" for node in second["nodes"])
    assert all(node["geometry"]["center"] == node["center_mm"] for node in second["nodes"])
    assert all(node["geometry"]["aabb_size"] == node["bbox_mm"] for node in second["nodes"])
    assert second["nodes"][0]["geometry"]["center"][0] == 25.0
