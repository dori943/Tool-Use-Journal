import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tuj.m1_scene import MockBackend, PropertyMemory, build_m1, ground_scene


def _points(center):
    x, y, z = np.meshgrid(
        np.linspace(center[0] - 10, center[0] + 10, 6),
        np.linspace(center[1] - 10, center[1] + 10, 6),
        np.linspace(center[2] - 3, center[2] + 3, 3),
    )
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def _scene():
    return build_m1([
        {"name": "spatula", "cls": "spatula", "points": _points((0, 0, 10))},
        {"name": "spoon", "cls": "spoon", "points": _points((60, 0, 10))},
    ])


def test_ground_scene_populates_all_scene_nodes_and_reuses_memory(tmp_path):
    memory_path = tmp_path / "memory.json"
    ee_pool = [{"ee_id": "2F", "type": "parallel_2f", "stroke_mm": 85,
                "payload_kg": 5.0, "grip_force_n": 235.0}]

    first = _scene()
    stats = ground_scene(first, backend=MockBackend(), memory=PropertyMemory(memory_path),
                         ee_pool=ee_pool, reach_mm=850)
    assert stats["grounded"] == 2
    assert stats["memory_update"]["new"] == 2
    for node in first["nodes"]:
        assert node["grounding_source"] == "backend"
        assert node["mass_kg"] is not None
        assert "2F" in node["ee"]

    second = _scene()
    stats = ground_scene(second, backend=MockBackend(), memory=PropertyMemory(memory_path),
                         ee_pool=ee_pool, reach_mm=850)
    assert stats["memory_hits"] == 2
    assert stats["grounded"] == 0
    assert all(node["grounding_source"] == "memory" for node in second["nodes"])
