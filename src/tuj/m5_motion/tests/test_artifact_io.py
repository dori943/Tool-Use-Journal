from pathlib import Path

import pytest

from tuj.m5_motion.artifact_io import world_snapshot_json, write_text_atomic


def test_world_snapshot_json_omits_triangle_soups():
    payload = {
        "objects": {
            "plate": {
                "dimensions_m": [0.1, 0.1, 0.01],
                "collision_points_m": [[0.0, 0.0, 0.0]],
                "collision_surface_triangles_m": [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]],
            }
        }
    }
    text = world_snapshot_json(payload)
    assert "collision_points_m" in text
    assert "collision_surface_triangles_m" not in text


def test_write_text_atomic_replaces_existing(tmp_path: Path):
    path = tmp_path / "initial_world.json"
    write_text_atomic(path, '{"a": 1}')
    write_text_atomic(path, '{"a": 2}')
    assert path.read_text(encoding="utf-8").strip() == '{"a": 2}'
    assert not path.with_suffix(".json.tmp").exists()


def test_write_text_atomic_retries_then_raises(tmp_path: Path, monkeypatch):
    path = tmp_path / "locked.json"
    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(Path, "write_text", boom)
    with pytest.raises(OSError):
        write_text_atomic(path, "{}", retries=3)
    assert calls["n"] == 3
