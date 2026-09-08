"""M1 ground_scene + M0 Object Knowledge retrieval integration tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tuj.m0_memory.density_only import DensityOnlyResult
from tuj.m0_memory.object_knowledge import ObjectKnowledgeManager, _as_size3
from tuj.m1_scene import (
    MockBackend,
    PropertyMemory,
    build_m1,
    ground_scene,
    serialize,
)


def _points(center, half=(10.0, 10.0, 3.0)):
    x, y, z = np.meshgrid(
        np.linspace(center[0] - half[0], center[0] + half[0], 6),
        np.linspace(center[1] - half[1], center[1] + half[1], 6),
        np.linspace(center[2] - half[2], center[2] + half[2], 3),
    )
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def _props(
    density=1000.0,
    extents=(100.0, 50.0, 40.0),
    material="plastic",
):
    return {
        "geometry": {
            "length_mm": 100.0,
            "diameter_mm": None,
            "extents_mm": list(extents),
            "height_z_mm": 100.0,
            "cylinder_like": False,
            "surface_rms_mm": 1.0,
            "footprint_mm": [100.0, 50.0],
            "height_mm": 40.0,
            "seal_patch_rms_mm": 1.0,
        },
        "material": material,
        "density_kgm3": density,
        "mass_kg": 0.2,
        "youngs_gpa": 2.0,
        "mu": {
            "mu": 0.4,
            "stage": 0,
            "material": material,
            "rms_mm": 1.0,
        },
        "confidence": 0.8,
        "materials_topk": [
            {"name": material, "prob": 0.8}
        ],
    }


class CountingBackend(MockBackend):
    def __init__(self):
        self.calls = 0

    def estimate(self, crop_rgb, cls_hint, points_mm=None):
        self.calls += 1
        return super().estimate(crop_rgb, cls_hint, points_mm)


def _spoon_scene(
    center=(0, 0, 10),
    half=(50.0, 25.0, 20.0),
):
    scene = build_m1([
        {
            "name": "spoon",
            "cls": "spoon",
            "points": _points(center, half),
        }
    ])

    # Stable bbox for retrieval tests
    # (orientation-invariant vs stored extents).
    scene["nodes"][0]["bbox_mm"] = [100.0, 50.0, 40.0]
    return scene


def _scene(x_offset=0):
    return build_m1([
        {
            "name": "spatula",
            "cls": "spatula",
            "points": _points((x_offset, 0, 10)),
        },
        {
            "name": "spoon",
            "cls": "spoon",
            "points": _points((60 + x_offset, 0, 10)),
        },
    ])


def test_ground_scene_populates_all_scene_nodes_and_reuses_memory(tmp_path):
    memory_path = tmp_path / "memory.json"

    ee_pool = [
        {
            "ee_id": "2F",
            "type": "parallel_2f",
            "stroke_mm": 85,
            "payload_kg": 5.0,
            "grip_force_n": 235.0,
        }
    ]

    # ---------------------------------------------------------
    # First episode: memory empty → Full grounding
    # ---------------------------------------------------------
    first = _scene()

    memory = PropertyMemory(
        memory_path,
        task_id="test-task",
    )

    stats = ground_scene(
        first,
        backend=MockBackend(),
        memory=memory,
        ee_pool=ee_pool,
        reach_mm=850,
    )

    assert stats["grounded"] == 2
    assert stats["memory_update"]["new"] == 2

    for node in first["nodes"]:
        assert node["grounding_source"] == "backend"
        assert node["mass_kg"] is not None
        assert "2F" in node["ee"]

        # Latest main M1 → M5 geometry contract
        assert node["geometry"]["center"] == node["center_mm"]
        assert node["geometry"]["aabb_size"] == node["bbox_mm"]

        assert "footprint_mm" in node["geometry"]

    # Serialized m1.json에도 current observation geometry가 존재해야 한다.
    serialized = serialize(first)

    assert all(
        "center" in node["geometry"]
        for node in serialized["nodes"]
    )
    assert all(
        "aabb_size" in node["geometry"]
        for node in serialized["nodes"]
    )
    assert all(
        "_points" not in node
        for node in serialized["nodes"]
    )

    # 반대로 Object Knowledge에는 episode-specific
    # center / aabb_size를 저장하지 않는다.
    cached = memory.lookup("obj_spatula_spatula")

    assert cached is not None
    assert "center" not in cached["geometry"]
    assert "aabb_size" not in cached["geometry"]

    # ---------------------------------------------------------
    # Second episode:
    # same task + same object_id → exact HIT
    # 위치가 달라져도 current observation geometry 사용
    # ---------------------------------------------------------
    second = _scene(x_offset=25)

    stats = ground_scene(
        second,
        backend=MockBackend(),
        memory=PropertyMemory(
            memory_path,
            task_id="test-task",
        ),
        ee_pool=ee_pool,
        reach_mm=850,
    )

    assert stats["memory_hits"] == 2
    assert stats["grounded"] == 0

    assert all(
        node["grounding_source"] == "memory"
        for node in second["nodes"]
    )

    # Latest main geometry contract 보존
    assert all(
        node["geometry"]["center"] == node["center_mm"]
        for node in second["nodes"]
    )

    assert all(
        node["geometry"]["aabb_size"] == node["bbox_mm"]
        for node in second["nodes"]
    )

    # 실제 위치가 first episode와 달라졌는지 확인
    assert second["nodes"][0]["geometry"]["center"][0] == 25.0


def test_same_task_exact_hit_skips_c3_and_full(tmp_path):
    memory = PropertyMemory(
        tmp_path / "m.json",
        task_id="same",
    )

    spoon_id = "obj_spoon_spoon"

    memory.update_entry(
        "same",
        spoon_id,
        _props(),
    )
    memory.save()

    c3 = {"calls": 0}
    full = CountingBackend()

    def infer(_):
        c3["calls"] += 1
        return DensityOnlyResult(1000, True)

    scene = _spoon_scene()

    assert scene["nodes"][0]["id"] == spoon_id

    debug = {}

    stats = ground_scene(
        scene,
        backend=full,
        memory=memory,
        density_infer=infer,
        retrieval_debug=debug,
    )

    assert stats["memory_hits"] == 1
    assert stats["grounded"] == 0

    assert c3["calls"] == 0
    assert full.calls == 0

    assert debug[spoon_id]["lookup_type"] == "intra_task_exact"
    assert debug[spoon_id]["result"] == "HIT"
    assert debug[spoon_id]["full_m3_skipped"] is True


def test_cross_task_hit_calls_c3_once_skips_full(tmp_path):
    memory = PropertyMemory(
        tmp_path / "m.json",
        task_id="now",
        bbox_relative_threshold=0.25,
        density_relative_threshold=0.20,
    )

    memory.update_entry(
        "past",
        "obj_spoon_spoon",
        _props(1000),
    )
    memory.save()

    c3 = {"calls": 0}
    full = CountingBackend()

    def infer(_):
        c3["calls"] += 1

        return DensityOnlyResult(
            1050,
            True,
            {"total_tokens": 42},
            materials_topk=[
                {
                    "name": "plastic",
                    "prob": 1.0,
                    "density_kgm3": [1000, 1100],
                }
            ],
            material_committed=True,
            top1_gap=0.5,
        )

    debug = {}

    stats = ground_scene(
        _spoon_scene(),
        backend=full,
        memory=memory,
        density_infer=infer,
        retrieval_debug=debug,
    )

    assert c3["calls"] == 1
    assert full.calls == 0

    assert stats["memory_hits"] == 1
    assert stats["grounded"] == 0

    d = debug["obj_spoon_spoon"]

    assert d["lookup_type"] == "cross_task"
    assert d["result"] == "HIT"
    assert d["c3_llm_called"] is True
    assert d["c3_materials_topk"] is not None
    assert d["query_bbox_mm"] == [100.0, 50.0, 40.0]
    assert d["memory_entry_count"] == 1


def test_cross_task_density_miss_runs_full_and_stores(tmp_path):
    path = tmp_path / "m.json"

    memory = PropertyMemory(
        path,
        task_id="later",
        density_relative_threshold=0.20,
    )

    memory.update_entry(
        "past",
        "obj_spoon_spoon",
        _props(1000),
    )
    memory.save()

    c3 = {"calls": 0}
    full = CountingBackend()

    def infer(_):
        c3["calls"] += 1
        return DensityOnlyResult(3000, True)

    debug = {}

    stats = ground_scene(
        _spoon_scene(),
        backend=full,
        memory=memory,
        density_infer=infer,
        retrieval_debug=debug,
        source="mock",
    )

    assert c3["calls"] == 1
    assert full.calls == 1

    assert stats["grounded"] == 1
    assert stats["memory_hits"] == 0

    assert (
        debug["obj_spoon_spoon"]["miss_reason"]
        == "DENSITY_THRESHOLD_EXCEEDED"
    )

    assert debug["obj_spoon_spoon"]["full_m3_called"] is True

    reloaded = PropertyMemory(
        path,
        task_id="later",
    )

    entry, _, _ = reloaded.lookup_exact(
        "later",
        "obj_spoon_spoon",
    )

    assert entry is not None


def test_no_bbox_candidate_skips_c3_runs_full(tmp_path):
    memory = PropertyMemory(
        tmp_path / "m.json",
        task_id="now",
    )

    memory.update_entry(
        "past",
        "obj_spoon_spoon",
        _props(extents=(300, 200, 150)),
    )
    memory.save()

    c3 = {"calls": 0}
    full = CountingBackend()

    def infer(_):
        c3["calls"] += 1
        return DensityOnlyResult(1000, True)

    debug = {}

    stats = ground_scene(
        _spoon_scene(),
        backend=full,
        memory=memory,
        density_infer=infer,
        retrieval_debug=debug,
    )

    assert c3["calls"] == 0
    assert full.calls == 1
    assert stats["grounded"] == 1

    assert (
        debug["obj_spoon_spoon"]["miss_reason"]
        == "NO_BBOX_CANDIDATE"
    )


def test_numpy_bbox_and_extents_filter(tmp_path):
    m = ObjectKnowledgeManager(
        tmp_path / "m.json",
        bbox_relative_threshold=0.25,
    )

    props = _props(
        extents=(100.0, 50.0, 40.0),
    )

    props["geometry"]["extents_mm"] = np.array(
        [100.0, 50.0, 40.0]
    )

    m.update(
        "past",
        "a",
        props,
    )

    found = m.filter_bbox(
        "now",
        np.array([102.0, 51.0, 39.0]),
    )

    assert len(found) == 1
    assert found[0]["object_id"] == "a"

    assert _as_size3(
        np.array([1.0, 2.0, 3.0])
    ) == [1.0, 2.0, 3.0]


def test_failure_recovery_preserved_after_m1_ground(tmp_path):
    path = tmp_path / "memory.json"

    path.write_text(
        json.dumps({
            "schema_version": "1.0",
            "object_knowledge": {
                "objects": {}
            },
            "failure_recovery_experience": {
                "experiences": [
                    {"id": "failure-1"}
                ]
            },
        }),
        encoding="utf-8",
    )

    memory = PropertyMemory(
        path,
        task_id="t1",
    )

    ground_scene(
        _spoon_scene(),
        backend=MockBackend(),
        memory=memory,
        source="mock",
    )

    saved = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert (
        saved["failure_recovery_experience"]["experiences"]
        == [{"id": "failure-1"}]
    )

    assert len(
        saved["object_knowledge"]["objects"]
    ) == 1


def test_stage_m1_argv_forwards_memory_and_thresholds():
    import importlib.util

    root = Path(__file__).resolve().parents[4]
    path = root / "scripts" / "run.py"

    spec = importlib.util.spec_from_file_location(
        "tuj_scripts_run",
        path,
    )

    run_mod = importlib.util.module_from_spec(spec)

    assert spec.loader is not None
    spec.loader.exec_module(run_mod)

    args = SimpleNamespace(
        seed=0,
        backend="siphy",
        model="gpt-4o",
        memory=r"output\m0_m1_test_memory.json",
        m0_bbox_threshold=0.25,
        m0_density_threshold=0.20,
        view=False,
    )

    argv = run_mod._stage_m1_argv(
        "c2_1",
        Path("output/c2_1"),
        args,
    )

    assert "--memory" in argv

    assert (
        argv[argv.index("--memory") + 1]
        == r"output\m0_m1_test_memory.json"
    )

    assert argv[argv.index("--model") + 1] == "gpt-4o"
    assert argv[argv.index("--backend") + 1] == "siphy"

    assert float(
        argv[argv.index("--m0-bbox-threshold") + 1]
    ) == 0.25

    assert float(
        argv[argv.index("--m0-density-threshold") + 1]
    ) == 0.20


def _vac_ee_pool():
    return [
        {
            "ee_id": "VAC",
            "type": "vacuum",
            "payload_kg": 5.0,
            "seal_rms_tol_mm": 1.5,
            "seal_diameter_mm": 30.0,
        }
    ]


def _hit_with_bad_surface_rms(
    tmp_path,
    bad_rms,
):
    """same-task HIT with invalid surface_rms
    → geometry refresh, no Full SiPhy.
    """
    from tuj.m1_scene.grounding import geometry_is_current

    spoon_id = "obj_spoon_spoon"

    props = _props(
        material="apple_flesh",
        density=900.0,
    )

    props["geometry"]["surface_rms_mm"] = bad_rms
    props["mu"]["rms_mm"] = bad_rms

    assert not geometry_is_current(
        props["geometry"]
    )

    memory = PropertyMemory(
        tmp_path / "m.json",
        task_id="same",
    )

    memory.update_entry(
        "same",
        spoon_id,
        props,
    )
    memory.save()

    c3 = {"calls": 0}
    full = CountingBackend()

    def infer(_):
        c3["calls"] += 1
        return DensityOnlyResult(900, True)

    # Dense enough point cloud so refreshed surface_rms is finite.
    scene = build_m1([
        {
            "name": "spoon",
            "cls": "spoon",
            "points": _points(
                (0, 0, 10),
                half=(40.0, 40.0, 20.0),
            ),
        }
    ])

    scene["nodes"][0]["bbox_mm"] = [
        100.0,
        50.0,
        40.0,
    ]

    stats = ground_scene(
        scene,
        backend=full,
        memory=memory,
        density_infer=infer,
        ee_pool=_vac_ee_pool(),
        reach_mm=850,
    )

    node = scene["nodes"][0]

    assert stats["memory_hits"] == 1
    assert stats["grounded"] == 0
    assert stats["geom_refreshed"] == 1

    assert c3["calls"] == 0
    assert full.calls == 0

    assert node["grounding_source"] == "memory"

    # Stored physical properties reused
    assert node["material"] == "apple_flesh"
    assert node["density_kgm3"] == 900.0
    assert node["mass_kg"] == 0.2

    # Geometry refreshed from current observation
    assert geometry_is_current(
        node["geometry"]
    )

    assert np.isfinite(
        node["geometry"]["surface_rms_mm"]
    )

    assert "VAC" in node["ee"]

    assert isinstance(
        node["ee"]["VAC"]["feasible"],
        bool,
    )


def test_memory_hit_refreshes_none_surface_rms_without_full_siphy(
    tmp_path,
):
    _hit_with_bad_surface_rms(
        tmp_path,
        None,
    )


def test_memory_hit_refreshes_nan_surface_rms_without_full_siphy(
    tmp_path,
):
    _hit_with_bad_surface_rms(
        tmp_path,
        float("nan"),
    )


def test_geometry_is_current_rejects_non_finite_ee_fields():
    from tuj.m1_scene.grounding import geometry_is_current

    base = _props()["geometry"]

    assert geometry_is_current(base)

    for bad in (
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
    ):
        g = dict(base)
        g["surface_rms_mm"] = bad

        assert not geometry_is_current(g)

    missing = dict(base)
    del missing["surface_rms_mm"]

    assert not geometry_is_current(missing)