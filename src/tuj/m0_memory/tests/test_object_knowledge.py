from __future__ import annotations

import json

from tuj.m0_memory.density_only import DensityOnlyResult
from tuj.m0_memory.memory_store import UnifiedMemoryStore
from tuj.m0_memory.object_knowledge import ObjectKnowledgeManager
from tuj.m3_grounding.materialize import Materializer, new_gk
from tuj.m3_grounding.memory import PropertyMemory


def props(density=1000.0, extents=(100.0, 50.0, 40.0), material="plastic"):
    return {"geometry": {"length_mm": 100.0, "diameter_mm": None,
            "extents_mm": list(extents), "height_z_mm": 100.0,
            "cylinder_like": False, "surface_rms_mm": 1.0},
            "material": material, "density_kgm3": density, "mass_kg": 0.2,
            "youngs_gpa": 2.0, "mu": {"mu": .4, "stage": 0,
            "material": material, "rms_mm": 1.0}, "confidence": .8,
            "materials_topk": [{"name": material, "prob": .8}]}


def test_schema_load_save_and_collision_free_keys(tmp_path):
    path = tmp_path / "object_knowledge.json"
    m = ObjectKnowledgeManager(path)
    m.update("task-a", "same-id", props())
    m.update("task-b", "same-id", props(900))
    m.save()
    loaded = ObjectKnowledgeManager(path)
    assert loaded.data["schema_version"] == "1.0"
    assert len(loaded.objects) == 2


def test_legacy_load_migrates_and_preserves_unknown_root_sections(tmp_path):
    path = tmp_path / "memory.json"
    legacy = {"objects": {"obj": {"props": props(), "stage": 2,
              "source": "siphy", "episodes_seen": 7,
              "updated_at": "2026-01-01T00:00:00+00:00"}},
              "custom_section": {"keep": True}}
    path.write_text(json.dumps(legacy), encoding="utf-8")
    m = ObjectKnowledgeManager(path)
    assert len(m.objects) == 1
    entry = next(iter(m.objects.values()))
    assert entry["metadata"]["stage"] == 2
    assert entry["metadata"]["episodes_seen"] == 7
    m.save()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "objects" not in saved
    assert saved["custom_section"] == {"keep": True}
    assert saved["failure_recovery_experience"] == {"experiences": []}


def test_failure_recovery_section_survives_object_update(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"schema_version": "1.0",
        "object_knowledge": {"objects": {}},
        "failure_recovery_experience": {"experiences": [{"id": "failure-1"}]}}),
        encoding="utf-8")
    m = ObjectKnowledgeManager(path)
    m.update("task", "obj", props())
    m.save()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["failure_recovery_experience"]["experiences"] == [{"id": "failure-1"}]


def test_bootstrap_from_m3(tmp_path):
    source = tmp_path / "m3.json"
    source.write_text(json.dumps({"responses": [
        {"node_id": "obj", **props()},
        {"node_id": "obj", **props()},
        {"queried_by": "relational", "value_mm": 2},
    ]}), encoding="utf-8")
    m = ObjectKnowledgeManager(tmp_path / "memory.json")
    result = m.update_from_m3(source, "c1-1")
    assert result["updated"] == ["obj"]
    entry, _, _ = m.lookup_exact("c1-1", "obj")
    assert entry["metadata"]["episodes_seen"] == 1


def test_bbox_filter_is_orientation_invariant_and_cross_task_only(tmp_path):
    m = ObjectKnowledgeManager(tmp_path / "m.json", bbox_relative_threshold=.20)
    m.update("past", "a", props(extents=(100, 50, 40)))
    m.update("current", "b", props(extents=(100, 50, 40)))
    m.update("past", "far", props(extents=(200, 50, 40)))
    found = m.filter_bbox("current", [42, 102, 51])
    assert [x["object_id"] for x in found] == ["a"]


def test_density_best_match_hit_and_miss(tmp_path):
    m = ObjectKnowledgeManager(tmp_path / "m.json", density_relative_threshold=.20)
    m.update("past", "a", props(1000))
    m.update("past", "b", props(1500))
    candidates = m.filter_bbox("now", [100, 50, 40])
    best, ranked = m.retrieve_by_density(candidates, 1100)
    assert best["object_id"] == "a"
    hit, debug = m.lookup_or_retrieve("now", "q", [100, 50, 40], "crop",
        lambda _: DensityOnlyResult(1100, True, {"total_tokens": 10}))
    assert hit["density_kgm3"] == 1000
    assert debug["result"] == "HIT" and debug["full_m3_skipped"]
    miss, debug = m.lookup_or_retrieve("now2", "q", [100, 50, 40], "crop",
        lambda _: DensityOnlyResult(2000, True))
    assert miss is None and debug["miss_reason"] == "DENSITY_THRESHOLD_EXCEEDED"


class CountingBackend:
    def __init__(self): self.calls = 0
    def estimate(self, crop_rgb, cls_hint, points_mm=None):
        self.calls += 1
        return props()["material"] and {"material": "plastic", "density_kgm3": 1000,
            "mass_kg": .2, "youngs_gpa": 2, "confidence": .8}


def node(object_id="obj"):
    return {"id": object_id, "class": "ignored", "bbox_mm": [100, 50, 40],
            "center_mm": [0, 0, 0], "_points": [[0, 0, 0], [100, 0, 0],
            [0, 50, 0], [0, 0, 40], [100, 50, 40]]}


def test_exact_hit_calls_neither_c3_nor_full_m3(tmp_path):
    m = ObjectKnowledgeManager(tmp_path / "m.json")
    m.update("same", "obj", props())
    c3 = {"calls": 0}; full = CountingBackend()
    def infer(_): c3["calls"] += 1; return DensityOnlyResult(1000, True)
    mat = Materializer({"nodes": [node()], "edges": []}, backend=full,
        task_id="same", object_knowledge=m, density_infer=infer)
    result = mat.query_intrinsic(new_gk("SG1"), "obj", "q", crop_rgb="crop")
    assert result["density_kgm3"] == 1000
    assert c3["calls"] == 0 and full.calls == 0


def test_cross_hit_skips_full_and_miss_runs_full_then_updates(tmp_path):
    m = ObjectKnowledgeManager(tmp_path / "m.json")
    m.update("past", "old", props())
    full = CountingBackend()
    mat = Materializer({"nodes": [node()], "edges": []}, backend=full,
        task_id="now", object_knowledge=m,
        density_infer=lambda _: DensityOnlyResult(1050, True))
    mat.query_intrinsic(new_gk("SG"), "obj", "q", crop_rgb="crop")
    assert full.calls == 0
    full2 = CountingBackend()
    mat2 = Materializer({"nodes": [node("new")], "edges": []}, backend=full2,
        task_id="later", object_knowledge=m,
        density_infer=lambda _: DensityOnlyResult(3000, True))
    mat2.query_intrinsic(new_gk("SG"), "new", "q", crop_rgb="crop")
    assert full2.calls == 1
    # Production persistence remains run-end, as in the original PropertyMemory flow.
    m.update("later", "new", mat2._cache["new"])
    assert m.lookup_exact("later", "new")[0] is not None


def test_no_bbox_candidate_skips_c3_and_full_m3_updates_at_run_end(tmp_path):
    memory = PropertyMemory(tmp_path / "memory.json", task_id="now",
                            source_episode="episode-1")
    calls = {"c3": 0}; full = CountingBackend()
    def infer(_): calls["c3"] += 1; return DensityOnlyResult(1000, True)
    mat = Materializer({"nodes": [node()], "edges": []}, backend=full,
        task_id="now", object_knowledge=memory, density_infer=infer)
    mat.query_intrinsic(new_gk("SG"), "obj", "q", crop_rgb="crop")
    assert calls["c3"] == 0 and full.calls == 1
    stats = memory.update(mat._cache, source="mock")
    memory.save()
    assert stats["new"] == 1
    entry, _, _ = memory.lookup_exact("now", "obj")
    assert entry["metadata"]["updated_at"] is not None
    assert entry["metadata"]["source_task"] == "now"
    assert entry["metadata"]["source_episode"] == "episode-1"


def test_stage_protection_and_stale_lookup(tmp_path):
    memory = PropertyMemory(tmp_path / "memory.json", task_id="task")
    high = props(); high["mass_stage"] = 2; high["mass_kg"] = 9.0
    memory.update({"obj": high}, source="probe")
    low = props(); low["mass_stage"] = 0; low["mass_kg"] = 1.0
    memory.update({"obj": low}, source="visual")
    assert memory.lookup("obj")["mass_kg"] == 9.0
    entry, _, _ = memory.lookup_exact("task", "obj")
    assert entry["metadata"]["episodes_seen"] == 2
    memory.invalidate("obj", "contents changed")
    assert memory.lookup("obj") is None
    assert memory.lookup_exact("task", "obj")[2] == "STALE_MEMORY"


def test_cross_task_matching_ignores_identity_text(tmp_path):
    m = ObjectKnowledgeManager(tmp_path / "memory.json")
    a = props(1000); a["caption"] = "completely unrelated words"
    m.update("past", "not-the-query-name", a)
    hit, debug = m.lookup_or_retrieve("now", "different-id", [100, 50, 40], "crop",
        lambda _: DensityOnlyResult(1000, True))
    assert hit is not None and debug["best_match"]["object_id"] == "not-the-query-name"
