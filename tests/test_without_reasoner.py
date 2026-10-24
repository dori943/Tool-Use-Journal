# -*- coding: utf-8 -*-
"""w/o Reasoner (vlm_prior) 정책 단위 테스트 — API 호출 없음."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _robot():
    return json.loads((ROOT / "configs" / "robot_spec.json").read_text(encoding="utf-8"))


def _full_m1():
    """Full M1 payload 축소판: 측정값(Reasoner)이 들어 있는 두 물체와 near edge 하나."""
    def node(nid, cls, center, bbox):
        return {
            "id": nid, "canonical_id": cls, "class": cls, "source_name": cls,
            "center_mm": center, "bbox_mm": bbox,
            "geometry": {"length_mm": 102.0, "diameter_mm": 82.4, "extents_mm": [102.0, 86.5, 78.3],
                         "height_z_mm": 78.4, "cylinder_like": True, "surface_rms_mm": 2.84,
                         "seal_patch_rms_mm": 1.218, "footprint_mm": bbox[:2], "height_mm": bbox[2],
                         "center": center, "aabb_size": bbox},
            "material": "ceramic", "density_kgm3": 3101.7, "mass_kg": 0.147, "youngs_gpa": 88.87,
            "mu": {"mu": 0.492, "stage": 0, "material": "ceramic", "rms_mm": 2.84},
            "confidence": 0.33, "geometry_source": "current_observation", "intrinsic_source": "backend",
            "mass_range_kg": [0.1, 0.2], "materials_topk": ["ceramic"], "caption": "a mug",
            "grounding_source": "backend",
            "ee": {"2F": {"feasible": True}, "3F": {"feasible": True}, "vac": {"feasible": False}},
            "reachability": {"pass": True}, "predicates": {"flat_face": {"pass": True}},
        }
    return {
        "nodes": [node("obj_mug_mug", "mug", [638.8, 470.7, 852.2], [72.8, 100.0, 78.4]),
                  node("obj_plate_plate", "plate", [500.0, 300.0, 840.0], [182.0, 182.0, 11.0])],
        "edges": [{"from": "obj_mug_mug", "to": "obj_plate_plate", "type": "near", "gap_mm": 51.2}],
    }


def _estimate(mass_kg=0.2, extents=(100.0, 85.0, 80.0), plate_extents=(180.0, 180.0, 12.0)):
    from tuj.ablations.vlm_prior import SceneEstimate
    return SceneEstimate.model_validate({
        "objects": [
            {"id": "obj_mug_mug", "extents_mm": list(extents), "cylinder_like": True,
             "surface_rms_mm": 3.0, "seal_patch_rms_mm": 6.0, "material": "ceramic",
             "density_kgm3": 2500.0, "mass_kg": mass_kg, "youngs_gpa": 70.0, "mu": 0.5, "confidence": 0.6},
            {"id": "obj_plate_plate", "extents_mm": list(plate_extents), "cylinder_like": True,
             "surface_rms_mm": 0.3, "seal_patch_rms_mm": 0.2, "material": "ceramic",
             "density_kgm3": 2500.0, "mass_kg": 0.4, "youngs_gpa": 70.0, "mu": 0.4, "confidence": 0.6},
        ],
        "edges": [{"from": "obj_mug_mug", "to": "obj_plate_plate", "gap_mm": 30.0}],
    })


def test_identity_view_hides_measurements():
    from tuj.ablations.vlm_prior import identity_view
    view = identity_view(_full_m1())
    assert [o["id"] for o in view["objects"]] == ["obj_mug_mug", "obj_plate_plate"]
    for o in view["objects"]:
        assert set(o) <= {"id", "class", "source_name", "is_container"}
    assert view["edges"] == [{"from": "obj_mug_mug", "to": "obj_plate_plate", "type": "near"}]


def test_prior_scene_replaces_measurements_keeps_observation():
    from tuj.ablations.vlm_prior import build_prior_scene, feasible_ee_table
    full = _full_m1()
    prior = build_prior_scene(full, _estimate(), _robot())
    mug = next(n for n in prior["nodes"] if n["id"] == "obj_mug_mug")
    # 관측값 유지
    assert mug["center_mm"] == [638.8, 470.7, 852.2] and mug["bbox_mm"] == [72.8, 100.0, 78.4]
    assert mug["geometry"]["center"] == [638.8, 470.7, 852.2]
    # 측정값 대체
    assert mug["geometry"]["extents_mm"] == [100.0, 85.0, 80.0]
    assert mug["geometry"]["footprint_mm"] == [100.0, 85.0]
    assert mug["mass_kg"] == 0.2 and mug["mu"]["mu"] == 0.5
    assert mug["grounding_source"] == "vlm_prior"
    for k in ("mass_range_kg", "materials_topk", "caption"):
        assert k not in mug
    # EE 판정은 Full 과 같은 규칙으로 다시 계산됨 (ee_pool 전체에 대해 verdict 존재)
    assert set(mug["ee"]) == {e["ee_id"] for e in _robot()["ee_pool"]}
    assert all({"feasible", "margin", "reason", "checks"} <= set(v) for v in mug["ee"].values())
    # 접시: 흡착 평면 OK → vac 통과, 2F/3F 는 개구 초과로 탈락해야 함
    table = feasible_ee_table(prior)
    assert table["obj_plate_plate"] and all("vac" in e.lower() or "suction" in e.lower()
                                            for e in table["obj_plate_plate"])
    # edge gap 대체
    assert prior["edges"][0]["gap_mm"] == 30.0 and prior["edges"][0]["gap_source"] == "vlm_prior"
    assert prior["ablation"]["mode"] == "without_reasoner"
    # 술어는 새 geometry 로 재계산
    assert "flat_face" in mug["predicates"]
    # 원본은 변경되지 않음
    assert full["nodes"][0]["mass_kg"] == 0.147


def test_prior_scene_always_yields_a_feasible_ee():
    """추정값으로는 어떤 EE 도 못 잡는 물체 → margin 최대 EE 하나를 fallback 으로 통과."""
    from tuj.ablations.vlm_prior import build_prior_scene, feasible_ee_table
    # 너무 무겁고(페이로드 초과) 너무 넓고(개구 초과) 상면이 울퉁불퉁한(흡착 불가) 물체
    est = _estimate(mass_kg=50.0, extents=(400.0, 400.0, 300.0))
    prior = build_prior_scene(_full_m1(), est, _robot())
    mug = next(n for n in prior["nodes"] if n["id"] == "obj_mug_mug")
    table = feasible_ee_table(prior)
    assert len(table["obj_mug_mug"]) == 1
    assert mug["ee_fallback"] == table["obj_mug_mug"][0]
    assert prior["ablation"]["ee_fallback"] == {"obj_mug_mug": mug["ee_fallback"]}
    assert "vlm_prior_fallback" in mug["ee"][mug["ee_fallback"]]["reason"]


def test_estimate_validation_requires_full_coverage():
    from tuj.ablations.vlm_prior import SceneEstimate
    with pytest.raises(Exception):
        SceneEstimate.model_validate({"objects": [{"id": "x"}], "edges": []})


def test_run_dispatch_routes_without_reasoner(monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import run as pipeline

    captured = []
    monkeypatch.setattr(pipeline, "_resolve_llm", lambda args: None)
    import run_without_reasoner as runner
    monkeypatch.setattr(runner, "run", lambda args, p: captured.append((args.grounding_mode, p)))
    monkeypatch.setattr(sys, "argv", ["run.py", "c2_1", "--grounding-mode", "without_reasoner"])
    pipeline.main()
    assert captured and captured[0][0] == "without_reasoner" and captured[0][1] is pipeline
