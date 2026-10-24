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


def _estimate(mass_kg=0.2, extents=(100.0, 85.0, 80.0), plate_extents=(180.0, 180.0, 12.0),
              height_z_mm=80.0, footprint_mm=(100.0, 85.0)):
    from tuj.ablations.vlm_prior import SceneEstimate
    return SceneEstimate.model_validate({
        "objects": [
            {"id": "obj_mug_mug", "extents_mm": list(extents), "cylinder_like": True,
             "height_z_mm": height_z_mm, "footprint_mm": list(footprint_mm),
             "surface_rms_mm": 3.0, "seal_patch_rms_mm": 6.0, "material": "ceramic",
             "density_kgm3": 2500.0, "mass_kg": mass_kg, "youngs_gpa": 70.0, "mu": 0.5, "confidence": 0.6},
            {"id": "obj_plate_plate", "extents_mm": list(plate_extents), "cylinder_like": True,
             "height_z_mm": 12.0, "footprint_mm": [180.0, 180.0],
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


@pytest.mark.parametrize("height,footprint", [(120.0, (100.0, 100.0)),
                                             (100.0, (100.0, 120.0))])
def test_geometry_preserves_estimated_vertical_axis(height, footprint):
    from tuj.ablations.vlm_prior import build_prior_scene
    est = _estimate(extents=(100.0, 100.0, 120.0), height_z_mm=height,
                    footprint_mm=footprint)
    mug = build_prior_scene(_full_m1(), est, _robot())["nodes"][0]
    assert mug["geometry"]["extents_mm"] == [120.0, 100.0, 100.0]
    assert mug["geometry"]["height_z_mm"] == height
    assert mug["geometry"]["height_mm"] == height
    assert mug["geometry"]["footprint_mm"] == list(footprint)
    assert mug["bbox_mm"] == _full_m1()["nodes"][0]["bbox_mm"]


@pytest.mark.parametrize("kwargs", [
    {"height_z_mm": 0}, {"height_z_mm": float("nan")},
    {"footprint_mm": (100, -2)}, {"footprint_mm": (100, float("inf"))},
    {"footprint_mm": (100,)}, {"extents": (100, 0, 20)},
])
def test_estimated_dimensions_must_be_positive_and_finite(kwargs):
    with pytest.raises(ValueError):
        _estimate(**kwargs)


def _runner_case(tmp_path, monkeypatch, with_keepout=True):
    from types import SimpleNamespace
    import numpy as np
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import run_without_reasoner as runner
    from tuj.common.ee_keepout import SweptArmHeightField, save_height_field
    from tuj.m2_subgoal.ground import load_scene

    def m1(task, execution, args):
        assert args.backend == "mock" and args.memory == "none"
        (execution / "m1.json").write_text(json.dumps(_full_m1()))
        (execution / "frame.png").write_bytes(b"frame supplied to mocked VLM")
        np.savez(execution / "m1_points.npz", points=np.array([1, 2, 3]))
        if with_keepout:
            save_height_field(execution / "m1_ee_keepout.npz", SweptArmHeightField(
                origin_xy=np.array([0., 0.]), cell_m=.01,
                floor_z=np.full((2, 2), 1.0), trajectory_ids=("test-sweep",)))

    def m2(task, out, args):
        scene = load_scene(out / "m1.json")
        if with_keepout:
            assert scene["_ee_keepout"].trajectory_ids == ("test-sweep",)
        else:
            assert scene["_ee_keepout"] is None
        assert not (out / "m1_points.npz").exists()
        (out / "m2.json").write_text(json.dumps({"m2_stats": {}}))

    class Estimator:
        usage = {}

        def __init__(self, *args):
            pass

        def estimate(self, *args):
            return _estimate()

    monkeypatch.setattr(runner, "VLMPriorEstimator", Estimator)
    args = SimpleNamespace(
        planner_mode="full", m5_physical=False, m5_args=[], start_from=None,
        stop_after="m2", task="c2_1", output_dir=tmp_path / "run", seed=0,
        backend="unused", memory="unused", scene_frame=None,
        robot_spec=ROOT / "configs/robot_spec.json", model="mock")
    pipeline = SimpleNamespace(ROOT=ROOT, STAGES=("m1", "m2", "m4", "m5"),
                               stage_m1=m1, stage_m2=m2)
    return runner, args, pipeline


@pytest.mark.parametrize("with_keepout", [True, False])
def test_reasoner_runner_passes_only_keepout_to_m2(tmp_path, monkeypatch, with_keepout):
    runner, args, pipeline = _runner_case(tmp_path, monkeypatch, with_keepout)
    runner.run(args, pipeline)
    manifest = json.loads((args.output_dir / "ablation_manifest.json").read_text())
    assert ("m1_ee_keepout.npz" in manifest["artifacts"]) is with_keepout


@pytest.mark.parametrize("change", ["remove", "modify", "unexpected"])
def test_resume_checks_keepout_provenance(tmp_path, monkeypatch, change):
    runner, args, pipeline = _runner_case(tmp_path, monkeypatch, change != "unexpected")
    runner.run(args, pipeline)
    keepout = args.output_dir / "m1_ee_keepout.npz"
    if change == "remove":
        keepout.unlink()
    else:
        keepout.write_bytes(b"different sweep")
    args.start_from = "m2"
    with pytest.raises(ValueError, match="m1_ee_keepout"):
        runner.run(args, pipeline)


def test_resume_rejects_old_geometry_policy(tmp_path, monkeypatch):
    runner, args, pipeline = _runner_case(tmp_path, monkeypatch)
    runner.run(args, pipeline)
    path = args.output_dir / "ablation_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["policy_version"] = "vlm-prior-attributes-v1"
    path.write_text(json.dumps(manifest))
    args.start_from = "m2"
    with pytest.raises(ValueError, match="policy_version"):
        runner.run(args, pipeline)


def test_current_policy_can_resume_with_matching_keepout(tmp_path, monkeypatch):
    runner, args, pipeline = _runner_case(tmp_path, monkeypatch)
    runner.run(args, pipeline)
    args.start_from = "m2"
    runner.run(args, pipeline)
    manifest = json.loads((args.output_dir / "ablation_manifest.json").read_text())
    assert manifest["stages"]["m2"] == "completed"


def test_vlm_request_and_response_include_pose_dimensions(tmp_path):
    from types import SimpleNamespace
    from tuj.ablations.vlm_prior import VLMPriorEstimator, identity_view
    expected = _estimate(extents=(100, 100, 120), height_z_mm=120,
                         footprint_mm=(100, 100))
    requests = []

    def create(**request):
        requests.append(request)
        message = SimpleNamespace(content=expected.model_dump_json(by_alias=True))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    frame = tmp_path / "frame.png"
    frame.write_bytes(b"test image; no external API")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = VLMPriorEstimator("mock", frame, client=client).estimate(
        identity_view(_full_m1()), "Move the mug")
    assert result.objects[0].height_z_mm == 120
    prompt = requests[0]["messages"][0]["content"][0]["text"]
    assert "height_z_mm" in prompt and "footprint_mm" in prompt
    assert "NOT world x/y/z axes" in prompt
