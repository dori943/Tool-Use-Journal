"""Isolation and real M4/M5 boundary tests; no API or simulator required."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from assemble_gk import assemble
from run_without_grounding import digest, prepare_directory, verify_artifact, write

from tuj.ablations.without_grounding import (
    MODE,
    POLICY_VERSION,
    VisualSemanticRough,
    build_m2,
    build_m4_request,
    decision_robot,
    decision_scene,
)
from tuj.m4_taskplanner.planner import plan
from tuj.m4_taskplanner.serialization import dump_result


def source():
    return json.loads((ROOT / "output/c1_2/m1.json").read_text(encoding="utf-8"))


def robot():
    return json.loads((ROOT / "configs/robot_spec.json").read_text(encoding="utf-8"))


class FakeClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(next(self.outputs)))
                )
            ],
        )


def responses(ees=None):
    return [
        [
            {
                "subgoal_id": "SG1",
                "goal": "Flatten dough",
                "kind": "flatten",
                "target_ids": ["obj_dough_dough"],
                "container_id": "obj_cutting_board_cutting_board",
                "ordered": False,
                "confidence": 0.6,
            }
        ],
        [
            {
                "subgoal_id": "SG1",
                "selected_tool_id": "obj_spatula_spatula",
                "ee_candidates_by_object": {
                    "obj_spatula_spatula": ["3F"] if ees is None else ees
                },
                "reason": "visual hypothesis",
                "confidence": 0.5,
            }
        ],
    ]


def build(tmp_path, ees=None):
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"test-image")
    client = FakeClient(responses(ees))
    rough = VisualSemanticRough("test-model", frame, robot(), client)
    m2 = build_m2("Flatten dough", source(), rough)
    request, bundle = build_m4_request(source(), m2, robot(), assemble)
    return m2, request, bundle, client


def test_allowlist_removes_nested_and_duplicate_measurements():
    raw = source()
    original = copy.deepcopy(raw)
    raw["nodes"][0]["future_grounding"] = {"secret_measurement": 123.456}
    raw["geometry_metadata"] = {"other": "secret"}
    safe = decision_scene(raw)
    assert safe["edges"] == []
    assert all(
        set(n) <= {"id", "canonical_id", "class", "source_name"} for n in safe["nodes"]
    )
    assert source() == original
    unsafe_robot = robot()
    unsafe_robot["facts"] = [{"flat_face": True}]
    unsafe_robot["ee_pool"][0]["compatible_tools"] = ["correct_answer"]
    assert "facts" not in decision_robot(unsafe_robot)
    assert "compatible_tools" not in decision_robot(unsafe_robot)["ee_pool"][0]


def test_real_m4_search_and_m5_contract(tmp_path):
    m2, request, bundle, client = build(tmp_path)
    assert m2["m2_subgoals"][0]["selected_tool_id"] == "obj_spatula_spatula"
    assert all(
        p["status"] == "unknown"
        for s in m2["m2_subgoals"]
        for d in s["details"]
        for p in d["pre"]
        if p["eval_by"] == "m3"
    )
    assert all(s.feasible_ee == ["3F"] for s in request.task_graph.subgoals)
    assert all(
        o.mass_kg is None and o.bbox_mm is None
        for o in request.resource_catalog.objects.values()
    )
    assert "geometry" not in json.dumps(bundle)
    assert len(client.requests) == 2
    for call in client.requests:
        assert call["messages"][0]["content"][1]["type"] == "image_url"
        assert "surface_rms_mm" not in json.dumps(call)
        assert "mass_range_kg" not in json.dumps(call)
    result = plan(request)
    assert result.status.value == "SUCCESS", result.model_dump()
    output = tmp_path / "m4.json"
    dump_result(result, output)
    from run_m5 import main

    assert (
        main(
            [
                "--task-planner",
                str(output),
                "--initial-world",
                str(ROOT / "output/c1_2/m5/initial_world.json"),
                "--output-dir",
                str(tmp_path / "m5"),
                "--validate-input-only",
            ]
        )
        == 0
    )


def test_wrong_ee_is_not_corrected_by_ground_truth(tmp_path):
    m2, request, _, _ = build(tmp_path, ["vac"])
    assert all(s.feasible_ee == ["vac"] for s in request.task_graph.subgoals)
    assert m2["m2_subgoals"][0]["selection_by"] == "visual_semantic"


def test_empty_hypothesis_is_failure_not_fallback(tmp_path):
    _, request, _, _ = build(tmp_path, [])
    result = plan(request)
    assert result.selected_plan is None


def test_foreign_m2_rejected(tmp_path):
    m2, _, _, _ = build(tmp_path)
    m2.pop("ablation")
    with pytest.raises(ValueError, match="M2 artifact"):
        build_m4_request(source(), m2, robot(), assemble)


def test_resume_checks_mode_and_content(tmp_path):
    identity = {"mode": MODE, "policy_version": POLICY_VERSION}
    out = tmp_path / "run"
    manifest = prepare_directory(out, identity, False)
    write(out / "m2.json", {"selected": "hypothesis"})
    manifest["artifacts"]["m2.json"] = digest(out / "m2.json")
    write(out / "ablation_manifest.json", manifest)
    verify_artifact(out, manifest, "m2.json")
    assert prepare_directory(out, identity, True) == manifest
    with pytest.raises(ValueError):
        prepare_directory(out, {"mode": "full"}, True)
    with pytest.raises(ValueError):
        prepare_directory(out, identity, False)
    write(out / "m2.json", {"selected": "full-answer"})
    with pytest.raises(ValueError, match="foreign"):
        verify_artifact(out, manifest, "m2.json")


def test_runner_routes_and_resumes_without_full_stages(tmp_path, monkeypatch):
    import run as pipeline
    import run_without_grounding as runner

    raw = tmp_path / "source.json"
    write(raw, source())
    frame = tmp_path / "source.png"
    frame.write_bytes(b"test-image")
    output = tmp_path / "experiment"
    args = pipeline.build_parser().parse_args(
        [
            "c1_2",
            "--grounding-mode",
            MODE,
            "--m1-json",
            str(raw),
            "--scene-frame",
            str(frame),
            "--output-dir",
            str(output),
            "--model",
            "test-model",
            "--stop-after",
            "m4",
        ]
    )
    monkeypatch.setenv("TUJ_LLM_PROVIDER", "openai")
    client = FakeClient(responses())
    monkeypatch.setattr(
        runner,
        "VisualSemanticRough",
        lambda model, image, spec: VisualSemanticRough(model, image, spec, client),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("the full M2/M4 path must not run")

    monkeypatch.setattr(pipeline, "stage_m2", forbidden)
    monkeypatch.setattr(pipeline, "stage_m4", forbidden)
    runner.run(args, pipeline)
    m4 = json.loads((output / "m4.json").read_text(encoding="utf-8"))
    assert m4["status"] == "SUCCESS"
    assert (
        m4["selected_plan"]["candidate_assignments"][0]["ee_feasible_set_source"]
        == "visual_semantic_unverified"
    )
    assert (output / "execution/m1.json").read_bytes() == raw.read_bytes()
    assert "mass_kg" not in (output / "m1.json").read_text()
    monkeypatch.setenv("MOTION_PLANNER_KEYFRAME_CACHE", "full-cache")
    attempts = []

    def mock_m5(task, out, options):
        import os

        attempts.append(out)
        assert os.environ["MOTION_PLANNER_KEYFRAME_CACHE"] == str(
            out / "keyframe_cache"
        )
        assert (out / "m4.json").read_bytes() == (output / "m4.json").read_bytes()

    monkeypatch.setattr(pipeline, "stage_m5", mock_m5)
    args.start_from, args.stop_after = "m5", "m5"
    runner.run(args, pipeline)
    runner.run(args, pipeline)
    assert attempts[0] != attempts[1]
    import os

    assert os.environ["MOTION_PLANNER_KEYFRAME_CACHE"] == "full-cache"
    # Re-preparing only M1 cannot leave an old M2/M4 eligible for resume.
    args.start_from, args.stop_after = "m1", "m1"
    runner.run(args, pipeline)
    args.start_from, args.stop_after = "m4", "m4"
    with pytest.raises(ValueError, match="m2.json"):
        runner.run(args, pipeline)


def test_main_dispatch_defaults_to_full(monkeypatch, tmp_path):
    import run as pipeline
    import run_without_grounding as runner

    captured = []
    monkeypatch.setattr(pipeline, "_resolve_llm", lambda args: None)
    monkeypatch.setattr(
        pipeline, "_run_integrated", lambda *args: captured.append("full")
    )
    monkeypatch.setattr(runner, "run", lambda *args: captured.append("ablation"))
    monkeypatch.setattr(sys, "argv", ["run.py", "c1_2", "--output-dir", str(tmp_path)])
    pipeline.main()
    monkeypatch.setattr(sys, "argv", ["run.py", "c1_2", "--grounding-mode", MODE])
    pipeline.main()
    assert captured == ["full", "ablation"]


def test_full_cannot_overwrite_ablation_output(monkeypatch, tmp_path):
    import run as pipeline

    write(tmp_path / "ablation_manifest.json", {})
    monkeypatch.setattr(pipeline, "_resolve_llm", lambda args: None)
    monkeypatch.setattr(sys, "argv", ["run.py", "c1_2", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="belongs to an ablation"):
        pipeline.main()


def test_visual_response_rejects_unknown_ids_and_retries(tmp_path):
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"image")
    good = responses()
    invalid = copy.deepcopy(good[1])
    invalid[0]["selected_tool_id"] = "invented_tool"
    client = FakeClient([good[0], invalid, good[1]])
    rough = VisualSemanticRough("test-model", frame, robot(), client)
    result = build_m2("Flatten dough", source(), rough)
    assert result["m2_subgoals"][0]["selected_tool_id"] == "obj_spatula_spatula"
    assert len(client.requests) == 3
