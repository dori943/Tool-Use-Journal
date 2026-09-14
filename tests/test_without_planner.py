"""Independent-assignment semantics and M5 contract, without an API call."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tuj.ablations.without_planner import METHOD, independent_plan
from tuj.m4_taskplanner.gk_adapter import build_request_from_gk
from tuj.m4_taskplanner.models import (
    CandidateProposal,
    InitialState,
    OrderConstraints,
    ResourceCatalog,
    Subgoal,
    TaskGraph,
    TaskPlannerRequest,
)
from tuj.m4_taskplanner.serialization import dump_result


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def c1_1_request():
    out = ROOT / "output" / "c1_1"
    return build_request_from_gk(
        read(out / "gk_bundle.json"),
        read(out / "m2.json"),
        m1_payload=read(out / "m1.json"),
        robot_spec_payload=read(ROOT / "configs" / "robot_spec.json"),
    )


def two_choice_request():
    catalog = ResourceCatalog.model_validate(
        {
            "end_effectors": {
                "A": {"capabilities": ["grip"], "home_slot": "SA"},
                "B": {"capabilities": ["grip"], "home_slot": "SB"},
            },
            "objects": {"box": {"mass_kg": 0.1}},
        }
    )
    subgoals = [
        Subgoal(
            subgoal_id=sid,
            action_type="relocate",
            target_ids=["box"],
            feasible_ee=["A", "B"],
        )
        for sid in ("SG1", "SG2")
    ]
    proposals = {
        "SG1": [
            CandidateProposal(candidate_id="a-B", subgoal_id="SG1", ee="B"),
            CandidateProposal(candidate_id="z-A", subgoal_id="SG1", ee="A"),
        ],
        "SG2": [
            CandidateProposal(candidate_id="a-A", subgoal_id="SG2", ee="A"),
            CandidateProposal(candidate_id="z-B", subgoal_id="SG2", ee="B"),
        ],
    }
    return TaskPlannerRequest(
        task_graph=TaskGraph(
            initial_state=InitialState(current_ee="A"), subgoals=subgoals
        ),
        resource_catalog=catalog,
        candidate_proposals=proposals,
    )


def test_independent_choice_is_not_global_ee_minimization(monkeypatch):
    from tuj.m4_taskplanner import planner, search

    def forbidden(*args, **kwargs):
        pytest.fail("M4 joint planner/search was invoked")

    monkeypatch.setattr(planner, "plan", forbidden)
    monkeypatch.setattr(search, "run_search", forbidden)
    result = independent_plan(two_choice_request())
    assert result.status.value == "SUCCESS"
    assert result.selected_plan.subgoal_order == ["SG1", "SG2"]
    assert [a.ee for a in result.selected_plan.candidate_assignments] == ["B", "A"]
    assert result.model_extra["metrics"]["N_EE"] == 2
    assert result.model_extra["metrics"]["N_tool"] == 0
    assert result.task["m4_invoked"] is False
    assert result.search_stats.states_expanded == 0


def test_fixed_m2_order_is_not_reordered_to_satisfy_constraints():
    request = two_choice_request()
    request.task_graph.order_constraints = OrderConstraints(edges=[["SG2", "SG1"]])
    result = independent_plan(request)
    assert result.status.value == "INFEASIBLE_NO_PLAN"
    assert result.selected_plan is None
    assert result.rejections[-1].subgoal_id == "SG1"


def test_real_c1_1_m2_gk_to_m5_contract(tmp_path, monkeypatch):
    from tuj.m4_taskplanner import planner, search

    monkeypatch.setattr(planner, "plan", lambda *a, **k: pytest.fail("plan() called"))
    monkeypatch.setattr(
        search, "run_search", lambda *a, **k: pytest.fail("run_search() called")
    )
    request = c1_1_request()
    result = independent_plan(request)
    assert result.status.value == "SUCCESS"
    assert result.selected_plan.subgoal_order == [
        s.subgoal_id for s in request.task_graph.subgoals
    ]
    assert result.model_extra["method"] == METHOD
    plan = tmp_path / "m4.json"
    dump_result(result, plan)
    from run_m5 import main

    assert (
        main(
            [
                "--task-planner",
                str(plan),
                "--initial-world",
                str(ROOT / "output" / "c1_1" / "m5" / "initial_world.json"),
                "--output-dir",
                str(tmp_path / "m5"),
                "--validate-input-only",
            ]
        )
        == 0
    )
    assert read(tmp_path / "m5" / "m5_summary.json")["status"] == "INPUT_VALIDATED"


def test_cli_default_and_separate_baseline_route(monkeypatch, tmp_path):
    import run as pipeline
    import run_without_planner as runner

    calls = []
    monkeypatch.setattr(pipeline, "_resolve_llm", lambda args: None)
    monkeypatch.setattr(pipeline, "_run_integrated", lambda *a: calls.append("full"))
    monkeypatch.setattr(runner, "run", lambda *a: calls.append("without-planner"))
    monkeypatch.setattr(sys, "argv", ["run.py", "c1_1", "--output-dir", str(tmp_path)])
    pipeline.main()
    monkeypatch.setattr(sys, "argv", ["run.py", "c1_1", "--planner-mode", METHOD])
    pipeline.main()
    assert calls == ["full", "without-planner"]


def test_runner_logs_skipped_m4_and_rejects_foreign_m5_plan(tmp_path, monkeypatch):
    import run as pipeline
    import run_without_planner as runner

    source = ROOT / "output" / "c1_2"
    for name in ("m1.json", "m2.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    args = pipeline.build_parser().parse_args(
        [
            "c1_2",
            "--planner-mode",
            METHOD,
            "--start-from",
            "m4",
            "--stop-after",
            "m4",
            "--output-dir",
            str(tmp_path),
            "--seed",
            "0",
        ]
    )
    monkeypatch.setattr(pipeline, "stage_m4", lambda *a: pytest.fail("M4 invoked"))
    runner.run(args, pipeline)
    report = read(tmp_path / "without_planner_result.json")
    assert report["method"] == METHOD and report["seed"] == 0
    assert report["plan_status"] == "SUCCESS" and report["m4_invoked"] is False
    assert read(tmp_path / "ablation_manifest.json")["stages"]["m4"] == "skipped"
    assert read(tmp_path / "m4.json")["search_stats"]["states_expanded"] == 0
    report["m5_failure_detail"] = "old attempt"
    (tmp_path / "without_planner_result.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    args.start_from, args.stop_after = "m5", "m5"
    args.m5_args = ["--stop-after-subgoal", "SG1_s1_d1"]

    def partial_m5(task, out, args):
        (out / "m5").mkdir(exist_ok=True)
        (out / "m5" / "m5_summary.json").write_text(
            '{"status": "SUCCESS"}', encoding="utf-8"
        )

    monkeypatch.setattr(pipeline, "stage_m5", partial_m5)
    runner.run(args, pipeline)
    report = read(tmp_path / "without_planner_result.json")
    assert report["m5_status"] == "SUCCESS"
    assert report["evaluation_scope"] == "partial" and report["success"] is None
    assert "m5_failure_detail" not in report
    (tmp_path / "m4.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unchanged independent plan"):
        runner.run(args, pipeline)


def test_runner_rejects_m2_from_obsolete_instruction(tmp_path, monkeypatch):
    import run as pipeline
    import run_without_planner as runner

    source = ROOT / "output" / "c4_2"
    for name in ("m1.json", "m2.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    args = pipeline.build_parser().parse_args(
        [
            "c4_2",
            "--planner-mode",
            METHOD,
            "--start-from",
            "m4",
            "--stop-after",
            "m4",
            "--output-dir",
            str(tmp_path),
        ]
    )
    monkeypatch.setattr(pipeline, "stage_gk", lambda *a: pytest.fail("stale M2 reached G_k"))
    with pytest.raises(ValueError, match="M2 task instruction mismatch for c4_2"):
        runner.run(args, pipeline)
    manifest = read(tmp_path / "ablation_manifest.json")
    assert manifest["failure"]["stage"] == "m2"
