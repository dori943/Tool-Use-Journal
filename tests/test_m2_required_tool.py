"""Unbound tool actions cannot borrow the direct-grasp batch verdict."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tuj.m2_subgoal.core import decompose, plan_evaluations
from tuj.m2_subgoal.ingest import apply_grounding, measurement_feedback
from tuj.m2_subgoal.pipeline import run_m2
from tuj.m2_subgoal.regroup import split_by_partition


def _node(oid, x):
    return {
        "id": oid, "class": "object", "center_mm": [x, 0, 20],
        "bbox_mm": [20, 20, 20],
        "geometry": {"extents_mm": [20, 20, 20], "height_z_mm": 20},
        "mass_kg": 0.1,
        "reachability": {"reachable": True, "margin_mm": 100},
        "ee": {"2F": {"feasible": True}},
        "predicates": {"top_exposed": {"value": True},
                       "clear": {"value": True}},
    }


def _scene():
    return {"nodes": [_node("a", 0), _node("b", 40),
                      _node("box", 200), _node("paddle", -100)], "edges": []}


def _proposal(kind="sweep_collect", candidates=None):
    s = {"subgoal_id": "SG1", "kind": kind, "goal": "move a and b",
         "target_ids": ["a", "b"], "container_id": "box",
         "tool_candidate_ids": list(candidates or [])}
    s["details"] = decompose(s)
    return {"task": "move a and b into box", "m2_subgoals": [s],
            "m2_stats": {"n_subgoals": 1, "n_details": len(s["details"]),
                         "n_edges": 0, "n_mutex": 0},
            "m2_partial_order": [], "m2_mutex": []}


@pytest.mark.parametrize("kind", ["sweep_collect", "scoop_transfer", "flatten", "extract"])
def test_empty_required_tool_is_a_feedback_failure_not_direct_batch(kind):
    out = _proposal(kind)
    scene = _scene()
    q = plan_evaluations(out["m2_subgoals"][0],
                         out["m2_subgoals"][0]["details"], scene)
    assert not any(x["call"].get("actor", {}).get("type") == "ee_pool" for x in q)
    apply_grounding(out, scene)
    s = out["m2_subgoals"][0]
    assert s["tool_validation"]["reason_code"] == "MISSING_REQUIRED_TOOL"
    assert s["tool_validation"]["status"] == "REJECTED"
    assert "partition_plan" not in s
    tool_checks = [p for d in s["details"] for p in d["pre"]
                   if p["eval_by"] == "m3" and "?tool" in p["expr"]]
    assert tool_checks and all(p["status"] == "unsat" for p in tool_checks)
    feedback = measurement_feedback(out)
    assert "MISSING_REQUIRED_TOOL" in feedback
    assert "[맨손 수행 불가]" not in feedback
    assert split_by_partition(out) == []


def test_logged_c42_payload_candidates_are_pruned_then_reported_missing():
    # The real run logged five scene IDs, all of them packing targets.
    targets = ["baguette", "cereal", "milk", "rolling_pin", "whisk"]
    scene = {"nodes": [_node(i, k * 40) for k, i in enumerate(targets + ["box"])],
             "edges": []}
    s = {"subgoal_id": "SG1", "kind": "sweep_collect", "goal": "pack",
         "target_ids": targets, "container_id": "box",
         "tool_candidate_ids": targets.copy()}
    rough = SimpleNamespace(generate=lambda *_a: [deepcopy(s)], usage={})
    out = run_m2("pack every object into box", scene, rough=rough)
    apply_grounding(out, scene)
    result = out["m2_subgoals"][0]
    assert result["tool_candidate_ids"] == []  # Existing payload policy is unchanged.
    assert "partition_plan" not in result
    assert "MISSING_REQUIRED_TOOL" in measurement_feedback(out)


def test_direct_transfer_without_external_tool_keeps_ee_batch_partition():
    out = _proposal("relocate")
    apply_grounding(out, _scene())
    s = out["m2_subgoals"][0]
    assert "tool_validation" not in s
    assert s["partition_plan"] == [["a"], ["b"]]
    assert split_by_partition(out)
    apply_grounding(out, _scene())
    assert len(out["m2_subgoals"]) == 2
    assert all("tool_validation" not in x for x in out["m2_subgoals"])
    assert all(d["action_type"] in {"acquire", "transport", "place"}
               for x in out["m2_subgoals"] for d in x["details"])


def test_real_tool_candidates_keep_object_batch_semantics():
    out = _proposal(candidates=["paddle"])
    s = out["m2_subgoals"][0]
    q = plan_evaluations(s, s["details"], _scene())
    actors = [x["call"]["actor"] for x in q if x["call"]["kind"] == "batch"]
    assert actors == [{"type": "object", "id": "paddle"}]
    apply_grounding(out, _scene())
    assert "tool_validation" not in s


def test_direct_stack_without_external_tool_is_not_rejected():
    out = _proposal("stack")
    apply_grounding(out, _scene())
    assert "tool_validation" not in out["m2_subgoals"][0]
    assert "MISSING_REQUIRED_TOOL" not in (measurement_feedback(out) or "")


def test_new_valid_tool_grounding_clears_stale_missing_tool_verdict():
    out = _proposal()
    apply_grounding(out, _scene())
    s = out["m2_subgoals"][0]
    assert s["tool_validation"]["status"] == "REJECTED"
    s["tool_candidate_ids"] = ["paddle"]
    apply_grounding(out, _scene())
    assert "tool_validation" not in s
    assert "MISSING_REQUIRED_TOOL" not in (measurement_feedback(out) or "")


def test_runner_rejects_missing_tool_before_confidence_split_or_packing(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_m2 as runner

    count = {"generation": 0}
    usage = {"rough": {"calls": 2, "tokens": 43}}
    rough = SimpleNamespace(_client=object(), model="mock", usage=usage)
    monkeypatch.setattr(runner, "LLMRough", lambda: rough)
    monkeypatch.setattr(runner, "load_scene", lambda _p: _scene())

    def generate(*_a, **_k):
        count["generation"] += 1
        return _proposal()

    def must_not_run(*_a, **_k):
        pytest.fail("invalid tool plan reached a later stage")

    monkeypatch.setattr(runner, "run_m2", generate)
    monkeypatch.setattr(runner, "update_confidence", must_not_run)
    monkeypatch.setattr(runner, "split_by_partition", must_not_run)
    monkeypatch.setattr(runner, "apply_task_packing_policy", must_not_run)
    monkeypatch.setattr(sys, "argv", ["run_m2.py", "c4_2", "--m1-json", "mock.json",
                                     "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="MISSING_REQUIRED_TOOL"):
        runner.main()
    assert count["generation"] == 2  # Existing identical-feedback stop, no new retry loop.
    assert not (tmp_path / "m2.json").exists()
    rejected = json.loads((tmp_path / "m2_rejected.json").read_text())
    assert rejected["m2_validation"]["reason_code"] == "MISSING_REQUIRED_TOOL"
    assert rejected["m2_stats"]["llm_usage"] == usage
    assert "MISSING_REQUIRED_TOOL" in rough.feedback


def test_existing_feedback_retry_can_recover_with_valid_direct_plan(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_m2 as runner

    count = {"generation": 0}
    rough = SimpleNamespace(_client=object(), model="mock", usage={})
    monkeypatch.setattr(runner, "LLMRough", lambda: rough)
    monkeypatch.setattr(runner, "load_scene", lambda _p: _scene())

    def generate(*_a, **_k):
        count["generation"] += 1
        return _proposal("sweep_collect" if count["generation"] == 1 else "stack")

    monkeypatch.setattr(runner, "run_m2", generate)
    monkeypatch.setattr(runner, "update_confidence", lambda *_a, **_k: [])
    monkeypatch.setattr(sys, "argv", ["run_m2.py", "c1_2", "--m1-json", "mock.json",
                                     "--output-dir", str(tmp_path)])
    runner.main()
    assert count["generation"] == 2
    assert "MISSING_REQUIRED_TOOL" in rough.feedback
    assert not (tmp_path / "m2_rejected.json").exists()
    result = json.loads((tmp_path / "m2.json").read_text())
    assert result["m2_subgoals"][0]["kind"] == "stack"
    assert "tool_validation" not in result["m2_subgoals"][0]
