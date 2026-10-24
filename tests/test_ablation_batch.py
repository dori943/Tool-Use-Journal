"""Offline regressions for the ablation error reports."""

import json

import pytest

from scripts import run_ablation_batch as batch


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def collect(out, exit_code=1):
    return batch.collect_record(
        out,
        task="c3_2",
        seed=0,
        method="without-reasoner",
        exit_code=exit_code,
        duration_s=5400,
        started_at="2026-10-24T17:00:00",
    )


@pytest.mark.parametrize("done", ["prepared", "resumed", "completed"])
def test_timeout_uses_active_stage_not_prepared_m1(tmp_path, done):
    write_json(
        tmp_path / "ablation_manifest.json",
        {
            "stages": {
                "m1": done,
                "vlm_prior": "completed",
                "m2": "completed",
                "m4": "SUCCESS",
                "m5": "running",
            }
        },
    )
    record = collect(tmp_path, -9)
    assert record["failure_stage"] == "m5"
    assert record["reason_code"] == "BATCH_TIMEOUT"
    assert record["success"] is False


def test_scripted_grasp_invalid_input_is_collected(tmp_path):
    (tmp_path / "run.log").write_text(
        "task_planner: invalid input: SCRIPTED_GRASP_EE_INFEASIBLE: "
        "cereal supports ['2F'], grounded feasible EEs=['vac']\n"
    )
    record = collect(tmp_path)
    assert record["reason_code"] == "SCRIPTED_GRASP_EE_INFEASIBLE"


def test_goal_failure_is_not_overridden_by_execution_success(tmp_path):
    write_json(
        tmp_path / "run_result.json",
        {
            "success": False,
            "exec_success": True,
            "goal_success": False,
            "failure_stage": "goal",
            "failure_code": "GOAL_NOT_ACHIEVED",
            "m5_status": "SUCCESS",
            "task_goal_status": "FAILED",
        },
    )
    record = collect(tmp_path, 0)
    assert record["failure_stage"] == "goal"
    assert record["reason_code"] == "GOAL_NOT_ACHIEVED"


def test_summarize_refreshes_latest_record_using_copied_windows_log(tmp_path):
    out = tmp_path / "c4_2" / "seed_0"
    old = {
        "task": "c4_2",
        "seed": 0,
        "rep": 0,
        "exit_code": 1,
        "duration_s": 40,
        "started_at": "2026-10-24",
        "reason_code": None,
        "log": r"C:\Users\user\repo\output\c4_2\logs\seed_0.log",
    }
    write_json(out / "batch_record.json", old)
    (out / "batch_records.jsonl").write_text(json.dumps(old) + "\n")
    (out / "seed_0.log").write_text(
        "task_planner: invalid input: SCRIPTED_GRASP_EE_INFEASIBLE: cereal\n"
    )
    rows = batch.load_all_records(tmp_path, "without-reasoner")
    assert len(rows) == 1
    assert rows[0]["reason_code"] == "SCRIPTED_GRASP_EE_INFEASIBLE"
    assert rows[0]["rep"] == 0
    assert json.loads((out / "batch_record.json").read_text()) == old


def test_refresh_does_not_replace_earlier_m5_repetitions(tmp_path):
    out = tmp_path / "c3_2" / "seed_0"
    out.mkdir(parents=True)
    previous = {
        "task": "c3_2",
        "seed": 0,
        "rep": 0,
        "success": True,
        "failure_stage": "none",
        "reason_code": None,
    }
    latest = {"task": "c3_2", "seed": 0, "rep": 1, "exit_code": -9}
    (out / "batch_records.jsonl").write_text(
        json.dumps(previous) + "\n" + json.dumps(latest) + "\n"
    )
    write_json(out / "ablation_manifest.json", {"stages": {"m5": "running"}})
    rows = batch.load_all_records(tmp_path, "without-reasoner")
    assert rows[0] == previous
    assert rows[1]["reason_code"] == "BATCH_TIMEOUT"


def test_report_distinguishes_goal_not_evaluated_from_success(tmp_path):
    write_json(
        tmp_path / "run_result.json",
        {
            "success": True,
            "exec_success": True,
            "goal_success": None,
            "failure_stage": "none",
            "m5_status": "SUCCESS",
            "task_goal_status": "NOT_EVALUATED",
        },
    )
    batch.write_summaries(tmp_path, [collect(tmp_path, 0)])
    report = (tmp_path / "errors_summary.md").read_text()
    assert "목표 미평가" in report
    assert "목표 달성" in report


def test_timeout_does_not_reuse_previous_attempt_success(tmp_path):
    write_json(
        tmp_path / "run_result.json",
        {
            "success": True,
            "exec_success": True,
            "goal_success": True,
            "failure_stage": "none",
            "m5_status": "SUCCESS",
        },
    )
    write_json(
        tmp_path / "ablation_manifest.json",
        {"stages": {"m1": "prepared", "m5": "running"}},
    )
    record = collect(tmp_path, -9)
    assert record["failure_stage"] == "m5"
    assert record["success"] is False
    assert record["exec_success"] is False
    assert record["goal_success"] is None


def test_explicit_log_precedes_stale_generic_log(tmp_path):
    out = tmp_path / "seed_0"
    out.mkdir()
    (out / "run.log").write_text("RuntimeError: OLD_FAILURE\n")
    current = tmp_path / "current.log"
    current.write_text("task_planner: invalid input: SCRIPTED_GRASP_EE_INFEASIBLE\n")
    record = batch.collect_record(
        out,
        task="c4_2",
        seed=0,
        method="without-reasoner",
        exit_code=1,
        duration_s=1,
        started_at=None,
        log=current,
    )
    assert record["reason_code"] == "SCRIPTED_GRASP_EE_INFEASIBLE"
