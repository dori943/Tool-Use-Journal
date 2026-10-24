"""Read local runtime artifacts without replaying physics or calling a model."""
import hashlib
import json
from pathlib import Path

from .task_goals import POLICY_VERSION, evaluate_task_goal, task_from_world


def unavailable(task, status, reason):
    return {"schema": "task-goal-evaluation-v1", "policy_version": POLICY_VERSION,
        "task": task, "task_goal_status": status, "goal_success": None,
        "basis": "INSUFFICIENT_TERMINAL_EVIDENCE", "reason": reason,
        "checks": [], "failed_checks": [], "unknown_checks": [reason]}


class ArtifactReader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.sources = {}

    def read(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("artifact outside supplied run directory")
        raw = path.read_bytes()
        self.sources[str(path.relative_to(self.root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def optional(self, path):
        return self.read(path) if path.is_file() else {}


def _basename(path):
    return str(path).replace("\\", "/").rstrip("/").split("/")[-1]


def _manifest_path(m5, summary):
    live = m5 / "live"
    named = summary.get("live_run_dir")
    if not named and summary.get("manifest"):
        parent = str(summary["manifest"]).replace("\\", "/").rsplit("/", 1)[0]
        if _basename(parent).startswith("run-"):
            named = parent
    if named:
        # Relocate Windows/other-checkout paths; never follow original paths.
        selected = live / _basename(named) / "live-execution-manifest.json"
        if not selected.is_file():
            raise ValueError("summary-selected live run is missing")
        return selected
    nested = sorted(live.glob("run-*/live-execution-manifest.json"))
    if len(nested) > 1:
        raise ValueError("multiple live runs without an unambiguous summary pointer")
    if nested:
        return nested[0]
    stable = live / "live-execution-manifest.json"
    return stable if stable.is_file() else None


def _history(reader, live, steps):
    events, cover_support = [], None
    for step in steps:
        event = {k: step.get(k) for k in ("index", "action_type", "status", "request_id")}
        name = _basename(step.get("request", ""))
        files = list(live.glob(f"**/requests/{name}")) if name else []
        request = None
        if len(files) == 1:
            request = reader.read(files[0])
            if request.get("request_id") != step.get("request_id"):
                request = None
        event["target_ids"] = request.get("task", {}).get("target_ids") if request else None
        if event["target_ids"] is None and step.get("object_id"):
            event["target_ids"] = [step["object_id"]]
        events.append(event)
        # Invalidate earlier cover evidence if the lid or contents move again.
        if request and set(event["target_ids"] or []) & {"lid", "packing_box", "rolling_pin", "baguette", "whisk", "cereal", "milk"}:
            cover_support = None
        if not request or "lid" not in (event["target_ids"] or []):
            continue
        reports = list(files[0].parent.parent.glob("simulation/reports/*execution-report.json"))
        if len(reports) != 1:
            continue
        report = reader.read(reports[0])
        tracking = report.get("metadata", {}).get("segment_tracking", [])
        settle = (tracking[-1].get("custom_settle") or {}).get("container_settle", {}) if tracking else {}
        if settle.get("request_id") == step.get("request_id") and settle.get("surface_target_id") == "lid" and settle.get("region_id") == "packing_box":
            cover_support = settle
    return events, cover_support


def evaluate_saved_run(run_dir, *, task=None, summary_override=None):
    """Return derived evidence only. Never alter raw experiment artifacts.

    Live CLI supplies its in-memory summary before writing m5_summary.json.
    Batch timeouts / IN_PROGRESS snapshots cannot be treated as final states.
    """
    run_dir = Path(run_dir).resolve()
    reader = ArtifactReader(run_dir)
    m5 = run_dir / "m5"
    result = None
    try:
        summary = summary_override if summary_override is not None else reader.optional(m5 / "m5_summary.json")
        rr = reader.optional(run_dir / "run_result.json") if summary_override is None else {}
        batch = reader.optional(run_dir / "batch_record.json") if summary_override is None else {}
        manifest_path = _manifest_path(m5, summary)
        if manifest_path is None:
            stage = reader.optional(run_dir / "ablation_manifest.json")
            early_failure = rr.get("plan_success") is False or any(str(v).lower() in ("failed", "failure", "infeasible") for k, v in stage.get("stages", {}).items() if k != "m5")
            result = unavailable(task, "NOT_EXECUTED" if early_failure or not m5.exists() else "UNKNOWN", "no measured M5 terminal snapshot")
        else:
            manifest = reader.read(manifest_path)
            final = manifest.get("final_world")
            task = task or (task_from_world(final) if final else None)
            if batch.get("exit_code") == -9 or manifest.get("status") not in ("SUCCESS", "FAILED"):
                result = unavailable(task, "INCOMPLETE", "interrupted/in-progress run; last snapshot is not terminal evidence")
                result["last_observed_step_count"] = len(manifest.get("steps", []))
            elif not final:
                result = unavailable(task, "UNKNOWN", "missing measured final world")
            else:
                initial = reader.optional(m5 / "initial_world.json")
                history, support = _history(reader, manifest_path.parent, manifest.get("steps", []))
                result = evaluate_task_goal(task, initial, final, history=history, cover_support=support)
                result["execution_status"] = manifest["status"]
                result["manifest"] = str(manifest_path.relative_to(run_dir))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = unavailable(task, "UNKNOWN", f"artifact evidence unavailable: {type(exc).__name__}: {exc}")
    result["source_directory"] = str(run_dir)
    result["source_sha256"] = reader.sources
    return result


def attach_live_goal_evaluation(output, summary):
    """Persist goal verdict even when M5 raised, without hiding execution status."""
    result = (evaluate_saved_run(Path(output).parent, summary_override=summary)
              if summary.get("manifest") else unavailable(None, "UNKNOWN", "current execution produced no terminal manifest"))
    path = Path(output) / "task_goal_evaluation.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary.update(task_goal_status=result["task_goal_status"], goal_success=result["goal_success"],
                   task_goal_evaluation=str(path.resolve()), task_goal_policy=result["policy_version"])
    return result
