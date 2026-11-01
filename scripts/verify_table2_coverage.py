# -*- coding: utf-8 -*-
"""Independent fixed-EE grasp trials, separate from whole-task ablation.

--all-ees enumerates every object/EE pair; it does not imply execution support.
--registered-only selects production recipes for a baseline. Validator-only
recipes remain experimental. All cases run in fresh processes, with a timeout.
Coverage over all task objects and success over inferred-feasible objects are
reported separately. Neither metric counts unexecuted cases as physical failures.
Use a NEW --out-root and --doc for every run. No LLM/API calls are made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from task_registry import TASKS
from audit_ee_coverage import TABLE2, TASK_GRASP_OBJECTS, load_ycb_masses, make_env, gt_scene_from_env


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def reasoner_feasible(task: str, m1_path: Path) -> dict[str, list[str]] | None:
    if not m1_path.is_file():
        return None
    m1 = json.loads(m1_path.read_text(encoding="utf-8"))
    out = {}
    for n in m1.get("nodes", []):
        nid, cls = n.get("id", ""), n.get("class", "")
        short = nid[len(f"obj_{cls}_"):] if nid.startswith(f"obj_{cls}_") else nid
        ee = n.get("ee") or {}
        if ee:
            out[short] = sorted(k for k, v in ee.items() if v.get("feasible"))
    names = TASK_GRASP_OBJECTS[task]
    if not any(o in out for o in names):
        return None
    # Missing records are unknown, not a negative verdict.
    return {o: out[o] for o in names if o in out}


def gt_rule_feasible(task: str, seed: int) -> dict[str, list[str]]:
    from tuj.gt.labels import intrinsic_from_gt, normalize_ee_pool
    from tuj.m1_scene.ee_rules import evaluate_ee
    robot = json.loads((ROOT / "configs" / "robot_spec.json").read_text(encoding="utf-8"))
    pool = normalize_ee_pool(robot["ee_pool"])
    env = make_env(task, seed)
    try:
        gt = gt_scene_from_env(env, TASK_GRASP_OBJECTS[task], load_ycb_masses(ROOT / "configs" / "ycb_mass_gt.json"))
    finally:
        env.close()
    return {name: [e["ee_id"] for e in pool if evaluate_ee(e, intrinsic_from_gt(rec))["feasible"]]
            for name, rec in gt.items() if "error" not in rec}


def summarize_trials(task, trials, ee_ids):
    total = len(TASK_GRASP_OBJECTS[task])
    per_ee = {}
    for ee in ee_ids:
        rows = [t for t in trials if t["ee"] == ee]
        inferred = [t for t in rows if t["inferred_feasible"] is True]
        passed = [t for t in rows if t["status"] == "PASS"]
        expected = TABLE2.get(task, {}).get(ee)
        complete = len({t["object"] for t in rows}) == total
        per_ee[ee] = {
            "task_objects": total, "evaluated_cases": len(rows),
            "grasp_attempted": sum(t.get("grasp_attempted", False) for t in rows),
            "grasped": len(passed), "coverage": f"{len(passed)}/{total}",
            "matrix_complete": complete,
            "inferred": len(inferred),
            "inferred_grasped": sum(t["status"] == "PASS" for t in inferred),
            "inference_known_cases": sum(t["inferred_feasible"] is not None for t in rows),
            "not_registered": [t["object"] for t in rows if t["status"] in ("NOT_REGISTERED", "EE_MISMATCH")],
            "failed": [t["object"] for t in rows if t["status"] == "FAIL"],
            "errors": [t["object"] for t in rows if t["status"] in ("ERROR", "TIMEOUT")],
            "table2": f"{expected[0]}/{expected[1]}" if expected else None,
            "table2_same_object_count": expected[1] == total if expected else None,
            # A numerical match is deliberately not treated as validation.
        }
    return per_ee


def run_case(task, name, ee, *, seed, video, camera, output, timeout_s):
    output.mkdir(parents=True, exist_ok=False)
    args = [sys.executable, str(Path(__file__).resolve()), "--worker-case", task, name, ee,
            "--out-root", str(output), "--seed", str(seed), "--camera", camera]
    if video:
        args.append("--video")
    started = time.monotonic()
    infrastructure = None
    with (output / "console.log").open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
            if proc.returncode:
                infrastructure = ("ERROR", f"WORKER_EXIT:{proc.returncode}")
        except subprocess.TimeoutExpired:
            infrastructure = ("TIMEOUT", f"WALL_TIME_LIMIT:{timeout_s}s")
    report_path = output / "validation_report.json"
    try:
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if not isinstance(report, dict) or "controller_status" not in report:
            infrastructure = infrastructure or ("ERROR", "INVALID_VALIDATION_REPORT")
            report = {}
    except (ValueError, OSError) as error:
        infrastructure = infrastructure or ("ERROR", f"REPORT_READ_ERROR:{type(error).__name__}")
        report = {}
    if infrastructure or not report:
        status, detail = infrastructure or ("ERROR", "MISSING_VALIDATION_REPORT")
        report.update(object_id=name, ee=ee, environment=TASKS[task].cls,
                      controller_status=status, failure_category="IMPLEMENTATION_OR_RUNTIME",
                      failures=[detail])
        # Keep real acquisition artifacts after a process interruption.
        report.setdefault("simulation_started", (output / "initial_state.npz").exists())
        report.setdefault("grasp_attempted", (output / "grasp" / "input.json").exists())
        save_json(report_path, report)
    result = report.get("result") or {}
    return {"object": name, "ee": ee, "status": report["controller_status"],
            "simulation_started": report.get("simulation_started", False),
            "grasp_attempted": report.get("grasp_attempted", False),
            "integration_status": report.get("integration_status"),
            "failure_category": report.get("failure_category"),
            "failures": report.get("failures", []), "detail": report.get("detail"),
            "result": result, "artifact_errors": report.get("artifact_errors", []),
            "seconds": round(time.monotonic() - started, 2), "report": str(report_path)}


def has_production_recipe(object_id, ee, environment):
    """Follow production resolution, including instance/type precedence."""
    from tuj.m5_motion.scripted_grasps import registry

    target = registry.ALIASES.get(object_id, object_id)
    entries = registry.ENTRIES + registry.ALTERNATIVE_ENTRIES
    for key in registry._recipe_keys_for_target(target, environment):
        matches = [e for e in entries
                   if (e.object_id, e.environment) == (key, environment)]
        if matches:
            # An exact instance registration shadows its type, even on EE mismatch.
            return any(e.ee == ee and e.object_id not in registry.PENDING_INTEGRATION
                       for e in matches)
    return False


def run_task(task, *, seed, denominator, all_ees, video, out_root, ee_ids,
             objects=None, registered_only=False, experimental_only=False, camera="agentview", timeout_s=600):
    names = objects or TASK_GRASP_OBJECTS[task]
    if any(n not in TASK_GRASP_OBJECTS[task] for n in names):
        raise ValueError(f"Unknown object for {task}: {names}")
    feasible = None
    source = denominator
    m1_path = ROOT / "output" / task / "m1.json"
    if denominator == "reasoner":
        feasible = reasoner_feasible(task, m1_path)
        if feasible is None:
            source = "gt-rule(fallback)"
    if feasible is None and denominator != "none":
        feasible = gt_rule_feasible(task, seed)
    feasible = feasible or {}
    res = {"task": task, "environment": TASKS[task].cls, "seed": seed,
           "denominator_source": source, "inferred_feasible": feasible, "trials": [],
           "registered_only": registered_only, "experimental_only": experimental_only}
    if source == "reasoner":
        res["inference_input"] = {"path": str(m1_path), "sha256": hashlib.sha256(m1_path.read_bytes()).hexdigest(),
                                  "note": "Stored M1; not independently verified as a Full run."}
    for name in names:
        ees = ee_ids if all_ees else [e for e in ee_ids if e in feasible.get(name, [])]
        for ee in ees:
            if registered_only or experimental_only:
                registered = has_production_recipe(name, ee, TASKS[task].cls)
                if (registered_only and not registered) or (experimental_only and registered):
                    continue
            output = out_root / task / f"{name}_{ee}"
            print(f"[table2] {task} {name} x {ee}", flush=True)
            trial = run_case(task, name, ee, seed=seed, video=video, camera=camera,
                             output=output, timeout_s=timeout_s)
            trial["inferred_feasible"] = ee in feasible[name] if name in feasible else None
            res["trials"].append(trial)
            res["per_ee"] = summarize_trials(task, res["trials"], ee_ids)
            save_json(out_root / task / "summary.json", res)
            print(f"[table2] -> {trial['status']} {trial['failures'][:1]}", flush=True)
    res["per_ee"] = summarize_trials(task, res["trials"], ee_ids)
    save_json(out_root / task / "summary.json", res)
    return res


def render_md(results):
    lines = ["# Fixed-EE grasp verification", "",
             "Coverage uses every task object as its denominator. Partial matrices are not final coverage results.",
             "Inference-conditioned counts are separate. Unknown inference is not a negative verdict.",
             "Historical Table II counts are not comparable when the current scene object set differs.", ""]
    for res in results:
        lines += [f"## {res['task']} (seed={res['seed']}, inference={res['denominator_source']})", "",
                  "| EE | PASS / all objects | tested cases | grasp attempted | inferred PASS / inferred | matrix complete |",
                  "|---|---|---|---|---|---|"]
        for ee, v in res["per_ee"].items():
            inferred = f"{v['inferred_grasped']}/{v['inferred']}" if v['inference_known_cases'] else "unknown"
            lines.append(f"| {ee} | {v['coverage']} | {v['evaluated_cases']} | {v['grasp_attempted']} | {inferred} | {v['matrix_complete']} |")
        lines += ["", "| object | EE | status | integration | attachment phase | detail |", "|---|---|---|---|---|---|"]
        for t in res["trials"]:
            result = t.get("result") or {}
            phase = result.get("attachment_phase") or ("ACQUISITION" if result.get("attachment_used") else "none/unknown")
            detail = str(t.get("detail") or "; ".join(t["failures"])).replace("|", "/").replace("\n", " ")
            lines.append(f"| {t['object']} | {t['ee']} | {t['status']} | {t.get('integration_status')} | {phase} | {detail[:240]} |")
        lines.append("")
    lines += ["NOT_REGISTERED/EE_MISMATCH means unexecuted, not physically impossible. FAIL describes this motion/grasp attempt only.",
              "ERROR/TIMEOUT is separate from grasp failure. Attachment-assisted PASS is not proof of unassisted retention.",
              "This checks grasp, lift and hold; it does not evaluate completion of the full task."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", default="c1_1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--denominator", choices=("reasoner", "gt-rule", "none"), default="reasoner")
    p.add_argument("--all-ees", action="store_true")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--registered-only", action="store_true")
    selection.add_argument("--experimental-only", action="store_true")
    p.add_argument("--objects", nargs="+")
    p.add_argument("--ees", nargs="+", choices=("2F", "3F", "vac"), default=["2F", "3F", "vac"])
    p.add_argument("--video", action="store_true")
    p.add_argument("--camera", default="agentview")
    p.add_argument("--timeout-seconds", type=float, default=600)
    p.add_argument("--out-root", type=Path, default=ROOT / "output" / "table2_verify")
    p.add_argument("--doc", type=Path, default=ROOT / "docs" / "table2_verification.md")
    p.add_argument("--worker-case", nargs=3, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.worker_case:
        from tuj.m5_motion.scripted_grasps.validate import controller_validate_case
        task, name, ee = args.worker_case
        controller_validate_case(name, ee, args.out_root, environment=TASKS[task].cls,
                                 seed=args.seed, camera=args.camera, repository=ROOT,
                                 video=args.out_root / f"{name}_{ee}.mp4" if args.video else None)
        return 0
    tasks = list(TASK_GRASP_OBJECTS) if args.tasks == "all" else args.tasks.split(",")
    if len(set(tasks)) != len(tasks) or any(t not in TASK_GRASP_OBJECTS for t in tasks):
        p.error("tasks must be distinct known task ids")
    if args.denominator == "none" and not args.all_ees:
        p.error("--denominator none requires --all-ees")
    if args.timeout_seconds <= 0:
        p.error("timeout must be positive")
    for path in (args.out_root, args.doc, args.doc.with_suffix(".json")):
        if path.exists():
            p.error(f"Refusing to overwrite existing output: {path}")
    args.out_root = args.out_root.resolve()
    args.out_root.mkdir(parents=True)
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    (args.out_root / "source.patch").write_bytes(diff)
    # Hash untracked experimental modules as well as tracked source files.
    source_files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in source_files if name.endswith(".py") and (ROOT / name).is_file()
              and (len(Path(name).parts) == 1 or Path(name).parts[0] in {"src", "scripts", "tests", "environments"})}
    # Include new source files that are absent from git diff.
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    for name in untracked:
        if name in hashes:
            destination = args.out_root / "untracked_source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
    save_json(args.out_root / "run_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "commit": git,
        "source_hashes": hashes, "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "scope": "GRASP_LIFT_HOLD_ONLY", "api_calls": 0})
    results = []
    for task in tasks:
        results.append(run_task(task, seed=args.seed, denominator=args.denominator, all_ees=args.all_ees,
                                video=args.video, out_root=args.out_root, ee_ids=args.ees,
                                objects=args.objects, registered_only=args.registered_only, experimental_only=args.experimental_only,
                                camera=args.camera, timeout_s=args.timeout_seconds))
        args.doc.parent.mkdir(parents=True, exist_ok=True)
        args.doc.write_text(render_md(results), encoding="utf-8")
        save_json(args.doc.with_suffix(".json"), results)
    print(render_md(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
