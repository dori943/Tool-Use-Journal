"""Compute SReg = actual EE switches - GT-optimal EE switches.

Uses the same M4 Dijkstra planner as the runtime, with feasibility inputs
replaced by MuJoCo GT mass/AABB (scene report) evaluated via ee_rules.

Example (c3_2)::

    python scripts/view_c3_2.py --headless --report output/c3_2/gt_scene_report.json
    python scripts/compute_sreg.py c3_2 --input-dir output/c3_2 \\
        --gt-scene output/c3_2/gt_scene_report.json --scripted
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main() -> int:
    from task_registry import TASK_ENVS
    from tuj.gt.labels import build_gt_payload_for_task_dir

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=sorted(TASK_ENVS))
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Run folder with m1.json / m2.json / gk_bundle.json / m4.json",
    )
    parser.add_argument(
        "--gt-scene",
        type=Path,
        required=True,
        help="GT scene report JSON from env.get_scene_report() / view_c*_*.py --report",
    )
    parser.add_argument(
        "--robot-spec",
        type=Path,
        default=ROOT / "configs" / "robot_spec.json",
    )
    parser.add_argument(
        "--initial-ee",
        choices=["2F", "3F", "vac"],
        default=None,
        help="Override initial EE (default: robot_spec.current_ee, usually null)",
    )
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Apply scripted-grasp EE constraints (ours / scripted live contract)",
    )
    parser.add_argument(
        "--live-run-dir",
        type=Path,
        default=None,
        help="Optional M5 live run dir; when present, prefer live exchange count",
    )
    args = parser.parse_args()

    # TASK_ENVS maps task id -> robosuite env class name (scripted registry key).
    scripted_environment = TASK_ENVS[args.task] if args.scripted else None

    payload = build_gt_payload_for_task_dir(
        args.input_dir,
        task=args.task,
        gt_scene_path=args.gt_scene,
        robot_spec_path=args.robot_spec,
        initial_ee=args.initial_ee,
        scripted_environment=scripted_environment,
        live_run_dir=args.live_run_dir,
    )
    out = Path(args.input_dir) / "gt.json"
    print(
        json.dumps(
            {
                "wrote": str(out),
                "actual_ee_switches": payload.get("actual_ee_switches"),
                "actual_source": payload.get("actual_ee_switches_source"),
                "optimal_ee_switches": payload.get("optimal_ee_switches"),
                "sreg": payload.get("sreg"),
                "planning_status": payload.get("planning_result_status"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if payload.get("optimal_ee_switches") is None:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
