"""Validate scripted grasps without M1→M5 or LLM calls.

Examples:
  python -m tuj.m5_motion.examples.scripted_grasp_validate c3_2 --object plate_b --ee vac --static
  python -m tuj.m5_motion.examples.scripted_grasp_validate c3_2 --object plate_b --ee vac --controller --output out/plate_b
  python -m tuj.m5_motion.examples.scripted_grasp_validate c3_2 --object plate_b --ee vac --controller --output out/plate_b --video out/plate_b.mp4 --camera robot0_robotview
  python -m tuj.m5_motion.examples.scripted_grasp_validate c3_2 --all --static
  python -m tuj.m5_motion.examples.scripted_grasp_validate c3_2 --all --static --world output/c3_2/m5/initial_world.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tuj.m5_motion.scripted_grasps.validate import (
    C3_2_ENVIRONMENT,
    C3_2_EXPECTED_ACQUIRES,
    batch_static_c3_2,
    c3_2_recipe_readiness,
    controller_validate_case,
    load_world_objects,
    static_validate_case,
)

DEFAULT_CAMERA = "agentview"

TASK_TO_ENVIRONMENT = {
    "c3_2": C3_2_ENVIRONMENT,
}


def _default_ee(task: str, object_id: str) -> str | None:
    if task != "c3_2":
        return None
    for oid, ee in C3_2_EXPECTED_ACQUIRES:
        if oid == object_id:
            return ee
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        choices=sorted(TASK_TO_ENVIRONMENT),
        help="task id (currently c3_2 is the supported validation matrix)",
    )
    parser.add_argument("--object", help="scene instance id, e.g. plate_b")
    parser.add_argument("--ee", help="end-effector id; defaults from the c3_2 M4 map")
    parser.add_argument(
        "--all",
        action="store_true",
        help="batch over the c3_2 M4 acquire matrix",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="geometry/resolve validation (cheap; default if neither mode set)",
    )
    parser.add_argument(
        "--controller",
        action="store_true",
        help="explicit physical PRE→GRASP→attach→LIFT via execute_grasp",
    )
    parser.add_argument(
        "--world",
        type=Path,
        help="optional WorldSnapshot JSON for object poses/dimensions",
    )
    parser.add_argument("--support-top-z", type=float, help="optional support surface z")
    parser.add_argument("--output", type=Path, help="artifact directory for controller mode")
    parser.add_argument("--video", type=Path, help="optional MP4 for controller mode")
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help=(
            "MuJoCo/robosuite camera for controller render/video "
            f"(default: {DEFAULT_CAMERA}; c3_2 typically needs robot0_robotview)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--readiness",
        action="store_true",
        help="print recipe readiness table and exit",
    )
    args = parser.parse_args(argv)

    if args.readiness:
        print(json.dumps(c3_2_recipe_readiness(), indent=2))
        return 0
    if not args.static and not args.controller:
        args.static = True
    if args.all and args.controller:
        parser.error(
            "--all --controller is disabled; run controller one object at a time "
            "with --object ... --controller"
        )
    if args.all and args.object:
        parser.error("use either --all or --object")
    if not args.all and not args.object:
        parser.error("provide --object or --all")

    environment = TASK_TO_ENVIRONMENT[args.task]

    if args.all:
        result = batch_static_c3_2(
            world_path=args.world, support_top_z=args.support_top_z,
        )
        print(result["summary_table"])
        print(json.dumps({
            "pass_count": result["pass_count"],
            "fail_count": result["fail_count"],
            "not_registered_count": result["not_registered_count"],
            "ee_mismatch_count": result["ee_mismatch_count"],
            "cases": [
                {
                    "object_id": c["object_id"],
                    "ee": c["ee"],
                    "recipe_id": c.get("recipe_id"),
                    "static_status": c["static_status"],
                    "failures": c.get("failures", []),
                }
                for c in result["cases"]
            ],
        }, indent=2))
        return 0 if result["fail_count"] == 0 and result["ee_mismatch_count"] == 0 else 2

    ee = args.ee or _default_ee(args.task, args.object)
    if ee is None:
        parser.error(f"--ee is required for object {args.object!r}")

    world_object = None
    if args.world is not None:
        worlds = load_world_objects(args.world)
        world_object = worlds.get(args.object)
        if world_object is None:
            parser.error(f"object {args.object!r} absent from {args.world}")

    if args.controller:
        if args.output is None:
            parser.error("--controller requires --output")
        report = controller_validate_case(
            args.object,
            ee,
            args.output,
            environment=environment,
            seed=args.seed,
            video=args.video,
            camera=args.camera,
        )
    else:
        report = static_validate_case(
            args.object,
            ee,
            environment=environment,
            world_object=world_object,
            support_top_z=args.support_top_z,
        )

    print(json.dumps(report, indent=2, default=str))
    status = (
        report.get("controller_status") if args.controller else report.get("static_status")
    )
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
