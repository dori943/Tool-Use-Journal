"""Exercise one integrated grasp in a deterministic M5 runtime, without an LLM."""
import argparse
import json
from pathlib import Path

from tuj.m5_motion.scripted_grasps.registry import ENTRIES
from tuj.m5_motion.scripted_grasps.context import execute_grasp
from tuj.m5_motion.scripted_grasps.settings import REPOSITORY
from tuj.m5_motion.scripted_grasps.profiles import settle_tool_use_journal_free_objects
from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object", choices=sorted({entry.object_id for entry in ENTRIES}))
    parser.add_argument("--ee", choices=("2F", "3F", "vac"),
        help="select an exact registered end effector when an object has alternatives")
    parser.add_argument("--environment",
        help="select an exact registered environment when an object appears in multiple tasks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dx", type=float, default=0., help="scenario placement offset, before grasp")
    parser.add_argument("--followup", action="store_true", help="also check ordinary M5 movement and physical release")
    args = parser.parse_args()
    matches = [e for e in ENTRIES if e.object_id == args.object
        and (args.ee is None or e.ee == args.ee)
        and (args.environment is None or e.environment == args.environment)]
    if not matches:
        parser.error(f"no scripted grasp for object={args.object!r}, "
            f"ee={args.ee!r}, environment={args.environment!r}")
    entry = matches[0]
    runtime = ToolUseJournalEERuntime.from_repository_for_controller(REPOSITORY, entry.environment,
        active_ee=entry.ee, seed=args.seed, scripted_grasps=True, ignore_done=True)
    try:
        if args.dx:
            body = runtime.env.obj_body_id[entry.object_id]
            model, data = runtime.env.sim.model, runtime.env.sim.data
            joint = int(model.body_jntadr[body])
            data.qpos[int(model.jnt_qposadr[joint])] += args.dx
            runtime.env.sim.forward()
        # Scenario initialization only. The grasp function itself never resets,
        # places an object or replaces the environment.
        if args.dx or entry.driver == "plate":
            settle_tool_use_journal_free_objects(runtime.env, duration_s=2.)
        result = execute_grasp(runtime, entry, args.output, seed=args.seed)
        if args.followup:
            from tuj.m5_motion.examples.scripted_grasp_followup import check_followup
            result["followup"] = check_followup(runtime, entry, args.output / "followup", args.seed)
        print(json.dumps({"status": result["status"], "object": entry.object_id,
            "environment": entry.environment, "ee": entry.ee,
            "recipe_id": result.get("recipe", {}).get("recipe_id"),
            "metrics": result.get("metrics"), "followup": result.get("followup")}, default=str, indent=2))
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
