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
    # 0912: 한 파지만 눈으로 확인할 방법이 없어 전체 태스크를 돌려야 했다.
    # 녹화 인자는 scripted_grasps/cli.py 와 같은 규약을 쓴다.
    parser.add_argument("--video", type=Path, help="record this grasp to an mp4")
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--video-fps", type=float, default=20.)
    parser.add_argument("--video-hold-seconds", type=float, default=1.5,
        help="freeze the last frame this long so the held pose is visible")
    args = parser.parse_args()
    matches = [e for e in ENTRIES if e.object_id == args.object
        and (args.ee is None or e.ee == args.ee)
        and (args.environment is None or e.environment == args.environment)]
    if not matches:
        parser.error(f"no scripted grasp for object={args.object!r}, "
            f"ee={args.ee!r}, environment={args.environment!r}")
    entry = matches[0]
    # The offscreen context is sized from the environment camera options
    # (256x256 by default); recording asks sim.render() for the video size, so a
    # 960x540 read out of a 256x256 buffer is noise. Declare the size up front,
    # exactly as scripted_grasps/cli.py does.
    recording_options = (
        {"camera_names": args.camera, "camera_heights": args.height,
         "camera_widths": args.width}
        if args.video is not None
        else {}
    )
    runtime = ToolUseJournalEERuntime.from_repository_for_controller(REPOSITORY, entry.environment,
        active_ee=entry.ee, seed=args.seed, scripted_grasps=True, ignore_done=True,
        has_renderer=False, has_offscreen_renderer=args.video is not None,
        use_camera_obs=False, render_camera=args.camera, **recording_options)
    recorder = None
    try:
        if args.video is not None:
            from tuj.m5_motion.generic_runner import GenericSimulationVideoRecorder
            camera = args.camera
            if camera not in runtime.env.sim.model.camera_names:
                camera = runtime.env.render_camera
            # bind_context() 는 --output 이 존재하지 않아야 한다 (mkdir exist_ok=False).
            # 영상 경로를 그 안에 두면 여기서 미리 만들어 버려 파지가 시작도 못 한다.
            if args.video.resolve().parent == args.output.resolve():
                parser.error("--video must not sit inside --output; "
                             "the grasp output directory has to be created fresh")
            args.video.parent.mkdir(parents=True, exist_ok=True)
            recorder = GenericSimulationVideoRecorder(runtime, args.video.resolve(),
                camera=camera, width=args.width, height=args.height, fps=args.video_fps)
            # The catalog runtime only renders when asked to.
            runtime.scripted_render = True
            runtime.scripted_realtime_factor = 0.
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
        if recorder is not None:
            recorder.hold_final_frame(args.video_hold_seconds)
            recorder.close()
            print(f"[video] {args.video.resolve()}")
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
