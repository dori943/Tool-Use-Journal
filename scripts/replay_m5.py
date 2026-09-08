"""Replay an existing generic M5 MotionPlan manifest in MuJoCo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    REPOSITORY / "src",
    REPOSITORY.parent / "dain-m3" / "src",
    REPOSITORY.parent / "tuj-m3" / "src",
)
for source_root in reversed(SOURCE_ROOTS):
    if source_root.is_dir() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from tuj.m5_motion.generic_runner import execute_planning_result
from tuj.m5_motion.orchestration import SelectedPlanPlanningResult
from tuj.m5_motion.schema import MotionPlan, MotionPlanRequest, WorldSnapshot


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--realtime-factor", type=float)
    parser.add_argument("--hold-seconds", type=float, default=10.0)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-hold-seconds", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan-count", type=int)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    manifest_path = result_dir / "motion-plan-manifest.json"
    manifest = _load(manifest_path)
    requests = tuple(
        MotionPlanRequest.model_validate(_load(Path(path)))
        for path in manifest["request_files"]
    )
    plans = tuple(
        MotionPlan.model_validate(_load(Path(path)))
        for path in manifest["plan_files"]
    )
    final_world = WorldSnapshot.model_validate(manifest["final_world"])
    if args.plan_count is not None:
        if args.plan_count <= 0:
            parser.error("--plan-count must be positive")
        requests = requests[: args.plan_count]
        plans = plans[: args.plan_count]
    initial_world = WorldSnapshot.model_validate(_load(result_dir / "initial_world.json"))
    planning = SelectedPlanPlanningResult(
        requests=requests,
        plans=plans,
        final_world=final_world,
        request_paths=tuple(Path(path) for path in manifest["request_files"]),
        plan_paths=tuple(Path(path) for path in manifest["plan_files"]),
        manifest_path=manifest_path,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else result_dir / "replay"
    )
    video_path = args.video.resolve() if args.video is not None else None
    realtime_factor = (
        args.realtime_factor
        if args.realtime_factor is not None
        else (0.0 if args.headless else 1.0)
    )
    execution = execute_planning_result(
        planning,
        repository=args.repository.resolve(),
        initial_world=initial_world,
        output_dir=output_dir,
        mode="controller",
        seed=0,
        show_viewer=not args.headless,
        realtime_factor=realtime_factor,
        hold_seconds=args.hold_seconds,
        video=video_path,
        camera=args.camera,
        width=args.width,
        height=args.height,
        video_fps=args.video_fps,
        video_hold_seconds=args.video_hold_seconds,
    )
    print(
        json.dumps(
            {
                "status": execution.status.value,
                "successful": execution.successful,
                "failed_index": execution.failed_index,
                "detail": execution.detail,
                "output_dir": str(output_dir),
                "video": str(video_path) if video_path is not None else None,
                "final_objects": {
                    object_id: record.get("pose")
                    for object_id, record in execution.final_world.objects.items()
                    if object_id in {"ladle", "plate", "fork", "scissors"}
                },
            },
            indent=2,
        )
    )
    return 0 if execution.successful else 2


if __name__ == "__main__":
    raise SystemExit(main())
