"""Render a contact sheet for every registered task's standard agentview."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPOSITORY / "src"), str(REPOSITORY)]

from task_registry import TASK_ENVS  # noqa: E402
from tuj.m5_motion.object_function_grasp import make_function_runtime  # noqa: E402


def main() -> int:
    output_dir = REPOSITORY / "output" / "camera_previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = 480, 270
    frames: list[np.ndarray] = []
    results: list[dict[str, object]] = []

    for task_id, environment_name in TASK_ENVS.items():
        runtime = make_function_runtime(
            REPOSITORY,
            environment_name,
            active_ee=None,
            seed=0,
            ignore_done=True,
            use_camera_obs=False,
            has_offscreen_renderer=True,
        )
        try:
            env = runtime.env
            rgb = env.sim.render(
                camera_name="agentview", width=width, height=height
            )[::-1]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(bgr, (0, 0), (150, 30), (0, 0, 0), -1)
            cv2.putText(
                bgr,
                task_id,
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            image_path = output_dir / f"{task_id}.png"
            cv2.imwrite(str(image_path), bgr)
            frames.append(bgr)
            camera_id = env.sim.model.camera_name2id("agentview")
            results.append(
                {
                    "task_id": task_id,
                    "environment": environment_name,
                    "camera": "agentview",
                    "fovy_deg": float(env.sim.model.cam_fovy[camera_id]),
                    "image": str(image_path),
                }
            )
        finally:
            runtime.close()

    while len(frames) % 3:
        frames.append(np.zeros_like(frames[0]))
    rows = [np.hstack(frames[index : index + 3]) for index in range(0, len(frames), 3)]
    contact_sheet = np.vstack(rows)
    sheet_path = output_dir / "all_tasks.png"
    cv2.imwrite(str(sheet_path), contact_sheet)
    print(json.dumps({"contact_sheet": str(sheet_path), "tasks": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
