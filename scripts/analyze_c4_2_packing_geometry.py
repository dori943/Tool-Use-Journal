"""Report C4-2 collision-mesh extents for candidate box-local orientations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from tuj.m5_motion.geometry import quaternion_matrix_xyzw  # noqa: E402
from tuj.m5_motion.object_function_grasp import make_function_runtime  # noqa: E402
from tuj.m5_motion.tool_use_journal import (  # noqa: E402
    ToolUseJournalEnvironmentAdapter,
    _geom_local_points,
    _raw_model_data,
    _subtree_geom_ids,
)


ORIENTATIONS_XYZW = {
    "whisk": (0.01277933, 0.71879474, -0.38785331, 0.57683674),
    "rolling_pin": (-0.38898768, -0.36902991, -0.12626111, 0.83460388),
    "baguette": (-0.19812896, -0.28660541, 0.33754624, 0.87445114),
    # Lay rigid cartons on their long side so their height stays below the
    # container opening and the lid remains physically placeable.
    "cereal": (0.0, 0.7071067811865476, 0.0, 0.7071067811865476),
    "milk": (0.0, 0.7071067811865476, 0.0, 0.7071067811865476),
}
PACKING_OBJECTS = ("rolling_pin", "whisk", "baguette", "cereal", "milk")


def _collision_points_in_body(adapter, object_id: str) -> np.ndarray:
    model, data = adapter.model, adapter.data
    body_id = adapter.object_body_ids[object_id]
    root_position = np.asarray(data.xpos[body_id], dtype=float)
    root_rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    points = []
    for geom_id in _subtree_geom_ids(model, body_id):
        if not (
            int(model.geom_contype[geom_id])
            or int(model.geom_conaffinity[geom_id])
        ):
            continue
        local = _geom_local_points(model, geom_id)
        if local is None:
            continue
        geom_rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        world = local @ geom_rotation.T + np.asarray(data.geom_xpos[geom_id], dtype=float)
        points.append((world - root_position) @ root_rotation)
    if not points:
        raise RuntimeError(f"{object_id}: no collision points")
    return np.concatenate(points, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--settle-seconds", type=float, default=0.0)
    parser.add_argument("--active-ee", default="2F")
    args = parser.parse_args()
    runtime = make_function_runtime(
        REPOSITORY,
        "C4_2_DiagonalFitPacking",
        active_ee=args.active_ee,
        seed=0,
        ignore_done=True,
        use_camera_obs=False,
        has_offscreen_renderer=False,
    )
    try:
        if args.state is not None:
            saved = np.load(args.state)
            model, data = _raw_model_data(runtime.env)
            data.qpos[:] = saved["qpos"]
            data.qvel[:] = saved["qvel"]
            data.ctrl[:] = saved["ctrl"]
            data.time = float(saved["time"])
            mujoco.mj_forward(model, data)
            if args.settle_seconds > 0.0:
                from tuj.m5_motion.tool_use_journal import (
                    settle_tool_use_journal_free_objects,
                )

                settle_tool_use_journal_free_objects(
                    runtime.env, duration_s=args.settle_seconds
                )
        adapter = ToolUseJournalEnvironmentAdapter(runtime.env)
        mujoco.mj_forward(adapter.model, adapter.data)
        report = {}
        for object_id, quaternion in ORIENTATIONS_XYZW.items():
            points = _collision_points_in_body(adapter, object_id)
            rotation = quaternion_matrix_xyzw(quaternion)
            rotated = points @ rotation.T
            lower = np.min(rotated, axis=0)
            upper = np.max(rotated, axis=0)
            report[object_id] = {
                "orientation_xyzw": list(quaternion),
                "lower_m": lower.tolist(),
                "upper_m": upper.tolist(),
                "dimensions_m": (upper - lower).tolist(),
                "center_offset_m": ((lower + upper) * 0.5).tolist(),
            }
        if args.state is not None:
            from tuj.m5_motion.push_to_region import target_fully_inside_region

            world = adapter.world_snapshot()
            box = world.objects["packing_box"]
            box_position = np.asarray(box["pose"]["position_m"], dtype=float)
            box_rotation = quaternion_matrix_xyzw(
                box["pose"]["orientation_xyzw"]
            )
            def observed_record(object_id):
                record = world.objects[object_id]
                pose = record["pose"]
                points = np.asarray(record["collision_points_m"], dtype=float)
                rotation = quaternion_matrix_xyzw(pose["orientation_xyzw"])
                position = np.asarray(pose["position_m"], dtype=float)
                points_world = position + (rotation @ points.T).T
                points_box = (box_rotation.T @ (points_world - box_position).T).T
                return {
                    "inside_packing_box": target_fully_inside_region(
                        world,
                        target_id=object_id,
                        region_id="packing_box",
                        include_vertical=True,
                    ),
                    "collision_point_count": len(points),
                    "bounds_in_box_m": {
                        "lower": np.min(points_box, axis=0).tolist(),
                        "upper": np.max(points_box, axis=0).tolist(),
                    },
                    "pose": pose,
                }
            report["observed_state"] = {
                object_id: observed_record(object_id)
                for object_id in PACKING_OBJECTS
            }
        print(json.dumps(report, indent=2))
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
