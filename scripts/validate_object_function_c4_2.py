"""C4-2 function -> attach -> transport -> release regression, without model calls.

Returns each object to its initial support location. This tests the attachment
lifecycle, not the complete M4 packing task or its packing-goal evaluator.
"""
from pathlib import Path
import argparse
import json
import math
import sys
from types import SimpleNamespace

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="+", default=["whisk", "rolling_pin", "baguette", "lid"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--release-offset-xy", type=float, nargs=2, default=(0., 0.),
                        metavar=("X_M", "Y_M"), help="Supported release location relative to the source")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import numpy as np
    from scipy.spatial.transform import Rotation
    from tuj.m5_motion.object_function_grasp import (
        catalog_library, execute_object_function_grasp, make_function_runtime, release_object,
    )
    from tuj.m5_motion.tool_use_journal import settle_tool_use_journal_free_objects
    catalog = catalog_library(REPOSITORY)
    from grasp_lab.catalog_runtime import CatalogContext
    from grasp_lab.frames import inverse
    from grasp_lab.runtime import save_json
    results = []
    for object_id in args.objects:
        recipe = catalog.get_recipe(object_id)
        if recipe.task_id != "c4_2":
            raise ValueError(f"not a C4-2 catalog object: {object_id}")
        directory = args.output_dir / object_id
        directory.mkdir()
        runtime = context = recorder = None
        rows = []
        record = {"object_id": object_id, "status": "FAILED",
                  "release_offset_xy_m": list(args.release_offset_xy)}
        try:
            runtime = make_function_runtime(REPOSITORY, "C4_2_DiagonalFitPacking",
                active_ee=recipe.ee_id, seed=0, has_offscreen_renderer=True,
                ignore_done=True, use_camera_obs=False)
            settle_tool_use_journal_free_objects(runtime.env, duration_s=3.)
            env_identity = id(runtime.env)
            body_id = runtime.env.obj_body_id[object_id]
            initial_position = runtime.env.sim.data.body_xpos[body_id].copy()
            if args.video:
                from tuj.m5_motion.generic_runner import GenericSimulationVideoRecorder
                recorder = GenericSimulationVideoRecorder(runtime, directory / "lifecycle.mp4",
                    camera="agentview", width=960, height=540, fps=20.)
            result = execute_object_function_grasp(runtime, object_id=object_id,
                output_dir=directory / "grasp", repository=REPOSITORY,
                on_control_step=runtime.render if recorder else None)
            assert id(runtime.env) == env_identity
            context = CatalogContext.from_runtime(runtime, recipe=recipe,
                output=directory / "transport", repository=REPOSITORY)
            context.carried_pose = inverse(context.grip_pose()) @ context.body_pose()
            reference = context.carried_pose.copy()
            plan = SimpleNamespace(joint_names=tuple(context.robot.robot_model.joints))

            def tick(q, phase):
                action = context.player._controller_action(plan, q)
                context.player._advance_controller(action)
                if recorder: runtime.render()
                relative = inverse(context.grip_pose()) @ context.body_pose()
                rows.append({"phase": phase, "time_s": float(context.data.time),
                    "attached": runtime.attached_object_id is not None,
                    "object_position": context.body_pose()[:3, 3].tolist(),
                    "position_error_m": float(np.linalg.norm(relative[:3, 3] - reference[:3, 3])),
                    "angle_error_deg": float(np.rad2deg(Rotation.from_matrix(
                        reference[:3, :3].T @ relative[:3, :3]).magnitude()))})
                # Include the carried object in online checks until release.
                bad = context.bad_contacts(context.data,
                    "HOLD" if runtime.attachment or phase == "RELEASE_SETTLE" else "RETREAT")
                if phase == "RELEASE_SETTLE":
                    # The island's countertop is split across multiple geoms.
                    # Contact with any of these is the intended placement, not
                    # an unexpected collision after releasing the free body.
                    supports = {context.model.geom(g).name for g in range(context.model.ngeom)
                        if context.model.geom(g).name.startswith('island_island_group_top_')}
                    supports.add(context.model.geom(context.support_gid).name)
                    object_geoms = {context.model.geom(g).name for g in context.object_geoms}
                    bad = [contact for contact in bad if not (
                        supports.intersection(contact["geoms"]) and object_geoms.intersection(contact["geoms"]))]
                if bad:
                    raise RuntimeError(f"UNEXPECTED_COLLISION: {bad[:1]}")

            def move(target, phase):
                # Reuse the catalog's existing full-model collision-checked IK path.
                path = context.plan_to(target, "HOLD" if runtime.attachment else "RETREAT", cartesian=True)
                lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
                arc = np.r_[0., np.cumsum(lengths)]
                duration = max(1., arc[-1] / recipe.joint_speed_rad_s * 1.5,
                    np.linalg.norm(target[:3, 3] - context.grip_pose()[:3, 3]) / recipe.cartesian_speed_m_s * 1.5)
                for frac in np.linspace(0., 1., math.ceil(duration * 50) + 1)[1:]:
                    u = frac * frac * (3. - 2. * frac) * arc[-1]
                    tick(np.array([np.interp(u, arc, path[:, j]) for j in range(6)]), phase)
                for _ in range(30): tick(path[-1], phase)
                return path[-1]

            transported = context.grip_pose().copy()
            # The box walls border the long tools' source locations. Use a
            # vertical transfer for this attachment test, not a packing path.
            transported[2, 3] += .04
            move(transported, "TRANSPORT")
            returned = context.body_pose().copy()
            returned[:2, 3] = initial_position[:2] + np.asarray(args.release_offset_xy)
            # Test release at the source support, in the actual held orientation.
            # Long tools can tilt during the existing function; use their lowest
            # point, not their original center height, to prevent penetration.
            returned[2, 3] += context.support_top_z + .012 - context.bottom_height()
            target_grip = returned @ inverse(reference)
            q = move(target_grip, "PLACE")
            release_position = context.body_pose()[:3, 3].copy()
            release_object(runtime, object_id)
            context.carried_pose = None
            for _ in range(75): tick(q, "RELEASE_SETTLE")
            settled = context.body_pose()[:3, 3].copy()
            retreat = context.grip_pose().copy()
            retreat[2, 3] += .12
            move(retreat, "RETREAT")
            final = context.body_pose()[:3, 3].copy()
            attached_rows = [r for r in rows if r["attached"]]
            drift = float(np.linalg.norm(final - settled))
            metrics = {
                "max_attached_position_error_m": max(r["position_error_m"] for r in attached_rows),
                "max_attached_angle_error_deg": max(r["angle_error_deg"] for r in attached_rows),
                "object_motion_during_retreat_m": drift,
                "drop_after_release_m": float(release_position[2] - settled[2]),
                "final_attached_object_id": runtime.attached_object_id,
                "environment_shared": id(runtime.env) == env_identity,
                "attachment_reused": result.attachment_reused,
            }
            assert metrics["max_attached_position_error_m"] < .001
            assert metrics["max_attached_angle_error_deg"] < 1.
            assert runtime.attached_object_id is None and not runtime.grasp_engaged
            assert drift < .01, f"object followed retreat: {drift}"
            record.update(status="SUCCESS", metrics=metrics)
            if recorder: recorder.hold_final_frame(2.)
        except Exception as error:
            import traceback
            record.update(detail=str(error))
            (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            save_json(directory / "lifecycle_trace.json", rows)
            save_json(directory / "validation.json", record)
            if context: context.close()
            if recorder: recorder.close()
            if runtime: runtime.close()
        results.append(record)
        save_json(args.output_dir / "summary.json", results)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if all(r["status"] == "SUCCESS" for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
