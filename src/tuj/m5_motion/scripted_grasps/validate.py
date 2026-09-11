"""Lightweight scripted-grasp validation without M1→M5 or LLM calls.

Static checks resolve a recipe, build PRE/GRASP/LIFT from object geometry, and
flag seating / binding / EE issues. Controller checks reuse ``execute_grasp``.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from tuj.m5_motion.schema import (
    ArtifactProvenance, MotionGoal, MotionPlanRequest, MotionTask,
    RobotState, SceneRef, WorldSnapshot,
)
from tuj.m5_motion.scripted_grasps.catalog_types import build_catalog_targets
from tuj.m5_motion.scripted_grasps.frames import pose_dict, transform
from tuj.m5_motion.scripted_grasps.registry import (
    GraspEntry, integration_status, resolve,
)

C3_2_ENVIRONMENT = "C3_2_BreakfastTrayPreparation"

# M4 EE map for breakfast-tray acquires. Validators never invent recipes for
# missing registrations; they report NOT_REGISTERED instead.
C3_2_EXPECTED_ACQUIRES: tuple[tuple[str, str], ...] = (
    ("plate_a", "vac"),
    ("plate_b", "vac"),
    ("bread_a", "vac"),
    ("bread_b", "vac"),
    ("fruit_a", "3F"),
    ("fruit_b", "3F"),
    ("spoon_a", "3F"),
    ("spoon_b", "3F"),
    ("fork_a", "3F"),
    ("fork_b", "3F"),
    ("mug_a", "3F"),
    ("mug_b", "3F"),
)

ATTACH_DISCONTINUITY_M = 0.001
GEOMETRY_SIZE_TOLERANCE_FRACTION = 0.12
GEOMETRY_SIZE_TOLERANCE_MIN_M = 0.004


def _acquire_request(environment: str, object_id: str, ee: str) -> MotionPlanRequest:
    return MotionPlanRequest(
        request_id="scripted-validate",
        provenance=ArtifactProvenance(
            artifact_id="scripted-validate",
            artifact_type="MotionPlanRequest",
            produced_by="MOTION_PLANNER",
            invocation_id="validate",
        ),
        world=WorldSnapshot(
            scene=SceneRef(signature="validate"),
            robot_state=RobotState(
                robot_id="robot", joint_names=["j1"], joint_positions_rad=[0.0],
            ),
            metadata={"environment_name": environment, "physical_active_ee": ee},
        ),
        task=MotionTask(
            task_id="validate",
            subgoal_id="pick",
            action_type="PICK",
            ee=ee,
            target_ids=[object_id],
            goal=MotionGoal(goal_type="POSE", target_object_id=object_id),
            metadata={
                "attach_target": False,
                "scripted_grasp_validator_experimental": True,
            },
        ),
    )


def resolve_case(
    object_id: str,
    ee: str,
    *,
    environment: str = C3_2_ENVIRONMENT,
) -> dict[str, Any]:
    """Resolve one acquire. Never invents a recipe for an unregistered type."""
    from tuj.m5_motion.scripted_grasps.registry import ScriptedGraspUnavailable

    request = _acquire_request(environment, object_id, ee)
    try:
        entry = resolve(request)
    except ValueError as error:
        return {
            "status": "EE_MISMATCH",
            "object_id": object_id,
            "ee": ee,
            "environment": environment,
            "entry": None,
            "detail": str(error),
        }
    except ScriptedGraspUnavailable as error:
        return {
            "status": "NOT_REGISTERED",
            "object_id": object_id,
            "ee": ee,
            "environment": environment,
            "entry": None,
            "detail": str(error),
        }
    if entry is None:
        return {
            "status": "NOT_REGISTERED",
            "object_id": object_id,
            "ee": ee,
            "environment": environment,
            "entry": None,
            "detail": f"no scripted grasp for {object_id!r} + {ee!r} in {environment}",
        }
    if entry.ee != ee:
        return {
            "status": "EE_MISMATCH",
            "object_id": object_id,
            "ee": ee,
            "environment": environment,
            "entry": entry,
            "detail": f"resolved ee {entry.ee!r} != requested {ee!r}",
        }
    if entry.scene_object_id != object_id:
        return {
            "status": "BODY_BINDING_ERROR",
            "object_id": object_id,
            "ee": ee,
            "environment": environment,
            "entry": entry,
            "detail": f"scene id {entry.scene_object_id!r} != request {object_id!r}",
        }
    return {
        "status": "RESOLVED",
        "object_id": object_id,
        "ee": ee,
        "environment": environment,
        "entry": entry,
        "detail": None,
    }


def build_targets_for_entry(
    entry: GraspEntry,
    T_WB: np.ndarray,
    center_in_body_m: np.ndarray,
    local_size_m: np.ndarray,
):
    """Call the object module's target builder; fall back to catalog/spoon."""
    recipe = entry.recipe()
    name = entry.module_name or entry.object_id
    module = import_module(f"tuj.m5_motion.scripted_grasps.objects.{name}")
    builder = getattr(module, f"build_{name}_targets", None)
    if builder is not None:
        return builder(T_WB, center_in_body_m, local_size_m, recipe), recipe
    if entry.driver == "catalog":
        return build_catalog_targets(T_WB, center_in_body_m, local_size_m, recipe), recipe
    if entry.driver == "spoon":
        from tuj.m5_motion.scripted_grasps.objects.spoon import build_spoon_targets
        return build_spoon_targets(T_WB, center_in_body_m, local_size_m, recipe), recipe
    raise ValueError(f"NO_TARGET_BUILDER: {entry.object_id}/{entry.driver}")


def _pose_from_world_record(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = record["pose"]
    T_WB = transform(
        pose["position_m"],
        quaternion_xyzw=pose["orientation_xyzw"],
    )
    center = np.asarray(
        record.get("anchors", {}).get("center", [0.0, 0.0, 0.0]), dtype=float,
    )
    size = np.asarray(record["dimensions_m"], dtype=float)
    return T_WB, center, size


def _aabb_top_z(T_WB: np.ndarray, center: np.ndarray, size: np.ndarray) -> float:
    world_center = T_WB[:3, 3] + T_WB[:3, :3] @ center
    return float(world_center[2] + np.abs(T_WB[2, :3]) @ (size / 2.0))


def _aabb_bottom_z(T_WB: np.ndarray, center: np.ndarray, size: np.ndarray) -> float:
    world_center = T_WB[:3, 3] + T_WB[:3, :3] @ center
    return float(world_center[2] - np.abs(T_WB[2, :3]) @ (size / 2.0))


def _size_mismatch(size: np.ndarray, expected: np.ndarray) -> bool:
    return bool(
        np.any(
            np.abs(size - expected)
            > np.maximum(
                GEOMETRY_SIZE_TOLERANCE_MIN_M,
                expected * GEOMETRY_SIZE_TOLERANCE_FRACTION,
            )
        )
    )


def attachment_discontinuity_m(
    pose_before: np.ndarray,
    pose_after: np.ndarray,
) -> float:
    """Translational jump between pre/post attach object poses."""
    return float(
        np.linalg.norm(
            np.asarray(pose_after)[:3, 3] - np.asarray(pose_before)[:3, 3]
        )
    )


def static_validate_case(
    object_id: str,
    ee: str,
    *,
    environment: str = C3_2_ENVIRONMENT,
    T_WB: np.ndarray | None = None,
    center_in_body_m: np.ndarray | None = None,
    local_size_m: np.ndarray | None = None,
    support_top_z: float | None = None,
    world_object: dict | None = None,
) -> dict[str, Any]:
    """Geometry-level validation for one instance. No MuJoCo / LLM required."""
    resolved = resolve_case(object_id, ee, environment=environment)
    report: dict[str, Any] = {
        "object_id": object_id,
        "ee": ee,
        "environment": environment,
        "static_status": resolved["status"],
        "controller_status": "NOT_RUN",
        "failures": [],
        "warnings": [],
    }
    if resolved["status"] != "RESOLVED":
        report["detail"] = resolved["detail"]
        return report

    entry: GraspEntry = resolved["entry"]
    recipe = entry.recipe()
    report.update(
        resolved_type=entry.object_id,
        recipe_id=getattr(recipe, "recipe_id", None),
        module_name=entry.module_name,
        scene_object_id=entry.scene_object_id,
        body_object_id=entry.body_object_id,
        integration_status=integration_status(entry),
        driver=entry.driver,
    )

    if world_object is not None:
        T_WB, center_in_body_m, local_size_m = _pose_from_world_record(world_object)
    if T_WB is None:
        T_WB = np.eye(4)
        T_WB[:3, 3] = (0.4, 0.0, 0.92)
    if center_in_body_m is None:
        center_in_body_m = np.zeros(3)
    if local_size_m is None:
        expected = getattr(recipe, "expected_size_m", None)
        if expected is None:
            report["failures"].append("MISSING_EXPECTED_SIZE")
            report["static_status"] = "FAIL"
            return report
        local_size_m = np.asarray(expected, dtype=float)
    else:
        local_size_m = np.asarray(local_size_m, dtype=float)
        expected = getattr(recipe, "expected_size_m", None)
        if expected is not None and _size_mismatch(
            local_size_m, np.asarray(expected, dtype=float)
        ):
            report["failures"].append("UNSUPPORTED_OBJECT_GEOMETRY")

    try:
        targets, recipe = build_targets_for_entry(
            entry,
            np.asarray(T_WB, dtype=float),
            np.asarray(center_in_body_m, dtype=float),
            local_size_m,
        )
    except Exception as error:
        report["failures"].append(
            f"TARGET_BUILD_FAILED: {type(error).__name__}: {error}"
        )
        report["static_status"] = "FAIL"
        return report

    report["targets"] = {
        "PRE_GRASP": pose_dict(targets["PRE_GRASP"]),
        "GRASP": pose_dict(targets["GRASP"]),
        "LIFT": pose_dict(targets["LIFT"]),
    }
    report["local_size_m"] = local_size_m.tolist()
    report["center_in_body_m"] = np.asarray(center_in_body_m, dtype=float).tolist()

    top_z = _aabb_top_z(
        np.asarray(T_WB), np.asarray(center_in_body_m), local_size_m
    )
    bottom_z = _aabb_bottom_z(
        np.asarray(T_WB), np.asarray(center_in_body_m), local_size_m
    )
    grasp_z = float(targets["GRASP"][2, 3])
    lift_z = float(targets["LIFT"][2, 3])
    report["object_aabb_top_z"] = top_z
    report["object_aabb_bottom_z"] = bottom_z
    report["grasp_tcp_z"] = grasp_z
    report["lift_tcp_z"] = lift_z
    report["grasp_to_top_m"] = grasp_z - top_z
    report["lift_delta_m"] = lift_z - grasp_z

    if entry.scene_object_id != object_id:
        report["failures"].append("WRONG_TARGET_BODY_BINDING")

    lift_distance = float(getattr(recipe, "lift_distance_m", lift_z - grasp_z))
    if abs(lift_z - grasp_z - lift_distance) > 1e-9:
        report["failures"].append("LIFT_STANDOFF_MISMATCH")

    # Vacuum recipes whose cup face coincides with the grip site must not seat
    # the TCP below the AABB top: that presses the free body into its support.
    if ee == "vac" and grasp_z < top_z - 1e-9:
        report["failures"].append("GRASP_SEATS_BELOW_AABB_TOP")
        report["failures"].append("GRASP_MAY_PRESS_OBJECT_INTO_SUPPORT")

    if support_top_z is not None:
        report["support_top_z"] = float(support_top_z)
        clearance = bottom_z - float(support_top_z)
        report["rest_support_clearance_m"] = clearance
        immersion = max(0.0, top_z - grasp_z)
        report["commanded_top_immersion_m"] = immersion
        estimated_lift_clearance = clearance - immersion
        report["estimated_first_lift_support_clearance_m"] = estimated_lift_clearance
        if clearance < -1e-9:
            report["failures"].append("OBJECT_ALREADY_PENETRATING_SUPPORT_AT_REST")
        if immersion > 1e-9 and estimated_lift_clearance < -1e-9:
            report["failures"].append("FIRST_LIFT_ESTIMATE_STILL_PENETRATING_SUPPORT")

    grasp_xy = targets["GRASP"][:2, 3]
    body_xy = np.asarray(T_WB)[:2, 3] + (
        np.asarray(T_WB)[:2, :2] @ np.asarray(center_in_body_m)[:2]
    )
    half_xy = local_size_m[:2] / 2.0
    if np.any(np.abs(grasp_xy - body_xy) > half_xy + 1e-6):
        report["warnings"].append("GRASP_XY_OUTSIDE_AABB")

    report["static_status"] = "PASS" if not report["failures"] else "FAIL"
    return report


def load_world_objects(path: Path) -> dict[str, dict]:
    import json
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("objects", payload)


def batch_static_c3_2(
    *,
    world_path: Path | None = None,
    support_top_z: float | None = None,
) -> dict[str, Any]:
    """Cheap static batch over the M4 c3_2 acquire matrix."""
    worlds = load_world_objects(world_path) if world_path is not None else {}
    rows = []
    for object_id, ee in C3_2_EXPECTED_ACQUIRES:
        record = worlds.get(object_id)
        rows.append(
            static_validate_case(
                object_id,
                ee,
                world_object=record,
                support_top_z=support_top_z,
            )
        )
    return {
        "environment": C3_2_ENVIRONMENT,
        "mode": "STATIC",
        "cases": rows,
        "summary_table": format_batch_table(rows),
        "pass_count": sum(1 for row in rows if row["static_status"] == "PASS"),
        "fail_count": sum(1 for row in rows if row["static_status"] == "FAIL"),
        "not_registered_count": sum(
            1 for row in rows if row["static_status"] == "NOT_REGISTERED"
        ),
        "ee_mismatch_count": sum(
            1 for row in rows if row["static_status"] == "EE_MISMATCH"
        ),
    }


def format_batch_table(rows: list[dict[str, Any]]) -> str:
    header = (
        f"{'object':<10} {'EE':<4} {'recipe':<28} "
        f"{'static':<14} {'controller':<10}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        recipe = row.get("recipe_id") or row.get("static_status") or "-"
        lines.append(
            f"{row['object_id']:<10} {row['ee']:<4} {str(recipe):<28} "
            f"{row.get('static_status', '-'):<14} "
            f"{row.get('controller_status', 'NOT_RUN'):<10}"
        )
    return "\n".join(lines)


def controller_validate_case(
    object_id: str,
    ee: str,
    output: Path,
    *,
    environment: str = C3_2_ENVIRONMENT,
    seed: int = 0,
    video: Path | None = None,
    camera: str = "agentview",
    repository: Path | None = None,
) -> dict[str, Any]:
    """Run PRE→GRASP→attach/close→LIFT only via existing execute_grasp."""
    from tuj.m5_motion.scripted_grasps.context import execute_grasp
    from tuj.m5_motion.scripted_grasps.profiles import settle_tool_use_journal_free_objects
    from tuj.m5_motion.scripted_grasps.runtime import GraspFailure, save_json
    from tuj.m5_motion.scripted_grasps.settings import REPOSITORY
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime

    static = static_validate_case(object_id, ee, environment=environment)
    report = {
        **static,
        "controller_status": "FAILED",
        "stages": {},
        "camera": camera,
    }
    if static["static_status"] in {
        "NOT_REGISTERED", "EE_MISMATCH", "BODY_BINDING_ERROR",
    }:
        report["controller_status"] = static["static_status"]
        return report

    entry = resolve_case(object_id, ee, environment=environment)["entry"]
    assert entry is not None
    root = Path(repository or REPOSITORY)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = recorder = None
    try:
        recording = {}
        if video is not None:
            recording = {
                "camera_names": camera,
                "camera_heights": 540,
                "camera_widths": 960,
            }
        runtime = ToolUseJournalEERuntime.from_repository_for_controller(
            root,
            environment,
            active_ee=entry.ee,
            seed=seed,
            scripted_grasps=True,
            ignore_done=True,
            has_renderer=False,
            has_offscreen_renderer=video is not None,
            use_camera_obs=False,
            render_camera=camera,
            **recording,
        )
        settle_tool_use_journal_free_objects(runtime.env, duration_s=2.0)
        if video is not None:
            from tuj.m5_motion.generic_runner import GenericSimulationVideoRecorder
            recorder = GenericSimulationVideoRecorder(
                runtime, Path(video).resolve(), camera=camera,
                width=960, height=540, fps=20.0,
            )
        from tuj.m5_motion.tool_use_journal import ToolUseJournalEnvironmentAdapter
        adapter = ToolUseJournalEnvironmentAdapter(runtime.env)
        body_id = int(runtime.env.obj_body_id[entry.scene_object_id])
        pose_before = transform(
            adapter.data.xpos[body_id],
            rotation=adapter.data.xmat[body_id].reshape(3, 3),
        )
        result = execute_grasp(runtime, entry, output / "grasp", seed=seed)
        pose_after = transform(
            adapter.data.xpos[body_id],
            rotation=adapter.data.xmat[body_id].reshape(3, 3),
        )
        sequence_displacement = attachment_discontinuity_m(pose_before, pose_after)
        report["object_sequence_displacement_m"] = sequence_displacement
        report["object_pose_before_grasp"] = pose_dict(pose_before)
        report["object_pose_after_grasp"] = pose_dict(pose_after)
        # The full-sequence displacement includes the commanded LIFT and is not
        # an attachment discontinuity. Vacuum execution already records the
        # actual object pose immediately before and after attach; validate that
        # narrow transition instead.
        vacuum_attachment = result.get("vacuum_attachment")
        if vacuum_attachment is not None:
            attach_jump = float(
                vacuum_attachment.get("raw_attach_translation_norm_m", 0.0)
            )
            report["attachment_translation_jump_m"] = attach_jump
            report["attachment_rebind_translation_m"] = float(
                vacuum_attachment.get("attach_rebind_translation_norm_m", 0.0)
            )
            if attach_jump > ATTACH_DISCONTINUITY_M:
                report["warnings"].append("ATTACHMENT_TRANSLATION_DISCONTINUITY")
        report["stages"] = {
            "PRE_GRASP": "EXECUTED",
            "GRASP": "EXECUTED",
            "CONTACT_OR_ATTACH": "EXECUTED",
            "LIFT": "EXECUTED",
            "RETENTION": (
                "SUCCESS" if result.get("status") == "SUCCESS" else "FAILED"
            ),
        }
        report["result"] = {
            "status": result.get("status"),
            "failure_stage": result.get("failure_stage"),
            "failure_reason": result.get("failure_reason"),
            "metrics": result.get("metrics"),
        }
        report["controller_status"] = (
            "PASS" if result.get("status") == "SUCCESS" else "FAIL"
        )
        if result.get("status") != "SUCCESS":
            report["failures"].append(
                f"{result.get('failure_stage')}: {result.get('failure_reason')}"
            )
        if recorder is not None:
            recorder.hold_final_frame(1.0)
    except GraspFailure as error:
        report["controller_status"] = "FAIL"
        report["failures"].append(str(error))
        report["stages"]["failure"] = str(error)
    except Exception as error:
        report["controller_status"] = "FAIL"
        report["failures"].append(f"{type(error).__name__}: {error}")
    finally:
        if recorder is not None:
            recorder.close()
        if runtime is not None:
            runtime.close()
        save_json(output / "validation_report.json", report)
    return report


def c3_2_recipe_readiness() -> list[dict[str, str]]:
    """Classify each expected acquire without inventing recipes."""
    rows = []
    for object_id, ee in C3_2_EXPECTED_ACQUIRES:
        resolved = resolve_case(object_id, ee)
        if resolved["status"] == "NOT_REGISTERED":
            rows.append(
                {
                    "object_id": object_id,
                    "ee": ee,
                    "recipe": "-",
                    "readiness": "NOT_REGISTERED",
                    "static_status": "NOT_REGISTERED",
                }
            )
            continue
        if resolved["status"] != "RESOLVED":
            rows.append(
                {
                    "object_id": object_id,
                    "ee": ee,
                    "recipe": "-",
                    "readiness": resolved["status"],
                    "static_status": resolved["status"],
                }
            )
            continue
        entry = resolved["entry"]
        recipe = entry.recipe().recipe_id
        static = static_validate_case(object_id, ee)
        flags = ["READY_FOR_STATIC_VALIDATION"]
        if integration_status(entry) == "EXPERIMENTAL":
            flags.append("REQUIRES_PHYSICAL_CALIBRATION")
        if static["static_status"] == "FAIL":
            # Geometry already fails static checks; physical tuning still needed.
            if "REQUIRES_PHYSICAL_CALIBRATION" not in flags:
                flags.append("REQUIRES_PHYSICAL_CALIBRATION")
        rows.append(
            {
                "object_id": object_id,
                "ee": ee,
                "recipe": recipe,
                "readiness": "|".join(flags),
                "static_status": static["static_status"],
                "static_failures": list(static.get("failures") or []),
            }
        )
    return rows


__all__ = [
    "ATTACH_DISCONTINUITY_M",
    "C3_2_ENVIRONMENT",
    "C3_2_EXPECTED_ACQUIRES",
    "attachment_discontinuity_m",
    "batch_static_c3_2",
    "build_targets_for_entry",
    "c3_2_recipe_readiness",
    "controller_validate_case",
    "format_batch_table",
    "load_world_objects",
    "resolve_case",
    "static_validate_case",
]
