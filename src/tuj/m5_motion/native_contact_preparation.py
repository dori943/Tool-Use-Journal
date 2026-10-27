"""A separate, collision-checked EEF preparation before a native side push.

This creates a MOVE request, not a contact or task-goal completion. Execute it
through the normal compiler/player, then snapshot the measured held-object pose
again before planning contact. No attachment, force, or target state is changed.
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import numpy as np

from .contact_keyframe_validation import ContactKeyframeGeometryError
from .native_side_push import native_side_push_geometry
from .profiles import GraspExecutionProfile
from .schema import (
    GoalType, KeyframePlannerType, KeyframeType, MotionGoal, MotionPlanRequest,
    MotionTask, Pose, RelativeKeyframeSpec,
)


def build_native_contact_preparation(
    request: MotionPlanRequest, *, eef_orientation_xyzw: Sequence[float],
) -> tuple[MotionPlanRequest, list[RelativeKeyframeSpec]]:
    """Lift at current XY, orient at hover, and settle using existing limits.

    The first TRANSFER retains the live attitude while reaching safe height;
    the second rotates there and holds for the ordinary grasp hold duration.
    EEF tracking uses the original request tolerances and the builder's default
    wait budget. A completed preparation says nothing about the original goal.
    """
    live = request.world.robot_state.eef_pose
    if live is None or live.frame_id != "world":
        raise ContactKeyframeGeometryError("native preparation requires a world-frame live EEF pose")
    # Pose validates finiteness and unit quaternion before geometry is computed.
    goal_pose = Pose(frame_id="world", position_m=live.position_m,
                     orientation_xyzw=tuple(eef_orientation_xyzw))
    geometry = native_side_push_geometry(request, eef_orientation_xyzw=goal_pose.orientation_xyzw)
    position = (*live.position_m[:2], max(live.position_m[2], geometry.hover_tcp_z_m))
    goal_pose = goal_pose.model_copy(update={"position_m": position})
    suffix = ":native-contact-preparation"
    prepared = request.model_copy(deep=True)
    prepared.request_id += suffix
    prepared.provenance = prepared.provenance.model_copy(deep=True, update={
        "artifact_id": request.provenance.artifact_id + suffix,
        "input_artifact_ids": [*request.provenance.input_artifact_ids, request.provenance.artifact_id],
    })
    prepared.task = MotionTask(
        task_id=request.task.task_id + suffix,
        subgoal_id=request.task.subgoal_id + suffix,
        action_type="MOVE", ee=request.task.ee, tool=request.task.tool,
        target_ids=[], allowed_touch_objects=[], contact=None,
        goal=MotionGoal(goal_type=GoalType.POSE, target_pose=goal_pose),
        metadata={
            "native_contact_preparation": True,
            "source_request_id": request.request_id,
            "source_subgoal_id": request.task.subgoal_id,
            **({"ee_capabilities": deepcopy(request.task.metadata["ee_capabilities"])}
               if "ee_capabilities" in request.task.metadata else {}),
        },
    )
    # The established world-origin encoding separates EEF attitude from its
    # position. No attached-object pose tag or virtual scene object is needed.
    distance = float(np.linalg.norm(position))
    direction = tuple(float(v) for v in np.asarray(position) / distance) if distance > 0. else (0., 0., 1.)
    frames = [RelativeKeyframeSpec(
        keyframe_id=f"{prepared.task.subgoal_id}:{label}",
        keyframe_type=KeyframeType.TRANSFER, frame_ref="world", anchor="origin",
        approach_axis_xyz=direction, offset_along_approach_m=distance,
        planner=KeyframePlannerType.CARTESIAN,
        metadata={"packing_orientation_xyzw": list(attitude)},
    ) for label, attitude in (
        ("hover", live.orientation_xyzw), ("orient-and-hold", goal_pose.orientation_xyzw),
    )]
    frames[-1].metadata.update({
        "hold_duration_after_s": GraspExecutionProfile().hold_duration_s,
        "tracking_settle": {
            "eef_tolerance_m": request.constraints.position_tolerance_m,
            "eef_orientation_tolerance_rad": request.constraints.orientation_tolerance_rad,
        },
    })
    return prepared, frames


def build_native_contact_local_preparation(
    request: MotionPlanRequest, *, eef_orientation_xyzw: Sequence[float],
) -> tuple[MotionPlanRequest, list[RelativeKeyframeSpec]]:
    """Move above the observed side-push start and settle before re-observing.

    This is only an auxiliary MOVE. Execute it through the normal compiler and
    player, acquire a new measured WorldSnapshot, and regenerate SIDE_PUSH from
    that snapshot. The supplied observation is copied, never projected forward
    or changed into a claimed contact/task success.
    """
    prepared, frames = build_native_contact_preparation(
        request, eef_orientation_xyzw=eef_orientation_xyzw,
    )
    geometry = native_side_push_geometry(
        request, eef_orientation_xyzw=eef_orientation_xyzw,
    )
    goal = prepared.task.goal.target_pose
    assert goal is not None  # The ordinary preparation always creates a POSE goal.
    tcp_start_xy = (np.asarray(geometry.patch_start_xy_m)
                    - np.asarray(geometry.working_patch_offset_xy_m))
    position = (float(tcp_start_xy[0]), float(tcp_start_xy[1]), goal.position_m[2])
    prepared.task.goal.target_pose = goal.model_copy(update={"position_m": position})
    suffix = ":local-observation"
    prepared.request_id += suffix
    prepared.task.task_id += suffix
    prepared.task.subgoal_id += suffix
    prepared.provenance.artifact_id += suffix
    prepared.task.metadata["requires_measured_replanning_after_execution"] = True
    # Retain the first vertical lift at the live XY, then translate at the same
    # existing hover height. The final hold/tracking limits are inherited intact.
    distance = float(np.linalg.norm(position))
    frames[-1].approach_axis_xyz = (
        tuple(float(v) for v in np.asarray(position) / distance)
        if distance > 0. else (0., 0., 1.)
    )
    frames[-1].offset_along_approach_m = distance
    for index, frame in enumerate(frames):
        frame.keyframe_id = f"{prepared.task.subgoal_id}:{index}"
    return prepared, frames
