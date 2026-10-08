"""Collision-checked SAFE RACK EXIT after EE attach / exchange.

This is the physical postcondition of EE exchange: after the requested EE is
mounted and verified, leave the rack interaction region under the *mounted* EE
geometry.  Orchestration no longer emits a separate task-level hop; the attach
or exchange planner appends this exit into the same success contract.

The commissioned ``bare_to_{ee}`` attach trajectory records a bare-flange
workspace start as ``start_joint_positions_rad``, but that q is not guaranteed
collision-free once the EE is mounted.  This planner therefore:

1. Prefer a collision-checked reverse of the commissioned attach trajectory
   when that reverse already reaches a mounted-valid workspace target.
2. If the reverse stops inside the rack corridor, try a direct or RRT route
   to that target before stitching the partial reverse to a free-space suffix.
3. Never accept a short reverse that remains in the rack corridor as SAFE EXIT.
"""

from __future__ import annotations

import enum
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from tuj.m5_motion.path_planning import RRTConnectEdgePlanner, validate_joint_segment
from tuj.m5_motion.precomputed_ee_attach import (
    CollisionChecker,
    EEAttachPathFailureCode,
    EEAttachTrajectoryTemplate,
    PrecomputedEEAttachRegistry,
    PrecomputedEEPathError,
    _robot_model,
    normalize_ee_id,
    rebase_portable_pose,
)
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    CollisionContext,
    GoalType,
    InterpolationType,
    KeyframePlannerType,
    KeyframeType,
    ModuleName,
    MotionPlan,
    MotionPlanRequest,
    Pose,
    RelativeKeyframeSpec,
    RobotState,
    SegmentType,
    TrajectoryProcessingStep,
    TrajectorySegment,
)
from tuj.m5_motion.task_semantics import task_operation
from tuj.m5_motion.trajectory_processing import (
    QuinticTimeParameterizer,
    TrajectoryProcessingError,
    deterministic_shortcut,
)

_JOINT_MATCH_TOL_RAD = 1e-6
# A timed sample can sit between the coarser geometric samples and miss the
# required clearance by a fraction of a millimetre. Repair that sample locally
# instead of treating the whole exit as blocked. Keep the shift small so the
# commissioned corridor is not replaced by a new route.
_CLEARANCE_REPAIR_ATTEMPTS = 4
_CLEARANCE_REPAIR_PROBE_RAD = 1e-3
_CLEARANCE_REPAIR_MAX_SHIFT_RAD = 0.02
# A path that only just meets the scene margin is inside it once the torque
# controller tracks it. Plan the exit this much further out. Execution still
# uses the requested margin.
_RACK_EXIT_CLEARANCE_PAD_M = 0.002


class _PaddedClearanceChecker:
    """Ask the exit checker for the scene margin plus a tracking pad."""

    def __init__(self, inner: CollisionChecker, pad_m: float) -> None:
        self._inner = inner
        self._pad = float(pad_m)

    def check(self, joint_config, keyframe=None, **kwargs):  # noqa: ANN001
        pad = float(kwargs.pop("extra_clearance_m", 0.0)) + self._pad
        try:
            return self._inner.check(
                joint_config, keyframe, extra_clearance_m=pad, **kwargs
            )
        except TypeError:
            return self._inner.check(joint_config, keyframe, **kwargs)

    def final_segment_validator(self, waypoints, context, **kwargs):  # noqa: ANN001
        inner = getattr(self._inner, "final_segment_validator", None)
        if not callable(inner):
            return True
        pad = float(kwargs.pop("extra_clearance_m", 0.0)) + self._pad
        try:
            return inner(waypoints, context, extra_clearance_m=pad, **kwargs)
        except TypeError:
            return inner(waypoints, context, **kwargs)

    @property
    def last_path_collision_check(self):
        return getattr(self._inner, "last_path_collision_check", None)


def safe_rack_exit_context_id(active_ee: str) -> str:
    """Distinct ACM id for mounted rack exit (neighbor-support allowances)."""

    return f"ee-attached-safe-rack-exit:{normalize_ee_id(active_ee)}"


def is_move_to_workspace_request(request: MotionPlanRequest) -> bool:
    return task_operation(request.task) == "MOVE_TO_WORKSPACE"


class MoveToWorkspaceFailureCode(str, enum.Enum):
    INVALID_REQUEST = "MOVE_TO_WORKSPACE_INVALID_REQUEST"
    PATH_NOT_FOUND = "MOVE_TO_WORKSPACE_PATH_NOT_FOUND"
    DYNAMICS_INVALID = "MOVE_TO_WORKSPACE_DYNAMICS_INVALID"
    FINAL_COLLISION_CHECK_FAILED = "MOVE_TO_WORKSPACE_FINAL_COLLISION_CHECK_FAILED"
    WORKSPACE_POSE_UNAVAILABLE = "MOVE_TO_WORKSPACE_POSE_UNAVAILABLE"


class MoveToWorkspacePlanningError(RuntimeError):
    def __init__(
        self,
        failure_code: MoveToWorkspaceFailureCode,
        detail: str,
        *,
        trajectory_id: str | None = None,
    ) -> None:
        self.failure_code = failure_code
        self.detail = detail
        self.trajectory_id = trajectory_id
        super().__init__(f"{failure_code.value}: {detail}")


class JointPositionLimitsProvider(Protocol):
    @property
    def joint_limits_rad(self) -> Sequence[tuple[float, float]]: ...


def _minimum_clearance(
    path: Sequence[Sequence[float]],
    keyframe: RelativeKeyframeSpec,
    collision_checker: CollisionChecker,
    max_joint_step_rad: float,
) -> float | None:
    minimum: float | None = None
    for source, target in zip(path, path[1:]):
        report = validate_joint_segment(
            source,
            target,
            keyframe,
            collision_checker,  # type: ignore[arg-type]
            max_joint_step_rad=max_joint_step_rad,
            wrap_joints=False,
        )
        if report.min_clearance_m is None:
            continue
        minimum = (
            report.min_clearance_m
            if minimum is None
            else min(minimum, report.min_clearance_m)
        )
    return minimum


def _joint_max_abs_delta(
    left: Sequence[float], right: Sequence[float]
) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def _joint_l2_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))
    )


def _blend_on_segment(
    point: Sequence[float],
    source: Sequence[float],
    target: Sequence[float],
) -> float | None:
    """Blend in (0, 1) when ``point`` lies on the open joint segment."""

    start = np.asarray(source, dtype=float)
    finish = np.asarray(target, dtype=float)
    sample = np.asarray(point, dtype=float)
    delta = finish - start
    length_sq = float(np.dot(delta, delta))
    if length_sq <= 1e-18:
        return None
    blend = float(np.dot(sample - start, delta) / length_sq)
    if not 1e-4 < blend < 1.0 - 1e-4:
        return None
    if float(np.max(np.abs(start + delta * blend - sample))) > 1e-5:
        return None
    return blend


def _insert_between(
    path: Sequence[Sequence[float]],
    failing: Sequence[float],
    repaired: Sequence[float],
) -> tuple[tuple[float, ...], ...] | None:
    points = [tuple(float(value) for value in item) for item in path]
    for index, (source, target) in enumerate(zip(points, points[1:])):
        if _blend_on_segment(failing, source, target) is None:
            continue
        if _joint_max_abs_delta(source, repaired) <= _JOINT_MATCH_TOL_RAD:
            return None
        if _joint_max_abs_delta(target, repaired) <= _JOINT_MATCH_TOL_RAD:
            return None
        points.insert(index + 1, tuple(float(value) for value in repaired))
        return tuple(points)
    return None


def _nudge_to_clearance(
    failing: Sequence[float],
    keyframe: RelativeKeyframeSpec,
    collision_checker: CollisionChecker,
    limits: Sequence[tuple[float, float]],
) -> tuple[float, ...] | None:
    """Step one grazing sample along the clearance gradient until it clears."""

    origin = np.asarray(failing, dtype=float)
    current = origin.copy()
    lower = np.asarray([bound[0] for bound in limits], dtype=float)
    upper = np.asarray([bound[1] for bound in limits], dtype=float)
    if current.shape != lower.shape:
        return None

    def report_at(config: np.ndarray) -> Any:
        return collision_checker.check(
            tuple(float(value) for value in config), keyframe
        )

    for _ in range(_CLEARANCE_REPAIR_ATTEMPTS):
        report = report_at(current)
        if bool(getattr(report, "valid", False)):
            if float(np.max(np.abs(current - origin))) <= (
                _CLEARANCE_REPAIR_MAX_SHIFT_RAD + 1e-9
            ):
                return tuple(float(value) for value in current)
            return None
        clearance = getattr(report, "min_clearance_m", None)
        if (
            clearance is None
            or not math.isfinite(float(clearance))
            or float(clearance) <= 0.0
        ):
            return None
        gradient = np.zeros(current.shape, dtype=float)
        for index in range(current.size):
            trial = current.copy()
            trial[index] = min(float(upper[index]), float(trial[index]) + _CLEARANCE_REPAIR_PROBE_RAD)
            sign = 1.0
            if trial[index] <= current[index] + 1e-12:
                trial[index] = max(
                    float(lower[index]),
                    float(current[index]) - _CLEARANCE_REPAIR_PROBE_RAD,
                )
                sign = -1.0
            moved = abs(float(trial[index] - current[index]))
            if moved <= 1e-12:
                continue
            probed = report_at(trial)
            if bool(getattr(probed, "valid", False)):
                if float(np.max(np.abs(trial - origin))) <= (
                    _CLEARANCE_REPAIR_MAX_SHIFT_RAD + 1e-9
                ):
                    return tuple(float(value) for value in trial)
                probed_clearance = float(clearance) + moved
            else:
                probed_clearance = getattr(probed, "min_clearance_m", None)
                if probed_clearance is None or not math.isfinite(float(probed_clearance)):
                    continue
                probed_clearance = float(probed_clearance)
            gradient[index] = sign * (probed_clearance - float(clearance)) / moved
        scale = float(np.linalg.norm(gradient))
        if scale <= 1e-12:
            return None
        remaining = _CLEARANCE_REPAIR_MAX_SHIFT_RAD - float(
            np.max(np.abs(current - origin))
        )
        if remaining <= 1e-6:
            return None
        current = np.clip(
            current + gradient / scale * min(remaining, _CLEARANCE_REPAIR_PROBE_RAD),
            lower,
            upper,
        )
    report = report_at(current)
    if bool(getattr(report, "valid", False)) and float(
        np.max(np.abs(current - origin))
    ) <= _CLEARANCE_REPAIR_MAX_SHIFT_RAD + 1e-9:
        return tuple(float(value) for value in current)
    return None


def _repair_timed_margin_graze(
    path: Sequence[Sequence[float]],
    timed_waypoints: Sequence[Any],
    report: Any,
    keyframe: RelativeKeyframeSpec,
    collision_checker: CollisionChecker,
    limits: Sequence[tuple[float, float]],
) -> tuple[tuple[float, ...], ...] | None:
    if getattr(report, "failure_code", None) != "COLLISION_MARGIN_VIOLATION":
        return None
    index = getattr(report, "failed_state_index", None)
    if not isinstance(index, int) or not 0 <= index < len(timed_waypoints):
        return None
    failing = tuple(
        float(value) for value in timed_waypoints[index].joint_positions_rad
    )
    state = collision_checker.check(failing, keyframe)
    clearance = getattr(state, "min_clearance_m", None)
    if (
        bool(getattr(state, "valid", False))
        or clearance is None
        or not math.isfinite(float(clearance))
        or float(clearance) <= 0.0
    ):
        return None
    nudged = _nudge_to_clearance(failing, keyframe, collision_checker, limits)
    if nudged is None:
        return None
    return _insert_between(path, failing, nudged)


def _flatten_attach_waypoints(
    template: EEAttachTrajectoryTemplate,
) -> list[tuple[tuple[float, ...], Pose | None]]:
    waypoints: list[tuple[tuple[float, ...], Pose | None]] = []
    for segment in template.segments:
        for item in segment.waypoints:
            joints = tuple(float(value) for value in item.joint_positions_rad)
            pose = item.eef_pose.model_copy(deep=True) if item.eef_pose is not None else None
            if waypoints and _joint_max_abs_delta(waypoints[-1][0], joints) <= _JOINT_MATCH_TOL_RAD:
                # Keep the later duplicate's pose (retreat end often equals staging end).
                waypoints[-1] = (joints, pose)
                continue
            waypoints.append((joints, pose))
    return waypoints


def _mount_aligned_index(
    waypoints: Sequence[tuple[tuple[float, ...], Pose | None]],
    start: Sequence[float],
) -> int:
    """Index of the current retreat state on the attach polyline.

    Prefer the earliest near-match so reverse walk leaves the rack via staging
    rather than re-entering the dock corridor (retreat end often duplicates
    staging end).
    """

    near = [
        index
        for index, (joints, _) in enumerate(waypoints)
        if _joint_max_abs_delta(joints, start) <= _JOINT_MATCH_TOL_RAD
    ]
    if near:
        return min(near)
    return min(
        range(len(waypoints)),
        key=lambda index: _joint_l2_distance(waypoints[index][0], start),
    )


def commissioned_reverse_workspace_path(
    start: Sequence[float],
    template: EEAttachTrajectoryTemplate,
    keyframe: RelativeKeyframeSpec,
    collision_checker: CollisionChecker,
    *,
    max_joint_step_rad: float,
) -> tuple[tuple[float, ...], ...]:
    """Longest collision-checked reverse along commissioned attach joints."""

    waypoints = _flatten_attach_waypoints(template)
    if not waypoints:
        return (tuple(float(value) for value in start),)
    aligned = _mount_aligned_index(waypoints, start)
    path: list[tuple[float, ...]] = [tuple(float(value) for value in start)]
    current = path[0]
    for index in range(aligned - 1, -1, -1):
        nxt = waypoints[index][0]
        state = collision_checker.check(nxt, keyframe)
        if not state.valid:
            break
        edge = validate_joint_segment(
            current,
            nxt,
            keyframe,
            collision_checker,  # type: ignore[arg-type]
            max_joint_step_rad=max_joint_step_rad,
            wrap_joints=False,
        )
        if not edge.valid:
            break
        path.append(nxt)
        current = nxt
    return tuple(path)


def select_mounted_workspace_target(
    template: EEAttachTrajectoryTemplate,
    keyframe: RelativeKeyframeSpec,
    collision_checker: CollisionChecker,
) -> tuple[tuple[float, ...], Pose | None, str]:
    """Pick a mounted-EE-valid workspace joint target from attach data."""

    bare_start = tuple(float(value) for value in template.start_joint_positions_rad)
    bare_pose = (
        template.start_eef_pose.model_copy(deep=True)
        if template.start_eef_pose is not None
        else None
    )
    if collision_checker.check(bare_start, keyframe).valid:
        return bare_start, bare_pose, "commissioned-start"
    best: tuple[float, tuple[float, ...], Pose | None] | None = None
    for joints, pose in _flatten_attach_waypoints(template):
        if not collision_checker.check(joints, keyframe).valid:
            continue
        distance = _joint_l2_distance(joints, bare_start)
        if best is None or distance < best[0]:
            best = (distance, joints, pose)
    if best is None:
        raise MoveToWorkspacePlanningError(
            MoveToWorkspaceFailureCode.WORKSPACE_POSE_UNAVAILABLE,
            "no mounted-EE collision-free waypoint on the attach template",
            trajectory_id=template.trajectory_id,
        )
    return best[1], best[2], "nearest-mounted-valid-attach-waypoint"


class MoveToWorkspacePlanner:
    """Plan rack-retreat → commissioned workspace-safe joint transit."""

    def __init__(
        self,
        registry: PrecomputedEEAttachRegistry,
        *,
        joint_position_limits_rad: Sequence[tuple[float, float]],
        log: Any = print,
    ) -> None:
        self.registry = registry
        self.joint_position_limits_rad = tuple(joint_position_limits_rad)
        self._log = log

    def load_workspace_template(
        self, request: MotionPlanRequest
    ) -> EEAttachTrajectoryTemplate:
        environment = str(request.world.metadata.get("environment_name") or "")
        active = normalize_ee_id(
            request.world.metadata.get("physical_active_ee") or request.task.ee
        )
        return self.registry.load(environment, active, world=request.world)

    def _validate_request(
        self,
        request: MotionPlanRequest,
        template: EEAttachTrajectoryTemplate,
    ) -> str:
        if not is_move_to_workspace_request(request):
            raise MoveToWorkspacePlanningError(
                MoveToWorkspaceFailureCode.INVALID_REQUEST,
                "request is not MOVE_TO_WORKSPACE",
                trajectory_id=template.trajectory_id,
            )
        state = request.world.robot_state
        try:
            physical = normalize_ee_id(
                request.world.metadata.get("physical_active_ee")
            )
            requested = normalize_ee_id(request.task.ee)
        except ValueError as error:
            raise MoveToWorkspacePlanningError(
                MoveToWorkspaceFailureCode.INVALID_REQUEST,
                str(error),
                trajectory_id=template.trajectory_id,
            ) from error
        if physical != requested or physical != template.target_active_ee:
            raise MoveToWorkspacePlanningError(
                MoveToWorkspaceFailureCode.INVALID_REQUEST,
                "mounted EE does not match MOVE_TO_WORKSPACE / attach template",
                trajectory_id=template.trajectory_id,
            )
        if state.attached_object_id is not None or state.held_tool_id is not None:
            raise MoveToWorkspacePlanningError(
                MoveToWorkspaceFailureCode.INVALID_REQUEST,
                "MOVE_TO_WORKSPACE requires an empty mounted end effector",
                trajectory_id=template.trajectory_id,
            )
        if state.joint_names != template.joint_names:
            raise MoveToWorkspacePlanningError(
                MoveToWorkspaceFailureCode.INVALID_REQUEST,
                "joint name or order differs from the workspace attach template",
                trajectory_id=template.trajectory_id,
            )
        if _robot_model(request.world) != template.robot_model:
            raise MoveToWorkspacePlanningError(
                MoveToWorkspaceFailureCode.WORKSPACE_POSE_UNAVAILABLE,
                "robot model differs from the workspace attach template",
                trajectory_id=template.trajectory_id,
            )
        if len(self.joint_position_limits_rad) != len(template.joint_names):
            raise MoveToWorkspacePlanningError(
                MoveToWorkspaceFailureCode.INVALID_REQUEST,
                "joint position limit DOF differs from the workspace template",
                trajectory_id=template.trajectory_id,
            )
        missing_dynamics = set(template.joint_names) - set(
            request.constraints.joint_limits
        )
        if missing_dynamics:
            raise MoveToWorkspacePlanningError(
                MoveToWorkspaceFailureCode.DYNAMICS_INVALID,
                f"request has no dynamic limits for {sorted(missing_dynamics)}",
                trajectory_id=template.trajectory_id,
            )
        return physical

    def _ensure_target_within_limits(
        self,
        target: Sequence[float],
        template: EEAttachTrajectoryTemplate,
    ) -> None:
        for name, value, (lower, upper) in zip(
            template.joint_names,
            target,
            self.joint_position_limits_rad,
        ):
            if value < lower - 1e-9 or value > upper + 1e-9:
                raise MoveToWorkspacePlanningError(
                    MoveToWorkspaceFailureCode.WORKSPACE_POSE_UNAVAILABLE,
                    f"workspace joint {name} exceeds its limit",
                    trajectory_id=template.trajectory_id,
                )

    def plan(
        self,
        request: MotionPlanRequest,
        *,
        collision_contexts: Mapping[str, CollisionContext],
        collision_checker: CollisionChecker,
        template: EEAttachTrajectoryTemplate | None = None,
    ) -> MotionPlan:
        selected = template or self.load_workspace_template(request)
        active = self._validate_request(request, selected)
        collision_checker = _PaddedClearanceChecker(
            collision_checker, _RACK_EXIT_CLEARANCE_PAD_M
        )
        preferred_ids = (
            (safe_rack_exit_context_id(active), f"ee-attached:{active}")
            if bool(request.task.metadata.get("safe_rack_exit"))
            else (f"ee-attached:{active}",)
        )
        context_id = next(
            (candidate for candidate in preferred_ids if candidate in collision_contexts),
            preferred_ids[-1],
        )
        context = collision_contexts.get(context_id)
        if context is None or context.active_ee != active:
            raise PrecomputedEEPathError(
                EEAttachPathFailureCode.PRECOMPUTED_EE_PATH_STALE,
                f"attached collision context for {active} is unavailable",
                trajectory_id=selected.trajectory_id,
            )
        keyframe = RelativeKeyframeSpec(
            keyframe_id=f"{request.task.subgoal_id}:move-to-workspace",
            keyframe_type=KeyframeType.CUSTOM,
            frame_ref="world",
            anchor="workspace",
            approach_axis_xyz=(0.0, 0.0, 1.0),
            planner=KeyframePlannerType.SAMPLING_BASED,
            collision_context_id=context_id,
            metadata={
                "active_ee": active,
                "workspace_trajectory_id": selected.trajectory_id,
            },
        )
        start = tuple(
            float(value) for value in request.world.robot_state.joint_positions_rad
        )
        if not collision_checker.check(start, keyframe).valid:
            report = collision_checker.check(start, keyframe)
            raise MoveToWorkspacePlanningError(
                MoveToWorkspaceFailureCode.PATH_NOT_FOUND,
                f"MOVE_TO_WORKSPACE start is invalid under {context_id}: {report.detail}",
                trajectory_id=selected.trajectory_id,
            )

        reverse_path = commissioned_reverse_workspace_path(
            start,
            selected,
            keyframe,
            collision_checker,
            max_joint_step_rad=request.constraints.max_joint_path_step_rad,
        )
        workspace_target, workspace_pose, workspace_source = (
            select_mounted_workspace_target(selected, keyframe, collision_checker)
        )
        self._ensure_target_within_limits(workspace_target, selected)

        # A short reverse that remains in the rack corridor must not count as
        # SAFE EXIT.  Prefer the mounted-valid workspace target, and only treat
        # pure reverse as done when it already reaches that target.
        reverse_reaches_workspace = (
            len(reverse_path) >= 2
            and _joint_max_abs_delta(reverse_path[-1], workspace_target)
            <= _JOINT_MATCH_TOL_RAD
        )
        if reverse_reaches_workspace:
            target = reverse_path[-1]
            target_pose = workspace_pose
            target_source = "commissioned-reverse"
            for joints, pose in _flatten_attach_waypoints(selected):
                if _joint_max_abs_delta(joints, target) <= _JOINT_MATCH_TOL_RAD:
                    target_pose = pose
                    break
        else:
            target = workspace_target
            target_pose = workspace_pose
            target_source = workspace_source

        seed = (
            reverse_path[-1]
            if len(reverse_path) >= 2 and not reverse_reaches_workspace
            else start
        )

        def _connect(source: Sequence[float]) -> tuple[tuple[float, ...], ...]:
            direct_edge = validate_joint_segment(
                source,
                target,
                keyframe,
                collision_checker,  # type: ignore[arg-type]
                max_joint_step_rad=request.constraints.max_joint_path_step_rad,
                wrap_joints=False,
            )
            if direct_edge.valid:
                return (tuple(float(v) for v in source), target)
            rrt = RRTConnectEdgePlanner(
                collision_checker,  # type: ignore[arg-type]
                self.joint_position_limits_rad,
                random_seed=request.options.random_seed,
                max_iterations=request.options.rrt_max_iterations,
                timeout_s=request.options.allowed_planning_time_s,
                extension_step_rad=request.options.rrt_extension_step_rad,
                validation_step_rad=request.constraints.max_joint_path_step_rad,
                goal_bias=request.options.rrt_goal_bias,
                wrap_joints=False,
            )
            result = rrt.plan(source, target, None, keyframe)
            if not result.valid:
                raise MoveToWorkspacePlanningError(
                    MoveToWorkspaceFailureCode.PATH_NOT_FOUND,
                    f"{result.failure_code or 'RRT_CONNECT_FAILED'}: {result.detail}",
                    trajectory_id=selected.trajectory_id,
                )
            path = result.joint_path
            if request.options.simplify_path:
                path = deterministic_shortcut(
                    path,
                    lambda left, right: validate_joint_segment(
                        left,
                        right,
                        keyframe,
                        collision_checker,  # type: ignore[arg-type]
                        max_joint_step_rad=request.constraints.max_joint_path_step_rad,
                        wrap_joints=False,
                    ).valid,
                )
            return path

        def _composed_from_reverse() -> tuple[tuple[float, ...], ...]:
            suffix = _connect(seed)
            if len(reverse_path) < 2:
                return suffix
            if _joint_max_abs_delta(reverse_path[-1], suffix[0]) > _JOINT_MATCH_TOL_RAD:
                raise MoveToWorkspacePlanningError(
                    MoveToWorkspaceFailureCode.PATH_NOT_FOUND,
                    "reverse seed does not match the free-space connector start",
                    trajectory_id=selected.trajectory_id,
                )
            return tuple(reverse_path) + tuple(suffix[1:])

        direct = validate_joint_segment(
            start,
            target,
            keyframe,
            collision_checker,  # type: ignore[arg-type]
            max_joint_step_rad=request.constraints.max_joint_path_step_rad,
            wrap_joints=False,
        )

        reverse_attempts: list[tuple[str, Any]] = []
        if reverse_reaches_workspace:
            reverse_attempts.append(("COMMISSIONED_REVERSE", lambda: reverse_path))
        elif len(reverse_path) >= 2:
            reverse_attempts.append(("COMMISSIONED_REVERSE_THEN_RRT", _composed_from_reverse))
        free_space_attempts: list[tuple[str, Any]] = []
        if direct.valid:
            free_space_attempts.append(("DIRECT_JOINT", lambda: (start, target)))
        free_space_attempts.append(("RRT_CONNECT", lambda: _connect(start)))
        # A reverse that already reaches the workspace stays in the commissioned
        # corridor. A free-space shortcut is only preferred when that corridor
        # stops short, so the exit is not sent across the rest of the scene.
        corridor_blocked = (
            request.task.metadata.get("safe_rack_exit") is True
            and not reverse_reaches_workspace
        )
        attempts = (
            free_space_attempts + reverse_attempts
            if corridor_blocked
            else reverse_attempts + free_space_attempts
        )

        final_detail = "timed trajectory failed final collision validation"
        planner_name = "RRT_CONNECT"
        timed = None
        minimum_clearance: float | None = None

        def _time(path: Sequence[Sequence[float]]) -> Any:
            return QuinticTimeParameterizer(
                sample_dt_s=request.options.interpolation_dt_s
            ).parameterize(
                selected.joint_names,
                path,
                request.constraints.joint_limits,
                velocity_scaling=request.constraints.velocity_scaling,
                acceleration_scaling=request.constraints.acceleration_scaling,
                jerk_scaling=request.constraints.jerk_scaling,
            )

        for planner_name, build_path in attempts:
            try:
                geometric_path = build_path()
            except MoveToWorkspacePlanningError as error:
                final_detail = str(error)
                continue
            try:
                timed = None
                for repair_index in range(_CLEARANCE_REPAIR_ATTEMPTS + 1):
                    minimum_clearance = _minimum_clearance(
                        geometric_path,
                        keyframe,
                        collision_checker,
                        request.constraints.max_joint_path_step_rad,
                    )
                    timed = _time(geometric_path)
                    final_validator = getattr(
                        collision_checker, "final_segment_validator", None
                    )
                    if not callable(final_validator) or final_validator(
                        timed.waypoints, context
                    ):
                        break
                    report = getattr(
                        collision_checker, "last_path_collision_check", None
                    )
                    if report is not None:
                        final_detail = f"{report.failure_code}: {report.detail}"
                    if repair_index == _CLEARANCE_REPAIR_ATTEMPTS:
                        timed = None
                        break
                    repaired = _repair_timed_margin_graze(
                        geometric_path,
                        timed.waypoints,
                        report,
                        keyframe,
                        collision_checker,
                        self.joint_position_limits_rad,
                    )
                    if repaired is None:
                        timed = None
                        break
                    geometric_path = repaired
            except TrajectoryProcessingError as error:
                raise MoveToWorkspacePlanningError(
                    MoveToWorkspaceFailureCode.DYNAMICS_INVALID,
                    str(error),
                    trajectory_id=selected.trajectory_id,
                ) from error
            if timed is not None:
                break
        else:
            raise MoveToWorkspacePlanningError(
                MoveToWorkspaceFailureCode.FINAL_COLLISION_CHECK_FAILED,
                final_detail,
                trajectory_id=selected.trajectory_id,
            )
        assert timed is not None
        digest = hashlib.sha256(
            (
                f"{request.request_id}|{request.world.scene.signature}|"
                f"{selected.trajectory_id}|move-to-workspace"
            ).encode("utf-8")
        ).hexdigest()[:24]
        segment = TrajectorySegment(
            segment_id=f"{request.task.subgoal_id}:move-to-workspace",
            segment_type=SegmentType.TRANSFER,
            start_time_s=0.0,
            end_time_s=timed.duration_s,
            interpolation=InterpolationType.QUINTIC,
            waypoints=list(timed.waypoints),
            collision_checked=True,
            min_clearance_m=minimum_clearance,
            collision_context_before=context,
            collision_context_after=context,
            processing_steps=(
                [TrajectoryProcessingStep.RAW_PATH]
                + (
                    [TrajectoryProcessingStep.SHORTCUT]
                    if planner_name == "RRT_CONNECT" and request.options.simplify_path
                    else []
                )
                + [
                    TrajectoryProcessingStep.TIME_PARAMETERIZATION,
                    TrajectoryProcessingStep.FINAL_COLLISION_CHECK,
                    TrajectoryProcessingStep.DYNAMICS_CHECK,
                ]
            ),
            metadata={
                "source": "dynamic-move-to-workspace",
                "planner": planner_name,
                "workspace_target_source": target_source,
                "workspace_trajectory_id": selected.trajectory_id,
            },
        )
        final_eef = rebase_portable_pose(selected, request.world, target_pose)
        if final_eef is None and selected.start_eef_pose is not None:
            final_eef = rebase_portable_pose(
                selected, request.world, selected.start_eef_pose
            )
        self._log(
            f"[M5][WORKSPACE] ee={active} planner={planner_name} "
            f"target={target_source} template={selected.trajectory_id}"
        )
        return MotionPlan(
            plan_id=f"motion-plan:{digest}",
            request_id=request.request_id,
            provenance=ArtifactProvenance(
                artifact_id=f"motion-plan-artifact:{digest}",
                artifact_type="MotionPlan",
                produced_by=ModuleName.MOTION_PLANNER,
                invocation_id=f"move-to-workspace:{request.request_id}",
                input_artifact_ids=[request.provenance.artifact_id],
                metadata={
                    "source": "dynamic-move-to-workspace",
                    "workspace_trajectory_id": selected.trajectory_id,
                    "workspace_target_source": target_source,
                    "target_active_ee": active,
                },
            ),
            scene_signature=request.world.scene.signature,
            robot_id=request.world.robot_state.robot_id,
            joint_names=list(selected.joint_names),
            duration_s=timed.duration_s,
            segments=[segment],
            events=[],
            expected_final_state=RobotState(
                robot_id=request.world.robot_state.robot_id,
                joint_names=list(selected.joint_names),
                joint_positions_rad=list(target),
                joint_velocities_rad_s=[0.0] * len(target),
                eef_pose=final_eef,
                gripper=(
                    request.world.robot_state.gripper.model_copy(deep=True)
                    if request.world.robot_state.gripper is not None
                    else None
                ),
                attached_object_id=None,
                held_tool_id=None,
            ),
            metadata={
                "source": "dynamic-move-to-workspace",
                "workspace_trajectory_id": selected.trajectory_id,
                "workspace_target_source": target_source,
                "target_active_ee": active,
                "goal_type": GoalType.JOINT.value,
                "safe_rack_exit": True,
            },
        )


def _unique_allowed_collision_pairs(
    *pair_lists: Sequence[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    """Merge ACM pairs without violating CollisionContext uniqueness."""

    merged: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pairs in pair_lists:
        if not pairs:
            continue
        for left, right in pairs:
            key = tuple(sorted((str(left), str(right))))
            if key in seen:
                continue
            seen.add(key)
            merged.append((str(left), str(right)))
    return merged


def append_safe_rack_exit_plan(
    exchange_plan: MotionPlan,
    exit_plan: MotionPlan,
) -> MotionPlan:
    """Append a validated SAFE RACK EXIT onto an attach/exchange plan.

    Exit failure must prevent treating the exchange as SUCCESS; callers raise
    before this merge when exit planning fails.
    """

    if exchange_plan.joint_names != exit_plan.joint_names:
        raise MoveToWorkspacePlanningError(
            MoveToWorkspaceFailureCode.INVALID_REQUEST,
            "safe-exit joint_names do not match the exchange plan",
        )
    offset = float(exchange_plan.duration_s)
    boundary = (
        exchange_plan.segments[-1].collision_context_after
        or exchange_plan.segments[-1].collision_context_before
    )
    shifted_segments: list[TrajectorySegment] = []
    for index, segment in enumerate(exit_plan.segments):
        waypoints = [
            waypoint.model_copy(
                update={
                    "time_from_start_s": float(waypoint.time_from_start_s) + offset
                }
            )
            for waypoint in segment.waypoints
        ]
        metadata = dict(segment.metadata)
        metadata["safe_rack_exit"] = True
        updates: dict[str, Any] = {
            "segment_id": f"{segment.segment_id}:safe-rack-exit",
            "start_time_s": float(segment.start_time_s) + offset,
            "end_time_s": float(segment.end_time_s) + offset,
            "waypoints": waypoints,
            "metadata": metadata,
        }
        exit_before = segment.collision_context_before
        exit_after = segment.collision_context_after
        if index == 0 and boundary is not None:
            # Preserve exit-only ACM pairs (neighbor rack supports) under a
            # distinct context_id so the merged plan does not reuse
            # ee-attached:{ee} with two incompatible definitions.
            active = (
                (exit_before.active_ee if exit_before is not None else None)
                or boundary.active_ee
                or "unknown"
            )
            stitched_id = (
                exit_before.context_id
                if exit_before is not None
                and exit_before.context_id != boundary.context_id
                else safe_rack_exit_context_id(str(active))
            )
            stitched = boundary.model_copy(
                update={
                    "context_id": stitched_id,
                    "allowed_collision_pairs": _unique_allowed_collision_pairs(
                        boundary.allowed_collision_pairs,
                        None
                        if exit_before is None
                        else exit_before.allowed_collision_pairs,
                        None
                        if exit_after is None
                        else exit_after.allowed_collision_pairs,
                    ),
                }
            )
            # Use one definition for both boundaries so free-pose / ACM fields
            # cannot diverge under the same context_id.
            updates["collision_context_before"] = stitched
            updates["collision_context_after"] = stitched.model_copy(deep=True)
        elif exit_before is not None and exit_after is not None:
            # Later exit segments must not keep the attach-era context_id when
            # they carry exit-only ACM pairs.
            active = exit_before.active_ee or exit_after.active_ee or "unknown"
            exit_id = safe_rack_exit_context_id(str(active))
            if exit_before.context_id != exit_id and exit_before.allowed_collision_pairs:
                updates["collision_context_before"] = exit_before.model_copy(
                    update={"context_id": exit_id}
                )
            if exit_after.context_id != exit_id and exit_after.allowed_collision_pairs:
                updates["collision_context_after"] = exit_after.model_copy(
                    update={"context_id": exit_id}
                )
        shifted_segments.append(segment.model_copy(update=updates))
    shifted_events = [
        event.model_copy(
            update={"time_from_start_s": float(event.time_from_start_s) + offset}
        )
        for event in exit_plan.events
    ]
    metadata = dict(exchange_plan.metadata)
    metadata["safe_rack_exit"] = True
    metadata["safe_rack_exit_planner"] = (
        exit_plan.segments[0].metadata.get("planner")
        if exit_plan.segments
        else None
    )
    metadata["workspace_target_source"] = exit_plan.metadata.get(
        "workspace_target_source"
    )
    metadata["workspace_trajectory_id"] = exit_plan.metadata.get(
        "workspace_trajectory_id"
    )
    return exchange_plan.model_copy(
        update={
            "duration_s": float(exchange_plan.duration_s)
            + float(exit_plan.duration_s),
            "segments": list(exchange_plan.segments) + shifted_segments,
            "events": list(exchange_plan.events) + shifted_events,
            "expected_final_state": exit_plan.expected_final_state.model_copy(
                deep=True
            ),
            "metadata": metadata,
        }
    )


__all__ = [
    "MoveToWorkspaceFailureCode",
    "MoveToWorkspacePlanner",
    "MoveToWorkspacePlanningError",
    "append_safe_rack_exit_plan",
    "commissioned_reverse_workspace_path",
    "is_move_to_workspace_request",
    "safe_rack_exit_context_id",
    "select_mounted_workspace_target",
]
