"""Sequential simulation execution, state validation, and task-goal evaluation.

The planning orchestrator deliberately predicts the state after each plan so it
can finalize a complete sequence without mutating a simulator.  This module is
the corresponding execution boundary: it replays those plans in order, checks
the observed state before continuing, evaluates the grounded motion goal, and
persists a manifest whose success status includes task completion.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from tuj.m5_motion.orchestration import SelectedPlanPlanningResult
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    ExecutionReport,
    ExecutionStatus,
    GoalType,
    ModuleName,
    MotionPlan,
    MotionPlanRequest,
    Pose,
    RobotState,
    SceneRef,
    SimulationConfig,
    SimulationRun,
    WorldSnapshot,
)
from tuj.m5_motion.task_semantics import (
    is_acquire_task,
    is_ee_exchange_task,
    is_release_task,
    task_operation,
)


class TrajectoryPlayer(Protocol):
    def execute(self, run: SimulationRun, *, report_id: str | None = None) -> ExecutionReport: ...


class SequenceExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    GOAL_FAILED = "GOAL_FAILED"
    STATE_DIVERGED = "STATE_DIVERGED"
    ABORTED = "ABORTED"


class GoalEvaluationStatus(str, Enum):
    SATISFIED = "SATISFIED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class GoalEvaluation:
    request_id: str
    goal_type: str
    status: GoalEvaluationStatus
    detail: str
    position_error_m: float | None = None
    orientation_error_rad: float | None = None
    joint_error_rad: float | None = None
    observed: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "goal_type": self.goal_type,
            "status": self.status.value,
            "detail": self.detail,
            "position_error_m": self.position_error_m,
            "orientation_error_rad": self.orientation_error_rad,
            "joint_error_rad": self.joint_error_rad,
            "observed": dict(self.observed or {}),
        }


@dataclass(frozen=True, slots=True)
class ExecutionAcceptance:
    """Execution-only tolerances, separate from geometric planning tolerances."""

    max_start_joint_error_rad: float = 0.20
    max_final_joint_error_rad: float = 0.20
    joint_goal_tolerance_rad: float = 0.02
    require_goal_verification: bool = True

    def __post_init__(self) -> None:
        if self.max_start_joint_error_rad < 0:
            raise ValueError("max_start_joint_error_rad must be non-negative")
        if self.max_final_joint_error_rad < 0:
            raise ValueError("max_final_joint_error_rad must be non-negative")
        if self.joint_goal_tolerance_rad <= 0:
            raise ValueError("joint_goal_tolerance_rad must be positive")


class GoalEvaluator(Protocol):
    def evaluate(
        self,
        request: MotionPlanRequest,
        report: ExecutionReport,
        observed_world: WorldSnapshot | None,
    ) -> GoalEvaluation: ...


PlayerSource = TrajectoryPlayer | Callable[[MotionPlanRequest, MotionPlan, int], TrajectoryPlayer]
ConfigSource = SimulationConfig | Callable[[MotionPlanRequest, MotionPlan, int], SimulationConfig]
WorldSnapshotProvider = Callable[[MotionPlanRequest, ExecutionReport], WorldSnapshot]
SequenceGoalEvaluator = Callable[
    [SelectedPlanPlanningResult, Sequence[ExecutionReport], WorldSnapshot],
    GoalEvaluation,
]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:120] or _digest(value)[:24]


def _quaternion_angle(left: Sequence[float], right: Sequence[float]) -> float:
    q_left = np.asarray(left, dtype=float)
    q_right = np.asarray(right, dtype=float)
    q_left /= np.linalg.norm(q_left)
    q_right /= np.linalg.norm(q_right)
    cosine = float(np.clip(abs(np.dot(q_left, q_right)), -1.0, 1.0))
    return 2.0 * math.acos(cosine)


def _pose_from_record(record: Any) -> Pose | None:
    if isinstance(record, Pose):
        return record
    if not isinstance(record, Mapping):
        return None
    nested = record.get("pose")
    if nested is not None and nested is not record:
        resolved = _pose_from_record(nested)
        if resolved is not None:
            return resolved
    position = record.get("position_m", record.get("position"))
    orientation = record.get(
        "orientation_xyzw",
        record.get("orientation", record.get("quaternion_xyzw")),
    )
    if position is None or orientation is None:
        return None
    try:
        return Pose(
            frame_id=str(record.get("frame_id") or "world"),
            position_m=tuple(float(value) for value in position),
            orientation_xyzw=tuple(float(value) for value in orientation),
        )
    except (TypeError, ValueError):
        return None


def _pose_errors(actual: Pose, target: Pose) -> tuple[float, float]:
    if actual.frame_id != target.frame_id:
        raise ValueError(
            f"pose frame mismatch: observed {actual.frame_id!r}, target {target.frame_id!r}"
        )
    position_error = float(
        np.linalg.norm(
            np.asarray(actual.position_m, dtype=float)
            - np.asarray(target.position_m, dtype=float)
        )
    )
    orientation_error = _quaternion_angle(
        actual.orientation_xyzw, target.orientation_xyzw
    )
    return position_error, orientation_error


class GroundedMotionGoalEvaluator:
    """Evaluate every current ``GoalType`` without treating missing data as PASS."""

    def __init__(self, *, joint_tolerance_rad: float = 0.02) -> None:
        if joint_tolerance_rad <= 0:
            raise ValueError("joint_tolerance_rad must be positive")
        self._joint_tolerance_rad = joint_tolerance_rad

    @staticmethod
    def _result(
        request: MotionPlanRequest,
        status: GoalEvaluationStatus,
        detail: str,
        **kwargs: Any,
    ) -> GoalEvaluation:
        return GoalEvaluation(
            request_id=request.request_id,
            goal_type=request.task.goal.goal_type.value,
            status=status,
            detail=detail,
            **kwargs,
        )

    def _evaluate_pose(
        self,
        request: MotionPlanRequest,
        actual: Pose | None,
        *,
        label: str,
    ) -> GoalEvaluation:
        target = request.task.goal.target_pose
        if target is None or actual is None:
            return self._result(
                request,
                GoalEvaluationStatus.UNKNOWN,
                f"{label} pose is unavailable",
            )
        try:
            position_error, orientation_error = _pose_errors(actual, target)
        except ValueError as error:
            return self._result(
                request, GoalEvaluationStatus.FAILED, str(error)
            )
        satisfied = (
            position_error <= request.constraints.position_tolerance_m
            and orientation_error <= request.constraints.orientation_tolerance_rad
        )
        return self._result(
            request,
            (
                GoalEvaluationStatus.SATISFIED
                if satisfied
                else GoalEvaluationStatus.FAILED
            ),
            (
                f"{label} pose satisfies the grounded tolerance"
                if satisfied
                else f"{label} pose exceeds the grounded tolerance"
            ),
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            observed={
                "position_tolerance_m": request.constraints.position_tolerance_m,
                "orientation_tolerance_rad": (
                    request.constraints.orientation_tolerance_rad
                ),
            },
        )

    def evaluate(
        self,
        request: MotionPlanRequest,
        report: ExecutionReport,
        observed_world: WorldSnapshot | None,
    ) -> GoalEvaluation:
        goal = request.task.goal
        state = report.final_robot_state
        if state is None:
            return self._result(
                request,
                GoalEvaluationStatus.UNKNOWN,
                "execution did not produce a final robot state",
            )

        operation = task_operation(request.task)

        if operation == "PICK_TOOL":
            target = goal.target_object_id or request.task.metadata.get("tool_id")
            held = state.held_tool_id or state.attached_object_id
            satisfied = held == target
            return self._result(
                request,
                GoalEvaluationStatus.SATISFIED if satisfied else GoalEvaluationStatus.FAILED,
                (
                    f"tool {target!r} is held"
                    if satisfied
                    else f"expected held tool {target!r}, observed {held!r}"
                ),
                observed={"held_tool_id": held, "target_tool_id": target},
            )

        if operation in {"RETURN_TOOL", "TERMINAL_RETURN_TOOL"}:
            target = goal.target_object_id or request.task.metadata.get("tool_id")
            held = state.held_tool_id or state.attached_object_id
            satisfied = held != target
            return self._result(
                request,
                GoalEvaluationStatus.SATISFIED if satisfied else GoalEvaluationStatus.FAILED,
                (
                    f"tool {target!r} was returned"
                    if satisfied
                    else f"tool {target!r} is still held"
                ),
                observed={"held_tool_id": held, "target_tool_id": target},
            )

        if is_acquire_task(request.task):
            target = goal.target_object_id or next(iter(request.task.target_ids), None)
            if not target:
                return self._result(
                    request,
                    GoalEvaluationStatus.UNKNOWN,
                    "PICK goal has no target object",
                )
            satisfied = state.attached_object_id == target
            return self._result(
                request,
                GoalEvaluationStatus.SATISFIED if satisfied else GoalEvaluationStatus.FAILED,
                (
                    f"target {target!r} is attached"
                    if satisfied
                    else f"target {target!r} is not attached"
                ),
                observed={"attached_object_id": state.attached_object_id},
            )

        if is_release_task(request.task):
            target = goal.target_object_id or next(iter(request.task.target_ids), None)
            if not target:
                return self._result(
                    request,
                    GoalEvaluationStatus.UNKNOWN,
                    "PLACE goal has no target object",
                )
            if state.attached_object_id == target:
                return self._result(
                    request,
                    GoalEvaluationStatus.FAILED,
                    f"placed object {target!r} is still attached",
                    observed={"attached_object_id": state.attached_object_id},
                )
            actual = (
                _pose_from_record(observed_world.objects.get(target))
                if observed_world is not None
                else None
            )
            return self._evaluate_pose(request, actual, label=f"object {target!r}")

        if is_ee_exchange_task(request.task):
            actual = report.metadata.get("final_active_ee")
            expected = request.task.ee
            try:
                from tuj.m5_motion.precomputed_ee_attach import normalize_ee_id

                expected = normalize_ee_id(expected)
                actual = normalize_ee_id(actual)
            except ValueError:
                # Non-rack custom EEs retain the general exact-match behavior.
                pass
            satisfied = actual == expected
            return self._result(
                request,
                GoalEvaluationStatus.SATISFIED if satisfied else GoalEvaluationStatus.FAILED,
                (
                    f"EE {expected!r} is active"
                    if satisfied
                    else f"expected active EE {expected!r}, observed {actual!r}"
                ),
                observed={"final_active_ee": actual},
            )

        if goal.goal_type is GoalType.POSE:
            return self._evaluate_pose(request, state.eef_pose, label="EEF")

        if goal.goal_type is GoalType.JOINT:
            target = goal.target_joint_positions_rad
            if target is None or len(target) != len(state.joint_positions_rad):
                return self._result(
                    request,
                    GoalEvaluationStatus.UNKNOWN,
                    "JOINT goal dimensions do not match the final robot state",
                )
            error = float(
                np.max(
                    np.abs(
                        np.asarray(state.joint_positions_rad, dtype=float)
                        - np.asarray(target, dtype=float)
                    )
                )
            )
            satisfied = error <= self._joint_tolerance_rad
            return self._result(
                request,
                GoalEvaluationStatus.SATISFIED if satisfied else GoalEvaluationStatus.FAILED,
                (
                    "joint goal satisfies the execution tolerance"
                    if satisfied
                    else "joint goal exceeds the execution tolerance"
                ),
                joint_error_rad=error,
                observed={"joint_tolerance_rad": self._joint_tolerance_rad},
            )

        return self._result(
            request,
            GoalEvaluationStatus.UNKNOWN,
            f"goal type {goal.goal_type.value!r} has no evaluator",
        )


class SimulationArtifactStore:
    """Atomically persist runs, reports, goal evaluations, and one manifest."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def _atomic_json(path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(path)

    def save_run(self, run: SimulationRun, *, index: int) -> Path:
        path = self.root / "runs" / f"{index:04d}-{_safe_name(run.run_id)}.json"
        self._atomic_json(path, run.model_dump_json(indent=2))
        return path.resolve()

    def save_report(self, report: ExecutionReport, *, index: int) -> Path:
        path = self.root / "reports" / (
            f"{index:04d}-{_safe_name(report.report_id)}.json"
        )
        self._atomic_json(path, report.model_dump_json(indent=2))
        return path.resolve()

    def save_goal(self, evaluation: GoalEvaluation, *, index: int) -> Path:
        path = self.root / "goals" / (
            f"{index:04d}-{_safe_name(evaluation.request_id)}.json"
        )
        self._atomic_json(
            path, json.dumps(evaluation.as_dict(), ensure_ascii=False, indent=2)
        )
        return path.resolve()

    def save_manifest(
        self,
        *,
        status: SequenceExecutionStatus,
        runs: Sequence[SimulationRun],
        reports: Sequence[ExecutionReport],
        goals: Sequence[GoalEvaluation],
        run_paths: Sequence[Path],
        report_paths: Sequence[Path],
        goal_paths: Sequence[Path],
        final_world: WorldSnapshot,
        failed_index: int | None,
        detail: str,
    ) -> Path:
        manifest = {
            "manifest_version": "1.0.0",
            "status": status.value,
            "successful": status is SequenceExecutionStatus.SUCCESS,
            "failed_index": failed_index,
            "detail": detail,
            "run_ids": [run.run_id for run in runs],
            "report_ids": [report.report_id for report in reports],
            "goal_evaluations": [goal.as_dict() for goal in goals],
            "run_files": [str(path) for path in run_paths],
            "report_files": [str(path) for path in report_paths],
            "goal_files": [str(path) for path in goal_paths],
            "final_scene_signature": final_world.scene.signature,
            "final_robot_state": final_world.robot_state.model_dump(mode="json"),
        }
        path = self.root / "simulation-manifest.json"
        self._atomic_json(path, json.dumps(manifest, ensure_ascii=False, indent=2))
        return path.resolve()


@dataclass(frozen=True, slots=True)
class SelectedPlanExecutionResult:
    planning: SelectedPlanPlanningResult
    status: SequenceExecutionStatus
    runs: tuple[SimulationRun, ...]
    reports: tuple[ExecutionReport, ...]
    goal_evaluations: tuple[GoalEvaluation, ...]
    final_world: WorldSnapshot
    failed_index: int | None = None
    detail: str = ""
    run_paths: tuple[Path, ...] = ()
    report_paths: tuple[Path, ...] = ()
    goal_paths: tuple[Path, ...] = ()
    manifest_path: Path | None = None

    @property
    def successful(self) -> bool:
        return self.status is SequenceExecutionStatus.SUCCESS


def _state_error(actual: RobotState, expected: RobotState) -> tuple[float, str | None]:
    if actual.robot_id != expected.robot_id:
        return math.inf, "robot_id differs from the planned state"
    if actual.joint_names != expected.joint_names:
        return math.inf, "joint_names differ from the planned state"
    error = float(
        np.max(
            np.abs(
                np.asarray(actual.joint_positions_rad, dtype=float)
                - np.asarray(expected.joint_positions_rad, dtype=float)
            )
        )
    )
    if actual.attached_object_id != expected.attached_object_id:
        return error, (
            "attached object differs from the planned state: "
            f"expected {expected.attached_object_id!r}, observed {actual.attached_object_id!r}"
        )
    if actual.held_tool_id != expected.held_tool_id:
        return error, (
            "held tool differs from the planned state: "
            f"expected {expected.held_tool_id!r}, observed {actual.held_tool_id!r}"
        )
    return error, None


def _fallback_world(
    request: MotionPlanRequest,
    report: ExecutionReport,
    *,
    goal_satisfied: bool,
) -> WorldSnapshot:
    result = request.world.model_copy(deep=True)
    if report.final_robot_state is not None:
        result.robot_state = report.final_robot_state.model_copy(deep=True)
    completed = list(result.scene.completed_subgoals)
    is_transition = is_ee_exchange_task(request.task) or ":transition:" in request.task.subgoal_id
    if goal_satisfied and not is_transition and request.task.subgoal_id not in completed:
        completed.append(request.task.subgoal_id)
    signature = _digest(
        {
            "previous": request.world.scene.signature,
            "report_id": report.report_id,
            "robot_state": result.robot_state.model_dump(mode="json"),
        }
    )
    result.scene = SceneRef(
        signature=f"observed:{signature}",
        completed_subgoals=completed,
        facts=list(result.scene.facts),
    )
    return result


def _observed_world_with_completion(
    world: WorldSnapshot,
    request: MotionPlanRequest,
    *,
    goal_satisfied: bool,
) -> WorldSnapshot:
    result = world.model_copy(deep=True)
    completed = list(result.scene.completed_subgoals)
    is_transition = is_ee_exchange_task(request.task) or ":transition:" in request.task.subgoal_id
    if goal_satisfied and not is_transition and request.task.subgoal_id not in completed:
        completed.append(request.task.subgoal_id)
    if completed != result.scene.completed_subgoals:
        signature = _digest(
            {
                "observed_scene_signature": result.scene.signature,
                "completed_subgoals": completed,
            }
        )
        result.scene = SceneRef(
            signature=f"observed:{signature}",
            completed_subgoals=completed,
            facts=list(result.scene.facts),
        )
    return result


class SelectedPlanSimulationOrchestrator:
    """Replay a finalized plan sequence and fail closed on unverified goals."""

    def __init__(
        self,
        player: PlayerSource,
        *,
        config: ConfigSource,
        store: SimulationArtifactStore | None = None,
        goal_evaluator: GoalEvaluator | None = None,
        world_snapshot_provider: WorldSnapshotProvider | None = None,
        sequence_goal_evaluator: SequenceGoalEvaluator | None = None,
        acceptance: ExecutionAcceptance | None = None,
    ) -> None:
        self._player = player
        self._config = config
        self._store = store
        self._acceptance = acceptance or ExecutionAcceptance()
        self._goal_evaluator = goal_evaluator or GroundedMotionGoalEvaluator(
            joint_tolerance_rad=self._acceptance.joint_goal_tolerance_rad
        )
        self._world_snapshot_provider = world_snapshot_provider
        self._sequence_goal_evaluator = sequence_goal_evaluator

    @staticmethod
    def _source(source: Any, request: MotionPlanRequest, plan: MotionPlan, index: int, expected: type, label: str) -> Any:
        value = source(request, plan, index) if callable(source) else source
        if not isinstance(value, expected):
            raise TypeError(f"{label} provider returned {type(value).__name__}")
        return value

    @staticmethod
    def _run(request: MotionPlanRequest, plan: MotionPlan, config: SimulationConfig, index: int) -> SimulationRun:
        identity = _digest(
            {
                "index": index,
                "request_id": request.request_id,
                "plan_id": plan.plan_id,
                "config": config.model_dump(mode="json"),
            }
        )
        run_id = f"simulation-run:{index:04d}:{identity[:20]}"
        return SimulationRun(
            run_id=run_id,
            provenance=ArtifactProvenance(
                artifact_id=f"simulation-run-artifact:{identity[:24]}",
                artifact_type="SimulationRun",
                produced_by=ModuleName.SIMULATOR,
                invocation_id=run_id,
                input_artifact_ids=[plan.provenance.artifact_id],
                metadata={"request_id": request.request_id, "sequence_index": index},
            ),
            plan=plan,
            config=config.model_copy(deep=True),
        )

    def execute(self, planning: SelectedPlanPlanningResult) -> SelectedPlanExecutionResult:
        if len(planning.requests) != len(planning.plans):
            raise ValueError("planning result request/plan counts differ")
        if not planning.plans:
            raise ValueError("planning result contains no MotionPlan")

        runs: list[SimulationRun] = []
        reports: list[ExecutionReport] = []
        goals: list[GoalEvaluation] = []
        run_paths: list[Path] = []
        report_paths: list[Path] = []
        goal_paths: list[Path] = []
        final_world = planning.requests[0].world.model_copy(deep=True)
        status = SequenceExecutionStatus.SUCCESS
        detail = "all plans executed and all grounded goals were satisfied"
        failed_index: int | None = None
        previous_state: RobotState | None = None

        for index, (request, plan) in enumerate(
            zip(planning.requests, planning.plans)
        ):
            if plan.request_id != request.request_id:
                raise ValueError(f"plan {index} does not belong to request {index}")
            if previous_state is not None:
                first_waypoint = plan.segments[0].waypoints[0]
                if previous_state.joint_names != plan.joint_names:
                    status = SequenceExecutionStatus.STATE_DIVERGED
                    detail = "next plan joint_names differ from the observed runtime state"
                    failed_index = index
                    break
                start_error = float(
                    np.max(
                        np.abs(
                            np.asarray(previous_state.joint_positions_rad, dtype=float)
                            - np.asarray(first_waypoint.joint_positions_rad, dtype=float)
                        )
                    )
                )
                if start_error > self._acceptance.max_start_joint_error_rad:
                    status = SequenceExecutionStatus.STATE_DIVERGED
                    detail = (
                        f"plan {index} starts {start_error:.6f} rad from the observed "
                        "runtime state"
                    )
                    failed_index = index
                    break

            config = self._source(
                self._config, request, plan, index, SimulationConfig, "SimulationConfig"
            )
            player = self._source(
                self._player, request, plan, index, object, "trajectory player"
            )
            if not hasattr(player, "execute"):
                raise TypeError("trajectory player provider returned an object without execute()")
            run = self._run(request, plan, config, index)
            report = player.execute(run)
            if report.run_id != run.run_id or report.plan_id != plan.plan_id:
                raise ValueError("ExecutionReport identity does not match its SimulationRun")
            runs.append(run)
            if self._store is not None:
                run_paths.append(self._store.save_run(run, index=index))

            if report.status is not ExecutionStatus.SUCCESS:
                reports.append(report)
                if self._store is not None:
                    report_paths.append(self._store.save_report(report, index=index))
                status = SequenceExecutionStatus.EXECUTION_FAILED
                detail = (
                    report.failure.message
                    if report.failure is not None
                    else f"plan {index} execution failed"
                )
                failed_index = index
                break
            if report.final_robot_state is None:
                reports.append(report)
                if self._store is not None:
                    report_paths.append(self._store.save_report(report, index=index))
                status = SequenceExecutionStatus.STATE_DIVERGED
                detail = f"plan {index} succeeded without a final robot state"
                failed_index = index
                break

            final_error, state_detail = _state_error(
                report.final_robot_state, plan.expected_final_state
            )
            if state_detail is not None or final_error > self._acceptance.max_final_joint_error_rad:
                reports.append(report)
                if self._store is not None:
                    report_paths.append(self._store.save_report(report, index=index))
                status = SequenceExecutionStatus.STATE_DIVERGED
                detail = state_detail or (
                    f"plan {index} final state differs by {final_error:.6f} rad"
                )
                failed_index = index
                previous_state = report.final_robot_state
                break

            observed_world = (
                self._world_snapshot_provider(request, report)
                if self._world_snapshot_provider is not None
                else None
            )
            evaluation = self._goal_evaluator.evaluate(
                request, report, observed_world
            )
            report = report.model_copy(
                update={
                    "metrics": report.metrics.model_copy(
                        update={
                            "goal_position_error_m": evaluation.position_error_m,
                            "goal_orientation_error_rad": (
                                evaluation.orientation_error_rad
                            ),
                        }
                    ),
                    "metadata": {
                        **report.metadata,
                        "goal_evaluation": evaluation.as_dict(),
                    },
                }
            )
            reports.append(report)
            if self._store is not None:
                report_paths.append(self._store.save_report(report, index=index))
            goals.append(evaluation)
            if self._store is not None:
                goal_paths.append(self._store.save_goal(evaluation, index=index))

            goal_satisfied = evaluation.status is GoalEvaluationStatus.SATISFIED
            final_world = (
                _observed_world_with_completion(
                    observed_world,
                    request,
                    goal_satisfied=goal_satisfied,
                )
                if observed_world is not None
                else _fallback_world(
                    request, report, goal_satisfied=goal_satisfied
                )
            )
            previous_state = report.final_robot_state
            if (
                self._acceptance.require_goal_verification
                and not goal_satisfied
            ):
                status = SequenceExecutionStatus.GOAL_FAILED
                detail = evaluation.detail
                failed_index = index
                break

        if (
            status is SequenceExecutionStatus.SUCCESS
            and self._sequence_goal_evaluator is not None
        ):
            sequence_goal = self._sequence_goal_evaluator(
                planning, tuple(reports), final_world
            )
            goals.append(sequence_goal)
            if self._store is not None:
                goal_paths.append(
                    self._store.save_goal(sequence_goal, index=len(goals) - 1)
                )
            if sequence_goal.status is not GoalEvaluationStatus.SATISFIED:
                status = SequenceExecutionStatus.GOAL_FAILED
                detail = sequence_goal.detail
                failed_index = len(planning.plans) - 1

        manifest: Path | None = None
        if self._store is not None:
            manifest = self._store.save_manifest(
                status=status,
                runs=runs,
                reports=reports,
                goals=goals,
                run_paths=run_paths,
                report_paths=report_paths,
                goal_paths=goal_paths,
                final_world=final_world,
                failed_index=failed_index,
                detail=detail,
            )
        return SelectedPlanExecutionResult(
            planning=planning,
            status=status,
            runs=tuple(runs),
            reports=tuple(reports),
            goal_evaluations=tuple(goals),
            final_world=final_world,
            failed_index=failed_index,
            detail=detail,
            run_paths=tuple(run_paths),
            report_paths=tuple(report_paths),
            goal_paths=tuple(goal_paths),
            manifest_path=manifest,
        )


__all__ = [
    "ConfigSource",
    "ExecutionAcceptance",
    "GoalEvaluation",
    "GoalEvaluationStatus",
    "GoalEvaluator",
    "GroundedMotionGoalEvaluator",
    "PlayerSource",
    "SelectedPlanExecutionResult",
    "SelectedPlanSimulationOrchestrator",
    "SequenceExecutionStatus",
    "SequenceGoalEvaluator",
    "SimulationArtifactStore",
    "TrajectoryPlayer",
    "WorldSnapshotProvider",
]
