"""Composable execution predicates for region and contact manipulation goals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tuj.m5_motion.execution import (
    GoalEvaluation,
    GoalEvaluationStatus,
    GoalEvaluator,
    GroundedMotionGoalEvaluator,
)
from tuj.m5_motion.push_to_region import target_fully_inside_region
from tuj.m5_motion.schema import ExecutionReport, MotionPlanRequest, WorldSnapshot
from tuj.m5_motion.task_semantics import (
    is_acquire_task,
    is_ee_exchange_task,
    is_release_task,
)


def _result(
    request: MotionPlanRequest,
    status: GoalEvaluationStatus,
    detail: str,
    *,
    observed: Mapping[str, Any] | None = None,
) -> GoalEvaluation:
    return GoalEvaluation(
        request_id=request.request_id,
        goal_type=request.task.goal.goal_type.value,
        status=status,
        detail=detail,
        observed=observed,
    )


class RegionContainmentEvaluator:
    """Require every grounded target footprint to fit inside its region."""

    def __init__(self, *, inset_margin_m: float = 0.0) -> None:
        if inset_margin_m < 0.0:
            raise ValueError("inset margin must be non-negative")
        self._inset = inset_margin_m

    def evaluate(
        self,
        request: MotionPlanRequest,
        report: ExecutionReport,
        observed_world: WorldSnapshot | None,
    ) -> GoalEvaluation:
        del report
        region_id = request.task.goal.target_region_id
        targets = list(request.task.target_ids)
        if not region_id or not targets:
            return _result(
                request,
                GoalEvaluationStatus.UNKNOWN,
                "region containment requires target_ids and target_region_id",
            )
        if observed_world is None:
            return _result(
                request,
                GoalEvaluationStatus.UNKNOWN,
                "observed world is unavailable",
            )
        inside: list[str] = []
        errors: dict[str, str] = {}
        for target_id in targets:
            try:
                if target_fully_inside_region(
                    observed_world,
                    target_id=target_id,
                    region_id=region_id,
                    inset_margin_m=self._inset,
                ):
                    inside.append(target_id)
            except ValueError as error:
                errors[target_id] = str(error)
        satisfied = len(inside) == len(targets) and not errors
        return _result(
            request,
            GoalEvaluationStatus.SATISFIED if satisfied else GoalEvaluationStatus.FAILED,
            (
                "all target footprints are inside the goal region"
                if satisfied
                else "one or more target footprints are outside the goal region"
            ),
            observed={
                "region_id": region_id,
                "inside_target_ids": inside,
                "outside_target_ids": [target for target in targets if target not in inside],
                "geometry_errors": errors,
                "inset_margin_m": self._inset,
            },
        )


class SupportStabilityEvaluator:
    """Compare observed target heights with an execution-start reference."""

    def __init__(
        self,
        reference_positions_m: Mapping[str, Sequence[float]],
        *,
        maximum_vertical_error_m: float,
    ) -> None:
        if maximum_vertical_error_m < 0.0:
            raise ValueError("maximum vertical error must be non-negative")
        self._reference = {
            key: tuple(float(value) for value in position)
            for key, position in reference_positions_m.items()
        }
        self._maximum = maximum_vertical_error_m

    def evaluate(
        self,
        request: MotionPlanRequest,
        report: ExecutionReport,
        observed_world: WorldSnapshot | None,
    ) -> GoalEvaluation:
        del report
        if observed_world is None:
            return _result(
                request,
                GoalEvaluationStatus.UNKNOWN,
                "observed world is unavailable",
            )
        errors: dict[str, float] = {}
        missing: list[str] = []
        for target_id in request.task.target_ids:
            reference = self._reference.get(target_id)
            record = observed_world.objects.get(target_id)
            pose = record.get("pose") if isinstance(record, Mapping) else None
            position = pose.get("position_m") if isinstance(pose, Mapping) else None
            if reference is None or position is None or len(reference) != 3:
                missing.append(target_id)
                continue
            observed = np.asarray(position, dtype=float)
            if observed.shape != (3,):
                missing.append(target_id)
                continue
            errors[target_id] = abs(float(observed[2] - reference[2]))
        if missing:
            return _result(
                request,
                GoalEvaluationStatus.UNKNOWN,
                "support reference or observed pose is unavailable",
                observed={"missing_target_ids": missing, "vertical_errors_m": errors},
            )
        satisfied = all(value <= self._maximum for value in errors.values())
        return _result(
            request,
            GoalEvaluationStatus.SATISFIED if satisfied else GoalEvaluationStatus.FAILED,
            "target support heights are stable" if satisfied else "target support height changed",
            observed={
                "vertical_errors_m": errors,
                "maximum_vertical_error_m": self._maximum,
            },
        )


class GraspRetentionEvaluator:
    """Evaluate an explicit physical validation or final held-tool state."""

    def evaluate(
        self,
        request: MotionPlanRequest,
        report: ExecutionReport,
        observed_world: WorldSnapshot | None,
    ) -> GoalEvaluation:
        del observed_world
        validation = report.metadata.get("grasp_retention_validation")
        if isinstance(validation, Mapping):
            raw_status = str(validation.get("status") or "").upper()
            if raw_status in {"SUCCESS", "SATISFIED"}:
                status = GoalEvaluationStatus.SATISFIED
            elif raw_status in {"FAILED", "FAILURE"}:
                status = GoalEvaluationStatus.FAILED
            else:
                status = GoalEvaluationStatus.UNKNOWN
            return _result(
                request,
                status,
                "physical grasp-retention validation was evaluated",
                observed=dict(validation),
            )
        state = report.final_robot_state
        expected = request.task.tool
        if state is None or expected is None:
            return _result(
                request,
                GoalEvaluationStatus.UNKNOWN,
                "grasp-retention evidence is unavailable",
            )
        actual = state.held_tool_id or state.attached_object_id
        satisfied = actual == expected
        return _result(
            request,
            GoalEvaluationStatus.SATISFIED if satisfied else GoalEvaluationStatus.FAILED,
            "selected tool remains held" if satisfied else "selected tool is no longer held",
            observed={"expected_tool_id": expected, "held_tool_id": actual},
        )


class ToolClearanceEvaluator:
    """Evaluate minimum tool clearance recorded by a physical executor."""

    def __init__(self, *, minimum_clearance_m: float = 0.0) -> None:
        if minimum_clearance_m < 0.0:
            raise ValueError("minimum clearance must be non-negative")
        self._minimum = minimum_clearance_m

    def evaluate(
        self,
        request: MotionPlanRequest,
        report: ExecutionReport,
        observed_world: WorldSnapshot | None,
    ) -> GoalEvaluation:
        del observed_world
        validation = report.metadata.get("tool_clearance_validation")
        if isinstance(validation, Mapping) and "status" in validation:
            raw_status = str(validation.get("status") or "").upper()
            status = (
                GoalEvaluationStatus.SATISFIED
                if raw_status in {"SUCCESS", "SATISFIED"}
                else GoalEvaluationStatus.FAILED
                if raw_status in {"FAILED", "FAILURE"}
                else GoalEvaluationStatus.UNKNOWN
            )
            return _result(
                request,
                status,
                "tool-clearance validation was evaluated",
                observed=dict(validation),
            )
        value = report.metadata.get("minimum_tool_clearance_m")
        if not isinstance(value, (int, float)):
            return _result(
                request,
                GoalEvaluationStatus.UNKNOWN,
                "minimum tool clearance is unavailable",
            )
        satisfied = float(value) >= self._minimum
        return _result(
            request,
            GoalEvaluationStatus.SATISFIED if satisfied else GoalEvaluationStatus.FAILED,
            "tool clearance is satisfied" if satisfied else "tool clearance is violated",
            observed={
                "minimum_tool_clearance_m": float(value),
                "required_clearance_m": self._minimum,
            },
        )


class CompositeGoalEvaluator:
    """Aggregate independent predicates without treating UNKNOWN as success."""

    def __init__(self, evaluators: Sequence[GoalEvaluator]) -> None:
        if not evaluators:
            raise ValueError("at least one goal evaluator is required")
        self._evaluators = tuple(evaluators)

    def evaluate(
        self,
        request: MotionPlanRequest,
        report: ExecutionReport,
        observed_world: WorldSnapshot | None,
    ) -> GoalEvaluation:
        evaluations = [
            evaluator.evaluate(request, report, observed_world)
            for evaluator in self._evaluators
        ]
        if any(item.status is GoalEvaluationStatus.FAILED for item in evaluations):
            status = GoalEvaluationStatus.FAILED
        elif any(item.status is GoalEvaluationStatus.UNKNOWN for item in evaluations):
            status = GoalEvaluationStatus.UNKNOWN
        else:
            status = GoalEvaluationStatus.SATISFIED
        return _result(
            request,
            status,
            "; ".join(item.detail for item in evaluations),
            observed={"evaluations": [item.as_dict() for item in evaluations]},
        )


class TaskAwareGoalEvaluator:
    """Route grounded region/contact work to its physical success predicate.

    Resource transitions (pick, place, tool return, and EE exchange) keep the
    state-based evaluator. Ordinary contact or transport tasks with a grounded
    region require every target footprint to be inside that region.
    """

    def __init__(self, *, joint_tolerance_rad: float = 0.02) -> None:
        self._state = GroundedMotionGoalEvaluator(
            joint_tolerance_rad=joint_tolerance_rad
        )
        self._region = RegionContainmentEvaluator()

    def evaluate(
        self,
        request: MotionPlanRequest,
        report: ExecutionReport,
        observed_world: WorldSnapshot | None,
    ) -> GoalEvaluation:
        task = request.task
        is_resource_transition = (
            is_acquire_task(task)
            or is_release_task(task)
            or is_ee_exchange_task(task)
        )
        if (
            not is_resource_transition
            and task.goal.target_region_id is not None
            and task.goal.target_region_id in request.world.objects
            and bool(task.target_ids)
        ):
            return self._region.evaluate(request, report, observed_world)
        return self._state.evaluate(request, report, observed_world)


__all__ = [
    "CompositeGoalEvaluator",
    "GraspRetentionEvaluator",
    "RegionContainmentEvaluator",
    "SupportStabilityEvaluator",
    "TaskAwareGoalEvaluator",
    "ToolClearanceEvaluator",
]
