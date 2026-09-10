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
from tuj.m5_motion.push_to_region import (
    target_above_region,
    target_fully_inside_region,
)
from tuj.m5_motion.schema import ExecutionReport, MotionPlanRequest, WorldSnapshot
from tuj.m5_motion.task_semantics import (
    is_acquire_task,
    is_ee_exchange_task,
    is_release_task,
    task_operation,
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

    def __init__(
        self,
        *,
        inset_margin_m: float = 0.0,
        include_vertical: bool = False,
        require_interior_geometry: bool = False,
    ) -> None:
        if inset_margin_m < 0.0:
            raise ValueError("inset margin must be non-negative")
        self._inset = inset_margin_m
        self._include_vertical = include_vertical
        self._require_interior_geometry = require_interior_geometry

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
        if self._require_interior_geometry:
            region_record = observed_world.objects.get(region_id)
            packing_metadata = (
                region_record.get("packing_metadata")
                if isinstance(region_record, Mapping)
                else None
            )
            interior_dimensions = np.asarray(
                packing_metadata.get("interior_dimensions_m")
                if isinstance(packing_metadata, Mapping)
                else None,
                dtype=float,
            )
            interior_center = np.asarray(
                packing_metadata.get("interior_center_m")
                if isinstance(packing_metadata, Mapping)
                else None,
                dtype=float,
            )
            if not (
                interior_dimensions.shape == (3,)
                and interior_center.shape == (3,)
                and np.all(np.isfinite(interior_dimensions))
                and np.all(interior_dimensions > 0.0)
                and np.all(np.isfinite(interior_center))
            ):
                return _result(
                    request,
                    GoalEvaluationStatus.UNKNOWN,
                    "container interior geometry is unavailable",
                    observed={"region_id": region_id, "include_vertical": True},
                )
        inside: list[str] = []
        errors: dict[str, str] = {}
        outside: list[str] = []
        for target_id in targets:
            try:
                if target_fully_inside_region(
                    observed_world,
                    target_id=target_id,
                    region_id=region_id,
                    inset_margin_m=self._inset,
                    include_vertical=self._include_vertical,
                ):
                    inside.append(target_id)
                else:
                    outside.append(target_id)
            except ValueError as error:
                errors[target_id] = str(error)
        if outside:
            status = GoalEvaluationStatus.FAILED
            detail = "one or more target footprints are outside the goal region"
        elif errors:
            status = GoalEvaluationStatus.UNKNOWN
            detail = "region containment geometry is unavailable"
        else:
            status = GoalEvaluationStatus.SATISFIED
            detail = "all target footprints are inside the goal region"
        return _result(
            request,
            status,
            detail,
            observed={
                "region_id": region_id,
                "inside_target_ids": inside,
                "outside_target_ids": outside,
                "geometry_errors": errors,
                "inset_margin_m": self._inset,
                "include_vertical": self._include_vertical,
            },
        )


class AboveRegionEvaluator:
    """Require each transported target to be horizontally over a region."""

    def __init__(
        self,
        *,
        horizontal_tolerance_m: float = 0.0,
        vertical_tolerance_m: float = 0.01,
    ) -> None:
        if horizontal_tolerance_m < 0.0 or vertical_tolerance_m < 0.0:
            raise ValueError("above-region tolerances must be non-negative")
        self._horizontal = horizontal_tolerance_m
        self._vertical = vertical_tolerance_m

    def evaluate(
        self,
        request: MotionPlanRequest,
        report: ExecutionReport,
        observed_world: WorldSnapshot | None,
    ) -> GoalEvaluation:
        del report
        region_id = request.task.goal.target_region_id
        targets = list(request.task.target_ids)
        if not region_id or not targets or observed_world is None:
            return _result(
                request,
                GoalEvaluationStatus.UNKNOWN,
                "above-region evaluation requires observed targets and region",
            )
        above: list[str] = []
        errors: dict[str, str] = {}
        for target_id in targets:
            try:
                if target_above_region(
                    observed_world,
                    target_id=target_id,
                    region_id=region_id,
                    horizontal_tolerance_m=self._horizontal,
                    vertical_tolerance_m=self._vertical,
                ):
                    above.append(target_id)
            except ValueError as error:
                errors[target_id] = str(error)
        satisfied = len(above) == len(targets) and not errors
        return _result(
            request,
            GoalEvaluationStatus.SATISFIED if satisfied else GoalEvaluationStatus.FAILED,
            "all targets are above the goal region"
            if satisfied
            else "one or more targets are not above the goal region",
            observed={
                "region_id": region_id,
                "above_target_ids": above,
                "not_above_target_ids": [item for item in targets if item not in above],
                "geometry_errors": errors,
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
        expected = (
            request.task.goal.target_object_id
            or next(iter(request.task.target_ids), None)
            or request.task.tool
        )
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
        self._container_region = RegionContainmentEvaluator(
            include_vertical=True,
            require_interior_geometry=True,
        )
        self._above_region = AboveRegionEvaluator()
        self._grasp = GraspRetentionEvaluator()

    def evaluate(
        self,
        request: MotionPlanRequest,
        report: ExecutionReport,
        observed_world: WorldSnapshot | None,
    ) -> GoalEvaluation:
        task = request.task
        from .flatten_contact import is_flatten_contact, flattening_outcome
        if is_flatten_contact(task):
            if observed_world is None or len(task.target_ids) != 1:
                return _result(request, GoalEvaluationStatus.UNKNOWN, 'measured deformable state unavailable')
            record = observed_world.objects.get(task.target_ids[0], {})
            passed, evidence = flattening_outcome(record)
            return _result(request, GoalEvaluationStatus.SATISFIED if passed else GoalEvaluationStatus.FAILED,
                           'measured contact and residual flattening', observed=evidence)
        if is_acquire_task(task):
            from tuj.m5_motion.physical_grasp import uses_contact_friction

            if uses_contact_friction(request):
                return self._grasp.evaluate(request, report, observed_world)
        is_resource_transition = (
            is_acquire_task(task)
            or is_release_task(task)
            or is_ee_exchange_task(task)
        )
        operation = task_operation(task)
        region_record = request.world.objects.get(task.goal.target_region_id or "")
        packing_metadata = (
            region_record.get("packing_metadata")
            if isinstance(region_record, Mapping)
            else None
        )
        region_evaluator = (
            self._container_region
            if isinstance(packing_metadata, Mapping)
            and str(packing_metadata.get("kind", "")).upper() == "CONTAINER"
            else self._region
        )
        if (
            operation == "TRANSPORT"
            and task.goal.target_region_id is not None
            and task.goal.target_region_id in request.world.objects
            and bool(task.target_ids)
        ):
            held_target = request.world.robot_state.held_tool_id
            if held_target is not None and held_target in task.target_ids:
                return CompositeGoalEvaluator(
                    (self._above_region, self._grasp)
                ).evaluate(request, report, observed_world)
            return self._above_region.evaluate(request, report, observed_world)
        if (
            is_release_task(task)
            and task.goal.target_region_id is not None
            and task.goal.target_region_id in request.world.objects
            and bool(task.target_ids)
        ):
            target = task.goal.target_object_id or task.target_ids[0]
            state = report.final_robot_state
            if state is None:
                return _result(
                    request,
                    GoalEvaluationStatus.UNKNOWN,
                    "release execution did not produce a final robot state",
                )
            if state.attached_object_id == target:
                return _result(
                    request,
                    GoalEvaluationStatus.FAILED,
                    f"placed object {target!r} is still attached",
                    observed={"attached_object_id": state.attached_object_id},
                )
            return region_evaluator.evaluate(request, report, observed_world)
        if (
            not is_resource_transition
            and task.goal.target_region_id is not None
            and task.goal.target_region_id in request.world.objects
            and bool(task.target_ids)
            and (
                task.contact is not None
                or operation
                in {
                    "PUSH",
                    "PUSH_TO_REGION",
                    "SWEEP",
                    "REGROUP",
                    "CLEANUP",
                    "TOOL_ACT",
                }
            )
        ):
            return region_evaluator.evaluate(request, report, observed_world)
        return self._state.evaluate(request, report, observed_world)


__all__ = [
    "AboveRegionEvaluator",
    "CompositeGoalEvaluator",
    "GraspRetentionEvaluator",
    "RegionContainmentEvaluator",
    "SupportStabilityEvaluator",
    "TaskAwareGoalEvaluator",
    "ToolClearanceEvaluator",
]
