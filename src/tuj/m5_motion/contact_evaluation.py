"""Composable execution predicates for region and contact manipulation goals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np

from tuj.m5_motion.execution import (
    GoalEvaluation,
    GoalEvaluationStatus,
    GoalEvaluator,
    GroundedMotionGoalEvaluator,
)
from tuj.m5_motion.push_to_region import (
    PLACE_SETTLE_FOOTPRINT_TOLERANCE_M,
    target_above_region,
    target_fully_inside_region,
    target_resting_on_region,
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
        contact_tolerance_m: float | None = None,
    ) -> None:
        if inset_margin_m < 0.0:
            raise ValueError("inset margin must be non-negative")
        if contact_tolerance_m is not None and contact_tolerance_m < 0.0:
            raise ValueError("contact tolerance must be non-negative")
        self._inset = inset_margin_m
        self._include_vertical = include_vertical
        self._require_interior_geometry = require_interior_geometry
        self._contact_tolerance_m = contact_tolerance_m

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
                    contact_tolerance_m=self._contact_tolerance_m,
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
                "contact_tolerance_m": self._contact_tolerance_m,
            },
        )


def _world_aabb(obj: Any):
    """Axis-aligned world centre and half-extents for an object snapshot.

    Uses the observed collision points (body frame; equal to world axes for the
    yaw-0 boxes in this scene) for the extents and the pose position (plus the
    points' own centre offset) for the centre.  Falls back to dimensions_m.
    Returns (None, None) when geometry is unavailable.
    """
    if not isinstance(obj, Mapping):
        return None, None
    pose = obj.get("pose") if isinstance(obj.get("pose"), Mapping) else {}
    pos = np.asarray(pose.get("position_m", obj.get("position_m", [0.0, 0.0, 0.0])), dtype=float)
    pts = obj.get("collision_points_m")
    if pts is not None:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] >= 2 and np.isfinite(pts).all():
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            return pos + (lo + hi) / 2.0, (hi - lo) / 2.0
    dims = obj.get("dimensions_m")
    if dims is not None:
        dims = np.asarray(dims, dtype=float)
        if dims.shape == (3,) and np.all(np.isfinite(dims)):
            return pos, dims / 2.0
    return None, None


class ToolExtractionEvaluator:
    """Success when the extracted target has left the gap it was wedged in.

    The extract goal names a flanking fixture (one wall of the gap) as its
    region, but extraction does not place the card INTO that fixture -- it pulls
    the card OUT of the channel between the flankers toward the open end.  So the
    criterion is: the target, initially trapped inside the flanker's span along
    the channel (the horizontal axis on which it started inside the flanker),
    has moved fully outside that span (its footprint clears the flanker face) and
    actually travelled a minimum distance (not merely jittered).
    """

    def __init__(self, *, min_travel_m: float = 0.05, exit_margin_m: float = 0.0) -> None:
        self._min_travel = min_travel_m
        self._exit_margin = exit_margin_m

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
                "extraction evaluation requires targets, a gap region, and an observed world",
            )
        region = observed_world.objects.get(region_id)
        if not isinstance(region, Mapping):
            region = request.world.objects.get(region_id)
        rc, rh = _world_aabb(region)
        if rc is None:
            return _result(
                request,
                GoalEvaluationStatus.UNKNOWN,
                "extraction gap-region geometry is unavailable",
                observed={"region_id": region_id},
            )
        extracted: list[str] = []
        remaining: list[str] = []
        errors: dict[str, str] = {}
        details: dict[str, Any] = {}
        for target_id in targets:
            ic, ih = _world_aabb(request.world.objects.get(target_id))
            fc, fh = _world_aabb(observed_world.objects.get(target_id))
            if ic is None or fc is None:
                errors[target_id] = "target geometry unavailable"
                continue
            # Channel axis: the horizontal axis on which the target STARTED inside
            # the flanker's span (that is the axis along which the gap traps it).
            inside = [ax for ax in (0, 1) if abs(ic[ax] - rc[ax]) <= rh[ax]]
            axis = max(inside, key=lambda a: rh[a]) if inside else (0 if rh[0] >= rh[1] else 1)
            travel = float(abs(fc[axis] - ic[axis]))
            cleared = (
                (fc[axis] + fh[axis]) < (rc[axis] - rh[axis] - self._exit_margin)
                or (fc[axis] - fh[axis]) > (rc[axis] + rh[axis] + self._exit_margin)
            )
            details[target_id] = {"axis": axis, "travel_m": travel, "cleared_span": bool(cleared)}
            if cleared and travel >= self._min_travel:
                extracted.append(target_id)
            else:
                remaining.append(target_id)
        if remaining:
            status = GoalEvaluationStatus.FAILED
            detail = "one or more targets are still inside the gap"
        elif errors:
            status = GoalEvaluationStatus.UNKNOWN
            detail = "extraction target geometry is unavailable"
        else:
            status = GoalEvaluationStatus.SATISFIED
            detail = "all targets have been extracted from the gap"
        return _result(
            request,
            status,
            detail,
            observed={
                "region_id": region_id,
                "extracted_target_ids": extracted,
                "remaining_target_ids": remaining,
                "geometry_errors": errors,
                "per_target": details,
                "min_travel_m": self._min_travel,
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


class StackPlacementEvaluator:
    """Require each placed target to rest stably on its base object.

    Stacking cannot use ``RegionContainmentEvaluator``.  Containment asks the
    placed footprint to fit inside the base footprint, which a sandwich can
    never satisfy: the top slice of bread (113.1 x 113.3 mm) goes onto the
    filling (turkey is 80.0 x 50.0 mm), and reordering the layers does not
    help because some layer is always wider than the one beneath it.  The
    physical success condition for a stack is support, not containment, so the
    target's centre must lie over the base and the target must rest on it.
    """

    def __init__(
        self,
        *,
        horizontal_tolerance_m: float = 0.0,
        maximum_gap_m: float = .005,
        maximum_penetration_m: float = .005,
    ) -> None:
        if horizontal_tolerance_m < 0.0:
            raise ValueError("horizontal tolerance must be non-negative")
        if maximum_gap_m < 0.0 or maximum_penetration_m < 0.0:
            raise ValueError("resting tolerances must be non-negative")
        self._horizontal = horizontal_tolerance_m
        self._gap = maximum_gap_m
        self._penetration = maximum_penetration_m

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
                "stack placement requires observed targets and a base region",
            )
        resting: list[str] = []
        errors: dict[str, str] = {}
        for target_id in targets:
            try:
                if target_resting_on_region(
                    observed_world,
                    target_id=target_id,
                    region_id=region_id,
                    horizontal_tolerance_m=self._horizontal,
                    maximum_gap_m=self._gap,
                    maximum_penetration_m=self._penetration,
                ):
                    resting.append(target_id)
            except ValueError as error:
                errors[target_id] = str(error)
        satisfied = len(resting) == len(targets) and not errors
        return _result(
            request,
            GoalEvaluationStatus.SATISFIED if satisfied else GoalEvaluationStatus.FAILED,
            "all targets rest stably on the base object"
            if satisfied
            else "one or more targets do not rest stably on the base object",
            observed={
                "region_id": region_id,
                "resting_target_ids": resting,
                "unsupported_target_ids": [
                    item for item in targets if item not in resting
                ],
                "geometry_errors": errors,
                "maximum_gap_m": self._gap,
                "maximum_penetration_m": self._penetration,
                "predicate": "STACK_SUPPORT",
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
        # Open-plate place/release: tolerate post-release settle rim graze.
        self._place_region = RegionContainmentEvaluator(
            contact_tolerance_m=PLACE_SETTLE_FOOTPRINT_TOLERANCE_M,
        )
        self._container_region = RegionContainmentEvaluator(
            include_vertical=True,
            require_interior_geometry=True,
        )
        self._above_region = AboveRegionEvaluator()
        self._stack_placement = StackPlacementEvaluator()
        self._grasp = GraspRetentionEvaluator()
        self._extraction = ToolExtractionEvaluator()

    def evaluate(
        self,
        request: MotionPlanRequest,
        report: ExecutionReport,
        observed_world: WorldSnapshot | None,
    ) -> GoalEvaluation:
        task = request.task
        # An 'extract' contact pulls the target OUT of the gap; its named region
        # is a flanking wall of the gap, not a container to place the target in,
        # so the containment check never fits.  Judge it by whether the target
        # left the gap instead.
        contact = getattr(task, "contact", None)
        if contact is not None and str(getattr(contact, "primitive", "")).lower() == "extract":
            return self._extraction.evaluate(request, report, observed_world)
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
            else (
                self._place_region
                if is_release_task(task)
                else self._region
            )
        )
        evaluation_world = observed_world
        if (
            (operation == "TRANSPORT" or is_release_task(task))
            and task.metadata.get("conceptual_tool_home_goal") is True
            and isinstance(region_record, Mapping)
            and isinstance(region_record.get("metadata"), Mapping)
            and region_record["metadata"].get("conceptual_tool_home") is True
            and observed_world is not None
            and task.goal.target_region_id not in observed_world.objects
        ):
            # ``tool_rest`` is a semantic destination built from the measured
            # pre-grasp pose.  It must stay out of the simulator collision
            # world, but the physical target pose is still observed there.
            # Add only that reference geometry to a private evaluation copy so
            # the ordinary transport/release predicate can verify the real body.
            evaluation_world = observed_world.model_copy(deep=True)
            evaluation_world.objects[task.goal.target_region_id] = deepcopy(
                region_record
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
                ).evaluate(request, report, evaluation_world)
            return self._above_region.evaluate(request, report, evaluation_world)
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
            from .container_surface import is_container_surface_task, surface_report_result
            if is_container_surface_task(task, request.world.objects):
                if evaluation_world is None:
                    return _result(request, GoalEvaluationStatus.UNKNOWN, 'observed cover state unavailable')
                try:
                    evidence = surface_report_result(request, report, evaluation_world)
                except (ValueError, KeyError, TypeError) as error:
                    return _result(request, GoalEvaluationStatus.UNKNOWN,
                                   'container cover evidence unavailable', observed={'error_type': type(error).__name__})
                return _result(request,
                    GoalEvaluationStatus.SATISFIED if evidence['succeeded'] else GoalEvaluationStatus.FAILED,
                    'container opening coverage, physical support and packed contents', observed=evidence)
            if operation == "PLACE_ON" and region_evaluator is self._region:
                # place_on names a base object to stack onto, so support is the
                # success condition rather than containment.  Containers keep
                # the containment predicate through _container_region above.
                return self._stack_placement.evaluate(
                    request, report, evaluation_world
                )
            return region_evaluator.evaluate(request, report, evaluation_world)
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
    "StackPlacementEvaluator",
    "SupportStabilityEvaluator",
    "TaskAwareGoalEvaluator",
    "ToolClearanceEvaluator",
    "ToolExtractionEvaluator",
]
