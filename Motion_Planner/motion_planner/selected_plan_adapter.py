"""Convert Task Planner's selected symbolic order into grounded motion requests."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

from task_planner.serialization import CandidateAssignment, PlanStep, SelectedPlan

from motion_planner.schema import (
    ArtifactProvenance,
    GoalType,
    ModuleName,
    MotionConstraints,
    MotionGoal,
    MotionPlanRequest,
    MotionTask,
    PlannerOptions,
    Pose,
    WorldSnapshot,
)


class SelectedPlanAdapterError(ValueError):
    """SelectedPlan lacks data required for a fail-closed motion request."""


WorldSnapshotSource: TypeAlias = (
    WorldSnapshot
    | Mapping[str, WorldSnapshot]
    | Callable[[str, int], WorldSnapshot]
)
ConstraintSource: TypeAlias = (
    MotionConstraints
    | Mapping[str, MotionConstraints]
    | Callable[[str, int], MotionConstraints]
)
OptionSource: TypeAlias = (
    PlannerOptions
    | Mapping[str, PlannerOptions]
    | Callable[[str, int], PlannerOptions]
)


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


def _float_sequence(value: Any, length: int) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != length:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    return result


def _canonical_frame(frame: Any, world: WorldSnapshot) -> str:
    value = str(frame or "world")
    if value == "world" or ":" in value:
        return value
    if value in world.objects:
        return f"object:{value}"
    if value in world.rack:
        return f"rack:{value}"
    return value


def _yaw_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    half = yaw_rad * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _pose_from_value(value: Any, world: WorldSnapshot) -> Pose | None:
    if value is None:
        return None
    if isinstance(value, Pose):
        return value
    if not isinstance(value, Mapping):
        return None

    for nested_key in ("pose", "target_pose", "dock_pose"):
        nested = value.get(nested_key)
        if isinstance(nested, Mapping):
            nested_pose = _pose_from_value(nested, world)
            if nested_pose is not None:
                return nested_pose

    position = _float_sequence(value.get("position_m"), 3)
    if position is None:
        millimetres = _float_sequence(value.get("position_mm"), 3)
        if millimetres is not None:
            position = tuple(item / 1000.0 for item in millimetres)
    if position is None:
        position = _float_sequence(
            value.get("position", value.get("xyz", value.get("translation"))), 3
        )
    if position is None and {"x", "y", "z"} <= set(value):
        position = _float_sequence([value["x"], value["y"], value["z"]], 3)
    if position is None:
        return None

    orientation = _float_sequence(
        value.get("orientation_xyzw", value.get("quaternion_xyzw")), 4
    )
    if orientation is None and "yaw_deg" in value:
        orientation = _yaw_quaternion(math.radians(float(value["yaw_deg"])))
    if orientation is None and "yaw_rad" in value:
        orientation = _yaw_quaternion(float(value["yaw_rad"]))
    if orientation is None:
        orientation = (0.0, 0.0, 0.0, 1.0)

    return Pose(
        frame_id=_canonical_frame(
            value.get("frame_id", value.get("frame", "world")), world
        ),
        position_m=position,
        orientation_xyzw=orientation,
    )


def _object_pose(world: WorldSnapshot, target_ids: Sequence[str]) -> Pose | None:
    for target_id in target_ids:
        record = world.objects.get(target_id)
        pose = _pose_from_value(record, world)
        if pose is not None:
            return pose
    return None


def _distance_m(parameters: Mapping[str, Any], name: str) -> float | None:
    if name in parameters:
        return float(parameters[name])
    millimetres = f"{name.removesuffix('_m')}_mm"
    if millimetres in parameters:
        return float(parameters[millimetres]) / 1000.0
    return None


def _steps_for_subgoal(selected: SelectedPlan, subgoal_id: str) -> list[PlanStep]:
    return sorted(
        [step for step in selected.steps if step.subgoal_id == subgoal_id],
        key=lambda step: step.step_index,
    )


def _execution_step(steps: Sequence[PlanStep], subgoal_id: str) -> PlanStep:
    matches = [step for step in steps if step.kind == "subgoal"]
    if len(matches) != 1:
        raise SelectedPlanAdapterError(
            f"subgoal {subgoal_id!r} requires exactly one executable step"
        )
    return matches[0]


def _merged_action_parameters(
    assignment: CandidateAssignment, execution: PlanStep
) -> dict[str, Any]:
    result = dict(assignment.action_parameters)
    nested = execution.parameters.get("action_parameters")
    if isinstance(nested, Mapping):
        result.update(nested)
    return result


def _action_type(
    assignment: CandidateAssignment, execution: PlanStep
) -> str:
    raw_action = assignment.action_type or execution.parameters.get("action_type")
    if not raw_action:
        raise SelectedPlanAdapterError(
            f"candidate {assignment.candidate_id!r} has no action_type; "
            "regenerate SelectedPlan with the current Task Planner schema"
        )
    return str(raw_action)


def _goal(
    assignment: CandidateAssignment,
    execution: PlanStep,
    world: WorldSnapshot,
    target_ids: Sequence[str],
) -> MotionGoal:
    parameters = _merged_action_parameters(assignment, execution)
    action_type = _action_type(assignment, execution)

    joint_target = _float_sequence(
        parameters.get("target_joint_positions_rad"),
        len(world.robot_state.joint_names),
    )
    target_pose = _pose_from_value(parameters.get("target_pose"), world)
    if target_pose is None and assignment.grasp is not None:
        target_pose = _pose_from_value(assignment.grasp.pose, world)
    if target_pose is None:
        target_pose = _object_pose(world, target_ids)

    normalized = action_type.strip().lower()
    explicit_goal = parameters.get("goal_type")
    if explicit_goal is not None:
        try:
            goal_type = GoalType(str(explicit_goal).upper())
        except ValueError as error:
            raise SelectedPlanAdapterError(
                f"unsupported goal_type {explicit_goal!r} for {assignment.subgoal_id!r}"
            ) from error
    elif joint_target is not None:
        goal_type = GoalType.JOINT
    elif any(token in normalized for token in ("pick", "grasp", "acquire")):
        goal_type = GoalType.PICK
    elif any(token in normalized for token in ("place", "position", "release")):
        goal_type = GoalType.PLACE
    elif "dock" in normalized:
        goal_type = GoalType.DOCK
    else:
        goal_type = GoalType.POSE

    if goal_type in {GoalType.POSE, GoalType.PLACE, GoalType.DOCK} and target_pose is None:
        raise SelectedPlanAdapterError(
            f"subgoal {assignment.subgoal_id!r} requires a target pose but none was "
            "preserved in the selected candidate, grasp, or world snapshot"
        )
    if goal_type is GoalType.PICK and assignment.grasp is None:
        raise SelectedPlanAdapterError(
            f"PICK subgoal {assignment.subgoal_id!r} requires a structured grasp"
        )

    approach = _float_sequence(
        parameters.get("approach_direction", parameters.get("approach_dir")), 3
    )
    return MotionGoal(
        goal_type=goal_type,
        target_pose=target_pose,
        target_joint_positions_rad=(list(joint_target) if joint_target else None),
        target_object_id=(target_ids[0] if target_ids else None),
        target_region_id=(
            assignment.goal_region_id
            or parameters.get("target_region_id")
            or parameters.get("target_region")
        ),
        approach_direction=approach,
        approach_distance_m=_distance_m(parameters, "approach_distance_m"),
        retreat_distance_m=_distance_m(parameters, "retreat_distance_m"),
    )


def _source_value(
    source: Any,
    subgoal_id: str,
    index: int,
    *,
    label: str,
    total: int,
    require_per_subgoal: bool = False,
) -> Any:
    if callable(source):
        return source(subgoal_id, index)
    if isinstance(source, Mapping):
        if subgoal_id not in source:
            raise SelectedPlanAdapterError(
                f"{label} is missing subgoal {subgoal_id!r}"
            )
        return source[subgoal_id]
    if require_per_subgoal and total != 1:
        raise SelectedPlanAdapterError(
            f"a distinct {label} is required for each of {total} ordered subgoals"
        )
    return source


class SelectedPlanMotionRequestAdapter:
    """Build one MotionPlanRequest per ordered Task Planner subgoal.

    Each request must be bound to the world snapshot at the start of that
    subgoal.  Passing a single snapshot for a multi-subgoal plan is rejected so
    stale joint/object state cannot silently contaminate later paths.
    """

    def convert(
        self,
        selected: SelectedPlan,
        *,
        worlds: WorldSnapshotSource,
        constraints: ConstraintSource,
        options: OptionSource | None = None,
        selected_plan_artifact_id: str | None = None,
        planner_invocation_id: str = "task-planner:selected-plan",
    ) -> tuple[MotionPlanRequest, ...]:
        order = list(selected.subgoal_order)
        if not order:
            raise SelectedPlanAdapterError("SelectedPlan has no ordered subgoals")
        if len(order) != len(set(order)):
            raise SelectedPlanAdapterError("SelectedPlan repeats a subgoal_id")

        assignments: dict[str, CandidateAssignment] = {}
        for assignment in selected.candidate_assignments:
            if assignment.subgoal_id in assignments:
                raise SelectedPlanAdapterError(
                    f"duplicate assignment for subgoal {assignment.subgoal_id!r}"
                )
            assignments[assignment.subgoal_id] = assignment
        missing = [subgoal_id for subgoal_id in order if subgoal_id not in assignments]
        if missing:
            raise SelectedPlanAdapterError(
                f"SelectedPlan is missing candidate assignments for {missing}"
            )

        selected_payload = selected.model_dump(mode="json")
        selected_hash = _digest(selected_payload)
        source_artifact_id = selected_plan_artifact_id or (
            f"selected-plan-artifact:{selected_hash[:24]}"
        )
        requests: list[MotionPlanRequest] = []
        total = len(order)
        default_options = options or PlannerOptions()

        for index, subgoal_id in enumerate(order):
            assignment = assignments[subgoal_id]
            world = _source_value(
                worlds,
                subgoal_id,
                index,
                label="WorldSnapshot",
                total=total,
                require_per_subgoal=True,
            )
            selected_constraints = _source_value(
                constraints,
                subgoal_id,
                index,
                label="MotionConstraints",
                total=total,
            )
            selected_options = _source_value(
                default_options,
                subgoal_id,
                index,
                label="PlannerOptions",
                total=total,
            )
            if not isinstance(world, WorldSnapshot):
                raise SelectedPlanAdapterError(
                    f"world provider returned {type(world).__name__}, not WorldSnapshot"
                )
            if not isinstance(selected_constraints, MotionConstraints):
                raise SelectedPlanAdapterError(
                    "constraints provider did not return MotionConstraints"
                )
            if not isinstance(selected_options, PlannerOptions):
                raise SelectedPlanAdapterError(
                    "options provider did not return PlannerOptions"
                )

            steps = _steps_for_subgoal(selected, subgoal_id)
            execution = _execution_step(steps, subgoal_id)
            if execution.candidate_id != assignment.candidate_id:
                raise SelectedPlanAdapterError(
                    f"step candidate {execution.candidate_id!r} does not match "
                    f"assignment {assignment.candidate_id!r}"
                )
            target_ids = list(assignment.target_ids)
            if not target_ids:
                parameters = _merged_action_parameters(assignment, execution)
                target = parameters.get("target_object_id")
                if target:
                    target_ids = [str(target)]
            goal = _goal(assignment, execution, world, target_ids)
            action_type = _action_type(assignment, execution)
            allowed_touch = _merged_action_parameters(
                assignment, execution
            ).get("allowed_touch_objects", [])
            if goal.goal_type is GoalType.PICK and not allowed_touch:
                allowed_touch = list(target_ids)
            if not isinstance(allowed_touch, list):
                raise SelectedPlanAdapterError(
                    "allowed_touch_objects must be a list when supplied"
                )

            request_digest = _digest(
                {
                    "selected_plan_hash": selected_hash,
                    "subgoal_id": subgoal_id,
                    "candidate_id": assignment.candidate_id,
                    "world": world.model_dump(mode="json"),
                    "constraints": selected_constraints.model_dump(mode="json"),
                    "options": selected_options.model_dump(mode="json"),
                }
            )
            request_id = f"motion-request:{index}:{request_digest[:20]}"
            requests.append(
                MotionPlanRequest(
                    request_id=request_id,
                    provenance=ArtifactProvenance(
                        artifact_id=f"motion-request-artifact:{request_digest[:24]}",
                        artifact_type="MotionPlanRequest",
                        produced_by=ModuleName.TASK_PLANNER,
                        invocation_id=planner_invocation_id,
                        input_artifact_ids=[source_artifact_id],
                        metadata={
                            "selected_plan_hash": selected_hash,
                            "selected_plan_index": index,
                            "candidate_id": assignment.candidate_id,
                        },
                    ),
                    world=world.model_copy(deep=True),
                    task=MotionTask(
                        task_id=f"{subgoal_id}:{assignment.candidate_id}",
                        subgoal_id=subgoal_id,
                        action_type=action_type,
                        ee=assignment.ee,
                        tool=assignment.tool,
                        target_ids=target_ids,
                        grasp=assignment.grasp,
                        goal=goal,
                        allowed_touch_objects=[str(item) for item in allowed_touch],
                        metadata={
                            "selected_plan_index": index,
                            "candidate_id": assignment.candidate_id,
                            "description": assignment.description,
                            "task_planner_steps": [
                                step.model_dump(mode="json") for step in steps
                            ],
                            "action_parameters": _merged_action_parameters(
                                assignment, execution
                            ),
                        },
                    ),
                    constraints=selected_constraints.model_copy(deep=True),
                    options=selected_options.model_copy(deep=True),
                )
            )
        return tuple(requests)


def selected_plan_to_motion_requests(
    selected: SelectedPlan,
    *,
    worlds: WorldSnapshotSource,
    constraints: ConstraintSource,
    options: OptionSource | None = None,
    selected_plan_artifact_id: str | None = None,
    planner_invocation_id: str = "task-planner:selected-plan",
) -> tuple[MotionPlanRequest, ...]:
    """Convenience entry point for the standard SelectedPlan conversion."""

    return SelectedPlanMotionRequestAdapter().convert(
        selected,
        worlds=worlds,
        constraints=constraints,
        options=options,
        selected_plan_artifact_id=selected_plan_artifact_id,
        planner_invocation_id=planner_invocation_id,
    )


__all__ = [
    "ConstraintSource",
    "OptionSource",
    "SelectedPlanAdapterError",
    "SelectedPlanMotionRequestAdapter",
    "WorldSnapshotSource",
    "selected_plan_to_motion_requests",
]
