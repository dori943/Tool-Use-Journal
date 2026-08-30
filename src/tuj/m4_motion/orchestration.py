"""Sequential SelectedPlan planning with predicted-state handoff and storage."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tuj.m3_taskplanner.serialization import PlanStep, SelectedPlan

from tuj.m4_motion.schema import (
    AttachedObjectTransform,
    ArtifactProvenance,
    GoalType,
    ModuleName,
    MotionConstraints,
    MotionGoal,
    MotionPlan,
    MotionPlanRequest,
    MotionTask,
    PlannerOptions,
    SceneRef,
    WorldSnapshot,
)
from tuj.m4_motion.selected_plan_adapter import (
    ConstraintSource,
    OptionSource,
    SelectedPlanAdapterError,
    SelectedPlanMotionRequestAdapter,
)


class MotionRequestPlanner(Protocol):
    def __call__(self, request: MotionPlanRequest) -> Any: ...


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


class MotionPlanStore:
    """Atomically persist finalized plans and one ordered manifest."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def _atomic_json(path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(path)

    def save_plan(self, plan: MotionPlan, *, index: int) -> Path:
        path = self.root / "plans" / (
            f"{index:04d}-{_safe_name(plan.plan_id)}.json"
        )
        self._atomic_json(path, plan.model_dump_json(indent=2))
        return path.resolve()

    def save_manifest(
        self,
        *,
        selected_plan_hash: str,
        requests: Sequence[MotionPlanRequest],
        plans: Sequence[MotionPlan],
        plan_paths: Sequence[Path],
        final_world: WorldSnapshot,
    ) -> Path:
        manifest = {
            "manifest_version": "1.0.0",
            "selected_plan_hash": selected_plan_hash,
            "request_ids": [request.request_id for request in requests],
            "plan_ids": [plan.plan_id for plan in plans],
            "plan_files": [str(path) for path in plan_paths],
            "final_scene_signature": final_world.scene.signature,
            "final_robot_state": final_world.robot_state.model_dump(mode="json"),
        }
        path = self.root / "motion-plan-manifest.json"
        self._atomic_json(path, json.dumps(manifest, ensure_ascii=False, indent=2))
        return path.resolve()


@dataclass(frozen=True, slots=True)
class SelectedPlanPlanningResult:
    requests: tuple[MotionPlanRequest, ...]
    plans: tuple[MotionPlan, ...]
    final_world: WorldSnapshot
    plan_paths: tuple[Path, ...] = ()
    manifest_path: Path | None = None


def _source_value(
    source: Any,
    subgoal_id: str,
    index: int,
    expected_type: type,
    label: str,
) -> Any:
    if callable(source):
        value = source(subgoal_id, index)
    elif isinstance(source, Mapping):
        if subgoal_id not in source:
            raise SelectedPlanAdapterError(
                f"{label} is missing subgoal {subgoal_id!r}"
            )
        value = source[subgoal_id]
    else:
        value = source
    if not isinstance(value, expected_type):
        raise SelectedPlanAdapterError(
            f"{label} provider returned {type(value).__name__}"
        )
    return value


def _single_subgoal_plan(selected: SelectedPlan, subgoal_id: str) -> SelectedPlan:
    assignment = [
        item
        for item in selected.candidate_assignments
        if item.subgoal_id == subgoal_id
    ]
    if len(assignment) != 1:
        raise SelectedPlanAdapterError(
            f"subgoal {subgoal_id!r} requires exactly one assignment"
        )
    steps = [step for step in selected.steps if step.subgoal_id == subgoal_id]
    payload = selected.model_dump(mode="python")
    payload["subgoal_order"] = [subgoal_id]
    payload["candidate_assignments"] = assignment
    payload["steps"] = steps
    return SelectedPlan.model_validate(payload)


def _transition_request(
    *,
    parent_subgoal_id: str,
    transition_index: int,
    action_type: str,
    world: WorldSnapshot,
    constraints: MotionConstraints,
    options: PlannerOptions,
    ee: str,
    target_ids: list[str],
    goal: MotionGoal,
    metadata: dict[str, Any],
    selected_plan_artifact_id: str,
) -> MotionPlanRequest:
    identity = _digest(
        {
            "parent_subgoal_id": parent_subgoal_id,
            "transition_index": transition_index,
            "action_type": action_type,
            "world": world.model_dump(mode="json"),
            "metadata": metadata,
            "constraints": constraints.model_dump(mode="json"),
            "options": options.model_dump(mode="json"),
        }
    )
    request_id = f"motion-request:transition:{identity[:20]}"
    return MotionPlanRequest(
        request_id=request_id,
        provenance=ArtifactProvenance(
            artifact_id=f"motion-request-artifact:{identity[:24]}",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="task-planner:selected-plan:transition",
            input_artifact_ids=[selected_plan_artifact_id],
            metadata={"parent_subgoal_id": parent_subgoal_id},
        ),
        world=world.model_copy(deep=True),
        task=MotionTask(
            task_id=f"{parent_subgoal_id}:transition:{transition_index}",
            subgoal_id=f"{parent_subgoal_id}:transition:{transition_index}",
            action_type=action_type,
            ee=ee,
            target_ids=target_ids,
            goal=goal,
            metadata=metadata,
        ),
        constraints=constraints.model_copy(deep=True),
        options=options.model_copy(deep=True),
    )


def _resource_transition_requests(
    *,
    parent_subgoal_id: str,
    steps: Sequence[PlanStep],
    world: WorldSnapshot,
    constraints: MotionConstraints,
    options: PlannerOptions,
    selected_plan_artifact_id: str,
    fallback_ee: str,
) -> list[MotionPlanRequest]:
    requests: list[MotionPlanRequest] = []
    transition_steps = [step for step in steps if step.kind == "transition"]
    detach = next(
        (step for step in transition_steps if step.action == "DETACH_EE"),
        None,
    )
    attach = next(
        (step for step in transition_steps if step.action == "ATTACH_EE"),
        None,
    )
    terminal_restore = next(
        (
            step
            for step in transition_steps
            if step.action == "TERMINAL_RESTORE_EE"
        ),
        None,
    )
    if terminal_restore is not None:
        from_ee = str(terminal_restore.parameters.get("from") or "")
        to_ee = str(terminal_restore.parameters.get("to") or "")
        exchange_steps = [terminal_restore]
    elif detach is not None and attach is not None:
        from_ee = str(detach.parameters.get("ee") or "")
        to_ee = str(attach.parameters.get("ee") or "")
        exchange_steps = [detach, attach]
    else:
        from_ee = to_ee = ""
        exchange_steps = []
    if exchange_steps:
        if not from_ee or not to_ee:
            raise SelectedPlanAdapterError(
                f"EE exchange before {parent_subgoal_id!r} lacks from/to EE"
            )
        requests.append(
            _transition_request(
                parent_subgoal_id=parent_subgoal_id,
                transition_index=len(requests),
                action_type="EE_EXCHANGE",
                world=world,
                constraints=constraints,
                options=options,
                ee=to_ee,
                target_ids=[from_ee, to_ee],
                goal=MotionGoal(goal_type=GoalType.EE_EXCHANGE),
                metadata={
                    "from_ee": from_ee,
                    "to_ee": to_ee,
                    "task_planner_steps": [
                        step.model_dump(mode="json") for step in exchange_steps
                    ],
                },
                selected_plan_artifact_id=selected_plan_artifact_id,
            )
        )

    tool_actions = {
        "RETURN_TOOL",
        "PICK_TOOL",
        "TERMINAL_RETURN_TOOL",
    }
    for step in transition_steps:
        if step.action not in tool_actions:
            continue
        tool_id = step.parameters.get("tool")
        if not tool_id:
            raise SelectedPlanAdapterError(
                f"{step.action} before {parent_subgoal_id!r} lacks tool"
            )
        requests.append(
            _transition_request(
                parent_subgoal_id=parent_subgoal_id,
                transition_index=len(requests),
                action_type=step.action,
                world=world,
                constraints=constraints,
                options=options,
                ee=to_ee or fallback_ee,
                target_ids=[str(tool_id)],
                goal=MotionGoal(
                    goal_type=GoalType.TOOL_CHANGE,
                    target_object_id=str(tool_id),
                ),
                metadata={
                    "operation": step.action,
                    "tool_id": str(tool_id),
                    "task_planner_steps": [step.model_dump(mode="json")],
                },
                selected_plan_artifact_id=selected_plan_artifact_id,
            )
        )
    return requests


def _unwrap_plan(value: Any, request: MotionPlanRequest) -> MotionPlan:
    plan = value if isinstance(value, MotionPlan) else getattr(value, "plan", None)
    if not isinstance(plan, MotionPlan):
        raise TypeError("motion request planner must return MotionPlan or .plan")
    if plan.request_id != request.request_id:
        raise ValueError("planned MotionPlan request_id does not match its request")
    if plan.scene_signature != request.world.scene.signature:
        raise ValueError("planned MotionPlan scene_signature does not match its request")
    return plan


def _predicted_world(
    world: WorldSnapshot,
    request: MotionPlanRequest,
    plan: MotionPlan,
    *,
    completed_subgoal: str | None,
) -> WorldSnapshot:
    result = world.model_copy(deep=True)
    result.robot_state = plan.expected_final_state.model_copy(deep=True)
    attached_object_id = result.robot_state.attached_object_id
    if attached_object_id is None:
        result.metadata["attached_object_transforms"] = {}
    else:
        final_context = None
        if plan.segments:
            final_segment = plan.segments[-1]
            final_context = (
                final_segment.collision_context_after
                or final_segment.collision_context_before
            )
        transform: AttachedObjectTransform | None = None
        if final_context is not None:
            transform = next(
                (
                    item
                    for item in final_context.attached_object_transforms
                    if item.object_id == attached_object_id
                ),
                None,
            )
        if transform is not None:
            result.metadata["attached_object_transforms"] = {
                attached_object_id: transform.model_dump(mode="json")
            }
        else:
            # Preserve an existing runtime-captured transform.  A subsequent
            # request's collision factory will fail closed if it is absent.
            existing = result.metadata.get("attached_object_transforms")
            if not isinstance(existing, Mapping) or attached_object_id not in existing:
                result.metadata["attached_object_transforms"] = {}
    if request.task.goal.goal_type is GoalType.PLACE:
        object_id = request.task.goal.target_object_id
        pose = request.task.goal.target_pose
        if object_id and pose is not None:
            record = result.objects.get(object_id)
            updated = dict(record) if isinstance(record, Mapping) else {}
            updated["pose"] = pose.model_dump(mode="json")
            result.objects[object_id] = updated
    completed = list(result.scene.completed_subgoals)
    if completed_subgoal is not None and completed_subgoal not in completed:
        completed.append(completed_subgoal)
    if request.task.goal.goal_type is GoalType.EE_EXCHANGE:
        result.metadata["physical_active_ee"] = request.task.ee
        result.metadata["declared_active_ee"] = request.task.ee
    operation = request.task.metadata.get("operation")
    if operation in {"PICK_TOOL"}:
        result.metadata["held_tool"] = request.task.metadata.get("tool_id")
    elif operation in {"RETURN_TOOL", "TERMINAL_RETURN_TOOL"}:
        result.metadata["held_tool"] = None
    signature = _digest(
        {
            "previous": world.scene.signature,
            "plan_id": plan.plan_id,
            "robot_state": result.robot_state.model_dump(mode="json"),
            "objects": result.objects,
            "metadata": result.metadata,
        }
    )
    result.scene = SceneRef(
        signature=f"predicted:{signature}",
        completed_subgoals=completed,
        facts=list(result.scene.facts),
    )
    return result


class SelectedPlanMotionOrchestrator:
    """Plan resource transitions and subgoals in order, carrying state forward."""

    def __init__(
        self,
        request_planner: MotionRequestPlanner,
        *,
        store: MotionPlanStore | None = None,
        adapter: SelectedPlanMotionRequestAdapter | None = None,
    ) -> None:
        self._request_planner = request_planner
        self._store = store
        self._adapter = adapter or SelectedPlanMotionRequestAdapter()

    def plan(
        self,
        selected: SelectedPlan,
        *,
        initial_world: WorldSnapshot,
        constraints: ConstraintSource,
        options: OptionSource | None = None,
        selected_plan_artifact_id: str | None = None,
    ) -> SelectedPlanPlanningResult:
        selected_hash = _digest(selected.model_dump(mode="json"))
        artifact_id = selected_plan_artifact_id or (
            f"selected-plan-artifact:{selected_hash[:24]}"
        )
        current_world = initial_world.model_copy(deep=True)
        requests: list[MotionPlanRequest] = []
        plans: list[MotionPlan] = []
        paths: list[Path] = []
        selected_options: OptionSource = options or PlannerOptions()

        def execute(request: MotionPlanRequest, completed: str | None) -> None:
            nonlocal current_world
            # A transition list is created from the current snapshot. Refresh
            # each request immediately before planning so prior transitions are
            # reflected in its start state and identity lineage.
            if request.world.scene.signature != current_world.scene.signature:
                request.world = current_world.model_copy(deep=True)
                rebound = _digest(
                    {
                        "previous_request_id": request.request_id,
                        "world": request.world.model_dump(mode="json"),
                        "task": request.task.model_dump(mode="json"),
                        "constraints": request.constraints.model_dump(mode="json"),
                        "options": request.options.model_dump(mode="json"),
                    }
                )
                request.request_id = f"motion-request:rebound:{rebound[:20]}"
                request.provenance = request.provenance.model_copy(
                    update={
                        "artifact_id": f"motion-request-artifact:{rebound[:24]}",
                        "metadata": {
                            **request.provenance.metadata,
                            "rebound_to_scene_signature": (
                                current_world.scene.signature
                            ),
                        },
                    }
                )
            plan = _unwrap_plan(self._request_planner(request), request)
            requests.append(request)
            plans.append(plan)
            if self._store is not None:
                paths.append(self._store.save_plan(plan, index=len(plans) - 1))
            current_world = _predicted_world(
                current_world, request, plan, completed_subgoal=completed
            )

        for index, subgoal_id in enumerate(selected.subgoal_order):
            selected_constraints = _source_value(
                constraints,
                subgoal_id,
                index,
                MotionConstraints,
                "MotionConstraints",
            )
            option = _source_value(
                selected_options,
                subgoal_id,
                index,
                PlannerOptions,
                "PlannerOptions",
            )
            sliced = _single_subgoal_plan(selected, subgoal_id)
            assignment = sliced.candidate_assignments[0]
            transition_requests = _resource_transition_requests(
                parent_subgoal_id=subgoal_id,
                steps=sliced.steps,
                world=current_world,
                constraints=selected_constraints,
                options=option,
                selected_plan_artifact_id=artifact_id,
                fallback_ee=assignment.ee,
            )
            for transition_request in transition_requests:
                execute(transition_request, None)
            subgoal_request = self._adapter.convert(
                sliced,
                worlds=current_world,
                constraints=selected_constraints,
                options=option,
                selected_plan_artifact_id=artifact_id,
            )[0]
            execute(subgoal_request, subgoal_id)

        terminal_steps = [step for step in selected.steps if step.subgoal_id is None]
        if terminal_steps:
            last_ee = str(
                current_world.metadata.get("physical_active_ee")
                or selected.candidate_assignments[-1].ee
            )
            terminal_constraints = _source_value(
                constraints,
                selected.subgoal_order[-1],
                len(selected.subgoal_order) - 1,
                MotionConstraints,
                "MotionConstraints",
            )
            terminal_options = _source_value(
                selected_options,
                selected.subgoal_order[-1],
                len(selected.subgoal_order) - 1,
                PlannerOptions,
                "PlannerOptions",
            )
            for request in _resource_transition_requests(
                parent_subgoal_id="terminal",
                steps=terminal_steps,
                world=current_world,
                constraints=terminal_constraints,
                options=terminal_options,
                selected_plan_artifact_id=artifact_id,
                fallback_ee=last_ee,
            ):
                execute(request, None)

        manifest: Path | None = None
        if self._store is not None:
            manifest = self._store.save_manifest(
                selected_plan_hash=selected_hash,
                requests=requests,
                plans=plans,
                plan_paths=paths,
                final_world=current_world,
            )
        return SelectedPlanPlanningResult(
            requests=tuple(requests),
            plans=tuple(plans),
            final_world=current_world,
            plan_paths=tuple(paths),
            manifest_path=manifest,
        )


__all__ = [
    "MotionPlanStore",
    "SelectedPlanMotionOrchestrator",
    "SelectedPlanPlanningResult",
]
