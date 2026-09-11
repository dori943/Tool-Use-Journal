"""Validate action-phase ownership before any IK or collision search."""

from __future__ import annotations

from tuj.m5_motion.schema import (
    KeyframeEventType,
    KeyframePlanArtifact,
    KeyframeType,
    MotionPlanRequest,
)
from tuj.m5_motion.task_semantics import (
    is_acquire_task,
    is_release_task,
    task_operation,
)
from tuj.m5_motion.contact_keyframe_validation import (
    ContactKeyframeGeometryError,
    is_tool_act_contact_geometry_scope,
    validate_sweep_keyframe_strategy,
)


class KeyframePhaseContractError(ValueError):
    """A generated strategy performs effects owned by another task phase."""


_RELEASE_EVENTS = {
    KeyframeEventType.DETACH_OBJECT,
    KeyframeEventType.GRIPPER_OPEN,
    KeyframeEventType.SUCTION_OFF,
}


def validate_keyframe_phase_contract(
    request: MotionPlanRequest,
    artifact: KeyframePlanArtifact,
) -> None:
    """Reject early release in TRANSPORT and malformed release sequences."""

    operation = task_operation(request.task)
    for strategy in artifact.candidates:
        keyframes = strategy.keyframes
        if is_acquire_task(request.task) and operation != "PICK_TOOL":
            grasp_indices = [
                index
                for index, keyframe in enumerate(keyframes)
                if keyframe.keyframe_type is KeyframeType.GRASP
            ]
            if len(grasp_indices) != 1:
                raise KeyframePhaseContractError(
                    f"acquire strategy {strategy.strategy_id!r} requires exactly one GRASP"
                )
            grasp_index = grasp_indices[0]
            if not any(
                keyframe.keyframe_type is KeyframeType.PRE_GRASP
                for keyframe in keyframes[:grasp_index]
            ):
                raise KeyframePhaseContractError(
                    f"acquire strategy {strategy.strategy_id!r} requires "
                    "PRE_GRASP before GRASP"
                )
        if operation == "TRANSPORT":
            forbidden = [
                keyframe.keyframe_id
                for keyframe in keyframes
                if keyframe.keyframe_type is KeyframeType.PLACE
                or any(event in _RELEASE_EVENTS for event in keyframe.events_after)
            ]
            if forbidden:
                raise KeyframePhaseContractError(
                    f"TRANSPORT strategy {strategy.strategy_id!r} contains PLACE or "
                    f"release effects at {', '.join(forbidden)}"
                )
        if is_tool_act_contact_geometry_scope(request):
            try:
                validate_sweep_keyframe_strategy(request, keyframes)
            except ContactKeyframeGeometryError as error:
                raise KeyframePhaseContractError(
                    f"sweep strategy {strategy.strategy_id!r}: {error}"
                ) from error
        if not is_release_task(request.task):
            continue
        place_indices = [
            index
            for index, keyframe in enumerate(keyframes)
            if keyframe.keyframe_type is KeyframeType.PLACE
        ]
        if len(place_indices) != 1:
            raise KeyframePhaseContractError(
                f"release strategy {strategy.strategy_id!r} requires exactly one PLACE"
            )
        place_index = place_indices[0]
        if not any(
            keyframe.keyframe_type is KeyframeType.RETREAT
            for keyframe in keyframes[place_index + 1 :]
        ):
            raise KeyframePhaseContractError(
                f"release strategy {strategy.strategy_id!r} requires RETREAT after PLACE"
            )
        release_locations = [
            (index, event)
            for index, keyframe in enumerate(keyframes)
            for event in keyframe.events_after
            if event in _RELEASE_EVENTS
        ]
        if any(index != place_index for index, _ in release_locations):
            raise KeyframePhaseContractError(
                f"release strategy {strategy.strategy_id!r} must release only at PLACE"
            )
        events = keyframes[place_index].events_after
        if KeyframeEventType.DETACH_OBJECT in events:
            detach_index = events.index(KeyframeEventType.DETACH_OBJECT)
            open_indices = [
                events.index(event)
                for event in (KeyframeEventType.GRIPPER_OPEN, KeyframeEventType.SUCTION_OFF)
                if event in events
            ]
            if open_indices and detach_index > min(open_indices):
                raise KeyframePhaseContractError(
                    f"release strategy {strategy.strategy_id!r} must DETACH_OBJECT "
                    "before opening the gripper or disabling suction"
                )


__all__ = ["KeyframePhaseContractError", "validate_keyframe_phase_contract"]
