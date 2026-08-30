"""Deterministic rack-relative EE dock/undock strategy generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from tuj.m4_motion.schema import (
    ArtifactProvenance,
    CollisionContext,
    KeyframeEventType,
    KeyframePlanCandidate,
    KeyframePlanArtifact,
    KeyframePlannerType,
    KeyframeType,
    RelativeKeyframeSpec,
    StrategyGenerationProvenance,
    StrategyGeneratorKind,
    ModuleName,
    MotionPlanRequest,
    WorldSnapshot,
)
from tuj.m4_motion.task_semantics import is_ee_exchange_task, task_operation


class EEExchangeTemplateError(ValueError):
    pass


def _unit_axis(record: Mapping[str, Any], ee: str) -> tuple[float, float, float]:
    raw = record.get("approach_axis_xyz")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
        raise EEExchangeTemplateError(
            f"rack slot for {ee!r} requires approach_axis_xyz"
        )
    values = tuple(float(value) for value in raw)
    # RelativeKeyframeSpec performs the authoritative unit-vector validation.
    return values


def _positive_distance(
    record: Mapping[str, Any], key: str, fallback: float, ee: str
) -> float:
    value = float(record.get(key, fallback))
    if value <= 0:
        raise EEExchangeTemplateError(f"rack slot {ee!r} has invalid {key}")
    return value


class EEExchangeTemplateGenerator:
    """Expand one Task Planner EE transition into a fixed semantic macro."""

    def __init__(
        self,
        *,
        template_id: str = "UR5E_RACK_EE_EXCHANGE_V1",
        default_staging_distance_m: float = 0.15,
        default_pre_dock_distance_m: float = 0.04,
    ) -> None:
        if not template_id:
            raise ValueError("template_id must not be empty")
        if default_staging_distance_m <= 0 or default_pre_dock_distance_m <= 0:
            raise ValueError("rack template distances must be positive")
        self._template_id = template_id
        self._staging = default_staging_distance_m
        self._pre_dock = default_pre_dock_distance_m

    def generate(
        self,
        world: WorldSnapshot,
        *,
        subgoal_id: str,
        from_ee: str | None,
        to_ee: str,
        attempt_index: int = 1,
    ) -> KeyframePlanCandidate:
        if from_ee is not None and from_ee == to_ee:
            raise EEExchangeTemplateError("from_ee and to_ee must differ")
        old_record = world.rack.get(from_ee) if from_ee is not None else None
        new_record = world.rack.get(to_ee)
        if (
            (from_ee is not None and not isinstance(old_record, Mapping))
            or not isinstance(new_record, Mapping)
        ):
            missing = [
                ee
                for ee, record in ((from_ee, old_record), (to_ee, new_record))
                if ee is not None and not isinstance(record, Mapping)
            ]
            raise EEExchangeTemplateError(f"missing rack slot records for {missing}")
        if "dock_pose" not in new_record or (
            from_ee is not None and "dock_pose" not in old_record
        ):
            raise EEExchangeTemplateError("required rack slots must define dock_pose")

        old_axis = (
            _unit_axis(old_record, from_ee) if from_ee is not None else None
        )
        new_axis = _unit_axis(new_record, to_ee)
        old_staging = (
            _positive_distance(
                old_record, "staging_distance_m", self._staging, from_ee
            )
            if from_ee is not None
            else None
        )
        new_staging = _positive_distance(
            new_record, "staging_distance_m", self._staging, to_ee
        )
        old_pre = (
            _positive_distance(
                old_record, "pre_dock_distance_m", self._pre_dock, from_ee
            )
            if from_ee is not None
            else None
        )
        new_pre = _positive_distance(
            new_record, "pre_dock_distance_m", self._pre_dock, to_ee
        )

        def keyframe(
            suffix: str,
            kind: KeyframeType,
            ee: str,
            axis: tuple[float, float, float],
            offset: float,
            planner: KeyframePlannerType,
            *,
            events: tuple[KeyframeEventType, ...] = (),
            context: str,
            context_after_events: str | None = None,
        ) -> RelativeKeyframeSpec:
            return RelativeKeyframeSpec(
                keyframe_id=f"{subgoal_id}:{suffix}",
                keyframe_type=kind,
                frame_ref=f"rack:{ee}",
                anchor="dock",
                approach_axis_xyz=axis,
                offset_along_approach_m=offset,
                planner=planner,
                events_after=list(events),
                collision_context_id=context,
                collision_context_after_events_id=context_after_events,
                metadata={"ee": ee, "template_id": self._template_id},
            )

        new_dock_contact_context = f"bare-flange-dock-contact:{to_ee}"
        new_attached_context = f"ee-attached:{to_ee}"

        keyframes: list[RelativeKeyframeSpec] = []
        if from_ee is not None:
            old_attached_context = f"ee-attached:{from_ee}"
            old_dock_contact_context = f"ee-attached-dock-contact:{from_ee}"
            keyframes.extend(
                [
                    keyframe(
                        "old-staging",
                        KeyframeType.EE_UNDOCK_STAGING,
                        from_ee,
                        old_axis,
                        -old_staging,
                        KeyframePlannerType.SAMPLING_BASED,
                        context=old_attached_context,
                    ),
                    keyframe(
                        "old-pre-undock",
                        KeyframeType.EE_PRE_UNDOCK,
                        from_ee,
                        old_axis,
                        -old_pre,
                        KeyframePlannerType.CARTESIAN,
                        context=old_attached_context,
                    ),
                    keyframe(
                        "old-undock",
                        KeyframeType.EE_UNDOCK,
                        from_ee,
                        old_axis,
                        0.0,
                        KeyframePlannerType.CARTESIAN,
                        events=(
                            KeyframeEventType.TOOL_UNLOCK,
                            KeyframeEventType.VERIFY_TOOL_RELEASE,
                        ),
                        context=old_dock_contact_context,
                        context_after_events="bare-flange",
                    ),
                    keyframe(
                        "old-retreat",
                        KeyframeType.EE_UNDOCK_STAGING,
                        from_ee,
                        old_axis,
                        -old_staging,
                        KeyframePlannerType.CARTESIAN,
                        context="bare-flange",
                    ),
                ]
            )
        keyframes.extend(
            [
                keyframe(
                    "new-staging",
                    KeyframeType.EE_DOCK_STAGING,
                    to_ee,
                    new_axis,
                    -new_staging,
                    KeyframePlannerType.SAMPLING_BASED,
                    context="bare-flange",
                ),
                keyframe(
                    "new-pre-dock",
                    KeyframeType.EE_PRE_DOCK,
                    to_ee,
                    new_axis,
                    -new_pre,
                    KeyframePlannerType.CARTESIAN,
                    context="bare-flange",
                ),
                keyframe(
                    "new-dock",
                    KeyframeType.EE_DOCK,
                    to_ee,
                    new_axis,
                    0.0,
                    KeyframePlannerType.CARTESIAN,
                    events=(
                        KeyframeEventType.TOOL_LOCK,
                        KeyframeEventType.VERIFY_TOOL_LOCK,
                    ),
                    context=new_dock_contact_context,
                    context_after_events=new_attached_context,
                ),
                keyframe(
                    "new-retreat",
                    KeyframeType.EE_DOCK_STAGING,
                    to_ee,
                    new_axis,
                    -new_staging,
                    KeyframePlannerType.CARTESIAN,
                    context=new_attached_context,
                ),
            ]
        )

        identity_payload = {
            "scene_signature": world.scene.signature,
            "subgoal_id": subgoal_id,
            "from_ee": from_ee,
            "to_ee": to_ee,
            "old_slot": old_record,
            "new_slot": new_record,
            "template_id": self._template_id,
        }
        encoded = json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        input_hash = hashlib.sha256(encoded).hexdigest()
        return KeyframePlanCandidate(
            strategy_id=(
                f"{subgoal_id}:ee-exchange:{from_ee}->{to_ee}"
                if from_ee is not None
                else f"{subgoal_id}:ee-attach:bare->{to_ee}"
            ),
            keyframes=keyframes,
            rationale=(
                "deterministic rack-relative undock, bare-flange transit, and dock"
                if from_ee is not None
                else "deterministic bare-flange transit and initial EE dock"
            ),
            provenance=StrategyGenerationProvenance(
                generator_kind=StrategyGeneratorKind.TEMPLATE,
                generator_id=self._template_id,
                input_hash=input_hash,
                attempt_index=attempt_index,
            ),
            metadata={"from_ee": from_ee, "to_ee": to_ee},
        )

    def build_collision_contexts(
        self,
        *,
        from_ee: str | None,
        to_ee: str,
        bare_flange_model_version: str = "ur5e-qc-bare-flange-v1",
        attached_model_versions: Mapping[str, str] | None = None,
        qc_master_entity: str = "qc_master",
        rack_support_prefix: str = "rack_support:",
    ) -> dict[str, CollisionContext]:
        """Build the physical-model and contact states used by the template.

        The two dock-contact contexts share their surrounding physical
        ``scene_state_id``.  They only relax the exact coupling/support contact
        needed at the event boundary.  Lock/unlock changes the physical scene
        and therefore selects a different compiled collision model.
        """

        if from_ee is not None and from_ee == to_ee:
            raise EEExchangeTemplateError("from_ee and to_ee must differ")
        versions = dict(attached_model_versions or {})

        def attached_version(ee: str) -> str:
            return versions.get(ee, f"ur5e-qc-{ee}-attached-v1")

        bare = CollisionContext(
            context_id="bare-flange",
            scene_state_id="bare-flange",
            collision_model_version=bare_flange_model_version,
        )
        new_contact = CollisionContext(
            context_id=f"bare-flange-dock-contact:{to_ee}",
            scene_state_id=bare.scene_state_id,
            allowed_collision_pairs=[(qc_master_entity, to_ee)],
            collision_model_version=bare.collision_model_version,
        )
        new_attached = CollisionContext(
            context_id=f"ee-attached:{to_ee}",
            scene_state_id=f"ee-attached:{to_ee}",
            active_ee=to_ee,
            collision_model_version=attached_version(to_ee),
        )
        contexts = [bare]
        if from_ee is not None:
            old_attached = CollisionContext(
                context_id=f"ee-attached:{from_ee}",
                scene_state_id=f"ee-attached:{from_ee}",
                active_ee=from_ee,
                collision_model_version=attached_version(from_ee),
            )
            old_contact = CollisionContext(
                context_id=f"ee-attached-dock-contact:{from_ee}",
                scene_state_id=old_attached.scene_state_id,
                active_ee=from_ee,
                allowed_collision_pairs=[
                    (from_ee, f"{rack_support_prefix}{from_ee}")
                ],
                collision_model_version=old_attached.collision_model_version,
            )
            contexts.extend((old_attached, old_contact))
        contexts.extend((new_contact, new_attached))
        return {context.context_id: context for context in contexts}


class EEExchangeKeyframeProvider:
    """Expose the deterministic EE exchange template through pipeline protocol."""

    def __init__(
        self, generator: EEExchangeTemplateGenerator | None = None
    ) -> None:
        self._generator = generator or EEExchangeTemplateGenerator()

    def generate(self, request: MotionPlanRequest) -> KeyframePlanArtifact:
        metadata = request.task.metadata
        from_ee = metadata.get("from_ee")
        to_ee = metadata.get("to_ee", request.task.ee)
        initial_attach = task_operation(request.task) in {
            "EE_ATTACH",
            "INITIAL_ATTACH_EE",
        }
        if not to_ee or (not initial_attach and not from_ee):
            raise EEExchangeTemplateError(
                "EE transition metadata lacks a required from_ee or to_ee"
            )
        candidate = self._generator.generate(
            request.world,
            subgoal_id=request.task.subgoal_id,
            from_ee=(str(from_ee) if from_ee else None),
            to_ee=str(to_ee),
        )
        identity = candidate.provenance.input_hash[:24]
        return KeyframePlanArtifact(
            artifact_id=f"keyframe-plan:ee-exchange:{identity}",
            provenance=ArtifactProvenance(
                artifact_id=f"keyframe-plan-artifact:ee-exchange:{identity}",
                artifact_type="KeyframePlanArtifact",
                produced_by=ModuleName.MOTION_PLANNER,
                invocation_id=f"ee-exchange:{request.request_id}",
                input_artifact_ids=[request.provenance.artifact_id],
                metadata={"generator": "EEExchangeTemplateGenerator"},
            ),
            scene_signature=request.world.scene.signature,
            subgoal_id=request.task.subgoal_id,
            candidates=[candidate],
        )


class RoutedKeyframeStrategyProvider:
    """Use deterministic templates for resource transitions and VLM otherwise."""

    def __init__(self, default_provider: object) -> None:
        self._default = default_provider
        self._ee_exchange = EEExchangeKeyframeProvider()

    def generate(self, request: MotionPlanRequest) -> KeyframePlanArtifact:
        if is_ee_exchange_task(request.task):
            return self._ee_exchange.generate(request)
        generate = getattr(self._default, "generate", None)
        if not callable(generate):
            raise TypeError("default keyframe provider must implement generate()")
        return generate(request)
