"""Physical suitability scoring for EE/tool assignments.

Task Planner evaluates only task-level payload and wrench constraints. Contact
pose, approach geometry, and pickup mechanics belong to the motion planner and
controller and are intentionally absent here.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping, Protocol

from tuj.m3_taskplanner.candidate_provider import Candidate
from tuj.m3_taskplanner.conditions import subgoal_required_wrench
from tuj.m3_taskplanner.diagnostics import ReasonCode, Rejection, make_rejection
from tuj.m3_taskplanner.models import EndEffectorSpec, ObjectSpec, ResourceCatalog, Subgoal


class SuitabilityStatus(str, enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class SuitabilityComponent:
    status: SuitabilityStatus
    score: float | None = None
    required: float | None = None
    capacity: float | None = None
    margin: float | None = None
    unit: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def not_applicable(cls, reason: str) -> "SuitabilityComponent":
        return cls(
            SuitabilityStatus.NOT_APPLICABLE,
            details={"reason": reason},
        )

    @classmethod
    def unknown(cls, reason: str, **details: Any) -> "SuitabilityComponent":
        return cls(
            SuitabilityStatus.UNKNOWN,
            details={"reason": reason, **details},
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"status": self.status.value}
        for key in ("score", "required", "capacity", "margin", "unit"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.details:
            result["details"] = dict(self.details)
        return result


@dataclass(frozen=True, slots=True)
class SuitabilityAssessment:
    overall_score: float | None
    components: Mapping[str, SuitabilityComponent]
    failure_reason: ReasonCode | None = None
    failure_message: str = ""

    @property
    def failed(self) -> bool:
        return self.failure_reason is not None

    @property
    def unknown_components(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, component in self.components.items()
            if component.status is SuitabilityStatus.UNKNOWN
        )

    @property
    def known_min_score(self) -> float | None:
        scores = [
            component.score
            for component in self.components.values()
            if component.status in (SuitabilityStatus.PASS, SuitabilityStatus.FAIL)
            and component.score is not None
        ]
        return min(scores) if scores else None

    def rejection(self, candidate: Candidate, subgoal: Subgoal) -> Rejection | None:
        if not self.failed or self.failure_reason is None:
            return None
        return make_rejection(
            "candidate",
            self.failure_reason,
            self.failure_message,
            subgoal_id=subgoal.subgoal_id,
            candidate_id=candidate.candidate_id,
            suitability=self.to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        applicable = [
            component
            for component in self.components.values()
            if component.status is not SuitabilityStatus.NOT_APPLICABLE
        ]
        unknown = list(self.unknown_components)
        known_count = sum(
            component.status in (SuitabilityStatus.PASS, SuitabilityStatus.FAIL)
            for component in applicable
        )
        result: dict[str, Any] = {
            "overall_suitability": self.overall_score,
            "known_min_score": self.known_min_score,
            "applicable_component_count": len(applicable),
            "known_component_count": known_count,
            "unknown_component_count": len(unknown),
            "unknown_components": unknown,
            "aggregation": "minimum_applicable_margin",
            "components": {
                name: component.to_dict()
                for name, component in self.components.items()
            },
        }
        if self.failure_reason is not None:
            result["failure_reason"] = self.failure_reason.value
            result["failure_message"] = self.failure_message
        return result


class SuitabilityScorer(Protocol):
    def score(
        self, candidate: Candidate, subgoal: Subgoal
    ) -> SuitabilityAssessment: ...


class PhysicsSuitabilityScorer:
    """Evaluate payload and task-wrench margins without contact-pose logic."""

    def __init__(self, catalog: ResourceCatalog) -> None:
        self._catalog = catalog

    def score(
        self, candidate: Candidate, subgoal: Subgoal
    ) -> SuitabilityAssessment:
        ee = self._catalog.end_effectors.get(candidate.ee)
        if ee is None:
            return SuitabilityAssessment(
                0.0,
                {},
                ReasonCode.UNKNOWN_EE,
                f"EE {candidate.ee!r} is not in the resource catalog",
            )

        obj = self._resolve_object(candidate, subgoal)
        components = {
            "payload": self._payload(candidate, subgoal, ee, obj),
            "wrench": self._wrench(candidate, subgoal),
        }
        failure = self._first_failure(components)
        if failure is not None:
            reason, message = failure
            return SuitabilityAssessment(0.0, components, reason, message)

        applicable = [
            component
            for component in components.values()
            if component.status is not SuitabilityStatus.NOT_APPLICABLE
        ]
        if any(c.status is SuitabilityStatus.UNKNOWN for c in applicable):
            overall: float | None = None
        else:
            scores = [c.score for c in applicable if c.score is not None]
            overall = min(scores) if scores else None
        return SuitabilityAssessment(overall, components)

    def _resolve_object(
        self, candidate: Candidate, subgoal: Subgoal
    ) -> ObjectSpec | None:
        object_id = candidate.metadata.get("target_object_id")
        if not isinstance(object_id, str):
            object_id = next(
                (target for target in subgoal.target_ids if target in self._catalog.objects),
                subgoal.target_ids[0] if subgoal.target_ids else None,
            )
        catalog_obj = self._catalog.objects.get(object_id) if object_id else None
        mass = candidate.metadata.get("object_mass_kg")
        if not isinstance(mass, (int, float)):
            nested = candidate.metadata.get("object")
            mass = nested.get("mass_kg") if isinstance(nested, Mapping) else None
        if isinstance(mass, (int, float)):
            return ObjectSpec(
                mass_kg=float(mass),
                bbox_mm=catalog_obj.bbox_mm if catalog_obj is not None else None,
                material=catalog_obj.material if catalog_obj is not None else None,
            )
        return catalog_obj

    def _payload(
        self,
        candidate: Candidate,
        subgoal: Subgoal,
        ee: EndEffectorSpec,
        obj: ObjectSpec | None,
    ) -> SuitabilityComponent:
        needs_object_mass = bool(subgoal.target_ids) and not bool(
            candidate.metadata.get("object_remains_supported")
        )
        tool_mass: float | None = 0.0
        if candidate.tool is not None:
            tool = self._catalog.tools.get(candidate.tool)
            tool_mass = tool.mass if tool is not None else None
        if ee.payload is None:
            return SuitabilityComponent.unknown("EE payload is missing")
        if tool_mass is None:
            return SuitabilityComponent.unknown("tool mass is missing")
        if needs_object_mass and (obj is None or obj.mass_kg is None):
            return SuitabilityComponent.unknown("object mass is missing")
        load = (obj.mass_kg or 0.0 if obj is not None and needs_object_mass else 0.0)
        load += tool_mass
        if not needs_object_mass and candidate.tool is None:
            return SuitabilityComponent.not_applicable("no carried object or tool")
        return _capacity_component(load, ee.payload, "kg")

    def _wrench(
        self, candidate: Candidate, subgoal: Subgoal
    ) -> SuitabilityComponent:
        required = subgoal_required_wrench(subgoal)
        meta_required = candidate.metadata.get("required_wrench")
        if isinstance(meta_required, (int, float)):
            required = max(required or 0.0, float(meta_required))
        if required is None:
            return SuitabilityComponent.not_applicable(
                "subgoal has no wrench requirement"
            )
        capacity = candidate.metadata.get("deliverable_wrench")
        if not isinstance(capacity, (int, float)) and candidate.tool is not None:
            tool = self._catalog.tools.get(candidate.tool)
            capacity = tool.deliverable_wrench if tool is not None else None
        if not isinstance(capacity, (int, float)):
            return SuitabilityComponent.unknown("deliverable wrench is missing")
        return _capacity_component(float(required), float(capacity), "wrench")

    @staticmethod
    def _first_failure(
        components: Mapping[str, SuitabilityComponent],
    ) -> tuple[ReasonCode, str] | None:
        if components["payload"].status is SuitabilityStatus.FAIL:
            return (
                ReasonCode.PAYLOAD_EXCEEDED,
                "combined object/tool load exceeds EE payload",
            )
        if components["wrench"].status is SuitabilityStatus.FAIL:
            return (
                ReasonCode.WRENCH_INSUFFICIENT,
                "tool wrench capacity is below the task requirement",
            )
        return None


def _clamp01(value: float) -> float:
    if not isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _margin_score(required: float, capacity: float) -> float:
    if required <= 0:
        return 1.0
    if capacity <= 0:
        return 0.0
    return _clamp01(capacity / (capacity + required))


def _capacity_component(
    required: float, capacity: float, unit: str
) -> SuitabilityComponent:
    return SuitabilityComponent(
        SuitabilityStatus.PASS if required <= capacity else SuitabilityStatus.FAIL,
        score=_margin_score(required, capacity),
        required=required,
        capacity=capacity,
        margin=capacity - required,
        unit=unit,
    )
