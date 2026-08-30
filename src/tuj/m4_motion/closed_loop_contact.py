"""Generic observe-adjust-retry execution for physical contact actions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from tuj.m4_motion.profiles import (
    ContactExecutionProfile,
    PushPlanningProfile,
    TaskRecoveryProfile,
)
from tuj.m4_motion.push_to_region import reduced_contact_step_distance


class ContactCheckpointBackend(Protocol):
    def checkpoint(self) -> object: ...

    def restore(self, checkpoint: object) -> None: ...


@dataclass(frozen=True, slots=True)
class ContactStepObservation:
    execution_succeeded: bool
    goal_satisfied: bool
    progress_m: float = 0.0
    contact_maintained: bool = True
    detail: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.progress_m < 0.0:
            raise ValueError("contact progress must be non-negative")


ContactStepRunner = Callable[[float, int, bool], ContactStepObservation]


class ClosedLoopContactStatus(str, Enum):
    SUCCESS = "SUCCESS"
    EXHAUSTED = "EXHAUSTED"
    ABORTED = "ABORTED"


@dataclass(frozen=True, slots=True)
class ContactAttemptRecord:
    attempt: int
    requested_distance_m: float
    reacquire_contact: bool
    observation: ContactStepObservation
    accepted: bool
    state_rolled_back: bool


@dataclass(frozen=True, slots=True)
class ClosedLoopContactResult:
    status: ClosedLoopContactStatus
    attempts: tuple[ContactAttemptRecord, ...]
    final_observation: ContactStepObservation | None

    @property
    def succeeded(self) -> bool:
        return self.status is ClosedLoopContactStatus.SUCCESS


class ClosedLoopContactExecutor:
    """Run bounded contact steps with observation-backed rollback and retry."""

    def __init__(
        self,
        backend: ContactCheckpointBackend,
        *,
        contact_profile: ContactExecutionProfile | None = None,
        push_profile: PushPlanningProfile | None = None,
        recovery_profile: TaskRecoveryProfile | None = None,
    ) -> None:
        self._backend = backend
        self._contact = contact_profile or ContactExecutionProfile()
        self._push = push_profile or PushPlanningProfile()
        self._recovery = recovery_profile or TaskRecoveryProfile()

    def run(self, step_runner: ContactStepRunner) -> ClosedLoopContactResult:
        requested_distance = self._push.nominal_step_distance_m
        reacquire = False
        records: list[ContactAttemptRecord] = []
        final: ContactStepObservation | None = None
        for attempt in range(1, self._recovery.maximum_execution_attempts + 1):
            checkpoint = self._backend.checkpoint()
            try:
                observation = step_runner(requested_distance, attempt, reacquire)
            except Exception as error:  # fail closed while preserving simulator state
                if self._recovery.rollback_failed_execution:
                    self._backend.restore(checkpoint)
                observation = ContactStepObservation(
                    execution_succeeded=False,
                    goal_satisfied=False,
                    contact_maintained=False,
                    detail=f"contact step raised {type(error).__name__}: {error}",
                )
                records.append(
                    ContactAttemptRecord(
                        attempt=attempt,
                        requested_distance_m=requested_distance,
                        reacquire_contact=reacquire,
                        observation=observation,
                        accepted=False,
                        state_rolled_back=self._recovery.rollback_failed_execution,
                    )
                )
                return ClosedLoopContactResult(
                    status=ClosedLoopContactStatus.ABORTED,
                    attempts=tuple(records),
                    final_observation=observation,
                )

            final = observation
            contact_ok = (
                observation.contact_maintained or not self._contact.maintain_contact
            )
            progress_ok = (
                observation.goal_satisfied
                or observation.progress_m >= self._contact.minimum_progress_m
            )
            accepted = observation.execution_succeeded and contact_ok and progress_ok
            rolled_back = False
            if not accepted and self._recovery.rollback_failed_execution:
                self._backend.restore(checkpoint)
                rolled_back = True
            records.append(
                ContactAttemptRecord(
                    attempt=attempt,
                    requested_distance_m=requested_distance,
                    reacquire_contact=reacquire,
                    observation=observation,
                    accepted=accepted,
                    state_rolled_back=rolled_back,
                )
            )
            if accepted and observation.goal_satisfied:
                return ClosedLoopContactResult(
                    status=ClosedLoopContactStatus.SUCCESS,
                    attempts=tuple(records),
                    final_observation=observation,
                )
            if accepted:
                requested_distance = self._push.nominal_step_distance_m
                reacquire = not observation.contact_maintained
            else:
                requested_distance = reduced_contact_step_distance(
                    requested_distance,
                    minimum_distance_m=self._push.minimum_step_distance_m,
                    retry_scale=self._push.retry_distance_scale,
                )
                reacquire = not observation.contact_maintained
        return ClosedLoopContactResult(
            status=ClosedLoopContactStatus.EXHAUSTED,
            attempts=tuple(records),
            final_observation=final,
        )


__all__ = [
    "ClosedLoopContactExecutor",
    "ClosedLoopContactResult",
    "ClosedLoopContactStatus",
    "ContactAttemptRecord",
    "ContactCheckpointBackend",
    "ContactStepObservation",
    "ContactStepRunner",
]
