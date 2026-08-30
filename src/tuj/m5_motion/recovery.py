"""Evidence-backed recovery attribution and localized module retry execution."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from tuj.m5_motion.execution import (
    GoalEvaluationStatus,
    SelectedPlanExecutionResult,
    SequenceExecutionStatus,
)
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    ExecutionReport,
    ExecutionStatus,
    FailureObservation,
    ModuleName,
    MotionPlan,
    MotionPlanRequest,
    RecoveryDirective,
    RetryPolicy,
    RootCause,
    SimulationRun,
)


class RecoveryAttributionError(RuntimeError):
    """A failure cannot be assigned to one module without guessing."""


class RecoveryExecutionError(RuntimeError):
    """A valid directive cannot be executed by the registered retry boundary."""


RecoveryHandler = Callable[[RecoveryDirective], Any]


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


class ArtifactLineageIndex:
    """Small in-memory DAG used to find retry inputs and invalidated outputs."""

    def __init__(self, artifacts: Iterable[ArtifactProvenance]) -> None:
        self._artifacts: dict[str, ArtifactProvenance] = {}
        self._children: dict[str, set[str]] = defaultdict(set)
        for artifact in artifacts:
            previous = self._artifacts.get(artifact.artifact_id)
            if previous is not None and previous != artifact:
                raise ValueError(
                    f"artifact {artifact.artifact_id!r} has conflicting provenance"
                )
            self._artifacts[artifact.artifact_id] = artifact
        for artifact in self._artifacts.values():
            for parent in artifact.input_artifact_ids:
                self._children[parent].add(artifact.artifact_id)

    def get(self, artifact_id: str) -> ArtifactProvenance | None:
        return self._artifacts.get(artifact_id)

    def produced_by(self, module: ModuleName) -> tuple[ArtifactProvenance, ...]:
        return tuple(
            artifact
            for artifact in self._artifacts.values()
            if artifact.produced_by is module
        )

    def descendants(self, artifact_id: str) -> tuple[str, ...]:
        result: list[str] = []
        pending = deque(sorted(self._children.get(artifact_id, ())))
        seen: set[str] = set()
        while pending:
            child = pending.popleft()
            if child in seen:
                continue
            seen.add(child)
            result.append(child)
            pending.extend(sorted(self._children.get(child, ())))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class RecoveryExecutionResult:
    directive: RecoveryDirective
    value: Any


class RecoveryOrchestrator:
    """Create a localized retry directive and dispatch only its target module."""

    _FAILURE_MODULE = {
        "EXECUTION_COLLISION": ModuleName.MOTION_PLANNER,
        "GRASP_LOST": ModuleName.MOTION_PLANNER,
        "SIMULATION_TIMEOUT": ModuleName.SIMULATOR,
        "PLAYBACK_RUNTIME_FAILED": ModuleName.SIMULATOR,
        "CONTROLLER_TRACKING_ERROR": ModuleName.CONTROLLER,
    }

    def __init__(
        self,
        handlers: Mapping[ModuleName, RecoveryHandler] | None = None,
    ) -> None:
        self._handlers = dict(handlers or {})

    def register(self, module: ModuleName, handler: RecoveryHandler) -> None:
        self._handlers[module] = handler

    @staticmethod
    def _explicit_module(report: ExecutionReport) -> ModuleName | None:
        if report.failure is None:
            return None
        value = report.failure.observed.get("root_cause_module")
        if value is None:
            return None
        try:
            return value if isinstance(value, ModuleName) else ModuleName(str(value))
        except ValueError as error:
            raise RecoveryAttributionError(
                f"failure declares unknown root_cause_module {value!r}"
            ) from error

    @staticmethod
    def _root_artifact(
        module: ModuleName,
        *,
        request: MotionPlanRequest,
        plan: MotionPlan,
        run: SimulationRun,
        report: ExecutionReport,
        lineage: ArtifactLineageIndex,
    ) -> ArtifactProvenance:
        direct = {
            ModuleName.TASK_PLANNER: request.provenance,
            ModuleName.MOTION_PLANNER: plan.provenance,
            ModuleName.SIMULATOR: run.provenance,
        }.get(module)
        if direct is not None and direct.produced_by is module:
            return direct
        candidates = lineage.produced_by(module)
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise RecoveryAttributionError(
                f"lineage has no artifact produced by {module.value}"
            )
        raise RecoveryAttributionError(
            f"lineage has multiple {module.value} artifacts; explicit artifact evidence is required"
        )

    def directive(
        self,
        *,
        request: MotionPlanRequest,
        plan: MotionPlan,
        run: SimulationRun,
        report: ExecutionReport,
        additional_artifacts: Iterable[ArtifactProvenance] = (),
        max_attempts: int = 3,
        parameter_overrides: Mapping[str, Any] | None = None,
    ) -> RecoveryDirective:
        if report.failure is None:
            raise RecoveryAttributionError("successful report has no failure to recover")
        if report.run_id != run.run_id or report.plan_id != plan.plan_id:
            raise RecoveryAttributionError("report/run/plan identities do not match")
        if plan.request_id != request.request_id:
            raise RecoveryAttributionError("plan/request identities do not match")

        module = self._explicit_module(report)
        confidence = 1.0
        if module is None:
            module = self._FAILURE_MODULE.get(report.failure.code)
            confidence = 0.85
        if module is None:
            raise RecoveryAttributionError(
                f"failure code {report.failure.code!r} has no unambiguous module attribution"
            )

        artifacts = (
            request.provenance,
            plan.provenance,
            run.provenance,
            report.provenance,
            *tuple(additional_artifacts),
        )
        lineage = ArtifactLineageIndex(artifacts)
        root = self._root_artifact(
            module,
            request=request,
            plan=plan,
            run=run,
            report=report,
            lineage=lineage,
        )
        if not root.input_artifact_ids:
            raise RecoveryAttributionError(
                f"root artifact {root.artifact_id!r} has no restart input"
            )
        if max_attempts < root.attempt:
            raise RecoveryAttributionError(
                "max_attempts is lower than the root artifact attempt"
            )
        invalidated = [root.artifact_id, *lineage.descendants(root.artifact_id)]
        evidence_refs = list(report.failure.evidence_refs)
        if report.trace_ref is not None and report.trace_ref not in evidence_refs:
            evidence_refs.append(report.trace_ref)
        identity = _digest(
            {
                "report_id": report.report_id,
                "root_artifact_id": root.artifact_id,
                "failure_code": report.failure.code,
                "attempt": root.attempt,
                "overrides": dict(parameter_overrides or {}),
            }
        )
        directive_id = f"recovery-directive:{identity[:20]}"
        return RecoveryDirective(
            directive_id=directive_id,
            source_report_id=report.report_id,
            provenance=ArtifactProvenance(
                artifact_id=f"recovery-directive-artifact:{identity[:24]}",
                artifact_type="RecoveryDirective",
                produced_by=ModuleName.RECOVERY_ORCHESTRATOR,
                invocation_id=directive_id,
                input_artifact_ids=[report.provenance.artifact_id],
            ),
            root_cause=RootCause(
                module=module,
                invocation_id=root.invocation_id,
                artifact_id=root.artifact_id,
                cause_code=report.failure.code,
                confidence=confidence,
                evidence_refs=evidence_refs,
                explanation=report.failure.message,
            ),
            target_module=module,
            restart_from_artifact_id=root.input_artifact_ids[0],
            invalidated_artifact_ids=invalidated,
            retry_policy=RetryPolicy(
                current_attempt=root.attempt,
                max_attempts=max_attempts,
                parameter_overrides=dict(parameter_overrides or {}),
            ),
        )

    def directive_for_sequence(
        self,
        result: SelectedPlanExecutionResult,
        *,
        max_attempts: int = 3,
        parameter_overrides: Mapping[str, Any] | None = None,
    ) -> RecoveryDirective:
        index = result.failed_index
        if index is None or index >= len(result.reports):
            raise RecoveryAttributionError(
                "sequence has no executed failed report to recover"
            )
        report = result.reports[index]
        if (
            result.status is SequenceExecutionStatus.GOAL_FAILED
            and report.status is ExecutionStatus.SUCCESS
        ):
            failed_goals = [
                evaluation
                for evaluation in result.goal_evaluations
                if evaluation.status is not GoalEvaluationStatus.SATISFIED
            ]
            if not failed_goals:
                raise RecoveryAttributionError(
                    "goal-failed sequence has no failed goal evidence"
                )
            evaluation = failed_goals[-1]
            module = (
                ModuleName.TASK_PLANNER
                if evaluation.goal_type == "TASK"
                else ModuleName.MOTION_PLANNER
            )
            observed = {
                **dict(evaluation.observed or {}),
                "root_cause_module": module.value,
                "goal_evaluation": evaluation.as_dict(),
            }
            report = ExecutionReport.model_validate(
                {
                    **report.model_dump(mode="json"),
                    "status": ExecutionStatus.FAILED.value,
                    "failure": FailureObservation(
                        code="GOAL_NOT_REACHED",
                        category="TASK_GOAL",
                        message=evaluation.detail,
                        observed=observed,
                    ).model_dump(mode="json"),
                }
            )
        return self.directive(
            request=result.planning.requests[index],
            plan=result.planning.plans[index],
            run=result.runs[index],
            report=report,
            max_attempts=max_attempts,
            parameter_overrides=parameter_overrides,
        )

    def execute(self, directive: RecoveryDirective) -> RecoveryExecutionResult:
        policy = directive.retry_policy
        if policy.current_attempt >= policy.max_attempts:
            raise RecoveryExecutionError(
                f"retry budget exhausted at attempt {policy.current_attempt}"
            )
        handler = self._handlers.get(directive.target_module)
        if handler is None:
            raise RecoveryExecutionError(
                f"no recovery handler registered for {directive.target_module.value}"
            )
        value = handler(directive)
        return RecoveryExecutionResult(directive=directive, value=value)


__all__ = [
    "ArtifactLineageIndex",
    "RecoveryAttributionError",
    "RecoveryExecutionError",
    "RecoveryExecutionResult",
    "RecoveryHandler",
    "RecoveryOrchestrator",
]
