"""Standalone M6 diagnosis/recovery orchestration skeleton."""

import logging

from .diagnosis import MockFailureDiagnoser, apply_diagnosis_output
from .diagnosis_aware_selection import DiagnosisAwareExperienceSelector
from .evidence import prepare_diagnosis_evidence, prepare_recovery_evidence
from .failure_context import FailureContextBuilder
from .memory_adapter import MemoryAdapter
from .recovery_config import create_recovery_router
from .recovery_router import MockRecoveryRouter, apply_recovery_output
from .schemas import empty_diagnosis, empty_recovery

logger = logging.getLogger(__name__)


class DiagnoseRouter:
    def __init__(
        self,
        context_builder=None,
        memory_adapter=None,
        failure_diagnoser=None,
        experience_selector=None,
        recovery_router=None,
    ):
        self.context_builder = context_builder or FailureContextBuilder()
        self.memory_adapter = memory_adapter or MemoryAdapter()
        self.failure_diagnoser = failure_diagnoser or MockFailureDiagnoser()
        self.experience_selector = experience_selector or DiagnosisAwareExperienceSelector()
        self.recovery_router = recovery_router or create_recovery_router()

    def run(self, pipeline_state, top_k=3) -> dict:
        failure_context = self.context_builder.build(pipeline_state)
        retrieved = self.memory_adapter.retrieve_experiences(failure_context, top_k=top_k)
        if not isinstance(retrieved, list):
            retrieved = []

        diagnosis_evidence = prepare_diagnosis_evidence(retrieved)
        recovery_evidence = prepare_recovery_evidence(retrieved)

        logger.debug(
            "evidence prepared diagnosis_ids=%s recovery_ids=%s",
            [item.get("experience_id") for item in diagnosis_evidence],
            [item.get("experience_id") for item in recovery_evidence],
        )

        diagnosis = empty_diagnosis()
        diagnosis["memory_context"]["retrieved_experiences"] = retrieved
        diagnosis["memory_context"]["diagnosis_evidence"] = diagnosis_evidence

        diagnosis_output = self.failure_diagnoser.diagnose(
            failure_context,
            diagnosis_evidence,
        )
        apply_diagnosis_output(diagnosis, diagnosis_output)

        selection_result = self.experience_selector.select(
            diagnosis,
            retrieved,
            recovery_evidence,
        )
        selected_recovery_evidence = selection_result.get("selected_recovery_evidence") or []

        recovery = empty_recovery()
        recovery["decision_mode"] = (
            "EXPERIENCE_GUIDED"
            if selection_result["selection_count"] > 0
            else "DIAGNOSIS_GUIDED"
        )
        recovery["guidance"]["experience_ids"] = list(
            selection_result["selected_experience_ids"]
        )
        recovery["guidance"]["recovery_evidence"] = selected_recovery_evidence
        recovery["guidance"]["selection"] = {
            "selected_experience_ids": list(selection_result["selected_experience_ids"]),
            "selection_count": selection_result["selection_count"],
            "selection_audit": list(selection_result["selection_audit"]),
        }

        logger.debug(
            "diagnosis-aware selection decision_mode=%s selected_ids=%s",
            recovery["decision_mode"],
            recovery["guidance"]["experience_ids"],
        )

        recovery_output = self.recovery_router.route(
            failure_context,
            diagnosis,
            recovery["decision_mode"],
            selected_recovery_evidence,
        )
        apply_recovery_output(recovery, recovery_output)

        logger.debug(
            "recovery routed category=%s action_type=%s restart_from=%s",
            recovery["recovery_category"],
            recovery["action"].get("action_type"),
            recovery["routing"].get("restart_from"),
        )

        return {"failure_context": failure_context, "diagnosis": diagnosis, "recovery": recovery}
