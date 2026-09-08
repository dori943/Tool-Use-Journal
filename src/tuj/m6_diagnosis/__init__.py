"""M6 - failure diagnosis and recovery routing (canonical package ``tuj.m6_diagnosis``)."""

from .artifact_adapter import (
    ArtifactAdapterError,
    FailureContextArtifactAdapter,
)
from .diagnosis import FailureDiagnoser, MockFailureDiagnoser
from .diagnosis_config import (
    create_failure_diagnoser,
    get_diagnoser_backend,
)
from .diagnosis_aware_selection import DiagnosisAwareExperienceSelector
from .failure_context import FailureContextBuilder
from .m6 import DiagnoseRouter
from .memory_adapter import MemoryAdapter, MemoryAppendError, append_experience
from .openai_recovery_router import OpenAIRecoveryRouter
from .openai_vlm_diagnoser import OpenAIVLMFailureDiagnoser
from .recovery_config import (
    create_recovery_router,
    get_recovery_router_backend,
)
from .recovery_dispatcher import RecoveryDispatchError, dispatch_recovery
from .recovery_outcome import (
    RecoveryResultError,
    process_recovery_outcome,
    validate_recovery_result,
)
from .recovery_request import RecoveryRequest, RecoveryRequestError, build_recovery_request
from .recovery_router import MockRecoveryRouter, RecoveryRouter
from .runtime_experience import (
    RuntimeExperienceBuilder,
    build_runtime_experience,
)
from .e2e_runner import M6E2EError, list_failure_task_availability, run_m6_e2e

__all__ = [
    "ArtifactAdapterError",
    "DiagnoseRouter",
    "DiagnosisAwareExperienceSelector",
    "FailureContextArtifactAdapter",
    "FailureContextBuilder",
    "FailureDiagnoser",
    "M6E2EError",
    "MemoryAdapter",
    "MemoryAppendError",
    "MockFailureDiagnoser",
    "MockRecoveryRouter",
    "OpenAIRecoveryRouter",
    "OpenAIVLMFailureDiagnoser",
    "RecoveryDispatchError",
    "RecoveryRequest",
    "RecoveryRequestError",
    "RecoveryResultError",
    "RecoveryRouter",
    "RuntimeExperienceBuilder",
    "append_experience",
    "build_recovery_request",
    "build_runtime_experience",
    "create_failure_diagnoser",
    "create_recovery_router",
    "dispatch_recovery",
    "get_diagnoser_backend",
    "get_recovery_router_backend",
    "list_failure_task_availability",
    "process_recovery_outcome",
    "run_m6_e2e",
    "validate_recovery_result",
]
