"""Optional live smoke test for OpenAI M6 recovery routing."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tuj.m6.openai_recovery_router import OpenAIRecoveryRouter
from tuj.m6.recovery_config import get_recovery_model
from tuj.m6.recovery_router import apply_recovery_output, validate_recovery_output
from tuj.m6.schemas import empty_failure_context, empty_recovery


SAMPLE_FAILURE_CONTEXT = {
    **empty_failure_context(),
    "failure_id": "openai-recovery-smoke-001",
    "task": {"task_id": "task-spatula-acquire", "instruction": "Acquire the wooden spatula."},
    "subgoal": {
        "subgoal_id": "sg-spatula-acquire",
        "description": "Approach the wooden spatula and get ready to grasp it.",
        "action_type": "acquire",
        "selected_object_id": "spatula_03",
        "selected_object_class": "spatula",
    },
    "verification": {"result": "FAIL"},
    "task_plan": {"selected_ee": "2F", "selected_tool": None},
    "motion_plan": {"planning_status": "FAILED", "planning_error": "invalid approach"},
}

SAMPLE_DIAGNOSIS = {
    "failure_type": "PLANNING",
    "failure_cause": {
        "code": "INVALID_APPROACH",
        "description": "The selected approach collided with the workspace boundary.",
    },
    "affected_module": "M5",
    "evidence": ["motion planning reported invalid approach"],
    "confidence": 0.85,
}


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set")
        return 1

    router = OpenAIRecoveryRouter()
    print(f"Model: {get_recovery_model()}")
    print("Decision Mode: DIAGNOSIS_GUIDED")
    print("Recovery Evidence: []")
    print()

    recovery_output = router.route(
        SAMPLE_FAILURE_CONTEXT,
        SAMPLE_DIAGNOSIS,
        "DIAGNOSIS_GUIDED",
        [],
    )

    recovery = empty_recovery()
    recovery["decision_mode"] = "DIAGNOSIS_GUIDED"
    recovery["guidance"]["experience_ids"] = []
    recovery["guidance"]["recovery_evidence"] = []
    apply_recovery_output(recovery, recovery_output)
    validate_recovery_output(recovery)

    print("Recovery:")
    print(json.dumps(recovery, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
