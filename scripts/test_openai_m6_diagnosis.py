"""Optional live smoke test for OpenAI VLM M6 failure diagnosis."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tuj.m6.diagnosis import validate_diagnosis_output
from tuj.m6.diagnosis_config import get_diagnosis_model
from tuj.m6.openai_vlm_diagnoser import OpenAIVLMFailureDiagnoser
from tuj.m6.schemas import empty_failure_context


SAMPLE_FAILURE_CONTEXT = {
    **empty_failure_context(),
    "failure_id": "openai-smoke-001",
    "task": {"task_id": "task-spatula-acquire", "instruction": "Acquire the wooden spatula."},
    "subgoal": {
        "subgoal_id": "sg-spatula-acquire",
        "description": "Approach the wooden spatula and get ready to grasp it.",
        "action_type": "acquire",
        "target_object_ids": ["spatula_03"],
        "selected_object_id": "spatula_03",
        "selected_object_class": "spatula",
        "postconditions": ["holding(spatula)"],
    },
    "verification": {
        "result": "FAIL",
        "expected_state": ["aligned(spatula)"],
        "observed_state": ["misaligned(spatula)"],
        "violated_predicates": [],
    },
    "task_plan": {"selected_ee": "2F", "selected_tool": None},
    "motion_plan": {"planning_status": "FAILED", "planning_error": "approach in collision"},
    "execution": {"controller_status": None},
}


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set")
        return 1

    diagnoser = OpenAIVLMFailureDiagnoser()
    print(f"Model: {get_diagnosis_model()}")
    print()

    diagnosis_output = diagnoser.diagnose(SAMPLE_FAILURE_CONTEXT, [])
    validate_diagnosis_output(diagnosis_output)

    print("Diagnosis:")
    print(json.dumps(diagnosis_output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
