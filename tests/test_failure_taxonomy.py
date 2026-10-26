"""Post-execution goal checks must not be counted as path-planning failures."""
import pytest
from tuj.m5_motion.failure_taxonomy import classify_failure

@pytest.mark.parametrize("code", ["PLACEMENT_GOAL_NOT_SATISFIED", "PLACEMENT_PRESERVATION_FAILED"])
@pytest.mark.parametrize("with_steps", [False, True])
def test_post_execution_check_is_goal_failure_not_physical_noise(code, with_steps):
    detail = f"PlacementPreservationError: {code}: fruit_b -> plate_b"
    steps = [{"status": "FAILED", "execution_status": "SUCCESS", "route": "M5_MOTION_PLAN", "error": detail}] if with_steps else None
    result = classify_failure(detail, steps=steps)
    assert result["failure_stage"] == "goal"
    assert result["failure_code"] == code
    assert result["physics_related"] is False

@pytest.mark.parametrize("detail,stage,physics", [
    ("MotionPlanningPipelineError: IK_SEARCH_EXHAUSTED", "planning", False),
    ("LivePlanExecutionError: GRASP_RETENTION_FAILED", "execution", True),
])
def test_planning_and_contact_failures_keep_their_classification(detail, stage, physics):
    result = classify_failure(detail)
    assert result["failure_stage"] == stage
    assert result["physics_related"] is physics
