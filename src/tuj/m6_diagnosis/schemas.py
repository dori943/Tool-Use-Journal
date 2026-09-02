"""Factories for fresh M6 JSON-compatible skeletons."""


def empty_failure_context() -> dict:
    return {
        "failure_id": None,
        "task": {"task_id": None, "instruction": None},
        "subgoal": {
            "subgoal_id": None, "description": None, "action_type": None,
            "target_object_ids": [], "selected_object_id": None, "selected_object_class": None,
            "preconditions": [], "postconditions": [], "invariants": [],
        },
        "verification": {"result": None, "expected_state": [], "observed_state": [], "violated_predicates": []},
        "scene": {"nodes": [], "relations": [], "object_states": {}},
        "grounding": {"physical_properties": {}, "geometry": {}, "metric_relations": {}, "ee_feasibility": {}, "confidence": {}},
        "task_plan": {"selected_ee": None, "selected_tool": None, "ee_candidates": [], "selection_score": None, "selection_reason": None, "final_order": [], "swap_plan": []},
        "motion_plan": {"keyframes": [], "ik_result": None, "collision_result": None, "approach_pose": None, "trajectory": None, "planning_status": None, "planning_error": None},
        "execution": {
            "executed_actions": [], "controller_status": None,
            "gripper": {"command": None, "position": None, "contact_detected": None, "force": None},
            "timeout": False, "error": None,
        },
        "observation": {"before_image": None, "after_image": None, "before_scene": None, "after_scene": None},
        "history": {"retry_count": 0, "previous_diagnoses": [], "previous_recoveries": [], "previous_outcomes": []},
    }


def empty_diagnosis() -> dict:
    return {
        "memory_context": {"retrieved_experiences": [], "diagnosis_evidence": []},
        "failure_type": None,
        "failure_cause": {"code": None, "description": None},
        "affected_module": None,
        "evidence": [],
        "confidence": None,
    }


def empty_recovery() -> dict:
    return {
        "decision_mode": None,
        "guidance": {
            "experience_ids": [],
            "past_recoveries": [],
            "recovery_evidence": [],
            "selection": {
                "selected_experience_ids": [],
                "selection_count": 0,
                "selection_audit": [],
            },
        },
        "recovery_category": None,
        "action": {
            "action_type": None, "target_module": None,
            "target": {"subgoal_id": None, "object_id": None, "property": None, "relation": None, "ee_id": None, "tool_id": None},
            "parameters": {},
        },
        "routing": {"restart_from": None, "rerun_modules": [], "invalidate": []},
        "outcome": {"status": None, "verification_result": None},
        "metadata": {"attempt": 1, "created_at": None},
    }
