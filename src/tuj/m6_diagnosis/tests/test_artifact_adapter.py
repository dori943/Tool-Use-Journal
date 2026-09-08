"""Unit tests for FailureContextArtifactAdapter."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tuj.m6_diagnosis.artifact_adapter import (
    ArtifactAdapterError,
    FailureContextArtifactAdapter,
)
from tuj.m6_diagnosis.failure_context import FailureContextBuilder
from tuj.m6_diagnosis.schemas import empty_failure_context


REPO_ROOT = Path(__file__).resolve().parents[4]
C1_1_OUTPUT = REPO_ROOT / "output" / "c1_1"
MEMORY_PATH = REPO_ROOT / "output" / "memory.json"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _minimal_m2(*, subgoal_id="SG1", detail_id="SG1_d1", tool_id="obj_ladle_ladle"):
    return {
        "task": "demo instruction",
        "m2_subgoals": [
            {
                "subgoal_id": subgoal_id,
                "goal": "pick the ladle",
                "target_ids": ["obj_block_block_0"],
                "selected_tool_id": tool_id,
                "selection_reason": "tool looks suitable",
                "selection_evidence": [
                    {"node": tool_id, "feasible_ees": ["2F", "3F"]}
                ],
                "details": [
                    {
                        "detail_id": detail_id,
                        "action_type": "acquire",
                        "pre": [{"id": "p0", "expr": "hand_empty"}],
                        "establish": [{"id": "e0", "expr": "holding(?tool)"}],
                        "destroy": [{"id": "d0", "expr": "hand_empty"}],
                    }
                ],
            }
        ],
        "m2_invariants": [{"id": "INV_SG1", "expr": "all_in({}, zone)", "check_at": "final"}],
    }


class ArtifactAdapterUnitTests(unittest.TestCase):
    def test_m1_nodes_edges_map_to_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "m1.json",
                {
                    "nodes": [{"id": "obj_a", "class": "ladle"}],
                    "edges": [{"from": "obj_a", "to": "obj_b", "type": "near"}],
                },
            )
            _write_json(root / "m2.json", _minimal_m2())
            state = FailureContextArtifactAdapter(root).build_pipeline_state(
                failed_subgoal_id="SG1_d1"
            )
            self.assertEqual(state["scene"]["nodes"][0]["id"], "obj_a")
            self.assertEqual(state["scene"]["relations"][0]["type"], "near")
            self.assertEqual(state["scene"]["object_states"], {})

    def test_m2_detail_subgoal_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "m2.json", _minimal_m2())
            state = FailureContextArtifactAdapter(root).build_pipeline_state(
                failed_subgoal_id="SG1_d1",
                failure_id="fail-fixed",
            )
            subgoal = state["subgoal"]
            self.assertEqual(subgoal["subgoal_id"], "SG1_d1")
            self.assertEqual(subgoal["parent_subgoal_id"], "SG1")
            self.assertEqual(subgoal["detail_id"], "SG1_d1")
            self.assertEqual(subgoal["description"], "pick the ladle")
            self.assertEqual(subgoal["action_type"], "acquire")
            self.assertEqual(subgoal["target_object_ids"], ["obj_block_block_0"])
            self.assertEqual(subgoal["preconditions"], ["hand_empty"])
            self.assertEqual(
                subgoal["postconditions"],
                ["holding(?tool)", "hand_empty"],
            )
            self.assertEqual(subgoal["invariants"], [])
            self.assertEqual(state["task"]["instruction"], "demo instruction")
            self.assertEqual(state["failure_id"], "fail-fixed")

    def test_selected_object_class_joins_m1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "m1.json",
                {"nodes": [{"id": "obj_ladle_ladle", "class": "ladle"}], "edges": []},
            )
            _write_json(root / "m2.json", _minimal_m2())
            state = FailureContextArtifactAdapter(root).build_pipeline_state(
                failed_subgoal_id="SG1_d1"
            )
            self.assertEqual(state["subgoal"]["selected_object_id"], "obj_ladle_ladle")
            self.assertEqual(state["subgoal"]["selected_object_class"], "ladle")

    def test_m3_grounding_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "m2.json", _minimal_m2())
            _write_json(
                root / "m3.json",
                {
                    "responses": [
                        {
                            "subgoal_id": "SG1",
                            "node_id": "obj_ladle_ladle",
                            "geometry": {"length_mm": 10},
                            "material": "unknown",
                            "density_kgm3": 500,
                            "mass_kg": 0.1,
                            "mu": {"mu": 0.4},
                            "ee": {"2F": {"feasible": True}},
                            "confidence": 0.5,
                        },
                        {
                            "subgoal_id": "OTHER",
                            "node_id": "obj_other",
                            "geometry": {"length_mm": 1},
                            "confidence": 0.1,
                        },
                    ]
                },
            )
            state = FailureContextArtifactAdapter(root).build_pipeline_state(
                failed_subgoal_id="SG1_d1"
            )
            grounding = state["grounding"]
            self.assertIn("obj_ladle_ladle", grounding["geometry"])
            self.assertIn("obj_ladle_ladle", grounding["physical_properties"])
            self.assertIn("obj_ladle_ladle", grounding["ee_feasibility"])
            self.assertEqual(grounding["confidence"]["obj_ladle_ladle"], 0.5)
            self.assertNotIn("obj_other", grounding["geometry"])
            self.assertEqual(grounding["metric_relations"], {})

    def test_m4_selected_plan_success_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "m2.json", _minimal_m2())
            _write_json(
                root / "m4.json",
                {
                    "status": "SUCCESS",
                    "selected_plan": {
                        "subgoal_order": ["SG1_d1", "SG1_d2"],
                        "group_ee_assignments": {"G": "2F"},
                        "candidate_assignments": [
                            {
                                "subgoal_id": "SG1_d1",
                                "group_id": "G",
                                "ee": "2F",
                                "tool": "ladle",
                                "suitability_score": 0.91,
                            }
                        ],
                    },
                },
            )
            state = FailureContextArtifactAdapter(root).build_pipeline_state(
                failed_subgoal_id="SG1_d1"
            )
            plan = state["task_plan"]
            self.assertEqual(plan["selected_ee"], "2F")
            self.assertEqual(plan["selected_tool"], "ladle")
            self.assertEqual(plan["final_order"], ["SG1_d1", "SG1_d2"])
            self.assertEqual(plan["selection_score"], 0.91)
            self.assertEqual(plan["selection_reason"], "tool looks suitable")
            self.assertEqual(plan["ee_candidates"], ["2F", "3F"])
            self.assertEqual(plan["swap_plan"], [])

    def test_m4_selected_plan_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "m2.json", _minimal_m2())
            _write_json(
                root / "m4.json",
                {"status": "INFEASIBLE_NO_PLAN", "selected_plan": None},
            )
            state = FailureContextArtifactAdapter(root).build_pipeline_state(
                failed_subgoal_id="SG1_d1"
            )
            plan = state["task_plan"]
            self.assertIsNone(plan["selected_ee"])
            self.assertIsNone(plan["selected_tool"])
            self.assertEqual(plan["final_order"], [])
            self.assertIsNone(plan["selection_score"])

    def test_m5_planning_failure_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "m2.json", _minimal_m2())
            _write_json(
                root / "m5" / "m5_failure.json",
                [
                    {
                        "strategy_id": "SG1_d1:transition:0",
                        "failure_code": "NO_CONNECTED_SEQUENCE",
                        "detail": "evaluated 2 edges",
                        "rejected_edge_counts": {
                            "CARTESIAN_INTERMEDIATE_IK_FAILED": 2,
                            "COLLISION_MARGIN_VIOLATION": 1,
                        },
                    }
                ],
            )
            state = FailureContextArtifactAdapter(root).build_pipeline_state(
                failed_subgoal_id="SG1_d1"
            )
            motion = state["motion_plan"]
            self.assertEqual(motion["planning_status"], "FAILED")
            self.assertIsInstance(motion["planning_error"], list)
            self.assertEqual(
                motion["ik_result"]["rejected_edge_counts"][
                    "CARTESIAN_INTERMEDIATE_IK_FAILED"
                ],
                2,
            )
            self.assertEqual(
                motion["collision_result"]["rejected_edge_counts"][
                    "COLLISION_MARGIN_VIOLATION"
                ],
                1,
            )

    def test_missing_execution_report_yields_empty_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "m2.json", _minimal_m2())
            (root / "m5").mkdir()
            state = FailureContextArtifactAdapter(root).build_pipeline_state(
                failed_subgoal_id="SG1_d1"
            )
            execution = state["execution"]
            self.assertEqual(execution["executed_actions"], [])
            self.assertIsNone(execution["controller_status"])
            self.assertFalse(execution["timeout"])
            self.assertIsNone(execution["error"])
            self.assertIsNone(execution["gripper"]["command"])

    def test_optional_artifacts_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "m2.json", _minimal_m2())
            state = FailureContextArtifactAdapter(root).build_pipeline_state(
                failed_subgoal_id="SG1_d1"
            )
            self.assertEqual(state["scene"]["nodes"], [])
            self.assertEqual(state["grounding"]["geometry"], {})
            self.assertIsNone(state["task_plan"]["selected_ee"])
            self.assertIsNone(state["motion_plan"]["planning_status"])
            self.assertIsNone(state["m5_result"])
            self.assertIsNone(state["observation"]["before_image"])

    def test_missing_m2_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ArtifactAdapterError):
                FailureContextArtifactAdapter(root).build_pipeline_state(
                    failed_subgoal_id="SG1_d1"
                )

    def test_unknown_subgoal_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "m2.json", _minimal_m2())
            with self.assertRaises(ArtifactAdapterError):
                FailureContextArtifactAdapter(root).build_pipeline_state(
                    failed_subgoal_id="MISSING"
                )

    def test_builder_schema_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "m2.json", _minimal_m2())
            context = FailureContextArtifactAdapter(root).build_failure_context(
                failed_subgoal_id="SG1_d1",
                failure_id="fail-schema",
            )
            expected = empty_failure_context()
            self.assertEqual(set(context.keys()), set(expected.keys()))
            for key, value in expected.items():
                if isinstance(value, dict):
                    self.assertEqual(set(context[key].keys()), set(value.keys()))

    def test_execution_report_explicit_subgoal_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "m2.json", _minimal_m2())
            reports = root / "m5" / "simulation" / "reports"
            _write_json(
                reports / "0001-plan-SG1_d1-report.json",
                {
                    "report_id": "report:SG1_d1",
                    "run_id": "run-1",
                    "plan_id": "plan-SG1_d1",
                    "status": "TIMEOUT",
                    "executed_events": [
                        {"event_id": "e1", "status": "SUCCESS"}
                    ],
                    "final_robot_state": {"gripper": {"command": 1.0}},
                    "failure": {
                        "code": "SIMULATION_TIMEOUT",
                        "message": "exceeded max_duration_s",
                        "observed": {"contact_force_n": 1.2, "contact_count": 2},
                    },
                    "metadata": {},
                },
            )
            state = FailureContextArtifactAdapter(root).build_pipeline_state(
                failed_subgoal_id="SG1_d1"
            )
            execution = state["execution"]
            self.assertEqual(execution["controller_status"], "TIMEOUT")
            self.assertTrue(execution["timeout"])
            self.assertEqual(len(execution["executed_actions"]), 1)
            self.assertEqual(execution["gripper"]["command"], 1.0)
            self.assertEqual(execution["gripper"]["force"], 1.2)
            self.assertTrue(execution["gripper"]["contact_detected"])
            self.assertIn("SIMULATION_TIMEOUT", execution["error"])

    def test_ambiguous_execution_reports_left_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "m2.json", _minimal_m2())
            reports = root / "m5" / "simulation" / "reports"
            for name in ("a-SG1_d1.json", "b-SG1_d1.json"):
                _write_json(
                    reports / name,
                    {
                        "report_id": f"report:{name}",
                        "status": "FAILED",
                        "executed_events": [],
                        "metadata": {},
                    },
                )
            state = FailureContextArtifactAdapter(root).build_pipeline_state(
                failed_subgoal_id="SG1_d1"
            )
            self.assertIsNone(state["execution"]["controller_status"])


class SubgoalResultArtifactTests(unittest.TestCase):
    def test_subgoal_result_fail_loads_into_m5_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "m2.json", _minimal_m2())
            _write_json(
                root / "m5" / "subgoal_result.json",
                {
                    "subgoal_id": "SG1_d1",
                    "status": "FAIL",
                    "phase": "planning",
                    "failure_code": "IK_FAILURE",
                    "detail": "No IK solution",
                },
            )
            context = FailureContextArtifactAdapter(root).build_failure_context(
                failed_subgoal_id="SG1_d1"
            )
            m5_result = context["m5_result"]
            self.assertEqual(m5_result["subgoal_id"], "SG1_d1")
            self.assertEqual(m5_result["status"], "FAIL")
            self.assertEqual(m5_result["phase"], "planning")
            self.assertEqual(m5_result["failure_code"], "IK_FAILURE")
            self.assertEqual(m5_result["detail"], "No IK solution")
            serialized = json.dumps(context)
            self.assertNotIn("verification", context)
            self.assertNotIn('"expected_state"', serialized)
            self.assertNotIn('"observed_state"', serialized)
            self.assertNotIn('"violated_predicates"', serialized)

    def test_subgoal_result_success_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "m2.json", _minimal_m2())
            _write_json(
                root / "m5" / "subgoal_result.json",
                {
                    "subgoal_id": "SG1_d1",
                    "status": "SUCCESS",
                    "phase": "execution",
                    "failure_code": None,
                    "detail": None,
                },
            )
            context = FailureContextArtifactAdapter(root).build_failure_context(
                failed_subgoal_id="SG1_d1"
            )
            self.assertEqual(context["m5_result"]["status"], "SUCCESS")
            self.assertEqual(context["m5_result"]["phase"], "execution")
            self.assertIsNone(context["m5_result"]["failure_code"])
            self.assertIsNone(context["m5_result"]["detail"])

    def test_subgoal_result_subgoal_id_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "m2.json", _minimal_m2())
            _write_json(
                root / "m5" / "subgoal_result.json",
                {
                    "subgoal_id": "OTHER_SG",
                    "status": "FAIL",
                    "phase": "planning",
                    "failure_code": "IK_FAILURE",
                    "detail": "wrong id",
                },
            )
            with self.assertRaises(ArtifactAdapterError) as ctx:
                FailureContextArtifactAdapter(root).build_pipeline_state(
                    failed_subgoal_id="SG1_d1"
                )
            self.assertIn("does not match", str(ctx.exception))

    def test_subgoal_result_missing_is_backward_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "m2.json", _minimal_m2())
            state = FailureContextArtifactAdapter(root).build_pipeline_state(
                failed_subgoal_id="SG1_d1"
            )
            context = FailureContextBuilder().build(state)
            self.assertIsNone(state["m5_result"])
            self.assertIsNone(context["m5_result"])
            self.assertNotIn("verification", context)


@unittest.skipUnless(C1_1_OUTPUT.is_dir(), "output/c1_1 is required for smoke test")
class ArtifactAdapterC11SmokeTests(unittest.TestCase):
    def test_c1_1_smoke_and_memory_unchanged(self):
        before = (
            hashlib.sha256(MEMORY_PATH.read_bytes()).hexdigest()
            if MEMORY_PATH.is_file()
            else None
        )
        context = FailureContextArtifactAdapter(C1_1_OUTPUT).build_failure_context(
            failed_subgoal_id="SG1_s1_d1",
            failure_id="fail_c1_1_SG1_s1_d1",
        )
        self.assertEqual(context["task"]["task_id"], "c1_1")
        self.assertIsInstance(context["task"]["instruction"], str)
        self.assertEqual(context["subgoal"]["subgoal_id"], "SG1_s1_d1")
        self.assertEqual(context["subgoal"]["parent_subgoal_id"], "SG1_s1")
        self.assertEqual(context["subgoal"]["detail_id"], "SG1_s1_d1")
        self.assertEqual(context["subgoal"]["action_type"], "acquire")
        self.assertIsInstance(context["subgoal"]["description"], str)
        self.assertGreater(len(context["scene"]["nodes"]), 0)
        self.assertGreater(len(context["scene"]["relations"]), 0)
        self.assertTrue(context["grounding"]["geometry"])
        self.assertEqual(context["task_plan"]["selected_ee"], "2F")
        self.assertEqual(context["motion_plan"]["planning_status"], "FAILED")
        self.assertIsInstance(context["motion_plan"]["planning_error"], list)
        self.assertIsNone(context["execution"]["controller_status"])
        self.assertEqual(
            context["m5_result"],
            {
                "subgoal_id": "SG1_s1_d1",
                "status": "FAIL",
                "phase": "planning",
                "failure_code": "IK_FAILURE",
                "detail": "Mock planning failure for M6 integration test",
            },
        )
        expected_keys = set(empty_failure_context().keys())
        self.assertEqual(set(context.keys()), expected_keys)
        self.assertNotIn("verification", context)
        serialized = json.dumps(context)
        self.assertNotIn("expected_state", serialized)
        self.assertNotIn("observed_state", serialized)
        self.assertNotIn("violated_predicates", serialized)
        if before is not None:
            after = hashlib.sha256(MEMORY_PATH.read_bytes()).hexdigest()
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
