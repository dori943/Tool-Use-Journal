"""Unit tests for M6 E2E runner (no live OpenAI calls)."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tuj.m6_diagnosis.e2e_runner import (
    M6E2EError,
    discover_failed_subgoal_candidates,
    has_m5_failure_artifact,
    list_failure_task_availability,
    match_subgoal_from_strategy_id,
    resolve_failed_subgoal_id,
    resolve_task_output_dir,
    run_m6_e2e,
    sha256_file,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _minimal_task_tree(
    root: Path,
    task_id: str = "demo_task",
    *,
    with_failure: bool = True,
    strategy_ids: list[str] | None = None,
) -> Path:
    task_dir = root / "output" / task_id
    m2 = {
        "task": "demo instruction",
        "m2_subgoals": [
            {
                "subgoal_id": "SG1_s1",
                "goal": "Acquire tool then transport",
                "target_ids": ["obj_block"],
                "selected_tool_id": "obj_tool",
                "selection_reason": "demo",
                "selection_evidence": [
                    {"node": "obj_tool", "feasible_ees": ["2F", "3F"]}
                ],
                "details": [
                    {
                        "detail_id": "SG1_s1_d1",
                        "action_type": "acquire",
                        "pre": [{"id": "p0", "expr": "hand_empty"}],
                        "establish": [{"id": "e0", "expr": "holding(?tool)"}],
                        "destroy": [{"id": "d0", "expr": "hand_empty"}],
                    },
                    {
                        "detail_id": "SG1_s1_d2",
                        "action_type": "transport",
                        "pre": [],
                        "establish": [],
                        "destroy": [],
                    },
                ],
            }
        ],
        "m2_invariants": [],
    }
    _write_json(task_dir / "m2.json", m2)
    _write_json(
        task_dir / "m1.json",
        {
            "nodes": [
                {"id": "obj_tool", "class": "tool"},
                {"id": "obj_block", "class": "block"},
            ],
            "edges": [],
        },
    )
    _write_json(task_dir / "m3.json", {"responses": []})
    _write_json(
        task_dir / "m4.json",
        {
            "selected_plan": {
                "subgoal_order": ["SG1_s1_d1"],
                "candidate_assignments": [
                    {
                        "subgoal_id": "SG1_s1_d1",
                        "ee": "2F",
                        "tool": "tool",
                    }
                ],
                "group_ee_assignments": {},
            }
        },
    )
    if with_failure:
        strategies = strategy_ids or ["SG1_s1_d1:transition:0:ee-attach:bare->2F"]
        _write_json(
            task_dir / "m5" / "m5_failure.json",
            [
                {
                    "strategy_id": sid,
                    "failure_code": "NO_CONNECTED_SEQUENCE",
                    "detail": "demo failure",
                    "rejected_edge_counts": {"CARTESIAN_INTERMEDIATE_IK_FAILED": 3},
                }
                for sid in strategies
            ],
        )
    return task_dir


def _memory_payload(experiences=None) -> dict:
    return {
        "failure_recovery_experience": {
            "experiences": experiences or []
        }
    }


class ResolvePathTests(unittest.TestCase):
    def test_task_output_path_resolution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task_dir = _minimal_task_tree(root, "c9_9")
            resolved = resolve_task_output_dir(
                "c9_9",
                output_root=root / "output",
                project_root=root,
            )
            self.assertEqual(resolved, task_dir.resolve())

    def test_missing_task_dir_raises(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "output").mkdir()
            with self.assertRaises(M6E2EError):
                resolve_task_output_dir("missing", output_root=root / "output", project_root=root)


class SubgoalSelectionTests(unittest.TestCase):
    def test_explicit_subgoal_wins(self):
        with tempfile.TemporaryDirectory() as temp:
            task_dir = _minimal_task_tree(Path(temp))
            self.assertEqual(
                resolve_failed_subgoal_id(task_dir, "SG1_s1_d2"),
                "SG1_s1_d2",
            )

    def test_auto_detect_unique_failure_subgoal(self):
        with tempfile.TemporaryDirectory() as temp:
            task_dir = _minimal_task_tree(Path(temp))
            self.assertTrue(has_m5_failure_artifact(task_dir))
            self.assertEqual(discover_failed_subgoal_candidates(task_dir), ["SG1_s1_d1"])
            self.assertEqual(resolve_failed_subgoal_id(task_dir, None), "SG1_s1_d1")

    def test_ambiguous_subgoal_raises(self):
        with tempfile.TemporaryDirectory() as temp:
            task_dir = _minimal_task_tree(
                Path(temp),
                strategy_ids=[
                    "SG1_s1_d1:transition:0:x",
                    "SG1_s1_d2:transition:0:y",
                ],
            )
            with self.assertRaises(M6E2EError) as ctx:
                resolve_failed_subgoal_id(task_dir, None)
            message = str(ctx.exception)
            self.assertIn("Ambiguous", message)
            self.assertIn("SG1_s1_d1", message)
            self.assertIn("SG1_s1_d2", message)

    def test_no_failure_artifact_raises(self):
        with tempfile.TemporaryDirectory() as temp:
            task_dir = _minimal_task_tree(Path(temp), with_failure=False)
            with self.assertRaises(M6E2EError) as ctx:
                resolve_failed_subgoal_id(task_dir, None)
            self.assertIn("No failure artifact available", str(ctx.exception))

    def test_strategy_prefix_match(self):
        known = ["SG1_s1", "SG1_s1_d1"]
        self.assertEqual(
            match_subgoal_from_strategy_id("SG1_s1_d1:transition:0:x", known),
            "SG1_s1_d1",
        )


class E2ERunTests(unittest.TestCase):
    def _prepare(self, root: Path, *, experiences=None) -> tuple[Path, Path]:
        task_dir = _minimal_task_tree(root, "c1_demo")
        memory_path = root / "output" / "memory.json"
        _write_json(memory_path, _memory_payload(experiences))
        return task_dir, memory_path

    def test_retrieval_empty_continues_diagnosis_guided(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, memory_path = self._prepare(root)
            before = sha256_file(memory_path)

            result = run_m6_e2e(
                task_id="c1_demo",
                subgoal_id="SG1_s1_d1",
                output_root=root / "output",
                project_root=root,
                memory_path=memory_path,
                diagnoser_backend="mock",
                recovery_backend="mock",
                save=True,
                print_summary=False,
            )

            self.assertEqual(result["summary"]["retrieval"]["candidate_count"], 0)
            self.assertEqual(result["recovery"]["decision_mode"], "DIAGNOSIS_GUIDED")
            self.assertEqual(result["summary"]["selection"]["selected_count"], 0)
            self.assertIn("recovery_type", result["recovery"]["action"])
            self.assertNotIn("action_type", result["recovery"]["action"])
            self.assertFalse(result["selective_reexecution_performed"])
            self.assertFalse(result["memory_updated"])
            self.assertEqual(sha256_file(memory_path), before)

            m6_dir = Path(result["m6_dir"])
            for name in (
                "failure_context.json",
                "retrieval_query.json",
                "retrieved_experiences.json",
                "diagnosis.json",
                "diagnosis_aware_selection.json",
                "recovery.json",
                "recovery_request.json",
                "e2e_summary.json",
            ):
                self.assertTrue((m6_dir / name).is_file(), name)
            summary = json.loads((m6_dir / "e2e_summary.json").read_text(encoding="utf-8"))
            self.assertIn("dispatch", summary)
            self.assertEqual(summary["dispatch"]["target_module"], "M5")
            self.assertFalse(result["dispatch"]["module_executed"])

    def test_experience_guided_when_selection_matches(self):
        experience = {
            "experience_id": "exp-match",
            "context_signature": {
                "action_type": "acquire",
                "target": {"object_id": "obj_tool", "object_class": "tool"},
                "violated_predicates": [],
                "selected_ee": "2F",
                "selected_tool": "tool",
                "execution_signature": {
                    "motion_planning_status": "FAILED",
                    "controller_status": None,
                },
            },
            "diagnosis_summary": {
                "failure_type": "PLANNING",
                "failure_cause": {"code": "INVALID_APPROACH", "description": "seed"},
                "affected_module": "M5",
            },
            "recovery_summary": {
                "recovery_category": "REPLAN_MOTION",
                "action": {"recovery_type": "CHANGE_APPROACH"},
                "routing": {"restart_from": "M5", "rerun_modules": ["M5"], "invalidate": []},
                "outcome": {"status": "NOT_EXECUTED", "verification_result": None},
            },
            "metadata": {"source": "test"},
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, memory_path = self._prepare(root, experiences=[experience])

            # Force retrieval to return the matching experience regardless of
            # similarity threshold edge cases in the fixture.
            with mock.patch(
                "tuj.m6_diagnosis.e2e_runner.MemoryAdapter.retrieve_experiences",
                return_value=[
                    {
                        "experience": experience,
                        "context_similarity": 1.0,
                        "comparison_coverage": 0.5,
                        "candidate_valid": True,
                        "candidate_validity": {"valid": True, "reason": "ok"},
                    }
                ],
            ):
                result = run_m6_e2e(
                    task_id="c1_demo",
                    subgoal_id="SG1_s1_d1",
                    output_root=root / "output",
                    project_root=root,
                    memory_path=memory_path,
                    diagnoser_backend="mock",
                    recovery_backend="mock",
                    save=False,
                    print_summary=False,
                )

            self.assertEqual(result["recovery"]["decision_mode"], "EXPERIENCE_GUIDED")
            self.assertEqual(result["summary"]["selection"]["selected_count"], 1)
            self.assertEqual(
                result["summary"]["selection"]["selected_experience_ids"],
                ["exp-match"],
            )

    def test_openai_backend_requires_api_key(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, memory_path = self._prepare(root)
            with mock.patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(M6E2EError) as ctx:
                    run_m6_e2e(
                        task_id="c1_demo",
                        subgoal_id="SG1_s1_d1",
                        output_root=root / "output",
                        project_root=root,
                        memory_path=memory_path,
                        diagnoser_backend="openai",
                        recovery_backend="mock",
                        save=False,
                        print_summary=False,
                    )
            self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_list_failure_task_availability(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _minimal_task_tree(root, "with_fail", with_failure=True)
            _minimal_task_tree(root, "no_fail", with_failure=False)
            rows = list_failure_task_availability(
                output_root=root / "output",
                project_root=root,
            )
            by_id = {row.task_id: row for row in rows}
            self.assertTrue(by_id["with_fail"].has_failure_artifact)
            self.assertFalse(by_id["no_fail"].has_failure_artifact)


class ShaHelperTests(unittest.TestCase):
    def test_sha256_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "x.bin"
            path.write_bytes(b"abc")
            self.assertEqual(sha256_file(path), hashlib.sha256(b"abc").hexdigest())


if __name__ == "__main__":
    unittest.main()
