"""Build M6 pipeline_state dicts from on-disk M1–M5 task artifacts.

This adapter is the only place that interprets module-specific JSON shapes.
Downstream M6 code continues to consume the canonical Failure Context schema
via ``FailureContextBuilder``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any


class ArtifactAdapterError(ValueError):
    """Raised when required artifacts or identifiers cannot be resolved."""


def _default_failure_id(output_dir: Path, failed_subgoal_id: str) -> str:
    return f"fail_{output_dir.name}_{failed_subgoal_id}"


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _expr_list(entries: Any) -> list[str]:
    exprs: list[str] = []
    for item in _as_list(entries):
        if isinstance(item, Mapping):
            expr = item.get("expr")
            if isinstance(expr, str):
                exprs.append(expr)
        elif isinstance(item, str):
            exprs.append(item)
    return exprs


class FailureContextArtifactAdapter:
    """Load ``output/<task>/`` module JSON files into an M6 pipeline_state."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        failure_id_factory: Callable[[Path, str], str] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.failure_id_factory = failure_id_factory or _default_failure_id

    def build_pipeline_state(
        self,
        *,
        failed_subgoal_id: str,
        failure_source: str | None = None,
        failure_id: str | None = None,
    ) -> dict[str, Any]:
        """Assemble a pipeline_state dict for ``FailureContextBuilder``.

        ``failed_subgoal_id`` is required. Auto-picking a subgoal from artifacts
        is intentionally unsupported.
        """
        del failure_source  # reserved for callers; unused for mapping rules

        if not isinstance(failed_subgoal_id, str) or not failed_subgoal_id.strip():
            raise ArtifactAdapterError("failed_subgoal_id is required")
        failed_subgoal_id = failed_subgoal_id.strip()

        if not self.output_dir.is_dir():
            raise ArtifactAdapterError(f"output_dir does not exist: {self.output_dir}")

        m2 = _read_json(self.output_dir / "m2.json")
        if not isinstance(m2, Mapping):
            raise ArtifactAdapterError(
                f"required artifact missing or invalid: {self.output_dir / 'm2.json'}"
            )

        m1 = _read_json(self.output_dir / "m1.json")
        m3 = _read_json(self.output_dir / "m3.json")
        m4 = _read_json(self.output_dir / "m4.json")

        parent_subgoal, detail = self._resolve_subgoal(m2, failed_subgoal_id)
        node_class_by_id = self._node_class_index(m1 if isinstance(m1, Mapping) else None)

        resolved_failure_id = (
            failure_id
            if failure_id is not None
            else self.failure_id_factory(self.output_dir, failed_subgoal_id)
        )

        return {
            "failure_id": resolved_failure_id,
            "task": self._map_task(m2),
            "subgoal": self._map_subgoal(
                parent_subgoal,
                detail,
                failed_subgoal_id=failed_subgoal_id,
                node_class_by_id=node_class_by_id,
            ),
            "scene": self._map_scene(m1 if isinstance(m1, Mapping) else None),
            "grounding": self._map_grounding(
                m3 if isinstance(m3, Mapping) else None,
                failed_subgoal_id=failed_subgoal_id,
                parent_subgoal_id=parent_subgoal.get("subgoal_id"),
            ),
            "task_plan": self._map_task_plan(
                m4 if isinstance(m4, Mapping) else None,
                parent_subgoal,
                failed_subgoal_id=failed_subgoal_id,
            ),
            "motion_plan": self._map_motion_plan(failed_subgoal_id=failed_subgoal_id),
            "execution": self._map_execution(failed_subgoal_id=failed_subgoal_id),
            "m5_result": self._map_m5_result(failed_subgoal_id=failed_subgoal_id),
            "observation": {
                "before_image": None,
                "after_image": None,
                "before_scene": None,
                "after_scene": None,
            },
            "history": {
                "retry_count": 0,
                "previous_diagnoses": [],
                "previous_recoveries": [],
                "previous_outcomes": [],
            },
        }

    def build_failure_context(
        self,
        *,
        failed_subgoal_id: str,
        failure_source: str | None = None,
        failure_id: str | None = None,
        context_builder=None,
    ) -> dict[str, Any]:
        from .failure_context import FailureContextBuilder

        builder = context_builder or FailureContextBuilder()
        pipeline_state = self.build_pipeline_state(
            failed_subgoal_id=failed_subgoal_id,
            failure_source=failure_source,
            failure_id=failure_id,
        )
        return builder.build(pipeline_state)

    # ------------------------------------------------------------------ #
    # Subgoal resolution
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_subgoal(
        m2: Mapping[str, Any],
        failed_subgoal_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        for raw in _as_list(m2.get("m2_subgoals")):
            if not isinstance(raw, Mapping):
                continue
            subgoal = dict(raw)
            if subgoal.get("subgoal_id") == failed_subgoal_id:
                return subgoal, None
            for detail_raw in _as_list(subgoal.get("details")):
                if not isinstance(detail_raw, Mapping):
                    continue
                detail = dict(detail_raw)
                if detail.get("detail_id") == failed_subgoal_id:
                    return subgoal, detail
        raise ArtifactAdapterError(
            f"failed_subgoal_id {failed_subgoal_id!r} not found in m2.json "
            f"m2_subgoals[].subgoal_id or details[].detail_id"
        )

    @staticmethod
    def _node_class_index(m1: Mapping[str, Any] | None) -> dict[str, str]:
        index: dict[str, str] = {}
        if m1 is None:
            return index
        for node in _as_list(m1.get("nodes")):
            if not isinstance(node, Mapping):
                continue
            node_id = node.get("id")
            node_class = node.get("class")
            if isinstance(node_id, str) and isinstance(node_class, str):
                index[node_id] = node_class
        return index

    # ------------------------------------------------------------------ #
    # Section mappers
    # ------------------------------------------------------------------ #

    def _map_task(self, m2: Mapping[str, Any]) -> dict[str, Any]:
        instruction = m2.get("task")
        return {
            "task_id": self.output_dir.name,
            "instruction": instruction if isinstance(instruction, str) else None,
        }

    def _map_subgoal(
        self,
        parent_subgoal: Mapping[str, Any],
        detail: Mapping[str, Any] | None,
        *,
        failed_subgoal_id: str,
        node_class_by_id: Mapping[str, str],
    ) -> dict[str, Any]:
        # Parent fields always come from the resolved M2 parent (never invent).
        parent_id = parent_subgoal.get("subgoal_id")
        if not isinstance(parent_id, str) or not parent_id.strip():
            parent_id = None
        else:
            parent_id = parent_id.strip()

        description = parent_subgoal.get("goal")
        if not isinstance(description, str):
            description = None

        action_type = None
        detail_id: str | None = None
        preconditions: list[str] = []
        postconditions: list[str] = []
        if detail is not None:
            raw_detail_id = detail.get("detail_id")
            if isinstance(raw_detail_id, str) and raw_detail_id.strip():
                detail_id = raw_detail_id.strip()
            raw_action = detail.get("action_type")
            action_type = raw_action if isinstance(raw_action, str) else None
            preconditions = _expr_list(detail.get("pre"))
            postconditions = _expr_list(detail.get("establish")) + _expr_list(
                detail.get("destroy")
            )
        else:
            # Failed id matched the parent itself — no M2 detail row.
            detail_id = None
            details = [
                item
                for item in _as_list(parent_subgoal.get("details"))
                if isinstance(item, Mapping)
            ]
            if len(details) == 1:
                only = details[0]
                raw_action = only.get("action_type")
                action_type = raw_action if isinstance(raw_action, str) else None
                preconditions = _expr_list(only.get("pre"))
                postconditions = _expr_list(only.get("establish")) + _expr_list(
                    only.get("destroy")
                )

        target_ids = parent_subgoal.get("target_ids")
        target_object_ids = (
            list(target_ids) if isinstance(target_ids, list) else []
        )

        selected_object_id = parent_subgoal.get("selected_tool_id")
        if not isinstance(selected_object_id, str):
            selected_object_id = None

        selected_object_class = None
        if selected_object_id is not None:
            selected_object_class = node_class_by_id.get(selected_object_id)

        return {
            # Lifecycle / M5 execution unit (detail id when failing a detail).
            "subgoal_id": failed_subgoal_id,
            "parent_subgoal_id": parent_id,
            "detail_id": detail_id,
            "description": description,
            "action_type": action_type,
            "target_object_ids": target_object_ids,
            "selected_object_id": selected_object_id,
            "selected_object_class": selected_object_class,
            "preconditions": preconditions,
            "postconditions": postconditions,
            # m2_invariants are task-level records, not subgoal-scoped fields.
            "invariants": [],
        }

    @staticmethod
    def _map_scene(m1: Mapping[str, Any] | None) -> dict[str, Any]:
        if m1 is None:
            return {"nodes": [], "relations": [], "object_states": {}}
        nodes = deepcopy(_as_list(m1.get("nodes")))
        relations = deepcopy(_as_list(m1.get("edges")))
        return {
            "nodes": nodes,
            "relations": relations,
            "object_states": {},
        }

    @staticmethod
    def _response_matches_subgoal(
        response_subgoal_id: Any,
        *,
        failed_subgoal_id: str,
        parent_subgoal_id: Any,
    ) -> bool:
        if not isinstance(response_subgoal_id, str):
            return False
        if response_subgoal_id == failed_subgoal_id:
            return True
        if (
            isinstance(parent_subgoal_id, str)
            and response_subgoal_id == parent_subgoal_id
        ):
            return True
        # Detail ids are parent_id + "_" + suffix (e.g. SG1_s1_d1 under SG1_s1).
        if failed_subgoal_id.startswith(response_subgoal_id + "_"):
            return True
        return False

    def _map_grounding(
        self,
        m3: Mapping[str, Any] | None,
        *,
        failed_subgoal_id: str,
        parent_subgoal_id: Any,
    ) -> dict[str, Any]:
        physical_properties: dict[str, Any] = {}
        geometry: dict[str, Any] = {}
        ee_feasibility: dict[str, Any] = {}
        confidence: dict[str, Any] = {}

        if m3 is None:
            return {
                "physical_properties": physical_properties,
                "geometry": geometry,
                "metric_relations": {},
                "ee_feasibility": ee_feasibility,
                "confidence": confidence,
            }

        for response in _as_list(m3.get("responses")):
            if not isinstance(response, Mapping):
                continue
            if not self._response_matches_subgoal(
                response.get("subgoal_id"),
                failed_subgoal_id=failed_subgoal_id,
                parent_subgoal_id=parent_subgoal_id,
            ):
                continue
            node_id = response.get("node_id")
            if not isinstance(node_id, str):
                continue

            props: dict[str, Any] = {}
            for key in ("material", "density_kgm3", "mass_kg", "youngs_gpa", "mu"):
                if key in response:
                    props[key] = deepcopy(response[key])
            if props:
                physical_properties[node_id] = props

            if isinstance(response.get("geometry"), Mapping):
                geometry[node_id] = deepcopy(response["geometry"])
            if isinstance(response.get("ee"), Mapping):
                ee_feasibility[node_id] = deepcopy(response["ee"])
            if "confidence" in response:
                confidence[node_id] = deepcopy(response["confidence"])

        return {
            "physical_properties": physical_properties,
            "geometry": geometry,
            "metric_relations": {},
            "ee_feasibility": ee_feasibility,
            "confidence": confidence,
        }

    def _map_task_plan(
        self,
        m4: Mapping[str, Any] | None,
        parent_subgoal: Mapping[str, Any],
        *,
        failed_subgoal_id: str,
    ) -> dict[str, Any]:
        # selection_reason originates from M2 subgoal records (not M4).
        selection_reason = parent_subgoal.get("selection_reason")
        if not isinstance(selection_reason, str):
            selection_reason = None

        ee_candidates: list[Any] = []
        for evidence in _as_list(parent_subgoal.get("selection_evidence")):
            if not isinstance(evidence, Mapping):
                continue
            feasible = evidence.get("feasible_ees")
            if isinstance(feasible, list):
                for item in feasible:
                    if item not in ee_candidates:
                        ee_candidates.append(item)

        task_plan: dict[str, Any] = {
            "selected_ee": None,
            "selected_tool": None,
            "ee_candidates": ee_candidates,
            "selection_score": None,
            "selection_reason": selection_reason,
            "final_order": [],
            "swap_plan": [],
        }
        if m4 is None:
            return task_plan

        selected_plan = m4.get("selected_plan")
        if not isinstance(selected_plan, Mapping):
            return task_plan

        order = selected_plan.get("subgoal_order")
        if isinstance(order, list):
            task_plan["final_order"] = list(order)

        assignment = None
        for item in _as_list(selected_plan.get("candidate_assignments")):
            if isinstance(item, Mapping) and item.get("subgoal_id") == failed_subgoal_id:
                assignment = item
                break

        group_ee = _as_mapping(selected_plan.get("group_ee_assignments"))
        if assignment is not None:
            ee = assignment.get("ee")
            tool = assignment.get("tool")
            task_plan["selected_ee"] = ee if isinstance(ee, str) else None
            task_plan["selected_tool"] = tool if isinstance(tool, str) else None
            if task_plan["selected_ee"] is None:
                group_id = assignment.get("group_id")
                if isinstance(group_id, str) and isinstance(group_ee.get(group_id), str):
                    task_plan["selected_ee"] = group_ee[group_id]

            score = assignment.get("suitability_score")
            if not isinstance(score, (int, float)):
                suitability = assignment.get("suitability")
                if isinstance(suitability, Mapping):
                    score = suitability.get("overall_suitability")
                else:
                    params = _as_mapping(assignment.get("action_parameters"))
                    suitability = _as_mapping(params.get("suitability"))
                    score = suitability.get("overall_suitability")
            if isinstance(score, (int, float)):
                task_plan["selection_score"] = float(score)
        elif len(group_ee) == 1:
            only_ee = next(iter(group_ee.values()))
            if isinstance(only_ee, str):
                task_plan["selected_ee"] = only_ee

        return task_plan

    def _map_motion_plan(self, *, failed_subgoal_id: str) -> dict[str, Any]:
        motion_plan: dict[str, Any] = {
            "keyframes": [],
            "ik_result": None,
            "collision_result": None,
            "approach_pose": None,
            "trajectory": None,
            "planning_status": None,
            "planning_error": None,
        }
        m5_dir = self.output_dir / "m5"
        if not m5_dir.is_dir():
            return motion_plan

        summary = _read_json(m5_dir / "m5_summary.json")
        if isinstance(summary, Mapping):
            status = summary.get("planning_status")
            if isinstance(status, str):
                motion_plan["planning_status"] = status
            elif isinstance(summary.get("status"), str) and summary.get(
                "planning_status"
            ) is None:
                # Prefer explicit planning_status; fall back only when absent.
                if summary["status"] in {"SUCCESS", "FAILED"}:
                    motion_plan["planning_status"] = summary["status"]

        failures = _read_json(m5_dir / "m5_failure.json")
        if isinstance(failures, list) and failures:
            matched = [
                item
                for item in failures
                if isinstance(item, Mapping)
                and failed_subgoal_id in str(item.get("strategy_id") or "")
            ]
            chosen = matched if matched else [
                item for item in failures if isinstance(item, Mapping)
            ]
            if chosen:
                motion_plan["planning_status"] = "FAILED"
                motion_plan["planning_error"] = deepcopy(chosen)
                ik_counts: dict[str, int] = {}
                collision_counts: dict[str, int] = {}
                for item in chosen:
                    rejected = _as_mapping(item.get("rejected_edge_counts"))
                    for code, count in rejected.items():
                        if not isinstance(count, int):
                            continue
                        code_text = str(code)
                        if "IK" in code_text or "SINGULARITY" in code_text:
                            ik_counts[code_text] = ik_counts.get(code_text, 0) + count
                        if "COLLISION" in code_text:
                            collision_counts[code_text] = (
                                collision_counts.get(code_text, 0) + count
                            )
                if ik_counts:
                    motion_plan["ik_result"] = {"rejected_edge_counts": ik_counts}
                if collision_counts:
                    motion_plan["collision_result"] = {
                        "rejected_edge_counts": collision_counts
                    }

        return motion_plan

    def _map_execution(self, *, failed_subgoal_id: str) -> dict[str, Any]:
        execution: dict[str, Any] = {
            "executed_actions": [],
            "controller_status": None,
            "gripper": {
                "command": None,
                "position": None,
                "contact_detected": None,
                "force": None,
            },
            "timeout": False,
            "error": None,
        }
        m5_dir = self.output_dir / "m5"
        reports_dir = m5_dir / "simulation" / "reports"
        report = self._select_execution_report(reports_dir, failed_subgoal_id)
        if report is None:
            # Scripted grasp result.json is only used when its object/subgoal id
            # matches explicitly; otherwise leave execution empty.
            scripted = self._select_scripted_result(m5_dir, failed_subgoal_id)
            if scripted is None:
                return execution
            status = scripted.get("status")
            if isinstance(status, str):
                execution["controller_status"] = status
            reason = scripted.get("failure_reason")
            if isinstance(reason, str):
                execution["error"] = reason
            return execution

        status = report.get("status")
        if isinstance(status, str):
            execution["controller_status"] = status
            execution["timeout"] = status == "TIMEOUT"

        events = []
        for event in _as_list(report.get("executed_events")):
            if isinstance(event, Mapping):
                events.append(deepcopy(event))
        execution["executed_actions"] = events

        robot_state = _as_mapping(report.get("final_robot_state"))
        gripper = _as_mapping(robot_state.get("gripper"))
        command = gripper.get("command")
        if isinstance(command, (int, float, str)) or command is None:
            execution["gripper"]["command"] = command

        failure = _as_mapping(report.get("failure"))
        if failure:
            code = failure.get("code")
            message = failure.get("message")
            if isinstance(code, str) and isinstance(message, str):
                execution["error"] = f"{code}: {message}"
            elif isinstance(code, str):
                execution["error"] = code
            elif isinstance(message, str):
                execution["error"] = message

            observed = _as_mapping(failure.get("observed"))
            if "contact_force_n" in observed:
                execution["gripper"]["force"] = deepcopy(observed["contact_force_n"])
            if "contact_count" in observed:
                count = observed["contact_count"]
                if isinstance(count, int):
                    execution["gripper"]["contact_detected"] = count > 0

        return execution

    def _map_m5_result(self, *, failed_subgoal_id: str) -> dict[str, Any] | None:
        """Load ``m5/subgoal_result.json`` when present; otherwise return None.

        ``subgoal_id`` must match ``failed_subgoal_id``. Mismatches fail loudly so
        a wrong subgoal outcome is never silently attached to the context.
        """
        path = self.output_dir / "m5" / "subgoal_result.json"
        if not path.is_file():
            return None

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactAdapterError(
                f"invalid or unreadable subgoal_result.json: {path}"
            ) from exc

        if not isinstance(raw, Mapping):
            raise ArtifactAdapterError(
                f"subgoal_result.json must be a JSON object: {path}"
            )

        result_subgoal_id = raw.get("subgoal_id")
        if not isinstance(result_subgoal_id, str) or not result_subgoal_id.strip():
            raise ArtifactAdapterError(
                f"subgoal_result.json missing subgoal_id: {path}"
            )
        result_subgoal_id = result_subgoal_id.strip()
        if result_subgoal_id != failed_subgoal_id:
            raise ArtifactAdapterError(
                f"subgoal_result.json subgoal_id {result_subgoal_id!r} does not "
                f"match failed_subgoal_id {failed_subgoal_id!r}"
            )

        # Preserve artifact values as-is (including nulls); do not coerce taxonomy.
        return {
            "subgoal_id": result_subgoal_id,
            "status": raw.get("status"),
            "phase": raw.get("phase"),
            "failure_code": raw.get("failure_code"),
            "detail": raw.get("detail"),
        }

    @staticmethod
    def _select_execution_report(
        reports_dir: Path,
        failed_subgoal_id: str,
    ) -> dict[str, Any] | None:
        """Pick a simulation report only when subgoal linkage is explicit.

        Safe rule: require ``failed_subgoal_id`` to appear in the report file
        name, report_id, run_id, plan_id, or metadata string fields. If no
        report qualifies, return None instead of guessing "latest".
        """
        if not reports_dir.is_dir():
            return None
        matches: list[dict[str, Any]] = []
        for path in sorted(reports_dir.glob("*.json")):
            payload = _read_json(path)
            if not isinstance(payload, Mapping):
                continue
            haystacks = [
                path.name,
                str(payload.get("report_id") or ""),
                str(payload.get("run_id") or ""),
                str(payload.get("plan_id") or ""),
                json.dumps(payload.get("metadata") or {}, ensure_ascii=False),
            ]
            if any(failed_subgoal_id in text for text in haystacks):
                matches.append(dict(payload))
        if len(matches) == 1:
            return matches[0]
        # Multiple explicit matches are ambiguous without richer metadata.
        return None

    @staticmethod
    def _select_scripted_result(
        m5_dir: Path,
        failed_subgoal_id: str,
    ) -> dict[str, Any] | None:
        if not m5_dir.is_dir():
            return None
        matches: list[dict[str, Any]] = []
        for path in sorted(m5_dir.rglob("result.json")):
            payload = _read_json(path)
            if not isinstance(payload, Mapping):
                continue
            object_id = payload.get("object_id")
            if object_id == failed_subgoal_id or failed_subgoal_id in str(path):
                matches.append(dict(payload))
        if len(matches) == 1:
            return matches[0]
        return None
