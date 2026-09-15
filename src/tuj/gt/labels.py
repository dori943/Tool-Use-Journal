"""GT feasibility + optimal EE-plan oracle for SReg.

SReg = actual_ee_switches - optimal_ee_switches

Optimal switches are produced by the *same* M4 Dijkstra planner
(:func:`tuj.m4_taskplanner.planner.plan`) with feasibility inputs replaced by
GT physical properties (MuJoCo ``get_scene_report`` mass + AABB), evaluated
with the same :func:`tuj.m1_scene.ee_rules.evaluate_ee` rules as runtime M1.

Surface RMS / seal-patch are not in the sim report; when an M1 node for the
same object exists those fields are reused so vac rules stay defined. Mass and
AABB extents always come from GT.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from tuj.m1_scene.ee_rules import evaluate_ee
from tuj.m4_taskplanner.diagnostics import PlanStatus
from tuj.m4_taskplanner.gk_adapter import build_request_from_gk
from tuj.m4_taskplanner.models import TaskPlannerRequest
from tuj.m4_taskplanner.planner import plan
from tuj.m4_taskplanner.serialization import PlanningResult

# Grasped binding roles — same contract as gk_adapter.GRASPED_ROLES.
_GRASPED_ROLES = frozenset({"?o", "?t"})


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_ee_pool(ee_pool: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mirror run_m3 / M1: flatness_tol_rms_mm → seal_rms_tol_mm for vac."""

    normalized: list[dict[str, Any]] = []
    for raw in ee_pool:
        entry = dict(raw)
        if "seal_rms_tol_mm" not in entry and "flatness_tol_rms_mm" in entry:
            entry["seal_rms_tol_mm"] = entry["flatness_tol_rms_mm"]
        normalized.append(entry)
    return normalized


def _size_to_geometry_mm(size_m: Sequence[float]) -> dict[str, Any]:
    extents = [float(v) * 1000.0 for v in size_m]
    if len(extents) < 3:
        raise ValueError(f"GT size_m must have 3 components, got {size_m!r}")
    return {
        "extents_mm": extents,
        "footprint_mm": [extents[0], extents[1]],
        "height_z_mm": extents[2],
    }


def intrinsic_from_gt(
    gt_object: Mapping[str, Any],
    *,
    m1_node: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build ee_rules intrinsic dict from a scene-report object (+ optional M1)."""

    if "mass_kg" not in gt_object:
        raise ValueError("GT object is missing mass_kg")
    if "size_m" not in gt_object:
        raise ValueError("GT object is missing size_m")

    geometry = _size_to_geometry_mm(gt_object["size_m"])
    mu: dict[str, Any] = {"mu": 0.5}
    if m1_node is not None:
        m1_geom = m1_node.get("geometry") or {}
        if isinstance(m1_geom, Mapping):
            if m1_geom.get("surface_rms_mm") is not None:
                geometry["surface_rms_mm"] = float(m1_geom["surface_rms_mm"])
            if m1_geom.get("seal_patch_rms_mm") is not None:
                geometry["seal_patch_rms_mm"] = float(m1_geom["seal_patch_rms_mm"])
            if m1_geom.get("diameter_mm") is not None:
                geometry["diameter_mm"] = float(m1_geom["diameter_mm"])
        m1_mu = m1_node.get("mu")
        if isinstance(m1_mu, Mapping) and m1_mu.get("mu") is not None:
            mu = dict(m1_mu)
    geometry.setdefault("surface_rms_mm", 0.5)

    return {
        "geometry": geometry,
        "mass_kg": float(gt_object["mass_kg"]),
        "mu": mu,
    }


def _index_m1_nodes(
    m1_payload: Mapping[str, Any],
    aliases: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    aliases = dict(aliases or {})
    index: dict[str, dict[str, Any]] = {}
    for node in m1_payload.get("nodes") or []:
        if not isinstance(node, Mapping):
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        short = aliases.get(node_id, node_id)
        record = dict(node)
        index[node_id] = record
        index[short] = record
        if node_id.startswith("obj_") and "_" in node_id[4:]:
            # obj_<class>_<instance> → instance (bread_a)
            instance = node_id.split("_", 2)[-1] if node_id.count("_") >= 2 else short
            # Prefer full alias short id when present.
            index.setdefault(instance, record)
    return index


def build_gt_object_feasibility(
    gt_scene: Mapping[str, Any],
    *,
    ee_pool: Sequence[Mapping[str, Any]],
    m1_payload: Mapping[str, Any] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Map object id → {ee_id: evaluate_ee(...), feasible_ees: [...]}."""

    pool = normalize_ee_pool(ee_pool)
    m1_index = _index_m1_nodes(m1_payload or {}, aliases)
    result: dict[str, dict[str, Any]] = {}
    for object_id, record in gt_scene.items():
        if not isinstance(object_id, str) or object_id.startswith("_"):
            continue
        if not isinstance(record, Mapping) or "mass_kg" not in record:
            continue
        intrinsic = intrinsic_from_gt(record, m1_node=m1_index.get(object_id))
        per_ee: dict[str, Any] = {}
        feasible: list[str] = []
        for ee_spec in pool:
            ee_id = str(ee_spec["ee_id"])
            verdict = evaluate_ee(ee_spec, intrinsic)
            per_ee[ee_id] = verdict
            if verdict.get("feasible"):
                feasible.append(ee_id)
        result[object_id] = {
            "mass_kg": float(record["mass_kg"]),
            "size_m": list(record.get("size_m") or []),
            "intrinsic": intrinsic,
            "ee": per_ee,
            "feasible_ees": feasible,
        }
    return result


def _owner_ids_for_subgoal(subgoal: Any) -> list[str]:
    owners: list[str] = []
    tool_id = getattr(subgoal, "tool_id", None)
    if isinstance(tool_id, str) and tool_id.strip():
        owners.append(tool_id)
    binding = getattr(subgoal, "source_binding", None) or {}
    if isinstance(binding, Mapping):
        for role, value in binding.items():
            if role in _GRASPED_ROLES and isinstance(value, str) and not value.startswith("?"):
                owners.append(value)
    if not owners:
        for target in getattr(subgoal, "target_ids", None) or []:
            if isinstance(target, str):
                owners.append(target)
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in owners:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def apply_gt_feasible_ee(
    request: TaskPlannerRequest,
    object_feasibility: Mapping[str, Mapping[str, Any]],
) -> TaskPlannerRequest:
    """Replace each subgoal's feasible_ee with the GT intersection over owners."""

    feas_sets = {
        object_id: set(record.get("feasible_ees") or [])
        for object_id, record in object_feasibility.items()
    }
    patched = request.model_copy(deep=True)
    for subgoal in patched.task_graph.subgoals:
        owners = _owner_ids_for_subgoal(subgoal)
        present = [feas_sets[owner] for owner in owners if owner in feas_sets]
        if not present:
            # No GT evidence for this subgoal's owners — leave upstream set.
            continue
        intersection = set.intersection(*present)
        subgoal.feasible_ee = sorted(intersection)
        # Literal stays in {"gk","m2","request"}; record GT in parameters.
        subgoal.feasible_ee_source = "request"
        subgoal.action_parameters = {
            **dict(subgoal.action_parameters or {}),
            "gt_feasible_ee": sorted(intersection),
            "gt_feasibility_owners": owners,
        }
    return patched


def solve_ee_plan(request: TaskPlannerRequest) -> PlanningResult:
    """Same EE planner the runtime uses (M4 Dijkstra)."""

    return plan(request)


def compute_sreg(*, actual_ee_switches: int, optimal_ee_switches: int) -> int:
    return int(actual_ee_switches) - int(optimal_ee_switches)


def _actual_switches_from_m4(m4_payload: Mapping[str, Any]) -> int | None:
    plan = m4_payload.get("selected_plan")
    if not isinstance(plan, Mapping):
        return None
    cost = plan.get("cost_vector")
    if not isinstance(cost, Mapping):
        return None
    value = cost.get("ee_switches")
    return int(value) if value is not None else None


def _count_live_ee_exchanges(live_run_dir: Path | None) -> int | None:
    """Count successful mid-task EE exchanges (not the initial bare→EE attach)."""

    from tuj.gt.ee_swap_metrics import (
        find_latest_live_manifest,
        load_executed_ee_metrics,
    )

    if live_run_dir is not None:
        manifest = Path(live_run_dir) / "live-execution-manifest.json"
        metrics = load_executed_ee_metrics(manifest)
        if metrics is not None and metrics.get("executed_ee_switches") is not None:
            return int(metrics["executed_ee_switches"])
        # Allow passing the m5/ directory or the live/ pointer file.
        pointer = find_latest_live_manifest(live_run_dir)
        if pointer is not None:
            metrics = load_executed_ee_metrics(pointer)
            if metrics is not None and metrics.get("executed_ee_switches") is not None:
                return int(metrics["executed_ee_switches"])
    return None


def compute_optimal_ee_plan(
    *,
    gk_payload: Mapping[str, Any],
    m2_payload: Mapping[str, Any],
    m1_payload: Mapping[str, Any],
    robot_spec: Mapping[str, Any],
    gt_scene: Mapping[str, Any],
    aliases: Mapping[str, str] | None = None,
    initial_ee: str | None = None,
    scripted_environment: str | None = None,
) -> dict[str, Any]:
    """Run M4 with GT feasible_ee; return gt.json-ready payload."""

    from tuj.m5_motion.scripted_grasps.task_constraints import (
        constrain_task_request,
        scene_id_aliases,
    )

    resolved_aliases = dict(aliases or scene_id_aliases(m1_payload))
    object_feas = build_gt_object_feasibility(
        gt_scene,
        ee_pool=list(robot_spec.get("ee_pool") or []),
        m1_payload=m1_payload,
        aliases=resolved_aliases,
    )

    request = build_request_from_gk(
        gk_payload,
        m2_payload,
        m1_payload=m1_payload,
        robot_spec_payload=robot_spec,
        id_aliases=resolved_aliases,
    )
    if initial_ee is not None:
        request.task_graph.initial_state.current_ee = initial_ee
    elif "current_ee" in robot_spec:
        request.task_graph.initial_state.current_ee = robot_spec.get("current_ee")

    request = apply_gt_feasible_ee(request, object_feas)
    scripted_changes: list[dict[str, Any]] = []
    if scripted_environment:
        request, scripted_changes = constrain_task_request(
            request, scripted_environment
        )

    result = solve_ee_plan(request)
    selected = result.selected_plan
    optimal_switches = (
        int(selected.cost_vector.ee_switches) if selected is not None else None
    )

    subgoal_feas = {
        sg.subgoal_id: list(sg.feasible_ee) for sg in request.task_graph.subgoals
    }

    return {
        "status": result.status.value if hasattr(result.status, "value") else str(result.status),
        "initial_ee": request.task_graph.initial_state.current_ee,
        "feasibility_rule": "tuj.m1_scene.ee_rules.evaluate_ee",
        "planner": "tuj.m4_taskplanner.planner.plan",
        "object_feasibility": {
            object_id: {
                "mass_kg": record["mass_kg"],
                "size_m": record["size_m"],
                "feasible_ees": list(record["feasible_ees"]),
                "ee": {
                    ee_id: {
                        "feasible": bool(verdict.get("feasible")),
                        "reason": verdict.get("reason"),
                    }
                    for ee_id, verdict in (record.get("ee") or {}).items()
                },
            }
            for object_id, record in object_feas.items()
        },
        "subgoal_feasible_ee": subgoal_feas,
        "scripted_environment": scripted_environment,
        "scripted_changes": scripted_changes,
        "optimal_ee_switches": optimal_switches,
        "optimal_ee_plan": (
            selected.model_dump(mode="json") if selected is not None else None
        ),
        "planning_result_status": (
            result.status.value if hasattr(result.status, "value") else str(result.status)
        ),
        "success": result.status is PlanStatus.SUCCESS,
    }


def write_gt_json(
    path: str | Path,
    *,
    task: str,
    optimal: Mapping[str, Any],
    actual_ee_switches: int | None,
    actual_source: str,
    extras: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    optimal_switches = optimal.get("optimal_ee_switches")
    payload: dict[str, Any] = {
        "task": task,
        "actual_ee_switches": actual_ee_switches,
        "actual_ee_switches_source": actual_source,
        "optimal_ee_switches": optimal_switches,
        "sreg": (
            compute_sreg(
                actual_ee_switches=int(actual_ee_switches),
                optimal_ee_switches=int(optimal_switches),
            )
            if actual_ee_switches is not None and optimal_switches is not None
            else None
        ),
        "fixed": {
            "planner": optimal.get("planner"),
            "feasibility_rule": optimal.get("feasibility_rule"),
            "initial_ee": optimal.get("initial_ee"),
            "note": (
                "CostVector.ee_switches is the discrete exchange count used by "
                "Dijkstra; robot_spec ee_swap_cost_s is not a search weight."
            ),
        },
        "object_feasibility": optimal.get("object_feasibility"),
        "subgoal_feasible_ee": optimal.get("subgoal_feasible_ee"),
        "optimal_ee_plan": optimal.get("optimal_ee_plan"),
        "planning_result_status": optimal.get("planning_result_status"),
        "scripted_environment": optimal.get("scripted_environment"),
        "scripted_changes": optimal.get("scripted_changes"),
    }
    if extras:
        payload.update(extras)
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def build_gt_payload_for_task_dir(
    task_dir: str | Path,
    *,
    task: str,
    gt_scene_path: str | Path,
    robot_spec_path: str | Path,
    initial_ee: str | None = None,
    scripted_environment: str | None = None,
    live_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience: load a run folder + GT scene report → gt.json body."""

    folder = Path(task_dir)
    m1 = load_json(folder / "m1.json")
    m2 = load_json(folder / "m2.json")
    gk = load_json(folder / "gk_bundle.json")
    robot_spec = load_json(robot_spec_path)
    gt_scene = load_json(gt_scene_path)
    m4_path = folder / "m4.json"
    m4 = load_json(m4_path) if m4_path.is_file() else {}

    from tuj.m5_motion.scripted_grasps.task_constraints import scene_id_aliases

    optimal = compute_optimal_ee_plan(
        gk_payload=gk,
        m2_payload=m2,
        m1_payload=m1,
        robot_spec=robot_spec,
        gt_scene=gt_scene,
        aliases=scene_id_aliases(m1),
        initial_ee=initial_ee,
        scripted_environment=scripted_environment,
    )

    live_count = _count_live_ee_exchanges(
        Path(live_run_dir) if live_run_dir is not None else None
    )
    if live_count is None:
        # Default: look under the task folder's m5/live manifests.
        live_count = _count_live_ee_exchanges(folder / "m5")
    m5_summary = folder / "m5" / "m5_summary.json"
    if live_count is None and m5_summary.is_file():
        summary = load_json(m5_summary)
        if summary.get("executed_ee_switches") is not None:
            live_count = int(summary["executed_ee_switches"])
            # Prefer m5_summary when it already stored the metric.
            if live_run_dir is None:
                live_run_dir = summary.get("live_run_dir")

    if live_count is not None:
        actual, source = live_count, "executed_ee_switches"
    else:
        planned = _actual_switches_from_m4(m4)
        actual, source = planned, "m4.selected_plan.cost_vector.ee_switches"

    return write_gt_json(
        folder / "gt.json",
        task=task,
        optimal=optimal,
        actual_ee_switches=actual,
        actual_source=source,
        extras={"gt_scene": str(Path(gt_scene_path).resolve())},
    )
