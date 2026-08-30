"""Integration coverage for the GK + M1 input boundary."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tuj.m3_taskplanner.cli import main
from tuj.m3_taskplanner.diagnostics import PlanStatus, ReasonCode
from tuj.m3_taskplanner.gk_adapter import build_request_from_gk
from tuj.m3_taskplanner.models import Condition, InitialState
from tuj.m3_taskplanner.planner import plan


def _gk() -> dict:
    # Uses the mixed-case class prefix found in the attached c1_1_gk.json.
    return {
        "task_id": "c1_1",
        "proposed_order": ["SG1"],
        "gk_by_subgoal": [
            {
                "subgoal_id": "SG1",
                "nodes": {
                    "obj_PlateObject_light_plate": {
                        "mass_kg": 0.2,
                        "material": "plastic",
                        "geometry": {"extents_mm": [170, 175, 10]},
                        "ee": {"2F": {"feasible": True}},
                    },
                    "obj_PlateObject_heavy_plate": {
                        "mass_kg": 2.0,
                        "material": "ceramic",
                        "geometry": {"extents_mm": [170, 175, 10]},
                        "ee": {"2F": {"feasible": True}},
                    },
                },
                "edges": [],
                "flags": {"near_threshold": []},
            }
        ],
    }


def _m1(*, tool_id: str = "obj_plate_light_plate") -> dict:
    # Uses the lower-case class prefix found in the repository's m1.json.
    return {
        "task": "레고 블록을 수거 영역으로 쓸어 담아라",
        "m1_subgoals": [
            {
                "subgoal_id": "SG1",
                "kind": "sweep_collect",
                "target_ids": ["obj_block_block_0"],
                "container_id": "obj_zone_collection_zone_visual",
                "tool_id": tool_id,
                "details": [
                    {
                        "detail_id": "SG1_d1",
                        "action_type": "acquire",
                        "binding": {"?o": "?tool"},
                        "group_id": "G_SG1_tool",
                        "pre": [
                            {
                                "id": "SG1_d1_p0",
                                "expr": "hand_empty",
                                "head": "hand_empty",
                                "eval_by": "m1",
                            }
                        ],
                        "establish": [
                            {
                                "id": "SG1_d1_e0",
                                "expr": "holding(?tool)",
                                "head": "holding",
                                "eval_by": "m1",
                            }
                        ],
                        "destroy": [
                            {
                                "id": "SG1_d1_d0",
                                "expr": "hand_empty",
                                "head": "hand_empty",
                                "eval_by": "m1",
                            }
                        ],
                    },
                    {
                        "detail_id": "SG1_d2",
                        "action_type": "tool_act:sweep",
                        "binding": {
                            "?t": "?tool",
                            "?targets": "{obj_block_block_0}",
                            "?r": "obj_zone_collection_zone_visual",
                        },
                        "group_id": "G_SG1_tool",
                        "pre": [
                            {
                                "id": "SG1_d2_p0",
                                "expr": "holding(?tool)",
                                "head": "holding",
                                "eval_by": "m1",
                            },
                            {
                                "id": "SG1_d2_p1",
                                "expr": "path_clear(?tool, "
                                "obj_zone_collection_zone_visual)",
                                "head": "path_clear",
                                "eval_by": "motion",
                            },
                        ],
                        "establish": [
                            {
                                "id": "SG1_d2_e0",
                                "expr": "in({obj_block_block_0}, "
                                "obj_zone_collection_zone_visual)",
                                "head": "in",
                                "eval_by": "m1",
                            }
                        ],
                        "destroy": [],
                    },
                ],
            }
        ],
        "m1_partial_order": [
            {"from": "SG1_d1", "to": "SG1_d2", "why": "causal_link"}
        ],
    }


def _m0() -> dict:
    return {
        "nodes": [
            {
                "id": "obj_block_block_0",
                "class": "block",
                "bbox_mm": [20, 20, 10],
                "mass_kg": 0.1,
            },
            {
                "id": "obj_zone_collection_zone_visual",
                "class": "zone",
                "bbox_mm": [200, 200, 0.1],
            },
        ],
        "edges": [],
    }


def _robot_spec() -> dict:
    return {
        "current_ee": "2F",
        "held_tool": None,
        "hand_empty": True,
        # Real Tool-Use-Journal robot_spec.json stores ee_pool as a list.
        "ee_pool": [
            {
                "ee_id": "2F",
                "payload_kg": 1.0,
                "home_slot": "ee-slot-2f",
            }
        ],
    }


def _bundle_gk_and_scene_m1() -> tuple[dict, dict]:
    """Mirror the attached gk_bundle + scene-graph M1 contract."""

    gk = deepcopy(_gk())
    rough = deepcopy(_m1()["m1_subgoals"][0])
    record = gk["gk_by_subgoal"][0]
    record.update(
        {
            "task": "레고 블록을 수거 영역으로 쓸어 담아라",
            "kind": rough["kind"],
            "roles": {
                "target": rough["target_ids"],
                "container": rough["container_id"],
                "tool_candidates": [
                    "obj_plate_light_plate",
                    "obj_plate_heavy_plate",
                ],
                "selected_tool": rough["tool_id"],
            },
            "details": rough["details"],
            "partial_order": _m1()["m1_partial_order"],
        }
    )
    return gk, _m0()


def test_gk_adapter_normalizes_c1_1_ids_actions_and_conditions() -> None:
    request = build_request_from_gk(
        _gk(), _m1(), m0_payload=_m0(), robot_spec_payload=_robot_spec()
    )

    first, sweep = request.task_graph.subgoals
    assert request.task_graph.log["adapter"] == "gk-m1-v1"
    assert first.tool_id == "light_plate"
    assert first.tool_selection_source == "upstream_fixed"
    assert first.goal_region_id is None
    assert sweep.action_type == "tool_act"
    assert sweep.mode == "sweep"
    assert sweep.target_ids == ["block_0"]
    assert sweep.goal_region_id == "collection_zone_visual"
    assert sweep.source_binding["?targets"] == ["block_0"]
    assert sweep.preconditions[1].args == ["?tool", "collection_zone_visual"]
    assert sweep.preconditions[1].eval_by == "motion"
    assert request.resource_catalog.objects["block_0"].bbox_mm == (
        20.0,
        20.0,
        10.0,
    )
    assert request.resource_catalog.tools["light_plate"].mass == 0.2
    assert "heavy_plate" not in request.resource_catalog.tools
    assert request.resource_catalog.end_effectors["2F"].payload == 1.0


def test_gk_bundle_uses_selected_tool_and_m1_scene_graph() -> None:
    gk, scene_m1 = _bundle_gk_and_scene_m1()
    gk["gk_by_subgoal"][0]["details"][0]["establish"] = [
        "holding(?tool)"
    ]
    gk["gk_by_subgoal"][0]["details"][0]["destroy"] = ["hand_empty"]

    request = build_request_from_gk(
        gk,
        scene_m1,
        robot_spec_payload=_robot_spec(),
    )
    result = plan(request)

    assert request.task_graph.log["adapter"] == "gk-bundle+m1-scene-v1"
    assert request.task_graph.task.instruction == (
        "레고 블록을 수거 영역으로 쓸어 담아라"
    )
    assert request.resource_catalog.objects["block_0"].bbox_mm == (
        20.0,
        20.0,
        10.0,
    )
    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    assert request.task_graph.subgoals[0].establish[0].type == "holding"
    assert request.task_graph.subgoals[0].destroy[0].type == "hand_empty"
    assert {
        assignment.tool
        for assignment in result.selected_plan.candidate_assignments
    } == {"light_plate"}


def test_repository_c1_1_bundle_selects_forwarded_light_plate() -> None:
    repository = Path(__file__).resolve().parents[4]
    output = repository / "output" / "c1_1"

    request = build_request_from_gk(
        json.loads((output / "gk_bundle.json").read_text(encoding="utf-8")),
        json.loads((output / "m1.json").read_text(encoding="utf-8")),
        robot_spec_payload=json.loads(
            (repository / "configs" / "robot_spec.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    result = plan(request)

    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    assignments = result.selected_plan.candidate_assignments
    assert {assignment.tool for assignment in assignments} == {"light_plate"}
    sweeps = [item for item in assignments if item.action_type == "tool_act"]
    assert len(sweeps) == 3
    assert len({target for item in sweeps for target in item.target_ids}) == 12


def test_gk_accepts_selected_tool_id_alias() -> None:
    m1 = _m1()
    rough = m1["m1_subgoals"][0]
    rough["selected_tool_id"] = rough.pop("tool_id")

    request = build_request_from_gk(
        _gk(), m1, m0_payload=_m0(), robot_spec_payload=_robot_spec()
    )

    assert {subgoal.tool_id for subgoal in request.task_graph.subgoals} == {
        "light_plate"
    }


def test_gk_upstream_selected_tool_is_preserved_across_group() -> None:
    request = build_request_from_gk(
        _gk(), _m1(), m0_payload=_m0(), robot_spec_payload=_robot_spec()
    )

    result = plan(request)

    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    assert result.selected_plan.group_ee_assignments == {"G_SG1_tool": "2F"}
    assert {
        assignment.tool for assignment in result.selected_plan.candidate_assignments
    } == {"light_plate"}
    assert not any(
        rejection.candidate_id and "heavy_plate" in rejection.candidate_id
        for rejection in result.rejections
    )


def test_gk_does_not_fallback_from_upstream_selected_tool() -> None:
    request = build_request_from_gk(
        _gk(),
        _m1(tool_id="obj_plate_heavy_plate"),
        m0_payload=_m0(),
        robot_spec_payload=_robot_spec(),
    )

    result = plan(request)

    assert result.selected_plan is None
    assert any(
        rejection.reason_code is ReasonCode.PAYLOAD_EXCEEDED
        and rejection.candidate_id
        and "heavy_plate" in rejection.candidate_id
        for rejection in result.rejections
    )


def test_gk_rejects_tool_candidates_without_upstream_selection() -> None:
    m1 = _m1()
    rough = m1["m1_subgoals"][0]
    rough.pop("tool_id")
    rough["tool_candidate_ids"] = [
        "obj_plate_light_plate",
        "obj_plate_heavy_plate",
    ]

    with pytest.raises(ValueError, match="Task Planner does not select tools"):
        build_request_from_gk(
            _gk(), m1, m0_payload=_m0(), robot_spec_payload=_robot_spec()
        )


def test_gk_cli_path_runs_end_to_end(tmp_path) -> None:
    inputs = {
        "gk": _gk(),
        "m1": _m1(),
        "m0": _m0(),
        "robot": _robot_spec(),
    }
    paths = {}
    for name, payload in inputs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    output = tmp_path / "result.json"

    exit_code = main(
        [
            "plan",
            "--gk",
            str(paths["gk"]),
            "--m1",
            str(paths["m1"]),
            "--m0",
            str(paths["m0"]),
            "--robot-spec",
            str(paths["robot"]),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "SUCCESS"
    assert {
        assignment["tool"]
        for assignment in payload["selected_plan"]["candidate_assignments"]
    } == {"light_plate"}
    assert "group_tool_assignments" not in payload["selected_plan"]


def test_gk_end_to_end_is_deterministic() -> None:
    first = plan(
        build_request_from_gk(
            _gk(), _m1(), m0_payload=_m0(), robot_spec_payload=_robot_spec()
        )
    ).model_dump(mode="json")
    second = plan(
        build_request_from_gk(
            _gk(), _m1(), m0_payload=_m0(), robot_spec_payload=_robot_spec()
        )
    ).model_dump(mode="json")

    first["search_stats"]["elapsed_ms"] = 0
    second["search_stats"]["elapsed_ms"] = 0
    assert first == second


def test_gk_requires_explicit_initial_robot_state() -> None:
    robot_spec = _robot_spec()
    robot_spec.pop("current_ee")

    with pytest.raises(ValueError, match="initial current_ee is missing"):
        build_request_from_gk(
            _gk(), _m1(), m0_payload=_m0(), robot_spec_payload=robot_spec
        )


def test_gk_accepts_explicit_null_initial_ee() -> None:
    robot_spec = _robot_spec()
    robot_spec["current_ee"] = None

    request = build_request_from_gk(
        _gk(), _m1(), m0_payload=_m0(), robot_spec_payload=robot_spec
    )
    result = plan(request)

    assert request.task_graph.initial_state.current_ee is None
    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    assert result.selected_plan.cost_vector.ee_switches == 0


def test_gk_accepts_normalized_initial_state_override() -> None:
    robot_spec = _robot_spec()
    robot_spec.pop("current_ee")
    robot_spec.pop("hand_empty")
    initial_state = InitialState(
        current_ee="2F",
        rack={"ee-slot-2f": "2F"},
        held_tool=None,
        facts=[
            Condition(
                cond_id="initial-hand-empty",
                type="hand_empty",
                args=[],
                pass_=True,
                eval_by="robot_state",
            )
        ],
    )

    request = build_request_from_gk(
        _gk(),
        _m1(),
        m0_payload=_m0(),
        robot_spec_payload=robot_spec,
        initial_state=initial_state,
    )

    assert request.task_graph.initial_state == initial_state


def test_gk_without_m1_fails_with_actionable_error() -> None:
    with pytest.raises(ValueError, match="no executable details"):
        build_request_from_gk(_gk(), {})
