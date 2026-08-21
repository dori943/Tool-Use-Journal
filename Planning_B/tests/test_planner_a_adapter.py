"""Integration tests for the current Planner-A detailed DAG schema."""

from __future__ import annotations

from copy import deepcopy

from planning_b.diagnostics import PlanStatus
from planning_b.models import ResourceCatalog
from planning_b.planner import plan
from planning_b.planner_a_adapter import build_request_from_current_planner_a


def _condition(
    kind: str,
    *args: str,
    pass_: bool | None,
    eval_by: str = "planner_a",
    depends_on: str | None = None,
    needs_observation: bool = False,
) -> dict:
    return {
        "type": kind,
        "args": list(args),
        "pass": pass_,
        "eval_by": eval_by,
        "depends_on": depends_on,
        "needs_observation": needs_observation,
    }


def _upstream_payload() -> dict:
    return {
        "scenario": "adapter test",
        "task": "box를 tray에 놓아라",
        "detailed_subgoals": [
            {
                "subgoal_id": "SG1_d1",
                "action_type": "acquire",
                "binding": {"?o": "box", "?ee": "2f"},
                "group_id": "G_box",
                "from_kg": "SG1",
                "note": "box 확보",
                "pre": [
                    _condition("reachable", "box", pass_=True),
                    _condition(
                        "attached_ee", "2f", pass_=None, eval_by="motion"
                    ),
                    _condition("hand_empty", pass_=True),
                ],
                "establish": [_condition("holding", "box", pass_=True)],
                "destroy": [_condition("hand_empty", pass_=False)],
            },
            {
                "subgoal_id": "SG1_d2",
                "action_type": "transport",
                "binding": {"?o": "box", "?r": "tray"},
                "group_id": "G_box",
                "from_kg": "SG1",
                "note": "box 운반",
                "pre": [
                    _condition("holding", "box", pass_=False),
                    _condition(
                        "path_clear", "box", "tray", pass_=None, eval_by="motion"
                    ),
                ],
                "establish": [_condition("above", "box", "tray", pass_=True)],
                "destroy": [],
            },
            {
                "subgoal_id": "SG1_d3",
                "action_type": "place",
                "binding": {"?o": "box", "?r": "tray"},
                "group_id": "G_box",
                "from_kg": "SG1",
                "note": "box 배치",
                "pre": [
                    _condition("holding", "box", pass_=False),
                    _condition("above", "box", "tray", pass_=False),
                    _condition(
                        "clear",
                        "tray",
                        pass_=None,
                        depends_on="SG1_d2",
                        needs_observation=True,
                    ),
                ],
                "establish": [
                    _condition("in_region", "box", "tray", pass_=True),
                    _condition("hand_empty", pass_=True),
                ],
                "destroy": [_condition("holding", "box", pass_=False)],
            },
        ],
        "edges": [
            {"from": "SG1_d1", "to": "SG1_d2", "reason": "causal_link"},
            {"from": "SG1_d2", "to": "SG1_d3", "reason": "observability"},
        ],
        "cycles": [],
        "redecompose_signals": [],
        "mutex": [],
        "deferred_conditions": [],
        "sg_observation_requests": [],
        "disjunctive_threats": [],
        "stats": {"n_orders_dp": 1},
    }


def _scenario_payload() -> dict:
    return {
        "sg": {
            "objects": {
                "box": {
                    "reachable": True,
                    "feasible_ee": ["2f", "vac"],
                    "at_rest": True,
                }
            },
            "object_specs": {
                "box": {
                    "mass_kg": 0.1,
                    "bbox_mm": [40.0, 30.0, 20.0],
                    "material": "cardboard",
                }
            },
            "per_subgoal": {
                "SG1": {
                    "target": "box",
                    "goal_region": "tray",
                    "tool_required": False,
                    "feasible_ee": ["2f", "vac"],
                    "ee_candidate": "2f",
                }
            },
        },
        "robot_state": {
            "in_hand": None,
            "current_ee": "2f",
            "ee_rack": ["2f", "vac"],
        },
    }


def test_current_planner_a_output_runs_end_to_end() -> None:
    request = build_request_from_current_planner_a(
        _upstream_payload(), _scenario_payload()
    )

    assert request.planner_a.task.instruction == "box를 tray에 놓아라"
    assert request.planner_a.initial_state.current_ee == "2f"
    assert all(
        subgoal.feasible_ee == ["2f", "vac"]
        for subgoal in request.planner_a.subgoals
    )
    assert all(
        condition.type != "attached_ee"
        for subgoal in request.planner_a.subgoals
        for condition in subgoal.preconditions
    )
    assert request.resource_catalog.objects["box"].mass_kg == 0.1
    assert request.resource_catalog.objects["box"].bbox_mm == (40.0, 30.0, 20.0)
    assert request.resource_catalog.objects["box"].material == "cardboard"

    result = plan(request)
    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan is not None
    assert result.selected_plan.subgoal_order == ["SG1_d1", "SG1_d2", "SG1_d3"]
    assert result.selected_plan.group_ee_assignments == {"G_box": "2f"}
    assert result.selected_plan.cost_vector.ee_switches == 0
    assert "in_region(box, tray)" in result.selected_plan.terminal_state["facts"]

    execution_steps = [
        step for step in result.selected_plan.steps if step.kind == "subgoal"
    ]
    assert execution_steps[1].verification_required is True  # path_clear
    assert execution_steps[2].verification_required is True  # deferred clear


def test_adapter_preserves_planner_a_diagnostics_in_log() -> None:
    request = build_request_from_current_planner_a(
        _upstream_payload(), _scenario_payload()
    )
    assert request.planner_a.order_constraints.n_topological_orders == 1
    assert request.planner_a.log["adapter"] == "current-planner-a-dag-v1"


def test_adapter_preserves_full_planner_a_constraint_contract() -> None:
    payload = deepcopy(_upstream_payload())
    payload["mutex"] = [
        {
            "a": "SG1_d1",
            "b": "SG1_d3",
            "condition": "C_mutex",
            "predicate": "hand_empty",
            "args": [],
            "reestablished_by": ["SG1_d2"],
        }
    ]
    payload["open_conditions"] = [
        {
            "subgoal": "SG1_d3",
            "condition": "C_open",
            "candidates": ["SG1_d1", "SG1_d2"],
        }
    ]
    payload["disjunctive_threats"] = [
        {
            "key": ["SG1_d1", "SG1_d3", "SG1_d2", "C_threat"],
            "link": ["SG1_d1", "SG1_d3"],
            "threat": "SG1_d2",
            "condition": "C_threat",
            "options": ["promotion", "demotion"],
        }
    ]
    payload["deferred_conditions"] = [
        {
            "subgoal": "SG1_d3",
            "condition": "C_deferred",
            "type": "clear",
            "args": ["tray"],
            "depends_on": "SG1_d2",
        }
    ]
    payload["sg_observation_requests"] = [
        {
            "subgoal": "SG1_d3",
            "condition": "C_observe",
            "type": "clear",
            "args": ["tray"],
            "depends_on": "SG1_d2",
            "needs_observation": True,
            "request": "SG1_d2 이후 재관측",
        }
    ]
    payload["kg_order_audit"] = {
        "verdicts": [{"pair": ["SG1", "SG2"], "verdict": "필요"}],
        "kg_missing": [],
        "counts": {"필요": 1},
    }
    payload["future_contract_field"] = {"kept": True}

    request = build_request_from_current_planner_a(payload, _scenario_payload())
    contract = request.planner_a.constraints

    assert contract.mutex[0].predicate == "hand_empty"
    assert contract.open_conditions[0].candidates == ["SG1_d1", "SG1_d2"]
    assert contract.disjunctive_threats[0].link == ("SG1_d1", "SG1_d3")
    assert contract.deferred_conditions[0].depends_on == "SG1_d2"
    assert contract.sg_observation_requests[0].needs_observation is True
    assert contract.kg_order_audit.counts == {"필요": 1}
    assert request.planner_a.future_contract_field == {"kept": True}
    assert "mutex" not in request.planner_a.log


def test_adapter_marks_planner_a_tool_as_fixed() -> None:
    payload = deepcopy(_upstream_payload())
    payload["detailed_subgoals"][1]["action_type"] = "tool_act"
    payload["detailed_subgoals"][1]["binding"] = {
        "?o": "box",
        "?r": "tray",
        "?t": "pusher",
    }

    request = build_request_from_current_planner_a(payload, _scenario_payload())

    assert {subgoal.tool_id for subgoal in request.planner_a.subgoals} == {
        "pusher"
    }
    assert {
        subgoal.tool_selection_source for subgoal in request.planner_a.subgoals
    } == {"planner_a_fixed"}


def test_external_resource_specs_merge_with_scenario_objects() -> None:
    supplied = ResourceCatalog.model_validate(
        {
            "end_effectors": {
                "2f": {
                    "payload": 5.0,
                    "max_opening_mm": 85.0,
                },
                "vac": {"payload": 2.0, "capabilities": ["suction"]},
            },
            "objects": {
                "box": {
                    "surface_condition": "wet",
                    "fragility": "medium",
                }
            },
        }
    )
    request = build_request_from_current_planner_a(
        _upstream_payload(),
        _scenario_payload(),
        resource_catalog=supplied,
    )

    box = request.resource_catalog.objects["box"]
    assert box.mass_kg == 0.1
    assert box.bbox_mm == (40.0, 30.0, 20.0)
    assert box.material == "cardboard"
    assert box.surface_condition == "wet"
    assert box.fragility == "medium"
    assert request.resource_catalog.end_effectors["2f"].payload == 5.0
