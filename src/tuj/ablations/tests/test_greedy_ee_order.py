"""Greedy EE-order policy and production isolation for C3_2 2F extras."""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from tuj.ablations.greedy_ee_order import METHOD, VERSION, greedy_ee_order_plan
from tuj.m4_taskplanner.diagnostics import PlanStatus
from tuj.m4_taskplanner.tests.conftest import make_request, sg
from tuj.m5_motion.scripted_grasps.registry import (
    ENABLED_ENTRIES,
    ENTRIES,
    GREEDY_EXTRA_ENTRIES,
    resolve,
)
from tuj.m5_motion.scripted_grasps.task_constraints import constrain_task_request
from tuj.m5_motion.scripted_grasps.validate import _acquire_request


C3_2 = "C3_2_BreakfastTrayPreparation"


def test_production_entries_have_no_c3_2_2f():
    assert not any(
        e.environment == C3_2 and e.ee == "2F" for e in ENTRIES
    )
    assert not any(
        e.environment == C3_2 and e.ee == "2F" for e in ENABLED_ENTRIES
    )


def test_greedy_extra_lists_c3_2_2f_objects():
    pairs = {(e.object_id, e.ee) for e in GREEDY_EXTRA_ENTRIES}
    assert ("bread", "2F") in pairs
    assert ("spoon", "2F") in pairs
    assert ("fork", "2F") in pairs
    assert ("fruit", "2F") in pairs
    assert ("mug", "2F") not in pairs


def test_constrain_without_extra_hides_c3_2_2f():
    from tuj.m4_taskplanner.models import (
        InitialState,
        OrderConstraints,
        TaskGraph,
        TaskPlannerRequest,
        TaskSpec,
        Subgoal,
        ResourceCatalog,
        PlanningPolicy,
    )

    request = TaskPlannerRequest(
        task_graph=TaskGraph(
            task=TaskSpec(instruction="t"),
            initial_state=InitialState(current_ee="3F"),
            subgoals=[
                Subgoal(
                    subgoal_id="pick_bread",
                    action_type="acquire",
                    target_ids=["bread_a"],
                    feasible_ee=["2F", "3F"],
                )
            ],
            order_constraints=OrderConstraints(),
        ),
        resource_catalog=ResourceCatalog.model_validate(
            {
                "end_effectors": {
                    "2F": {"capabilities": ["grip"], "payload": 5.0, "home_slot": "S2"},
                    "3F": {"capabilities": ["grip"], "payload": 5.0, "home_slot": "S3"},
                },
                "tools": {},
            }
        ),
        planning_policy=PlanningPolicy(),
    )
    constrained, _ = constrain_task_request(request, C3_2)
    assert constrained.task_graph.subgoals[0].feasible_ee == ["3F"]


def test_constrain_with_greedy_extra_keeps_c3_2_2f():
    from tuj.m4_taskplanner.models import (
        InitialState,
        OrderConstraints,
        TaskGraph,
        TaskPlannerRequest,
        TaskSpec,
        Subgoal,
        ResourceCatalog,
        PlanningPolicy,
    )

    request = TaskPlannerRequest(
        task_graph=TaskGraph(
            task=TaskSpec(instruction="t"),
            initial_state=InitialState(current_ee="3F"),
            subgoals=[
                Subgoal(
                    subgoal_id="pick_bread",
                    action_type="acquire",
                    target_ids=["bread_a"],
                    feasible_ee=["2F", "3F"],
                )
            ],
            order_constraints=OrderConstraints(),
        ),
        resource_catalog=ResourceCatalog.model_validate(
            {
                "end_effectors": {
                    "2F": {"capabilities": ["grip"], "payload": 5.0, "home_slot": "S2"},
                    "3F": {"capabilities": ["grip"], "payload": 5.0, "home_slot": "S3"},
                },
                "tools": {},
            }
        ),
        planning_policy=PlanningPolicy(),
    )
    constrained, changes = constrain_task_request(
        request, C3_2, extra_entries=GREEDY_EXTRA_ENTRIES
    )
    assert set(constrained.task_graph.subgoals[0].feasible_ee) == {"2F", "3F"}
    assert changes


def test_resolve_2f_hidden_without_greedy_flag():
    request = _acquire_request(C3_2, "bread_b", "2F")
    request.task.metadata.pop("scripted_grasp_validator_experimental", None)
    assert resolve(request) is None


def test_resolve_2f_visible_with_greedy_flag():
    request = _acquire_request(C3_2, "bread_b", "2F")
    request.task.metadata.pop("scripted_grasp_validator_experimental", None)
    request.task.metadata["scripted_grasp_greedy_extra"] = True
    entry = resolve(request)
    assert entry is not None
    assert entry.ee == "2F"
    assert entry.scene_object_id == "bread_b"


def test_resolve_fruit_2f_with_greedy_flag():
    request = _acquire_request(C3_2, "fruit_a", "2F")
    request.task.metadata.pop("scripted_grasp_validator_experimental", None)
    request.task.metadata["scripted_grasp_greedy_extra"] = True
    entry = resolve(request)
    assert entry is not None
    assert entry.object_id == "fruit"
    assert entry.ee == "2F"
    assert entry.recipe().recipe_id  # recipe builds


def test_greedy_prefers_zero_switch_ready_subgoal():
    # Both ready; initial EE=A. sg_switch only accepts B; sg_keep only accepts A.
    # Greedy must schedule sg_keep first (0 switch), then sg_switch (1 switch).
    request = make_request(
        [
            sg("sg_switch", targets=["o1"], feasible=["B"]),
            sg("sg_keep", targets=["o2"], feasible=["A"]),
        ],
        initial_ee="A",
    )
    result = greedy_ee_order_plan(request)
    assert result.status is PlanStatus.SUCCESS
    assert result.model_extra["method"] == METHOD
    assert result.task["scripted_grasp_greedy_extra"] is True
    order = result.selected_plan.subgoal_order
    assert order == ["sg_keep", "sg_switch"]
    assert result.selected_plan.cost_vector.ee_switches == 1
    sources = {
        a.subgoal_id: a.ee_selection_source
        for a in result.selected_plan.candidate_assignments
    }
    assert sources["sg_keep"] == VERSION
    assert sources["sg_switch"] == VERSION


def test_greedy_respects_hard_order_edges():
    request = make_request(
        [
            sg("first", targets=["o1"], feasible=["B"]),
            sg("second", targets=["o2"], feasible=["A"]),
        ],
        edges=[("first", "second")],
        initial_ee="A",
    )
    result = greedy_ee_order_plan(request)
    assert result.status is PlanStatus.SUCCESS
    assert result.selected_plan.subgoal_order == ["first", "second"]
    # Must switch to B for first, then to A for second → 2 switches after attach.
    assert result.selected_plan.cost_vector.ee_switches == 2
