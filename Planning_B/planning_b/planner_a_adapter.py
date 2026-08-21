"""Adapter for the current Planner-A output and scenario schemas.

Planner A intentionally emits a partial-order DAG, while Planning B uses a
normalized request containing an initial state, resource catalog, and
per-subgoal feasible end effectors.  The current Planner-A JSON does not carry
all of that execution information, so its companion scenario JSON is required
for a resource-aware hand-off.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from planning_b.models import (
    CandidateProposal,
    Condition,
    EndEffectorSpec,
    InitialState,
    ObjectSpec,
    OrderConstraints,
    PlannerAOutput,
    PlannerAConstraints,
    PlanningBRequest,
    PlanningPolicy,
    ResourceCatalog,
    Subgoal,
    TaskSpec,
    ToolSpec,
)

_CURRENT_PLANNER_A_FIELDS = frozenset(
    "scenario task condition_source decidability_source detailed_subgoals edges "
    "mutex deferred_conditions sg_observation_requests open_conditions "
    "disjunctive_threats cycles redecompose_signals kg_order_audit stats".split()
)
_CONSTRAINT_LIST_FIELDS = (
    "mutex open_conditions disjunctive_threats deferred_conditions "
    "sg_observation_requests redecompose_signals".split()
)


def build_request_from_current_planner_a(
    planner_a_payload: Mapping[str, Any],
    scenario_payload: Mapping[str, Any],
    *,
    resource_catalog: ResourceCatalog | None = None,
    candidate_proposals: dict[str, list[CandidateProposal]] | None = None,
    planning_policy: PlanningPolicy | None = None,
) -> PlanningBRequest:
    """Build a Planning-B request from a current Planner-A result.

    The scenario supplies the fields deliberately absent from Planner A's DAG:
    the mounted EE, the EE rack, and the complete feasible-EE sets.  When no
    external resource catalog is supplied, a symbolic catalog is derived from
    the scenario. Task poses remain owned by the motion-planning pipeline.
    """

    planner_a = adapt_current_planner_a_output(planner_a_payload, scenario_payload)
    uses_symbolic_catalog = resource_catalog is None
    derived_catalog = derive_symbolic_resource_catalog(
        planner_a_payload, scenario_payload, planner_a
    )
    catalog = (
        derived_catalog
        if resource_catalog is None
        else _merge_resource_catalog(derived_catalog, resource_catalog)
    )
    policy = planning_policy or PlanningPolicy(
        unknown_feasibility_policy=("defer" if uses_symbolic_catalog else "reject")
    )
    return PlanningBRequest(
        planner_a=planner_a,
        resource_catalog=catalog,
        candidate_proposals=candidate_proposals,
        planning_policy=policy,
    )


def adapt_current_planner_a_output(
    payload: Mapping[str, Any], scenario: Mapping[str, Any]
) -> PlannerAOutput:
    """Normalize Planner A's DAG JSON to :class:`PlannerAOutput`."""

    if "detailed_subgoals" not in payload:
        raise ValueError("Planner-A payload has no detailed_subgoals field")

    detailed = payload.get("detailed_subgoals")
    if not isinstance(detailed, list) or not detailed:
        raise ValueError("Planner-A detailed_subgoals must be a non-empty list")

    sg_input = _mapping(scenario.get("sg"))
    per_kg = _mapping(sg_input.get("per_subgoal"))
    objects = _mapping(sg_input.get("objects"))
    robot_state = _mapping(scenario.get("robot_state"))
    if robot_state.get("in_hand") is not None:
        raise ValueError(
            "the current Planner-A adapter requires an empty initial hand; "
            "provide a normalized Planning-B initial_state for held resources"
        )

    group_primary: dict[str, str] = {}
    group_tool: dict[str, str] = {}
    for raw in detailed:
        item = _mapping(raw)
        group = str(item.get("group_id") or item.get("subgoal_id"))
        binding = _mapping(item.get("binding"))
        if item.get("action_type") == "acquire" and isinstance(binding.get("?o"), str):
            group_primary[group] = binding["?o"]
        if item.get("action_type") == "tool_act" and isinstance(binding.get("?t"), str):
            group_tool[group] = binding["?t"]

    group_feasible: dict[str, list[str]] = {}
    for raw in detailed:
        item = _mapping(raw)
        group = str(item.get("group_id") or item.get("subgoal_id"))
        if group in group_feasible:
            continue
        from_kg = str(item.get("from_kg") or "")
        kg_record = _mapping(per_kg.get(from_kg))
        feasible = _string_list(kg_record.get("feasible_ee"))
        primary = group_primary.get(group)
        if not feasible and primary is not None:
            feasible = _string_list(_mapping(objects.get(primary)).get("feasible_ee"))
        if not feasible:
            feasible = _attached_ee_candidates(item)
        if not feasible:
            raise ValueError(
                f"cannot determine feasible_ee for Planner-A group {group!r}; "
                "provide a companion scenario with sg.per_subgoal"
            )
        group_feasible[group] = sorted(set(feasible))

    subgoals: list[Subgoal] = []
    initial_conditions: dict[tuple[str, tuple[str, ...]], Condition] = {}
    for raw in detailed:
        item = _mapping(raw)
        subgoal_id = _required_string(item, "subgoal_id")
        group = str(item.get("group_id") or subgoal_id)
        binding = _mapping(item.get("binding"))
        primary = group_primary.get(group)
        tool_id = group_tool.get(group)

        pre_raw = item.get("pre") or []
        establish_raw = item.get("establish") or []
        destroy_raw = item.get("destroy") or []
        preconditions = [
            Condition.model_validate(deepcopy(cond))
            for cond in pre_raw
            # EE attachment is candidate-dependent and is enforced by Planning
            # B's explicit KEEP/DETACH/ATTACH transitions.
            if _mapping(cond).get("type") != "attached_ee"
        ]
        establish = [
            Condition.model_validate(deepcopy(cond)) for cond in establish_raw
        ]
        destroy = [Condition.model_validate(deepcopy(cond)) for cond in destroy_raw]

        for cond in preconditions:
            if cond.pass_ is True:
                initial_conditions.setdefault((cond.type, tuple(cond.args)), cond)

        target_ids = [primary] if primary is not None else []
        if item.get("action_type") == "tool_act" and isinstance(binding.get("?o"), str):
            # Keep the acted-on object visible in the normalized record.  Tool
            # candidate generation uses tool_id rather than target_ids.
            target_ids = [binding["?o"]]

        subgoals.append(
            Subgoal(
                subgoal_id=subgoal_id,
                description=item.get("note"),
                action_type=item.get("action_type"),
                mode=item.get("mode"),
                group_id=group,
                source_kg_subgoal_id=(
                    item.get("from_kg")
                    if isinstance(item.get("from_kg"), str)
                    else None
                ),
                source_binding=deepcopy(dict(binding)),
                condition_source=(
                    item.get("condition_source")
                    if isinstance(item.get("condition_source"), str)
                    else None
                ),
                target_ids=target_ids,
                goal_region_id=(
                    binding.get("?r") if isinstance(binding.get("?r"), str) else None
                ),
                tool_id=tool_id,
                preconditions=preconditions,
                postconditions=establish,
                establish=establish,
                destroy=destroy,
                feasible_ee=group_feasible[group],
                tool_required=tool_id is not None,
                tool_selection_source=(
                    "planner_a_fixed" if tool_id is not None else "not_required"
                ),
                # The temporary ?ee grounding in Planner A is deliberately
                # removed. B selects from the complete scenario-derived set.
                feasible_ee_source="scenario",
            )
        )

    current_ee = robot_state.get("current_ee")
    if not isinstance(current_ee, str) or not current_ee:
        candidates = sorted({ee for values in group_feasible.values() for ee in values})
        if not candidates:
            raise ValueError("scenario.robot_state.current_ee is missing")
        current_ee = candidates[0]

    rack_ees = _string_list(robot_state.get("ee_rack"))
    if not rack_ees:
        rack_ees = sorted({current_ee, *(ee for values in group_feasible.values() for ee in values)})

    edges = payload.get("edges") or []
    cycles = payload.get("cycles") or []
    redecompose = payload.get("redecompose_signals") or []
    stats = _mapping(payload.get("stats"))
    task_value = payload.get("task")
    task = TaskSpec.model_validate(
        {
            "instruction": (
            task_value
            if isinstance(task_value, str)
            else str(_mapping(task_value).get("instruction") or "")
            ),
            "scenario": payload.get("scenario"),
        }
    )
    contract_payload = {
        key: deepcopy(payload.get(key) or []) for key in _CONSTRAINT_LIST_FIELDS
    }
    contract_payload["kg_order_audit"] = deepcopy(
        payload.get("kg_order_audit") or {}
    )

    return PlannerAOutput(
        task=task,
        initial_state=InitialState(
            current_ee=current_ee,
            rack={f"slot-{ee}": ee for ee in rack_ees},
            facts=list(initial_conditions.values()),
            held_tool=None,
        ),
        subgoals=subgoals,
        order_constraints=OrderConstraints(
            edges=deepcopy(edges),
            cycle_detected=bool(cycles),
            cycles=deepcopy(cycles),
            unestablishable=deepcopy(redecompose),
            n_topological_orders=(
                int(stats["n_orders_dp"])
                if isinstance(stats.get("n_orders_dp"), int)
                else None
            ),
        ),
        constraints=PlannerAConstraints.model_validate(contract_payload),
        log={
            "adapter": "current-planner-a-dag-v1",
            "condition_source": payload.get("condition_source"),
            "decidability_source": payload.get("decidability_source"),
            "source_stats": deepcopy(stats),
        },
        **{
            key: deepcopy(value)
            for key, value in payload.items()
            if key not in _CURRENT_PLANNER_A_FIELDS
        },
    )


def derive_symbolic_resource_catalog(
    payload: Mapping[str, Any],
    scenario: Mapping[str, Any],
    planner_a: PlannerAOutput,
) -> ResourceCatalog:
    """Derive the symbolic EE/tool/object catalog available in the scenario."""

    sg_input = _mapping(scenario.get("sg"))
    object_specs = _mapping(sg_input.get("object_specs"))
    robot_state = _mapping(scenario.get("robot_state"))
    ee_ids = set(_string_list(robot_state.get("ee_rack")))
    for subgoal in planner_a.subgoals:
        ee_ids.update(subgoal.feasible_ee)
    ee_ids.add(planner_a.initial_state.current_ee)

    tools_by_group: dict[str, str] = {}
    primary_by_group: dict[str, str] = {}
    raw_by_id = {
        str(_mapping(item).get("subgoal_id")): _mapping(item)
        for item in payload.get("detailed_subgoals") or []
    }
    for subgoal in planner_a.subgoals:
        raw = raw_by_id.get(subgoal.subgoal_id, {})
        binding = _mapping(raw.get("binding"))
        if subgoal.action_type == "acquire" and isinstance(binding.get("?o"), str):
            primary_by_group[subgoal.group_id or subgoal.subgoal_id] = binding["?o"]
        if subgoal.tool_id is not None:
            tools_by_group[subgoal.group_id or subgoal.subgoal_id] = subgoal.tool_id

    tools: dict[str, ToolSpec] = {}
    ee_tools: dict[str, set[str]] = {ee: set() for ee in ee_ids}
    for group, primary in sorted(primary_by_group.items()):
        feasible = next(
            sg.feasible_ee
            for sg in planner_a.subgoals
            if (sg.group_id or sg.subgoal_id) == group
        )
        if group in tools_by_group:
            tool_id = tools_by_group[group]
            spec = _mapping(object_specs.get(tool_id))
            mass = spec.get("mass_kg")
            tools[tool_id] = ToolSpec(
                mass=float(mass) if isinstance(mass, (int, float)) else None,
                compatible_ee=feasible,
                home_slot=f"tool-slot-{tool_id}",
            )
            for ee in feasible:
                ee_tools.setdefault(ee, set()).add(tool_id)

    end_effectors = {
        ee: EndEffectorSpec(
            compatible_tools=sorted(ee_tools.get(ee, set())),
            home_slot=f"slot-{ee}",
        )
        for ee in sorted(ee_ids)
    }
    objects = {
        object_id: ObjectSpec.model_validate(dict(raw_spec))
        for object_id, raw_spec in sorted(object_specs.items())
        if isinstance(object_id, str) and isinstance(raw_spec, Mapping)
    }
    return ResourceCatalog(
        end_effectors=end_effectors,
        tools=tools,
        objects=objects,
    )


def _merge_resource_catalog(
    derived: ResourceCatalog, supplied: ResourceCatalog
) -> ResourceCatalog:
    """Merge scenario-derived records with an external static catalog.

    Supplied EE/tool specifications override symbolic defaults. Scenario object
    fields override supplied object defaults only when the scenario explicitly
    carried that field, so model defaults such as ``surface_condition='dry'``
    do not erase a real catalog value.
    """

    end_effectors = dict(derived.end_effectors)
    for resource_id, supplied_spec in supplied.end_effectors.items():
        if resource_id not in end_effectors:
            end_effectors[resource_id] = supplied_spec
            continue
        values = end_effectors[resource_id].model_dump()
        values.update(
            supplied_spec.model_dump(include=supplied_spec.model_fields_set)
        )
        end_effectors[resource_id] = EndEffectorSpec.model_validate(values)

    tools = dict(derived.tools)
    for resource_id, supplied_spec in supplied.tools.items():
        if resource_id not in tools:
            tools[resource_id] = supplied_spec
            continue
        values = tools[resource_id].model_dump()
        values.update(
            supplied_spec.model_dump(include=supplied_spec.model_fields_set)
        )
        tools[resource_id] = ToolSpec.model_validate(values)

    objects = dict(supplied.objects)
    for object_id, scenario_spec in derived.objects.items():
        if object_id not in objects:
            objects[object_id] = scenario_spec
            continue
        values = objects[object_id].model_dump()
        values.update(
            scenario_spec.model_dump(include=scenario_spec.model_fields_set)
        )
        objects[object_id] = ObjectSpec.model_validate(values)

    return ResourceCatalog(
        end_effectors=end_effectors,
        tools=tools,
        objects=objects,
    )


def _attached_ee_candidates(item: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    binding = _mapping(item.get("binding"))
    if isinstance(binding.get("?ee"), str):
        result.append(binding["?ee"])
    for cond in item.get("pre") or []:
        record = _mapping(cond)
        if record.get("type") == "attached_ee":
            result.extend(_string_list(record.get("args")))
    return result


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _required_string(item: Mapping[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Planner-A detailed subgoal has no valid {key!r}: {item!r}")
    return value
