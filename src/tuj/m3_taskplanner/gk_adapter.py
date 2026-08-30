"""Normalize GK bundle + M1 scene/legacy artifacts into a planner request.

The current upstream contract places executable details, ordering, and the
selected tool in each ``gk_by_subgoal`` record while M1 provides the scene
``nodes``/``edges``.  The legacy contract, where M1 owns ``m1_subgoals``, is
kept for compatibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from tuj.m3_taskplanner.models import (
    CandidateProposal,
    Condition,
    EndEffectorSpec,
    InitialState,
    ObjectSpec,
    OrderConstraints,
    TaskGraph,
    TaskConstraints,
    TaskPlannerRequest,
    PlanningPolicy,
    ResourceCatalog,
    Subgoal,
    TaskSpec,
    ToolSpec,
)


_TRUE_STATUSES = frozenset({"sat", "pass", "passed", "true", "feasible"})
_FALSE_STATUSES = frozenset(
    {"unsat", "fail", "failed", "false", "infeasible"}
)


def build_request_from_gk(
    gk_payload: Mapping[str, Any],
    m1_payload: Mapping[str, Any],
    *,
    m0_payload: Mapping[str, Any] | None = None,
    robot_spec_payload: Mapping[str, Any] | None = None,
    resource_catalog: ResourceCatalog | None = None,
    candidate_proposals: dict[str, list[CandidateProposal]] | None = None,
    planning_policy: PlanningPolicy | None = None,
    id_aliases: Mapping[str, str] | None = None,
    initial_state: InitialState | None = None,
) -> TaskPlannerRequest:
    """Build a complete request from the upstream GK family of artifacts.

    In the current contract, GK carries executable action details and the
    upstream-selected tool while M1 carries scene objects.  Legacy
    ``m1_subgoals`` input remains supported. ``m0_payload`` can explicitly
    override the scene source, and ``robot_spec_payload`` enriches EE data.
    """

    aliases = dict(id_aliases or {})
    task_graph, derived_catalog = adapt_gk_m1_output(
        gk_payload,
        m1_payload,
        m0_payload=m0_payload,
        robot_spec_payload=robot_spec_payload,
        id_aliases=aliases,
        initial_state=initial_state,
    )
    uses_derived_only = resource_catalog is None
    catalog = (
        derived_catalog
        if resource_catalog is None
        else _merge_resource_catalog(derived_catalog, resource_catalog)
    )
    policy = planning_policy or PlanningPolicy(
        # A GK may omit payload, wrench, or surface information. Preserve the
        # strict behavior when a complete external catalog is supplied,
        # but defer genuinely missing derived knowledge to motion validation.
        unknown_feasibility_policy="defer" if uses_derived_only else "reject"
    )
    return TaskPlannerRequest(
        task_graph=task_graph,
        resource_catalog=catalog,
        candidate_proposals=candidate_proposals,
        planning_policy=policy,
    )


def adapt_gk_m1_output(
    gk_payload: Mapping[str, Any],
    m1_payload: Mapping[str, Any],
    *,
    m0_payload: Mapping[str, Any] | None = None,
    robot_spec_payload: Mapping[str, Any] | None = None,
    id_aliases: Mapping[str, str] | None = None,
    initial_state: InitialState | None = None,
) -> tuple[TaskGraph, ResourceCatalog]:
    """Return the normalized planner input and a GK-derived catalog."""

    aliases = dict(id_aliases or {})
    gk_records = _gk_records(gk_payload)
    rough_subgoals, raw_partial_order, contract = _executable_contract(
        gk_records, m1_payload
    )

    nodes, raw_to_canonical = _collect_gk_nodes(gk_records, aliases)
    catalog = _derive_catalog(
        nodes,
        rough_subgoals,
        _scene_payload(m1_payload, m0_payload),
        robot_spec_payload or {},
        aliases,
        raw_to_canonical,
    )

    all_ees = sorted(catalog.end_effectors)
    if not all_ees:
        raise ValueError(
            "cannot determine any end effector from GK nodes or robot_spec"
        )

    initial_conditions: dict[tuple[str, tuple[str, ...]], Condition] = {}
    subgoals: list[Subgoal] = []
    detail_ids: set[str] = set()
    rough_ids: set[str] = set()

    for rough_raw in rough_subgoals:
        rough = _mapping(rough_raw)
        rough_id = _required_string(rough, "subgoal_id", "upstream subgoal")
        rough_ids.add(rough_id)
        details = rough.get("details")
        if not isinstance(details, list) or not details:
            raise ValueError(
                f"upstream subgoal {rough_id!r} has no action details"
            )

        tool_id = _selected_tool_id(
            rough,
            aliases,
            raw_to_canonical,
            context=f"upstream subgoal {rough_id!r}",
        )
        rough_targets = [
            _canonical_id(item, aliases, raw_to_canonical)
            for item in _string_list(rough.get("target_ids"))
        ]
        feasible_ee = _feasible_ees_for_group(
            [tool_id] if tool_id is not None else [],
            rough_targets,
            nodes,
            all_ees,
        )

        for detail_raw in details:
            detail = _mapping(detail_raw)
            detail_id = _required_string(detail, "detail_id", "upstream detail")
            if detail_id in detail_ids:
                raise ValueError(f"duplicate upstream detail_id {detail_id!r}")
            detail_ids.add(detail_id)
            group_id = str(detail.get("group_id") or f"G_{rough_id}")
            raw_action = str(detail.get("action_type") or "") or None
            action_type, mode = _split_action_type(raw_action)
            binding = _normalize_binding(
                _mapping(detail.get("binding")), aliases, raw_to_canonical
            )

            preconditions = [
                _condition_from_m1(item, aliases, raw_to_canonical)
                for item in detail.get("pre") or []
            ]
            establish = [
                _condition_from_m1(item, aliases, raw_to_canonical)
                for item in detail.get("establish") or []
            ]
            destroy = [
                _condition_from_m1(item, aliases, raw_to_canonical)
                for item in detail.get("destroy") or []
            ]
            for condition in preconditions:
                if condition.pass_ is True:
                    initial_conditions.setdefault(
                        (condition.type, tuple(condition.args)), condition
                    )

            target_ids = _detail_target_ids(
                binding,
                action_type,
                rough_targets,
                selected_tool_id=tool_id,
            )
            action_type = _normalize_tool_resource_action(
                action_type,
                target_ids=target_ids,
                selected_tool_id=tool_id,
            )
            region = binding.get("?r")
            goal_region_id = region if isinstance(region, str) else None
            if (
                goal_region_id is None
                and action_type == "tool_act"
                and isinstance(rough.get("container_id"), str)
            ):
                goal_region_id = _canonical_id(
                    rough["container_id"], aliases, raw_to_canonical
                )

            subgoals.append(
                Subgoal(
                    subgoal_id=detail_id,
                    description=(
                        str(detail["note"])
                        if detail.get("note") is not None
                        else None
                    ),
                    action_type=action_type,
                    mode=mode,
                    group_id=group_id,
                    source_kg_subgoal_id=rough_id,
                    source_binding=deepcopy(binding),
                    condition_source=contract,
                    target_ids=target_ids,
                    goal_region_id=goal_region_id,
                    tool_id=tool_id,
                    preconditions=preconditions,
                    postconditions=establish,
                    establish=establish,
                    destroy=destroy,
                    feasible_ee=feasible_ee,
                    tool_required=tool_id is not None,
                    tool_selection_source=(
                        "upstream_fixed" if tool_id is not None else "not_required"
                    ),
                    feasible_ee_source="gk",
                )
            )

    gk_ids = {str(record.get("subgoal_id")) for record in gk_records}
    missing_gk = sorted(rough_ids - gk_ids)
    if missing_gk:
        raise ValueError(
            f"upstream subgoals have no matching GK record: {missing_gk}"
        )

    edges = _normalize_partial_order(raw_partial_order, detail_ids)
    normalized_initial_state = _initial_state(
        robot_spec_payload or {},
        catalog,
        initial_conditions,
        explicit=initial_state,
    )

    task_text = m1_payload.get("task")
    if task_text is None:
        task_text = next(
            (record.get("task") for record in gk_records if record.get("task")),
            None,
        )
    task = TaskSpec(
        instruction=(
            task_text
            if isinstance(task_text, str)
            else str(_mapping(task_text).get("instruction") or "")
        )
    )
    task_graph = TaskGraph(
        task=task,
        initial_state=normalized_initial_state,
        subgoals=subgoals,
        order_constraints=OrderConstraints(edges=edges),
        constraints=TaskConstraints(),
        log={
            "adapter": contract,
            "task_id": gk_payload.get("task_id"),
            "gk_proposed_order": deepcopy(
                gk_payload.get("proposed_order") or []
            ),
            "id_aliases": dict(sorted(raw_to_canonical.items())),
            "m1_stats": deepcopy(m1_payload.get("m1_stats") or {}),
        },
    )
    return task_graph, catalog


def _executable_contract(
    gk_records: list[Mapping[str, Any]],
    m1_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[Any], str]:
    """Locate executable subgoals without inventing an upstream selection."""

    legacy = m1_payload.get("m1_subgoals")
    if isinstance(legacy, list) and legacy:
        return (
            [deepcopy(dict(_mapping(item))) for item in legacy],
            list(m1_payload.get("m1_partial_order") or []),
            "gk-m1-v1",
        )

    if not all(isinstance(record.get("details"), list) for record in gk_records):
        raise ValueError(
            "GK bundle has no executable details and companion M1 JSON has "
            "no non-empty m1_subgoals"
        )

    rough_subgoals: list[dict[str, Any]] = []
    partial_order: list[Any] = []
    for record in gk_records:
        rough = deepcopy(dict(record))
        roles = _mapping(record.get("roles"))
        if not any(
            rough.get(field) is not None
            for field in ("tool_id", "selected_tool_id", "selected_tool")
        ) and roles.get("selected_tool") is not None:
            rough["selected_tool"] = roles["selected_tool"]
        if (
            not rough.get("tool_candidate_ids")
            and roles.get("tool_candidates")
        ):
            rough["tool_candidate_ids"] = deepcopy(roles["tool_candidates"])
        if not rough.get("target_ids") and roles.get("target"):
            rough["target_ids"] = deepcopy(roles["target"])
        if (
            rough.get("container_id") is None
            and roles.get("container") is not None
        ):
            rough["container_id"] = roles["container"]
        rough_subgoals.append(rough)
        raw_edges = record.get("partial_order") or []
        if not isinstance(raw_edges, list):
            raise ValueError(
                f"GK subgoal {record.get('subgoal_id')!r} partial_order "
                "must be a list"
            )
        partial_order.extend(deepcopy(raw_edges))
    return rough_subgoals, partial_order, "gk-bundle+m1-scene-v1"


def _scene_payload(
    m1_payload: Mapping[str, Any],
    m0_payload: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if m0_payload is not None:
        return m0_payload
    return m1_payload if isinstance(m1_payload.get("nodes"), list) else {}


def _merge_resource_catalog(
    derived: ResourceCatalog, supplied: ResourceCatalog
) -> ResourceCatalog:
    """Merge GK-derived records with an explicitly supplied catalog.

    Supplied EE and Tool specifications override derived defaults. GK/M0 object
    fields override supplied defaults only when the upstream artifact actually
    provides that field.
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
    for object_id, derived_spec in derived.objects.items():
        if object_id not in objects:
            objects[object_id] = derived_spec
            continue
        values = objects[object_id].model_dump()
        values.update(
            derived_spec.model_dump(include=derived_spec.model_fields_set)
        )
        objects[object_id] = ObjectSpec.model_validate(values)

    return ResourceCatalog(
        end_effectors=end_effectors,
        tools=tools,
        objects=objects,
    )


def _gk_records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if isinstance(payload.get("gk_by_subgoal"), list):
        records = [_mapping(item) for item in payload["gk_by_subgoal"]]
    elif isinstance(payload.get("gks"), list):
        records = [_mapping(item) for item in payload["gks"]]
    elif isinstance(payload.get("subgoal_id"), str) and isinstance(
        payload.get("nodes"), Mapping
    ):
        records = [payload]
    else:
        raise ValueError(
            "unrecognized GK schema: expected gk_by_subgoal, gks, or a "
            "single {subgoal_id,nodes} record"
        )
    if not records:
        raise ValueError("GK contains no subgoal records")
    for record in records:
        _required_string(record, "subgoal_id", "GK record")
    return records


def _collect_gk_nodes(
    records: list[Mapping[str, Any]], aliases: Mapping[str, str]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    result: dict[str, dict[str, Any]] = {}
    raw_to_canonical: dict[str, str] = {}
    for record in records:
        for raw_id, raw_node in _mapping(record.get("nodes")).items():
            if not isinstance(raw_id, str):
                continue
            canonical = _canonical_id(raw_id, aliases, raw_to_canonical)
            raw_to_canonical[raw_id] = canonical
            node = deepcopy(dict(_mapping(raw_node)))
            if canonical in result:
                _deep_fill(result[canonical], node)
            else:
                result[canonical] = node
    return result, raw_to_canonical


def _selected_tool_id(
    rough: Mapping[str, Any],
    aliases: Mapping[str, str],
    raw_to_canonical: dict[str, str],
    *,
    context: str,
) -> str | None:
    """Read an upstream-fixed tool without choosing among candidates."""

    supplied: list[tuple[str, str]] = []
    for field in ("tool_id", "selected_tool_id", "selected_tool"):
        value = rough.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{context} field {field!r} must be a non-empty string")
        supplied.append(
            (field, _canonical_id(value.strip(), aliases, raw_to_canonical))
        )

    selected_values = {value for _, value in supplied}
    if len(selected_values) > 1:
        raise ValueError(
            f"{context} has conflicting selected-tool fields: {dict(supplied)!r}"
        )

    candidate_values = {
        _canonical_id(item, aliases, raw_to_canonical)
        for item in _string_list(
            rough.get("tool_candidate_ids")
            or rough.get("candidate_tool_ids")
        )
    }
    selected = next(iter(selected_values), None)
    if selected is None:
        if candidate_values or _rough_subgoal_uses_tool(rough):
            raise ValueError(
                f"{context} requires a tool but has no selected tool_id; "
                "Task Planner does not select tools"
            )
        return None
    if candidate_values and selected not in candidate_values:
        raise ValueError(
            f"{context} selected tool {selected!r} is not present in the "
            f"declared candidate set {sorted(candidate_values)!r}"
        )
    return selected


def _rough_subgoal_uses_tool(rough: Mapping[str, Any]) -> bool:
    for detail_raw in rough.get("details") or []:
        detail = _mapping(detail_raw)
        action_type = detail.get("action_type")
        if isinstance(action_type, str) and action_type.startswith("tool_act"):
            return True
        binding = _mapping(detail.get("binding"))
        if any(
            isinstance(value, str) and "?tool" in value
            for value in binding.values()
        ):
            return True
    return False


def _derive_catalog(
    nodes: dict[str, dict[str, Any]],
    rough_subgoals: list[Any],
    m0: Mapping[str, Any],
    robot_spec: Mapping[str, Any],
    aliases: Mapping[str, str],
    raw_to_canonical: dict[str, str],
) -> ResourceCatalog:
    tool_ids = {
        tool_id
        for rough_raw in rough_subgoals
        if (
            tool_id := _selected_tool_id(
                _mapping(rough_raw),
                aliases,
                raw_to_canonical,
                context="M1 rough subgoal",
            )
        )
        is not None
    }

    ee_ids: set[str] = set()
    compatible_by_ee: dict[str, set[str]] = {}
    tools: dict[str, ToolSpec] = {}
    for tool_id in sorted(tool_ids):
        node = _mapping(nodes.get(tool_id))
        compatible = sorted(
            ee
            for ee, record in _mapping(node.get("ee")).items()
            if isinstance(ee, str) and _is_feasible(record)
        )
        ee_ids.update(compatible)
        for ee in compatible:
            compatible_by_ee.setdefault(ee, set()).add(tool_id)
        mass = node.get("mass_kg")
        tools[tool_id] = ToolSpec(
            mass=float(mass) if isinstance(mass, (int, float)) else None,
            compatible_ee=compatible,
            home_slot=f"tool-slot-{tool_id}",
            gk=deepcopy(dict(node)),
        )

    robot_records = _robot_ee_records(robot_spec)
    ee_ids.update(robot_records)
    for node in nodes.values():
        ee_ids.update(
            ee for ee in _mapping(node.get("ee")) if isinstance(ee, str)
        )

    # Missing per-tool EE results mean "not measured", not "incompatible".
    # Leave the uncertainty to the policy/motion checks while still producing
    # searchable candidates.
    for tool_id, tool in tools.items():
        if not tool.compatible_ee:
            tool.compatible_ee = sorted(ee_ids)
            for ee in ee_ids:
                compatible_by_ee.setdefault(ee, set()).add(tool_id)

    end_effectors: dict[str, EndEffectorSpec] = {}
    for ee in sorted(ee_ids):
        raw = _mapping(robot_records.get(ee))
        raw_tools = {
            _canonical_id(item, aliases, raw_to_canonical)
            for item in _string_list(raw.get("compatible_tools"))
        }
        payload = _first_number(
            raw.get("payload"),
            raw.get("payload_kg"),
            raw.get("max_payload_kg"),
        )
        end_effectors[ee] = EndEffectorSpec(
            capabilities=_string_list(raw.get("capabilities")),
            payload=payload,
            compatible_tools=sorted(
                raw_tools | compatible_by_ee.get(ee, set())
            ),
            home_slot=(
                str(raw["home_slot"])
                if isinstance(raw.get("home_slot"), str)
                else f"slot-{ee}"
            ),
            robot_spec=deepcopy(dict(raw)),
        )

    objects: dict[str, ObjectSpec] = {}
    for raw_node in m0.get("nodes") or []:
        record = _mapping(raw_node)
        raw_id = record.get("id")
        if not isinstance(raw_id, str):
            continue
        object_id = _canonical_id(raw_id, aliases, raw_to_canonical)
        raw_to_canonical[raw_id] = object_id
        objects[object_id] = ObjectSpec(
            mass_kg=_optional_float(record.get("mass_kg")),
            bbox_mm=_positive_bbox(record.get("bbox_mm")),
            material=(
                str(record["material"])
                if isinstance(record.get("material"), str)
                else None
            ),
            center_mm=deepcopy(record.get("center_mm")),
            object_class=record.get("class"),
            source_id=raw_id,
        )
    for object_id, raw_node in nodes.items():
        node = _mapping(raw_node)
        geometry = _mapping(node.get("geometry"))
        bbox = _positive_bbox(
            geometry.get("extents_mm") or node.get("bbox_mm")
        )
        values: dict[str, Any] = {
            "mass_kg": _optional_float(node.get("mass_kg")),
            "bbox_mm": bbox,
            "material": (
                str(node["material"])
                if isinstance(node.get("material"), str)
                else None
            ),
            "gk": deepcopy(dict(node)),
        }
        if object_id in objects:
            base = objects[object_id].model_dump()
            base.update({key: value for key, value in values.items() if value is not None})
            objects[object_id] = ObjectSpec.model_validate(base)
        else:
            objects[object_id] = ObjectSpec.model_validate(values)

    return ResourceCatalog(
        end_effectors=end_effectors,
        tools=tools,
        objects=objects,
    )


def _condition_from_m1(
    raw: Any,
    aliases: Mapping[str, str],
    raw_to_canonical: dict[str, str],
) -> Condition:
    record: Mapping[str, Any]
    if isinstance(raw, str):
        record = {"expr": raw}
    else:
        record = _mapping(raw)
    expr = record.get("expr") if isinstance(record.get("expr"), str) else ""
    head = record.get("head")
    condition_type = str(head) if isinstance(head, str) and head else ""
    args = _string_list(record.get("args"))
    if not condition_type or not args:
        parsed_head, parsed_args = _parse_expr(expr)
        condition_type = condition_type or parsed_head
        if not args:
            args = parsed_args
    if not condition_type:
        raise ValueError(f"M1 condition has no head/type: {record!r}")
    normalized_args = [
        _normalize_condition_arg(arg, aliases, raw_to_canonical) for arg in args
    ]
    status = str(record.get("status") or "").lower()
    pass_: bool | None = None
    if status in _TRUE_STATUSES:
        pass_ = True
    elif status in _FALSE_STATUSES:
        pass_ = False
    eval_by = str(record.get("eval_by") or "") or None
    deferred = pass_ is None and eval_by in {"m2", "motion"}
    return Condition(
        cond_id=(
            str(record["id"])
            if isinstance(record.get("id"), str)
            else None
        ),
        type=condition_type,
        args=normalized_args,
        pass_=pass_,
        eval_by=eval_by,
        needs_observation=deferred and eval_by == "m2",
        nl=expr or None,
        status=record.get("status"),
        evidence=deepcopy(record.get("evidence") or []),
    )


def _parse_expr(expr: str) -> tuple[str, list[str]]:
    expr = expr.strip()
    if not expr:
        return "", []
    if "(" not in expr or not expr.endswith(")"):
        return expr, []
    head, remainder = expr.split("(", 1)
    inner = remainder[:-1]
    args: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(inner):
        if char in "{[(":
            depth += 1
        elif char in "}])":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            args.append(inner[start:index].strip())
            start = index + 1
    tail = inner[start:].strip()
    if tail:
        args.append(tail)
    return head.strip(), args


def _normalize_condition_arg(
    value: str,
    aliases: Mapping[str, str],
    raw_to_canonical: dict[str, str],
) -> str:
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        members = [
            _canonical_id(item.strip(), aliases, raw_to_canonical)
            for item in value[1:-1].split(",")
            if item.strip()
        ]
        return "{" + ",".join(members) + "}"
    return _canonical_id(value, aliases, raw_to_canonical)


def _normalize_binding(
    binding: Mapping[str, Any],
    aliases: Mapping[str, str],
    raw_to_canonical: dict[str, str],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in binding.items():
        if isinstance(value, str):
            if value.startswith("{") and value.endswith("}"):
                normalized[key] = [
                    _canonical_id(item.strip(), aliases, raw_to_canonical)
                    for item in value[1:-1].split(",")
                    if item.strip()
                ]
            else:
                normalized[key] = _canonical_id(
                    value, aliases, raw_to_canonical
                )
        elif isinstance(value, list):
            normalized[key] = [
                _canonical_id(item, aliases, raw_to_canonical)
                if isinstance(item, str)
                else deepcopy(item)
                for item in value
            ]
        else:
            normalized[key] = deepcopy(value)
    return normalized


def _detail_target_ids(
    binding: Mapping[str, Any],
    action_type: str | None,
    rough_targets: list[str],
    *,
    selected_tool_id: str | None,
) -> list[str]:
    targets = binding.get("?targets")
    if isinstance(targets, list):
        return [
            selected_tool_id if item == "?tool" else item
            for item in targets
            if isinstance(item, str) and (item != "?tool" or selected_tool_id)
        ]
    obj = binding.get("?o")
    if obj == "?tool" and selected_tool_id is not None:
        return [selected_tool_id]
    if isinstance(obj, str) and not obj.startswith("?"):
        return [obj]
    if action_type == "tool_act":
        return list(rough_targets)
    return []


def _normalize_tool_resource_action(
    action_type: str | None,
    *,
    target_ids: list[str],
    selected_tool_id: str | None,
) -> str | None:
    """Give Tool acquisition/release one unambiguous physical operation."""

    if selected_tool_id is None or target_ids != [selected_tool_id]:
        return action_type
    normalized = str(action_type or "").strip().lower().replace("-", "_")
    if normalized in {"acquire", "grasp", "pick", "pick_object"}:
        return "PICK_TOOL"
    if normalized in {"place", "release", "return"}:
        return "RETURN_TOOL"
    return action_type


def _feasible_ees_for_group(
    tools: list[str],
    targets: list[str],
    nodes: Mapping[str, Mapping[str, Any]],
    fallback: list[str],
) -> list[str]:
    owners = tools or targets
    feasible: set[str] = set()
    for owner in owners:
        for ee, record in _mapping(_mapping(nodes.get(owner)).get("ee")).items():
            if isinstance(ee, str) and _is_feasible(record):
                feasible.add(ee)
    return sorted(feasible) if feasible else list(fallback)


def _normalize_partial_order(
    raw_edges: Any, detail_ids: set[str]
) -> list[dict[str, Any]]:
    if not isinstance(raw_edges, list):
        raise ValueError("M1 m1_partial_order must be a list")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_edges:
        edge = _mapping(raw)
        pred = edge.get("from")
        succ = edge.get("to")
        if not isinstance(pred, str) or not isinstance(succ, str):
            raise ValueError(f"malformed M1 partial-order edge: {edge!r}")
        if pred not in detail_ids or succ not in detail_ids:
            raise ValueError(
                f"M1 edge {pred!r}->{succ!r} references an unknown detail"
            )
        if (pred, succ) in seen:
            continue
        seen.add((pred, succ))
        result.append(
            {
                "from": pred,
                "to": succ,
                "reason": edge.get("why") or edge.get("reason"),
            }
        )
    return result


def _canonical_id(
    value: str,
    aliases: Mapping[str, str],
    raw_to_canonical: Mapping[str, str] | None = None,
) -> str:
    if value in aliases:
        return aliases[value]
    if raw_to_canonical is not None and value in raw_to_canonical:
        return raw_to_canonical[value]
    if value.startswith("?") or value in {"tool_rest", "world"}:
        return value
    if value.startswith("obj_"):
        remainder = value[4:]
        if "_" in remainder:
            _object_class, instance_id = remainder.split("_", 1)
            return instance_id
    return value


def _robot_ee_records(robot_spec: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("ee_pool", "end_effectors", "ees"):
        value = robot_spec.get(key)
        if isinstance(value, Mapping):
            return {str(ee): raw for ee, raw in value.items()}
        if isinstance(value, list):
            result: dict[str, Any] = {}
            for item in value:
                record = _mapping(item)
                ee_id = (
                    record.get("ee_id")
                    or record.get("id")
                    or record.get("name")
                )
                if isinstance(ee_id, str):
                    result[ee_id] = record
            if result:
                return result
    nested = _mapping(robot_spec.get("robot"))
    if nested:
        return _robot_ee_records(nested)
    return {}


def _initial_state(
    robot_spec: Mapping[str, Any],
    catalog: ResourceCatalog,
    initial_conditions: Mapping[tuple[str, tuple[str, ...]], Condition],
    *,
    explicit: InitialState | None,
) -> InitialState:
    """Build initial state without guessing a robot or resource identity."""

    if explicit is not None:
        facts = {
            (condition.type, tuple(condition.args)): condition
            for condition in initial_conditions.values()
        }
        facts.update(
            {
                (condition.type, tuple(condition.args)): condition
                for condition in explicit.facts
            }
        )
        return explicit.model_copy(
            deep=True,
            update={"facts": list(facts.values())},
        )

    all_ees = sorted(catalog.end_effectors)
    current_ee = _current_ee(robot_spec, all_ees)
    state_record = _mapping(robot_spec.get("robot_state"))
    nested_robot = _mapping(robot_spec.get("robot"))
    sources = (state_record, robot_spec, nested_robot)

    held_tool: str | None = None
    for source in sources:
        if "held_tool" not in source:
            continue
        value = source["held_tool"]
        if value is not None and not isinstance(value, str):
            raise ValueError("initial held_tool must be a string or null")
        held_tool = value
        break

    facts = dict(initial_conditions)
    hand_empty_value: bool | None = None
    for source in sources:
        value = source.get("hand_empty")
        if value is None:
            continue
        if not isinstance(value, bool):
            raise ValueError("initial hand_empty must be a boolean")
        hand_empty_value = value
        break
    if hand_empty_value is None and ("hand_empty", ()) not in facts:
        raise ValueError(
            "initial hand state is missing; set robot_spec.hand_empty or "
            "provide --initial-state facts"
        )
    if hand_empty_value is True:
        facts.setdefault(
            ("hand_empty", ()),
            Condition(
                cond_id="GK_INIT_hand_empty",
                type="hand_empty",
                args=[],
                pass_=True,
                eval_by="robot_state",
            ),
        )
    elif hand_empty_value is False:
        facts.pop(("hand_empty", ()), None)

    rack = {
        spec.home_slot: ee
        for ee, spec in sorted(catalog.end_effectors.items())
        if spec.home_slot
    }
    raw_occupancy = next(
        (
            source.get("rack_occupancy")
            for source in sources
            if source.get("rack_occupancy") is not None
        ),
        None,
    )
    if raw_occupancy is not None and not isinstance(raw_occupancy, Mapping):
        raise ValueError("initial rack_occupancy must be a JSON object")
    rack_occupancy = (
        {str(slot): str(value) for slot, value in raw_occupancy.items()}
        if isinstance(raw_occupancy, Mapping)
        else None
    )
    return InitialState(
        current_ee=current_ee,
        rack=rack,
        facts=list(facts.values()),
        held_tool=held_tool,
        rack_occupancy=rack_occupancy,
    )


def _current_ee(
    robot_spec: Mapping[str, Any], all_ees: list[str]
) -> str | None:
    sources = (
        robot_spec,
        _mapping(robot_spec.get("robot_state")),
        _mapping(robot_spec.get("robot")),
    )
    for source in sources:
        if "current_ee" not in source:
            continue
        value = source["current_ee"]
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("initial current_ee must be a string or null")
        if value not in all_ees:
            raise ValueError(
                f"initial current_ee {value!r} is not in the EE catalog {all_ees!r}"
            )
        return value
    raise ValueError(
        "initial current_ee is missing; set robot_spec.current_ee or provide "
        "--initial-state"
    )


def _is_feasible(record: Any) -> bool:
    if isinstance(record, bool):
        return record
    mapping = _mapping(record)
    value = mapping.get("feasible")
    if isinstance(value, bool):
        return value
    status = str(mapping.get("status") or "").lower()
    return status in _TRUE_STATUSES


def _positive_bbox(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if not all(isinstance(item, (int, float)) and item > 0 for item in value):
        return None
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and value >= 0 else None


def _first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)
    return None


def _split_action_type(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    if ":" not in value:
        return value, None
    action, mode = value.split(":", 1)
    return action or None, mode or None


def _deep_fill(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Fill missing GK fields without erasing earlier non-null knowledge."""
    for key, value in source.items():
        if key not in target or target[key] is None:
            target[key] = deepcopy(value)
        elif isinstance(target[key], dict) and isinstance(value, Mapping):
            _deep_fill(target[key], value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _required_string(
    item: Mapping[str, Any], key: str, owner: str
) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} has no valid {key!r}: {item!r}")
    return value
