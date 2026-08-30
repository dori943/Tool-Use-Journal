"""Transition primitives for EE changes and upstream-fixed tool use.

A search edge prepares the selected EE and fixed tool, moves to the workspace,
and executes one subgoal. Contact-pose state is deliberately outside Task Planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tuj.m4_taskplanner.candidate_provider import Candidate
from tuj.m4_taskplanner.conditions import requires_runtime_verification
from tuj.m4_taskplanner.cost import CostVector
from tuj.m4_taskplanner.diagnostics import ReasonCode, Rejection, make_rejection
from tuj.m4_taskplanner.models import PlanningPolicy, ResourceCatalog, Subgoal
from tuj.m4_taskplanner.state import SearchState


class P:
    """Primitive action names."""

    KEEP_EE = "KEEP_EE"
    KEEP_TOOL = "KEEP_TOOL"
    MOVE_TO_TOOL_RACK = "MOVE_TO_TOOL_RACK"
    RETURN_TOOL = "RETURN_TOOL"
    PICK_TOOL = "PICK_TOOL"
    MOVE_TO_EE_RACK = "MOVE_TO_EE_RACK"
    DETACH_EE = "DETACH_EE"
    ATTACH_EE = "ATTACH_EE"
    VERIFY_ATTACHMENT = "VERIFY_ATTACHMENT"
    MOVE_TO_WORKSPACE = "MOVE_TO_WORKSPACE"
    EXECUTE_SUBGOAL = "EXECUTE_SUBGOAL"
    TERMINAL_RETURN_TOOL = "TERMINAL_RETURN_TOOL"
    TERMINAL_RESTORE_EE = "TERMINAL_RESTORE_EE"


@dataclass(frozen=True, slots=True)
class Primitive:
    action: str
    parameters: tuple[tuple[str, Any], ...] = ()
    preconditions: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    cost_delta: CostVector = field(default_factory=CostVector)
    verification_required: bool = False

    def parameters_dict(self) -> dict[str, Any]:
        return dict(self.parameters)


@dataclass(slots=True)
class TransitionContext:
    catalog: ResourceCatalog
    policy: PlanningPolicy
    initial_ee: str | None


@dataclass(slots=True)
class TransitionResult:
    feasible: bool
    primitives: tuple[Primitive, ...] = ()
    cost: CostVector = field(default_factory=CostVector)
    rejection: Rejection | None = None
    next_held_tool: str | None = None
    next_rack_signature: tuple[tuple[str, str], ...] | None = None


def transition_signature(
    state: SearchState, candidate: Candidate
) -> tuple[str, str, str]:
    """Identity of a transition for no-good bans and geometry caching."""
    return state.current_ee or "", state.held_tool or "", candidate.candidate_id


def held_object_ids(
    state: SearchState, context: TransitionContext
) -> frozenset[str]:
    """Return objects represented by symbolic ``holding(object)`` facts."""
    return frozenset(
        args[0]
        for fluent, args in state.symbolic_facts
        if fluent == "holding"
        and args
        and not args[0].startswith("?")
        and args[0] not in context.catalog.tools
    )


def _prim(
    action: str,
    parameters: dict[str, Any] | None = None,
    preconditions: tuple[str, ...] = (),
    effects: tuple[str, ...] = (),
    cost: CostVector | None = None,
    verify: bool = False,
) -> Primitive:
    return Primitive(
        action=action,
        parameters=tuple(sorted((parameters or {}).items())),
        preconditions=preconditions,
        effects=effects,
        cost_delta=cost or CostVector(),
        verification_required=verify,
    )


def _infeasible(
    code: ReasonCode, message: str, candidate: Candidate
) -> TransitionResult:
    return TransitionResult(
        feasible=False,
        rejection=make_rejection(
            "transition",
            code,
            message,
            subgoal_id=candidate.subgoal_id,
            candidate_id=candidate.candidate_id,
        ),
    )


def _rack_after_exchange(
    rack: tuple[tuple[str, str], ...] | None,
    catalog: ResourceCatalog,
    old_ee: str,
    new_ee: str,
) -> tuple[tuple[tuple[str, str], ...] | None, str | None]:
    if rack is None:
        return None, None
    occupancy = dict(rack)
    old_slot = catalog.end_effectors[old_ee].home_slot
    new_slot = catalog.end_effectors[new_ee].home_slot
    if old_slot is None or new_slot is None:
        return None, f"missing home slot for {old_ee!r} or {new_ee!r}"
    if occupancy.get(old_slot) != "empty":
        return None, (
            f"slot {old_slot!r} is {occupancy.get(old_slot)!r}, cannot dock "
            f"{old_ee!r}"
        )
    if occupancy.get(new_slot) != new_ee:
        return None, (
            f"slot {new_slot!r} holds {occupancy.get(new_slot)!r}, not "
            f"{new_ee!r}"
        )
    occupancy[old_slot] = old_ee
    occupancy[new_slot] = "empty"
    return tuple(sorted(occupancy.items())), None


def _rack_after_initial_attach(
    rack: tuple[tuple[str, str], ...] | None,
    catalog: ResourceCatalog,
    new_ee: str,
) -> tuple[tuple[tuple[str, str], ...] | None, str | None]:
    """Remove the first selected EE from its home slot.

    Unlike a normal exchange there is no mounted EE to detach and therefore no
    newly occupied rack slot.
    """
    if rack is None:
        return None, None
    occupancy = dict(rack)
    new_slot = catalog.end_effectors[new_ee].home_slot
    if new_slot is None:
        return None, f"missing home slot for {new_ee!r}"
    if occupancy.get(new_slot) != new_ee:
        return None, (
            f"slot {new_slot!r} holds {occupancy.get(new_slot)!r}, not "
            f"{new_ee!r}"
        )
    occupancy[new_slot] = "empty"
    return tuple(sorted(occupancy.items())), None


def build_transition(
    state: SearchState,
    candidate: Candidate,
    context: TransitionContext,
    subgoal: Subgoal | None = None,
) -> TransitionResult:
    """Prepare ``candidate.ee/tool`` and execute its subgoal."""
    prims: list[Primitive] = []
    motion = context.policy.motion_costs
    initial_attach = state.current_ee is None
    ee_change = not initial_attach and state.current_ee != candidate.ee
    operation = str(candidate.action_type or "").strip().upper().replace("-", "_")
    acquires_tool = operation == "PICK_TOOL"
    releases_tool = operation in {"RETURN_TOOL", "TERMINAL_RETURN_TOOL"}
    if (acquires_tool or releases_tool) and candidate.tool is None:
        return _infeasible(
            ReasonCode.TOOL_REQUIRED_MISSING,
            f"explicit {operation} subgoal has no grounded tool",
            candidate,
        )
    if releases_tool and state.held_tool != candidate.tool:
        return _infeasible(
            ReasonCode.TOOL_MISMATCH,
            f"cannot return tool {candidate.tool!r}; held tool is "
            f"{state.held_tool!r}",
            candidate,
        )
    required_tool = None if acquires_tool else candidate.tool
    resulting_tool = None if releases_tool else candidate.tool
    held_objects = held_object_ids(state, context)
    rack = state.rack_signature

    if held_objects and (initial_attach or ee_change):
        return _infeasible(
            ReasonCode.OBJECT_HELD_EE_SWITCH,
            f"holding object(s) {sorted(held_objects)!r}; EE attach/detach is "
            "forbidden",
            candidate,
        )
    if held_objects and state.held_tool != required_tool:
        return _infeasible(
            ReasonCode.OBJECT_HELD_TOOL_ACQUIRE,
            f"holding object(s) {sorted(held_objects)!r}; changing the held "
            "tool is forbidden",
            candidate,
        )

    cur_tool = state.held_tool
    next_rack = rack
    if initial_attach:
        # With no mounted EE there can be no held tool or object (validated at
        # input time).  Attach the search-selected EE without charging an EE
        # *exchange*; only subsequent EE changes increment ee_switches.
        next_rack, rack_error = _rack_after_initial_attach(
            rack, context.catalog, candidate.ee
        )
        if rack_error is not None:
            return _infeasible(
                ReasonCode.RACK_SLOT_UNAVAILABLE, rack_error, candidate
            )
        prims.extend(
            [
                _prim(
                    P.MOVE_TO_EE_RACK,
                    cost=CostVector(motion_cost=motion.move_to_ee_rack),
                ),
                _prim(
                    P.ATTACH_EE,
                    {"ee": candidate.ee},
                    preconditions=(f"docked({candidate.ee})",),
                    effects=(f"mounted({candidate.ee})",),
                ),
                _prim(P.VERIFY_ATTACHMENT, {"ee": candidate.ee}, verify=True),
            ]
        )
        if required_tool is not None:
            prims.extend(
                _pick_tool_prims(
                    candidate,
                    motion.move_to_tool_rack,
                    switch=True,
                )
            )
    elif ee_change:
        if cur_tool is not None:
            prims.append(
                _prim(
                    P.MOVE_TO_TOOL_RACK,
                    cost=CostVector(motion_cost=motion.move_to_tool_rack),
                )
            )
            prims.append(
                _prim(
                    P.RETURN_TOOL,
                    {"tool": cur_tool},
                    preconditions=(f"holding_tool({cur_tool})",),
                    effects=("hand_empty",),
                    cost=CostVector(
                        tool_switches=0 if required_tool is not None else 1
                    ),
                )
            )
        next_rack, rack_error = _rack_after_exchange(
            rack, context.catalog, state.current_ee, candidate.ee
        )
        if rack_error is not None:
            return _infeasible(
                ReasonCode.RACK_SLOT_UNAVAILABLE, rack_error, candidate
            )
        prims.extend(
            [
                _prim(
                    P.MOVE_TO_EE_RACK,
                    cost=CostVector(motion_cost=motion.move_to_ee_rack),
                ),
                _prim(
                    P.DETACH_EE,
                    {"ee": state.current_ee},
                    preconditions=("hand_empty", f"mounted({state.current_ee})"),
                    effects=(f"docked({state.current_ee})",),
                ),
                _prim(
                    P.ATTACH_EE,
                    {"ee": candidate.ee},
                    preconditions=(f"docked({candidate.ee})",),
                    effects=(f"mounted({candidate.ee})",),
                    cost=CostVector(ee_switches=1),
                ),
                _prim(P.VERIFY_ATTACHMENT, {"ee": candidate.ee}, verify=True),
            ]
        )
        if required_tool is not None:
            prims.extend(
                _pick_tool_prims(
                    candidate,
                    motion.move_to_tool_rack,
                    switch=cur_tool != required_tool,
                )
            )
    elif cur_tool == required_tool:
        prims.append(_prim(P.KEEP_EE, {"ee": state.current_ee}))
        if cur_tool is not None:
            prims.append(_prim(P.KEEP_TOOL, {"tool": cur_tool}))
    else:
        prims.append(
            _prim(
                P.MOVE_TO_TOOL_RACK,
                cost=CostVector(motion_cost=motion.move_to_tool_rack),
            )
        )
        if cur_tool is not None:
            prims.append(
                _prim(
                    P.RETURN_TOOL,
                    {"tool": cur_tool},
                    preconditions=(f"holding_tool({cur_tool})",),
                    effects=("hand_empty",),
                    cost=CostVector(
                        tool_switches=0 if required_tool is not None else 1
                    ),
                )
            )
        if required_tool is not None:
            prims.extend(_pick_tool_prims(candidate, None, switch=True))

    if acquires_tool or releases_tool:
        prims.extend(
            _tool_resource_and_execute(
                candidate,
                motion.move_to_tool_rack,
                switch=state.held_tool != resulting_tool,
                subgoal=subgoal,
            )
        )
    else:
        prims.extend(
            _workspace_and_execute(
                candidate,
                motion.move_to_workspace,
                subgoal,
            )
        )
    return TransitionResult(
        feasible=True,
        primitives=tuple(prims),
        cost=_sum_cost(prims),
        next_held_tool=resulting_tool,
        next_rack_signature=next_rack,
    )


def _pick_tool_prims(
    candidate: Candidate, move_cost: int | None, switch: bool
) -> list[Primitive]:
    prims: list[Primitive] = []
    if move_cost is not None:
        prims.append(
            _prim(P.MOVE_TO_TOOL_RACK, cost=CostVector(motion_cost=move_cost))
        )
    prims.append(
        _prim(
            P.PICK_TOOL,
            {"tool": candidate.tool},
            preconditions=("hand_empty", f"tool_at_home({candidate.tool})"),
            effects=(f"holding_tool({candidate.tool})",),
            cost=CostVector(tool_switches=1 if switch else 0),
        )
    )
    return prims


def _workspace_and_execute(
    candidate: Candidate, workspace_cost: int, subgoal: Subgoal | None = None
) -> list[Primitive]:
    verify = bool(
        subgoal
        and any(requires_runtime_verification(cond) for cond in subgoal.preconditions)
    )
    return [
        _prim(
            P.MOVE_TO_WORKSPACE,
            {"subgoal": candidate.subgoal_id},
            cost=CostVector(motion_cost=workspace_cost),
        ),
        _prim(
            P.EXECUTE_SUBGOAL,
            {
                "subgoal": candidate.subgoal_id,
                "candidate": candidate.candidate_id,
                "ee": candidate.ee,
                "tool": candidate.tool,
                "action_parameters": dict(candidate.metadata),
            },
            cost=CostVector(execution_cost=candidate.nominal_execution_cost),
            verify=verify,
        ),
    ]


def _tool_resource_and_execute(
    candidate: Candidate,
    tool_rack_cost: int,
    *,
    switch: bool,
    subgoal: Subgoal | None = None,
) -> list[Primitive]:
    """Execute an explicit PICK_TOOL/RETURN_TOOL subgoal exactly once."""

    verify = bool(
        subgoal
        and any(requires_runtime_verification(cond) for cond in subgoal.preconditions)
    )
    return [
        _prim(
            P.MOVE_TO_TOOL_RACK,
            {"tool": candidate.tool},
            cost=CostVector(motion_cost=tool_rack_cost),
        ),
        _prim(
            P.EXECUTE_SUBGOAL,
            {
                "subgoal": candidate.subgoal_id,
                "candidate": candidate.candidate_id,
                "ee": candidate.ee,
                "tool": candidate.tool,
                "action_parameters": dict(candidate.metadata),
            },
            cost=CostVector(
                tool_switches=1 if switch else 0,
                execution_cost=candidate.nominal_execution_cost,
            ),
            verify=verify,
        ),
    ]


def _sum_cost(prims: list[Primitive]) -> CostVector:
    total = CostVector()
    for primitive in prims:
        total = total + primitive.cost_delta
    return total


@dataclass(slots=True)
class TerminalResult:
    feasible: bool
    primitives: tuple[Primitive, ...] = ()
    cost: CostVector = field(default_factory=CostVector)
    rejection: Rejection | None = None
    next_ee: str | None = None
    next_rack_signature: tuple[tuple[str, str], ...] | None = None


def build_terminal_transition(
    state: SearchState, context: TransitionContext
) -> TerminalResult:
    """Synthetic terminal edge enforcing the terminal policy at real cost."""
    policy = context.policy.terminal
    motion = context.policy.motion_costs
    prims: list[Primitive] = []
    next_ee = state.current_ee
    next_rack = state.rack_signature
    held_objects = held_object_ids(state, context)

    if policy.require_empty_object_hand_at_end and held_objects:
        return TerminalResult(
            feasible=False,
            rejection=make_rejection(
                "terminal",
                ReasonCode.TERMINAL_OBJECT_HELD,
                f"still holding object(s) {sorted(held_objects)!r} at the end",
            ),
        )

    needs_restore = (
        policy.restore_initial_ee_at_end
        and context.initial_ee is not None
        and state.current_ee != context.initial_ee
    )
    if needs_restore and held_objects:
        return TerminalResult(
            feasible=False,
            rejection=make_rejection(
                "terminal",
                ReasonCode.TERMINAL_OBJECT_HELD,
                "cannot restore the initial EE while holding object(s) "
                f"{sorted(held_objects)!r}",
            ),
        )
    must_return_tool = state.held_tool is not None and (
        policy.return_tool_at_end or needs_restore
    )
    if must_return_tool and held_objects:
        return TerminalResult(
            feasible=False,
            rejection=make_rejection(
                "terminal",
                ReasonCode.TERMINAL_OBJECT_HELD,
                f"cannot return tool {state.held_tool!r} while holding "
                f"object(s) {sorted(held_objects)!r}",
            ),
        )
    if must_return_tool:
        prims.append(
            _prim(
                P.TERMINAL_RETURN_TOOL,
                {"tool": state.held_tool},
                preconditions=(f"holding_tool({state.held_tool})",),
                effects=("hand_empty",),
                cost=CostVector(
                    tool_switches=1, motion_cost=motion.move_to_tool_rack
                ),
            )
        )
    if needs_restore:
        next_rack, rack_error = _rack_after_exchange(
            next_rack, context.catalog, state.current_ee, context.initial_ee
        )
        if rack_error is not None:
            return TerminalResult(
                feasible=False,
                rejection=make_rejection(
                    "terminal", ReasonCode.RACK_SLOT_UNAVAILABLE, rack_error
                ),
            )
        prims.append(
            _prim(
                P.TERMINAL_RESTORE_EE,
                {"from": state.current_ee, "to": context.initial_ee},
                preconditions=("hand_empty",),
                effects=(f"mounted({context.initial_ee})",),
                cost=CostVector(
                    ee_switches=1, motion_cost=motion.move_to_ee_rack
                ),
                verify=True,
            )
        )
        next_ee = context.initial_ee
    return TerminalResult(
        feasible=True,
        primitives=tuple(prims),
        cost=_sum_cost(prims),
        next_ee=next_ee,
        next_rack_signature=next_rack,
    )
