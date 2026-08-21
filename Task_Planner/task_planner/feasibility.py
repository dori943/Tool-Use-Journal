"""Eager symbolic/static checks and lazy geometric checks with caching.

Static checks run once per candidate right after generation. Geometric checks
run only when the search actually expands a node (lazy), and results are
cached per (scene_signature, candidate) and per transition context.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Protocol

from task_planner.candidate_provider import Candidate
from task_planner.conditions import subgoal_required_wrench
from task_planner.diagnostics import ReasonCode, Rejection, make_rejection
from task_planner.models import PlanningPolicy, ResourceCatalog, Subgoal


class FeasibilityStatus(str, enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class CheckResult:
    status: FeasibilityStatus
    reason_code: ReasonCode | None = None
    message: str = ""

    @classmethod
    def ok(cls) -> "CheckResult":
        return cls(FeasibilityStatus.PASS)


class StaticFeasibilityChecker:
    """Schema/inventory/capability/compatibility/payload/rack checks.

    Payload (carrying) and task wrench (force/torque delivery) are checked
    separately. With no numeric data present the corresponding check reports
    UNKNOWN and the ``unknown_feasibility_policy`` decides (default: reject).
    Checks that have nothing to verify (no tool, no wrench requirement) PASS
    vacuously.
    """

    def __init__(
        self,
        catalog: ResourceCatalog,
        policy: PlanningPolicy,
        group_feasible_ee: dict[str, frozenset[str]],
    ) -> None:
        self._catalog = catalog
        self._policy = policy
        self._group_feasible_ee = group_feasible_ee

    def check(self, candidate: Candidate, subgoal: Subgoal) -> Rejection | None:
        """Return a Rejection if the candidate is statically infeasible."""
        unknowns: list[CheckResult] = []
        for result in self._run_checks(candidate, subgoal):
            if result.status is FeasibilityStatus.FAIL:
                return self._to_rejection(result, candidate, subgoal)
            if result.status is FeasibilityStatus.UNKNOWN:
                unknowns.append(result)
        if unknowns and self._policy.unknown_feasibility_policy == "reject":
            first = unknowns[0]
            return make_rejection(
                "candidate",
                ReasonCode.UNKNOWN_FEASIBILITY_REJECTED,
                f"unknown feasibility rejected by policy: {first.message}",
                subgoal_id=subgoal.subgoal_id,
                candidate_id=candidate.candidate_id,
            )
        # "allow" and "defer" both let the candidate through eagerly; "defer"
        # simply leaves final judgement to the lazy geometric stage.
        return None

    def _to_rejection(
        self, result: CheckResult, candidate: Candidate, subgoal: Subgoal
    ) -> Rejection:
        return make_rejection(
            "candidate",
            result.reason_code or ReasonCode.GEOMETRY_FAILED,
            result.message,
            subgoal_id=subgoal.subgoal_id,
            candidate_id=candidate.candidate_id,
        )

    def _run_checks(self, cand: Candidate, sg: Subgoal) -> list[CheckResult]:
        results: list[CheckResult] = []
        catalog = self._catalog

        # Inventory existence.
        ee_spec = catalog.end_effectors.get(cand.ee)
        if ee_spec is None:
            return [
                CheckResult(
                    FeasibilityStatus.FAIL,
                    ReasonCode.UNKNOWN_EE,
                    f"EE {cand.ee!r} not in catalog",
                )
            ]
        if sg.tool_required and cand.tool is None:
            return [
                CheckResult(
                    FeasibilityStatus.FAIL,
                    ReasonCode.TOOL_REQUIRED_MISSING,
                    "tool_required=true but candidate has no tool",
                )
            ]
        if sg.tool_id is not None and cand.tool != sg.tool_id:
            code = (
                ReasonCode.TOOL_REQUIRED_MISSING
                if sg.tool_id is not None and cand.tool is None
                else ReasonCode.TOOL_MISMATCH
            )
            return [
                CheckResult(
                    FeasibilityStatus.FAIL,
                    code,
                    f"subgoal fixes tool to {sg.tool_id!r} from "
                    f"{sg.tool_selection_source}; candidate uses {cand.tool!r}",
                )
            ]
        if sg.tool_id is None and cand.tool is not None:
            return [
                CheckResult(
                    FeasibilityStatus.FAIL,
                    ReasonCode.TOOL_MISMATCH,
                    f"subgoal does not declare a tool; candidate injects "
                    f"{cand.tool!r}",
                )
            ]
        tool_spec = None
        if cand.tool is not None:
            tool_spec = catalog.tools.get(cand.tool)
            if tool_spec is None:
                return [
                    CheckResult(
                        FeasibilityStatus.FAIL,
                        ReasonCode.UNKNOWN_TOOL,
                        f"tool {cand.tool!r} not in catalog",
                    )
                ]
        # Subgoal-declared feasible EEs.
        if cand.ee not in sg.feasible_ee:
            results.append(
                CheckResult(
                    FeasibilityStatus.FAIL,
                    ReasonCode.EE_NOT_FEASIBLE_FOR_SUBGOAL,
                    f"EE {cand.ee!r} not in feasible_ee {sg.feasible_ee!r}",
                )
            )
            return results

        # Group-feasible EE intersection.
        if sg.group_id is not None:
            allowed = self._group_feasible_ee.get(sg.group_id)
            if allowed is not None and cand.ee not in allowed:
                results.append(
                    CheckResult(
                        FeasibilityStatus.FAIL,
                        ReasonCode.GROUP_EE_CONFLICT,
                        f"EE {cand.ee!r} not in group {sg.group_id!r} "
                        f"intersection {sorted(allowed)}",
                    )
                )
                return results

        ee_caps = set(ee_spec.capabilities)
        available_caps = set(ee_caps)
        if tool_spec is not None:
            available_caps.update(tool_spec.capabilities)

        # Planner-A task requirements are checked against the capabilities
        # jointly supplied by the EE and upstream-fixed tool.
        missing_subgoal = set(sg.required_capabilities) - available_caps
        if missing_subgoal:
            results.append(
                CheckResult(
                    FeasibilityStatus.FAIL,
                    ReasonCode.CAPABILITY_MISMATCH,
                    f"EE/tool pair ({cand.ee!r}, {cand.tool!r}) lacks "
                    f"subgoal-required capabilities {sorted(missing_subgoal)}",
                )
            )

        # Tool checks: capability, compatibility, payload, and home slot.
        if cand.tool is not None and tool_spec is not None:
            missing = set(tool_spec.required_capabilities) - ee_caps
            if missing:
                results.append(
                    CheckResult(
                        FeasibilityStatus.FAIL,
                        ReasonCode.CAPABILITY_MISMATCH,
                        f"EE {cand.ee!r} lacks capabilities {sorted(missing)} "
                        f"required by tool {cand.tool!r}",
                    )
                )
            if cand.ee not in tool_spec.compatible_ee:
                results.append(
                    CheckResult(
                        FeasibilityStatus.FAIL,
                        ReasonCode.EE_TOOL_INCOMPATIBLE,
                        f"tool {cand.tool!r} lists compatible_ee="
                        f"{tool_spec.compatible_ee!r}, not {cand.ee!r}",
                    )
                )
            if cand.tool not in ee_spec.compatible_tools:
                results.append(
                    CheckResult(
                        FeasibilityStatus.FAIL,
                        ReasonCode.EE_TOOL_INCOMPATIBLE,
                        f"EE {cand.ee!r} lists compatible_tools="
                        f"{ee_spec.compatible_tools!r}, not {cand.tool!r}",
                    )
                )
            # Carrying payload.
            if tool_spec.mass is None or ee_spec.payload is None:
                results.append(
                    CheckResult(
                        FeasibilityStatus.UNKNOWN,
                        ReasonCode.PAYLOAD_EXCEEDED,
                        f"payload check needs tool mass and EE payload "
                        f"(tool.mass={tool_spec.mass}, ee.payload="
                        f"{ee_spec.payload})",
                    )
                )
            elif tool_spec.mass > ee_spec.payload:
                results.append(
                    CheckResult(
                        FeasibilityStatus.FAIL,
                        ReasonCode.PAYLOAD_EXCEEDED,
                        f"tool mass {tool_spec.mass} exceeds EE payload "
                        f"{ee_spec.payload}",
                    )
                )
            if tool_spec.home_slot is None:
                results.append(
                    CheckResult(
                        FeasibilityStatus.FAIL,
                        ReasonCode.HOME_SLOT_MISSING,
                        f"tool {cand.tool!r} has no home_slot; cannot be "
                        "returned",
                    )
                )

        # Candidate-declared required capabilities.
        missing_cand = set(cand.required_capabilities) - ee_caps
        if missing_cand:
            results.append(
                CheckResult(
                    FeasibilityStatus.FAIL,
                    ReasonCode.CAPABILITY_MISMATCH,
                    f"EE {cand.ee!r} lacks candidate-required capabilities "
                    f"{sorted(missing_cand)}",
                )
            )

        # Task wrench delivery (separate from carrying payload).
        results.append(self._check_wrench(cand, sg))

        # EE home slot must exist (needed for any future exchange).
        if ee_spec.home_slot is None:
            results.append(
                CheckResult(
                    FeasibilityStatus.FAIL,
                    ReasonCode.HOME_SLOT_MISSING,
                    f"EE {cand.ee!r} has no home_slot in the rack",
                )
            )
        return results

    def _check_wrench(self, cand: Candidate, sg: Subgoal) -> CheckResult:
        """Compare a required task wrench against the candidate's capacity.

        The requirement comes from subgoal conditions typed force/torque (with
        a numeric ``limit``) or ``metadata['required_wrench']``. The capacity
        comes from ``candidate.metadata['deliverable_wrench']``. No
        requirement -> vacuous PASS; requirement without capacity -> UNKNOWN.
        """
        required = subgoal_required_wrench(sg)
        meta_req = cand.metadata.get("required_wrench")
        if isinstance(meta_req, (int, float)):
            required = max(required or 0.0, float(meta_req))
        if required is None:
            return CheckResult.ok()
        capacity = cand.metadata.get("deliverable_wrench")
        if not isinstance(capacity, (int, float)) and cand.tool is not None:
            tool = self._catalog.tools.get(cand.tool)
            capacity = tool.deliverable_wrench if tool is not None else None
        if not isinstance(capacity, (int, float)):
            return CheckResult(
                FeasibilityStatus.UNKNOWN,
                ReasonCode.WRENCH_INSUFFICIENT,
                f"task requires wrench {required} but candidate declares no "
                "deliverable_wrench",
            )
        if float(capacity) < required:
            return CheckResult(
                FeasibilityStatus.FAIL,
                ReasonCode.WRENCH_INSUFFICIENT,
                f"deliverable wrench {capacity} < required {required}",
            )
        return CheckResult.ok()


# --------------------------------------------------------------------------- #
# Lazy geometric checking                                                     #
# --------------------------------------------------------------------------- #


class GeometricFeasibilityChecker(Protocol):
    """External geometric/motion feasibility oracle (IK, collision, docking).

    Implementations must be deterministic for a given (scene, arguments) pair.
    """

    def check_candidate(
        self, scene_signature: str, candidate: Candidate
    ) -> CheckResult: ...

    def check_transition(
        self,
        scene_signature: str,
        current_ee: str,
        held_tool: str | None,
        candidate: Candidate,
    ) -> CheckResult: ...


class AlwaysPassGeometryChecker:
    """Deterministic mock: everything is geometrically feasible."""

    def __init__(self) -> None:
        self.candidate_calls: list[tuple[str, str]] = []
        self.transition_calls: list[tuple[str, str, str, str]] = []

    def check_candidate(
        self, scene_signature: str, candidate: Candidate
    ) -> CheckResult:
        self.candidate_calls.append((scene_signature, candidate.candidate_id))
        return CheckResult.ok()

    def check_transition(
        self,
        scene_signature: str,
        current_ee: str,
        held_tool: str | None,
        candidate: Candidate,
    ) -> CheckResult:
        self.transition_calls.append(
            (
                scene_signature,
                current_ee,
                held_tool or "",
                candidate.candidate_id,
            )
        )
        return CheckResult.ok()


class TableGeometryChecker:
    """Rule-table mock: deterministic failures for specific keys.

    ``candidate_rules`` maps (scene_signature | "*", candidate_id) to a
    CheckResult; ``transition_rules`` maps
    (scene_signature | "*", current_ee, held_tool | "", candidate_id) to a
    CheckResult. Unlisted keys PASS.
    """

    def __init__(
        self,
        candidate_rules: dict[tuple[str, str], CheckResult] | None = None,
        transition_rules: dict[tuple[str, str, str, str], CheckResult]
        | None = None,
    ) -> None:
        self.candidate_rules = candidate_rules or {}
        self.transition_rules = transition_rules or {}
        self.candidate_calls: list[tuple[str, str]] = []
        self.transition_calls: list[tuple[str, str, str, str]] = []

    def check_candidate(
        self, scene_signature: str, candidate: Candidate
    ) -> CheckResult:
        self.candidate_calls.append((scene_signature, candidate.candidate_id))
        for key in (
            (scene_signature, candidate.candidate_id),
            ("*", candidate.candidate_id),
        ):
            if key in self.candidate_rules:
                return self.candidate_rules[key]
        return CheckResult.ok()

    def check_transition(
        self,
        scene_signature: str,
        current_ee: str,
        held_tool: str | None,
        candidate: Candidate,
    ) -> CheckResult:
        sig = (
            scene_signature,
            current_ee,
            held_tool or "",
            candidate.candidate_id,
        )
        self.transition_calls.append(sig)
        for key in (sig, ("*",) + sig[1:]):
            if key in self.transition_rules:
                return self.transition_rules[key]
        return CheckResult.ok()


@dataclass
class GeometryCache:
    """Caches lazy check results keyed by scene signature and context.

    Accepts either the legacy :class:`GeometricFeasibilityChecker` (two
    signature-keyed methods) or a
    :class:`~task_planner.motion_interface.MotionFeasibilityOracle` (structured
    queries plus EE-exchange and terminal hooks). The oracle is detected by
    duck typing so neither module has to import the other at load time.
    """

    checker: GeometricFeasibilityChecker
    candidate_cache: dict[tuple[str, str], CheckResult] = field(default_factory=dict)
    transition_cache: dict[tuple[str, str, str, str], CheckResult] = field(
        default_factory=dict
    )
    exchange_cache: dict[tuple[str, str, str, str], CheckResult] = field(
        default_factory=dict
    )
    terminal_cache: dict[tuple[str, str, str, str], CheckResult] = field(
        default_factory=dict
    )
    checks_executed: int = 0
    cache_hits: int = 0

    @property
    def is_motion_oracle(self) -> bool:
        """Whether the checker implements the richer motion-planner contract."""
        return hasattr(self.checker, "check_ee_exchange")

    def check_candidate(
        self,
        scene_signature: str,
        candidate: Candidate,
        scene_ref: Any | None = None,
    ) -> CheckResult:
        key = (scene_signature, candidate.candidate_id)
        if key in self.candidate_cache:
            self.cache_hits += 1
            return self.candidate_cache[key]
        self.checks_executed += 1
        if self.is_motion_oracle:
            from task_planner.motion_interface import build_candidate_query

            result = self.checker.check_candidate(
                build_candidate_query(scene_ref or scene_signature, candidate)
            )
        else:
            result = self.checker.check_candidate(scene_signature, candidate)
        self.candidate_cache[key] = result
        return result

    def check_transition(
        self,
        scene_signature: str,
        current_ee: str,
        held_tool: str | None,
        candidate: Candidate,
        scene_ref: Any | None = None,
        primitives: tuple[str, ...] = (),
        held_object_id: str | None = None,
    ) -> CheckResult:
        key = (
            scene_signature,
            current_ee,
            held_tool or "",
            candidate.candidate_id,
        )
        if key in self.transition_cache:
            self.cache_hits += 1
            return self.transition_cache[key]
        self.checks_executed += 1
        if self.is_motion_oracle:
            from task_planner.motion_interface import build_transition_query

            result = self.checker.check_transition(
                build_transition_query(
                    scene_ref or scene_signature,
                    candidate,
                    current_ee=current_ee,
                    held_tool=held_tool,
                    held_object_id=held_object_id,
                    primitives=primitives,
                )
            )
        else:
            result = self.checker.check_transition(
                scene_signature, current_ee, held_tool, candidate
            )
        self.transition_cache[key] = result
        return result

    def check_ee_exchange(
        self,
        scene_signature: str,
        from_ee: str,
        to_ee: str,
        *,
        held_tool: str | None = None,
        rack_occupancy: tuple[tuple[str, str], ...] | None = None,
        scene_ref: Any | None = None,
    ) -> CheckResult:
        """Vacuous PASS unless the checker models EE exchange."""
        if not self.is_motion_oracle:
            return CheckResult.ok()
        key = (scene_signature, from_ee, to_ee, held_tool or "")
        if key in self.exchange_cache:
            self.cache_hits += 1
            return self.exchange_cache[key]
        self.checks_executed += 1
        from task_planner.motion_interface import EEExchangeQuery, as_scene_ref

        result = self.checker.check_ee_exchange(
            EEExchangeQuery(
                scene=as_scene_ref(scene_ref or scene_signature),
                from_ee=from_ee,
                to_ee=to_ee,
                held_tool=held_tool,
                rack_occupancy=rack_occupancy,
            )
        )
        self.exchange_cache[key] = result
        return result

    def check_terminal(
        self,
        scene_signature: str,
        current_ee: str,
        *,
        held_tool: str | None = None,
        held_object_id: str | None = None,
        restore_ee: str | None = None,
        return_tool: str | None = None,
        rack_occupancy: tuple[tuple[str, str], ...] | None = None,
        scene_ref: Any | None = None,
    ) -> CheckResult:
        """Vacuous PASS unless the checker models terminal cleanup."""
        if not self.is_motion_oracle:
            return CheckResult.ok()
        key = (scene_signature, current_ee, restore_ee or "", return_tool or "")
        if key in self.terminal_cache:
            self.cache_hits += 1
            return self.terminal_cache[key]
        self.checks_executed += 1
        from task_planner.motion_interface import (
            ResourceState,
            TerminalQuery,
            as_scene_ref,
        )

        result = self.checker.check_terminal(
            TerminalQuery(
                scene=as_scene_ref(scene_ref or scene_signature),
                from_state=ResourceState(
                    current_ee=current_ee,
                    held_tool=held_tool,
                    held_object_id=held_object_id,
                ),
                restore_ee=restore_ee,
                return_tool=return_tool,
                rack_occupancy=rack_occupancy,
            )
        )
        self.terminal_cache[key] = result
        return result
