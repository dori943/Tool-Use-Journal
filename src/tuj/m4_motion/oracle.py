"""MuJoCo-backed implementation of the Task Planner motion contract.

Scope of this increment (see MOTION_PLANNER_INTERFACE.md step ladder):

* step 2 -- ``check_candidate``: IK reachability of the task pose
* step 3 -- ``check_ee_exchange``: IK reachability of the rack dock pose

Everything else deliberately answers ``UNKNOWN`` rather than ``PASS``. Under the
default ``unknown_feasibility_policy="reject"`` that makes unmodelled checks
visible instead of silently optimistic; set the policy to ``allow`` while the
remaining steps are unimplemented.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from tuj.m3_taskplanner.diagnostics import ReasonCode
from tuj.m3_taskplanner.feasibility import CheckResult, FeasibilityStatus
from tuj.m3_taskplanner.motion_interface import (
    CandidateQuery,
    EEExchangeQuery,
    TerminalQuery,
    TransitionQuery,
    WorldSnapshot,
)

from tuj.m4_motion.kinematics import UR5eKinematics


def _as_position(value: Any) -> tuple[float, float, float] | None:
    """Accept ``[x,y,z]``, ``{"position": [...]}`` or ``{"x":..,"y":..,"z":..}``."""
    if isinstance(value, Mapping):
        for key in (
            "position_m",
            "position",
            "pos",
            "xyz",
            "translation",
            "pose",
            "dock_pose",
        ):
            if key in value:
                return _as_position(value[key])
        if {"x", "y", "z"} <= set(value):
            try:
                return (float(value["x"]), float(value["y"]), float(value["z"]))
            except (TypeError, ValueError):
                return None
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) >= 3:
            try:
                return (float(value[0]), float(value[1]), float(value[2]))
            except (TypeError, ValueError):
                return None
    return None


class MuJoCoMotionOracle:
    """Reachability-only feasibility oracle. Deterministic by construction."""

    def __init__(
        self,
        kinematics: UR5eKinematics | None = None,
        *,
        tolerance_m: float = 5e-3,
    ) -> None:
        self._kin = kinematics or UR5eKinematics()
        self._tolerance = tolerance_m
        self._objects: Mapping[str, Any] = {}
        self._rack: Mapping[str, Any] = {}
        self._initialized = False
        # Query counters, useful for the lazy-vs-eager geometry ablation.
        self.ik_calls = 0

    # -- lifecycle ---------------------------------------------------------- #

    def initialize(self, world: WorldSnapshot) -> None:
        self._objects = dict(world.objects or {})
        self._rack = dict(world.rack or {})
        self._initialized = True

    # -- pose resolution ---------------------------------------------------- #

    def _candidate_position(
        self, query: CandidateQuery
    ) -> tuple[tuple[float, float, float] | None, str]:
        """Where the EEF must be. Returns (position, why-not) for diagnostics."""
        if query.grasp is not None:
            position = _as_position(query.grasp.pose)
            if position is not None:
                return position, ""
        position = _as_position(query.metadata.get("grasp_pose"))
        if position is not None:
            return position, ""
        for target in query.target_ids:
            record = self._objects.get(target)
            if record is not None:
                position = _as_position(record)
                if position is not None:
                    return position, ""
        if not self._initialized:
            return None, "oracle was never initialized with a WorldSnapshot"
        return None, (
            f"no pose for candidate {query.candidate_id!r}: grasp.pose is empty "
            f"and targets {list(query.target_ids)} have no pose in the world model"
        )

    def _dock_position(self, ee: str) -> tuple[float, float, float] | None:
        record = self._rack.get(ee)
        if record is None:
            return None
        if isinstance(record, Mapping):
            for key in ("dock_pose", "dock", "pre_dock", "approach"):
                if key in record:
                    position = _as_position(record[key])
                    if position is not None:
                        return position
        return _as_position(record)

    def _reachable(self, position, subject: str) -> CheckResult:
        self.ik_calls += 1
        result = self._kin.solve_ik(position, tolerance_m=self._tolerance)
        if result.solved:
            return CheckResult.ok()
        return CheckResult(
            FeasibilityStatus.FAIL,
            ReasonCode.IK_UNREACHABLE,
            f"{subject}: {result.detail} (best error {result.position_error_m:.3f} m)",
        )

    # -- contract ----------------------------------------------------------- #

    def check_candidate(self, query: CandidateQuery) -> CheckResult:
        position, why_not = self._candidate_position(query)
        if position is None:
            return CheckResult(
                FeasibilityStatus.UNKNOWN, None, why_not
            )
        return self._reachable(position, f"candidate {query.candidate_id!r}")

    def check_transition(self, query: TransitionQuery) -> CheckResult:
        # Approach-path collision checking is step 5 and is not implemented, so
        # this must not claim PASS.
        return CheckResult(
            FeasibilityStatus.UNKNOWN,
            None,
            "transition path checking is not implemented in this increment",
        )

    def check_ee_exchange(self, query: EEExchangeQuery) -> CheckResult:
        undock = (
            self._dock_position(query.from_ee)
            if query.from_ee is not None
            else None
        )
        dock = self._dock_position(query.to_ee)
        if dock is None or (query.from_ee is not None and undock is None):
            missing = [
                ee
                for ee, position in ((query.from_ee, undock), (query.to_ee, dock))
                if position is None
            ]
            return CheckResult(
                FeasibilityStatus.UNKNOWN,
                None,
                f"no rack dock pose for {missing}",
            )
        # Both halves of the swap must be reachable: park the worn EE, then
        # collect the new one.
        rack_moves = [(query.to_ee, dock)]
        if query.from_ee is not None:
            rack_moves.insert(0, (query.from_ee, undock))
        for ee, position in rack_moves:
            result = self._reachable(position, f"rack slot for EE {ee!r}")
            if result.status is not FeasibilityStatus.PASS:
                return CheckResult(
                    result.status,
                    ReasonCode.RACK_SLOT_UNAVAILABLE,
                    result.message,
                )
        return CheckResult.ok()

    def check_terminal(self, query: TerminalQuery) -> CheckResult:
        if query.restore_ee is None:
            # Nothing geometric to verify beyond the tool return, which is
            # step 5 territory.
            return CheckResult(
                FeasibilityStatus.UNKNOWN,
                None,
                "terminal tool return is not modelled in this increment",
            )
        return self.check_ee_exchange(
            EEExchangeQuery(
                scene=query.scene,
                from_ee=query.from_state.current_ee,
                to_ee=query.restore_ee,
                held_tool=query.from_state.held_tool,
                rack_occupancy=query.rack_occupancy,
            )
        )
