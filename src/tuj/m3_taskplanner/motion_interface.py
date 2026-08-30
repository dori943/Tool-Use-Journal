"""Contract between Task Planner and the motion planner.

Task Planner never generates trajectories. It asks a motion-side oracle two kinds
of question:

* *Is this assignment geometrically realizable?* -> :class:`MotionFeasibilityOracle`
* *What does moving there actually cost?*        -> :class:`MotionCostOracle`

Scene identity
--------------
Task Planner does not own a geometric world model, so it cannot ship object poses
and obstacle meshes on every call. Instead every query carries a
:class:`SceneRef`: an opaque signature plus the *causal history* that produced
it (completed subgoals, applied symbolic effects). The motion side keeps its own
world model, seeded once through :meth:`MotionFeasibilityOracle.initialize`, and
reconstructs a predicted scene by replaying that history. Because Task Planner
derives the signature deterministically from the same history, the pair
``(signature, history)`` is a stable cache key on both sides.

Determinism
-----------
Both oracles MUST be deterministic: identical query -> identical result within a
planning run. Task Planner caches by query identity and treats a cached answer as
authoritative, so a sampling-based implementation must fix its seed per query
(deriving it from the query fields is the simplest correct choice). A
non-deterministic oracle silently breaks Dijkstra's optimality argument.

Cost units
----------
:class:`~tuj.m3_taskplanner.cost.CostVector` is integer-valued on purpose: lexicographic
comparison of floats makes tie-breaking depend on rounding, which would make
plans non-reproducible across machines. Motion cost is therefore reported as a
non-negative integer in a fixed quantum (see :data:`MOTION_COST_UNIT`), not as
seconds or metres. Quantize once, in the oracle, and never inside the search.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from tuj.m3_taskplanner.feasibility import CheckResult, FeasibilityStatus
from tuj.m3_taskplanner.models import GraspSpec

# One unit of ``CostVector.motion_cost``. Oracles convert their native metric
# (seconds, metres of end-effector travel, ...) into this quantum exactly once.
MOTION_COST_UNIT = "millisecond of nominal execution time"


class MotionQueryKind(str, enum.Enum):
    CANDIDATE = "CANDIDATE"
    TRANSITION = "TRANSITION"
    EE_EXCHANGE = "EE_EXCHANGE"
    TERMINAL = "TERMINAL"


# --------------------------------------------------------------------------- #
# Scene and resource identity                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SceneRef:
    """Identity of a *predicted* scene, plus the history that produced it.

    ``signature`` is opaque to the motion side and must only be used as a cache
    key. ``completed_subgoals`` and ``facts`` are the reconstruction recipe: the
    motion planner replays them against the world seeded by ``initialize`` to
    obtain the predicted object placements.
    """

    signature: str
    completed_subgoals: tuple[str, ...] = ()
    # Normalized fluent strings, e.g. ``"holding(obj_milk)"``, sorted.
    facts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResourceState:
    """What the arm is carrying when the queried motion begins."""

    current_ee: str
    held_tool: str | None = None
    held_object_id: str | None = None


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    """One-time seed for the motion planner's world model.

    Task Planner passes through whatever the upstream scene graph supplied; it
    does not interpret these payloads. ``objects`` and ``obstacles`` are the
    scene-graph records, ``rack`` the EE rack layout.
    """

    scene: SceneRef
    objects: Mapping[str, Any] = field(default_factory=dict)
    obstacles: tuple[Any, ...] = ()
    rack: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Queries                                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CandidateQuery:
    """Can this EE/tool assignment execute this subgoal here?

    Covers design-doc feasibility checks 4 (reachability) and 5 (collision) for
    the externally supplied task pose. Task Planner passes ``target_pose``
    through without generating, interpreting, or selecting it.
    """

    scene: SceneRef
    subgoal_id: str
    candidate_id: str
    ee: str
    tool: str | None
    grasp_id: str | None = None
    grasp: GraspSpec | None = None
    target_ids: tuple[str, ...] = ()
    action_type: str | None = None
    target_pose: Any = None
    # Unit approach vector in the scene frame, when the candidate carries one.
    approach_dir: tuple[float, float, float] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.grasp is None:
            return
        if self.grasp_id is None:
            object.__setattr__(self, "grasp_id", self.grasp.grasp_id)
        elif self.grasp_id != self.grasp.grasp_id:
            raise ValueError(
                "grasp_id must match grasp.grasp_id when both are supplied"
            )


@dataclass(frozen=True, slots=True)
class TransitionQuery:
    """Can the arm get from ``from_state`` into the candidate's configuration?

    ``primitives`` is the ordered primitive-action sequence Task Planner intends
    to emit (``MOVE_TO_TOOL_RACK``, ``PICK_TOOL``, ...). The oracle may check the
    whole chain or only the parts it models; a partial check must still be
    deterministic and must report UNKNOWN rather than PASS for what it skipped.
    """

    scene: SceneRef
    from_state: ResourceState
    candidate: CandidateQuery
    primitives: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EEExchangeQuery:
    """Can the EE be swapped here? (design-doc check 6 and limitation 8)

    Task Planner currently validates rack slot *bookkeeping* symbolically but has
    no way to know whether the dock pose is reachable or the approach is
    collision-free. That judgement belongs here.
    """

    scene: SceneRef
    from_ee: str
    to_ee: str
    held_tool: str | None = None
    # slot -> occupant EE id / "empty" / "reserved" / "unknown".
    rack_occupancy: tuple[tuple[str, str], ...] | None = None


@dataclass(frozen=True, slots=True)
class TerminalQuery:
    """Is the terminal cleanup (tool return, EE restore) realizable?"""

    scene: SceneRef
    from_state: ResourceState
    restore_ee: str | None = None
    return_tool: str | None = None
    rack_occupancy: tuple[tuple[str, str], ...] | None = None


@dataclass(frozen=True, slots=True)
class MotionCostResult:
    """Quantized motion cost for one edge.

    ``cost`` is a non-negative integer in :data:`MOTION_COST_UNIT`. ``status``
    lets an oracle answer "I cannot price this" without inventing a number;
    Task Planner then falls back to its configured nominal ``MotionCosts``.
    """

    cost: int
    status: FeasibilityStatus = FeasibilityStatus.PASS
    detail: str = ""

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError("MotionCostResult.cost must be non-negative")


# --------------------------------------------------------------------------- #
# Oracles                                                                     #
# --------------------------------------------------------------------------- #


class MotionFeasibilityOracle(Protocol):
    """Geometric judgement. Returns PASS / FAIL / UNKNOWN, never a trajectory.

    Reason codes an implementation is expected to use:
    ``IK_UNREACHABLE``, ``GEOMETRY_COLLISION``, ``RACK_SLOT_UNAVAILABLE``,
    ``GEOMETRY_FAILED`` (anything else). UNKNOWN is routed through
    ``PlanningPolicy.unknown_feasibility_policy``, so returning UNKNOWN for an
    unmodelled check is safe and honest; returning PASS is not.
    """

    def initialize(self, world: WorldSnapshot) -> None:
        """Seed the world model. Called once per planning run, before search."""
        ...

    def check_candidate(self, query: CandidateQuery) -> CheckResult: ...

    def check_transition(self, query: TransitionQuery) -> CheckResult: ...

    def check_ee_exchange(self, query: EEExchangeQuery) -> CheckResult: ...

    def check_terminal(self, query: TerminalQuery) -> CheckResult: ...


class MotionCostOracle(Protocol):
    """Optional. Replaces the nominal integer ``MotionCosts`` with real values."""

    def transition_cost(self, query: TransitionQuery) -> MotionCostResult: ...

    def ee_exchange_cost(self, query: EEExchangeQuery) -> MotionCostResult: ...

    def terminal_cost(self, query: TerminalQuery) -> MotionCostResult: ...


# --------------------------------------------------------------------------- #
# Query builders (used by GeometryCache to bridge the legacy call sites)       #
# --------------------------------------------------------------------------- #


def as_scene_ref(scene: "SceneRef | str") -> SceneRef:
    """Accept either a full SceneRef or a bare signature.

    A bare signature loses the reconstruction history, so an oracle that needs
    to replay effects should require the full ref and report UNKNOWN otherwise.
    """
    if isinstance(scene, SceneRef):
        return scene
    return SceneRef(signature=scene)


def build_candidate_query(
    scene: "SceneRef | str", candidate: Any
) -> CandidateQuery:
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    grasp = getattr(candidate, "grasp", None)
    if grasp is None and isinstance(metadata.get("grasp"), Mapping):
        grasp = GraspSpec.model_validate(metadata["grasp"])
    grasp_id = getattr(candidate, "grasp_id", None) or metadata.get("grasp_id")
    if grasp_id is None and grasp is not None:
        grasp_id = grasp.grasp_id
    approach = metadata.get("approach_dir")
    if isinstance(approach, (list, tuple)) and len(approach) == 3:
        approach_dir: tuple[float, float, float] | None = (
            float(approach[0]),
            float(approach[1]),
            float(approach[2]),
        )
    else:
        approach_dir = None
    targets = getattr(candidate, "target_ids", None) or metadata.get("target_ids")
    action_type = getattr(candidate, "action_type", None) or metadata.get(
        "action_type"
    )
    return CandidateQuery(
        scene=as_scene_ref(scene),
        subgoal_id=candidate.subgoal_id,
        candidate_id=candidate.candidate_id,
        ee=candidate.ee,
        tool=candidate.tool,
        grasp_id=grasp_id,
        grasp=grasp,
        target_ids=tuple(targets) if isinstance(targets, (list, tuple)) else (),
        action_type=action_type,
        target_pose=metadata.get("target_pose"),
        approach_dir=approach_dir,
        metadata=metadata,
    )


def build_transition_query(
    scene: "SceneRef | str",
    candidate: Any,
    *,
    current_ee: str,
    held_tool: str | None,
    held_object_id: str | None = None,
    primitives: tuple[str, ...] = (),
) -> TransitionQuery:
    return TransitionQuery(
        scene=as_scene_ref(scene),
        from_state=ResourceState(
            current_ee=current_ee,
            held_tool=held_tool,
            held_object_id=held_object_id,
        ),
        candidate=build_candidate_query(scene, candidate),
        primitives=primitives,
    )


# --------------------------------------------------------------------------- #
# Reference implementation                                                    #
# --------------------------------------------------------------------------- #


class UnknownMotionOracle:
    """Honest no-op oracle: models nothing, so answers UNKNOWN everywhere.

    Useful as a scaffold and as an ablation baseline. With the default
    ``unknown_feasibility_policy="reject"`` this makes every candidate fail,
    which is the point: it proves the policy wiring works. Use
    ``unknown_feasibility_policy="allow"`` to reproduce today's
    ``AlwaysPassGeometryChecker`` behaviour.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[MotionQueryKind, str]] = []

    def initialize(self, world: WorldSnapshot) -> None:
        self.calls.append((MotionQueryKind.CANDIDATE, f"init:{world.scene.signature}"))

    def _unknown(self, kind: MotionQueryKind, key: str) -> CheckResult:
        self.calls.append((kind, key))
        return CheckResult(
            FeasibilityStatus.UNKNOWN,
            None,
            f"{kind.value} is not modelled by UnknownMotionOracle",
        )

    def check_candidate(self, query: CandidateQuery) -> CheckResult:
        return self._unknown(MotionQueryKind.CANDIDATE, query.candidate_id)

    def check_transition(self, query: TransitionQuery) -> CheckResult:
        return self._unknown(
            MotionQueryKind.TRANSITION, query.candidate.candidate_id
        )

    def check_ee_exchange(self, query: EEExchangeQuery) -> CheckResult:
        return self._unknown(
            MotionQueryKind.EE_EXCHANGE, f"{query.from_ee}->{query.to_ee}"
        )

    def check_terminal(self, query: TerminalQuery) -> CheckResult:
        return self._unknown(
            MotionQueryKind.TERMINAL, query.from_state.current_ee
        )
