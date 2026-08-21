"""Planning B: search over subgoal order, EE, and resource transitions.

Given a normalized Planner-A contract or GK+M1 artifacts, a single Dijkstra
search over a joint symbolic/resource state space selects subgoal order and
end-effector while preserving upstream-fixed tools and planning transitions.
"""

from planning_b.cost import CostVector
from planning_b.diagnostics import PlanStatus, ReasonCode, Rejection
from planning_b.models import (
    CandidateProposal,
    Condition,
    EndEffectorSpec,
    ExecutionState,
    FailureFeedback,
    FailureType,
    GraspSpec,
    InitialState,
    ObjectSpec,
    PlannerAConstraints,
    PlannerAOutput,
    PlanningBRequest,
    PlanningPolicy,
    ResourceCatalog,
    Subgoal,
    ToolSpec,
)
from planning_b.constraints import PlannerAConstraintEngine
from planning_b.planner import plan
from planning_b.planner_a_adapter import (
    adapt_current_planner_a_output,
    build_request_from_current_planner_a,
)
from planning_b.gk_adapter import adapt_gk_m1_output, build_request_from_gk
from planning_b.motion_interface import (
    MOTION_COST_UNIT,
    CandidateQuery,
    EEExchangeQuery,
    MotionCostOracle,
    MotionCostResult,
    MotionFeasibilityOracle,
    ResourceState,
    SceneRef,
    TerminalQuery,
    TransitionQuery,
    UnknownMotionOracle,
    WorldSnapshot,
)
from planning_b.replanning import NoGoodSet, replan
from planning_b.serialization import PLANNER_VERSION, PlanningResult
from planning_b.state import SearchState
from planning_b.suitability import (
    PhysicsSuitabilityScorer,
    SuitabilityAssessment,
    SuitabilityComponent,
    SuitabilityStatus,
)

__all__ = [
    "MOTION_COST_UNIT",
    "PLANNER_VERSION",
    "CandidateProposal",
    "CandidateQuery",
    "EEExchangeQuery",
    "MotionCostOracle",
    "MotionCostResult",
    "MotionFeasibilityOracle",
    "ResourceState",
    "SceneRef",
    "TerminalQuery",
    "TransitionQuery",
    "UnknownMotionOracle",
    "WorldSnapshot",
    "Condition",
    "CostVector",
    "EndEffectorSpec",
    "ExecutionState",
    "FailureFeedback",
    "FailureType",
    "GraspSpec",
    "InitialState",
    "NoGoodSet",
    "ObjectSpec",
    "PlannerAConstraintEngine",
    "PlannerAConstraints",
    "PlanStatus",
    "PlannerAOutput",
    "PlanningBRequest",
    "PlanningPolicy",
    "PlanningResult",
    "ReasonCode",
    "Rejection",
    "ResourceCatalog",
    "SearchState",
    "PhysicsSuitabilityScorer",
    "SuitabilityAssessment",
    "SuitabilityComponent",
    "SuitabilityStatus",
    "Subgoal",
    "ToolSpec",
    "plan",
    "adapt_current_planner_a_output",
    "build_request_from_current_planner_a",
    "adapt_gk_m1_output",
    "build_request_from_gk",
    "replan",
]
