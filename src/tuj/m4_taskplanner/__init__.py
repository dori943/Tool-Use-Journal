"""Task Planner: search over subgoal order, EE, and resource transitions.

Given GK + M2 artifacts, a single Dijkstra
search over a joint symbolic/resource state space selects subgoal order and
end-effector while preserving upstream-fixed tools and planning transitions.
"""

from tuj.m4_taskplanner.cost import CostVector
from tuj.m4_taskplanner.diagnostics import PlanStatus, ReasonCode, Rejection
from tuj.m4_taskplanner.models import (
    CandidateProposal,
    Condition,
    EndEffectorSpec,
    ExecutionState,
    FailureFeedback,
    FailureType,
    GraspSpec,
    InitialState,
    ObjectSpec,
    TaskConstraints,
    TaskGraph,
    TaskPlannerRequest,
    PlanningPolicy,
    ResourceCatalog,
    Subgoal,
    ToolSpec,
)
from tuj.m4_taskplanner.constraints import TaskConstraintEngine
from tuj.m4_taskplanner.planner import plan
from tuj.m4_taskplanner.gk_adapter import adapt_gk_m2_output, build_request_from_gk
from tuj.m4_taskplanner.motion_interface import (
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
from tuj.m4_taskplanner.replanning import NoGoodSet, replan
from tuj.m4_taskplanner.serialization import PLANNER_VERSION, PlanningResult
from tuj.m4_taskplanner.state import SearchState
from tuj.m4_taskplanner.suitability import (
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
    "TaskConstraintEngine",
    "TaskConstraints",
    "PlanStatus",
    "TaskGraph",
    "TaskPlannerRequest",
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
    "adapt_gk_m2_output",
    "build_request_from_gk",
    "replan",
]
