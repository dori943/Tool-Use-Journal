"""Task Planner: search over subgoal order, EE, and resource transitions.

Given a normalized Planner-A contract or GK+M1 artifacts, a single Dijkstra
search over a joint symbolic/resource state space selects subgoal order and
end-effector while preserving upstream-fixed tools and planning transitions.
"""

from task_planner.cost import CostVector
from task_planner.diagnostics import PlanStatus, ReasonCode, Rejection
from task_planner.models import (
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
    TaskPlannerRequest,
    PlanningPolicy,
    ResourceCatalog,
    Subgoal,
    ToolSpec,
)
from task_planner.constraints import PlannerAConstraintEngine
from task_planner.planner import plan
from task_planner.planner_a_adapter import (
    adapt_current_planner_a_output,
    build_request_from_current_planner_a,
)
from task_planner.gk_adapter import adapt_gk_m1_output, build_request_from_gk
from task_planner.motion_interface import (
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
from task_planner.replanning import NoGoodSet, replan
from task_planner.serialization import PLANNER_VERSION, PlanningResult
from task_planner.state import SearchState
from task_planner.suitability import (
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
    "adapt_current_planner_a_output",
    "build_request_from_current_planner_a",
    "adapt_gk_m1_output",
    "build_request_from_gk",
    "replan",
]
