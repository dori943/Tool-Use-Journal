"""Motion planner for the EE-exchange TAMP pipeline.

Implements the feasibility side of the Task Planner contract documented in
``Task_Planner/MOTION_PLANNER_INTERFACE.md``. Task Planner decides *what* to do with
*which* resources; this package answers whether the arm can physically do it.

Current increment: deterministic multi-branch IK, scene-relative keyframe
resolution, first-feasible branch backtracking, rack EE exchange templates, and
constraint-respecting trajectory timing. Full-workcell MuJoCo collision checks
cover endpoints, sampled edges, and final timed waypoints; MJCF-derived scene
variants represent bare and hard-attached EE states. Unimplemented checks answer
UNKNOWN, never PASS.
"""

from motion_planner.compiler import FirstFeasibleStrategyCompiler
from motion_planner.ee_exchange import (
    EEExchangeKeyframeProvider,
    EEExchangeTemplateGenerator,
    RoutedKeyframeStrategyProvider,
)
from motion_planner.geometry import RelativePoseResolver
from motion_planner.kinematics import (
    IKResult,
    IKSolutionSet,
    UR5eKinematics,
    default_model_path,
)
from motion_planner.oracle import MuJoCoMotionOracle
from motion_planner.mujoco_collision import (
    CollisionCheckResult,
    MuJoCoCollisionModelRegistry,
    MuJoCoCollisionValidator,
    MuJoCoInterpolatingEdgePlanner,
    PathCollisionCheckResult,
)
from motion_planner.plan_builder import MotionPlanBuilder
from motion_planner.orchestration import (
    MotionPlanStore,
    SelectedPlanMotionOrchestrator,
    SelectedPlanPlanningResult,
)
from motion_planner.path_planning import (
    CartesianEdgePlanner,
    PlannerDispatchEdgePlanner,
    RRTConnectEdgePlanner,
)
from motion_planner.safety import KinematicSafetyValidator
from motion_planner.selected_plan_adapter import (
    SelectedPlanAdapterError,
    SelectedPlanMotionRequestAdapter,
    selected_plan_to_motion_requests,
)
from motion_planner.pipeline import (
    CollisionContextFactory,
    CollisionPlanningSetup,
    KeyframeStrategyProvider,
    MotionPlanningPipeline,
    MotionPlanningPipelineError,
    MotionPlanningResult,
)
from motion_planner.strategy import (
    FirstFeasibleBranchSelector,
    InterpolatingEdgePlanner,
)
from motion_planner.trajectory_processing import (
    QuinticTimeParameterizer,
    deterministic_shortcut,
)
from motion_planner.tool_use_journal import (
    CompiledToolUseJournalCollisionModel,
    ToolUseJournalCollisionModelCompiler,
    ToolUseJournalCompatibilityError,
    ToolUseJournalEnvironmentAdapter,
    make_tool_use_journal_env,
)
from motion_planner.tool_use_journal_planning import (
    ToolUseJournalCollisionBindingError,
    ToolUseJournalCollisionContextFactory,
    ToolUseJournalMotionRequestPlanner,
    WorkcellMotionRequestRouter,
    attached_object_transform_from_state,
)
from motion_planner.tool_use_journal_runtime import (
    AttachmentBreakObservation,
    AttachmentContactMetrics,
    AttachmentMode,
    AttachedObjectState,
    BreakableWeldConfig,
    EERuntimeTransition,
    ToolUseJournalControllerTrajectoryPlayer,
    ToolUseJournalEERuntime,
    ToolUseJournalKinematicTrajectoryPlayer,
    ToolUseJournalRuntimeError,
    ToolUseJournalAttachmentBroken,
    tool_use_journal_joint_position_controller_config,
)
from motion_planner.workcell_models import (
    CompiledWorkcellCollisionModel,
    EEWorkcellCollisionModelCompiler,
    WorkcellModelCompilationError,
)
from motion_planner.vlm_provider import (
    MissingOpenAIAPIKeyError,
    OpenAIKeyframeProvider,
    OpenAIKeyframeProviderConfig,
    OpenAIKeyframeProviderError,
)

__all__ = [
    "AttachmentBreakObservation",
    "AttachmentContactMetrics",
    "AttachmentMode",
    "EEExchangeTemplateGenerator",
    "EEExchangeKeyframeProvider",
    "AttachedObjectState",
    "BreakableWeldConfig",
    "EERuntimeTransition",
    "FirstFeasibleBranchSelector",
    "FirstFeasibleStrategyCompiler",
    "IKResult",
    "IKSolutionSet",
    "InterpolatingEdgePlanner",
    "CollisionCheckResult",
    "CollisionContextFactory",
    "CollisionPlanningSetup",
    "CompiledWorkcellCollisionModel",
    "CompiledToolUseJournalCollisionModel",
    "EEWorkcellCollisionModelCompiler",
    "MuJoCoCollisionValidator",
    "MuJoCoCollisionModelRegistry",
    "MuJoCoInterpolatingEdgePlanner",
    "MuJoCoMotionOracle",
    "KeyframeStrategyProvider",
    "MissingOpenAIAPIKeyError",
    "MotionPlanBuilder",
    "MotionPlanStore",
    "MotionPlanningPipeline",
    "MotionPlanningPipelineError",
    "MotionPlanningResult",
    "OpenAIKeyframeProvider",
    "OpenAIKeyframeProviderConfig",
    "OpenAIKeyframeProviderError",
    "QuinticTimeParameterizer",
    "PathCollisionCheckResult",
    "RelativePoseResolver",
    "RoutedKeyframeStrategyProvider",
    "CartesianEdgePlanner",
    "PlannerDispatchEdgePlanner",
    "RRTConnectEdgePlanner",
    "KinematicSafetyValidator",
    "SelectedPlanAdapterError",
    "SelectedPlanMotionRequestAdapter",
    "SelectedPlanMotionOrchestrator",
    "SelectedPlanPlanningResult",
    "selected_plan_to_motion_requests",
    "ToolUseJournalCollisionModelCompiler",
    "ToolUseJournalCollisionBindingError",
    "ToolUseJournalCollisionContextFactory",
    "ToolUseJournalCompatibilityError",
    "ToolUseJournalControllerTrajectoryPlayer",
    "ToolUseJournalEnvironmentAdapter",
    "ToolUseJournalEERuntime",
    "ToolUseJournalKinematicTrajectoryPlayer",
    "ToolUseJournalMotionRequestPlanner",
    "ToolUseJournalRuntimeError",
    "ToolUseJournalAttachmentBroken",
    "UR5eKinematics",
    "WorkcellModelCompilationError",
    "WorkcellMotionRequestRouter",
    "attached_object_transform_from_state",
    "default_model_path",
    "deterministic_shortcut",
    "make_tool_use_journal_env",
    "tool_use_journal_joint_position_controller_config",
]
