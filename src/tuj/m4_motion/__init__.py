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

from tuj.m4_motion.compiler import FirstFeasibleStrategyCompiler
from tuj.m4_motion.ee_exchange import (
    EEExchangeKeyframeProvider,
    EEExchangeTemplateGenerator,
    RoutedKeyframeStrategyProvider,
)
from tuj.m4_motion.geometry import RelativePoseResolver
from tuj.m4_motion.kinematics import (
    IKResult,
    IKSolutionSet,
    UR5eKinematics,
    default_model_path,
)
from tuj.m4_motion.oracle import MuJoCoMotionOracle
from tuj.m4_motion.mujoco_collision import (
    CollisionCheckResult,
    MuJoCoCollisionModelRegistry,
    MuJoCoCollisionValidator,
    MuJoCoInterpolatingEdgePlanner,
    PathCollisionCheckResult,
)
from tuj.m4_motion.plan_builder import MotionPlanBuilder
from tuj.m4_motion.orchestration import (
    MotionPlanStore,
    SelectedPlanMotionOrchestrator,
    SelectedPlanPlanningResult,
)
from tuj.m4_motion.path_planning import (
    CartesianEdgePlanner,
    PlannerDispatchEdgePlanner,
    RRTConnectEdgePlanner,
)
from tuj.m4_motion.safety import KinematicSafetyValidator
from tuj.m4_motion.selected_plan_adapter import (
    SelectedPlanAdapterError,
    SelectedPlanMotionRequestAdapter,
    selected_plan_to_motion_requests,
)
from tuj.m4_motion.pipeline import (
    CollisionContextFactory,
    CollisionPlanningSetup,
    KeyframeStrategyProvider,
    MotionPlanningPipeline,
    MotionPlanningPipelineError,
    MotionPlanningResult,
)
from tuj.m4_motion.strategy import (
    FirstFeasibleBranchSelector,
    InterpolatingEdgePlanner,
)
from tuj.m4_motion.trajectory_processing import (
    QuinticTimeParameterizer,
    deterministic_shortcut,
)
from tuj.m4_motion.tool_use_journal import (
    CompiledToolUseJournalCollisionModel,
    ToolUseJournalCollisionModelCompiler,
    ToolUseJournalCompatibilityError,
    ToolUseJournalEnvironmentAdapter,
    make_tool_use_journal_env,
)
from tuj.m4_motion.tool_use_journal_planning import (
    ToolUseJournalCollisionBindingError,
    ToolUseJournalCollisionContextFactory,
    ToolUseJournalMotionRequestPlanner,
    WorkcellMotionRequestRouter,
    attached_object_transform_from_state,
)
from tuj.m4_motion.tool_use_journal_runtime import (
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
from tuj.m4_motion.workcell_models import (
    CompiledWorkcellCollisionModel,
    EEWorkcellCollisionModelCompiler,
    WorkcellModelCompilationError,
)
from tuj.m4_motion.vlm_provider import (
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
