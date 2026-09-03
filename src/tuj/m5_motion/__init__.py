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

from tuj.m5_motion.compiler import FirstFeasibleStrategyCompiler
from tuj.m5_motion.ee_exchange import (
    EEExchangeKeyframeProvider,
    EEExchangeTemplateGenerator,
    RoutedKeyframeStrategyProvider,
)
from tuj.m5_motion.execution import (
    ExecutionAcceptance,
    GoalEvaluation,
    GoalEvaluationStatus,
    GroundedMotionGoalEvaluator,
    SelectedPlanExecutionResult,
    SelectedPlanSimulationOrchestrator,
    SequenceExecutionStatus,
    SimulationArtifactStore,
)
from tuj.m5_motion.closed_loop_contact import (
    ClosedLoopContactExecutor,
    ClosedLoopContactResult,
    ClosedLoopContactStatus,
    ContactAttemptRecord,
    ContactCheckpointBackend,
    ContactStepObservation,
    ContactStepRunner,
)
from tuj.m5_motion.contact_evaluation import (
    CompositeGoalEvaluator,
    GraspRetentionEvaluator,
    RegionContainmentEvaluator,
    SupportStabilityEvaluator,
    TaskAwareGoalEvaluator,
    ToolClearanceEvaluator,
)
from tuj.m5_motion.geometry import RelativePoseResolver
from tuj.m5_motion.kinematics import (
    IKResult,
    IKSolutionSet,
    UR5eKinematics,
    default_model_path,
)
from tuj.m5_motion.oracle import MuJoCoMotionOracle
from tuj.m5_motion.mujoco_collision import (
    CollisionCheckResult,
    MuJoCoCollisionModelRegistry,
    MuJoCoCollisionValidator,
    MuJoCoInterpolatingEdgePlanner,
    PathCollisionCheckResult,
)
from tuj.m5_motion.plan_builder import MotionPlanBuilder
from tuj.m5_motion.precomputed_ee_attach import (
    EEAttachPathFailureCode,
    EEAttachPolicy,
    EEAttachTrajectoryTemplate,
    PrecomputedEEAttachPlanner,
    PrecomputedEEAttachRegistry,
    PrecomputedEEPathError,
    compute_rack_signature,
    compute_workcell_signature,
    save_ee_attach_template,
)
from tuj.m5_motion.ee_exchange_entry import (
    EEExchangeEntryFailureCode,
    EEExchangeEntryPlanner,
    EEExchangeEntryPlanningError,
    is_ee_exchange_entry_request,
)
from tuj.m5_motion.precomputed_ee_exchange import (
    EEReturnTrajectoryTemplate,
    PrecomputedEEExchangePlanner,
    PrecomputedEEReturnRegistry,
    derive_return_template_from_attach,
    is_ee_exchange_request,
    save_ee_return_template,
)
from tuj.m5_motion.orchestration import (
    MotionPlanStore,
    SelectedPlanMotionOrchestrator,
    SelectedPlanPlanningResult,
)
from tuj.m5_motion.path_planning import (
    CartesianEdgePlanner,
    PlannerDispatchEdgePlanner,
    RRTConnectEdgePlanner,
)
from tuj.m5_motion.profiles import (
    ContactExecutionProfile,
    GraspExecutionProfile,
    PushPlanningProfile,
    RobotControlProfile,
    TaskRecoveryProfile,
    ToolAffordanceProfile,
)
from tuj.m5_motion.push_to_region import (
    ContactPoseResolver,
    DirectToolContactPoseResolver,
    PushToRegionError,
    PushToRegionGeometry,
    PushToRegionStrategyProvider,
    cleanup_target_ids,
    order_targets_around_region,
    push_to_region_geometry,
    reduced_contact_step_distance,
    target_fully_inside_region,
)
from tuj.m5_motion.safety import KinematicSafetyValidator
from tuj.m5_motion.selected_plan_adapter import (
    SelectedPlanAdapterError,
    SelectedPlanMotionRequestAdapter,
    selected_plan_to_motion_requests,
)
from tuj.m5_motion.pipeline import (
    CollisionContextFactory,
    CollisionPlanningSetup,
    KeyframeStrategyProvider,
    MotionPlanningPipeline,
    MotionPlanningPipelineError,
    MotionPlanningResult,
)
from tuj.m5_motion.recovery import (
    ArtifactLineageIndex,
    RecoveryAttributionError,
    RecoveryExecutionError,
    RecoveryExecutionResult,
    RecoveryOrchestrator,
)
from tuj.m5_motion.strategy import (
    FirstFeasibleBranchSelector,
    InterpolatingEdgePlanner,
)
from tuj.m5_motion.trajectory_processing import (
    QuinticTimeParameterizer,
    deterministic_shortcut,
)
from tuj.m5_motion.tool_use_journal import (
    CompiledToolUseJournalCollisionModel,
    ToolUseJournalCollisionModelCompiler,
    ToolUseJournalCompatibilityError,
    ToolUseJournalEnvironmentAdapter,
    make_tool_use_journal_env,
    registered_tool_use_journal_environments,
)
from tuj.m5_motion.tool_affordance import (
    CircularPlateAffordanceProvider,
    CompositeToolAffordanceProvider,
    StaticToolAffordanceProvider,
    ToolAffordanceError,
    ToolAffordanceProvider,
    select_contact_patch,
)
from tuj.m5_motion.tool_use_journal_planning import (
    ToolUseJournalCollisionBindingError,
    ToolUseJournalCollisionContextFactory,
    ToolUseJournalMotionRequestPlanner,
    WorkcellMotionRequestRouter,
    attached_object_transform_from_state,
)
from tuj.m5_motion.tool_use_journal_execution import (
    ToolUseJournalExecutionAdapter,
)
from tuj.m5_motion.tool_use_journal_runtime import (
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
from tuj.m5_motion.workcell_models import (
    CompiledWorkcellCollisionModel,
    EEWorkcellCollisionModelCompiler,
    WorkcellModelCompilationError,
)
from tuj.m5_motion.vlm_provider import (
    MissingOpenAIAPIKeyError,
    OpenAIKeyframeProvider,
    OpenAIKeyframeProviderConfig,
    OpenAIKeyframeProviderError,
)

__all__ = [
    "AttachmentBreakObservation",
    "AttachmentContactMetrics",
    "AttachmentMode",
    "CircularPlateAffordanceProvider",
    "ClosedLoopContactExecutor",
    "ClosedLoopContactResult",
    "ClosedLoopContactStatus",
    "CompositeGoalEvaluator",
    "CompositeToolAffordanceProvider",
    "ContactAttemptRecord",
    "ContactCheckpointBackend",
    "ContactExecutionProfile",
    "ContactPoseResolver",
    "ContactStepObservation",
    "ContactStepRunner",
    "DirectToolContactPoseResolver",
    "ArtifactLineageIndex",
    "EEExchangeTemplateGenerator",
    "EEExchangeKeyframeProvider",
    "EEAttachPathFailureCode",
    "EEAttachPolicy",
    "EEAttachTrajectoryTemplate",
    "AttachedObjectState",
    "BreakableWeldConfig",
    "EERuntimeTransition",
    "ExecutionAcceptance",
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
    "GoalEvaluation",
    "GoalEvaluationStatus",
    "GroundedMotionGoalEvaluator",
    "GraspExecutionProfile",
    "GraspRetentionEvaluator",
    "MissingOpenAIAPIKeyError",
    "MotionPlanBuilder",
    "MotionPlanStore",
    "MotionPlanningPipeline",
    "MotionPlanningPipelineError",
    "MotionPlanningResult",
    "PrecomputedEEAttachPlanner",
    "PrecomputedEEAttachRegistry",
    "PrecomputedEEPathError",
    "OpenAIKeyframeProvider",
    "OpenAIKeyframeProviderConfig",
    "OpenAIKeyframeProviderError",
    "QuinticTimeParameterizer",
    "PathCollisionCheckResult",
    "PushPlanningProfile",
    "PushToRegionError",
    "PushToRegionGeometry",
    "PushToRegionStrategyProvider",
    "RegionContainmentEvaluator",
    "RelativePoseResolver",
    "RecoveryAttributionError",
    "RecoveryExecutionError",
    "RecoveryExecutionResult",
    "RecoveryOrchestrator",
    "RoutedKeyframeStrategyProvider",
    "RobotControlProfile",
    "CartesianEdgePlanner",
    "PlannerDispatchEdgePlanner",
    "RRTConnectEdgePlanner",
    "KinematicSafetyValidator",
    "compute_rack_signature",
    "compute_workcell_signature",
    "save_ee_attach_template",
    "SelectedPlanAdapterError",
    "SelectedPlanMotionRequestAdapter",
    "SelectedPlanMotionOrchestrator",
    "SelectedPlanPlanningResult",
    "SelectedPlanExecutionResult",
    "SelectedPlanSimulationOrchestrator",
    "SequenceExecutionStatus",
    "SimulationArtifactStore",
    "StaticToolAffordanceProvider",
    "SupportStabilityEvaluator",
    "TaskAwareGoalEvaluator",
    "TaskRecoveryProfile",
    "selected_plan_to_motion_requests",
    "ToolUseJournalCollisionModelCompiler",
    "ToolUseJournalCollisionBindingError",
    "ToolUseJournalCollisionContextFactory",
    "ToolUseJournalCompatibilityError",
    "ToolUseJournalControllerTrajectoryPlayer",
    "ToolUseJournalEnvironmentAdapter",
    "ToolUseJournalExecutionAdapter",
    "ToolUseJournalEERuntime",
    "ToolUseJournalKinematicTrajectoryPlayer",
    "ToolUseJournalMotionRequestPlanner",
    "ToolUseJournalRuntimeError",
    "ToolUseJournalAttachmentBroken",
    "ToolAffordanceError",
    "ToolAffordanceProvider",
    "ToolAffordanceProfile",
    "ToolClearanceEvaluator",
    "UR5eKinematics",
    "WorkcellModelCompilationError",
    "WorkcellMotionRequestRouter",
    "attached_object_transform_from_state",
    "cleanup_target_ids",
    "default_model_path",
    "deterministic_shortcut",
    "make_tool_use_journal_env",
    "registered_tool_use_journal_environments",
    "order_targets_around_region",
    "push_to_region_geometry",
    "reduced_contact_step_distance",
    "select_contact_patch",
    "target_fully_inside_region",
    "tool_use_journal_joint_position_controller_config",
]
