"""Versioned contract for trajectory generation and failure-localized replay.

Task Planner is the task planner.  It supplies a grounded motion task, a world
snapshot, and planning constraints.  The Motion Planner emits an internal,
time-parameterized trajectory artifact which is replayed directly by the
simulator.  Execution reports stay inside the runtime; on failure, artifact
lineage identifies the responsible module and a recovery orchestrator reruns
only that module.  There is no result callback to Task Planner.
"""

from __future__ import annotations

import enum
import json
import math
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from tuj.m4_taskplanner.models import GraspSpec

SCHEMA_VERSION = "3.2.0"
SCHEMA_ID = (
    "https://schemas.local/ee-motion-planner/"
    f"motion-planner-{SCHEMA_VERSION}.schema.json"
)


class _ContractModel(BaseModel):
    """Strict transport model; extensions belong in an explicit metadata map."""

    model_config = ConfigDict(extra="forbid")


class ModuleName(str, enum.Enum):
    TASK_PLANNER = "TASK_PLANNER"
    GRASP_PLANNER = "GRASP_PLANNER"
    MOTION_PLANNER = "MOTION_PLANNER"
    SIMULATOR = "SIMULATOR"
    CONTROLLER = "CONTROLLER"
    SCENE_MODEL = "SCENE_MODEL"
    RECOVERY_ORCHESTRATOR = "RECOVERY_ORCHESTRATOR"


class ArtifactProvenance(_ContractModel):
    """Lineage needed to trace a failed execution back to one module run."""

    artifact_id: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    produced_by: ModuleName
    invocation_id: str = Field(min_length=1)
    input_artifact_ids: list[str] = Field(default_factory=list)
    attempt: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Pose(_ContractModel):
    """Rigid pose in SI units; quaternion order is x, y, z, w."""

    frame_id: str = Field(min_length=1)
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]

    @model_validator(mode="after")
    def _validate_pose(self) -> "Pose":
        values = (*self.position_m, *self.orientation_xyzw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("pose values must be finite")
        norm = math.sqrt(sum(value * value for value in self.orientation_xyzw))
        if norm < 1e-9:
            raise ValueError("orientation quaternion must be non-zero")
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("orientation quaternion must be normalized")
        return self


class SceneRef(_ContractModel):
    """Deterministic identity and causal history for the predicted scene."""

    signature: str = Field(min_length=1)
    completed_subgoals: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)


class GripperMode(str, enum.Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    HOLDING = "HOLDING"
    UNKNOWN = "UNKNOWN"


class GripperState(_ContractModel):
    mode: GripperMode = GripperMode.UNKNOWN
    command: float | None = None


class RobotState(_ContractModel):
    """Complete state required to initialize or verify trajectory playback."""

    robot_id: str = Field(min_length=1)
    joint_names: list[str] = Field(min_length=1)
    joint_positions_rad: list[float] = Field(min_length=1)
    joint_velocities_rad_s: list[float] | None = None
    eef_pose: Pose | None = None
    gripper: GripperState | None = None
    attached_object_id: str | None = None
    held_tool_id: str | None = None

    @model_validator(mode="after")
    def _validate_joint_dimensions(self) -> "RobotState":
        dof = len(self.joint_names)
        if len(self.joint_positions_rad) != dof:
            raise ValueError("joint_positions_rad length must match joint_names")
        if (
            self.joint_velocities_rad_s is not None
            and len(self.joint_velocities_rad_s) != dof
        ):
            raise ValueError(
                "joint_velocities_rad_s length must match joint_names"
            )
        values = self.joint_positions_rad + (self.joint_velocities_rad_s or [])
        if not all(math.isfinite(value) for value in values):
            raise ValueError("robot joint values must be finite")
        return self


class WorldSnapshot(_ContractModel):
    """Atomic world and robot state used as the planning start state."""

    scene: SceneRef
    robot_state: RobotState
    objects: dict[str, Any] = Field(default_factory=dict)
    obstacles: list[Any] = Field(default_factory=list)
    rack: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyGeneratorKind(str, enum.Enum):
    VLM = "VLM"
    TEMPLATE = "TEMPLATE"
    TASK_GEOMETRY = "TASK_GEOMETRY"


class KeyframeType(str, enum.Enum):
    PRE_GRASP = "PRE_GRASP"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TRANSFER = "TRANSFER"
    PRE_PLACE = "PRE_PLACE"
    PLACE = "PLACE"
    RETREAT = "RETREAT"
    EE_UNDOCK_STAGING = "EE_UNDOCK_STAGING"
    EE_PRE_UNDOCK = "EE_PRE_UNDOCK"
    EE_UNDOCK = "EE_UNDOCK"
    EE_DOCK_STAGING = "EE_DOCK_STAGING"
    EE_PRE_DOCK = "EE_PRE_DOCK"
    EE_DOCK = "EE_DOCK"
    CUSTOM = "CUSTOM"


class KeyframePlannerType(str, enum.Enum):
    CARTESIAN = "CARTESIAN"
    JOINT = "JOINT"
    SAMPLING_BASED = "SAMPLING_BASED"


class KeyframeEventType(str, enum.Enum):
    GRIPPER_OPEN = "GRIPPER_OPEN"
    GRIPPER_CLOSE = "GRIPPER_CLOSE"
    SUCTION_ON = "SUCTION_ON"
    SUCTION_OFF = "SUCTION_OFF"
    ATTACH_OBJECT = "ATTACH_OBJECT"
    DETACH_OBJECT = "DETACH_OBJECT"
    TOOL_LOCK = "TOOL_LOCK"
    TOOL_UNLOCK = "TOOL_UNLOCK"
    VERIFY_TOOL_LOCK = "VERIFY_TOOL_LOCK"
    VERIFY_TOOL_RELEASE = "VERIFY_TOOL_RELEASE"


class StrategyGenerationProvenance(_ContractModel):
    """Identity that freezes a stochastic strategy proposal as an artifact."""

    generator_kind: StrategyGeneratorKind
    generator_id: str = Field(min_length=1)
    input_hash: str = Field(min_length=1)
    model_id: str | None = None
    prompt_hash: str | None = None
    temperature: float | None = Field(default=None, ge=0)
    provider_request_id: str | None = None
    attempt_index: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _validate_vlm_identity(self) -> "StrategyGenerationProvenance":
        if self.generator_kind is StrategyGeneratorKind.VLM:
            if not self.model_id or not self.prompt_hash:
                raise ValueError("VLM strategy provenance requires model_id and prompt_hash")
        return self


class RelativeKeyframeSpec(_ContractModel):
    """Symbolic keyframe resolved deterministically against a scene frame.

    A VLM selects a frame, named anchor, approach axis, offset, and roll.  It
    never emits a world-frame position or a free-form quaternion.
    """

    keyframe_id: str = Field(min_length=1)
    keyframe_type: KeyframeType
    frame_ref: str = Field(min_length=1)
    anchor: str = Field(min_length=1)
    approach_axis_xyz: tuple[float, float, float]
    tool_axis_to_align: Literal["+z", "-z"] = "+z"
    offset_along_approach_m: float = 0.0
    roll_rad: float = 0.0
    planner: KeyframePlannerType
    events_after: list[KeyframeEventType] = Field(default_factory=list)
    collision_context_id: str | None = None
    collision_context_after_events_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_axis(self) -> "RelativeKeyframeSpec":
        if not all(math.isfinite(value) for value in self.approach_axis_xyz):
            raise ValueError("approach_axis_xyz must be finite")
        norm = math.sqrt(sum(value * value for value in self.approach_axis_xyz))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError("approach_axis_xyz must be a unit vector")
        if not math.isfinite(self.offset_along_approach_m) or not math.isfinite(
            self.roll_rad
        ):
            raise ValueError("keyframe offset and roll must be finite")
        return self


class KeyframePlanCandidate(_ContractModel):
    """A semantically coherent approach strategy, not an independent pose bag."""

    strategy_id: str = Field(min_length=1)
    keyframes: list[RelativeKeyframeSpec] = Field(min_length=2)
    rationale: str = ""
    provenance: StrategyGenerationProvenance
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_unique_keyframes(self) -> "KeyframePlanCandidate":
        ids = [keyframe.keyframe_id for keyframe in self.keyframes]
        if len(ids) != len(set(ids)):
            raise ValueError("keyframe_id values must be unique within a strategy")
        return self


class KeyframePlanArtifact(_ContractModel):
    schema_version: Literal["3.2.0"] = SCHEMA_VERSION
    artifact_id: str = Field(min_length=1)
    provenance: ArtifactProvenance
    scene_signature: str = Field(min_length=1)
    subgoal_id: str = Field(min_length=1)
    candidates: list[KeyframePlanCandidate] = Field(min_length=1, max_length=20)


class GoalType(str, enum.Enum):
    """Representation of the motion target, independent of task semantics."""

    POSE = "POSE"
    JOINT = "JOINT"


_LEGACY_POSE_GOAL_TYPES = {
    "PICK",
    "PLACE",
    "DOCK",
    "TOOL_CHANGE",
    "EE_EXCHANGE",
    "PUSH",
    "PULL",
    "SWEEP",
    "INSERT",
    "POUR",
}


class MotionGoal(_ContractModel):
    """Grounded geometric target; manipulation meaning lives on MotionTask."""

    goal_type: GoalType
    target_pose: Pose | None = None
    target_joint_positions_rad: list[float] | None = None
    target_object_id: str | None = None
    target_region_id: str | None = None
    approach_direction: tuple[float, float, float] | None = None
    approach_distance_m: float | None = Field(default=None, gt=0)
    retreat_distance_m: float | None = Field(default=None, gt=0)

    @field_validator("goal_type", mode="before")
    @classmethod
    def _migrate_legacy_goal_type(cls, value: object) -> object:
        """Read v3.2 artifacts without retaining action taxonomy in GoalType."""

        normalized = str(getattr(value, "value", value)).strip().upper()
        if normalized in _LEGACY_POSE_GOAL_TYPES:
            return GoalType.POSE
        return value

    @model_validator(mode="after")
    def _validate_goal_target(self) -> "MotionGoal":
        if self.goal_type is GoalType.JOINT:
            if not self.target_joint_positions_rad:
                raise ValueError("JOINT goal requires target_joint_positions_rad")
        elif (
            self.target_pose is None
            and self.target_object_id is None
            and self.target_region_id is None
        ):
            # Specialized transition providers (for example EE exchange) can
            # still ground a pose from MotionTask metadata and the world.  The
            # enclosing MotionTask validator rejects an ungrounded ordinary
            # action while allowing those explicit transition operations.
            pass
        if self.approach_direction is not None:
            norm = math.sqrt(sum(v * v for v in self.approach_direction))
            if norm < 1e-9:
                raise ValueError("approach_direction must be non-zero")
        return self


class ContactSurfaceType(str, enum.Enum):
    """Stable geometric contact categories, not manipulation-action labels."""

    AUTO = "AUTO"
    BROAD_FACE = "BROAD_FACE"
    RIM = "RIM"
    EDGE = "EDGE"
    POINT = "POINT"


class ToolContactPatch(_ContractModel):
    """A usable tool surface expressed in the tool object's local frame."""

    patch_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)
    surface_type: ContactSurfaceType
    position_in_tool_m: tuple[float, float, float]
    normal_in_tool_xyz: tuple[float, float, float]
    tangent_in_tool_xyz: tuple[float, float, float] | None = None
    extent_m: tuple[float, float] | None = None
    curvature_radius_m: float | None = Field(default=None, gt=0)
    collision_geometry_refs: list[str] = Field(default_factory=list)
    supported_primitives: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_patch_frame(self) -> "ToolContactPatch":
        values = (*self.position_in_tool_m, *self.normal_in_tool_xyz)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("contact patch position and normal must be finite")
        normal = math.sqrt(sum(value * value for value in self.normal_in_tool_xyz))
        if not math.isclose(normal, 1.0, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError("contact patch normal must be a unit vector")
        if self.tangent_in_tool_xyz is not None:
            tangent = math.sqrt(
                sum(value * value for value in self.tangent_in_tool_xyz)
            )
            if not math.isclose(tangent, 1.0, rel_tol=0.0, abs_tol=1e-3):
                raise ValueError("contact patch tangent must be a unit vector")
            dot = sum(
                left * right
                for left, right in zip(
                    self.normal_in_tool_xyz, self.tangent_in_tool_xyz
                )
            )
            if not math.isclose(dot, 0.0, rel_tol=0.0, abs_tol=1e-3):
                raise ValueError("contact patch tangent must be perpendicular to normal")
        if self.extent_m is not None and any(value <= 0 for value in self.extent_m):
            raise ValueError("contact patch extents must be positive")
        return self


class ContactManipulationSpec(_ContractModel):
    """Grounded contact intent carried alongside an opaque task action label."""

    primitive: str = Field(min_length=1)
    contact_surface: ContactSurfaceType = ContactSurfaceType.AUTO
    contact_patch_id: str | None = None
    path_pattern: str = "AUTO"
    target_grouping: str = "SINGLE"
    maintain_contact: bool = False
    max_contact_force_n: float | None = Field(default=None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MotionTask(_ContractModel):
    """One executable motion task grounded by Task Planner."""

    task_id: str = Field(min_length=1)
    subgoal_id: str = Field(min_length=1)
    action_type: str = Field(min_length=1)
    ee: str = Field(min_length=1)
    tool: str | None = None
    target_ids: list[str] = Field(default_factory=list)
    grasp: GraspSpec | None = None
    goal: MotionGoal
    contact: ContactManipulationSpec | None = None
    allowed_touch_objects: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_pick_grasp(self) -> "MotionTask":
        from tuj.m5_motion.task_semantics import is_ee_exchange_task

        if self.metadata.get("require_structured_grasp") and self.grasp is None:
            raise ValueError("task explicitly requires a structured grasp")
        if (
            self.goal.goal_type is GoalType.POSE
            and self.goal.target_pose is None
            and self.goal.target_object_id is None
            and self.goal.target_region_id is None
            and not is_ee_exchange_task(self)
        ):
            raise ValueError(
                "POSE goal requires target_pose, target_object_id, or target_region_id"
            )
        return self


class JointDynamicLimit(_ContractModel):
    max_velocity_rad_s: float = Field(gt=0)
    max_acceleration_rad_s2: float = Field(gt=0)
    max_jerk_rad_s3: float | None = Field(default=None, gt=0)


class MotionConstraints(_ContractModel):
    collision_margin_m: float = Field(default=0.005, ge=0)
    position_tolerance_m: float = Field(default=0.005, gt=0)
    orientation_tolerance_rad: float = Field(default=0.05, gt=0)
    velocity_scaling: float = Field(default=0.5, gt=0, le=1)
    acceleration_scaling: float = Field(default=0.5, gt=0, le=1)
    jerk_scaling: float = Field(default=0.5, gt=0, le=1)
    max_cartesian_speed_m_s: float | None = Field(default=None, gt=0)
    min_jacobian_singular_value: float = Field(default=1e-4, ge=0)
    max_jacobian_condition_number: float = Field(default=1e4, gt=1)
    max_joint_path_step_rad: float = Field(default=0.02, gt=0)
    allowed_collision_pairs: list[tuple[str, str]] = Field(default_factory=list)
    joint_limits: dict[str, JointDynamicLimit] = Field(default_factory=dict)


class PlannerOptions(_ContractModel):
    algorithm: str = "RRT_CONNECT"
    allowed_planning_time_s: float = Field(default=5.0, gt=0)
    max_attempts: int = Field(default=5, ge=1)
    simplify_path: bool = True
    time_parameterization_algorithm: str = "QUINTIC_STOP"
    interpolation_dt_s: float = Field(default=0.02, gt=0)
    cartesian_translation_step_m: float = Field(default=0.01, gt=0)
    cartesian_rotation_step_rad: float = Field(default=0.1, gt=0)
    rrt_extension_step_rad: float = Field(default=0.25, gt=0)
    rrt_max_iterations: int = Field(default=2000, ge=1)
    rrt_goal_bias: float = Field(default=0.15, ge=0, le=1)
    random_seed: int = 0


class MotionPlanRequest(_ContractModel):
    """One-way Task Planner -> Motion Planner trajectory-generation command."""

    schema_version: Literal["3.2.0"] = SCHEMA_VERSION
    request_id: str = Field(min_length=1)
    provenance: ArtifactProvenance
    world: WorldSnapshot
    task: MotionTask
    constraints: MotionConstraints = Field(default_factory=MotionConstraints)
    options: PlannerOptions = Field(default_factory=PlannerOptions)


class InterpolationType(str, enum.Enum):
    LINEAR = "LINEAR"
    CUBIC = "CUBIC"
    QUINTIC = "QUINTIC"


class SegmentType(str, enum.Enum):
    APPROACH = "APPROACH"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TRANSFER = "TRANSFER"
    PLACE = "PLACE"
    RETREAT = "RETREAT"
    TOOL_CHANGE = "TOOL_CHANGE"
    EE_EXCHANGE = "EE_EXCHANGE"
    EE_DOCK = "EE_DOCK"
    EE_UNDOCK = "EE_UNDOCK"
    CUSTOM = "CUSTOM"


class FreeObjectPose(_ContractModel):
    """World pose override for an object backed by a MuJoCo free joint."""

    object_id: str = Field(min_length=1)
    free_joint_name: str = Field(min_length=1)
    pose: Pose

    @model_validator(mode="after")
    def _validate_world_pose(self) -> "FreeObjectPose":
        if self.pose.frame_id != "world":
            raise ValueError("free-object pose must use the world frame")
        return self


class AttachedObjectTransform(_ContractModel):
    """Rigid grasp transform used to move a free-joint object during planning."""

    object_id: str = Field(min_length=1)
    free_joint_name: str = Field(min_length=1)
    reference_kind: Literal["body", "site"] = "body"
    reference_name: str = Field(min_length=1)
    position_in_reference_m: tuple[float, float, float]
    orientation_in_reference_xyzw: tuple[float, float, float, float]

    @model_validator(mode="after")
    def _validate_transform(self) -> "AttachedObjectTransform":
        values = (
            *self.position_in_reference_m,
            *self.orientation_in_reference_xyzw,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("attached-object transform must be finite")
        norm = math.sqrt(
            sum(value * value for value in self.orientation_in_reference_xyzw)
        )
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "attached-object orientation quaternion must be normalized"
            )
        return self


class CollisionContext(_ContractModel):
    """Collision model and ACM state active at one trajectory boundary."""

    context_id: str = Field(min_length=1)
    scene_state_id: str | None = None
    active_ee: str | None = None
    attached_object_ids: list[str] = Field(default_factory=list)
    attached_object_transforms: list[AttachedObjectTransform] = Field(
        default_factory=list
    )
    free_object_poses: list[FreeObjectPose] = Field(default_factory=list)
    touch_links: list[str] = Field(default_factory=list)
    kinematic_joint_positions: dict[str, float] = Field(default_factory=dict)
    allowed_collision_pairs: list[tuple[str, str]] = Field(default_factory=list)
    collision_model_version: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_unique_state(self) -> "CollisionContext":
        if len(self.attached_object_ids) != len(set(self.attached_object_ids)):
            raise ValueError("attached_object_ids must be unique")
        canonical_pairs = [tuple(sorted(pair)) for pair in self.allowed_collision_pairs]
        if len(canonical_pairs) != len(set(canonical_pairs)):
            raise ValueError("allowed_collision_pairs must be unique")
        transform_ids = [item.object_id for item in self.attached_object_transforms]
        if len(transform_ids) != len(set(transform_ids)):
            raise ValueError("attached_object_transforms object_id values must be unique")
        if not set(transform_ids).issubset(self.attached_object_ids):
            raise ValueError(
                "attached_object_transforms must reference attached_object_ids"
            )
        free_pose_ids = [item.object_id for item in self.free_object_poses]
        if len(free_pose_ids) != len(set(free_pose_ids)):
            raise ValueError("free_object_poses object_id values must be unique")
        if set(free_pose_ids) & set(self.attached_object_ids):
            raise ValueError(
                "free_object_poses and attached_object_ids must be disjoint"
            )
        if not all(
            math.isfinite(value)
            for value in self.kinematic_joint_positions.values()
        ):
            raise ValueError("kinematic_joint_positions must be finite")
        return self


class TrajectoryProcessingStep(str, enum.Enum):
    RAW_PATH = "RAW_PATH"
    SHORTCUT = "SHORTCUT"
    TIME_PARAMETERIZATION = "TIME_PARAMETERIZATION"
    JERK_SMOOTHING = "JERK_SMOOTHING"
    FINAL_COLLISION_CHECK = "FINAL_COLLISION_CHECK"
    DYNAMICS_CHECK = "DYNAMICS_CHECK"


class TrajectoryWaypoint(_ContractModel):
    """One time-parameterized robot command sample."""

    time_from_start_s: float = Field(ge=0)
    joint_positions_rad: list[float] = Field(min_length=1)
    joint_velocities_rad_s: list[float] | None = None
    joint_accelerations_rad_s2: list[float] | None = None
    eef_pose: Pose | None = None

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "TrajectoryWaypoint":
        dof = len(self.joint_positions_rad)
        for name, values in (
            ("joint_velocities_rad_s", self.joint_velocities_rad_s),
            ("joint_accelerations_rad_s2", self.joint_accelerations_rad_s2),
        ):
            if values is not None and len(values) != dof:
                raise ValueError(f"{name} length must match joint_positions_rad")
        values = (
            self.joint_positions_rad
            + (self.joint_velocities_rad_s or [])
            + (self.joint_accelerations_rad_s2 or [])
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("trajectory joint values must be finite")
        return self


class TrajectorySegment(_ContractModel):
    """Semantically labelled, collision-checked part of the trajectory."""

    segment_id: str = Field(min_length=1)
    segment_type: SegmentType
    start_time_s: float = Field(ge=0)
    end_time_s: float = Field(gt=0)
    interpolation: InterpolationType = InterpolationType.CUBIC
    waypoints: list[TrajectoryWaypoint] = Field(min_length=2)
    collision_checked: bool
    min_clearance_m: float | None = Field(default=None, ge=0)
    collision_context_before: CollisionContext | None = None
    collision_context_after: CollisionContext | None = None
    processing_steps: list[TrajectoryProcessingStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_timeline(self) -> "TrajectorySegment":
        if self.end_time_s <= self.start_time_s:
            raise ValueError("segment end_time_s must be greater than start_time_s")
        times = [waypoint.time_from_start_s for waypoint in self.waypoints]
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("waypoint times must be strictly increasing")
        if not math.isclose(times[0], self.start_time_s, abs_tol=1e-9):
            raise ValueError("first waypoint time must match segment start_time_s")
        if not math.isclose(times[-1], self.end_time_s, abs_tol=1e-9):
            raise ValueError("last waypoint time must match segment end_time_s")
        dof = len(self.waypoints[0].joint_positions_rad)
        if any(len(w.joint_positions_rad) != dof for w in self.waypoints):
            raise ValueError("all segment waypoints must use the same DOF")
        return self


class EventType(str, enum.Enum):
    GRIPPER_OPEN = "GRIPPER_OPEN"
    GRIPPER_CLOSE = "GRIPPER_CLOSE"
    SUCTION_ON = "SUCTION_ON"
    SUCTION_OFF = "SUCTION_OFF"
    ATTACH_OBJECT = "ATTACH_OBJECT"
    DETACH_OBJECT = "DETACH_OBJECT"
    TOOL_LOCK = "TOOL_LOCK"
    TOOL_UNLOCK = "TOOL_UNLOCK"
    VERIFY_TOOL_LOCK = "VERIFY_TOOL_LOCK"
    VERIFY_TOOL_RELEASE = "VERIFY_TOOL_RELEASE"
    WAIT = "WAIT"


class TrajectoryEvent(_ContractModel):
    """Discrete command synchronized to the trajectory clock."""

    event_id: str = Field(min_length=1)
    time_from_start_s: float = Field(ge=0)
    event_type: EventType
    target_id: str | None = None
    command: float | bool | str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class MotionPlan(_ContractModel):
    """Internal executable artifact consumed directly by the simulator."""

    plan_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    provenance: ArtifactProvenance
    scene_signature: str = Field(min_length=1)
    robot_id: str = Field(min_length=1)
    joint_names: list[str] = Field(min_length=1)
    duration_s: float = Field(gt=0)
    segments: list[TrajectorySegment] = Field(min_length=1)
    events: list[TrajectoryEvent] = Field(default_factory=list)
    expected_final_state: RobotState
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_plan(self) -> "MotionPlan":
        if self.segments[0].start_time_s != 0:
            raise ValueError("the first trajectory segment must start at t=0")
        for previous, current in zip(self.segments, self.segments[1:]):
            if current.start_time_s < previous.end_time_s:
                raise ValueError("trajectory segments must not overlap")
            after = previous.collision_context_after
            before = current.collision_context_before
            if after is not None and before is not None:
                after_state = (
                    after.scene_state_id or after.context_id,
                    after.active_ee,
                    tuple(sorted(after.attached_object_ids)),
                    after.collision_model_version,
                )
                before_state = (
                    before.scene_state_id or before.context_id,
                    before.active_ee,
                    tuple(sorted(before.attached_object_ids)),
                    before.collision_model_version,
                )
                if after_state != before_state:
                    raise ValueError(
                        "adjacent segment collision contexts must preserve scene state"
                    )
        final_time = self.segments[-1].end_time_s
        if not math.isclose(self.duration_s, final_time, abs_tol=1e-9):
            raise ValueError("duration_s must match the final segment end time")
        dof = len(self.joint_names)
        for segment in self.segments:
            if any(len(w.joint_positions_rad) != dof for w in segment.waypoints):
                raise ValueError("waypoint DOF must match plan joint_names")
        if any(event.time_from_start_s > self.duration_s for event in self.events):
            raise ValueError("trajectory events must occur within plan duration")
        if self.expected_final_state.robot_id != self.robot_id:
            raise ValueError("expected_final_state robot_id must match plan robot_id")
        if self.expected_final_state.joint_names != self.joint_names:
            raise ValueError("expected_final_state joint_names must match plan")
        return self


class SimulationConfig(_ContractModel):
    physics_timestep_s: float = Field(default=0.002, gt=0)
    control_timestep_s: float = Field(default=0.02, gt=0)
    realtime_factor: float = Field(default=0.0, ge=0)
    max_duration_s: float | None = Field(default=None, gt=0)
    terminate_on_collision: bool = True
    render: bool = False
    random_seed: int = 0


class SimulationRun(_ContractModel):
    """Internal command that binds an exact plan to simulator settings."""

    schema_version: Literal["3.2.0"] = SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    provenance: ArtifactProvenance
    plan: MotionPlan
    config: SimulationConfig = Field(default_factory=SimulationConfig)


class EventExecutionStatus(str, enum.Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ExecutedEvent(_ContractModel):
    event_id: str = Field(min_length=1)
    scheduled_time_s: float = Field(ge=0)
    actual_time_s: float = Field(ge=0)
    status: EventExecutionStatus
    message: str = ""


class ExecutionStatus(str, enum.Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    ABORTED = "ABORTED"


class SimulationMetrics(_ContractModel):
    executed_duration_s: float = Field(ge=0)
    max_joint_tracking_error_rad: float = Field(ge=0)
    max_eef_tracking_error_m: float | None = Field(default=None, ge=0)
    collision_count: int = Field(default=0, ge=0)
    goal_position_error_m: float | None = Field(default=None, ge=0)
    goal_orientation_error_rad: float | None = Field(default=None, ge=0)


class FailureObservation(_ContractModel):
    """What failed in simulation, before assigning a root cause."""

    code: str = Field(min_length=1)
    category: str = Field(min_length=1)
    message: str = Field(min_length=1)
    segment_id: str | None = None
    waypoint_index: int | None = Field(default=None, ge=0)
    event_id: str | None = None
    expected: dict[str, Any] = Field(default_factory=dict)
    observed: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)


class ExecutionReport(_ContractModel):
    """Internal execution record stored for evaluation and failure tracing."""

    schema_version: Literal["3.2.0"] = SCHEMA_VERSION
    report_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    provenance: ArtifactProvenance
    status: ExecutionStatus
    final_robot_state: RobotState | None = None
    metrics: SimulationMetrics
    executed_events: list[ExecutedEvent] = Field(default_factory=list)
    failure: FailureObservation | None = None
    trace_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_execution_report(self) -> "ExecutionReport":
        if self.status is ExecutionStatus.SUCCESS and self.final_robot_state is None:
            raise ValueError("successful simulation requires final_robot_state")
        if self.status is not ExecutionStatus.SUCCESS and self.failure is None:
            raise ValueError("failed simulation requires a failure observation")
        return self


class RootCause(_ContractModel):
    """Module invocation selected by causal analysis as the retry boundary."""

    module: ModuleName
    invocation_id: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    cause_code: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    evidence_refs: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)


class RetryPolicy(_ContractModel):
    current_attempt: int = Field(default=1, ge=1)
    max_attempts: int = Field(default=3, ge=1)
    parameter_overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_attempts(self) -> "RetryPolicy":
        if self.current_attempt > self.max_attempts:
            raise ValueError("current_attempt must not exceed max_attempts")
        return self


class RecoveryDirective(_ContractModel):
    """Recovery Orchestrator command: rerun only the root-cause module."""

    schema_version: Literal["3.2.0"] = SCHEMA_VERSION
    directive_id: str = Field(min_length=1)
    source_report_id: str = Field(min_length=1)
    provenance: ArtifactProvenance
    root_cause: RootCause
    target_module: ModuleName
    restart_from_artifact_id: str = Field(min_length=1)
    invalidated_artifact_ids: list[str] = Field(default_factory=list)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)

    @model_validator(mode="after")
    def _validate_target_module(self) -> "RecoveryDirective":
        if self.target_module is not self.root_cause.module:
            raise ValueError("target_module must match root_cause.module")
        return self


MotionPlannerPayload: TypeAlias = (
    KeyframePlanArtifact
    | MotionPlanRequest
    | MotionPlan
    | SimulationRun
    | ExecutionReport
    | RecoveryDirective
)

_PAYLOAD_ADAPTER = TypeAdapter(MotionPlannerPayload)


def motion_planner_json_schema() -> dict[str, object]:
    """Return the canonical direct-execution/recovery contract as JSON Schema."""

    schema = _PAYLOAD_ADAPTER.json_schema(
        ref_template="#/$defs/{model}",
        union_format="any_of",
    )
    schema.update(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": SCHEMA_ID,
            "title": "EE Motion Planner Direct Execution Contract",
            "description": (
                "One-way Task Planner task commands, internal trajectory artifacts, "
                "simulation execution reports, artifact lineage, and localized "
                "module retry directives. No result is returned to Task Planner."
            ),
            "x-schema-version": SCHEMA_VERSION,
            "x-units": {
                "position": "metre",
                "orientation": "quaternion_xyzw or radian",
                "joint_position": "radian",
                "joint_velocity": "radian/second",
                "joint_acceleration": "radian/second^2",
                "joint_jerk": "radian/second^3",
                "time": "second",
            },
            "x-contract-roots": [
                "KeyframePlanArtifact",
                "MotionPlanRequest",
                "MotionPlan",
                "SimulationRun",
                "ExecutionReport",
                "RecoveryDirective",
            ],
            "x-operations": {
                "freeze_keyframe_candidates": {
                    "input": "MotionPlanRequest",
                    "emits": "KeyframePlanArtifact",
                    "visibility": "internal",
                },
                "generate_trajectory": {
                    "input": "MotionPlanRequest",
                    "emits": "MotionPlan",
                    "visibility": "internal",
                },
                "execute_simulation": {
                    "input": "SimulationRun",
                    "emits": "ExecutionReport",
                    "visibility": "internal",
                },
                "recover_failure": {
                    "input": "ExecutionReport",
                    "emits": "RecoveryDirective",
                    "visibility": "internal",
                },
            },
            "x-result-routing": {
                "return_to_task_planner": False,
                "success": "archive ExecutionReport and continue pipeline",
                "failure": "trace artifact lineage and rerun root-cause module",
            },
        }
    )
    return dict(sorted(schema.items()))


def write_json_schema(path: str | Path) -> Path:
    """Write the schema as stable UTF-8 JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(motion_planner_json_schema(), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return destination.resolve()


if __name__ == "__main__":
    default_path = (
        Path(__file__).resolve().parent
        / "schemas"
        / "motion_planner.schema.json"
    )
    print(write_json_schema(default_path))
