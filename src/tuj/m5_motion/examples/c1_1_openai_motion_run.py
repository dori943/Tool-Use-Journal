"""Plan and replay the C1_1 plate pick and sweep with OpenAI keyframes.

The OpenAI model proposes only scene-relative Cartesian keyframe strategies.
UR5e IK branches, collision contexts, path connections, timing, attachment,
controller tracking, and MuJoCo execution remain deterministic local code.

``OPENAI_API_KEY`` must be supplied in the process environment.  It is never
written to the generated artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable

import cv2
import mujoco
import numpy as np

SRC = Path(__file__).resolve().parents[3]
REPOSITORY = SRC.parent
TASK_PLANNER_SOURCES = (
    REPOSITORY.parent / "dain-m3" / "src",
    REPOSITORY.parent / "tuj-m3" / "src",
)
for package_root in reversed((SRC, *TASK_PLANNER_SOURCES)):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from tuj.m4_taskplanner.models import GraspSpec  # noqa: E402
from tuj.m4_taskplanner.serialization import PlanningResult  # noqa: E402

from tuj.m5_motion.geometry import RelativePoseResolver  # noqa: E402
from tuj.m5_motion.pipeline import (  # noqa: E402
    MotionPlanningPipeline,
    MotionPlanningPipelineError,
)
from tuj.m5_motion.push_to_region import (  # noqa: E402
    order_targets_around_region,
    reduced_contact_step_distance,
)
from tuj.m5_motion.schema import (  # noqa: E402
    ArtifactProvenance,
    AttachedObjectTransform,
    CollisionContext,
    ContactManipulationSpec,
    ContactSurfaceType,
    GoalType,
    JointDynamicLimit,
    KeyframeEventType,
    KeyframePlanArtifact,
    KeyframePlanCandidate,
    KeyframePlannerType,
    KeyframeType,
    ModuleName,
    MotionConstraints,
    MotionGoal,
    MotionPlanRequest,
    MotionTask,
    PlannerOptions,
    Pose,
    RelativeKeyframeSpec,
    SimulationConfig,
    SimulationRun,
    StrategyGenerationProvenance,
    StrategyGeneratorKind,
    TrajectorySegment,
)
from tuj.m5_motion.tool_use_journal import (  # noqa: E402
    ToolUseJournalCollisionModelCompiler,
    ToolUseJournalEnvironmentAdapter,
)
from tuj.m5_motion.tool_use_journal_runtime import (  # noqa: E402
    ToolUseJournalControllerTrajectoryPlayer,
    ToolUseJournalEERuntime,
)
from tuj.m5_motion.tool_use_journal_planning import (  # noqa: E402
    ToolUseJournalCollisionContextFactory,
    attached_object_transform_from_state,
)
from tuj.m5_motion.vlm_provider import (  # noqa: E402
    OpenAIKeyframeProvider,
    OpenAIKeyframeProviderConfig,
)


@dataclass(frozen=True)
class _FrozenProvider:
    artifact: KeyframePlanArtifact

    def generate(self, request: MotionPlanRequest) -> KeyframePlanArtifact:
        if self.artifact.scene_signature != request.world.scene.signature:
            raise ValueError("frozen keyframes belong to another scene")
        return self.artifact


@dataclass(frozen=True)
class _C1MotionProfile:
    """M4-owned, scenario-tunable metric and controller-settle parameters."""

    plate_table_clearance_m: float = 0.0015
    sweep_start_offset_m: float = 0.060
    sweep_end_offset_m: float = 0.040
    hover_height_m: float = 0.10
    sweep_tool_axis_elevation_rad: float = -float(np.pi / 12.0)
    sweep_tool_roll_rad: float = -float(np.pi / 3.0)
    sweep_collision_margin_m: float = 0.002
    sweep_allowed_planning_time_s: float = 30.0
    sweep_min_jacobian_singular_value: float = 5.0e-5
    sweep_max_jacobian_condition_number: float = 2.0e4
    sweep_max_joint_path_step_rad: float = 0.015
    sweep_velocity_scaling: float = 0.60
    sweep_acceleration_scaling: float = 0.60
    sweep_jerk_scaling: float = 0.60
    sweep_tool_xy_tolerance_m: float = 0.015
    sweep_clearance_tolerance_m: float = 0.003
    sweep_max_table_penetration_m: float = 0.001
    sweep_max_tool_speed_m_s: float = 0.020
    sweep_clearance_control_gain: float = 4.0
    sweep_clearance_control_rate_m_s: float = 0.120
    sweep_clearance_control_max_offset_m: float = 0.120
    sweep_clearance_control_activation_band_m: float = 0.080
    sweep_tool_control_max_joint_offset_rad: float = 0.150
    sweep_push_contact_penetration_m: float = 0.001
    sweep_push_control_max_offset_m: float = 0.080
    sweep_push_plan_time_scale: float = 0.50
    sweep_push_reacquire_timeout_s: float = 8.0
    sweep_push_max_reacquire_attempts: int = 2
    sweep_push_contact_height_fraction: float = 0.50
    sweep_push_contact_height_gain: float = 2.0
    sweep_push_contact_height_rate_m_s: float = 0.030
    sweep_push_contact_height_max_offset_m: float = 0.020
    sweep_push_contact_height_max_downward_offset_m: float = 0.002
    sweep_block_support_tolerance_m: float = 0.001
    sweep_micro_push_distance_m: float = 0.030
    sweep_micro_push_min_distance_m: float = 0.0075
    sweep_micro_push_retry_scale: float = 0.50
    sweep_micro_push_min_progress_m: float = 0.005
    sweep_micro_push_max_attempts_per_block: int = 24
    sweep_max_planning_retries_per_block: int = 12
    sweep_cleanup_max_passes: int = 3
    sweep_max_contact_continuation_duration_s: float = 15.0
    sweep_max_recovery_duration_s: float = 45.0
    sweep_recovery_standoff_m: float = 0.020
    sweep_approach_standoff_step_m: float = 0.015
    sweep_recovery_lift_m: float = 0.025
    sweep_max_block_lift_m: float = 0.003
    sweep_goal_inset_margin_m: float = 0.003
    sweep_plane_alignment_blend: float = 1.0
    sweep_plane_alignment_min_blend: float = 0.0
    sweep_plane_alignment_candidate_count: int = 5
    sweep_broad_face_max_normal_deviation_rad: float = float(np.pi / 4.0)
    sweep_positive_y_roll_switch: float = 0.80
    far_corner_min_x_m: float = 0.23
    far_corner_min_abs_y_from_zone_m: float = 0.20
    far_corner_roll_rad: float = float(np.pi / 3.0)
    engage_hold_s: float = 0.20
    sweep_hold_s: float = 0.50
    intermediate_retract_hold_s: float = 0.10
    final_retract_hold_s: float = 0.75
    settle_joint_tolerance_rad: float = 0.020
    settle_eef_tolerance_m: float = 0.010
    settle_max_wait_s: float = 3.0
    settle_required_consecutive_ticks: int = 5
    finger_centerline_inset_m: float = 0.021
    pick_grasp_site_offset_m: float = 0.0
    pick_approach_distance_m: float = 0.08
    pick_retreat_distance_m: float = 0.10
    pick_grasp_hold_s: float = 1.50
    pick_gripper_close_rate: float = 1.0
    pick_gripper_closure_actuator_kp: float = 20.0
    pick_gripper_max_actuator_kp: float = 500.0
    pick_gripper_force_feedback_gain: float = 0.20
    pick_grip_force_safety_factor: float = 2.0
    pick_velocity_scaling: float = 0.25
    pick_acceleration_scaling: float = 0.25
    pick_jerk_scaling: float = 0.25
    pick_contact_freeze_ticks: int = 2
    pick_min_lift_m: float = 0.075
    pick_min_bottom_clearance_m: float = 0.030
    pick_final_hold_s: float = 5.0
    pick_contact_loss_grace_s: float = 0.06
    pick_required_contact_ticks: int = 5
    pick_min_contact_separation_m: float = 0.003
    pick_min_normal_opposition: float = 0.50
    pick_max_friction_utilization: float = 0.95
    pick_contact_follow_gain: float = 0.50
    pick_contact_follow_max_m: float = 0.040
    pick_contact_follow_activation_ticks: int = 10
    pick_contact_follow_max_tick_m: float = 0.002
    pick_contact_follow_max_joint_step_rad: float = 0.020
    pick_regrasp_roll_rad: float = -float(np.pi / 2.0)
    pick_regrasp_roll_rate_rad_s: float = 1.0
    pick_regrasp_min_separation_ratio: float = 0.75
    pick_preshape_clearance_m: float = 0.020
    pick_preshape_tolerance_m: float = 0.002
    pick_preshape_settle_ticks: int = 25
    pick_side_grasp_radial_inset_m: float = 0.0
    pick_side_grasp_vertical_offset_m: float = 0.0015
    pick_side_grasp_variant_index: int = 0
    pick_side_grasp_lift_m: float = 0.10
    pick_side_grasp_seat_start_lift_m: float = 0.0
    pick_side_grasp_seat_descent_m: float = 0.0
    pick_side_grasp_seat_radial_inset_m: float = 0.0
    pick_side_grasp_approach_elevation_rad: float = float(np.pi / 4.0)
    pick_side_grasp_roll_rad: float = 0.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "_C1MotionProfile":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown C1 motion-profile fields: {unknown}")
        profile = cls(**payload)  # type: ignore[arg-type]
        for name in allowed:
            value = getattr(profile, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"motion-profile {name} must be numeric")
        non_negative = {
            "plate_table_clearance_m",
            "sweep_max_table_penetration_m",
            "sweep_push_contact_penetration_m",
            "sweep_plane_alignment_min_blend",
            "sweep_push_contact_height_fraction",
            "pick_preshape_clearance_m",
            "pick_preshape_tolerance_m",
            "pick_side_grasp_radial_inset_m",
            "pick_side_grasp_vertical_offset_m",
            "pick_side_grasp_variant_index",
            "pick_side_grasp_seat_descent_m",
            "pick_side_grasp_seat_start_lift_m",
            "pick_side_grasp_seat_radial_inset_m",
        }
        signed = {
            "pick_grasp_site_offset_m",
            "pick_regrasp_roll_rad",
            "sweep_tool_axis_elevation_rad",
            "sweep_tool_roll_rad",
            "pick_side_grasp_roll_rad",
        }
        positive = allowed - non_negative - signed
        for name in positive:
            value = getattr(profile, name)
            if float(value) <= 0.0:
                raise ValueError(f"motion-profile {name} must be positive")
        for name in non_negative:
            if float(getattr(profile, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if abs(profile.pick_grasp_site_offset_m) > 0.02:
            raise ValueError("pick_grasp_site_offset_m must be within +/- 0.02 m")
        for name in (
            "settle_required_consecutive_ticks",
            "pick_required_contact_ticks",
            "pick_contact_freeze_ticks",
            "pick_contact_follow_activation_ticks",
            "pick_preshape_settle_ticks",
            "pick_side_grasp_variant_index",
            "sweep_push_max_reacquire_attempts",
            "sweep_micro_push_max_attempts_per_block",
            "sweep_max_planning_retries_per_block",
            "sweep_cleanup_max_passes",
            "sweep_plane_alignment_candidate_count",
        ):
            if not isinstance(getattr(profile, name), int):
                raise ValueError(f"{name} must be an integer")
        if profile.pick_gripper_close_rate > 1.0:
            raise ValueError("pick_gripper_close_rate must be at most 1.0")
        if profile.pick_gripper_force_feedback_gain > 1.0:
            raise ValueError(
                "pick_gripper_force_feedback_gain must be at most 1.0"
            )
        if (
            profile.pick_gripper_max_actuator_kp
            < profile.pick_gripper_closure_actuator_kp
        ):
            raise ValueError(
                "pick_gripper_max_actuator_kp must be at least the initial kp"
            )
        for name in (
            "pick_velocity_scaling",
            "pick_acceleration_scaling",
            "pick_jerk_scaling",
            "sweep_velocity_scaling",
            "sweep_acceleration_scaling",
            "sweep_jerk_scaling",
        ):
            if float(getattr(profile, name)) > 1.0:
                raise ValueError(f"{name} must be at most 1.0")
        if profile.pick_side_grasp_variant_index > 3:
            raise ValueError("pick_side_grasp_variant_index must be within 0..3")
        if profile.sweep_push_plan_time_scale > 1.0:
            raise ValueError("sweep_push_plan_time_scale must be at most 1.0")
        if profile.sweep_micro_push_retry_scale > 1.0:
            raise ValueError("sweep_micro_push_retry_scale must be at most 1.0")
        if (
            profile.sweep_micro_push_min_distance_m
            > profile.sweep_micro_push_distance_m
        ):
            raise ValueError(
                "sweep_micro_push_min_distance_m must not exceed "
                "sweep_micro_push_distance_m"
            )
        if profile.sweep_plane_alignment_blend > 1.0:
            raise ValueError("sweep_plane_alignment_blend must be at most 1.0")
        if profile.sweep_plane_alignment_min_blend > 1.0:
            raise ValueError(
                "sweep_plane_alignment_min_blend must be at most 1.0"
            )
        if (
            profile.sweep_plane_alignment_min_blend
            > profile.sweep_plane_alignment_blend
        ):
            raise ValueError(
                "sweep_plane_alignment_min_blend must not exceed the maximum"
            )
        if profile.sweep_recovery_standoff_m > profile.sweep_start_offset_m:
            raise ValueError(
                "sweep_recovery_standoff_m must not exceed sweep_start_offset_m"
            )
        if not 0.0 <= profile.sweep_push_contact_height_fraction <= 1.0:
            raise ValueError(
                "sweep_push_contact_height_fraction must be within [0, 1]"
            )
        if profile.pick_side_grasp_approach_elevation_rad >= np.pi / 2.0:
            raise ValueError(
                "pick_side_grasp_approach_elevation_rad must be below pi/2"
            )
        if abs(profile.sweep_tool_axis_elevation_rad) >= np.pi / 2.0:
            raise ValueError(
                "sweep_tool_axis_elevation_rad must be within (-pi/2, pi/2)"
            )
        if abs(profile.sweep_tool_roll_rad) > np.pi:
            raise ValueError("sweep_tool_roll_rad must be within +/- pi")
        if profile.sweep_broad_face_max_normal_deviation_rad > np.pi / 4.0:
            raise ValueError(
                "sweep_broad_face_max_normal_deviation_rad must be at most pi/4"
            )
        if abs(profile.pick_side_grasp_roll_rad) > np.pi:
            raise ValueError("pick_side_grasp_roll_rad must be within +/- pi")
        if abs(profile.pick_regrasp_roll_rad) > np.pi:
            raise ValueError("pick_regrasp_roll_rad must be within +/- pi")
        if not 0.0 < profile.pick_regrasp_min_separation_ratio <= 1.0:
            raise ValueError(
                "pick_regrasp_min_separation_ratio must be within (0, 1]"
            )
        if not 0.0 < profile.pick_min_normal_opposition <= 1.0:
            raise ValueError("pick_min_normal_opposition must be within (0, 1]")
        if not 0.0 < profile.pick_max_friction_utilization <= 1.0:
            raise ValueError(
                "pick_max_friction_utilization must be within (0, 1]"
            )
        if profile.pick_contact_follow_gain > 1.0:
            raise ValueError("pick_contact_follow_gain must be at most 1.0")
        if profile.sweep_positive_y_roll_switch > 1.0:
            raise ValueError(
                "sweep_positive_y_roll_switch must be at most 1.0"
            )
        if profile.pick_contact_follow_max_tick_m > profile.pick_contact_follow_max_m:
            raise ValueError(
                "pick_contact_follow_max_tick_m must not exceed total follow limit"
            )
        if (
            profile.pick_side_grasp_seat_descent_m
            >= profile.pick_side_grasp_lift_m
        ):
            raise ValueError(
                "pick_side_grasp_seat_descent_m must be below lift distance"
            )
        if (
            profile.pick_side_grasp_seat_start_lift_m
            >= profile.pick_side_grasp_lift_m
        ):
            raise ValueError(
                "pick_side_grasp_seat_start_lift_m must be below lift distance"
            )
        return profile


_DEFAULT_MOTION_PROFILE = _C1MotionProfile()


def _sweep_alignment_blends(profile: _C1MotionProfile) -> tuple[float, ...]:
    """Return deterministic alignment candidates from best to most reachable."""

    count = profile.sweep_plane_alignment_candidate_count
    if count == 1:
        return (float(profile.sweep_plane_alignment_blend),)
    return tuple(
        float(value)
        for value in np.linspace(
            profile.sweep_plane_alignment_blend,
            profile.sweep_plane_alignment_min_blend,
            count,
        )
    )


def _nearest_alignment_variant_index(
    profile: _C1MotionProfile, selected_blend: float
) -> int:
    blends = _sweep_alignment_blends(profile)
    return min(
        range(len(blends)),
        key=lambda index: abs(blends[index] - selected_blend),
    )


def _circular_plate_rim_contact_offset_m(
    rotation: np.ndarray,
    dimensions_m: Sequence[float],
    *,
    vertical_offset_m: float,
    preferred_direction_xy: Sequence[float],
) -> np.ndarray:
    """Find a physical rim point at block height, favoring push direction."""

    plate_rotation = np.asarray(rotation, dtype=float)
    dimensions = np.asarray(dimensions_m, dtype=float)
    preferred_xy = np.asarray(preferred_direction_xy, dtype=float)
    if plate_rotation.shape != (3, 3):
        raise ValueError("plate rotation must be 3x3")
    if dimensions.shape != (3,) or not np.all(dimensions > 0.0):
        raise ValueError("plate dimensions must contain three positive values")
    if preferred_xy.shape != (2,) or np.linalg.norm(preferred_xy) <= 1e-9:
        raise ValueError("preferred rim direction must be a non-zero XY vector")
    preferred_xy /= np.linalg.norm(preferred_xy)
    radius_m = float(max(dimensions[0], dimensions[1]) * 0.5)
    vertical_basis = np.asarray(
        (plate_rotation[2, 0], plate_rotation[2, 1]), dtype=float
    )
    vertical_amplitude = float(np.linalg.norm(vertical_basis))
    if vertical_amplitude <= 1e-9:
        raise RuntimeError("plate plane is horizontal and cannot sweep a block")
    normalized_height = float(
        np.clip(
            vertical_offset_m / (radius_m * vertical_amplitude),
            -1.0,
            1.0,
        )
    )
    phase = float(np.arctan2(vertical_basis[1], vertical_basis[0]))
    angle_delta = float(np.arccos(normalized_height))
    candidates: list[np.ndarray] = []
    for theta in (phase + angle_delta, phase - angle_delta):
        local_offset = np.asarray(
            (radius_m * np.cos(theta), radius_m * np.sin(theta), 0.0),
            dtype=float,
        )
        candidates.append(plate_rotation @ local_offset)
    return max(
        candidates,
        key=lambda offset: float(np.dot(offset[:2], preferred_xy)),
    )


def _plate_sweep_roll_for_vertical_radial_plane(
    *,
    tool_axis_world: np.ndarray,
    attachment_rotation_in_reference: np.ndarray,
    outward_xy: np.ndarray,
    preferred_roll_rad: float,
    axis_blend: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """Choose EEF axis and roll that make the plate plane radial/vertical."""

    preferred_tool_axis = np.asarray(tool_axis_world, dtype=float)
    preferred_tool_axis /= np.linalg.norm(preferred_tool_axis)
    attachment_rotation = np.asarray(
        attachment_rotation_in_reference, dtype=float
    )
    desired_normal = np.asarray(
        (-float(outward_xy[1]), float(outward_xy[0]), 0.0), dtype=float
    )
    desired_normal /= np.linalg.norm(desired_normal)
    attachment_normal = attachment_rotation[:, 2]
    normal_axis_component = float(attachment_normal[2])
    preferred_perpendicular = (
        preferred_tool_axis
        - float(np.dot(preferred_tool_axis, desired_normal)) * desired_normal
    )
    preferred_perpendicular /= np.linalg.norm(preferred_perpendicular)
    fully_aligned_tool_axis = (
        normal_axis_component * desired_normal
        + np.sqrt(max(0.0, 1.0 - normal_axis_component**2))
        * preferred_perpendicular
    )
    tool_axis = (
        (1.0 - axis_blend) * preferred_tool_axis
        + axis_blend * fully_aligned_tool_axis
    )
    tool_axis /= np.linalg.norm(tool_axis)

    best_roll = preferred_roll_rad
    best_alignment = -1.0
    best_preference_distance = float("inf")
    for roll_rad in np.linspace(-np.pi, np.pi, 1441):
        plate_normal = (
            _tool_rotation_from_axis_roll(tool_axis, float(roll_rad))
            @ attachment_rotation
        )[:, 2]
        alignment = abs(float(np.dot(plate_normal, desired_normal)))
        preference_distance = abs(
            float(
                np.arctan2(
                    np.sin(float(roll_rad) - preferred_roll_rad),
                    np.cos(float(roll_rad) - preferred_roll_rad),
                )
            )
        )
        if (
            alignment > best_alignment + 1e-9
            or (
                abs(alignment - best_alignment) <= 1e-9
                and preference_distance < best_preference_distance
            )
        ):
            best_roll = float(roll_rad)
            best_alignment = alignment
            best_preference_distance = preference_distance
    return tool_axis, best_roll, best_alignment


def _broad_face_normal_deviation_for_retry(
    profile: _C1MotionProfile,
    retry_index: int,
) -> float:
    """Schedule bounded broad-face normal relaxations across planning retries."""

    if retry_index < 0:
        raise ValueError("planning retry index must be non-negative")
    levels = (0.0, 0.5, -0.5, 1.0, -1.0)
    return float(
        levels[retry_index % len(levels)]
        * profile.sweep_broad_face_max_normal_deviation_rad
    )


def _circular_plate_vertical_half_extent_m(
    rotation: np.ndarray,
    dimensions_m: Sequence[float],
) -> float:
    """Project a circular plate's live shape onto the world vertical axis."""

    plate_rotation = np.asarray(rotation, dtype=float)
    dimensions = np.asarray(dimensions_m, dtype=float)
    if plate_rotation.shape != (3, 3):
        raise ValueError("plate rotation must be 3x3")
    if dimensions.shape != (3,) or not np.all(dimensions > 0.0):
        raise ValueError("plate dimensions must contain three positive values")
    radius_m = float(max(dimensions[0], dimensions[1]) * 0.5)
    half_thickness_m = float(dimensions[2] * 0.5)
    in_plane_projection = float(np.linalg.norm(plate_rotation[2, :2]))
    normal_projection = abs(float(plate_rotation[2, 2]))
    return (
        radius_m * in_plane_projection
        + half_thickness_m * normal_projection
    )


def _tool_rotation_from_axis_roll(
    tool_axis_world: Sequence[float], roll_rad: float
) -> np.ndarray:
    """Build the EEF rotation represented by a tool axis and axial roll."""

    z_axis = np.asarray(tool_axis_world, dtype=float)
    if z_axis.shape != (3,) or np.linalg.norm(z_axis) <= 1e-9:
        raise ValueError("tool axis must be a non-zero 3-vector")
    z_axis /= np.linalg.norm(z_axis)
    reference = np.asarray((0.0, 0.0, 1.0), dtype=float)
    if abs(float(np.dot(reference, z_axis))) > 0.95:
        reference = np.asarray((1.0, 0.0, 0.0), dtype=float)
    x_axis = reference - float(np.dot(reference, z_axis)) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    base = np.column_stack((x_axis, y_axis, z_axis))
    cosine, sine = np.cos(roll_rad), np.sin(roll_rad)
    return base @ np.asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        )
    )


def _tool_axis_roll_from_rotation(
    rotation: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Recover the schema's tool-axis/roll representation from a rotation."""

    eef_rotation = np.asarray(rotation, dtype=float)
    if eef_rotation.shape != (3, 3):
        raise ValueError("EEF rotation must be 3x3")
    tool_axis_world = eef_rotation[:, 2].copy()
    zero_roll_rotation = _tool_rotation_from_axis_roll(tool_axis_world, 0.0)
    relative_roll = zero_roll_rotation.T @ eef_rotation
    tool_roll_rad = float(
        np.arctan2(relative_roll[1, 0], relative_roll[0, 0])
    )
    return tool_axis_world, tool_roll_rad


def _plate_broad_face_orientation(
    *,
    outward_xy: Sequence[float],
    attachment_rotation_in_reference: np.ndarray,
    variant_index: int,
    variant_count: int,
    normal_deviation_rad: float,
) -> tuple[np.ndarray, np.ndarray, float, float, int, float]:
    """Point a plate broad face along the inward sweep direction.

    The plate local ``+/-z`` surfaces are its two broad faces.  Every returned
    candidate keeps one of those surface normals within the configured
    broad-face cone around the inward radial direction. Variants change which
    equivalent face is used and rotate the circular plate within that plane
    to give IK multiple wrist configurations; none turns the circumference
    toward the block.
    """

    outward = np.asarray(outward_xy, dtype=float)
    if outward.shape != (2,) or np.linalg.norm(outward) <= 1e-9:
        raise ValueError("outward direction must be a non-zero XY vector")
    outward /= np.linalg.norm(outward)
    attachment_rotation = np.asarray(
        attachment_rotation_in_reference, dtype=float
    )
    if attachment_rotation.shape != (3, 3):
        raise ValueError("plate/EEF relative rotation must be 3x3")
    if variant_index < 0:
        raise ValueError("orientation variant index must be non-negative")
    if variant_count <= 0 or variant_index >= variant_count:
        raise ValueError("orientation variant index must be within candidate count")
    if abs(normal_deviation_rad) > np.pi / 4.0:
        raise ValueError("broad-face normal deviation must be within +/- pi/4")

    inward_normal = np.asarray((-outward[0], -outward[1], 0.0), dtype=float)
    face_sign = 1 if variant_index % 2 == 0 else -1
    in_plane_index = variant_index // 2
    in_plane_candidate_count = max(1, (variant_count + 1) // 2)
    tangent = np.asarray(
        (-inward_normal[1], inward_normal[0], 0.0), dtype=float
    )
    broad_face_normal = (
        np.cos(normal_deviation_rad) * inward_normal
        + np.sin(normal_deviation_rad) * tangent
    )
    plate_z_axis = float(face_sign) * broad_face_normal
    plate_x_axis = np.asarray((0.0, 0.0, 1.0), dtype=float)
    plate_y_axis = np.cross(plate_z_axis, plate_x_axis)
    base_plate_rotation = np.column_stack(
        (plate_x_axis, plate_y_axis, plate_z_axis)
    )
    in_plane_roll_rad = float(
        in_plane_index * (2.0 * np.pi / in_plane_candidate_count)
    )
    cosine, sine = np.cos(in_plane_roll_rad), np.sin(in_plane_roll_rad)
    plate_rotation = base_plate_rotation @ np.asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    eef_rotation = plate_rotation @ attachment_rotation.T
    tool_axis_world, tool_roll_rad = _tool_axis_roll_from_rotation(eef_rotation)
    return (
        eef_rotation,
        tool_axis_world,
        tool_roll_rad,
        in_plane_roll_rad,
        face_sign,
        normal_deviation_rad,
    )


def _circular_plate_broad_face_contact_offset_m(
    rotation: np.ndarray,
    dimensions_m: Sequence[float],
    *,
    vertical_offset_m: float,
    face_sign: int,
) -> np.ndarray:
    """Select a block-height point on one of a circular plate's broad faces."""

    plate_rotation = np.asarray(rotation, dtype=float)
    dimensions = np.asarray(dimensions_m, dtype=float)
    if plate_rotation.shape != (3, 3):
        raise ValueError("plate rotation must be 3x3")
    if dimensions.shape != (3,) or not np.all(dimensions > 0.0):
        raise ValueError("plate dimensions must contain three positive values")
    if face_sign not in (-1, 1):
        raise ValueError("face_sign must be -1 or +1")
    plate_normal = plate_rotation[:, 2]
    surface_normal = float(face_sign) * plate_normal
    world_vertical = np.asarray((0.0, 0.0, 1.0), dtype=float)
    in_plane_vertical = world_vertical - float(
        np.dot(world_vertical, plate_normal)
    ) * plate_normal
    vertical_gain = float(np.linalg.norm(in_plane_vertical))
    if vertical_gain <= 1e-6:
        raise RuntimeError("plate broad face is horizontal and cannot sweep a block")
    in_plane_vertical /= vertical_gain
    signed_distance_m = float(vertical_offset_m / vertical_gain)
    radius_m = float(max(dimensions[0], dimensions[1]) * 0.5)
    # A friction-held plate can tilt slightly between micro-plans. If the
    # requested block-center height falls just beyond the projected disk,
    # use the nearest realizable point on the same broad face. The closed-loop
    # height/clearance controller then resolves the remaining millimetres
    # without switching to a circumference-contact strategy.
    signed_distance_m = float(
        np.clip(signed_distance_m, -radius_m, radius_m)
    )
    return (
        surface_normal * float(dimensions[2] * 0.5)
        + in_plane_vertical * signed_distance_m
    )


def _rim_grasp_retention_force_n(
    *,
    mass_kg: float,
    sliding_friction: float,
    dimensions_m: Sequence[float],
    safety_factor: float,
    max_grip_force_n: float,
) -> float:
    """Return clamp force needed for weight and off-centre rim torque.

    A rim grasp carries gravity roughly one plate radius away from the contact
    pair.  The old mass-only calculation ignored that moment and reduced a
    valid light-plate contact from about 19 N to 4.9 N.  Use the declared plate
    thickness as the conservative opposing-contact span and bound the result
    by the M1 gripper force limit.
    """

    dimensions = np.asarray(dimensions_m, dtype=float)
    if dimensions.shape != (3,) or not np.all(dimensions > 0.0):
        raise ValueError("dimensions_m must contain three positive values")
    for name, value in {
        "mass_kg": mass_kg,
        "sliding_friction": sliding_friction,
        "safety_factor": safety_factor,
        "max_grip_force_n": max_grip_force_n,
    }.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    radius_m = float(max(dimensions[0], dimensions[1]) * 0.5)
    contact_span_m = float(dimensions[2])
    gravity_force_n = float(mass_kg * 9.81)
    weight_limited_force_n = gravity_force_n / sliding_friction
    torque_limited_force_n = (
        gravity_force_n * radius_m / (sliding_friction * contact_span_m)
    )
    requested_force_n = safety_factor * max(
        weight_limited_force_n,
        torque_limited_force_n,
    )
    return min(float(max_grip_force_n), requested_force_n)


@dataclass
class _PhysicalGraspMonitor:
    runtime: ToolUseJournalEERuntime
    object_id: str
    initial_position_m: tuple[float, float, float]
    min_lift_m: float
    object_dimensions_m: tuple[float, float, float]
    table_surface_z_m: float
    min_bottom_clearance_m: float
    required_final_hold_s: float
    contact_loss_grace_s: float
    required_contact_ticks: int
    contact_freeze_ticks: int
    closure_actuator_kp: float
    max_closure_actuator_kp: float
    force_feedback_gain: float
    retention_target_normal_force_n: float
    max_grip_force_n: float
    sliding_friction: float
    min_contact_separation_m: float
    min_normal_opposition: float
    max_friction_utilization: float
    contact_follow_gain: float
    contact_follow_max_m: float
    contact_follow_activation_ticks: int
    contact_follow_max_tick_m: float
    contact_follow_max_joint_step_rad: float
    regrasp_roll_rad: float
    regrasp_roll_rate_rad_s: float
    regrasp_min_separation_ratio: float
    samples: list[dict[str, object]] = field(default_factory=list)
    _bilateral_ticks: int = 0
    _closure_frozen_time_s: float | None = None
    _stable_clearance_ticks: int = 0
    _follow_object_to_reference_local_m: np.ndarray | None = None
    _regrasp_peak_contact_separation_m: float = 0.0
    current_regrasp_roll_offset_rad: float = 0.0
    _active_closure_actuator_kp: float = 0.0
    active_segment_id: str | None = None
    active_keyframe_id: str | None = None
    active_target_block_id: str | None = None
    active_physical_tool_control: dict[str, object] | None = None
    active_physical_tool_settle: dict[str, object] | None = None
    active_physical_push_control: dict[str, object] | None = None
    push_contact_acquired: bool = False
    push_reacquire_attempts: int = 0
    push_contact_loss_started_time_s: float | None = None
    push_recovery_exhausted: bool = False
    push_recovery_events: list[dict[str, object]] = field(default_factory=list)
    _push_control_segment_id: str | None = None
    learned_push_contact_local_m: np.ndarray | None = None
    current_control_translation_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )

    _CHECKPOINT_FIELDS = (
        "_bilateral_ticks",
        "_closure_frozen_time_s",
        "_stable_clearance_ticks",
        "_follow_object_to_reference_local_m",
        "_regrasp_peak_contact_separation_m",
        "current_regrasp_roll_offset_rad",
        "_active_closure_actuator_kp",
        "active_segment_id",
        "active_keyframe_id",
        "active_target_block_id",
        "active_physical_tool_control",
        "active_physical_tool_settle",
        "active_physical_push_control",
        "push_contact_acquired",
        "push_reacquire_attempts",
        "push_contact_loss_started_time_s",
        "push_recovery_exhausted",
        "push_recovery_events",
        "_push_control_segment_id",
        "learned_push_contact_local_m",
        "current_control_translation_m",
    )

    def checkpoint(self) -> dict[str, object]:
        """Capture monitor state for a speculative controller replay."""

        return {
            "sample_count": len(self.samples),
            "fields": {
                name: copy.deepcopy(getattr(self, name))
                for name in self._CHECKPOINT_FIELDS
            },
        }

    def restore(self, checkpoint: Mapping[str, object]) -> None:
        """Discard a rejected replay without poisoning later validation."""

        sample_count = int(checkpoint["sample_count"])
        del self.samples[sample_count:]
        state = checkpoint["fields"]
        if not isinstance(state, Mapping):
            raise TypeError("physical grasp checkpoint fields must be a mapping")
        for name in self._CHECKPOINT_FIELDS:
            setattr(self, name, copy.deepcopy(state[name]))
        if self._closure_frozen_time_s is not None:
            self.runtime.set_finger_gripper_force_target(
                total_force_n=self.retention_target_normal_force_n,
                actuator_kp=self._active_closure_actuator_kp,
                max_grip_force_n=self.max_grip_force_n,
            )
            self.runtime.hold_gripper_position()

    def set_active_segment(self, segment: object | None) -> None:
        """Expose controller phase metadata to physical samples and feedback."""

        metadata = getattr(segment, "metadata", {}) if segment is not None else {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        self.active_segment_id = (
            str(getattr(segment, "segment_id"))
            if segment is not None and getattr(segment, "segment_id", None)
            else None
        )
        self.active_keyframe_id = (
            str(metadata["keyframe_id"])
            if metadata.get("keyframe_id") is not None
            else None
        )
        self.active_target_block_id = (
            str(metadata["target_block_id"])
            if metadata.get("target_block_id") is not None
            else None
        )
        raw_control = metadata.get("physical_tool_control")
        self.active_physical_tool_control = (
            dict(raw_control) if isinstance(raw_control, Mapping) else None
        )
        raw_settle = metadata.get("physical_tool_settle")
        self.active_physical_tool_settle = (
            dict(raw_settle) if isinstance(raw_settle, Mapping) else None
        )
        raw_push_control = metadata.get("physical_push_control")
        self.active_physical_push_control = (
            dict(raw_push_control)
            if isinstance(raw_push_control, Mapping)
            else None
        )
        if self._push_control_segment_id != self.active_segment_id:
            self.push_contact_acquired = False
            self.push_reacquire_attempts = 0
            self.push_contact_loss_started_time_s = None
            self.push_recovery_exhausted = False
            self.learned_push_contact_local_m = None
            self._push_control_segment_id = self.active_segment_id

    def _contact_details(self, object_body_id: int) -> list[dict[str, object]]:
        """Return the actual MuJoCo finger/object contact geometry."""

        model = self.runtime.env.sim.model._model
        data = self.runtime.env.sim.data._data

        def belongs_to_object(geom_id: int) -> bool:
            body_id = int(model.geom_bodyid[geom_id])
            while body_id > 0:
                if body_id == object_body_id:
                    return True
                body_id = int(model.body_parentid[body_id])
            return body_id == object_body_id

        details: list[dict[str, object]] = []
        for contact_id in range(int(data.ncon)):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            object_is_geom1 = belongs_to_object(geom1)
            object_is_geom2 = belongs_to_object(geom2)
            if object_is_geom1 == object_is_geom2:
                continue
            finger_geom = geom2 if object_is_geom1 else geom1
            finger_name = (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, finger_geom
                )
                or ""
            )
            if "gripper0_right_" not in finger_name.lower():
                continue
            wrench = np.empty(6, dtype=float)
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            details.append(
                {
                    "finger_geom": finger_name,
                    "object_geom": (
                        mujoco.mj_id2name(
                            model,
                            mujoco.mjtObj.mjOBJ_GEOM,
                            geom1 if object_is_geom1 else geom2,
                        )
                        or ""
                    ),
                    "position_m": [float(value) for value in contact.pos],
                    "frame_normal_xyz": [
                        float(value) for value in contact.frame[:3]
                    ],
                    "object_outward_normal_xyz": [
                        float(value)
                        for value in (
                            np.asarray(contact.frame[:3], dtype=float)
                            * (1.0 if object_is_geom1 else -1.0)
                        )
                    ],
                    "object_is_geom1": object_is_geom1,
                    "distance_m": float(contact.dist),
                    "normal_force_n": abs(float(wrench[0])),
                    "tangential_force_n": float(np.linalg.norm(wrench[1:3])),
                }
            )
        return details

    def _environment_contact_details(
        self, object_body_id: int
    ) -> list[dict[str, object]]:
        """Describe contacts between the plate and the active target block."""

        model = self.runtime.env.sim.model._model
        data = self.runtime.env.sim.data._data

        def descends_from(body_id: int, ancestor_id: int) -> bool:
            while body_id > 0:
                if body_id == ancestor_id:
                    return True
                body_id = int(model.body_parentid[body_id])
            return body_id == ancestor_id

        target_object_id = self.active_target_block_id
        if (
            target_object_id is None
            or target_object_id not in self.runtime.env.obj_body_id
        ):
            return []
        target_body_id = int(
            self.runtime.env.obj_body_id[target_object_id]
        )
        details: list[dict[str, object]] = []
        for contact_id in range(int(data.ncon)):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])
            object_is_geom1 = descends_from(body1, object_body_id)
            object_is_geom2 = descends_from(body2, object_body_id)
            if object_is_geom1 == object_is_geom2:
                continue
            other_geom = geom2 if object_is_geom1 else geom1
            other_body = body2 if object_is_geom1 else body1
            if not descends_from(other_body, target_body_id):
                continue
            other_geom_name = (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
                )
                or ""
            )
            if "gripper0_right_" in other_geom_name.lower():
                continue
            wrench = np.empty(6, dtype=float)
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            details.append(
                {
                    "other_object_id": target_object_id,
                    "other_body": (
                        mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_BODY, other_body
                        )
                        or ""
                    ),
                    "other_geom": other_geom_name,
                    "position_m": [float(value) for value in contact.pos],
                    "distance_m": float(contact.dist),
                    "normal_force_n": abs(float(wrench[0])),
                    "tangential_force_n": float(np.linalg.norm(wrench[1:3])),
                }
            )
        return details

    def sample(self, simulation_time_s: float) -> None:
        if not self.runtime.grasp_engaged:
            return
        data = self.runtime.env.sim.data._data
        body_id = self.runtime.env.obj_body_id[self.object_id]
        position = tuple(float(value) for value in data.xpos[body_id])
        object_rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
        vertical_half_extent_m = _circular_plate_vertical_half_extent_m(
            object_rotation,
            self.object_dimensions_m,
        )
        bottom_clearance_m = (
            position[2] - vertical_half_extent_m - self.table_surface_z_m
        )
        model = self.runtime.env.sim.model._model
        gripper_actuator_forces = {
            (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
                )
                or f"actuator_{actuator_id}"
            ): float(data.actuator_force[actuator_id])
            for actuator_id in range(model.nu)
            if (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
                )
                or ""
            ).startswith("gripper0_right_")
        }
        _, _, reference_position, reference_rotation = self.runtime._grasp_reference(
            self.runtime.env
        )
        contact = self.runtime.object_contact_metrics(self.object_id)
        contact_details = self._contact_details(body_id)
        environment_contacts = self._environment_contact_details(body_id)
        target_environment_contacts = [
            detail
            for detail in environment_contacts
            if detail.get("other_object_id") == self.active_target_block_id
        ]
        if target_environment_contacts:
            self.push_contact_acquired = True
            strongest_target_contact = max(
                target_environment_contacts,
                key=lambda detail: float(detail["normal_force_n"]),
            )
            contact_position = np.asarray(
                strongest_target_contact["position_m"], dtype=float
            )
            self.learned_push_contact_local_m = (
                object_rotation.T
                @ (contact_position - np.asarray(position, dtype=float))
            )
        strongest_target_contact_position_m = (
            list(strongest_target_contact["position_m"])
            if target_environment_contacts
            else None
        )
        target_block_position_m: list[float] | None = None
        if (
            self.active_target_block_id is not None
            and self.active_target_block_id in self.runtime.env.obj_body_id
        ):
            target_body_id = self.runtime.env.obj_body_id[
                self.active_target_block_id
            ]
            target_block_position_m = [
                float(value) for value in data.xpos[target_body_id]
            ]
        stability = self._contact_stability(contact_details)
        sample = {
            "simulation_time_s": float(simulation_time_s),
            "object_position_m": list(position),
            "object_rotation": [
                [float(value) for value in row] for row in object_rotation
            ],
            "object_vertical_half_extent_m": vertical_half_extent_m,
            "bottom_clearance_m": bottom_clearance_m,
            "gripper_actuator_forces_n": gripper_actuator_forces,
            "fingerpad_separation_m": self.runtime.fingerpad_separation_m(),
            "grasp_reference_position_m": [
                float(value) for value in reference_position
            ],
            "grasp_reference_rotation": [
                [float(value) for value in row] for row in reference_rotation
            ],
            "contact_count": contact.contact_count,
            "normal_force_n": contact.normal_force_n,
            "tangential_force_n": contact.tangential_force_n,
            "total_force_n": contact.total_force_n,
            "contact_groups": list(contact.contact_groups),
            "contact_details": contact_details,
            "environment_contact_count": len(environment_contacts),
            "environment_contact_objects": sorted(
                {
                    str(detail["other_object_id"])
                    for detail in environment_contacts
                    if detail.get("other_object_id") is not None
                }
            ),
            "target_block_contact_count": len(target_environment_contacts),
            "target_block_normal_force_n": sum(
                float(detail["normal_force_n"])
                for detail in target_environment_contacts
            ),
            "environment_contact_details": sorted(
                environment_contacts,
                key=lambda detail: float(detail["normal_force_n"]),
                reverse=True,
            )[:8],
            "execution_segment_id": self.active_segment_id,
            "execution_keyframe_id": self.active_keyframe_id,
            "target_block_id": self.active_target_block_id,
            "physical_tool_control": self.active_physical_tool_control,
            "physical_tool_settle": self.active_physical_tool_settle,
            "physical_push_control": self.active_physical_push_control,
            "push_contact_acquired": self.push_contact_acquired,
            "push_reacquire_attempts": self.push_reacquire_attempts,
            "push_recovery_exhausted": self.push_recovery_exhausted,
            "strongest_target_contact_position_m": (
                strongest_target_contact_position_m
            ),
            "learned_push_contact_local_m": (
                [
                    float(value)
                    for value in self.learned_push_contact_local_m
                ]
                if self.learned_push_contact_local_m is not None
                else None
            ),
            "target_block_position_m": target_block_position_m,
            "controller_translation_correction_m": [
                float(value) for value in self.current_control_translation_m
            ],
            **stability,
        }
        self.samples.append(sample)
        current_object_to_reference_local = (
            np.asarray(reference_rotation, dtype=float).T
            @ (
                np.asarray(position, dtype=float)
                - np.asarray(reference_position, dtype=float)
            )
        )
        if self._closure_frozen_time_s is None:
            self._bilateral_ticks = (
                self._bilateral_ticks + 1 if self._bilateral(sample) else 0
            )
            if self._bilateral_ticks >= self.contact_freeze_ticks:
                self.runtime.set_finger_gripper_force_target(
                    total_force_n=self.retention_target_normal_force_n,
                    actuator_kp=self.closure_actuator_kp,
                    max_grip_force_n=self.max_grip_force_n,
                )
                self.runtime.hold_gripper_position()
                self._closure_frozen_time_s = float(simulation_time_s)
                self._active_closure_actuator_kp = self.closure_actuator_kp
        else:
            # The retention target is a *contact normal force*.  Actuator
            # force is not equivalent because the finger linkage has a
            # configuration-dependent mechanical advantage.  Closing the
            # loop on actuator force over-squeezes and ejects a thin rim.
            if self._bilateral(sample) and contact.normal_force_n > 1e-6:
                normalized_error = np.clip(
                    (
                        self.retention_target_normal_force_n
                        - contact.normal_force_n
                    )
                    / self.retention_target_normal_force_n,
                    -1.0,
                    1.0,
                )
                next_kp = float(
                    np.clip(
                        self._active_closure_actuator_kp
                        * (1.0 + self.force_feedback_gain * normalized_error),
                        self.closure_actuator_kp,
                        self.max_closure_actuator_kp,
                    )
                )
                if not np.isclose(
                    next_kp,
                    self._active_closure_actuator_kp,
                    rtol=1e-4,
                    atol=1e-4,
                ):
                    self.runtime.set_finger_gripper_force_target(
                        total_force_n=self.retention_target_normal_force_n,
                        actuator_kp=next_kp,
                        max_grip_force_n=self.max_grip_force_n,
                    )
                    self._active_closure_actuator_kp = next_kp
        if (
            self._stable_bilateral(sample)
            and bottom_clearance_m >= self.min_bottom_clearance_m
        ):
            self._stable_clearance_ticks += 1
            if (
                self._follow_object_to_reference_local_m is None
                and self._stable_clearance_ticks
                >= self.contact_follow_activation_ticks
            ):
                self._follow_object_to_reference_local_m = (
                    current_object_to_reference_local
                )
        else:
            self._stable_clearance_ticks = 0
        sample["gripper_actuator_kp"] = (
            self._active_closure_actuator_kp
            if self._closure_frozen_time_s is not None
            else self.closure_actuator_kp
        )
        sample["gripper_force_target_n"] = (
            self.retention_target_normal_force_n
            if self._closure_frozen_time_s is not None
            else None
        )
        sample["contact_follow_active"] = (
            self._follow_object_to_reference_local_m is not None
        )
        if sample["contact_follow_active"] and self._stable_bilateral(sample):
            self._regrasp_peak_contact_separation_m = max(
                self._regrasp_peak_contact_separation_m,
                float(sample["contact_point_separation_m"]),
            )
        sample["regrasp_roll_offset_rad"] = (
            self.current_regrasp_roll_offset_rad
        )
        sample["contact_follow_baseline_local_m"] = (
            [
                float(value)
                for value in self._follow_object_to_reference_local_m
            ]
            if self._follow_object_to_reference_local_m is not None
            else None
        )

    def contact_follow_translation_xy_m(self) -> np.ndarray | None:
        """Damp tick-to-tick plate motion relative to the grasp frame.

        A frictionally held free body can settle at a new relative pose after a
        deliberate wrist or sweep reorientation.  A position-restoring term
        would keep fighting that valid equilibrium and prevent trajectory
        settling.  The tick delta below is a relative-velocity term: it damps
        pendulum motion, becomes zero at rest, and ignores common-mode arm
        translation.  Expressing the delta in the hand frame also makes rigid
        hand rotation invariant.
        """

        if (
            self._follow_object_to_reference_local_m is None
            or len(self.samples) < 2
        ):
            return None
        latest = self.samples[-1]
        previous = self.samples[-2]
        if (
            not self._bilateral(latest)
            or not self._bilateral(previous)
            or not bool(previous.get("contact_follow_active"))
        ):
            return None

        def local_offset(sample: Mapping[str, object]) -> np.ndarray:
            object_position = np.asarray(
                sample["object_position_m"], dtype=float
            )
            reference_position = np.asarray(
                sample["grasp_reference_position_m"], dtype=float
            )
            reference_rotation = np.asarray(
                sample["grasp_reference_rotation"], dtype=float
            )
            return reference_rotation.T @ (
                object_position - reference_position
            )

        latest_rotation = np.asarray(
            latest["grasp_reference_rotation"], dtype=float
        )
        relative_tick_delta_world = latest_rotation @ (
            local_offset(latest) - local_offset(previous)
        )
        correction = -self.contact_follow_gain * relative_tick_delta_world[:2]
        correction_norm = float(np.linalg.norm(correction))
        if correction_norm > self.contact_follow_max_m:
            correction *= self.contact_follow_max_m / correction_norm
        correction_norm = float(np.linalg.norm(correction))
        if correction_norm > self.contact_follow_max_tick_m:
            correction *= self.contact_follow_max_tick_m / correction_norm
        return correction

    @property
    def contact_stabilization_active(self) -> bool:
        return self._follow_object_to_reference_local_m is not None

    @property
    def regrasp_roll_permitted(self) -> bool:
        """Allow wrist reorientation only while the opposed pinch stays wide.

        The controller learns its separation reference from the live contacts;
        it does not assume a plate radius or a fixed wrist angle.  Once the
        contact span contracts beyond the configured ratio, the wrist target
        is frozen instead of continuing to roll the two fingertips toward the
        same rim point.
        """

        if (
            not self.contact_stabilization_active
            or not self.samples
            or self._regrasp_peak_contact_separation_m <= 0.0
        ):
            return False
        latest = self.samples[-1]
        return self._stable_bilateral(latest) and float(
            latest["contact_point_separation_m"]
        ) >= (
            self.regrasp_min_separation_ratio
            * self._regrasp_peak_contact_separation_m
        )

    @staticmethod
    def _bilateral(sample: Mapping[str, object]) -> bool:
        return (
            int(sample["contact_count"]) >= 2
            and {"left_finger", "right_finger"}.issubset(
                set(sample["contact_groups"])  # type: ignore[arg-type]
            )
        )

    def _contact_stability(
        self, details: Sequence[Mapping[str, object]]
    ) -> dict[str, object]:
        """Measure whether two fingers form a separated, opposed pinch.

        A contact count alone is insufficient for a thin round plate.  The old
        path ended with both fingertips on almost the same rim point.  Use the
        strongest contact on each finger and reject coincident, non-opposed,
        or friction-saturated contact pairs.
        """

        strongest: dict[str, Mapping[str, object]] = {}
        for detail in details:
            name = str(detail.get("finger_geom", "")).lower()
            side = (
                "left"
                if "gripper0_right_left_" in name
                else "right"
                if "gripper0_right_right_" in name
                else None
            )
            if side is None:
                continue
            previous = strongest.get(side)
            if previous is None or float(detail["normal_force_n"]) > float(
                previous["normal_force_n"]
            ):
                strongest[side] = detail
        if set(strongest) != {"left", "right"}:
            return {
                "opposed_contact": False,
                "contact_point_separation_m": 0.0,
                "normal_opposition": -1.0,
                # Keep trace JSON standards-compliant; opposed_contact already
                # marks this sample invalid for stability decisions.
                "friction_utilization": None,
            }
        left, right = strongest["left"], strongest["right"]
        left_position = np.asarray(left["position_m"], dtype=float)
        right_position = np.asarray(right["position_m"], dtype=float)
        left_normal = np.asarray(left["object_outward_normal_xyz"], dtype=float)
        right_normal = np.asarray(
            right["object_outward_normal_xyz"], dtype=float
        )
        separation_m = float(np.linalg.norm(left_position - right_position))
        normal_opposition = -float(np.dot(left_normal, right_normal))
        normal_force_n = float(left["normal_force_n"]) + float(
            right["normal_force_n"]
        )
        tangential_force_n = float(left["tangential_force_n"]) + float(
            right["tangential_force_n"]
        )
        friction_capacity_n = self.sliding_friction * normal_force_n
        friction_utilization = (
            tangential_force_n / friction_capacity_n
            if friction_capacity_n > 1e-9
            else float("inf")
        )
        opposed_contact = (
            separation_m >= self.min_contact_separation_m
            and normal_opposition >= self.min_normal_opposition
            and friction_utilization <= self.max_friction_utilization
        )
        return {
            "opposed_contact": opposed_contact,
            "contact_point_separation_m": separation_m,
            "normal_opposition": normal_opposition,
            "friction_utilization": friction_utilization,
        }

    @staticmethod
    def _stable_bilateral(sample: Mapping[str, object]) -> bool:
        return _PhysicalGraspMonitor._bilateral(sample) and bool(
            sample.get("opposed_contact", False)
        )

    @staticmethod
    def _sample_tool_speed_m_s(
        previous: Mapping[str, object], latest: Mapping[str, object]
    ) -> float:
        delta_time_s = float(latest["simulation_time_s"]) - float(
            previous["simulation_time_s"]
        )
        if delta_time_s <= 0.0:
            return float("inf")
        delta_position = np.asarray(
            latest["object_position_m"], dtype=float
        ) - np.asarray(previous["object_position_m"], dtype=float)
        return float(np.linalg.norm(delta_position) / delta_time_s)

    def live_tool_observation(self) -> dict[str, object] | None:
        """Return the live free-tool pose used by physics-aware settling."""

        if not self.samples:
            return None
        latest = self.samples[-1]
        speed_m_s = (
            self._sample_tool_speed_m_s(self.samples[-2], latest)
            if len(self.samples) >= 2
            else float("inf")
        )
        return {
            "position_m": [float(value) for value in latest["object_position_m"]],
            "bottom_clearance_m": float(latest["bottom_clearance_m"]),
            "linear_speed_m_s": speed_m_s,
            "stable_bilateral_contact": self._stable_bilateral(latest),
            "contact_count": int(latest["contact_count"]),
            "normal_force_n": float(latest["normal_force_n"]),
            "target_block_id": latest.get("target_block_id"),
        }

    def tool_clearance_summary(self) -> dict[str, object]:
        """Evaluate controlled sweep clearance independently of grasp retention."""

        controlled = [
            sample
            for sample in self.samples
            if isinstance(sample.get("physical_tool_control"), Mapping)
        ]
        if not controlled:
            return {
                "status": "NOT_EVALUATED",
                "sample_count": 0,
                "minimum_bottom_clearance_m": None,
                "maximum_table_penetration_m": None,
                "within_clearance_ratio": None,
            }
        operational = [
            sample
            for sample in controlled
            if isinstance(sample.get("physical_push_control"), Mapping)
        ]
        evaluated = operational or controlled
        minimum_clearance_m = min(
            float(sample["bottom_clearance_m"]) for sample in evaluated
        )
        within_count = 0
        for sample in evaluated:
            control = sample["physical_tool_control"]
            assert isinstance(control, Mapping)
            clearance_m = float(sample["bottom_clearance_m"])
            target_m = float(control["target_clearance_m"])
            tolerance_m = float(control["clearance_tolerance_m"])
            max_penetration_m = float(control["max_table_penetration_m"])
            if (
                abs(clearance_m - target_m) <= tolerance_m
                and clearance_m >= -max_penetration_m
            ):
                within_count += 1
        allowed_penetration_m = max(
            float(sample["physical_tool_control"]["max_table_penetration_m"])
            for sample in evaluated
        )
        penetration_m = max(0.0, -minimum_clearance_m)
        return {
            "status": (
                "SUCCESS"
                if penetration_m <= allowed_penetration_m
                else "FAILED"
            ),
            "sample_count": len(evaluated),
            "evaluation_phase": (
                "PHYSICAL_PUSH" if operational else "PHYSICAL_TOOL_CONTROL"
            ),
            "minimum_bottom_clearance_m": minimum_clearance_m,
            "maximum_table_penetration_m": penetration_m,
            "allowed_table_penetration_m": allowed_penetration_m,
            "within_clearance_ratio": within_count / len(evaluated),
            "approach_minimum_bottom_clearance_m": min(
                float(sample["bottom_clearance_m"])
                for sample in controlled
            ),
        }

    def summary(self) -> dict[str, object]:
        formation_index: int | None = None
        consecutive = 0
        for index, sample in enumerate(self.samples):
            consecutive = (
                consecutive + 1 if self._stable_bilateral(sample) else 0
            )
            if consecutive >= self.required_contact_ticks:
                formation_index = index - self.required_contact_ticks + 1
                break

        retained = self.samples[formation_index:] if formation_index is not None else []
        max_zero_contact_s = 0.0
        zero_started_at: float | None = None
        for sample in retained:
            time_s = float(sample["simulation_time_s"])
            if int(sample["contact_count"]) == 0:
                if zero_started_at is None:
                    zero_started_at = time_s
            elif zero_started_at is not None:
                max_zero_contact_s = max(max_zero_contact_s, time_s - zero_started_at)
                zero_started_at = None
        if retained and zero_started_at is not None:
            max_zero_contact_s = max(
                max_zero_contact_s,
                float(retained[-1]["simulation_time_s"]) - zero_started_at,
            )

        final = self.samples[-1] if self.samples else None
        final_position = (
            tuple(float(value) for value in final["object_position_m"])
            if final is not None
            else self.initial_position_m
        )
        final_lift_m = final_position[2] - self.initial_position_m[2]
        max_lift_m = max(
            (
                float(sample["object_position_m"][2])  # type: ignore[index]
                - self.initial_position_m[2]
                for sample in self.samples
            ),
            default=0.0,
        )
        bilateral_ratio = (
            sum(1 for sample in retained if self._stable_bilateral(sample))
            / len(retained)
            if retained
            else 0.0
        )
        final_bilateral = final is not None and self._stable_bilateral(final)
        formation_succeeded = formation_index is not None
        retention_succeeded = (
            formation_succeeded
            and max_zero_contact_s <= self.contact_loss_grace_s
            and final_bilateral
        )
        final_bottom_clearance_m = (
            float(final["bottom_clearance_m"])
            if final is not None
            else 0.0
        )
        max_bottom_clearance_m = max(
            (float(sample["bottom_clearance_m"]) for sample in self.samples),
            default=0.0,
        )
        continuous_clearance_started_at_s: float | None = None
        for sample in retained:
            if (
                float(sample["bottom_clearance_m"])
                >= self.min_bottom_clearance_m
                and self._stable_bilateral(sample)
            ):
                if continuous_clearance_started_at_s is None:
                    continuous_clearance_started_at_s = float(
                        sample["simulation_time_s"]
                    )
            else:
                continuous_clearance_started_at_s = None
        final_clearance_hold_s = (
            float(retained[-1]["simulation_time_s"])
            - continuous_clearance_started_at_s
            if retained and continuous_clearance_started_at_s is not None
            else 0.0
        )
        bottom_clearance_succeeded = (
            final_bottom_clearance_m >= self.min_bottom_clearance_m
        )
        final_hold_succeeded = (
            final_clearance_hold_s + 1e-9 >= self.required_final_hold_s
        )
        lift_succeeded = (
            final_lift_m >= self.min_lift_m
            and bottom_clearance_succeeded
            and final_hold_succeeded
        )
        weld_absent = self.runtime.attachment is None
        contact_hold_succeeded = self._closure_frozen_time_s is not None
        grasp_retention = {
            "status": (
                "SUCCESS"
                if formation_succeeded
                and retention_succeeded
                and weld_absent
                and contact_hold_succeeded
                else "FAILED"
            ),
            "formation_succeeded": formation_succeeded,
            "retention_succeeded": retention_succeeded,
            "contact_triggered_hold_succeeded": contact_hold_succeeded,
            "bilateral_contact_ratio_after_formation": bilateral_ratio,
            "max_zero_contact_duration_s": max_zero_contact_s,
            "allowed_zero_contact_duration_s": self.contact_loss_grace_s,
            "final_bilateral_contact": final_bilateral,
            "weld_absent": weld_absent,
        }
        pick_lift_validation = {
            "status": "SUCCESS" if lift_succeeded else "FAILED",
            "lift_succeeded": lift_succeeded,
            "bottom_clearance_succeeded": bottom_clearance_succeeded,
            "final_hold_succeeded": final_hold_succeeded,
            "required_lift_m": self.min_lift_m,
            "final_lift_m": final_lift_m,
            "required_bottom_clearance_m": self.min_bottom_clearance_m,
            "final_bottom_clearance_m": final_bottom_clearance_m,
            "required_final_hold_s": self.required_final_hold_s,
            "final_clearance_hold_s": final_clearance_hold_s,
        }
        return {
            "status": (
                "SUCCESS"
                if formation_succeeded
                and retention_succeeded
                and lift_succeeded
                and weld_absent
                and contact_hold_succeeded
                else "FAILED"
            ),
            "grasp_mode": "CONTACT_FRICTION",
            "weld_absent": weld_absent,
            "grasp_retention": grasp_retention,
            "pick_lift_validation": pick_lift_validation,
            "tool_clearance": self.tool_clearance_summary(),
            "sample_count": len(self.samples),
            "formation_succeeded": formation_succeeded,
            "formation_time_s": (
                float(self.samples[formation_index]["simulation_time_s"])
                if formation_index is not None
                else None
            ),
            "retention_succeeded": retention_succeeded,
            "contact_triggered_hold_succeeded": contact_hold_succeeded,
            "contact_triggered_hold_time_s": self._closure_frozen_time_s,
            "bilateral_contact_ratio_after_formation": bilateral_ratio,
            "max_zero_contact_duration_s": max_zero_contact_s,
            "allowed_zero_contact_duration_s": self.contact_loss_grace_s,
            "lift_succeeded": lift_succeeded,
            "required_lift_m": self.min_lift_m,
            "final_lift_m": final_lift_m,
            "max_lift_m": max_lift_m,
            "bottom_clearance_succeeded": bottom_clearance_succeeded,
            "required_bottom_clearance_m": self.min_bottom_clearance_m,
            "final_bottom_clearance_m": final_bottom_clearance_m,
            "max_bottom_clearance_m": max_bottom_clearance_m,
            "final_hold_succeeded": final_hold_succeeded,
            "required_final_hold_s": self.required_final_hold_s,
            "final_clearance_hold_s": final_clearance_hold_s,
            "table_surface_z_m": self.table_surface_z_m,
            "initial_object_position_m": list(self.initial_position_m),
            "final_object_position_m": list(final_position),
            "final_contact": (
                {
                    "contact_count": final["contact_count"],
                    "normal_force_n": final["normal_force_n"],
                    "tangential_force_n": final["tangential_force_n"],
                    "total_force_n": final["total_force_n"],
                    "contact_groups": final["contact_groups"],
                    "opposed_contact": final["opposed_contact"],
                    "contact_point_separation_m": final[
                        "contact_point_separation_m"
                    ],
                    "normal_opposition": final["normal_opposition"],
                    "friction_utilization": final["friction_utilization"],
                }
                if final is not None
                else None
            ),
        }


class _PhysicalGraspControllerTrajectoryPlayer(
    ToolUseJournalControllerTrajectoryPlayer
):
    def __init__(self, *args: object, monitor: _PhysicalGraspMonitor, **kwargs: object):
        super().__init__(*args, **kwargs)
        self._physical_grasp_monitor = monitor
        # A new player is created for each MotionPlan.  The next plan is built
        # from the live joint state, so it already contains any wrist regrasp
        # accumulated by the preceding plan.  Preserve that controller state
        # and add only changes relative to this plan's starting baseline.
        self._regrasp_roll_offset_rad = float(
            monitor.current_regrasp_roll_offset_rad
        )
        self._plan_regrasp_roll_baseline_rad = self._regrasp_roll_offset_rad
        self._clearance_offset_m = 0.0
        self._tool_xy_offset_m = np.zeros(2, dtype=float)
        self._push_contact_height_offset_m = 0.0
        self._push_control_segment_id: str | None = None

    def _advance_controller(self, action: np.ndarray) -> float:
        corrected_action = self._contact_follow_action(action)
        simulation_time_s = super()._advance_controller(corrected_action)
        self._physical_grasp_monitor.sample(simulation_time_s)
        return simulation_time_s

    def _plan_time_step_s(
        self,
        *,
        segment: TrajectorySegment,
        control_timestep_s: float,
    ) -> float:
        self._physical_grasp_monitor.set_active_segment(segment)
        push_control = (
            self._physical_grasp_monitor.active_physical_push_control
        )
        if push_control is None or not self._physical_grasp_monitor.samples:
            return control_timestep_s
        latest = self._physical_grasp_monitor.samples[-1]
        if int(latest.get("target_block_contact_count", 0)) > 0:
            if (
                self._physical_grasp_monitor.push_contact_loss_started_time_s
                is not None
                and self._physical_grasp_monitor.push_recovery_events
            ):
                now_s = float(latest["simulation_time_s"])
                event = self._physical_grasp_monitor.push_recovery_events[-1]
                event.update(
                    {
                        "ended_at_simulation_time_s": now_s,
                        "loss_duration_s": now_s
                        - float(
                            self._physical_grasp_monitor
                            .push_contact_loss_started_time_s
                        ),
                        "status": "REACQUIRED",
                    }
                )
            self._physical_grasp_monitor.push_contact_loss_started_time_s = None
            return control_timestep_s * float(
                push_control["contact_plan_time_scale"]
            )
        if (
            not self._physical_grasp_monitor.push_contact_acquired
            or self._physical_grasp_monitor.push_recovery_exhausted
        ):
            return control_timestep_s
        now_s = float(latest["simulation_time_s"])
        if self._physical_grasp_monitor.push_contact_loss_started_time_s is None:
            self._physical_grasp_monitor.push_contact_loss_started_time_s = now_s
            self._physical_grasp_monitor.push_reacquire_attempts += 1
            self._physical_grasp_monitor.push_recovery_events.append(
                {
                    "segment_id": self._physical_grasp_monitor.active_segment_id,
                    "started_at_simulation_time_s": now_s,
                    "attempt": (
                        self._physical_grasp_monitor.push_reacquire_attempts
                    ),
                }
            )
        loss_duration_s = now_s - float(
            self._physical_grasp_monitor.push_contact_loss_started_time_s
        )
        attempts_exhausted = (
            self._physical_grasp_monitor.push_reacquire_attempts
            > int(push_control["max_reacquire_attempts"])
        )
        timeout = loss_duration_s >= float(push_control["reacquire_timeout_s"])
        if not attempts_exhausted and not timeout:
            return 0.0
        self._physical_grasp_monitor.push_recovery_exhausted = True
        self._physical_grasp_monitor.push_recovery_events[-1].update(
            {
                "ended_at_simulation_time_s": now_s,
                "loss_duration_s": loss_duration_s,
                "status": "EXHAUSTED",
            }
        )
        # Finish the bounded micro-push. The outer observation/replan loop can
        # then approach again with the next reachability-ranked orientation.
        return control_timestep_s

    def _contact_follow_action(self, action: np.ndarray) -> np.ndarray:
        self._physical_grasp_monitor.set_active_segment(self._active_segment)
        active_segment_id = self._physical_grasp_monitor.active_segment_id
        if active_segment_id != self._push_control_segment_id:
            self._push_contact_height_offset_m = 0.0
            self._push_control_segment_id = active_segment_id
        translation_xy = (
            self._physical_grasp_monitor.contact_follow_translation_xy_m()
        )
        translation = np.zeros(3, dtype=float)
        if translation_xy is not None:
            translation[:2] = translation_xy
        control = self._physical_grasp_monitor.active_physical_tool_control
        if control is not None and self._physical_grasp_monitor.samples:
            latest = self._physical_grasp_monitor.samples[-1]
            settle = self._physical_grasp_monitor.active_physical_tool_settle
            if settle is not None:
                current_xy = np.asarray(
                    latest["object_position_m"], dtype=float
                )[:2]
                target_xy = np.asarray(
                    settle["target_position_m"], dtype=float
                )[:2]
                timestep_s = float(self.runtime.env.control_timestep)
                xy_step = (
                    float(control["gain"])
                    * (target_xy - current_xy)
                    * timestep_s
                )
                max_xy_step_m = float(control["rate_m_s"]) * timestep_s
                xy_step_norm = float(np.linalg.norm(xy_step))
                if xy_step_norm > max_xy_step_m:
                    xy_step *= max_xy_step_m / xy_step_norm
                self._tool_xy_offset_m += xy_step
                max_xy_offset_m = float(control["max_offset_m"])
                xy_offset_norm = float(np.linalg.norm(self._tool_xy_offset_m))
                if xy_offset_norm > max_xy_offset_m:
                    self._tool_xy_offset_m *= (
                        max_xy_offset_m / xy_offset_norm
                    )
            clearance_m = float(latest["bottom_clearance_m"])
            target_m = float(control["target_clearance_m"])
            activation_band_m = float(control["activation_band_m"])
            if (
                clearance_m <= target_m + activation_band_m
                or self._clearance_offset_m > 0.0
            ):
                timestep_s = float(self.runtime.env.control_timestep)
                requested_step_m = (
                    float(control["gain"])
                    * (target_m - clearance_m)
                    * timestep_s
                )
                max_step_m = float(control["rate_m_s"]) * timestep_s
                self._clearance_offset_m = float(
                    np.clip(
                        self._clearance_offset_m
                        + np.clip(
                            requested_step_m,
                            -max_step_m,
                            max_step_m,
                        ),
                        0.0,
                        float(control["max_offset_m"]),
                    )
                )
        translation[:2] += self._tool_xy_offset_m
        push_control = (
            self._physical_grasp_monitor.active_physical_push_control
        )
        if (
            push_control is not None
            and self._physical_grasp_monitor.push_contact_acquired
            and self._physical_grasp_monitor.samples
        ):
            latest = self._physical_grasp_monitor.samples[-1]
            target_block_position = latest.get("target_block_position_m")
            if target_block_position is not None:
                plate_position = np.asarray(
                    latest["object_position_m"], dtype=float
                )
                plate_rotation = np.asarray(
                    latest["object_rotation"], dtype=float
                )
                learned_contact_offset = getattr(
                    self._physical_grasp_monitor,
                    "learned_push_contact_local_m",
                    None,
                )
                contact_offset_local = np.asarray(
                    (
                        learned_contact_offset
                        if learned_contact_offset is not None
                        else push_control.get(
                            "rim_contact_offset_local_m",
                            push_control.get("contact_offset_local_m"),
                        )
                    ),
                    dtype=float,
                )
                actual_contact_xy = (
                    plate_position + plate_rotation @ contact_offset_local
                )[:2]
                push_axis_xy = np.asarray(
                    push_control["push_axis_world"], dtype=float
                )[:2]
                push_axis_xy /= np.linalg.norm(push_axis_xy)
                desired_contact_xy = (
                    np.asarray(target_block_position, dtype=float)[:2]
                    + push_axis_xy
                    * (
                        float(push_control["block_support_m"])
                        - float(push_control["contact_penetration_m"])
                    )
                )
                push_correction = desired_contact_xy - actual_contact_xy
                correction_norm = float(np.linalg.norm(push_correction))
                max_correction_m = float(
                    push_control["max_correction_m"]
                )
                if correction_norm > max_correction_m:
                    push_correction *= max_correction_m / correction_norm
                translation[:2] += push_correction
                actual_contact_z = float(
                    (plate_position + plate_rotation @ contact_offset_local)[2]
                )
                desired_contact_z = float(
                    push_control["contact_height_target_m"]
                )
                timestep_s = float(self.runtime.env.control_timestep)
                height_step_m = np.clip(
                    float(push_control["contact_height_gain"])
                    * (desired_contact_z - actual_contact_z)
                    * timestep_s,
                    -float(push_control["contact_height_rate_m_s"])
                    * timestep_s,
                    float(push_control["contact_height_rate_m_s"])
                    * timestep_s,
                )
                support_floor_m = float(
                    push_control["block_support_center_z_m"]
                ) - float(push_control["block_support_tolerance_m"])
                support_deficit_m = max(
                    0.0,
                    support_floor_m - float(target_block_position[2]),
                )
                if support_deficit_m > 0.0:
                    height_step_m = max(
                        float(height_step_m),
                        min(
                            float(push_control["contact_height_rate_m_s"])
                            * timestep_s,
                            float(push_control["contact_height_gain"])
                            * support_deficit_m
                            * timestep_s,
                        ),
                    )
                self._push_contact_height_offset_m = float(
                    np.clip(
                        self._push_contact_height_offset_m + height_step_m,
                        -float(
                            push_control[
                                "contact_height_max_downward_offset_m"
                            ]
                        ),
                        float(push_control["contact_height_max_offset_m"]),
                    )
                )
        combined_height_offset_m = (
            self._clearance_offset_m + self._push_contact_height_offset_m
        )
        if control is not None and self._physical_grasp_monitor.samples:
            latest_clearance_m = float(
                self._physical_grasp_monitor.samples[-1]["bottom_clearance_m"]
            )
            available_downward_m = max(
                0.0,
                latest_clearance_m + float(control["max_table_penetration_m"]),
            )
            combined_height_offset_m = max(
                combined_height_offset_m,
                self._clearance_offset_m - available_downward_m,
            )
        translation[2] = combined_height_offset_m
        self._physical_grasp_monitor.current_control_translation_m = (
            translation.copy()
        )
        stabilization_active = (
            self._physical_grasp_monitor.contact_stabilization_active
        )
        if not np.any(np.abs(translation) > 1e-12) and not stabilization_active:
            return action
        env = self.runtime.env
        model = env.sim.model._model
        data = env.sim.data._data
        robot = env.robots[0]
        joint_names = tuple(str(name) for name in robot.robot_model.joints)
        split_indexes = robot.composite_controller._action_split_indexes
        arm_start, arm_end = split_indexes["right"]
        corrected = np.asarray(action, dtype=float).copy()
        if np.any(np.abs(translation) > 1e-12):
            reference_kind, reference_name, _, _ = self.runtime._grasp_reference(
                env
            )
            jacobian_position = np.zeros((3, model.nv), dtype=float)
            jacobian_rotation = np.zeros((3, model.nv), dtype=float)
            if reference_kind == "site":
                reference_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_SITE, reference_name
                )
                mujoco.mj_jacSite(
                    model,
                    data,
                    jacobian_position,
                    jacobian_rotation,
                    reference_id,
                )
            elif reference_kind == "body":
                reference_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, reference_name
                )
                mujoco.mj_jacBody(
                    model,
                    data,
                    jacobian_position,
                    jacobian_rotation,
                    reference_id,
                )
            else:
                return action
            dof_ids = []
            for joint_name in joint_names:
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                dof_ids.append(int(model.jnt_dofadr[joint_id]))
            jacobian_xyz = jacobian_position[:, dof_ids]
            damping = 1e-4
            joint_delta = jacobian_xyz.T @ np.linalg.solve(
                jacobian_xyz @ jacobian_xyz.T + damping * np.eye(3),
                translation,
            )
            max_step = (
                float(control["max_joint_offset_rad"])
                if control is not None
                else self._physical_grasp_monitor.contact_follow_max_joint_step_rad
            )
            joint_delta = np.clip(joint_delta, -max_step, max_step)
            corrected[arm_start:arm_end] += joint_delta
            for action_index, joint_name in enumerate(joint_names):
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                if bool(model.jnt_limited[joint_id]):
                    lower, upper = model.jnt_range[joint_id]
                    corrected[arm_start + action_index] = np.clip(
                        corrected[arm_start + action_index],
                        float(lower),
                        float(upper),
                    )

        if stabilization_active:
            roll_target = self._physical_grasp_monitor.regrasp_roll_rad
            max_roll_step = (
                self._physical_grasp_monitor.regrasp_roll_rate_rad_s
                * float(env.control_timestep)
            )
            if self._physical_grasp_monitor.regrasp_roll_permitted:
                self._regrasp_roll_offset_rad += float(
                    np.clip(
                        roll_target - self._regrasp_roll_offset_rad,
                        -max_roll_step,
                        max_roll_step,
                    )
                )
            self._physical_grasp_monitor.current_regrasp_roll_offset_rad = (
                self._regrasp_roll_offset_rad
            )
            wrist_index = next(
                (
                    index
                    for index, name in enumerate(joint_names)
                    if name.endswith("wrist_3_joint")
                ),
                None,
            )
            if wrist_index is None:
                raise RuntimeError("UR5e wrist-3 joint is absent")
            wrist_joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_names[wrist_index],
            )
            target = (
                corrected[arm_start + wrist_index]
                + self._regrasp_roll_offset_rad
                - self._plan_regrasp_roll_baseline_rad
            )
            if bool(model.jnt_limited[wrist_joint_id]):
                target = float(
                    np.clip(target, *model.jnt_range[wrist_joint_id])
                )
            corrected[arm_start + wrist_index] = target
        return corrected

    def _custom_settle_evaluation(
        self,
        *,
        segment: object,
        settle_config: Mapping[str, float | int],
        joint_error_rad: float,
        eef_position_error_m: float | None,
    ) -> Mapping[str, object] | None:
        metadata = getattr(segment, "metadata", {})
        raw = (
            metadata.get("physical_tool_settle")
            if isinstance(metadata, Mapping)
            else None
        )
        if not isinstance(raw, Mapping):
            return None
        observation = self._physical_grasp_monitor.live_tool_observation()
        if observation is None:
            return {
                "succeeded": False,
                "reason": "physical tool has no live observation",
            }
        actual_position = np.asarray(observation["position_m"], dtype=float)
        target_position = np.asarray(raw["target_position_m"], dtype=float)
        xy_error_m = float(
            np.linalg.norm(actual_position[:2] - target_position[:2])
        )
        bottom_clearance_m = float(observation["bottom_clearance_m"])
        target_clearance_m = float(raw["target_clearance_m"])
        clearance_error_m = abs(bottom_clearance_m - target_clearance_m)
        max_penetration_m = float(raw["max_table_penetration_m"])
        joint_ok = (
            "joint_tolerance_rad" not in settle_config
            or joint_error_rad <= float(settle_config["joint_tolerance_rad"])
        )
        xy_ok = xy_error_m <= float(raw["xy_tolerance_m"])
        clearance_ok = (
            clearance_error_m <= float(raw["clearance_tolerance_m"])
            and bottom_clearance_m >= -max_penetration_m
        )
        speed_ok = float(observation["linear_speed_m_s"]) <= float(
            raw["max_tool_speed_m_s"]
        )
        contact_ok = bool(observation["stable_bilateral_contact"])
        return {
            # Physics-aware feedback intentionally offsets nominal joints to
            # place the free plate correctly. Nominal joint error remains a
            # diagnostic; the live tool pose is the convergence criterion.
            "succeeded": (xy_ok and clearance_ok and speed_ok and contact_ok),
            "joint_ok": joint_ok,
            "tool_xy_ok": xy_ok,
            "clearance_ok": clearance_ok,
            "speed_ok": speed_ok,
            "stable_bilateral_contact": contact_ok,
            "tool_xy_error_m": xy_error_m,
            "tool_xy_tolerance_m": float(raw["xy_tolerance_m"]),
            "bottom_clearance_m": bottom_clearance_m,
            "target_clearance_m": target_clearance_m,
            "clearance_error_m": clearance_error_m,
            "clearance_tolerance_m": float(raw["clearance_tolerance_m"]),
            "maximum_table_penetration_m": max_penetration_m,
            "tool_linear_speed_m_s": float(observation["linear_speed_m_s"]),
            "maximum_tool_speed_m_s": float(raw["max_tool_speed_m_s"]),
            "controller_clearance_offset_m": self._clearance_offset_m,
            "eef_position_error_m": eef_position_error_m,
        }


def _load_motion_profile(path: Path | None) -> _C1MotionProfile:
    if path is None:
        return _DEFAULT_MOTION_PROFILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("motion profile must be a JSON object")
    return _C1MotionProfile.from_mapping(payload)


@dataclass(frozen=True)
class _TaskBinding:
    subgoal_id: str
    action_type: str | None
    mode: str | None
    ee: str
    tool: str
    target_ids: tuple[str, ...] = ()
    goal_region_id: str | None = None
    grasp: GraspSpec | None = None


@dataclass(frozen=True)
class _SweepTaskBinding:
    subgoal_id: str
    action_type: str | None
    mode: str | None
    ee: str
    tool: str
    target_ids: tuple[str, ...]
    goal_region_id: str


def _condition_payload(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        rendered = model_dump(mode="python")
        if isinstance(rendered, Mapping):
            return rendered
    raise RuntimeError("Task Planner condition is not a structured mapping")


def _task_planner_ids(value: object, *, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        raw_ids: Sequence[object] = [part.strip() for part in text.split(",")]
    elif isinstance(value, Sequence):
        raw_ids = value
    else:
        raise RuntimeError(f"Task Planner {label} must be a list or set string")
    ids = tuple(str(value).strip() for value in raw_ids if str(value).strip())
    if not ids:
        raise RuntimeError(f"Task Planner {label} is empty")
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"Task Planner {label} contains duplicate ids")
    return ids


def _selected_sweep_binding(selected: object) -> _SweepTaskBinding:
    """Read the M4-selected sweep targets without assuming their count."""

    assignments = list(getattr(selected, "candidate_assignments", ()))
    assignment_by_candidate = {
        assignment.candidate_id: assignment for assignment in assignments
    }
    assignment_by_subgoal = {
        assignment.subgoal_id: assignment for assignment in assignments
    }
    selected_steps = []
    for step in getattr(selected, "steps", ()):
        if getattr(step, "kind", None) != "subgoal":
            continue
        conditions = [
            _condition_payload(condition)
            for condition in (
                *getattr(step, "preconditions", ()),
                *getattr(step, "postconditions", ()),
            )
        ]
        assignment = assignment_by_candidate.get(getattr(step, "candidate_id", None))
        if assignment is None:
            assignment = assignment_by_subgoal.get(getattr(step, "subgoal_id", None))
        structured_sweep = assignment is not None and (
            str(getattr(assignment, "mode", "")).lower() == "sweep"
            or str(getattr(assignment, "action_type", "")).lower()
            in {"sweep", "tool_act:sweep"}
            or (
                str(getattr(assignment, "action_type", "")).lower()
                == "tool_act"
                and bool(getattr(assignment, "target_ids", ()))
                and bool(getattr(assignment, "goal_region_id", None))
            )
        )
        if structured_sweep or any(
            condition.get("type") == "tool_sweepable" for condition in conditions
        ):
            selected_steps.append((step, conditions))
    if not selected_steps:
        raise RuntimeError(
            "Task Planner selected plan must contain at least one sweep subgoal"
        )

    bindings: list[_SweepTaskBinding] = []
    for step, conditions in selected_steps:
        assignment = assignment_by_candidate.get(
            getattr(step, "candidate_id", None)
        )
        if assignment is None:
            assignment = assignment_by_subgoal.get(
                getattr(step, "subgoal_id", None)
            )
        if assignment is None:
            raise RuntimeError("selected sweep step has no candidate assignment")

        target_ids = tuple(getattr(assignment, "target_ids", ()))
        goal_region_id = getattr(assignment, "goal_region_id", None)
        for condition in conditions:
            args = condition.get("args", ())
            if not isinstance(args, Sequence) or isinstance(args, str):
                continue
            if condition.get("type") == "tool_sweepable" and not target_ids:
                if len(args) >= 2:
                    target_ids = _task_planner_ids(
                        args[1], label="sweep targets"
                    )
            if condition.get("type") == "in" and len(args) >= 2:
                if not target_ids:
                    target_ids = _task_planner_ids(
                        args[0], label="sweep targets"
                    )
                if not goal_region_id:
                    goal_region_id = str(args[1])

        target_ids = _task_planner_ids(target_ids, label="sweep targets")
        if not goal_region_id:
            raise RuntimeError("Task Planner sweep subgoal has no goal region")
        ee = str(getattr(assignment, "ee", "")).strip()
        tool = str(getattr(assignment, "tool", "")).strip()
        subgoal_id = str(getattr(step, "subgoal_id", "")).strip()
        if not ee or not tool or not subgoal_id:
            raise RuntimeError(
                "Task Planner sweep binding lacks subgoal, EE, or tool"
            )
        bindings.append(
            _SweepTaskBinding(
                subgoal_id=subgoal_id,
                action_type=getattr(assignment, "action_type", None),
                mode=getattr(assignment, "mode", None),
                ee=ee,
                tool=tool,
                target_ids=target_ids,
                goal_region_id=str(goal_region_id),
            )
        )

    first = bindings[0]
    if any(
        (item.ee, item.tool, item.goal_region_id)
        != (first.ee, first.tool, first.goal_region_id)
        for item in bindings[1:]
    ):
        raise RuntimeError(
            "split sweep assignments must use the same EE, tool, and goal region"
        )
    target_ids = _task_planner_ids(
        tuple(target for item in bindings for target in item.target_ids),
        label="combined sweep targets",
    )
    split_parts = [item.subgoal_id.split("_") for item in bindings]
    collapsible_split_ids = bool(split_parts) and all(
        len(parts) >= 3
        and parts[-2].startswith("s")
        and parts[-2][1:].isdigit()
        and parts[-1].startswith("d")
        and parts[-1][1:].isdigit()
        for parts in split_parts
    )
    if collapsible_split_ids:
        common_prefixes = {"_".join(parts[:-2]) for parts in split_parts}
        common_suffixes = {parts[-1] for parts in split_parts}
        collapsible_split_ids = (
            len(common_prefixes) == 1 and len(common_suffixes) == 1
        )
    if collapsible_split_ids:
        motion_subgoal_id = (
            f"{next(iter(common_prefixes))}_{next(iter(common_suffixes))}"
        )
    else:
        motion_subgoal_id = "+".join(item.subgoal_id for item in bindings)
    inferred_mode = first.mode
    if (
        not inferred_mode
        and str(first.action_type or "").lower() == "tool_act"
        and target_ids
        and first.goal_region_id
    ):
        inferred_mode = "sweep"
    return _SweepTaskBinding(
        subgoal_id=motion_subgoal_id,
        action_type=first.action_type,
        mode=inferred_mode,
        ee=first.ee,
        tool=first.tool,
        target_ids=target_ids,
        goal_region_id=first.goal_region_id,
    )


def _selected_pick_binding(selected: object) -> _TaskBinding:
    """Resolve the M4 assignment responsible for acquiring the selected tool."""

    assignments = list(getattr(selected, "candidate_assignments", ()))
    assignment_by_candidate = {
        assignment.candidate_id: assignment for assignment in assignments
    }
    assignment_by_subgoal = {
        assignment.subgoal_id: assignment for assignment in assignments
    }
    candidates = []
    for step in getattr(selected, "steps", ()):
        if getattr(step, "action", None) != "PICK_TOOL":
            continue
        assignment = assignment_by_candidate.get(getattr(step, "candidate_id", None))
        if assignment is None:
            assignment = assignment_by_subgoal.get(getattr(step, "subgoal_id", None))
        if assignment is not None and assignment not in candidates:
            candidates.append(assignment)
    if not candidates:
        candidates = [
            assignment
            for assignment in assignments
            if str(getattr(assignment, "action_type", "")).lower()
            in {"acquire", "pick", "pick_tool"}
            and getattr(assignment, "tool", None)
        ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Task Planner selected plan must identify exactly one tool-acquire "
            f"assignment; found {len(candidates)}"
        )
    assignment = candidates[0]
    ee = str(getattr(assignment, "ee", "")).strip()
    tool = str(getattr(assignment, "tool", "")).strip()
    if not ee or not tool:
        raise RuntimeError("selected tool-acquire assignment has no EE or tool")
    return _TaskBinding(
        subgoal_id=str(assignment.subgoal_id),
        action_type=getattr(assignment, "action_type", None),
        mode=getattr(assignment, "mode", None),
        ee=ee,
        tool=tool,
        target_ids=(tool,),
        grasp=getattr(assignment, "grasp", None),
    )


def _provenance(
    artifact_id: str,
    artifact_type: str,
    module: ModuleName,
    *inputs: str,
) -> ArtifactProvenance:
    return ArtifactProvenance(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        produced_by=module,
        invocation_id=f"{artifact_id}:invocation",
        input_artifact_ids=list(inputs),
    )


def _write_model(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = value.model_dump_json(indent=2)  # type: ignore[attr-defined]
    path.write_text(rendered + "\n", encoding="utf-8")


def _rotation_from_xyzw(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = quaternion
    result = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(result, np.asarray((w, x, y, z), dtype=float))
    return result.reshape(3, 3)


def _xyzw_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    result = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(result, np.ascontiguousarray(rotation.reshape(9)))
    return (float(result[1]), float(result[2]), float(result[3]), float(result[0]))


def _world_pose(record: dict) -> tuple[np.ndarray, np.ndarray]:
    raw = record["pose"]
    return (
        np.asarray(raw["position_m"], dtype=float),
        _rotation_from_xyzw(tuple(raw["orientation_xyzw"])),
    )


def _target_site_pose(
    env: object,
    runtime: ToolUseJournalEERuntime,
    target_hand_pose: Pose,
) -> tuple[str, str, np.ndarray, np.ndarray]:
    model = env.sim.model._model  # type: ignore[attr-defined]
    data = env.sim.data._data  # type: ignore[attr-defined]
    adapter = ToolUseJournalEnvironmentAdapter(env)
    hand_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, adapter.hand_body
    )
    reference_kind, reference_name, site_position, site_rotation = (
        runtime._grasp_reference(env)
    )
    if reference_kind != "site":
        raise RuntimeError("C1_1 runtime did not expose a grasp site")
    hand_position = np.asarray(data.xpos[hand_id], dtype=float)
    hand_rotation = np.asarray(data.xmat[hand_id], dtype=float).reshape(3, 3)
    site_in_hand = hand_rotation.T @ (site_position - hand_position)
    site_rotation_in_hand = hand_rotation.T @ site_rotation
    target_hand_position = np.asarray(target_hand_pose.position_m, dtype=float)
    target_hand_rotation = _rotation_from_xyzw(
        tuple(target_hand_pose.orientation_xyzw)
    )
    target_site_position = (
        target_hand_position + target_hand_rotation @ site_in_hand
    )
    target_site_rotation = target_hand_rotation @ site_rotation_in_hand
    return (
        reference_kind,
        reference_name,
        target_site_position,
        target_site_rotation,
    )


def _attachment_at_keyframe(
    runtime: ToolUseJournalEERuntime,
    request: MotionPlanRequest,
    keyframe: object,
    object_id: str,
) -> AttachedObjectTransform:
    target_hand_pose = RelativePoseResolver(request.world).resolve(keyframe)
    kind, name, site_position, site_rotation = _target_site_pose(
        runtime.env, runtime, target_hand_pose
    )
    record = request.world.objects[object_id]
    object_position, object_rotation = _world_pose(record)
    relative_position = site_rotation.T @ (object_position - site_position)
    relative_rotation = site_rotation.T @ object_rotation
    return AttachedObjectTransform(
        object_id=object_id,
        free_joint_name=str(record["free_joint_name"]),
        reference_kind=kind,
        reference_name=name,
        position_in_reference_m=tuple(float(value) for value in relative_position),
        orientation_in_reference_xyzw=_xyzw_from_rotation(relative_rotation),
    )


def _runtime_attachment_transform(
    runtime: ToolUseJournalEERuntime,
) -> AttachedObjectTransform:
    attachment = runtime.attachment
    if attachment is None:
        raise RuntimeError("runtime has no attached plate")
    return AttachedObjectTransform(
        object_id=attachment.object_id,
        free_joint_name=attachment.free_joint_name,
        reference_kind=attachment.reference_kind,  # type: ignore[arg-type]
        reference_name=attachment.reference_name,
        position_in_reference_m=attachment.position_in_reference_m,
        orientation_in_reference_xyzw=_xyzw_from_rotation(
            np.asarray(attachment.rotation_in_reference, dtype=float)
        ),
    )


def _limits(joint_names: list[str]) -> dict[str, JointDynamicLimit]:
    return {
        name: JointDynamicLimit(
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=2.0,
            max_jerk_rad_s3=20.0,
        )
        for name in joint_names
    }


def _constraints(joint_names: list[str]) -> MotionConstraints:
    return MotionConstraints(
        collision_margin_m=0.002,
        position_tolerance_m=0.008,
        orientation_tolerance_rad=0.08,
        velocity_scaling=0.25,
        acceleration_scaling=0.25,
        jerk_scaling=0.25,
        max_cartesian_speed_m_s=0.18,
        max_joint_path_step_rad=0.035,
        joint_limits=_limits(joint_names),
    )


def _options(
    seed: int,
    *,
    allowed_planning_time_s: float,
    rrt_max_iterations: int,
) -> PlannerOptions:
    return PlannerOptions(
        allowed_planning_time_s=allowed_planning_time_s,
        max_attempts=5,
        interpolation_dt_s=0.02,
        cartesian_translation_step_m=0.01,
        cartesian_rotation_step_rad=0.08,
        rrt_extension_step_rad=0.2,
        rrt_max_iterations=rrt_max_iterations,
        rrt_goal_bias=0.2,
        random_seed=seed,
    )


def _request(
    *,
    request_id: str,
    world: object,
    task: MotionTask,
    seed: int,
    task_planner_artifact: Path,
    allowed_planning_time_s: float,
    rrt_max_iterations: int,
) -> MotionPlanRequest:
    return MotionPlanRequest(
        request_id=request_id,
        provenance=_provenance(
            f"{request_id}:artifact",
            "MotionPlanRequest",
            ModuleName.TASK_PLANNER,
            str(task_planner_artifact.resolve()),
        ),
        world=world,
        task=task,
        constraints=_constraints(list(world.robot_state.joint_names)),
        options=_options(
            seed,
            allowed_planning_time_s=allowed_planning_time_s,
            rrt_max_iterations=rrt_max_iterations,
        ),
    )


def _openai_artifact(
    request: MotionPlanRequest,
    *,
    model: str,
    candidates: int,
    cache_dir: Path,
) -> KeyframePlanArtifact:
    config = OpenAIKeyframeProviderConfig.from_environment(
        model=model,
        candidate_count=candidates,
        reasoning_effort="medium",
        max_output_tokens=16_000,
        timeout_s=120.0,
        cache_dir=cache_dir,
    )
    return OpenAIKeyframeProvider(config).generate(request)


def _install_sweep_reference_frames(
    world: object,
    *,
    target_ids: Sequence[str],
    goal_region_id: str,
    tool_id: str,
    attachment_position_in_reference_m: tuple[float, float, float],
    attachment_rotation_in_reference: np.ndarray | None = None,
    profile: _C1MotionProfile = _DEFAULT_MOTION_PROFILE,
    alignment_blend: float | None = None,
    broad_face_normal_deviation_rad: float = 0.0,
    frame_prefix: str = "sweep_target",
    max_push_distance_m: float | None = None,
    eef_rotation_override: np.ndarray | None = None,
    start_offset_m: float | None = None,
    table_surface_z_override_m: float | None = None,
    recovery_eef_position_world_m: np.ndarray | None = None,
    recovery_lift_m: float | None = None,
) -> tuple[str, ...]:
    """Add collision-free transit and radial broad-face push frames.

    The old three-lane template had two physical defects: each lane preserved
    the block's lateral coordinate instead of steering it into the narrow
    collection zone, and the plate travelled back out to the next lane at
    table height, undoing earlier pushes.  These frames target each observed
    block radially and lift the plate clear of the table between pushes.

    The measured friction-grasp transform is inverted for every target pose.
    The plate's broad-face normal points along the inward push direction and a
    point on that face at the observed block height follows the requested push
    line.  Computing the plate-center work height from observed table/block
    geometry keeps the vertical plate tangent to the table.
    """

    zone = world.objects.get(goal_region_id)
    plate = world.objects.get(tool_id)
    if not isinstance(zone, dict) or not isinstance(plate, dict):
        raise RuntimeError("selected sweep region or tool is missing from the world")
    zone_position = np.asarray(zone["pose"]["position_m"], dtype=float)
    plate_dimensions = np.asarray(plate.get("dimensions_m"), dtype=float)
    if plate_dimensions.shape != (3,) or not np.all(plate_dimensions > 0.0):
        raise RuntimeError(f"{tool_id} dimensions are missing or invalid")

    block_records: list[tuple[str, dict]] = []
    for name in target_ids:
        record = world.objects.get(name)
        if not isinstance(record, dict):
            raise RuntimeError(
                f"Task Planner sweep target {name!r} is missing from the world"
            )
        block_records.append((name, record))
    if not block_records:
        raise RuntimeError("Task Planner sweep target list is empty")

    table_surface_samples: list[float] = []
    for _, record in block_records:
        position = np.asarray(record["pose"]["position_m"], dtype=float)
        dimensions = np.asarray(record.get("dimensions_m"), dtype=float)
        if position.shape != (3,) or dimensions.shape != (3,):
            raise RuntimeError("C1_1 block pose or dimensions are invalid")
        table_surface_samples.append(float(position[2] - dimensions[2] * 0.5))
    table_surface_z = (
        float(np.median(table_surface_samples))
        if table_surface_z_override_m is None
        else float(table_surface_z_override_m)
    )
    if not np.isfinite(table_surface_z):
        raise ValueError("table surface override must be finite")
    attachment_offset = np.asarray(
        attachment_position_in_reference_m, dtype=float
    )
    has_live_physical_rotation = attachment_rotation_in_reference is not None
    # The live relative rotation generalizes the verified rim sweep geometry
    # to a weld-free, angled friction grasp.
    attachment_rotation = (
        np.diag((1.0, -1.0, -1.0))
        if attachment_rotation_in_reference is None
        else np.asarray(attachment_rotation_in_reference, dtype=float)
    )
    if attachment_rotation.shape != (3, 3):
        raise RuntimeError("plate/EFF relative rotation must be 3x3")
    live_eef_rotation = (
        None
        if eef_rotation_override is None
        else np.asarray(eef_rotation_override, dtype=float)
    )
    if live_eef_rotation is not None and live_eef_rotation.shape != (3, 3):
        raise RuntimeError("EEF rotation override must be 3x3")
    names: list[str] = []
    for block_name, record in block_records:
        block_position = np.asarray(record["pose"]["position_m"], dtype=float)
        block_dimensions = np.asarray(record["dimensions_m"], dtype=float)
        outward_xy = block_position[:2] - zone_position[:2]
        norm = float(np.linalg.norm(outward_xy))
        if norm <= 1e-9:
            continue
        outward_xy /= norm
        axis = [float(outward_xy[0]), float(outward_xy[1]), 0.0]
        block_support_m = 0.5 * float(
            np.dot(np.abs(outward_xy), block_dimensions[:2])
        )
        # Start at the configured non-contact stand-off. End close to the zone
        # center so the observed block geometry is safely inside the region.
        selected_start_offset_m = (
            profile.sweep_start_offset_m
            if start_offset_m is None
            else start_offset_m
        )
        if not np.isfinite(selected_start_offset_m) or selected_start_offset_m <= 0:
            raise ValueError("start_offset_m must be finite and positive")
        start_contact_xy = block_position[:2] + outward_xy * selected_start_offset_m
        end_radial_distance_m = profile.sweep_end_offset_m
        if max_push_distance_m is not None:
            if not np.isfinite(max_push_distance_m) or max_push_distance_m <= 0.0:
                raise ValueError("max_push_distance_m must be finite and positive")
            end_radial_distance_m = max(
                end_radial_distance_m,
                norm - max_push_distance_m,
            )
        end_block_center_xy = (
            zone_position[:2] + outward_xy * end_radial_distance_m
        )
        # Preserve the verified V3 behavior: the requested radial endpoint is
        # used as the rim contact target. The closed-loop contact controller
        # stops on the observed block state rather than on an open-loop tool
        # center displacement.
        end_contact_xy = end_block_center_xy
        # The verified rim sweep uses task-sector roll preferences, then
        # rotates the acquired plate into a vertical radial plane.
        far_corner = (
            float(block_position[0]) > profile.far_corner_min_x_m
            and abs(float(block_position[1] - zone_position[1]))
            > profile.far_corner_min_abs_y_from_zone_m
        )
        singularity_avoidance_sector = (
            float(outward_xy[1]) >= profile.sweep_positive_y_roll_switch
        )
        selected_variant_value = (
            profile.sweep_plane_alignment_blend
            if alignment_blend is None
            else float(alignment_blend)
        )
        orientation_variant_index = _nearest_alignment_variant_index(
            profile,
            selected_variant_value,
        )
        selected_elevation = (
            profile.sweep_tool_axis_elevation_rad
            if has_live_physical_rotation
            else 0.0
        )
        selected_roll = (
            (
                profile.far_corner_roll_rad
                if far_corner or singularity_avoidance_sector
                else profile.sweep_tool_roll_rad
            )
            if has_live_physical_rotation
            else (
                -float(np.sign(block_position[1] - zone_position[1]))
                * profile.far_corner_roll_rad
                if far_corner
                else 0.0
            )
        )
        tool_axis_world = np.asarray(
            (
                np.cos(selected_elevation) * outward_xy[0],
                np.cos(selected_elevation) * outward_xy[1],
                np.sin(selected_elevation),
            ),
            dtype=float,
        )
        plate_plane_alignment: float | None = None
        if live_eef_rotation is not None:
            eef_rotation = live_eef_rotation
            tool_axis_world, tool_roll_rad = _tool_axis_roll_from_rotation(
                eef_rotation
            )
            plate_rotation = eef_rotation @ attachment_rotation
            desired_normal = np.asarray(
                (-outward_xy[1], outward_xy[0], 0.0), dtype=float
            )
            desired_normal /= np.linalg.norm(desired_normal)
            plate_plane_alignment = abs(
                float(np.dot(plate_rotation[:, 2], desired_normal))
            )
            orientation_source = "LIVE_EEF_CONTINUATION"
        elif has_live_physical_rotation:
            tool_axis_world, tool_roll_rad, plate_plane_alignment = (
                _plate_sweep_roll_for_vertical_radial_plane(
                    tool_axis_world=tool_axis_world,
                    attachment_rotation_in_reference=attachment_rotation,
                    outward_xy=outward_xy,
                    preferred_roll_rad=selected_roll,
                    axis_blend=selected_variant_value,
                )
            )
            eef_rotation = _tool_rotation_from_axis_roll(
                tool_axis_world, tool_roll_rad
            )
            plate_rotation = eef_rotation @ attachment_rotation
            orientation_source = "REACHABILITY_RANKED_ALIGNMENT"
        else:
            tool_roll_rad = selected_roll
            eef_rotation = _tool_rotation_from_axis_roll(
                tool_axis_world, tool_roll_rad
            )
            plate_rotation = eef_rotation @ attachment_rotation
            orientation_source = "LEGACY_STATIC_ALIGNMENT"
        plate_vertical_half_extent_m = _circular_plate_vertical_half_extent_m(
            plate_rotation,
            plate_dimensions,
        )
        plate_center_work_z = (
            table_surface_z
            + plate_vertical_half_extent_m
            + profile.plate_table_clearance_m
        )
        target_contact_world_z_m = float(
            table_surface_z
            + profile.sweep_push_contact_height_fraction * block_dimensions[2]
        )
        block_support_center_z_m = float(
            table_surface_z + 0.5 * block_dimensions[2]
        )
        contact_offset = _circular_plate_rim_contact_offset_m(
            plate_rotation,
            plate_dimensions,
            vertical_offset_m=float(
                target_contact_world_z_m - plate_center_work_z
            ),
            preferred_direction_xy=-outward_xy,
        )
        contact_offset_local = plate_rotation.T @ contact_offset
        start_center_xy = start_contact_xy - contact_offset[:2]
        end_center_xy = end_contact_xy - contact_offset[:2]

        def plate_center_position(xy: np.ndarray, *, hover: bool) -> np.ndarray:
            return np.asarray(
                (
                    float(xy[0]),
                    float(xy[1]),
                    plate_center_work_z + (profile.hover_height_m if hover else 0.0),
                ),
                dtype=float,
            )

        def eef_position(xy: np.ndarray, *, hover: bool) -> np.ndarray:
            return (
                plate_center_position(xy, hover=hover)
                - eef_rotation @ attachment_offset
            )

        start_eef = eef_position(start_center_xy, hover=False)
        end_eef = eef_position(end_center_xy, hover=False)
        frame_specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
            (
                "hover_start",
                eef_position(start_center_xy, hover=True),
                plate_center_position(start_center_xy, hover=True),
                start_contact_xy,
            ),
            (
                "engage",
                start_eef,
                plate_center_position(start_center_xy, hover=False),
                start_contact_xy,
            ),
            (
                "end",
                end_eef,
                plate_center_position(end_center_xy, hover=False),
                end_contact_xy,
            ),
            (
                "hover_end",
                eef_position(end_center_xy, hover=True),
                plate_center_position(end_center_xy, hover=True),
                end_contact_xy,
            ),
        ]
        if recovery_eef_position_world_m is not None:
            recovery_eef = np.asarray(
                recovery_eef_position_world_m, dtype=float
            )
            if recovery_eef.shape != (3,):
                raise ValueError("recovery EEF position must contain 3 values")
            if recovery_lift_m is None or recovery_lift_m <= 0.0:
                raise ValueError("recovery_lift_m must be positive")
            lift = np.asarray((0.0, 0.0, recovery_lift_m), dtype=float)
            current_plate_center = recovery_eef + eef_rotation @ attachment_offset
            frame_specs[0:0] = [
                (
                    "recovery_lift",
                    recovery_eef + lift,
                    current_plate_center + lift,
                    start_contact_xy,
                ),
                (
                    "recovery_backoff",
                    start_eef + lift,
                    plate_center_position(start_center_xy, hover=False) + lift,
                    start_contact_xy,
                ),
            ]
        for phase, position, plate_center_target, contact_target_xy in frame_specs:
            name = f"{frame_prefix}_{block_name}_{phase}"
            world.objects[name] = {
                "pose": {
                    "frame_id": "world",
                    "position_m": [float(value) for value in position],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "anchors": {"center": [0.0, 0.0, 0.0]},
                "collision_enabled": False,
                "reference_frame_kind": "TASK_GEOMETRY",
                "target_block_id": block_name,
                "push_axis_world": axis,
                "tool_axis_world": [
                    float(value) for value in tool_axis_world
                ],
                "roll_rad": tool_roll_rad,
                "tool_roll_rad": tool_roll_rad,
                "contact_surface": "RIM",
                "vertical_radial_plane_alignment": plate_plane_alignment,
                "orientation_source": orientation_source,
                "orientation_variant_index": orientation_variant_index,
                "orientation_variant_value": selected_variant_value,
                "far_corner": far_corner,
                "singularity_avoidance_sector": (
                    singularity_avoidance_sector
                ),
                "table_surface_z_m": table_surface_z,
                "plate_center_work_z_m": plate_center_work_z,
                "plate_vertical_half_extent_m": plate_vertical_half_extent_m,
                "plate_contact_offset_m": [
                    float(value) for value in contact_offset
                ],
                "plate_contact_offset_local_m": [
                    float(value) for value in contact_offset_local
                ],
                "target_block_support_m": block_support_m,
                "target_block_height_m": float(block_dimensions[2]),
                "target_contact_height_offset_from_block_center_m": float(
                    (profile.sweep_push_contact_height_fraction - 0.5)
                    * block_dimensions[2]
                ),
                "target_contact_world_z_m": target_contact_world_z_m,
                "block_support_center_z_m": block_support_center_z_m,
                "contact_point_target_position_m": [
                    float(contact_target_xy[0]),
                    float(contact_target_xy[1]),
                    target_contact_world_z_m,
                ],
                "tool_target_position_m": [
                    float(value) for value in plate_center_target
                ],
            }
            names.append(name)
    return tuple(names)


def _physical_grasp_transform_in_reference(
    runtime: ToolUseJournalEERuntime,
    object_id: str,
) -> tuple[tuple[float, float, float], np.ndarray]:
    """Measure a free object's live offset from the gripper reference.

    CONTACT_FRICTION deliberately leaves ``runtime.attachment`` empty.  The
    sweep geometry still needs the plate-center offset acquired by the real
    finger contact, so measure that transform from MuJoCo instead of creating
    a synthetic attachment state.
    """

    env = runtime.env
    if object_id not in env.obj_body_id:  # type: ignore[attr-defined]
        raise RuntimeError(f"physical grasp object {object_id!r} is absent")
    data = env.sim.data._data  # type: ignore[attr-defined]
    body_id = env.obj_body_id[object_id]  # type: ignore[attr-defined]
    _, _, reference_position, reference_rotation = runtime._grasp_reference(env)
    object_position = np.asarray(data.xpos[body_id], dtype=float)
    object_rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    relative = np.asarray(reference_rotation, dtype=float).T @ (
        object_position - np.asarray(reference_position, dtype=float)
    )
    relative_rotation = (
        np.asarray(reference_rotation, dtype=float).T @ object_rotation
    )
    return (
        tuple(float(value) for value in relative),
        relative_rotation,
    )


def _sweep_template_artifact(
    request: MotionPlanRequest,
    *,
    profile: _C1MotionProfile = _DEFAULT_MOTION_PROFILE,
    alignment_variant_offset: int = 0,
    contact_continuation: bool = False,
    planar_recovery: bool = False,
    include_retract: bool = False,
) -> KeyframePlanArtifact:
    """Build radial pushes ordered by alignment quality and live reachability."""

    input_payload = {
        "scene_signature": request.world.scene.signature,
        "subgoal_id": request.task.subgoal_id,
        "motion_profile": asdict(profile),
        "contact_continuation": contact_continuation,
        "planar_recovery": planar_recovery,
        "include_retract": include_retract,
        "frames": {
            name: request.world.objects[name]
            for name in sorted(request.world.objects)
            if name.startswith(("sweep_target_", "sweep_alignment_"))
        },
    }
    digest = hashlib.sha256(
        json.dumps(input_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    block_ids = sorted(
        (
            name
            for name in request.task.target_ids
            if f"sweep_target_{name}_hover_start" in request.world.objects
        ),
        key=lambda name: int(name.split("_")[-1]),
    )
    goal_region_id = request.task.goal.target_region_id
    if not goal_region_id or goal_region_id not in request.world.objects:
        raise RuntimeError("sweep request has no world-backed goal region")
    zone_position = np.asarray(
        request.world.objects[goal_region_id]["pose"]["position_m"], dtype=float
    )
    block_ids.sort(
        key=lambda name: float(
            np.arctan2(
                request.world.objects[name]["pose"]["position_m"][1]
                - zone_position[1],
                request.world.objects[name]["pose"]["position_m"][0]
                - zone_position[0],
            )
        )
    )
    frame_prefixes = ["sweep_target"]
    for variant_index in range(1, profile.sweep_plane_alignment_candidate_count):
        prefix = f"sweep_alignment_{variant_index}_target"
        if all(
            f"{prefix}_{block_id}_hover_start" in request.world.objects
            for block_id in block_ids
        ):
            frame_prefixes.append(prefix)
    if frame_prefixes:
        offset = alignment_variant_offset % len(frame_prefixes)
        frame_prefixes = frame_prefixes[offset:] + frame_prefixes[:offset]

    forward_order = tuple(block_ids)
    reverse_order = tuple(reversed(block_ids))
    block_orders = (forward_order,) if forward_order == reverse_order else (
        forward_order,
        reverse_order,
    )
    candidates: list[KeyframePlanCandidate] = []
    candidate_index = 0
    for frame_prefix in frame_prefixes:
        for block_order in block_orders:
            candidate_index += 1
            keyframes: list[RelativeKeyframeSpec] = []
            for sequence_index, block_id in enumerate(block_order, start=1):
                frame = request.world.objects[
                    f"{frame_prefix}_{block_id}_hover_start"
                ]
                engage_frame = request.world.objects[
                    f"{frame_prefix}_{block_id}_engage"
                ]
                end_frame = request.world.objects[
                    f"{frame_prefix}_{block_id}_end"
                ]
                physical_tool_control = {
                    "target_clearance_m": profile.plate_table_clearance_m,
                    "clearance_tolerance_m": profile.sweep_clearance_tolerance_m,
                    "max_table_penetration_m": profile.sweep_max_table_penetration_m,
                    "gain": profile.sweep_clearance_control_gain,
                    "rate_m_s": profile.sweep_clearance_control_rate_m_s,
                    "max_offset_m": profile.sweep_clearance_control_max_offset_m,
                    "activation_band_m": (
                        profile.sweep_clearance_control_activation_band_m
                    ),
                    "max_joint_offset_rad": (
                        profile.sweep_tool_control_max_joint_offset_rad
                    ),
                }
                physical_push_control = {
                    "push_axis_world": list(end_frame["push_axis_world"]),
                    "rim_contact_offset_local_m": list(
                        end_frame["plate_contact_offset_local_m"]
                    ),
                    "block_support_m": float(end_frame["target_block_support_m"]),
                    "contact_penetration_m": (
                        profile.sweep_push_contact_penetration_m
                    ),
                    "max_correction_m": profile.sweep_push_control_max_offset_m,
                    "contact_plan_time_scale": profile.sweep_push_plan_time_scale,
                    "reacquire_timeout_s": profile.sweep_push_reacquire_timeout_s,
                    "max_reacquire_attempts": (
                        profile.sweep_push_max_reacquire_attempts
                    ),
                    "contact_height_offset_from_block_center_m": float(
                        end_frame[
                            "target_contact_height_offset_from_block_center_m"
                        ]
                    ),
                    "contact_height_target_m": float(
                        end_frame.get(
                            "target_contact_world_z_m",
                            request.world.objects[block_id]["pose"][
                                "position_m"
                            ][2],
                        )
                    ),
                    "block_support_center_z_m": float(
                        end_frame["block_support_center_z_m"]
                    ),
                    "block_support_tolerance_m": (
                        profile.sweep_block_support_tolerance_m
                    ),
                    "contact_height_gain": profile.sweep_push_contact_height_gain,
                    "contact_height_rate_m_s": (
                        profile.sweep_push_contact_height_rate_m_s
                    ),
                    "contact_height_max_offset_m": (
                        profile.sweep_push_contact_height_max_offset_m
                    ),
                    "contact_height_max_downward_offset_m": (
                        profile.sweep_push_contact_height_max_downward_offset_m
                    ),
                }
                common: dict[str, object] = {
                    "anchor": "center",
                    "approach_axis_xyz": tuple(
                        frame.get("tool_axis_world", frame["push_axis_world"])
                    ),
                    "tool_axis_to_align": "+z",
                    "offset_along_approach_m": 0.0,
                    "roll_rad": float(
                        frame.get("tool_roll_rad", frame["roll_rad"])
                    ),
                }
                if not contact_continuation:
                    approach_keyframes = (
                            RelativeKeyframeSpec(
                                keyframe_id=(
                                    f"{request.task.subgoal_id}:{block_id}:"
                                    "hover-start"
                                ),
                                keyframe_type=KeyframeType.TRANSFER,
                                frame_ref=(
                                    f"object:{frame_prefix}_{block_id}_hover_start"
                                ),
                                planner=KeyframePlannerType.SAMPLING_BASED,
                                metadata={"target_block_id": block_id},
                                **common,
                            ),
                            RelativeKeyframeSpec(
                                keyframe_id=(
                                    f"{request.task.subgoal_id}:{block_id}:engage"
                                ),
                                keyframe_type=KeyframeType.CUSTOM,
                                frame_ref=(
                                    f"object:{frame_prefix}_{block_id}_engage"
                                ),
                                planner=KeyframePlannerType.CARTESIAN,
                                metadata={
                                    "target_block_id": block_id,
                                    "hold_duration_after_s": profile.engage_hold_s,
                                    "tracking_settle": {
                                        "joint_tolerance_rad": (
                                            profile.settle_joint_tolerance_rad
                                        ),
                                        "eef_tolerance_m": (
                                            profile.settle_eef_tolerance_m
                                        ),
                                        "max_wait_s": profile.settle_max_wait_s,
                                        "required_consecutive_ticks": (
                                            profile.settle_required_consecutive_ticks
                                        ),
                                    },
                                    "physical_tool_control": physical_tool_control,
                                    "physical_tool_settle": {
                                        "target_position_m": engage_frame[
                                            "tool_target_position_m"
                                        ],
                                        "target_clearance_m": (
                                            profile.plate_table_clearance_m
                                        ),
                                        "xy_tolerance_m": (
                                            profile.sweep_tool_xy_tolerance_m
                                        ),
                                        "clearance_tolerance_m": (
                                            profile.sweep_clearance_tolerance_m
                                        ),
                                        "max_table_penetration_m": (
                                            profile.sweep_max_table_penetration_m
                                        ),
                                        "max_tool_speed_m_s": (
                                            profile.sweep_max_tool_speed_m_s
                                        ),
                                    },
                                },
                                **common,
                            ),
                        )
                    if planar_recovery:
                        keyframes.extend(
                            (
                                RelativeKeyframeSpec(
                                    keyframe_id=(
                                        f"{request.task.subgoal_id}:{block_id}:"
                                        "recovery-lift"
                                    ),
                                    keyframe_type=KeyframeType.CUSTOM,
                                    frame_ref=(
                                        f"object:{frame_prefix}_{block_id}_"
                                        "recovery_lift"
                                    ),
                                    planner=KeyframePlannerType.CARTESIAN,
                                    metadata={"target_block_id": block_id},
                                    **common,
                                ),
                                RelativeKeyframeSpec(
                                    keyframe_id=(
                                        f"{request.task.subgoal_id}:{block_id}:"
                                        "recovery-backoff"
                                    ),
                                    keyframe_type=KeyframeType.CUSTOM,
                                    frame_ref=(
                                        f"object:{frame_prefix}_{block_id}_"
                                        "recovery_backoff"
                                    ),
                                    planner=KeyframePlannerType.CARTESIAN,
                                    metadata={
                                        "target_block_id": block_id,
                                        "hold_duration_after_s": (
                                            profile.engage_hold_s
                                        ),
                                    },
                                    **common,
                                ),
                            )
                        )
                        keyframes.extend(approach_keyframes[1:])
                    else:
                        keyframes.extend(approach_keyframes)
                keyframes.append(
                    RelativeKeyframeSpec(
                        keyframe_id=f"{request.task.subgoal_id}:{block_id}:sweep",
                        keyframe_type=KeyframeType.CUSTOM,
                        frame_ref=f"object:{frame_prefix}_{block_id}_end",
                        planner=KeyframePlannerType.CARTESIAN,
                        metadata={
                            "target_block_id": block_id,
                            "preserve_endpoint_continuity": contact_continuation,
                            "hold_duration_after_s": profile.sweep_hold_s,
                            "physical_tool_control": physical_tool_control,
                            "physical_push_control": physical_push_control,
                            "physical_tool_target_position_m": end_frame[
                                "tool_target_position_m"
                            ],
                        },
                        **common,
                    )
                )
                if contact_continuation:
                    # The schema requires at least two keyframes. The first
                    # segment performs the observed 3 cm continuation; this
                    # colocated second keyframe is a short physical hold, not
                    # another open-loop displacement.
                    keyframes.append(
                        RelativeKeyframeSpec(
                            keyframe_id=(
                                f"{request.task.subgoal_id}:{block_id}:"
                                "sweep-hold"
                            ),
                            keyframe_type=KeyframeType.CUSTOM,
                            frame_ref=f"object:{frame_prefix}_{block_id}_end",
                            planner=KeyframePlannerType.CARTESIAN,
                            metadata={
                                "target_block_id": block_id,
                                "preserve_endpoint_continuity": True,
                                "hold_duration_after_s": profile.sweep_hold_s,
                                "physical_tool_control": physical_tool_control,
                                "physical_push_control": physical_push_control,
                                "physical_tool_target_position_m": end_frame[
                                    "tool_target_position_m"
                                ],
                            },
                            **common,
                        )
                    )
                if include_retract:
                    keyframes.append(
                        RelativeKeyframeSpec(
                            keyframe_id=(
                                f"{request.task.subgoal_id}:{block_id}:retract"
                            ),
                            keyframe_type=KeyframeType.RETREAT,
                            frame_ref=(
                                f"object:{frame_prefix}_{block_id}_hover_end"
                            ),
                            planner=KeyframePlannerType.CARTESIAN,
                            metadata={
                                "target_block_id": block_id,
                                "hold_duration_after_s": (
                                    profile.final_retract_hold_s
                                    if sequence_index == len(block_order)
                                    else profile.intermediate_retract_hold_s
                                ),
                            },
                            **common,
                        )
                    )
            alignment_blend = request.world.objects[
                f"{frame_prefix}_{block_order[0]}_engage"
            ].get("orientation_variant_value")
            candidates.append(
                KeyframePlanCandidate(
                    strategy_id=(
                        f"{request.task.subgoal_id}:radial-block-push-"
                        f"{candidate_index}:alignment-{alignment_blend}"
                    ),
                    keyframes=keyframes,
                    rationale=(
                        "Use the highest-ranked IK-valid plate-plane alignment, "
                        "push radially, and preserve contact for observation-based "
                        "continuation unless a re-approach is requested."
                    ),
                    provenance=StrategyGenerationProvenance(
                        generator_kind=StrategyGeneratorKind.TASK_GEOMETRY,
                        generator_id="C1_1_MJCF_RADIAL_SWEEP_V3",
                        input_hash=digest,
                        attempt_index=candidate_index,
                    ),
                )
            )
    return KeyframePlanArtifact(
        artifact_id=f"keyframe-plan:c1-1-radial-sweep:{digest[:24]}",
        provenance=_provenance(
            f"keyframe-plan-artifact:c1-1-radial-sweep:{digest[:24]}",
            "KeyframePlanArtifact",
            ModuleName.MOTION_PLANNER,
            request.provenance.artifact_id,
        ),
        scene_signature=request.world.scene.signature,
        subgoal_id=request.task.subgoal_id,
        candidates=candidates,
    )


def _task_geometry_motion_phase(frame_name: str) -> str:
    return next(
        (
            candidate
            for candidate in (
                "recovery_lift",
                "recovery_backoff",
                "hover_start",
                "hover_end",
                "engage",
                "end",
            )
            if frame_name.endswith(f"_{candidate}")
        ),
        "",
    )


def _expand_task_geometry_orientation_variants(
    raw: KeyframePlanArtifact,
    request: MotionPlanRequest,
    *,
    maximum_candidates: int = 20,
) -> KeyframePlanArtifact:
    """Expand a provider-selected route over grounded orientation variants."""

    if maximum_candidates < 1:
        raise ValueError("maximum_candidates must be positive")
    variant_frames: dict[int, dict[tuple[str, str], str]] = {}
    for frame_name, frame in request.world.objects.items():
        if not isinstance(frame, Mapping) or (
            frame.get("reference_frame_kind") != "TASK_GEOMETRY"
        ):
            continue
        variant_value = frame.get("orientation_variant_index")
        target_block_id = frame.get("target_block_id")
        phase = _task_geometry_motion_phase(frame_name)
        if variant_value is None or target_block_id is None or not phase:
            continue
        variant_index = int(variant_value)
        variant_frames.setdefault(variant_index, {})[
            (str(target_block_id), phase)
        ] = frame_name
    if len(variant_frames) <= 1:
        return raw

    expanded: list[KeyframePlanCandidate] = []
    for expansion_index, variant_index in enumerate(sorted(variant_frames)):
        if len(expanded) >= maximum_candidates:
            break
        source_candidate = raw.candidates[expansion_index % len(raw.candidates)]
        keyframes: list[RelativeKeyframeSpec] = []
        compatible = True
        for keyframe in source_candidate.keyframes:
            if not keyframe.frame_ref.startswith("object:"):
                keyframes.append(keyframe)
                continue
            source_name = keyframe.frame_ref.removeprefix("object:")
            source_frame = request.world.objects.get(source_name)
            if not isinstance(source_frame, Mapping) or (
                source_frame.get("reference_frame_kind") != "TASK_GEOMETRY"
            ):
                keyframes.append(keyframe)
                continue
            target_block_id = source_frame.get("target_block_id")
            phase = _task_geometry_motion_phase(source_name)
            destination_name = variant_frames[variant_index].get(
                (str(target_block_id), phase)
            )
            if destination_name is None:
                compatible = False
                break
            keyframes.append(
                keyframe.model_copy(
                    update={"frame_ref": f"object:{destination_name}"}
                )
            )
        if not compatible:
            continue
        expanded.append(
            source_candidate.model_copy(
                update={
                    "strategy_id": (
                        f"{source_candidate.strategy_id}:grounded-orientation-"
                        f"{variant_index}"
                    ),
                    "keyframes": keyframes,
                    "metadata": {
                        **source_candidate.metadata,
                        "source_strategy_id": source_candidate.strategy_id,
                        "grounded_orientation_variant_index": variant_index,
                    },
                }
            )
        )
    if not expanded:
        return raw
    digest = hashlib.sha256(
        json.dumps(
            {
                "source_artifact_id": raw.artifact_id,
                "orientation_variants": sorted(variant_frames),
                "candidate_count": len(expanded),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return raw.model_copy(
        update={
            "artifact_id": f"{raw.artifact_id}:orientation-expansion:{digest}",
            "provenance": raw.provenance.model_copy(
                update={
                    "artifact_id": (
                        f"{raw.provenance.artifact_id}:orientation-expansion:"
                        f"{digest}"
                    ),
                    "metadata": {
                        **raw.provenance.metadata,
                        "orientation_expander": (
                            "TASK_GEOMETRY_ORIENTATION_EXPANDER_V1"
                        ),
                        "source_keyframe_artifact_id": raw.artifact_id,
                        "expanded_candidate_count": len(expanded),
                    },
                }
            ),
            "candidates": expanded,
        }
    )


def _bind_grounded_task_geometry_keyframes(
    raw: KeyframePlanArtifact,
    request: MotionPlanRequest,
    *,
    execution_metadata_resolver: (
        Callable[[str, Mapping[str, object]], Mapping[str, object]] | None
    ) = None,
) -> KeyframePlanArtifact:
    """Preserve metric poses already grounded in TASK_GEOMETRY frames.

    A stochastic provider selects the semantic frame sequence. It must not
    perturb the metric axis, roll, or position that an upstream geometry
    grounding stage attached to those frames. This binder is generic to the
    reference-frame contract and contains no C1_1 object names or poses.
    """

    candidates: list[KeyframePlanCandidate] = []
    bound_frame_payload: dict[str, dict[str, object]] = {}
    for candidate in raw.candidates:
        keyframes: list[RelativeKeyframeSpec] = []
        for keyframe in candidate.keyframes:
            frame_name = (
                keyframe.frame_ref.removeprefix("object:")
                if keyframe.frame_ref.startswith("object:")
                else None
            )
            frame = (
                request.world.objects.get(frame_name)
                if frame_name is not None
                else None
            )
            if not isinstance(frame, Mapping) or (
                frame.get("reference_frame_kind") != "TASK_GEOMETRY"
            ):
                keyframes.append(keyframe)
                continue
            axis_value = frame.get("tool_axis_world")
            roll_value = frame.get("tool_roll_rad", frame.get("roll_rad"))
            if not isinstance(axis_value, Sequence) or len(axis_value) != 3:
                raise RuntimeError(
                    f"grounded frame {frame_name!r} has no 3D tool_axis_world"
                )
            axis = np.asarray(axis_value, dtype=float)
            axis_norm = float(np.linalg.norm(axis))
            if not np.isfinite(axis).all() or axis_norm <= 1e-9:
                raise RuntimeError(
                    f"grounded frame {frame_name!r} has an invalid tool axis"
                )
            if roll_value is None or not np.isfinite(float(roll_value)):
                raise RuntimeError(
                    f"grounded frame {frame_name!r} has an invalid tool roll"
                )
            normalized_axis = tuple(float(value) for value in axis / axis_norm)
            grounded_roll = float(roll_value)
            execution_metadata = (
                {}
                if execution_metadata_resolver is None
                else dict(execution_metadata_resolver(frame_name, frame))
            )
            keyframes.append(
                keyframe.model_copy(
                    update={
                        "anchor": "center",
                        "approach_axis_xyz": normalized_axis,
                        "tool_axis_to_align": "+z",
                        "offset_along_approach_m": 0.0,
                        "roll_rad": grounded_roll,
                        "metadata": {
                            **keyframe.metadata,
                            **execution_metadata,
                            "metric_grounding_source": (
                                "TASK_GEOMETRY_REFERENCE_FRAME"
                            ),
                        },
                    }
                )
            )
            bound_frame_payload[frame_name] = {
                "tool_axis_world": list(normalized_axis),
                "tool_roll_rad": grounded_roll,
            }
        candidates.append(candidate.model_copy(update={"keyframes": keyframes}))
    if not bound_frame_payload:
        return raw
    digest = hashlib.sha256(
        json.dumps(
            {
                "source_artifact_id": raw.artifact_id,
                "bound_frames": bound_frame_payload,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return raw.model_copy(
        update={
            "artifact_id": f"{raw.artifact_id}:task-geometry:{digest}",
            "provenance": raw.provenance.model_copy(
                update={
                    "artifact_id": (
                        f"{raw.provenance.artifact_id}:task-geometry:{digest}"
                    ),
                    "metadata": {
                        **raw.provenance.metadata,
                        "geometry_binder": "TASK_GEOMETRY_KEYFRAME_BINDER_V1",
                        "source_keyframe_artifact_id": raw.artifact_id,
                        "bound_frame_count": len(bound_frame_payload),
                    },
                }
            ),
            "candidates": candidates,
        }
    )


def _sweep_frame_execution_metadata(
    frame_name: str,
    frame: Mapping[str, object],
    *,
    profile: _C1MotionProfile,
) -> dict[str, object]:
    """Build controller metadata from a grounded sweep reference frame."""

    phase = _task_geometry_motion_phase(frame_name)
    target_block_id = frame.get("target_block_id")
    metadata: dict[str, object] = {}
    if target_block_id is not None:
        metadata["target_block_id"] = str(target_block_id)
    if phase not in {"engage", "end"}:
        if phase == "hover_end":
            metadata["hold_duration_after_s"] = profile.final_retract_hold_s
        return metadata
    physical_tool_control = {
        "target_clearance_m": profile.plate_table_clearance_m,
        "clearance_tolerance_m": profile.sweep_clearance_tolerance_m,
        "max_table_penetration_m": profile.sweep_max_table_penetration_m,
        "gain": profile.sweep_clearance_control_gain,
        "rate_m_s": profile.sweep_clearance_control_rate_m_s,
        "max_offset_m": profile.sweep_clearance_control_max_offset_m,
        "activation_band_m": profile.sweep_clearance_control_activation_band_m,
        "max_joint_offset_rad": profile.sweep_tool_control_max_joint_offset_rad,
    }
    metadata["physical_tool_control"] = physical_tool_control
    if phase == "engage":
        metadata.update(
            {
                "hold_duration_after_s": profile.engage_hold_s,
                "tracking_settle": {
                    "joint_tolerance_rad": profile.settle_joint_tolerance_rad,
                    "eef_tolerance_m": profile.settle_eef_tolerance_m,
                    "max_wait_s": profile.settle_max_wait_s,
                    "required_consecutive_ticks": (
                        profile.settle_required_consecutive_ticks
                    ),
                },
                "physical_tool_settle": {
                    "target_position_m": list(frame["tool_target_position_m"]),
                    "target_clearance_m": profile.plate_table_clearance_m,
                    "xy_tolerance_m": profile.sweep_tool_xy_tolerance_m,
                    "clearance_tolerance_m": (
                        profile.sweep_clearance_tolerance_m
                    ),
                    "max_table_penetration_m": (
                        profile.sweep_max_table_penetration_m
                    ),
                    "max_tool_speed_m_s": profile.sweep_max_tool_speed_m_s,
                },
            }
        )
        return metadata
    metadata.update(
        {
            "preserve_endpoint_continuity": False,
            "hold_duration_after_s": profile.sweep_hold_s,
            "physical_push_control": {
                "push_axis_world": list(frame["push_axis_world"]),
                "contact_offset_local_m": list(
                    frame["plate_contact_offset_local_m"]
                ),
                "block_support_m": float(frame["target_block_support_m"]),
                "contact_penetration_m": (
                    profile.sweep_push_contact_penetration_m
                ),
                "max_correction_m": profile.sweep_push_control_max_offset_m,
                "contact_plan_time_scale": profile.sweep_push_plan_time_scale,
                "reacquire_timeout_s": profile.sweep_push_reacquire_timeout_s,
                "max_reacquire_attempts": (
                    profile.sweep_push_max_reacquire_attempts
                ),
                "contact_height_offset_from_block_center_m": float(
                    frame["target_contact_height_offset_from_block_center_m"]
                ),
                "contact_height_target_m": float(
                    frame["target_contact_world_z_m"]
                ),
                "block_support_center_z_m": float(
                    frame["block_support_center_z_m"]
                ),
                "block_support_tolerance_m": (
                    profile.sweep_block_support_tolerance_m
                ),
                "contact_height_gain": profile.sweep_push_contact_height_gain,
                "contact_height_rate_m_s": (
                    profile.sweep_push_contact_height_rate_m_s
                ),
                "contact_height_max_offset_m": (
                    profile.sweep_push_contact_height_max_offset_m
                ),
                "contact_height_max_downward_offset_m": (
                    profile.sweep_push_contact_height_max_downward_offset_m
                ),
            },
            "physical_tool_target_position_m": list(
                frame["tool_target_position_m"]
            ),
        }
    )
    return metadata


def _bind_2f_plate_rim_grasps(
    raw: KeyframePlanArtifact,
    request: MotionPlanRequest,
    *,
    tool_id: str,
    finger_centerline_inset_m: float = (
        _DEFAULT_MOTION_PROFILE.finger_centerline_inset_m
    ),
    grasp_site_offset_m: float = (
        _DEFAULT_MOTION_PROFILE.pick_grasp_site_offset_m
    ),
) -> KeyframePlanArtifact:
    """Bind symbolic top-down proposals to MJCF-derived plate rim grasps.

    OpenAI still supplies the high-level approach strategy.  Exact metric
    anchors come from the plate collision bounds and the Robotiq 85 finger-pad
    geometry.  Center grasps are invalid for this thin plate because the two
    fingers never oppose one another around an edge.
    """

    plate = request.world.objects.get(tool_id)
    if not isinstance(plate, dict):
        raise RuntimeError(f"{tool_id} MJCF record is absent")
    dimensions = plate.get("dimensions_m")
    anchors = plate.get("anchors")
    if (
        not isinstance(dimensions, list)
        or len(dimensions) < 3
        or not isinstance(anchors, dict)
        or not isinstance(anchors.get("center"), list)
        or not isinstance(anchors.get("top"), list)
    ):
        raise RuntimeError(f"{tool_id} MJCF bounds and anchors are absent")
    center = np.asarray(anchors["center"], dtype=float)
    top = np.asarray(anchors["top"], dtype=float)
    half_x = float(dimensions[0]) * 0.5
    half_y = float(dimensions[1]) * 0.5
    if min(half_x, half_y) <= finger_centerline_inset_m:
        raise RuntimeError(f"{tool_id} is too small for the 2F rim template")
    radial_x = half_x - finger_centerline_inset_m
    radial_y = half_y - finger_centerline_inset_m
    rim_anchors = {
        "2f_rim_x_pos": [center[0] + radial_x, center[1], top[2]],
        "2f_rim_x_neg": [center[0] - radial_x, center[1], top[2]],
        "2f_rim_y_pos": [center[0], center[1] + radial_y, top[2]],
        "2f_rim_y_neg": [center[0], center[1] - radial_y, top[2]],
    }
    anchors.update(
        {
            name: [float(value) for value in position]
            for name, position in rim_anchors.items()
        }
    )
    rim_variants = (
        ("2f_rim_x_pos", 0.0),
        ("2f_rim_x_neg", 0.0),
        ("2f_rim_y_pos", np.pi / 2.0),
        ("2f_rim_y_neg", np.pi / 2.0),
    )
    candidates: list[KeyframePlanCandidate] = []
    for candidate_index, candidate in enumerate(raw.candidates):
        anchor, roll = rim_variants[candidate_index % len(rim_variants)]
        keyframes: list[RelativeKeyframeSpec] = []
        for keyframe in candidate.keyframes:
            updates: dict[str, object] = {}
            if keyframe.frame_ref == f"object:{tool_id}":
                updates.update(
                    anchor=anchor,
                    approach_axis_xyz=(0.0, 0.0, 1.0),
                    tool_axis_to_align="-z",
                    roll_rad=float(roll),
                )
                if keyframe.keyframe_type is KeyframeType.PRE_GRASP:
                    updates["offset_along_approach_m"] = max(
                        0.08, keyframe.offset_along_approach_m
                    ) + grasp_site_offset_m
                elif keyframe.keyframe_type is KeyframeType.GRASP:
                    updates["offset_along_approach_m"] = grasp_site_offset_m
                elif keyframe.keyframe_type in {
                    KeyframeType.LIFT,
                    KeyframeType.RETREAT,
                }:
                    updates["offset_along_approach_m"] = max(
                        0.10, keyframe.offset_along_approach_m
                    ) + grasp_site_offset_m
            keyframes.append(keyframe.model_copy(update=updates))
        candidates.append(
            candidate.model_copy(
                update={
                    "strategy_id": f"{candidate.strategy_id}:mjcf-2f-rim",
                    "keyframes": keyframes,
                    "rationale": (
                        f"{candidate.rationale} Exact grasp pose is bound to "
                        f"MJCF rim anchor {anchor!r} for opposed 2F contact."
                    ),
                    "metadata": {
                        **candidate.metadata,
                        "geometry_binder": "C1_1_MJCF_2F_PLATE_RIM_V1",
                        "rim_anchor": anchor,
                        "finger_centerline_inset_m": (
                            finger_centerline_inset_m
                        ),
                        "grasp_site_offset_m": grasp_site_offset_m,
                    },
                }
            )
        )
    digest = hashlib.sha256(
        json.dumps(
            {
                "source_artifact_id": raw.artifact_id,
                "tool_id": tool_id,
                "plate_dimensions_m": dimensions,
                "rim_anchors": rim_anchors,
                "finger_centerline_inset_m": finger_centerline_inset_m,
                "grasp_site_offset_m": grasp_site_offset_m,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return raw.model_copy(
        update={
            "artifact_id": f"{raw.artifact_id}:mjcf-2f-rim:{digest}",
            "provenance": raw.provenance.model_copy(
                update={
                    "artifact_id": (
                        f"{raw.provenance.artifact_id}:mjcf-2f-rim:{digest}"
                    ),
                    "metadata": {
                        **raw.provenance.metadata,
                        "geometry_binder": "C1_1_MJCF_2F_PLATE_RIM_V1",
                        "source_keyframe_artifact_id": raw.artifact_id,
                    },
                }
            ),
            "candidates": candidates,
        }
    )


def _bind_2f_plate_side_grasps(
    raw: KeyframePlanArtifact,
    request: MotionPlanRequest,
    *,
    tool_id: str,
    radial_inset_m: float,
    vertical_offset_m: float,
    lift_distance_m: float,
    seat_start_lift_m: float,
    seat_descent_m: float,
    seat_radial_inset_m: float,
    lateral_compensation_m: float,
    approach_elevation_rad: float,
    roll_rad: float,
    side_variant_index: int,
) -> KeyframePlanArtifact:
    """Bind a pre-shaped 2F gripper to a horizontal plate edge.

    The initial edge pinch is retained throughout lift. Physical validation,
    rather than an attachment or EE/object proximity, decides whether that
    contact topology can carry the selected plate.
    """

    if radial_inset_m < 0.0:
        raise ValueError("radial_inset_m must be non-negative")
    if vertical_offset_m < 0.0:
        raise ValueError("vertical_offset_m must be non-negative")
    if lift_distance_m <= 0.0:
        raise ValueError("lift_distance_m must be positive")
    if not 0.0 <= seat_start_lift_m < lift_distance_m:
        raise ValueError(
            "seat_start_lift_m must be within [0, lift distance)"
        )
    if not 0.0 <= seat_descent_m < lift_distance_m:
        raise ValueError("seat_descent_m must be within [0, lift distance)")
    if seat_radial_inset_m < 0.0:
        raise ValueError("seat_radial_inset_m must be non-negative")
    if lateral_compensation_m < 0.0:
        raise ValueError("lateral_compensation_m must be non-negative")
    if not 0.0 < approach_elevation_rad < np.pi / 2.0:
        raise ValueError("approach_elevation_rad must be within (0, pi/2)")
    if not -np.pi <= roll_rad <= np.pi:
        raise ValueError("roll_rad must be within +/- pi")
    if not isinstance(side_variant_index, int) or not 0 <= side_variant_index < 4:
        raise ValueError("side_variant_index must be within 0..3")
    plate = request.world.objects.get(tool_id)
    if not isinstance(plate, dict):
        raise RuntimeError(f"{tool_id} MJCF record is absent")
    dimensions = plate.get("dimensions_m")
    anchors = plate.get("anchors")
    if (
        not isinstance(dimensions, list)
        or len(dimensions) < 3
        or not isinstance(anchors, dict)
        or not isinstance(anchors.get("center"), list)
    ):
        raise RuntimeError(f"{tool_id} MJCF bounds and anchors are absent")
    center = np.asarray(anchors["center"], dtype=float)
    half_x = float(dimensions[0]) * 0.5
    half_y = float(dimensions[1]) * 0.5
    if radial_inset_m >= min(half_x, half_y):
        raise RuntimeError(f"{tool_id} side-grasp inset exceeds plate radius")
    base_z = float(center[2]) + vertical_offset_m
    radial_component = float(np.cos(approach_elevation_rad))
    vertical_component = float(np.sin(approach_elevation_rad))
    side_variants = (
        (
            "2f_side_x_pos",
            np.asarray((center[0] + half_x - radial_inset_m, center[1], base_z)),
            (radial_component, 0.0, vertical_component),
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
        ),
        (
            "2f_side_x_neg",
            np.asarray((center[0] - half_x + radial_inset_m, center[1], base_z)),
            (-radial_component, 0.0, vertical_component),
            np.asarray((-1.0, 0.0, 0.0)),
            np.asarray((0.0, -1.0, 0.0)),
        ),
        (
            "2f_side_y_pos",
            np.asarray((center[0], center[1] + half_y - radial_inset_m, base_z)),
            (0.0, radial_component, vertical_component),
            np.asarray((0.0, 1.0, 0.0)),
            np.asarray((-1.0, 0.0, 0.0)),
        ),
        (
            "2f_side_y_neg",
            np.asarray((center[0], center[1] - half_y + radial_inset_m, base_z)),
            (0.0, -radial_component, vertical_component),
            np.asarray((0.0, -1.0, 0.0)),
            np.asarray((1.0, 0.0, 0.0)),
        ),
    )
    for name, position, _, radial, tangent in side_variants:
        anchors[name] = [float(value) for value in position]
        seating_start_height_m = (
            seat_start_lift_m
            if seat_start_lift_m > 0.0
            else lift_distance_m
        )
        overshoot = (
            position
            + np.asarray((0.0, 0.0, seating_start_height_m))
        )
        seated = (
            position
            + tangent * lateral_compensation_m
            + np.asarray((0.0, 0.0, lift_distance_m))
            - radial * seat_radial_inset_m
            - np.asarray((0.0, 0.0, seat_descent_m))
        )
        anchors[f"{name}_lift_overshoot"] = [
            float(value) for value in overshoot
        ]
        anchors[f"{name}_lift"] = [
            float(value) for value in seated
        ]

    candidates: list[KeyframePlanCandidate] = []
    for candidate_index, candidate in enumerate(raw.candidates):
        anchor, _, approach_axis, _, _ = side_variants[
            (side_variant_index + candidate_index) % len(side_variants)
        ]
        keyframes: list[RelativeKeyframeSpec] = []
        for keyframe in candidate.keyframes:
            updates: dict[str, object] = {}
            if keyframe.frame_ref == f"object:{tool_id}":
                updates.update(
                    approach_axis_xyz=approach_axis,
                    tool_axis_to_align="-z",
                    roll_rad=roll_rad,
                )
                if keyframe.keyframe_type is KeyframeType.PRE_GRASP:
                    updates.update(
                        anchor=anchor,
                        offset_along_approach_m=max(
                            request.task.goal.approach_distance_m or 0.08,
                            0.08,
                        ),
                    )
                elif keyframe.keyframe_type is KeyframeType.GRASP:
                    updates.update(
                        anchor=anchor,
                        offset_along_approach_m=0.0,
                    )
                elif keyframe.keyframe_type in {
                    KeyframeType.LIFT,
                    KeyframeType.RETREAT,
                }:
                    updates.update(
                        anchor=f"{anchor}_lift",
                        offset_along_approach_m=0.0,
                    )
            bound_keyframe = keyframe.model_copy(update=updates)
            if (
                (seat_descent_m > 0.0 or seat_radial_inset_m > 0.0)
                and keyframe.frame_ref == f"object:{tool_id}"
                and keyframe.keyframe_type is KeyframeType.LIFT
            ):
                keyframes.append(
                    bound_keyframe.model_copy(
                        update={
                            "keyframe_id": f"{keyframe.keyframe_id}:overshoot",
                            "keyframe_type": KeyframeType.TRANSFER,
                            "anchor": f"{anchor}_lift_overshoot",
                            "metadata": {
                                **bound_keyframe.metadata,
                                "continuous_grasp_seating": "OVERSHOOT",
                            },
                        }
                    )
                )
                bound_keyframe = bound_keyframe.model_copy(
                    update={
                        "metadata": {
                            **bound_keyframe.metadata,
                            "continuous_grasp_seating": "SEAT",
                            "seat_descent_m": seat_descent_m,
                            "seat_start_lift_m": seat_start_lift_m,
                            "seat_radial_inset_m": seat_radial_inset_m,
                        }
                    }
                )
            keyframes.append(bound_keyframe)
        candidates.append(
            candidate.model_copy(
                update={
                    "strategy_id": f"{candidate.strategy_id}:mjcf-2f-side",
                    "keyframes": keyframes,
                    "rationale": (
                        f"{candidate.rationale} Exact pose is bound to MJCF "
                        f"side anchor {anchor!r}; the pre-shaped 2F closing "
                        "axis pinches the selected rim sector without a weld."
                    ),
                    "metadata": {
                        **candidate.metadata,
                        "geometry_binder": "C1_1_MJCF_2F_PLATE_SIDE_V2",
                        "side_anchor": anchor,
                        "radial_inset_m": radial_inset_m,
                        "vertical_offset_m": vertical_offset_m,
                        "lift_distance_m": lift_distance_m,
                        "seat_descent_m": seat_descent_m,
                        "seat_start_lift_m": seat_start_lift_m,
                        "seat_radial_inset_m": seat_radial_inset_m,
                        "lateral_compensation_m": lateral_compensation_m,
                        "approach_elevation_rad": approach_elevation_rad,
                    },
                }
            )
        )
    digest = hashlib.sha256(
        json.dumps(
            {
                "source_artifact_id": raw.artifact_id,
                "tool_id": tool_id,
                "plate_dimensions_m": dimensions,
                "radial_inset_m": radial_inset_m,
                "vertical_offset_m": vertical_offset_m,
                "lift_distance_m": lift_distance_m,
                "seat_descent_m": seat_descent_m,
                "seat_start_lift_m": seat_start_lift_m,
                "seat_radial_inset_m": seat_radial_inset_m,
                "lateral_compensation_m": lateral_compensation_m,
                "approach_elevation_rad": approach_elevation_rad,
                "roll_rad": roll_rad,
                "side_variant_index": side_variant_index,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return raw.model_copy(
        update={
            "artifact_id": f"{raw.artifact_id}:mjcf-2f-side:{digest}",
            "provenance": raw.provenance.model_copy(
                update={
                    "artifact_id": (
                        f"{raw.provenance.artifact_id}:mjcf-2f-side:{digest}"
                    ),
                    "metadata": {
                        **raw.provenance.metadata,
                        "geometry_binder": "C1_1_MJCF_2F_PLATE_SIDE_V2",
                        "source_keyframe_artifact_id": raw.artifact_id,
                    },
                }
            ),
            "candidates": candidates,
        }
    )


def _with_contact_friction_grasp(
    raw: KeyframePlanArtifact,
    *,
    hold_duration_s: float,
    lift_hold_duration_s: float,
    gripper_close_rate: float,
    grasp_clearance_reserve_m: float = 0.0,
    settle_joint_tolerance_rad: float | None = None,
    settle_eef_tolerance_m: float | None = None,
    settle_max_wait_s: float = 3.0,
    settle_required_consecutive_ticks: int = 5,
) -> KeyframePlanArtifact:
    """Use persistent gripper force and MuJoCo contact friction without a weld."""

    if grasp_clearance_reserve_m < 0.0:
        raise ValueError("grasp_clearance_reserve_m must be non-negative")
    if hold_duration_s <= 0.0:
        raise ValueError("hold_duration_s must be positive")
    if lift_hold_duration_s <= 0.0:
        raise ValueError("lift_hold_duration_s must be positive")
    if not 0.0 < gripper_close_rate <= 1.0:
        raise ValueError("gripper_close_rate must be within (0, 1]")
    if settle_joint_tolerance_rad is not None and settle_joint_tolerance_rad <= 0:
        raise ValueError("settle_joint_tolerance_rad must be positive")
    if settle_eef_tolerance_m is not None and settle_eef_tolerance_m <= 0:
        raise ValueError("settle_eef_tolerance_m must be positive")

    bound = raw.model_copy(deep=True)
    valid_candidates: list[KeyframePlanCandidate] = []
    rejected_candidates: list[dict[str, str]] = []
    for candidate in bound.candidates:
        grasp = next(
            (
                keyframe
                for keyframe in candidate.keyframes
                if keyframe.keyframe_type is KeyframeType.GRASP
            ),
            None,
        )
        if grasp is None:
            rejected_candidates.append(
                {
                    "strategy_id": candidate.strategy_id,
                    "reason": "MISSING_GRASP_KEYFRAME",
                }
            )
            continue
        lift = next(
            (
                keyframe
                for keyframe in reversed(candidate.keyframes)
                if keyframe.keyframe_type is KeyframeType.LIFT
            ),
            None,
        )
        if lift is None:
            rejected_candidates.append(
                {
                    "strategy_id": candidate.strategy_id,
                    "reason": "MISSING_LIFT_KEYFRAME",
                }
            )
            continue
        grasp.offset_along_approach_m += grasp_clearance_reserve_m
        grasp.events_after = [
            event
            for event in grasp.events_after
            if event is not KeyframeEventType.ATTACH_OBJECT
        ]
        if KeyframeEventType.GRIPPER_CLOSE not in grasp.events_after:
            grasp.events_after.insert(0, KeyframeEventType.GRIPPER_CLOSE)
        event_parameters = dict(grasp.metadata.get("event_parameters", {}))
        event_parameters.pop("ATTACH_OBJECT", None)
        event_parameters["GRIPPER_CLOSE"] = {"command": gripper_close_rate}
        grasp.metadata = {
            **grasp.metadata,
            "grasp_execution_mode": "CONTACT_FRICTION",
            "hold_duration_after_s": hold_duration_s,
            "event_time_offsets_s": {"GRIPPER_CLOSE": 0.0},
            "event_parameters": event_parameters,
            **(
                {
                    "tracking_settle": {
                        **(
                            {"joint_tolerance_rad": settle_joint_tolerance_rad}
                            if settle_joint_tolerance_rad is not None
                            else {}
                        ),
                        **(
                            {"eef_tolerance_m": settle_eef_tolerance_m}
                            if settle_eef_tolerance_m is not None
                            else {}
                        ),
                        "max_wait_s": settle_max_wait_s,
                        "required_consecutive_ticks": (
                            settle_required_consecutive_ticks
                        ),
                    }
                }
                if settle_joint_tolerance_rad is not None
                or settle_eef_tolerance_m is not None
                else {}
            ),
        }
        lift.metadata = {
            **lift.metadata,
            "hold_duration_after_s": lift_hold_duration_s,
            "physical_retention_hold": True,
        }
        valid_candidates.append(candidate)
    if not valid_candidates:
        details = ", ".join(
            f"{item['strategy_id']}={item['reason']}"
            for item in rejected_candidates
        )
        raise RuntimeError(
            "no PICK strategy contains both GRASP and LIFT keyframes: " + details
        )
    bound.candidates = valid_candidates
    digest = hashlib.sha256(
        json.dumps(
            {
                "source_artifact_id": raw.artifact_id,
                "hold_duration_s": hold_duration_s,
                "lift_hold_duration_s": lift_hold_duration_s,
                "gripper_close_rate": gripper_close_rate,
                "grasp_clearance_reserve_m": grasp_clearance_reserve_m,
                "settle_joint_tolerance_rad": settle_joint_tolerance_rad,
                "settle_eef_tolerance_m": settle_eef_tolerance_m,
                "settle_max_wait_s": settle_max_wait_s,
                "settle_required_consecutive_ticks": (
                    settle_required_consecutive_ticks
                ),
                "accepted_strategy_ids": [
                    candidate.strategy_id for candidate in valid_candidates
                ],
                "rejected_candidates": rejected_candidates,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    bound.artifact_id = f"{raw.artifact_id}:contact-friction:{digest}"
    bound.provenance = raw.provenance.model_copy(
        update={
            "artifact_id": f"{raw.provenance.artifact_id}:contact-friction:{digest}",
            "metadata": {
                **raw.provenance.metadata,
                "grasp_execution_mode": "CONTACT_FRICTION",
                "rejected_candidates": rejected_candidates,
            },
        }
    )
    return bound


def _retarget_equivalent_pick_keyframes(
    raw: KeyframePlanArtifact,
    request: MotionPlanRequest,
    *,
    target_tool_id: str,
) -> KeyframePlanArtifact:
    """Reuse saved VLM geometry for an equivalent reselected tool.

    M4 physical validation may reject the originally selected tool and cause
    M4 to select another instance of the same plate asset.  The VLM's symbolic
    pre/grasp/lift strategy remains valid, so retarget only object references
    after verifying that source and target dimensions match.
    """

    source_ids = {
        keyframe.frame_ref.removeprefix("object:")
        for candidate in raw.candidates
        for keyframe in candidate.keyframes
        if keyframe.frame_ref.startswith("object:")
    }
    if source_ids == {target_tool_id}:
        return raw
    if len(source_ids) != 1:
        raise RuntimeError(
            "saved PICK keyframes must reference exactly one source object; "
            f"found {sorted(source_ids)}"
        )
    source_tool_id = next(iter(source_ids))
    source = request.world.objects.get(source_tool_id)
    target = request.world.objects.get(target_tool_id)
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise RuntimeError(
            "saved PICK source and selected target must both exist in the scene"
        )
    source_dimensions = np.asarray(source.get("dimensions_m"), dtype=float)
    target_dimensions = np.asarray(target.get("dimensions_m"), dtype=float)
    if (
        source_dimensions.shape != (3,)
        or target_dimensions.shape != (3,)
        or not np.allclose(source_dimensions, target_dimensions, atol=1e-3)
    ):
        raise RuntimeError(
            "saved PICK keyframes cannot be retargeted across different geometry"
        )

    bound = raw.model_copy(deep=True)
    for candidate in bound.candidates:
        for keyframe in candidate.keyframes:
            if keyframe.frame_ref == f"object:{source_tool_id}":
                keyframe.frame_ref = f"object:{target_tool_id}"
            metadata = dict(keyframe.metadata)
            if metadata.get("event_target_id") == source_tool_id:
                metadata["event_target_id"] = target_tool_id
            keyframe.metadata = metadata
        candidate.rationale = (
            f"{candidate.rationale} Reused for physically equivalent "
            f"{target_tool_id!r} after {source_tool_id!r} failed M4 retention."
        )
        candidate.metadata = {
            **candidate.metadata,
            "retargeted_from_tool_id": source_tool_id,
            "retargeted_to_tool_id": target_tool_id,
            "retarget_reason": "M4_PHYSICAL_GRASP_REPLAN",
        }
    digest = hashlib.sha256(
        f"{raw.artifact_id}:{source_tool_id}:{target_tool_id}".encode("utf-8")
    ).hexdigest()[:16]
    bound.artifact_id = f"{raw.artifact_id}:retarget:{digest}"
    bound.provenance = raw.provenance.model_copy(
        update={
            "artifact_id": f"{raw.provenance.artifact_id}:retarget:{digest}",
            "metadata": {
                **raw.provenance.metadata,
                "retargeted_from_tool_id": source_tool_id,
                "retargeted_to_tool_id": target_tool_id,
                "retarget_reason": "M4_PHYSICAL_GRASP_REPLAN",
            },
        }
    )
    return bound


def _contextualize_sweep(
    raw: KeyframePlanArtifact,
    compiler: ToolUseJournalCollisionModelCompiler,
    transform: AttachedObjectTransform,
    block_ids: list[str],
    *,
    ee: str,
    tool_id: str,
) -> tuple[KeyframePlanArtifact, dict[str, CollisionContext], str]:
    context_id = "c1_1:sweep:plate-attached"
    allowed = [
        (ee, tool_id),
        (tool_id, "table*"),
        *((tool_id, block_id) for block_id in block_ids),
    ]
    context = CollisionContext(
        context_id=context_id,
        scene_state_id="c1_1:sweep:plate-attached",
        active_ee=ee,
        attached_object_ids=[tool_id],
        attached_object_transforms=[transform],
        touch_links=[ee],
        allowed_collision_pairs=allowed,
        collision_model_version=compiler.model_version_for(ee),
    )
    candidates = [
        candidate.model_copy(
            update={
                "keyframes": [
                    keyframe.model_copy(
                        update={"collision_context_id": context_id}
                    )
                    for keyframe in candidate.keyframes
                ]
            }
        )
        for candidate in raw.candidates
    ]
    artifact = raw.model_copy(
        update={
            "artifact_id": f"{raw.artifact_id}:sweep-context",
            "provenance": raw.provenance.model_copy(
                update={
                    "artifact_id": (
                        f"{raw.provenance.artifact_id}:sweep-context"
                    ),
                    "metadata": {
                        **raw.provenance.metadata,
                        "contextualizer": "C1_1_SWEEP_ATTACHED_TOOL_V1",
                    },
                }
            ),
            "candidates": candidates,
        }
    )
    return artifact, {context_id: context}, context_id


def _plan(
    request: MotionPlanRequest,
    artifact: KeyframePlanArtifact,
    adapter: ToolUseJournalEnvironmentAdapter,
    collision_factory: ToolUseJournalCollisionContextFactory,
):
    setup = collision_factory.prepare(request, artifact)
    pipeline = MotionPlanningPipeline(
        _FrozenProvider(setup.keyframe_artifact), adapter.make_kinematics()
    )
    result = pipeline.plan(
        request,
        state_validator=setup.state_validator,
        collision_contexts=setup.collision_contexts,
        initial_collision_context_id=setup.initial_collision_context_id,
        final_segment_validator=setup.final_segment_validator,
    )
    return result, setup.state_validator


def _run(
    runtime: ToolUseJournalEERuntime,
    plan: object,
    collision_registry: object,
    run_id: str,
    *,
    collision_check_stride: int = 1,
    physical_grasp_monitor: _PhysicalGraspMonitor | None = None,
    render_video: bool = False,
):
    env = runtime.env
    adaptive_settle_budget_s = sum(
        float(segment.metadata.get("tracking_settle", {}).get("max_wait_s", 0.0))
        for segment in plan.segments
        if isinstance(segment.metadata.get("tracking_settle"), dict)
    )
    adaptive_push_budget_s = 0.0
    for segment in plan.segments:
        push_control = segment.metadata.get("physical_push_control")
        if not isinstance(push_control, dict):
            continue
        time_scale = float(push_control["contact_plan_time_scale"])
        segment_duration_s = float(
            segment.end_time_s - segment.start_time_s
        )
        adaptive_push_budget_s += (
            segment_duration_s * (1.0 / time_scale - 1.0)
            + float(push_control["reacquire_timeout_s"])
            * int(push_control["max_reacquire_attempts"])
        )
    simulation_run = SimulationRun(
        run_id=run_id,
        provenance=_provenance(
            f"{run_id}:artifact",
            "SimulationRun",
            ModuleName.SIMULATOR,
            plan.provenance.artifact_id,
        ),
        plan=plan,
        config=SimulationConfig(
            physics_timestep_s=float(env.model_timestep),
            control_timestep_s=float(env.control_timestep),
            realtime_factor=0.0,
            max_duration_s=max(
                30.0,
                float(plan.duration_s)
                + adaptive_settle_budget_s
                + adaptive_push_budget_s
                + 5.0,
            ),
            terminate_on_collision=True,
            render=render_video,
            random_seed=0,
        ),
    )
    player_type = (
        _PhysicalGraspControllerTrajectoryPlayer
        if physical_grasp_monitor is not None
        else ToolUseJournalControllerTrajectoryPlayer
    )
    player_kwargs: dict[str, object] = {
        "collision_probe": collision_registry,
        "collision_check_stride": collision_check_stride,
    }
    if physical_grasp_monitor is not None:
        player_kwargs["monitor"] = physical_grasp_monitor
    report = player_type(runtime, **player_kwargs).execute(simulation_run)
    return simulation_run, report


class _OffscreenVideoRecorder:
    """Capture controller ticks without changing simulated-time execution."""

    def __init__(
        self,
        env: object,
        path: Path,
        *,
        camera: str,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        self.env = env
        self.path = path.resolve()
        self.camera = camera
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self._capture_credit = 0.0
        self._pending_frames: list[bytes] | None = None
        self._transaction_capture_credit: float | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open video writer for {self.path}")
        self.write_current_frame()

    def _write_bgr_frame(self, bgr: np.ndarray) -> None:
        if self._pending_frames is not None:
            encoded, buffer = cv2.imencode(
                ".jpg",
                bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            if not encoded:
                raise RuntimeError("could not buffer a transactional video frame")
            self._pending_frames.append(buffer.tobytes())
            return
        self._writer.write(bgr)
        self.frame_count += 1

    def write_current_frame(self) -> None:
        rgb = self.env.sim.render(  # type: ignore[attr-defined]
            camera_name=self.camera,
            width=self.width,
            height=self.height,
        )[::-1]
        self._write_bgr_frame(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def capture_controller_tick(self) -> None:
        self._capture_credit += self.fps * float(
            self.env.control_timestep  # type: ignore[attr-defined]
        )
        if self._capture_credit + 1e-12 < 1.0:
            return
        self._capture_credit -= 1.0
        self.write_current_frame()

    def hold_final_frame(self, duration_s: float) -> None:
        for _ in range(max(0, round(duration_s * self.fps))):
            self.write_current_frame()

    def begin_transaction(self) -> None:
        if self._pending_frames is not None:
            raise RuntimeError("video transaction is already active")
        self._pending_frames = []
        self._transaction_capture_credit = self._capture_credit

    def commit_transaction(self) -> int:
        if self._pending_frames is None:
            raise RuntimeError("no video transaction is active")
        pending = self._pending_frames
        self._pending_frames = None
        self._transaction_capture_credit = None
        for encoded in pending:
            bgr = cv2.imdecode(
                np.frombuffer(encoded, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if bgr is None:
                raise RuntimeError("could not decode a transactional video frame")
            self._writer.write(bgr)
            self.frame_count += 1
        return len(pending)

    def rollback_transaction(self) -> int:
        if self._pending_frames is None:
            raise RuntimeError("no video transaction is active")
        discarded_count = len(self._pending_frames)
        self._pending_frames = None
        if self._transaction_capture_credit is not None:
            self._capture_credit = self._transaction_capture_credit
        self._transaction_capture_credit = None
        return discarded_count

    def close(self) -> None:
        if self._pending_frames is not None:
            self.rollback_transaction()
        self._writer.release()


def _configure_fixed_camera(
    env: object,
    camera: str,
    *,
    eye_m: Sequence[float] | None,
    look_at_m: Sequence[float] | None,
    fovy_deg: float | None,
) -> None:
    model: mujoco.MjModel = env.sim.model._model  # type: ignore[attr-defined]
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if camera_id < 0:
        raise RuntimeError(f"camera {camera!r} is absent")
    if eye_m is not None and look_at_m is not None:
        eye = np.asarray(eye_m, dtype=float)
        target = np.asarray(look_at_m, dtype=float)
        forward = target - eye
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm <= 1e-9:
            raise ValueError("camera eye and look-at point must differ")
        forward /= forward_norm
        world_up = np.asarray((0.0, 0.0, 1.0), dtype=float)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
        if right_norm <= 1e-9:
            raise ValueError("camera viewing direction cannot be vertical")
        right /= right_norm
        up = np.cross(right, forward)
        rotation = np.column_stack((right, up, -forward))
        quaternion_wxyz = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(
            quaternion_wxyz,
            np.ascontiguousarray(rotation.reshape(9)),
        )
        model.cam_pos[camera_id] = eye
        model.cam_quat[camera_id] = quaternion_wxyz
    if fovy_deg is not None:
        model.cam_fovy[camera_id] = fovy_deg


def _block_positions(
    env: object, target_ids: Sequence[str]
) -> dict[str, list[float]]:
    data = env.sim.data._data  # type: ignore[attr-defined]
    missing = set(target_ids) - set(env.obj_body_id)  # type: ignore[attr-defined]
    if missing:
        raise RuntimeError(
            f"Task Planner targets are absent from runtime: {sorted(missing)}"
        )
    return {
        object_id: [
            round(
                float(data.xpos[env.obj_body_id[object_id], axis]),  # type: ignore[attr-defined]
                6,
            )
            for axis in range(3)
        ]
        for object_id in target_ids
    }


@dataclass(frozen=True)
class _SimulatorCheckpoint:
    state: object
    arrays: dict[str, np.ndarray]
    env_timestep: int | None
    env_cur_time: float | None


def _simulator_checkpoint(env: object) -> _SimulatorCheckpoint:
    """Capture all dynamic MuJoCo state needed for a retryable replay."""

    sim = env.sim  # type: ignore[attr-defined]
    data = sim.data._data  # type: ignore[attr-defined]
    arrays = {
        name: np.asarray(getattr(data, name), dtype=float).copy()
        for name in (
            "act",
            "ctrl",
            "mocap_pos",
            "mocap_quat",
            "qacc_warmstart",
            "userdata",
        )
        if getattr(data, name, None) is not None
    }
    return _SimulatorCheckpoint(
        state=sim.get_state(),
        arrays=arrays,
        env_timestep=(
            int(env.timestep)  # type: ignore[attr-defined]
            if hasattr(env, "timestep")
            else None
        ),
        env_cur_time=(
            float(env.cur_time)  # type: ignore[attr-defined]
            if hasattr(env, "cur_time")
            else None
        ),
    )


def _restore_simulator_checkpoint(
    env: object, checkpoint: _SimulatorCheckpoint
) -> None:
    """Restore a rejected controller replay to its exact start state."""

    sim = env.sim  # type: ignore[attr-defined]
    data = sim.data._data  # type: ignore[attr-defined]
    sim.set_state(checkpoint.state)
    for name, values in checkpoint.arrays.items():
        target = getattr(data, name, None)
        if target is not None and np.asarray(target).shape == values.shape:
            target[...] = values
    if checkpoint.env_timestep is not None:
        env.timestep = checkpoint.env_timestep  # type: ignore[attr-defined]
    if checkpoint.env_cur_time is not None:
        env.cur_time = checkpoint.env_cur_time  # type: ignore[attr-defined]
    sim.forward()
    update_observables = getattr(env, "_update_observables", None)
    if callable(update_observables):
        update_observables()


def _angularly_order_sweep_targets(
    world: object,
    target_ids: Sequence[str],
    goal_region_id: str,
) -> list[str]:
    """Order targets by observed polar angle to avoid large wrist reversals."""

    return list(order_targets_around_region(world, target_ids, goal_region_id))


def _inside_goal_region(
    env: object,
    world: object,
    target_ids: Sequence[str],
    goal_region_id: str,
) -> list[str]:
    data = env.sim.data._data  # type: ignore[attr-defined]
    region = world.objects.get(goal_region_id)
    if not isinstance(region, dict):
        raise RuntimeError(f"goal region {goal_region_id!r} is missing from world")
    zone = np.asarray(region["pose"]["position_m"], dtype=float)[:2]
    size = np.asarray(region.get("dimensions_m"), dtype=float)[:2]
    if zone.shape != (2,) or size.shape != (2,) or not np.all(size > 0.0):
        raise RuntimeError(f"goal region {goal_region_id!r} has invalid geometry")
    missing = set(target_ids) - set(env.obj_body_id)  # type: ignore[attr-defined]
    if missing:
        raise RuntimeError(
            f"Task Planner sweep targets are absent from runtime: {sorted(missing)}"
        )
    inside = []
    for block_id in target_ids:
        body_id = env.obj_body_id[block_id]  # type: ignore[attr-defined]
        position = np.asarray(data.xpos[body_id, :2], dtype=float)
        block = world.objects.get(block_id)
        block_dimensions = (
            np.asarray(block.get("dimensions_m"), dtype=float)[:2]
            if isinstance(block, Mapping)
            and block.get("dimensions_m") is not None
            else np.zeros(2, dtype=float)
        )
        available_half_size = size / 2.0 - block_dimensions / 2.0
        if np.all(available_half_size >= 0.0) and np.all(
            np.abs(position - zone) <= available_half_size
        ):
            inside.append(block_id)
    return inside


def _adaptive_micro_push_distance_m(
    world: object,
    *,
    block_id: str,
    goal_region_id: str,
    maximum_distance_m: float,
    inset_margin_m: float,
) -> float:
    """Limit the last radial push to the remaining full-block goal distance."""

    block = world.objects.get(block_id)
    region = world.objects.get(goal_region_id)
    if not isinstance(block, Mapping) or not isinstance(region, Mapping):
        raise RuntimeError("adaptive push geometry is missing")
    block_xy = np.asarray(block["pose"]["position_m"], dtype=float)[:2]
    region_xy = np.asarray(region["pose"]["position_m"], dtype=float)[:2]
    block_size = np.asarray(block["dimensions_m"], dtype=float)[:2]
    region_size = np.asarray(region["dimensions_m"], dtype=float)[:2]
    outward = block_xy - region_xy
    radius_m = float(np.linalg.norm(outward))
    if radius_m <= 1e-9:
        return min(maximum_distance_m, inset_margin_m)
    outward /= radius_m
    usable_half_size = region_size / 2.0 - block_size / 2.0
    if np.any(usable_half_size <= inset_margin_m):
        raise RuntimeError("goal region is too small for the selected block")
    radial_limits = [
        float(usable_half_size[axis] / abs(outward[axis]))
        for axis in range(2)
        if abs(float(outward[axis])) > 1e-9
    ]
    boundary_radius_m = min(radial_limits)
    remaining_m = max(0.0, radius_m - boundary_radius_m)
    return min(
        maximum_distance_m,
        max(inset_margin_m, remaining_m + inset_margin_m),
    )


def _cleanup_block_ids(
    block_ids: Sequence[str],
    *,
    inside_ids: Sequence[str],
    initial_positions_m: Mapping[str, Sequence[float]],
    current_positions_m: Mapping[str, Sequence[float]],
    max_support_error_m: float,
) -> list[str]:
    """Return blocks displaced from the goal or their original table support."""

    if max_support_error_m < 0.0:
        raise ValueError("max_support_error_m must be non-negative")
    inside = set(inside_ids)
    cleanup: list[str] = []
    for block_id in block_ids:
        initial = np.asarray(initial_positions_m[block_id], dtype=float)
        current = np.asarray(current_positions_m[block_id], dtype=float)
        if initial.shape != (3,) or current.shape != (3,):
            raise ValueError("block positions must be 3D")
        support_error_m = abs(float(current[2] - initial[2]))
        if block_id not in inside or support_error_m > max_support_error_m:
            cleanup.append(block_id)
    return cleanup


def _reduced_micro_push_limit_m(
    current_distance_m: float,
    *,
    minimum_distance_m: float,
    retry_scale: float,
) -> float:
    """Back off a failed contact-continuation step without abandoning contact."""

    return reduced_contact_step_distance(
        current_distance_m,
        minimum_distance_m=minimum_distance_m,
        retry_scale=retry_scale,
    )


def _physical_push_rejection_reason(
    *,
    execution_succeeded: bool,
    reached_goal: bool,
    target_contact_sample_count: int,
    radial_progress_m: float,
    minimum_progress_m: float,
    block_support_lift_m: float,
    maximum_block_lift_m: float,
) -> str | None:
    """Return why an observed physical push must be rolled back, if any.

    The decision deliberately uses task-independent execution evidence: a
    motion is committed only when the controller succeeded and the observed
    object state either reached its goal or made supported, contact-backed
    progress toward it.
    """

    if not execution_succeeded:
        return "EXECUTION_FAILED"
    if abs(block_support_lift_m) > maximum_block_lift_m:
        return "EXCESSIVE_BLOCK_LIFT"
    if reached_goal:
        return None
    if target_contact_sample_count <= 0:
        return "NO_TARGET_CONTACT"
    if radial_progress_m < minimum_progress_m:
        return "INSUFFICIENT_GOAL_PROGRESS"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("--task-planner", type=Path, required=True)
    parser.add_argument(
        "--pick-keyframes",
        type=Path,
        help=(
            "Reuse an existing raw PICK keyframe artifact instead of requesting "
            "new OpenAI output."
        ),
    )
    parser.add_argument(
        "--pick-candidate-limit",
        type=int,
        help="Limit reused PICK candidates for focused physical-grasp validation.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--planning-time",
        type=float,
        default=12.0,
        help="Per-strategy connected-path planning budget in seconds.",
    )
    parser.add_argument(
        "--rrt-max-iterations",
        type=int,
        default=5000,
        help="Maximum RRT iterations per connection attempt.",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--environment", default="C1_1_LegoSweep")
    parser.add_argument("--camera", default="agentview")
    parser.add_argument(
        "--video",
        type=Path,
        help="Record the physical closed-loop execution directly to MP4.",
    )
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--video-hold-seconds", type=float, default=4.0)
    parser.add_argument(
        "--camera-eye",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Override the fixed camera world position in metres.",
    )
    parser.add_argument(
        "--camera-look-at",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="World point aimed at by --camera-eye.",
    )
    parser.add_argument("--camera-fovy", type=float)
    parser.add_argument(
        "--motion-profile",
        type=Path,
        help="Optional JSON overrides for M4-owned C1 geometry and settle values.",
    )
    parser.add_argument(
        "--save-physical-traces",
        action="store_true",
        help=(
            "Persist per-tick physical grasp and push traces. Disabled by "
            "default because a full C1_1 run can exceed 500 MB."
        ),
    )
    parser.add_argument(
        "--stop-after-pick",
        action="store_true",
        help="Replay and report only the OpenAI-generated selected-tool pick plan.",
    )
    parser.add_argument(
        "--validate-input-only",
        action="store_true",
        help=(
            "Validate the M4 Task Planner binding without OpenAI or MuJoCo "
            "execution."
        ),
    )
    parser.add_argument(
        "--sweep-provider",
        choices=("task-geometry", "openai"),
        default="task-geometry",
        help=(
            "Use MJCF-derived radial block pushes or request general sweep "
            "keyframes."
        ),
    )
    parser.add_argument(
        "--sweep-target-ids",
        nargs="+",
        help=(
            "Run a staged sweep on this Task-Planner target subset. The IDs "
            "must already be present in the selected M4 assignment."
        ),
    )
    parser.add_argument(
        "--execution-collision-check-stride",
        type=int,
        default=5,
        help=(
            "Check the already collision-validated controller trajectory every "
            "N control ticks during execution (segment boundaries and the final "
            "tick are always checked)."
        ),
    )
    parser.add_argument(
        "--pick-controller-kp",
        type=float,
        default=50.0,
        help="Selected-tool pick-phase joint-position proportional gain.",
    )
    parser.add_argument(
        "--pick-controller-damping-ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--controller-kp",
        type=float,
        default=80.0,
        help=(
            "Sweep-phase absolute joint-position proportional gain."
        ),
    )
    parser.add_argument(
        "--controller-damping-ratio",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()
    if args.planning_time <= 0.0:
        parser.error("--planning-time must be positive")
    if args.rrt_max_iterations <= 0:
        parser.error("--rrt-max-iterations must be positive")
    if args.execution_collision_check_stride <= 0:
        parser.error("--execution-collision-check-stride must be positive")
    if args.pick_candidate_limit is not None and args.pick_candidate_limit <= 0:
        parser.error("--pick-candidate-limit must be positive")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.video_fps <= 0.0:
        parser.error("--video-fps must be positive")
    if args.video_hold_seconds < 0.0:
        parser.error("--video-hold-seconds must be non-negative")
    if (args.camera_eye is None) != (args.camera_look_at is None):
        parser.error("--camera-eye and --camera-look-at must be supplied together")
    if args.camera_fovy is not None and not 1.0 < args.camera_fovy < 179.0:
        parser.error("--camera-fovy must be between 1 and 179 degrees")
    try:
        motion_profile = _load_motion_profile(args.motion_profile)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"invalid --motion-profile: {exc}")

    repository = args.repository.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    planning_result = PlanningResult.model_validate_json(
        args.task_planner.read_text(encoding="utf-8")
    )
    if planning_result.selected_plan is None:
        raise RuntimeError("Task Planner did not select a plan")
    selected = planning_result.selected_plan
    pick_binding = _selected_pick_binding(selected)
    sweep_binding = _selected_sweep_binding(selected)
    if (pick_binding.ee, pick_binding.tool) != (
        sweep_binding.ee,
        sweep_binding.tool,
    ):
        raise RuntimeError(
            "C1_1 pick and sweep must use the same M4-selected EE and tool; "
            f"pick={(pick_binding.ee, pick_binding.tool)}, "
            f"sweep={(sweep_binding.ee, sweep_binding.tool)}"
        )
    if pick_binding.ee != "2F":
        raise RuntimeError(
            "the C1_1 rim-grasp geometry binder currently supports only 2F; "
            f"M4 selected {pick_binding.ee!r}"
        )

    selected_target_ids = list(sweep_binding.target_ids)
    if args.sweep_target_ids is not None:
        requested = list(dict.fromkeys(args.sweep_target_ids))
        unknown = sorted(set(requested) - set(selected_target_ids))
        if unknown:
            parser.error(
                "--sweep-target-ids contains IDs not selected by M4: "
                f"{unknown}"
            )
        selected_target_ids = requested

    if args.validate_input_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "environment": args.environment,
                    "selected_ee": pick_binding.ee,
                    "selected_tool": pick_binding.tool,
                    "pick_subgoal_id": pick_binding.subgoal_id,
                    "sweep": {
                        "subgoal_id": sweep_binding.subgoal_id,
                        "target_ids": selected_target_ids,
                        "goal_region_id": sweep_binding.goal_region_id,
                    },
                    "all_sweep_target_ids": selected_target_ids,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    runtime = ToolUseJournalEERuntime.from_repository_for_controller(
        repository,
        args.environment,
        active_ee=pick_binding.ee,
        seed=args.seed,
        joint_position_kp=args.pick_controller_kp,
        joint_position_damping_ratio=args.pick_controller_damping_ratio,
        ignore_done=True,
        # Camera observations would render a 512x512 image at every physics
        # control tick even though validation only needs the final frame.
        use_camera_obs=False,
        has_offscreen_renderer=True,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
        render_camera=args.camera,
    )
    reports = []
    plans = []
    video_recorder: _OffscreenVideoRecorder | None = None
    try:
        _configure_fixed_camera(
            runtime.env,
            args.camera,
            eye_m=args.camera_eye,
            look_at_m=args.camera_look_at,
            fovy_deg=args.camera_fovy,
        )
        if args.video is not None:
            video_recorder = _OffscreenVideoRecorder(
                runtime.env,
                args.video,
                camera=args.camera,
                width=args.width,
                height=args.height,
                fps=args.video_fps,
            )
            runtime.env.render = video_recorder.capture_controller_tick
        ee_pool = getattr(runtime.env, "robot_spec", {}).get("ee_pool", [])
        selected_ee_spec = next(
            (
                record
                for record in ee_pool
                if isinstance(record, dict)
                and record.get("ee_id") == pick_binding.ee
            ),
            None,
        )
        if not isinstance(selected_ee_spec, dict) or not isinstance(
            selected_ee_spec.get("grip_force_n"), (int, float)
        ):
            raise RuntimeError(
                f"M1 robot spec is missing grip_force_n for {pick_binding.ee}"
            )
        fingerpad_friction = selected_ee_spec.get("fingerpad_friction")
        if (
            not isinstance(fingerpad_friction, (list, tuple))
            or len(fingerpad_friction) != 3
        ):
            raise RuntimeError(
                f"M1 robot spec is missing fingerpad_friction for "
                f"{pick_binding.ee}"
            )
        runtime.set_finger_gripper_contact_friction(fingerpad_friction)
        tool_metadata = runtime.env.get_tool_physical_metadata(  # type: ignore[attr-defined]
            pick_binding.tool
        )
        full_size_mm = tool_metadata.get("full_size_mm")
        if not isinstance(full_size_mm, (list, tuple)) or len(full_size_mm) != 3:
            raise RuntimeError(
                f"M1 tool metadata is missing full_size_mm for {pick_binding.tool}"
            )
        target_preshape_aperture_m = (
            float(full_size_mm[2]) / 1000.0
            + motion_profile.pick_preshape_clearance_m
        )
        tool_mass_kg = tool_metadata.get("mass_kg")
        tool_friction = tool_metadata.get("friction")
        if (
            not isinstance(tool_mass_kg, (int, float))
            or not isinstance(tool_friction, (list, tuple))
            or not tool_friction
            or not isinstance(tool_friction[0], (int, float))
            or float(tool_mass_kg) <= 0.0
            or float(tool_friction[0]) <= 0.0
        ):
            raise RuntimeError(
                f"M1 tool metadata is missing mass/friction for {pick_binding.tool}"
            )
        retention_target_normal_force_n = _rim_grasp_retention_force_n(
            mass_kg=float(tool_mass_kg),
            sliding_friction=float(tool_friction[0]),
            dimensions_m=[float(value) / 1000.0 for value in full_size_mm],
            safety_factor=motion_profile.pick_grip_force_safety_factor,
            max_grip_force_n=float(selected_ee_spec["grip_force_n"]),
        )
        preshape_aperture_m = (
            ToolUseJournalControllerTrajectoryPlayer(runtime)
            .preshape_finger_gripper_to_aperture(
                target_aperture_m=target_preshape_aperture_m,
                tolerance_m=motion_profile.pick_preshape_tolerance_m,
                final_settle_ticks=motion_profile.pick_preshape_settle_ticks,
            )
        )
        adapter = ToolUseJournalEnvironmentAdapter(runtime.env)
        adapter.require_physical_ee(pick_binding.ee)
        world = adapter.world_snapshot()
        pick_grasp = pick_binding.grasp or GraspSpec(
            grasp_id=f"openai:c1_1:{pick_binding.tool}",
            owner_kind="tool",
            owner_id=pick_binding.tool,
            source="openai_keyframe_provider",
        )
        pick_task = MotionTask(
            task_id=f"c1_1:pick-tool:{pick_binding.tool}",
            subgoal_id=pick_binding.subgoal_id,
            action_type="PICK",
            ee=pick_binding.ee,
            tool=pick_binding.tool,
            target_ids=list(pick_binding.target_ids),
            grasp=pick_grasp,
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_object_id=pick_binding.tool,
                approach_distance_m=motion_profile.pick_approach_distance_m,
                retreat_distance_m=motion_profile.pick_retreat_distance_m,
            ),
            allowed_touch_objects=[pick_binding.tool],
            metadata={
                "task_planner_subgoal": pick_binding.subgoal_id,
                "source_action_type": pick_binding.action_type,
                "source_mode": pick_binding.mode,
                "operation": "PICK_TOOL",
                "attach_target": True,
                "grasp_execution_mode": "CONTACT_FRICTION",
            },
        )
        pick_request = _request(
            request_id=f"c1_1:motion-request:pick-tool:{pick_binding.tool}",
            world=world,
            task=pick_task,
            seed=args.seed,
            task_planner_artifact=args.task_planner,
            allowed_planning_time_s=args.planning_time,
            rrt_max_iterations=args.rrt_max_iterations,
        )
        pick_request.constraints = pick_request.constraints.model_copy(
            update={
                "velocity_scaling": motion_profile.pick_velocity_scaling,
                "acceleration_scaling": (
                    motion_profile.pick_acceleration_scaling
                ),
                "jerk_scaling": motion_profile.pick_jerk_scaling,
            }
        )
        pick_request.constraints.allowed_collision_pairs = [
            (pick_binding.ee, "table*"),
            (pick_binding.tool, "table*"),
        ]
        _write_model(output / "pick_request.json", pick_request)
        raw_pick = (
            KeyframePlanArtifact.model_validate_json(
                args.pick_keyframes.read_text(encoding="utf-8")
            )
            if args.pick_keyframes is not None
            else _openai_artifact(
                pick_request,
                model=args.model,
                candidates=args.candidates,
                cache_dir=output / "keyframe_cache",
            )
        )
        if raw_pick.scene_signature != pick_request.world.scene.signature:
            if args.pick_keyframes is None:
                raise RuntimeError("generated PICK keyframes belong to another scene")
            source_scene_signature = raw_pick.scene_signature
            rebase_digest = hashlib.sha256(
                (
                    source_scene_signature
                    + ":"
                    + pick_request.world.scene.signature
                ).encode("utf-8")
            ).hexdigest()[:12]
            raw_pick = raw_pick.model_copy(
                update={
                    "artifact_id": f"{raw_pick.artifact_id}:rebase:{rebase_digest}",
                    "scene_signature": pick_request.world.scene.signature,
                    "provenance": raw_pick.provenance.model_copy(
                        update={
                            "artifact_id": (
                                f"{raw_pick.provenance.artifact_id}:rebase:"
                                f"{rebase_digest}"
                            ),
                            "metadata": {
                                **raw_pick.provenance.metadata,
                                "scene_signature_rebase_reason": (
                                    "PHYSICAL_GRIPPER_PRESHAPE"
                                ),
                                "source_scene_signature": (
                                    source_scene_signature
                                ),
                                "rebased_scene_signature": (
                                    pick_request.world.scene.signature
                                ),
                            },
                        }
                    ),
                }
            )
        if raw_pick.subgoal_id != pick_request.task.subgoal_id:
            if args.pick_keyframes is None:
                raise RuntimeError(
                    "generated PICK keyframes belong to another subgoal"
                )
            source_subgoal_id = raw_pick.subgoal_id
            raw_pick = raw_pick.model_copy(
                update={
                    "subgoal_id": pick_request.task.subgoal_id,
                    "provenance": raw_pick.provenance.model_copy(
                        update={
                            "metadata": {
                                **raw_pick.provenance.metadata,
                                "source_subgoal_id": source_subgoal_id,
                                "rebound_subgoal_id": (
                                    pick_request.task.subgoal_id
                                ),
                            }
                        }
                    ),
                }
            )
        if args.pick_candidate_limit is not None:
            raw_pick = raw_pick.model_copy(
                update={
                    "candidates": raw_pick.candidates[: args.pick_candidate_limit]
                }
            )
            if not raw_pick.candidates:
                raise RuntimeError("PICK keyframe artifact has no candidates")
        _write_model(output / "pick_keyframes_source.json", raw_pick)
        raw_pick = _retarget_equivalent_pick_keyframes(
            raw_pick,
            pick_request,
            target_tool_id=pick_binding.tool,
        )
        _write_model(output / "pick_keyframes_raw.json", raw_pick)
        rim_pick = _bind_2f_plate_side_grasps(
            raw_pick,
            pick_request,
            tool_id=pick_binding.tool,
            radial_inset_m=motion_profile.pick_side_grasp_radial_inset_m,
            vertical_offset_m=(
                motion_profile.pick_side_grasp_vertical_offset_m
            ),
            lift_distance_m=motion_profile.pick_side_grasp_lift_m,
            seat_start_lift_m=(
                motion_profile.pick_side_grasp_seat_start_lift_m
            ),
            seat_descent_m=motion_profile.pick_side_grasp_seat_descent_m,
            seat_radial_inset_m=(
                motion_profile.pick_side_grasp_seat_radial_inset_m
            ),
            lateral_compensation_m=0.0,
            approach_elevation_rad=(
                motion_profile.pick_side_grasp_approach_elevation_rad
            ),
            roll_rad=motion_profile.pick_side_grasp_roll_rad,
            side_variant_index=motion_profile.pick_side_grasp_variant_index,
        )
        _write_model(output / "pick_keyframes_side_bound.json", rim_pick)
        pick_compiler = ToolUseJournalCollisionModelCompiler.from_repository(
            runtime.env,
            repository,
            seed=args.seed,
            ignore_done=True,
            use_camera_obs=False,
            has_offscreen_renderer=False,
        )
        pick_artifact = _with_contact_friction_grasp(
            rim_pick,
            hold_duration_s=motion_profile.pick_grasp_hold_s,
            lift_hold_duration_s=motion_profile.pick_final_hold_s,
            gripper_close_rate=motion_profile.pick_gripper_close_rate,
        )
        reference_kind, reference_name, _, _ = runtime._grasp_reference(
            runtime.env
        )
        pick_factory = ToolUseJournalCollisionContextFactory(
            pick_compiler,
            attachment_reference_name=reference_name,
            attachment_reference_kind=reference_kind,
        )
        pick_result, pick_registry = _plan(
            pick_request,
            pick_artifact,
            adapter,
            pick_factory,
        )
        _write_model(
            output / "pick_keyframes_contextualized.json",
            pick_result.keyframe_artifact,
        )
        pick_plan = pick_result.plan
        plans.append(pick_plan)
        _write_model(output / "pick_motion_plan.json", pick_plan)
        plate_body_id = runtime.env.obj_body_id[pick_binding.tool]
        initial_plate_position = tuple(
            float(value)
            for value in runtime.env.sim.data._data.xpos[plate_body_id]
        )
        plate_record = world.objects.get(pick_binding.tool)
        if not isinstance(plate_record, dict):
            raise RuntimeError(f"{pick_binding.tool} is missing from the world")
        plate_dimensions = tuple(
            float(value) for value in plate_record.get("dimensions_m", ())
        )
        if len(plate_dimensions) != 3:
            raise RuntimeError(
                f"{pick_binding.tool} dimensions are missing or invalid"
            )
        initial_plate_rotation = np.asarray(
            runtime.env.sim.data._data.xmat[plate_body_id], dtype=float
        ).reshape(3, 3)
        table_surface_z_m = (
            initial_plate_position[2]
            - _circular_plate_vertical_half_extent_m(
                initial_plate_rotation,
                plate_dimensions,
            )
        )
        physical_grasp_monitor = _PhysicalGraspMonitor(
            runtime=runtime,
            object_id=pick_binding.tool,
            initial_position_m=initial_plate_position,
            min_lift_m=motion_profile.pick_min_lift_m,
            object_dimensions_m=plate_dimensions,
            table_surface_z_m=table_surface_z_m,
            min_bottom_clearance_m=(
                motion_profile.pick_min_bottom_clearance_m
            ),
            required_final_hold_s=motion_profile.pick_final_hold_s,
            contact_loss_grace_s=motion_profile.pick_contact_loss_grace_s,
            required_contact_ticks=motion_profile.pick_required_contact_ticks,
            contact_freeze_ticks=motion_profile.pick_contact_freeze_ticks,
            closure_actuator_kp=(
                motion_profile.pick_gripper_closure_actuator_kp
            ),
            max_closure_actuator_kp=(
                motion_profile.pick_gripper_max_actuator_kp
            ),
            force_feedback_gain=(
                motion_profile.pick_gripper_force_feedback_gain
            ),
            retention_target_normal_force_n=retention_target_normal_force_n,
            max_grip_force_n=float(selected_ee_spec["grip_force_n"]),
            sliding_friction=float(fingerpad_friction[0]),
            min_contact_separation_m=(
                motion_profile.pick_min_contact_separation_m
            ),
            min_normal_opposition=(
                motion_profile.pick_min_normal_opposition
            ),
            max_friction_utilization=(
                motion_profile.pick_max_friction_utilization
            ),
            contact_follow_gain=motion_profile.pick_contact_follow_gain,
            contact_follow_max_m=motion_profile.pick_contact_follow_max_m,
            contact_follow_activation_ticks=(
                motion_profile.pick_contact_follow_activation_ticks
            ),
            contact_follow_max_tick_m=(
                motion_profile.pick_contact_follow_max_tick_m
            ),
            contact_follow_max_joint_step_rad=(
                motion_profile.pick_contact_follow_max_joint_step_rad
            ),
            regrasp_roll_rad=motion_profile.pick_regrasp_roll_rad,
            regrasp_roll_rate_rad_s=(
                motion_profile.pick_regrasp_roll_rate_rad_s
            ),
            regrasp_min_separation_ratio=(
                motion_profile.pick_regrasp_min_separation_ratio
            ),
        )
        pick_run, pick_report = _run(
            runtime,
            pick_plan,
            pick_registry,
            f"c1_1:controller:pick-tool:{pick_binding.tool}",
            collision_check_stride=args.execution_collision_check_stride,
            physical_grasp_monitor=physical_grasp_monitor,
            render_video=video_recorder is not None,
        )
        _write_model(output / "pick_simulation_run.json", pick_run)
        _write_model(output / "pick_execution_report.json", pick_report)
        grasp_validation = physical_grasp_monitor.summary()
        if args.save_physical_traces:
            (output / "pick_physical_grasp_trace.json").write_text(
                json.dumps(
                    physical_grasp_monitor.samples,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        (output / "pick_physical_grasp_validation.json").write_text(
            json.dumps(grasp_validation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        reports.append(pick_report)
        runtime.set_joint_position_controller_gains(
            kp=args.controller_kp,
            damping_ratio=args.controller_damping_ratio,
        )

        adapter = ToolUseJournalEnvironmentAdapter(runtime.env)
        if args.stop_after_pick:
            frame = runtime.env.sim.render(
                camera_name=args.camera, height=args.height, width=args.width
            )[::-1]
            final_frame = output / "pick_controller_final.png"
            cv2.imwrite(str(final_frame), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if video_recorder is not None:
                video_recorder.hold_final_frame(args.video_hold_seconds)
            summary = {
                "status": (
                    "SUCCESS"
                    if pick_report.status.value == "SUCCESS"
                    and grasp_validation["status"] == "SUCCESS"
                    else "FAILED"
                ),
                "phase": "PICK",
                "selected_ee": pick_binding.ee,
                "selected_tool": pick_binding.tool,
                "selected_pick_subgoal": pick_binding.subgoal_id,
                "model": args.model,
                "candidate_count": len(raw_pick.candidates),
                "seed": args.seed,
                "pick_controller_kp": args.pick_controller_kp,
                "pick_controller_damping_ratio": (
                    args.pick_controller_damping_ratio
                ),
                "preshape_aperture_m": preshape_aperture_m,
                "motion_profile": asdict(motion_profile),
                "plan_id": pick_plan.plan_id,
                "plan_duration_s": pick_plan.duration_s,
                "trajectory_waypoint_count": sum(
                    len(segment.waypoints) for segment in pick_plan.segments
                ),
                "execution_status": pick_report.status.value,
                "grasp_mode": "CONTACT_FRICTION",
                "weld_absent": runtime.attachment is None,
                "final_attached_object_id": runtime.attached_object_id,
                "physical_grasp_validation": grasp_validation,
                "last_attachment_break": (
                    runtime.last_attachment_break.as_mapping()
                    if runtime.last_attachment_break is not None
                    else None
                ),
                "artifacts": {
                    "pick_plan": str((output / "pick_motion_plan.json").resolve()),
                    "pick_report": str(
                        (output / "pick_execution_report.json").resolve()
                    ),
                    "final_frame": str(final_frame.resolve()),
                    "physical_grasp_trace": (
                        str(
                            (output / "pick_physical_grasp_trace.json").resolve()
                        )
                        if args.save_physical_traces
                        else None
                    ),
                    "physical_grasp_validation": str(
                        (output / "pick_physical_grasp_validation.json").resolve()
                    ),
                    "video": (
                        str(video_recorder.path)
                        if video_recorder is not None
                        else None
                    ),
                },
            }
            (output / "pick_run_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if summary["status"] == "SUCCESS" else 2
        if pick_report.status.value != "SUCCESS":
            raise RuntimeError(
                f"pick controller replay failed: {pick_report.failure}"
            )
        if grasp_validation["status"] != "SUCCESS":
            raise RuntimeError(
                "contact-friction pick did not satisfy physical grasp validation"
            )
        (
            physical_grasp_offset,
            physical_grasp_rotation,
        ) = _physical_grasp_transform_in_reference(
            runtime,
            sweep_binding.tool,
        )
        block_ids = list(sweep_binding.target_ids)
        if args.sweep_target_ids:
            requested_targets = list(dict.fromkeys(args.sweep_target_ids))
            unknown_targets = sorted(set(requested_targets) - set(block_ids))
            if unknown_targets:
                raise RuntimeError(
                    "staged sweep targets are absent from the selected M4 "
                    f"assignment: {unknown_targets}"
                )
            block_ids = requested_targets
        # The plate stays a free MuJoCo body. Every micro-plan receives a new
        # live world snapshot and a newly measured friction-grasp transform.
        world = adapter.world_snapshot()
        if not args.sweep_target_ids:
            block_ids = _angularly_order_sweep_targets(
                world,
                block_ids,
                sweep_binding.goal_region_id,
            )
        initial_block_positions = _block_positions(runtime.env, block_ids)
        zone_pose = Pose.model_validate(
            world.objects[sweep_binding.goal_region_id]["pose"]
        )
        sweep_action_type = sweep_binding.action_type or "tool_act:sweep"
        if (
            sweep_binding.mode
            and ":" not in sweep_action_type
            and sweep_binding.mode.lower() not in sweep_action_type.lower()
        ):
            sweep_action_type = f"{sweep_action_type}:{sweep_binding.mode}"
        zone_xy = np.asarray(zone_pose.position_m, dtype=float)[:2]
        micro_push_records: list[dict[str, object]] = []
        sweep_reports = []
        sweep_artifact_paths: list[dict[str, str]] = []
        block_queue = list(block_ids)
        block_cursor = 0
        cleanup_passes_completed = 0
        cleanup_history: list[dict[str, object]] = []
        executed_block_sequence: list[str] = []
        while block_cursor < len(block_queue):
            block_id = block_queue[block_cursor]
            block_cursor += 1
            executed_block_sequence.append(block_id)
            alignment_variant_offset = 0
            continue_from_contact = False
            planar_recovery = False
            approach_standoff_m = motion_profile.sweep_start_offset_m
            attempt_index = max(
                (
                    int(record["attempt"])
                    for record in micro_push_records
                    if record.get("block_id") == block_id
                ),
                default=0,
            )
            physical_attempt_count = 0
            planning_retry_count = 0
            committed_push_count = 0
            micro_push_limit_m = motion_profile.sweep_micro_push_distance_m
            # A semantic VLM route can occasionally be infeasible even though
            # the grounded broad-face pose at the same normal deviation is
            # reachable.  In that case, retry the same grounded geometry once
            # with the deterministic task-geometry route before moving to the
            # next bounded normal deviation.
            force_task_geometry_retry = False
            while (
                physical_attempt_count
                < motion_profile.sweep_micro_push_max_attempts_per_block
                and planning_retry_count
                < motion_profile.sweep_max_planning_retries_per_block
            ):
                attempt_index += 1
                world = adapter.world_snapshot()
                world.robot_state = world.robot_state.model_copy(
                    update={"held_tool_id": sweep_binding.tool}
                )
                world.metadata["held_tool"] = sweep_binding.tool
                world.metadata["attachment_mode"] = "CONTACT_FRICTION"
                block_is_inside = bool(_inside_goal_region(
                    runtime.env,
                    world,
                    (block_id,),
                    sweep_binding.goal_region_id,
                ))
                live_block_position = np.asarray(
                    _block_positions(runtime.env, (block_id,))[block_id],
                    dtype=float,
                )
                support_error_m = abs(
                    float(
                        live_block_position[2]
                        - initial_block_positions[block_id][2]
                    )
                )
                if (
                    block_is_inside
                    and support_error_m <= motion_profile.sweep_max_block_lift_m
                ):
                    break

                before_position = np.asarray(
                    _block_positions(runtime.env, (block_id,))[block_id],
                    dtype=float,
                )
                before_radius_m = float(
                    np.linalg.norm(before_position[:2] - zone_xy)
                )
                requested_micro_push_distance_m = (
                    _adaptive_micro_push_distance_m(
                        world,
                        block_id=block_id,
                        goal_region_id=sweep_binding.goal_region_id,
                        maximum_distance_m=(
                            micro_push_limit_m
                        ),
                        inset_margin_m=(
                            motion_profile.sweep_goal_inset_margin_m
                        ),
                    )
                )
                (
                    physical_grasp_offset,
                    physical_grasp_rotation,
                ) = _physical_grasp_transform_in_reference(
                    runtime,
                    sweep_binding.tool,
                )
                for variant_index, alignment_blend in enumerate(
                    _sweep_alignment_blends(motion_profile)
                ):
                    (
                        _,
                        _,
                        current_eef_position,
                        current_eef_rotation,
                    ) = runtime._grasp_reference(runtime.env)
                    _install_sweep_reference_frames(
                        world,
                        target_ids=(block_id,),
                        goal_region_id=sweep_binding.goal_region_id,
                        tool_id=sweep_binding.tool,
                        attachment_position_in_reference_m=(
                            physical_grasp_offset
                        ),
                        attachment_rotation_in_reference=(
                            physical_grasp_rotation
                        ),
                        profile=motion_profile,
                        alignment_blend=alignment_blend,
                        broad_face_normal_deviation_rad=(
                            _broad_face_normal_deviation_for_retry(
                                motion_profile,
                                planning_retry_count,
                            )
                        ),
                        frame_prefix=(
                            "sweep_target"
                            if variant_index == 0
                            else f"sweep_alignment_{variant_index}_target"
                        ),
                        max_push_distance_m=(
                            requested_micro_push_distance_m
                        ),
                        start_offset_m=(
                            motion_profile.sweep_recovery_standoff_m
                            if planar_recovery
                            else (
                                None
                                if continue_from_contact
                                else approach_standoff_m
                            )
                        ),
                        table_surface_z_override_m=(
                            physical_grasp_monitor.table_surface_z_m
                        ),
                        eef_rotation_override=(
                            np.asarray(current_eef_rotation, dtype=float)
                            if continue_from_contact or planar_recovery
                            else None
                        ),
                        recovery_eef_position_world_m=(
                            np.asarray(current_eef_position, dtype=float)
                            if planar_recovery
                            else None
                        ),
                        recovery_lift_m=(
                            motion_profile.sweep_recovery_lift_m
                            if planar_recovery
                            else None
                        ),
                    )

                sweep_task = MotionTask(
                    task_id=f"c1_1:micro-sweep:{block_id}:{attempt_index}",
                    subgoal_id=sweep_binding.subgoal_id,
                    action_type=sweep_action_type,
                    ee=sweep_binding.ee,
                    tool=sweep_binding.tool,
                    target_ids=[block_id],
                    goal=MotionGoal(
                        goal_type=GoalType.POSE,
                        target_pose=zone_pose,
                        target_region_id=sweep_binding.goal_region_id,
                    ),
                    contact=(
                        ContactManipulationSpec(
                            primitive="SWEEP",
                            contact_surface=ContactSurfaceType.RIM,
                            path_pattern="RADIAL",
                            target_grouping="SINGLE",
                            maintain_contact=continue_from_contact,
                            metadata={
                                "geometry_provider": (
                                    "C1_1_MJCF_RADIAL_SWEEP_V3"
                                )
                            },
                        )
                        if args.sweep_provider == "openai"
                        else None
                    ),
                    allowed_touch_objects=[sweep_binding.tool, block_id],
                    metadata={
                        "task_planner_subgoal": sweep_binding.subgoal_id,
                        "held_tool": sweep_binding.tool,
                        "source_action_type": sweep_binding.action_type,
                        "source_mode": sweep_binding.mode,
                        "micro_push_attempt": attempt_index,
                    },
                )
                sweep_request = _request(
                    request_id=(
                        f"c1_1:motion-request:micro-sweep:{block_id}:"
                        f"{attempt_index}"
                    ),
                    world=world,
                    task=sweep_task,
                    seed=args.seed,
                    task_planner_artifact=args.task_planner,
                    allowed_planning_time_s=args.planning_time,
                    rrt_max_iterations=args.rrt_max_iterations,
                )
                sweep_request.constraints.allowed_collision_pairs = [
                    ("robot", sweep_binding.tool),
                    (sweep_binding.ee, "table*"),
                    (sweep_binding.tool, "table*"),
                    (sweep_binding.tool, block_id),
                ]
                sweep_request.constraints = sweep_request.constraints.model_copy(
                    update={
                        "max_jacobian_condition_number": (
                            motion_profile.sweep_max_jacobian_condition_number
                        ),
                        "min_jacobian_singular_value": (
                            motion_profile.sweep_min_jacobian_singular_value
                        ),
                        "max_joint_path_step_rad": (
                            motion_profile.sweep_max_joint_path_step_rad
                        ),
                        "collision_margin_m": (
                            motion_profile.sweep_collision_margin_m
                        ),
                        "velocity_scaling": (
                            motion_profile.sweep_velocity_scaling
                        ),
                        "acceleration_scaling": (
                            motion_profile.sweep_acceleration_scaling
                        ),
                        "jerk_scaling": motion_profile.sweep_jerk_scaling,
                    }
                )
                sweep_request.options = sweep_request.options.model_copy(
                    update={
                        "allowed_planning_time_s": (
                            motion_profile.sweep_allowed_planning_time_s
                        )
                    }
                )
                step_dir = (
                    output
                    / "sweep_steps"
                    / block_id
                    / f"attempt_{attempt_index:02d}"
                )
                step_dir.mkdir(parents=True, exist_ok=True)
                _write_model(step_dir / "request.json", sweep_request)
                use_openai_sweep_provider = bool(
                    args.sweep_provider == "openai"
                    and not continue_from_contact
                    and not planar_recovery
                    and not force_task_geometry_retry
                )
                source_sweep = (
                    _openai_artifact(
                        sweep_request,
                        model=args.model,
                        candidates=args.candidates,
                        cache_dir=output / "keyframe_cache",
                    )
                    if use_openai_sweep_provider
                    else _sweep_template_artifact(
                        sweep_request,
                        profile=motion_profile,
                        alignment_variant_offset=alignment_variant_offset,
                        contact_continuation=continue_from_contact,
                        planar_recovery=planar_recovery,
                    )
                )
                if use_openai_sweep_provider:
                    _write_model(step_dir / "keyframes_source.json", source_sweep)
                    source_sweep = _expand_task_geometry_orientation_variants(
                        source_sweep,
                        sweep_request,
                    )
                    raw_sweep = _bind_grounded_task_geometry_keyframes(
                        source_sweep,
                        sweep_request,
                        execution_metadata_resolver=lambda frame_name, frame: (
                            _sweep_frame_execution_metadata(
                                frame_name,
                                frame,
                                profile=motion_profile,
                            )
                        ),
                    )
                else:
                    raw_sweep = source_sweep
                _write_model(step_dir / "keyframes_raw.json", raw_sweep)
                sweep_compiler = pick_compiler.with_reference_environment(
                    runtime.env
                )
                reference_kind, reference_name, _, _ = (
                    runtime._grasp_reference(runtime.env)
                )
                sweep_factory = ToolUseJournalCollisionContextFactory(
                    sweep_compiler,
                    attachment_reference_name=reference_name,
                    attachment_reference_kind=reference_kind,
                )
                try:
                    sweep_result, sweep_registry = _plan(
                        sweep_request,
                        raw_sweep,
                        adapter,
                        sweep_factory,
                    )
                except MotionPlanningPipelineError as error:
                    block_dimensions = np.asarray(
                        world.objects[block_id]["dimensions_m"], dtype=float
                    )
                    supported_block_center_z_m = float(
                        physical_grasp_monitor.table_surface_z_m
                        + 0.5 * block_dimensions[2]
                    )
                    planning_failure_record = {
                        "block_id": block_id,
                        "attempt": attempt_index,
                        "selected_strategy": None,
                        "selected_orientation_variant_value": None,
                        "alignment_variant_offset": alignment_variant_offset,
                        "approach_standoff_m": approach_standoff_m,
                        "requested_micro_push_distance_m": (
                            requested_micro_push_distance_m
                        ),
                        "before_position_m": before_position.tolist(),
                        "after_position_m": before_position.tolist(),
                        "radial_progress_m": 0.0,
                        "block_vertical_displacement_m": 0.0,
                        "block_support_lift_m": float(
                            before_position[2] - supported_block_center_z_m
                        ),
                        "target_contact_sample_count": 0,
                        "mean_contact_height_error_m": None,
                        "max_abs_contact_height_error_m": None,
                        "recovery_exhausted": False,
                        "execution_status": "PLANNING_FAILED",
                        "route_provider": (
                            "OPENAI"
                            if use_openai_sweep_provider
                            else "TASK_GEOMETRY"
                        ),
                        "planning_failure": str(error),
                        "goal_reached": False,
                    }
                    micro_push_records.append(planning_failure_record)
                    (step_dir / "micro_push_result.json").write_text(
                        json.dumps(
                            planning_failure_record,
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    if use_openai_sweep_provider:
                        # Keep planning_retry_count unchanged so the reference
                        # frames retain the same broad-face normal deviation.
                        # Only the route topology changes on the next attempt.
                        force_task_geometry_retry = True
                        continue
                    force_task_geometry_retry = False
                    if (
                        args.sweep_provider == "task-geometry"
                        and continue_from_contact
                    ):
                        continue_from_contact = False
                        planar_recovery = True
                        micro_push_limit_m = (
                            motion_profile.sweep_micro_push_distance_m
                        )
                        continue
                    if continue_from_contact:
                        reduced_limit_m = _reduced_micro_push_limit_m(
                            requested_micro_push_distance_m,
                            minimum_distance_m=(
                                motion_profile.sweep_micro_push_min_distance_m
                            ),
                            retry_scale=(
                                motion_profile.sweep_micro_push_retry_scale
                            ),
                        )
                        if reduced_limit_m < requested_micro_push_distance_m - 1e-9:
                            micro_push_limit_m = reduced_limit_m
                            planar_recovery = False
                        else:
                            continue_from_contact = False
                            planar_recovery = True
                    elif planar_recovery:
                        continue_from_contact = False
                        planar_recovery = False
                    else:
                        approach_standoff_m = max(
                            motion_profile.sweep_recovery_standoff_m,
                            approach_standoff_m
                            - motion_profile.sweep_approach_standoff_step_m,
                        )
                        planar_recovery = False
                    alignment_variant_offset = (
                        alignment_variant_offset + 1
                    ) % motion_profile.sweep_plane_alignment_candidate_count
                    if not planar_recovery and micro_push_limit_m < (
                        motion_profile.sweep_micro_push_distance_m
                    ):
                        continue_from_contact = True
                    planning_retry_count += 1
                    continue
                _write_model(
                    step_dir / "keyframes_contextualized.json",
                    sweep_result.keyframe_artifact,
                )
                sweep_plan = sweep_result.plan
                _write_model(step_dir / "motion_plan.json", sweep_plan)
                plan_phase = (
                    "CONTACT_CONTINUATION"
                    if continue_from_contact
                    else (
                        "PLANAR_RECOVERY"
                        if planar_recovery
                        else (
                            "REACQUIRE_AFTER_PUSH"
                            if committed_push_count > 0
                            else "INITIAL_APPROACH"
                        )
                    )
                )
                maximum_phase_duration_s = (
                    motion_profile.sweep_max_contact_continuation_duration_s
                    if plan_phase == "CONTACT_CONTINUATION"
                    else (
                        motion_profile.sweep_max_recovery_duration_s
                        if plan_phase
                        in {"PLANAR_RECOVERY", "REACQUIRE_AFTER_PUSH"}
                        else None
                    )
                )
                if (
                    maximum_phase_duration_s is not None
                    and sweep_plan.duration_s > maximum_phase_duration_s
                ):
                    block_dimensions = np.asarray(
                        world.objects[block_id]["dimensions_m"], dtype=float
                    )
                    supported_block_center_z_m = float(
                        physical_grasp_monitor.table_surface_z_m
                        + 0.5 * block_dimensions[2]
                    )
                    long_plan_record = {
                        "block_id": block_id,
                        "attempt": attempt_index,
                        "selected_strategy": None,
                        "selected_orientation_variant_value": None,
                        "alignment_variant_offset": alignment_variant_offset,
                        "approach_standoff_m": approach_standoff_m,
                        "requested_micro_push_distance_m": (
                            requested_micro_push_distance_m
                        ),
                        "before_position_m": before_position.tolist(),
                        "after_position_m": before_position.tolist(),
                        "radial_progress_m": 0.0,
                        "block_vertical_displacement_m": 0.0,
                        "block_support_lift_m": float(
                            before_position[2] - supported_block_center_z_m
                        ),
                        "target_contact_sample_count": 0,
                        "mean_contact_height_error_m": None,
                        "max_abs_contact_height_error_m": None,
                        "recovery_exhausted": False,
                        "execution_status": (
                            "PLAN_REJECTED_LONG_CONTINUATION"
                            if plan_phase == "CONTACT_CONTINUATION"
                            else (
                                "PLAN_REJECTED_LONG_RECOVERY"
                                if plan_phase == "PLANAR_RECOVERY"
                                else "PLAN_REJECTED_LONG_REACQUIRE"
                            )
                        ),
                        "route_provider": (
                            "OPENAI"
                            if use_openai_sweep_provider
                            else "TASK_GEOMETRY"
                        ),
                        "plan_phase": plan_phase,
                        "planned_duration_s": float(sweep_plan.duration_s),
                        "maximum_phase_duration_s": maximum_phase_duration_s,
                        "goal_reached": False,
                    }
                    micro_push_records.append(long_plan_record)
                    (step_dir / "micro_push_result.json").write_text(
                        json.dumps(
                            long_plan_record, ensure_ascii=False, indent=2
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    if use_openai_sweep_provider:
                        # The grounded goal is retained; only replace the VLM
                        # route by the deterministic task-geometry route.
                        force_task_geometry_retry = True
                        continue
                    force_task_geometry_retry = False
                    if (
                        args.sweep_provider == "task-geometry"
                        and plan_phase == "CONTACT_CONTINUATION"
                    ):
                        # Verified V3 fallback: a continuation that would run
                        # too long is replaced by lift/backoff/re-engage at the
                        # same plate-plane alignment. Changing orientation here
                        # loses the contact geometry that produced progress.
                        continue_from_contact = False
                        planar_recovery = True
                        micro_push_limit_m = (
                            motion_profile.sweep_micro_push_distance_m
                        )
                        continue
                    alignment_variant_offset = (
                        alignment_variant_offset + 1
                    ) % motion_profile.sweep_plane_alignment_candidate_count
                    if plan_phase == "CONTACT_CONTINUATION":
                        reduced_limit_m = _reduced_micro_push_limit_m(
                            requested_micro_push_distance_m,
                            minimum_distance_m=(
                                motion_profile.sweep_micro_push_min_distance_m
                            ),
                            retry_scale=(
                                motion_profile.sweep_micro_push_retry_scale
                            ),
                        )
                        if reduced_limit_m < requested_micro_push_distance_m - 1e-9:
                            micro_push_limit_m = reduced_limit_m
                            continue_from_contact = True
                            planar_recovery = False
                        else:
                            continue_from_contact = False
                            planar_recovery = True
                    elif plan_phase in {
                        "PLANAR_RECOVERY",
                        "REACQUIRE_AFTER_PUSH",
                    }:
                        # One deterministic recovery attempt is enough to test
                        # the restored contact geometry. If it also fails,
                        # request a newly grounded approach at the next bounded
                        # broad-face normal deviation instead of exhausting
                        # every physical attempt on the same stalled contact.
                        continue_from_contact = False
                        planar_recovery = False
                    else:
                        continue_from_contact = False
                        planar_recovery = False
                    planning_retry_count += 1
                    continue
                force_task_geometry_retry = False
                plans.append(sweep_plan)
                sample_start_index = len(physical_grasp_monitor.samples)
                simulator_checkpoint = _simulator_checkpoint(runtime.env)
                grasp_monitor_checkpoint = physical_grasp_monitor.checkpoint()
                if video_recorder is not None:
                    video_recorder.begin_transaction()
                try:
                    sweep_run, sweep_report = _run(
                        runtime,
                        sweep_plan,
                        sweep_registry,
                        f"c1_1:controller:micro-sweep:{block_id}:{attempt_index}",
                        collision_check_stride=(
                            args.execution_collision_check_stride
                        ),
                        physical_grasp_monitor=physical_grasp_monitor,
                        render_video=video_recorder is not None,
                    )
                except BaseException:
                    if video_recorder is not None:
                        video_recorder.rollback_transaction()
                    raise
                physical_attempt_count += 1
                _write_model(step_dir / "simulation_run.json", sweep_run)
                _write_model(step_dir / "execution_report.json", sweep_report)
                attempt_samples = copy.deepcopy(
                    physical_grasp_monitor.samples[sample_start_index:]
                )
                execution_succeeded = sweep_report.status.value == "SUCCESS"
                execution_state_rolled_back = not execution_succeeded
                if execution_state_rolled_back:
                    _restore_simulator_checkpoint(
                        runtime.env, simulator_checkpoint
                    )
                    physical_grasp_monitor.restore(grasp_monitor_checkpoint)
                else:
                    reports.append(sweep_report)
                    sweep_reports.append(sweep_report)

                after_position = np.asarray(
                    _block_positions(runtime.env, (block_id,))[block_id],
                    dtype=float,
                )
                after_radius_m = float(
                    np.linalg.norm(after_position[:2] - zone_xy)
                )
                radial_progress_m = before_radius_m - after_radius_m
                block_dimensions = np.asarray(
                    world.objects[block_id]["dimensions_m"], dtype=float
                )
                supported_block_center_z_m = float(
                    physical_grasp_monitor.table_surface_z_m
                    + 0.5 * block_dimensions[2]
                )
                block_support_lift_m = float(
                    after_position[2] - supported_block_center_z_m
                )
                target_contact_samples = [
                    sample
                    for sample in attempt_samples
                    if sample.get("target_block_id") == block_id
                    and int(sample.get("target_block_contact_count", 0)) > 0
                ]
                if (
                    sweep_report.status.value == "SUCCESS"
                    and target_contact_samples
                ):
                    committed_push_count += 1
                contact_height_errors_m = [
                    float(sample["strongest_target_contact_position_m"][2])
                    - float(
                        sample["physical_push_control"][
                            "contact_height_target_m"
                        ]
                    )
                    for sample in target_contact_samples
                    if sample.get("strongest_target_contact_position_m")
                    is not None
                    and sample.get("target_block_position_m") is not None
                    and isinstance(sample.get("physical_push_control"), Mapping)
                ]
                attempt_recovery_exhausted = any(
                    bool(sample.get("push_recovery_exhausted"))
                    for sample in attempt_samples
                    if sample.get("target_block_id") == block_id
                )
                selected_strategy = (
                    sweep_plan.segments[0].metadata.get("strategy_id")
                    if sweep_plan.segments
                    else None
                )
                selected_orientation_variant_value = None
                for marker in ("alignment-", "orientation-"):
                    if marker not in str(selected_strategy):
                        continue
                    try:
                        selected_orientation_variant_value = float(
                            str(selected_strategy).rsplit(marker, 1)[1]
                        )
                    except (IndexError, TypeError, ValueError):
                        selected_orientation_variant_value = None
                    break
                reached_goal = bool(
                    _inside_goal_region(
                        runtime.env,
                        world,
                        (block_id,),
                        sweep_binding.goal_region_id,
                    )
                )
                insufficient_progress = (
                    radial_progress_m
                    < motion_profile.sweep_micro_push_min_progress_m
                )
                excessive_block_lift = (
                    abs(block_support_lift_m)
                    > motion_profile.sweep_max_block_lift_m
                )
                physical_rejection_reason = _physical_push_rejection_reason(
                    execution_succeeded=execution_succeeded,
                    reached_goal=reached_goal,
                    target_contact_sample_count=len(target_contact_samples),
                    radial_progress_m=radial_progress_m,
                    minimum_progress_m=(
                        motion_profile.sweep_micro_push_min_progress_m
                    ),
                    block_support_lift_m=block_support_lift_m,
                    maximum_block_lift_m=(
                        motion_profile.sweep_max_block_lift_m
                    ),
                )
                if (
                    args.sweep_provider == "task-geometry"
                    and physical_rejection_reason == "EXCESSIVE_BLOCK_LIFT"
                    and block_support_lift_m < 0.0
                ):
                    # A negative support error is slight table penetration,
                    # not a lifted block. Preserve the verified V3 state and
                    # let planar recovery re-seat the contact. Positive lift
                    # remains a rejected, rolled-back safety violation.
                    physical_rejection_reason = None
                if execution_succeeded and physical_rejection_reason is not None:
                    _restore_simulator_checkpoint(
                        runtime.env, simulator_checkpoint
                    )
                    physical_grasp_monitor.restore(grasp_monitor_checkpoint)
                    execution_state_rolled_back = True
                    reached_goal = False
                    reports.pop()
                    sweep_reports.pop()
                    if target_contact_samples:
                        committed_push_count -= 1
                committed_video_frame_count = 0
                discarded_video_frame_count = 0
                if video_recorder is not None:
                    if execution_state_rolled_back:
                        discarded_video_frame_count = (
                            video_recorder.rollback_transaction()
                        )
                    else:
                        committed_video_frame_count = (
                            video_recorder.commit_transaction()
                        )
                committed_after_position = np.asarray(
                    _block_positions(runtime.env, (block_id,))[block_id],
                    dtype=float,
                )
                record = {
                    "block_id": block_id,
                    "attempt": attempt_index,
                    "selected_strategy": selected_strategy,
                    "selected_orientation_variant_value": (
                        selected_orientation_variant_value
                    ),
                    "alignment_variant_offset": alignment_variant_offset,
                    "approach_standoff_m": approach_standoff_m,
                    "requested_micro_push_distance_m": (
                        requested_micro_push_distance_m
                    ),
                    "before_position_m": before_position.tolist(),
                    "after_position_m": after_position.tolist(),
                    "committed_after_position_m": (
                        committed_after_position.tolist()
                    ),
                    "radial_progress_m": radial_progress_m,
                    "block_vertical_displacement_m": float(
                        after_position[2] - before_position[2]
                    ),
                    "block_support_lift_m": block_support_lift_m,
                    "target_contact_sample_count": len(target_contact_samples),
                    "mean_contact_height_error_m": (
                        float(np.mean(contact_height_errors_m))
                        if contact_height_errors_m
                        else None
                    ),
                    "max_abs_contact_height_error_m": (
                        float(np.max(np.abs(contact_height_errors_m)))
                        if contact_height_errors_m
                        else None
                    ),
                    "recovery_exhausted": (
                        attempt_recovery_exhausted
                    ),
                    "execution_status": sweep_report.status.value,
                    "route_provider": (
                        "OPENAI"
                        if use_openai_sweep_provider
                        else "TASK_GEOMETRY"
                    ),
                    "physical_validation_status": (
                        "REJECTED"
                        if physical_rejection_reason is not None
                        else "ACCEPTED"
                    ),
                    "physical_rejection_reason": physical_rejection_reason,
                    "execution_state_rolled_back": (
                        execution_state_rolled_back
                    ),
                    "committed_video_frame_count": committed_video_frame_count,
                    "discarded_video_frame_count": discarded_video_frame_count,
                    "goal_reached": reached_goal,
                }
                micro_push_records.append(record)
                if args.save_physical_traces:
                    (step_dir / "physical_trace.json").write_text(
                        json.dumps(
                            attempt_samples,
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                (step_dir / "micro_push_result.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                sweep_artifact_paths.append(
                    {
                        "block_id": block_id,
                        "attempt": str(attempt_index),
                        "plan": str((step_dir / "motion_plan.json").resolve()),
                        "report": str(
                            (step_dir / "execution_report.json").resolve()
                        ),
                    }
                )
                continue_from_contact = bool(
                    not insufficient_progress
                    and not attempt_recovery_exhausted
                    and not excessive_block_lift
                    and target_contact_samples
                    and sweep_report.status.value == "SUCCESS"
                )
                if (
                    continue_from_contact
                    and selected_orientation_variant_value is not None
                ):
                    alignment_variant_offset = (
                        _nearest_alignment_variant_index(
                            motion_profile,
                            selected_orientation_variant_value,
                        )
                    )
                planar_recovery = bool(
                    not continue_from_contact
                    and (
                        insufficient_progress
                        or attempt_recovery_exhausted
                        or excessive_block_lift
                    )
                    and sweep_report.status.value == "SUCCESS"
                )
                if execution_state_rolled_back:
                    alignment_variant_offset = (
                        alignment_variant_offset + 1
                    ) % motion_profile.sweep_plane_alignment_candidate_count
                    if plan_phase == "CONTACT_CONTINUATION":
                        reduced_limit_m = _reduced_micro_push_limit_m(
                            requested_micro_push_distance_m,
                            minimum_distance_m=(
                                motion_profile.sweep_micro_push_min_distance_m
                            ),
                            retry_scale=(
                                motion_profile.sweep_micro_push_retry_scale
                            ),
                        )
                        if reduced_limit_m < requested_micro_push_distance_m - 1e-9:
                            micro_push_limit_m = reduced_limit_m
                            continue_from_contact = True
                            planar_recovery = False
                        else:
                            continue_from_contact = False
                            planar_recovery = True
                    elif plan_phase in {
                        "PLANAR_RECOVERY",
                        "REACQUIRE_AFTER_PUSH",
                    }:
                        # A failed recovery should return to a newly grounded
                        # approach at the next bounded broad-face normal
                        # deviation, not consume every physical attempt on the
                        # same stalled contact geometry.
                        continue_from_contact = False
                        planar_recovery = False
                        planning_retry_count += 1
                    else:
                        continue_from_contact = False
                        if use_openai_sweep_provider:
                            # A VLM route can be kinematically valid yet make
                            # poor physical contact.  Keep the same grounded
                            # broad-face normal deviation and retry with the
                            # deterministic initial route before attempting a
                            # lift/backoff recovery or changing orientation.
                            force_task_geometry_retry = True
                            planar_recovery = False
                        else:
                            # A deterministic initial approach that failed
                            # physical validation is followed by one
                            # lift/backoff/re-engage recovery.
                            planar_recovery = True
                    continue
                if reached_goal:
                    break
                if continue_from_contact:
                    micro_push_limit_m = motion_profile.sweep_micro_push_distance_m
                current_retention = physical_grasp_monitor.summary()[
                    "grasp_retention"
                ]
                if current_retention["status"] != "SUCCESS":
                    break

            if (
                block_cursor == len(block_queue)
                and cleanup_passes_completed
                < motion_profile.sweep_cleanup_max_passes
            ):
                cleanup_world = adapter.world_snapshot()
                cleanup_inside = _inside_goal_region(
                    runtime.env,
                    cleanup_world,
                    block_ids,
                    sweep_binding.goal_region_id,
                )
                cleanup_positions = _block_positions(runtime.env, block_ids)
                cleanup_targets = _cleanup_block_ids(
                    block_ids,
                    inside_ids=cleanup_inside,
                    initial_positions_m=initial_block_positions,
                    current_positions_m=cleanup_positions,
                    max_support_error_m=motion_profile.sweep_max_block_lift_m,
                )
                if cleanup_targets:
                    cleanup_passes_completed += 1
                    cleanup_history.append(
                        {
                            "pass": cleanup_passes_completed,
                            "target_ids": list(cleanup_targets),
                            "inside_before": list(cleanup_inside),
                            "support_errors_before_m": {
                                target_id: float(
                                    cleanup_positions[target_id][2]
                                    - initial_block_positions[target_id][2]
                                )
                                for target_id in cleanup_targets
                            },
                        }
                    )
                    block_queue.extend(cleanup_targets)

        (output / "micro_push_summary.json").write_text(
            json.dumps(micro_push_records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        sweep_report = sweep_reports[-1] if sweep_reports else None

        final_grasp_validation = physical_grasp_monitor.summary()
        # Pick lift/hold validation belongs to the end of the pick phase. A
        # later table-level tool action must not retroactively redefine it
        # using the plate's final sweep/retract pose.
        final_grasp_validation["pick_lift_validation"] = grasp_validation[
            "pick_lift_validation"
        ]
        for field_name in (
            "lift_succeeded",
            "required_lift_m",
            "final_lift_m",
            "bottom_clearance_succeeded",
            "required_bottom_clearance_m",
            "final_bottom_clearance_m",
            "final_hold_succeeded",
            "required_final_hold_s",
            "final_clearance_hold_s",
        ):
            if field_name in grasp_validation:
                final_grasp_validation[field_name] = grasp_validation[field_name]
        final_grasp_validation["status"] = (
            "SUCCESS"
            if (
                final_grasp_validation["grasp_retention"]["status"]
                == "SUCCESS"
                and final_grasp_validation["pick_lift_validation"]["status"]
                == "SUCCESS"
            )
            else "FAILED"
        )
        if args.save_physical_traces:
            (output / "pick_physical_grasp_trace.json").write_text(
                json.dumps(
                    physical_grasp_monitor.samples,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        (output / "pick_physical_grasp_validation.json").write_text(
            json.dumps(final_grasp_validation, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        grasp_retention_validation = final_grasp_validation["grasp_retention"]
        tool_clearance_validation = final_grasp_validation["tool_clearance"]
        (output / "grasp_retention_validation.json").write_text(
            json.dumps(
                grasp_retention_validation, ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        (output / "tool_clearance_validation.json").write_text(
            json.dumps(
                tool_clearance_validation, ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )

        inside = _inside_goal_region(
            runtime.env,
            world,
            block_ids,
            sweep_binding.goal_region_id,
        )
        final_block_positions = _block_positions(runtime.env, block_ids)
        block_support_errors_m = {
            block_id: float(
                final_block_positions[block_id][2]
                - initial_block_positions[block_id][2]
            )
            for block_id in block_ids
        }
        blocks_support_stable = all(
            abs(error_m) <= motion_profile.sweep_max_block_lift_m
            for error_m in block_support_errors_m.values()
        )
        frame = runtime.env.sim.render(
            camera_name=args.camera, height=args.height, width=args.width
        )[::-1]
        final_frame = output / "controller_final.png"
        cv2.imwrite(str(final_frame), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if video_recorder is not None:
            video_recorder.hold_final_frame(args.video_hold_seconds)
        execution_succeeded = all(
            report.status.value == "SUCCESS" for report in reports
        )
        task_goal_satisfied = (
            set(inside) == set(block_ids) and blocks_support_stable
        )
        grasp_retained = grasp_retention_validation["status"] == "SUCCESS"
        tool_clearance_satisfied = (
            tool_clearance_validation["status"] == "SUCCESS"
        )
        summary = {
            # A collision-free controller replay is necessary but not sufficient:
            # this scenario succeeds only when every block reaches the collection
            # zone.  Keep both facts explicit so task failure cannot be archived
            # as a successful end-to-end run.
            "status": (
                "SUCCESS"
                if (
                    execution_succeeded
                    and task_goal_satisfied
                    and grasp_retained
                    and tool_clearance_satisfied
                )
                else "FAILED"
            ),
            "execution_succeeded": execution_succeeded,
            "task_goal_satisfied": task_goal_satisfied,
            "blocks_support_stable": blocks_support_stable,
            "block_support_errors_m": block_support_errors_m,
            "grasp_retained": grasp_retained,
            "tool_clearance_satisfied": tool_clearance_satisfied,
            "selected_ee": sweep_binding.ee,
            "selected_tool": sweep_binding.tool,
            "selected_pick_subgoal": pick_binding.subgoal_id,
            "selected_sweep_subgoal": sweep_binding.subgoal_id,
            "selected_goal_region": sweep_binding.goal_region_id,
            "model": args.model,
            "candidate_count": len(raw_pick.candidates),
            "sweep_provider": args.sweep_provider,
            "seed": args.seed,
            "pick_controller_kp": args.pick_controller_kp,
            "pick_controller_damping_ratio": args.pick_controller_damping_ratio,
            "sweep_controller_kp": args.controller_kp,
            "sweep_controller_damping_ratio": args.controller_damping_ratio,
            "motion_profile": asdict(motion_profile),
            "plan_ids": [plan.plan_id for plan in plans],
            "execution_statuses": [report.status.value for report in reports],
            "blocks_inside": inside,
            "blocks_inside_count": len(inside),
            "blocks_total": len(block_ids),
            "blocks_outside": sorted(set(block_ids) - set(inside)),
            "block_execution_order": executed_block_sequence,
            "initial_block_execution_order": block_ids,
            "cleanup_passes_completed": cleanup_passes_completed,
            "cleanup_history": cleanup_history,
            "initial_block_positions_m": initial_block_positions,
            "final_block_positions_m": final_block_positions,
            "execution_collision_check_stride": (
                args.execution_collision_check_stride
            ),
            "adaptive_settle_extra_duration_s": sum(
                float(
                    report.metadata.get(
                        "adaptive_settle_extra_duration_s", 0.0
                    )
                )
                for report in sweep_reports
            ),
            "micro_push_attempts": micro_push_records,
            "micro_push_attempt_count": len(micro_push_records),
            "rolled_back_execution_count": sum(
                bool(record.get("execution_state_rolled_back"))
                for record in micro_push_records
            ),
            "physical_traces_saved": args.save_physical_traces,
            "attachment_mode": "CONTACT_FRICTION",
            "weld_absent": runtime.attachment is None,
            "physical_grasp_validation": final_grasp_validation,
            "grasp_retention_validation": grasp_retention_validation,
            "tool_clearance_validation": tool_clearance_validation,
            "final_attached_object_id": runtime.attached_object_id,
            "last_attachment_break": (
                runtime.last_attachment_break.as_mapping()
                if runtime.last_attachment_break is not None
                else None
            ),
            "artifacts": {
                "pick_plan": str((output / "pick_motion_plan.json").resolve()),
                "sweep_steps": sweep_artifact_paths,
                "micro_push_summary": str(
                    (output / "micro_push_summary.json").resolve()
                ),
                "pick_report": str((output / "pick_execution_report.json").resolve()),
                "final_frame": str(final_frame.resolve()),
                "physical_grasp_trace": (
                    str(
                        (output / "pick_physical_grasp_trace.json").resolve()
                    )
                    if args.save_physical_traces
                    else None
                ),
                "physical_grasp_validation": str(
                    (output / "pick_physical_grasp_validation.json").resolve()
                ),
                "grasp_retention_validation": str(
                    (output / "grasp_retention_validation.json").resolve()
                ),
                "tool_clearance_validation": str(
                    (output / "tool_clearance_validation.json").resolve()
                ),
                "video": (
                    str(video_recorder.path)
                    if video_recorder is not None
                    else None
                ),
            },
            "video": (
                {
                    "path": str(video_recorder.path),
                    "width": video_recorder.width,
                    "height": video_recorder.height,
                    "fps": video_recorder.fps,
                    "frame_count": video_recorder.frame_count,
                    "camera": args.camera,
                    "camera_eye_m": args.camera_eye,
                    "camera_look_at_m": args.camera_look_at,
                    "camera_fovy_deg": args.camera_fovy,
                }
                if video_recorder is not None
                else None
            ),
        }
        (output / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["status"] == "SUCCESS" else 2
    finally:
        if video_recorder is not None:
            video_recorder.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
