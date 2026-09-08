"""Reusable, validated planning and execution profiles.

Scenario files may override these values, but generic algorithms depend on
profile roles rather than a monolithic scenario-specific parameter bag.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RobotControlProfile:
    joint_position_kp: float = 80.0
    damping_ratio: float = 1.0
    velocity_scaling: float = 0.5
    acceleration_scaling: float = 0.5
    jerk_scaling: float = 0.5
    max_joint_path_step_rad: float = 0.02

    def __post_init__(self) -> None:
        _positive("joint_position_kp", self.joint_position_kp)
        _positive("damping_ratio", self.damping_ratio)
        _positive("max_joint_path_step_rad", self.max_joint_path_step_rad)
        for name in ("velocity_scaling", "acceleration_scaling", "jerk_scaling"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be within (0, 1]")


@dataclass(frozen=True, slots=True)
class GraspExecutionProfile:
    approach_distance_m: float = 0.08
    retreat_distance_m: float = 0.10
    hold_duration_s: float = 1.0
    minimum_lift_m: float = 0.05
    contact_loss_grace_s: float = 0.05
    required_contact_ticks: int = 3
    max_friction_utilization: float = 0.95

    def __post_init__(self) -> None:
        for name in (
            "approach_distance_m",
            "retreat_distance_m",
            "hold_duration_s",
            "minimum_lift_m",
            "contact_loss_grace_s",
        ):
            _positive(name, getattr(self, name))
        if self.required_contact_ticks < 1:
            raise ValueError("required_contact_ticks must be positive")
        if not 0.0 < self.max_friction_utilization <= 1.0:
            raise ValueError("max_friction_utilization must be within (0, 1]")


@dataclass(frozen=True, slots=True)
class GripperPreshapeProfile:
    """Geometry-independent tolerances for opening a parallel gripper."""

    clearance_m: float = 0.02
    tolerance_m: float = 0.002
    settle_ticks: int = 25

    def __post_init__(self) -> None:
        _non_negative("clearance_m", self.clearance_m)
        _positive("tolerance_m", self.tolerance_m)
        if self.settle_ticks < 1:
            raise ValueError("settle_ticks must be positive")


@dataclass(frozen=True, slots=True)
class GraspForceProfile:
    """Persistent close and force-feedback settings for a friction grasp."""

    close_command: float = 1.0
    closure_actuator_kp: float = 50.0
    maximum_actuator_kp: float = 70.0
    force_feedback_gain: float = 0.2
    grip_force_safety_factor: float = 2.0
    fallback_maximum_grip_force_n: float = 80.0
    fallback_sliding_friction: float = 1.0
    fallback_retention_force_n: float = 10.0

    def __post_init__(self) -> None:
        if not 0.0 < self.close_command <= 1.0:
            raise ValueError("close_command must be within (0, 1]")
        for name in (
            "closure_actuator_kp",
            "maximum_actuator_kp",
            "force_feedback_gain",
            "grip_force_safety_factor",
            "fallback_maximum_grip_force_n",
            "fallback_sliding_friction",
            "fallback_retention_force_n",
        ):
            _positive(name, getattr(self, name))
        if self.maximum_actuator_kp < self.closure_actuator_kp:
            raise ValueError(
                "maximum_actuator_kp cannot be smaller than closure_actuator_kp"
            )


@dataclass(frozen=True, slots=True)
class GraspValidationProfile:
    """Physical evidence required before a free object is considered held."""

    minimum_lift_m: float = 0.05
    contact_loss_grace_s: float = 0.06
    required_contact_ticks: int = 5
    contact_freeze_ticks: int = 2
    final_hold_duration_s: float = 1.0
    minimum_normal_force_n: float = 0.1
    max_friction_utilization: float = 0.95

    def __post_init__(self) -> None:
        for name in (
            "minimum_lift_m",
            "contact_loss_grace_s",
            "final_hold_duration_s",
            "minimum_normal_force_n",
        ):
            _positive(name, getattr(self, name))
        for name in ("required_contact_ticks", "contact_freeze_ticks"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.max_friction_utilization <= 1.0:
            raise ValueError("max_friction_utilization must be within (0, 1]")


@dataclass(frozen=True, slots=True)
class GraspStabilizationProfile:
    """Bounded post-contact damping settings shared by grasp controllers."""

    contact_follow_gain: float = 0.5
    maximum_translation_m: float = 0.003
    maximum_translation_per_tick_m: float = 0.0005
    maximum_joint_step_rad: float = 0.01

    def __post_init__(self) -> None:
        _non_negative("contact_follow_gain", self.contact_follow_gain)
        for name in (
            "maximum_translation_m",
            "maximum_translation_per_tick_m",
            "maximum_joint_step_rad",
        ):
            _positive(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class PhysicalGraspProfile:
    """Task-independent configuration for a two-finger contact-friction pick.

    ``from_mapping`` accepts both the generic field names and the historical
    C1_1 ``pick_*`` names so validated profiles can be migrated without keeping
    the scenario runner in the execution path.
    """

    grasp_hold_duration_s: float = 1.5
    lift_hold_duration_s: float = 1.0
    clearance_reserve_m: float = 0.0
    preshape: GripperPreshapeProfile = GripperPreshapeProfile()
    force: GraspForceProfile = GraspForceProfile()
    validation: GraspValidationProfile = GraspValidationProfile()
    stabilization: GraspStabilizationProfile = GraspStabilizationProfile()

    def __post_init__(self) -> None:
        _positive("grasp_hold_duration_s", self.grasp_hold_duration_s)
        _positive("lift_hold_duration_s", self.lift_hold_duration_s)
        _non_negative("clearance_reserve_m", self.clearance_reserve_m)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "PhysicalGraspProfile":
        if not raw:
            return cls()

        def value(name: str, legacy: str, default: Any) -> Any:
            return raw.get(name, raw.get(legacy, default))

        defaults = cls()
        preshape = GripperPreshapeProfile(
            clearance_m=float(
                value(
                    "preshape_clearance_m",
                    "pick_preshape_clearance_m",
                    defaults.preshape.clearance_m,
                )
            ),
            tolerance_m=float(
                value(
                    "preshape_tolerance_m",
                    "pick_preshape_tolerance_m",
                    defaults.preshape.tolerance_m,
                )
            ),
            settle_ticks=int(
                value(
                    "preshape_settle_ticks",
                    "pick_preshape_settle_ticks",
                    defaults.preshape.settle_ticks,
                )
            ),
        )
        force = GraspForceProfile(
            close_command=float(
                value(
                    "gripper_close_command",
                    "pick_gripper_close_rate",
                    defaults.force.close_command,
                )
            ),
            closure_actuator_kp=float(
                value(
                    "closure_actuator_kp",
                    "pick_gripper_closure_actuator_kp",
                    defaults.force.closure_actuator_kp,
                )
            ),
            maximum_actuator_kp=float(
                value(
                    "maximum_actuator_kp",
                    "pick_gripper_max_actuator_kp",
                    defaults.force.maximum_actuator_kp,
                )
            ),
            force_feedback_gain=float(
                value(
                    "force_feedback_gain",
                    "pick_gripper_force_feedback_gain",
                    defaults.force.force_feedback_gain,
                )
            ),
            grip_force_safety_factor=float(
                value(
                    "grip_force_safety_factor",
                    "pick_grip_force_safety_factor",
                    defaults.force.grip_force_safety_factor,
                )
            ),
            fallback_maximum_grip_force_n=float(
                raw.get(
                    "fallback_maximum_grip_force_n",
                    defaults.force.fallback_maximum_grip_force_n,
                )
            ),
            fallback_sliding_friction=float(
                raw.get(
                    "fallback_sliding_friction",
                    defaults.force.fallback_sliding_friction,
                )
            ),
            fallback_retention_force_n=float(
                raw.get(
                    "fallback_retention_force_n",
                    defaults.force.fallback_retention_force_n,
                )
            ),
        )
        validation = GraspValidationProfile(
            minimum_lift_m=float(
                value(
                    "minimum_lift_m",
                    "pick_min_lift_m",
                    defaults.validation.minimum_lift_m,
                )
            ),
            contact_loss_grace_s=float(
                value(
                    "contact_loss_grace_s",
                    "pick_contact_loss_grace_s",
                    defaults.validation.contact_loss_grace_s,
                )
            ),
            required_contact_ticks=int(
                value(
                    "required_contact_ticks",
                    "pick_required_contact_ticks",
                    defaults.validation.required_contact_ticks,
                )
            ),
            contact_freeze_ticks=int(
                value(
                    "contact_freeze_ticks",
                    "pick_contact_freeze_ticks",
                    defaults.validation.contact_freeze_ticks,
                )
            ),
            final_hold_duration_s=float(
                value(
                    "final_hold_duration_s",
                    "pick_final_hold_s",
                    defaults.validation.final_hold_duration_s,
                )
            ),
            minimum_normal_force_n=float(
                raw.get(
                    "minimum_normal_force_n",
                    defaults.validation.minimum_normal_force_n,
                )
            ),
            max_friction_utilization=float(
                raw.get(
                    "max_friction_utilization",
                    defaults.validation.max_friction_utilization,
                )
            ),
        )
        stabilization = GraspStabilizationProfile(
            contact_follow_gain=float(
                raw.get(
                    "contact_follow_gain",
                    defaults.stabilization.contact_follow_gain,
                )
            ),
            maximum_translation_m=float(
                raw.get(
                    "contact_follow_max_m",
                    defaults.stabilization.maximum_translation_m,
                )
            ),
            maximum_translation_per_tick_m=float(
                raw.get(
                    "contact_follow_max_tick_m",
                    defaults.stabilization.maximum_translation_per_tick_m,
                )
            ),
            maximum_joint_step_rad=float(
                raw.get(
                    "contact_follow_max_joint_step_rad",
                    defaults.stabilization.maximum_joint_step_rad,
                )
            ),
        )
        return cls(
            grasp_hold_duration_s=float(
                value(
                    "grasp_hold_duration_s",
                    "pick_grasp_hold_s",
                    defaults.grasp_hold_duration_s,
                )
            ),
            lift_hold_duration_s=float(
                raw.get(
                    "lift_hold_duration_s",
                    validation.final_hold_duration_s,
                )
            ),
            clearance_reserve_m=float(
                raw.get(
                    "grasp_clearance_reserve_m",
                    defaults.clearance_reserve_m,
                )
            ),
            preshape=preshape,
            force=force,
            validation=validation,
            stabilization=stabilization,
        )


@dataclass(frozen=True, slots=True)
class ContactExecutionProfile:
    contact_penetration_m: float = 0.001
    maximum_correction_m: float = 0.05
    minimum_progress_m: float = 0.003
    contact_loss_grace_s: float = 0.05
    maintain_contact: bool = True

    def __post_init__(self) -> None:
        _non_negative("contact_penetration_m", self.contact_penetration_m)
        _positive("maximum_correction_m", self.maximum_correction_m)
        _non_negative("minimum_progress_m", self.minimum_progress_m)
        _non_negative("contact_loss_grace_s", self.contact_loss_grace_s)


@dataclass(frozen=True, slots=True)
class PushPlanningProfile:
    approach_standoff_m: float = 0.06
    hover_height_m: float = 0.10
    nominal_step_distance_m: float = 0.03
    minimum_step_distance_m: float = 0.0075
    retry_distance_scale: float = 0.5
    goal_inset_margin_m: float = 0.003
    contact_height_fraction: float = 0.5
    path_pattern: str = "RADIAL"

    def __post_init__(self) -> None:
        for name in (
            "approach_standoff_m",
            "hover_height_m",
            "nominal_step_distance_m",
            "minimum_step_distance_m",
        ):
            _positive(name, getattr(self, name))
        _non_negative("goal_inset_margin_m", self.goal_inset_margin_m)
        if self.minimum_step_distance_m > self.nominal_step_distance_m:
            raise ValueError("minimum step distance cannot exceed nominal distance")
        if not 0.0 < self.retry_distance_scale <= 1.0:
            raise ValueError("retry_distance_scale must be within (0, 1]")
        if not 0.0 <= self.contact_height_fraction <= 1.0:
            raise ValueError("contact_height_fraction must be within [0, 1]")
        if not self.path_pattern.strip():
            raise ValueError("path_pattern must not be empty")


@dataclass(frozen=True, slots=True)
class TaskRecoveryProfile:
    maximum_execution_attempts: int = 12
    maximum_planning_attempts: int = 12
    maximum_cleanup_passes: int = 3
    rollback_failed_execution: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_execution_attempts",
            "maximum_planning_attempts",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.maximum_cleanup_passes < 0:
            raise ValueError("maximum_cleanup_passes must be non-negative")


@dataclass(frozen=True, slots=True)
class ToolAffordanceProfile:
    preferred_surface: str = "AUTO"
    preferred_patch_id: str | None = None
    maximum_contact_force_n: float | None = None

    def __post_init__(self) -> None:
        if not self.preferred_surface.strip():
            raise ValueError("preferred_surface must not be empty")
        if self.maximum_contact_force_n is not None:
            _positive("maximum_contact_force_n", self.maximum_contact_force_n)


__all__ = [
    "ContactExecutionProfile",
    "GraspExecutionProfile",
    "GraspForceProfile",
    "GraspStabilizationProfile",
    "GraspValidationProfile",
    "GripperPreshapeProfile",
    "PhysicalGraspProfile",
    "PushPlanningProfile",
    "RobotControlProfile",
    "TaskRecoveryProfile",
    "ToolAffordanceProfile",
]
