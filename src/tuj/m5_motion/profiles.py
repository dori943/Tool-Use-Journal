"""Reusable, validated planning and execution profiles.

Scenario files may override these values, but generic algorithms depend on
profile roles rather than a monolithic scenario-specific parameter bag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


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
    "PushPlanningProfile",
    "RobotControlProfile",
    "TaskRecoveryProfile",
    "ToolAffordanceProfile",
]
