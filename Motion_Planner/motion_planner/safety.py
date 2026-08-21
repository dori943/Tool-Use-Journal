"""Composed collision and kinematic state safety checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from motion_planner.mujoco_collision import CollisionCheckResult
from motion_planner.schema import RelativeKeyframeSpec
from motion_planner.strategy import JointConfig, StateValidator


class JacobianKinematics(Protocol):
    def jacobian_singular_values(
        self, qpos: Sequence[float]
    ) -> tuple[float, ...]: ...


@dataclass(slots=True)
class KinematicSafetyValidator:
    """Preserve collision detail and additionally reject singular states."""

    base: StateValidator
    kinematics: JacobianKinematics
    min_singular_value: float = 1e-4
    max_condition_number: float = 1e4

    def __post_init__(self) -> None:
        if self.min_singular_value < 0 or not math.isfinite(
            self.min_singular_value
        ):
            raise ValueError("min_singular_value must be finite and non-negative")
        if self.max_condition_number <= 1 or not math.isfinite(
            self.max_condition_number
        ):
            raise ValueError("max_condition_number must be finite and greater than 1")

    def _base_check(
        self,
        joint_config: JointConfig,
        keyframe: RelativeKeyframeSpec | None,
        **kwargs: Any,
    ) -> CollisionCheckResult:
        check = getattr(self.base, "check", None)
        if callable(check):
            return check(joint_config, keyframe, **kwargs)
        if keyframe is None:
            return CollisionCheckResult(
                valid=False,
                failure_code="KEYFRAME_CONTEXT_REQUIRED",
                detail="base state validator requires a keyframe",
            )
        return CollisionCheckResult(valid=bool(self.base(joint_config, keyframe)))

    def check(
        self,
        joint_config: Sequence[float],
        keyframe: RelativeKeyframeSpec | None = None,
        **kwargs: Any,
    ) -> CollisionCheckResult:
        config = tuple(float(value) for value in joint_config)
        base = self._base_check(config, keyframe, **kwargs)
        if not base.valid:
            return base
        try:
            singular_values = self.kinematics.jacobian_singular_values(config)
        except Exception as error:  # noqa: BLE001 - backend boundary
            return CollisionCheckResult(
                valid=False,
                failure_code="JACOBIAN_EVALUATION_FAILED",
                detail=f"{type(error).__name__}: {error}",
                min_clearance_m=base.min_clearance_m,
                contacts=base.contacts,
            )
        if not singular_values or not all(
            math.isfinite(value) for value in singular_values
        ):
            return CollisionCheckResult(
                valid=False,
                failure_code="JACOBIAN_EVALUATION_FAILED",
                detail="Jacobian singular values are empty or non-finite",
                min_clearance_m=base.min_clearance_m,
                contacts=base.contacts,
            )
        minimum = min(singular_values)
        maximum = max(singular_values)
        condition = math.inf if minimum <= 0 else maximum / minimum
        if minimum < self.min_singular_value or condition > self.max_condition_number:
            return CollisionCheckResult(
                valid=False,
                failure_code="KINEMATIC_SINGULARITY",
                detail=(
                    f"Jacobian min singular value={minimum:.6g}, "
                    f"condition={condition:.6g}"
                ),
                min_clearance_m=base.min_clearance_m,
                contacts=base.contacts,
            )
        return base

    def __call__(
        self, joint_config: JointConfig, keyframe: RelativeKeyframeSpec
    ) -> bool:
        return self.check(joint_config, keyframe).valid


__all__ = ["KinematicSafetyValidator"]
