from __future__ import annotations

from tuj.m5_motion.mujoco_collision import CollisionCheckResult
from tuj.m5_motion.safety import KinematicSafetyValidator
from tuj.m5_motion.schema import (
    KeyframePlannerType,
    KeyframeType,
    RelativeKeyframeSpec,
)


def _keyframe() -> RelativeKeyframeSpec:
    return RelativeKeyframeSpec(
        keyframe_id="target",
        keyframe_type=KeyframeType.CUSTOM,
        frame_ref="world",
        anchor="origin",
        approach_axis_xyz=(0.0, 0.0, 1.0),
        planner=KeyframePlannerType.JOINT,
    )


class _BaseValidator:
    def check(self, joint_config, keyframe=None, **kwargs):
        del joint_config, keyframe, kwargs
        return CollisionCheckResult(valid=True, min_clearance_m=0.02)


class _Kinematics:
    def __init__(self, values) -> None:
        self.values = values

    def jacobian_singular_values(self, qpos):
        del qpos
        return self.values


def test_singularity_is_rejected_after_collision_passes() -> None:
    validator = KinematicSafetyValidator(
        _BaseValidator(),
        _Kinematics((1.0, 0.5, 1e-6)),
        min_singular_value=1e-4,
        max_condition_number=1e4,
    )

    result = validator.check((0.0, 0.0, 0.0), _keyframe())

    assert not result.valid
    assert result.failure_code == "KINEMATIC_SINGULARITY"
    assert result.min_clearance_m == 0.02


def test_well_conditioned_state_preserves_collision_result() -> None:
    validator = KinematicSafetyValidator(
        _BaseValidator(),
        _Kinematics((1.0, 0.5, 0.1)),
    )

    result = validator.check((0.0, 0.0, 0.0), _keyframe())

    assert result.valid
    assert result.min_clearance_m == 0.02
