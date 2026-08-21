"""Planner-specific edge generation for Cartesian and free-space motion.

The branch selector asks for one incoming edge per keyframe.  This module
dispatches that request according to ``RelativeKeyframeSpec.planner`` instead
of treating the planner label as metadata only.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from motion_planner.geometry import RelativePoseResolver
from motion_planner.schema import (
    KeyframePlannerType,
    RelativeKeyframeSpec,
    WorldSnapshot,
)
from motion_planner.strategy import (
    EdgePlanResult,
    EdgePlanner,
    JointConfig,
    StateValidator,
    wrapped_joint_delta,
    wrapped_joint_distance,
)


class CartesianKinematics(Protocol):
    def forward_pose_world(
        self, qpos: Sequence[float]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]: ...

    def solve_all_ik(
        self,
        world_pos: Sequence[float],
        orientation_xyzw: Sequence[float],
        **kwargs: Any,
    ) -> Any: ...


def _state_report(
    validator: StateValidator,
    config: JointConfig,
    keyframe: RelativeKeyframeSpec,
) -> tuple[bool, str | None, str, float | None]:
    check = getattr(validator, "check", None)
    if callable(check):
        result = check(config, keyframe)
        return (
            bool(result.valid),
            result.failure_code,
            result.detail,
            result.min_clearance_m,
        )
    return bool(validator(config, keyframe)), None, "", None


def validate_joint_segment(
    source: Sequence[float],
    target: Sequence[float],
    keyframe: RelativeKeyframeSpec,
    validator: StateValidator,
    *,
    max_joint_step_rad: float,
) -> EdgePlanResult:
    """Conservatively sample an entire joint segment, including endpoints."""

    if max_joint_step_rad <= 0:
        raise ValueError("max_joint_step_rad must be positive")
    if len(source) != len(target):
        raise ValueError("joint configurations must have the same DOF")
    delta = wrapped_joint_delta(source, target)
    steps = max(
        1,
        int(math.ceil(float(np.max(np.abs(delta))) / max_joint_step_rad)),
    )
    start = np.asarray(source, dtype=float)
    path: list[JointConfig] = []
    minimum: float | None = None
    for index in range(steps + 1):
        config = tuple(float(value) for value in start + delta * (index / steps))
        valid, failure_code, detail, clearance = _state_report(
            validator, config, keyframe
        )
        if clearance is not None:
            minimum = clearance if minimum is None else min(minimum, clearance)
        if not valid:
            return EdgePlanResult(
                valid=False,
                failure_code=failure_code or "SWEPT_STATE_INVALID",
                detail=f"invalid swept sample {index}/{steps}: {detail}",
                min_clearance_m=minimum,
            )
        path.append(config)
    return EdgePlanResult(
        valid=True,
        joint_path=tuple(path),
        min_clearance_m=minimum,
    )


def _quaternion_angle(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))


def _slerp(
    left: Sequence[float], right: Sequence[float], fraction: float
) -> tuple[float, float, float, float]:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        value = a + fraction * (b - a)
        value /= np.linalg.norm(value)
    else:
        angle = math.acos(dot)
        sine = math.sin(angle)
        value = (
            math.sin((1.0 - fraction) * angle) / sine * a
            + math.sin(fraction * angle) / sine * b
        )
    return tuple(float(item) for item in value)


@dataclass(slots=True)
class CartesianEdgePlanner:
    """Follow a straight SE(3) line and solve a continuous IK branch along it."""

    kinematics: CartesianKinematics
    world: WorldSnapshot
    state_validator: StateValidator
    translation_step_m: float = 0.01
    rotation_step_rad: float = 0.1
    max_joint_step_rad: float = 0.02

    def __post_init__(self) -> None:
        if min(
            self.translation_step_m,
            self.rotation_step_rad,
            self.max_joint_step_rad,
        ) <= 0:
            raise ValueError("Cartesian planner step sizes must be positive")

    def plan(
        self,
        source: JointConfig,
        target: JointConfig,
        source_keyframe: RelativeKeyframeSpec | None,
        target_keyframe: RelativeKeyframeSpec,
    ) -> EdgePlanResult:
        del source_keyframe
        try:
            start_position, start_orientation = (
                self.kinematics.forward_pose_world(source)
            )
            target_pose = RelativePoseResolver(self.world).resolve(target_keyframe)
        except Exception as error:  # noqa: BLE001 - kinematics backends vary
            return EdgePlanResult(
                valid=False,
                failure_code="CARTESIAN_ENDPOINT_UNAVAILABLE",
                detail=f"{type(error).__name__}: {error}",
            )

        start_xyz = np.asarray(start_position, dtype=float)
        target_xyz = np.asarray(target_pose.position_m, dtype=float)
        translation = float(np.linalg.norm(target_xyz - start_xyz))
        rotation = _quaternion_angle(
            start_orientation, target_pose.orientation_xyzw
        )
        steps = max(
            1,
            int(math.ceil(translation / self.translation_step_m)),
            int(math.ceil(rotation / self.rotation_step_rad)),
        )
        previous = tuple(float(value) for value in source)
        path: list[JointConfig] = [previous]
        minimum: float | None = None
        for index in range(1, steps + 1):
            fraction = index / steps
            position = start_xyz + fraction * (target_xyz - start_xyz)
            orientation = _slerp(
                start_orientation, target_pose.orientation_xyzw, fraction
            )
            if index == steps:
                selected = tuple(float(value) for value in target)
            else:
                solutions = self.kinematics.solve_all_ik(position, orientation)
                valid_solutions = []
                for solution in solutions.solutions:
                    candidate = tuple(float(value) for value in solution.qpos)
                    valid, _, _, _ = _state_report(
                        self.state_validator, candidate, target_keyframe
                    )
                    if valid:
                        valid_solutions.append(candidate)
                if not valid_solutions:
                    return EdgePlanResult(
                        valid=False,
                        failure_code="CARTESIAN_INTERMEDIATE_IK_FAILED",
                        detail=f"no valid IK at Cartesian sample {index}/{steps}",
                        min_clearance_m=minimum,
                    )
                selected = min(
                    valid_solutions,
                    key=lambda candidate: wrapped_joint_distance(
                        previous, candidate
                    ),
                )
            segment = validate_joint_segment(
                previous,
                selected,
                target_keyframe,
                self.state_validator,
                max_joint_step_rad=self.max_joint_step_rad,
            )
            if not segment.valid:
                return EdgePlanResult(
                    valid=False,
                    failure_code=segment.failure_code,
                    detail=f"Cartesian sample {index}/{steps}: {segment.detail}",
                    min_clearance_m=(
                        segment.min_clearance_m
                        if minimum is None
                        else min(
                            minimum,
                            segment.min_clearance_m
                            if segment.min_clearance_m is not None
                            else minimum,
                        )
                    ),
                )
            if segment.min_clearance_m is not None:
                minimum = (
                    segment.min_clearance_m
                    if minimum is None
                    else min(minimum, segment.min_clearance_m)
                )
            path.extend(segment.joint_path[1:])
            # Keep the unwrapped endpoint returned by validation.  Resetting to
            # an analytically equivalent [-π, π] IK value here introduces a 2π
            # discontinuity between adjacent Cartesian samples.
            previous = segment.joint_path[-1]
        return EdgePlanResult(
            valid=True,
            joint_path=tuple(path),
            min_clearance_m=minimum,
        )


@dataclass(frozen=True, slots=True)
class _TreeNode:
    q: JointConfig
    parent: int | None


@dataclass(slots=True)
class RRTConnectEdgePlanner:
    """Deterministic, bounded bidirectional RRT-Connect in joint space."""

    state_validator: StateValidator
    joint_limits_rad: Sequence[tuple[float, float]]
    random_seed: int = 0
    max_iterations: int = 2000
    timeout_s: float = 5.0
    extension_step_rad: float = 0.25
    validation_step_rad: float = 0.02
    goal_bias: float = 0.15

    def __post_init__(self) -> None:
        if not self.joint_limits_rad:
            raise ValueError("RRTConnect requires joint limits")
        if self.max_iterations < 1 or self.timeout_s <= 0:
            raise ValueError("RRTConnect budget must be positive")
        if self.extension_step_rad <= 0 or self.validation_step_rad <= 0:
            raise ValueError("RRTConnect step sizes must be positive")
        if not 0.0 <= self.goal_bias <= 1.0:
            raise ValueError("goal_bias must be in [0, 1]")
        if any(
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower >= upper
            for lower, upper in self.joint_limits_rad
        ):
            raise ValueError("RRTConnect joint limits must be finite intervals")

    @staticmethod
    def _root_path(tree: list[_TreeNode], index: int) -> list[JointConfig]:
        result: list[JointConfig] = []
        while True:
            node = tree[index]
            result.append(node.q)
            if node.parent is None:
                break
            index = node.parent
        result.reverse()
        return result

    def _nearest(self, tree: list[_TreeNode], target: JointConfig) -> int:
        return min(
            range(len(tree)),
            key=lambda index: wrapped_joint_distance(tree[index].q, target),
        )

    def _steer(self, source: JointConfig, target: JointConfig) -> JointConfig:
        delta = wrapped_joint_delta(source, target)
        distance = float(np.linalg.norm(delta))
        if distance <= self.extension_step_rad:
            return target
        value = np.asarray(source, dtype=float) + (
            delta * (self.extension_step_rad / distance)
        )
        return tuple(float(item) for item in value)

    def _extend(
        self,
        tree: list[_TreeNode],
        target: JointConfig,
        keyframe: RelativeKeyframeSpec,
    ) -> tuple[str, int | None, float | None]:
        nearest_index = self._nearest(tree, target)
        candidate = self._steer(tree[nearest_index].q, target)
        segment = validate_joint_segment(
            tree[nearest_index].q,
            candidate,
            keyframe,
            self.state_validator,
            max_joint_step_rad=self.validation_step_rad,
        )
        if not segment.valid:
            return "TRAPPED", None, segment.min_clearance_m
        tree.append(_TreeNode(q=candidate, parent=nearest_index))
        reached = wrapped_joint_distance(candidate, target) <= 1e-9
        return ("REACHED" if reached else "ADVANCED"), len(tree) - 1, segment.min_clearance_m

    def _connect(
        self,
        tree: list[_TreeNode],
        target: JointConfig,
        keyframe: RelativeKeyframeSpec,
    ) -> tuple[bool, int | None, float | None]:
        minimum: float | None = None
        while True:
            status, index, clearance = self._extend(tree, target, keyframe)
            if clearance is not None:
                minimum = clearance if minimum is None else min(minimum, clearance)
            if status == "TRAPPED":
                return False, None, minimum
            if status == "REACHED":
                return True, index, minimum

    def plan(
        self,
        source: JointConfig,
        target: JointConfig,
        source_keyframe: RelativeKeyframeSpec | None,
        target_keyframe: RelativeKeyframeSpec,
    ) -> EdgePlanResult:
        del source_keyframe
        if len(source) != len(self.joint_limits_rad) or len(target) != len(
            self.joint_limits_rad
        ):
            raise ValueError("RRTConnect joint limit DOF mismatch")
        direct = validate_joint_segment(
            source,
            target,
            target_keyframe,
            self.state_validator,
            max_joint_step_rad=self.validation_step_rad,
        )
        if direct.valid:
            return direct

        rng = np.random.default_rng(self.random_seed)
        lower = np.asarray([item[0] for item in self.joint_limits_rad])
        upper = np.asarray([item[1] for item in self.joint_limits_rad])
        start_tree = [_TreeNode(tuple(source), None)]
        goal_tree = [_TreeNode(tuple(target), None)]
        tree_a, tree_b = start_tree, goal_tree
        a_is_start = True
        started = time.monotonic()
        # Clearance from the rejected straight line does not describe the
        # alternate path eventually selected by RRT-Connect.
        minimum: float | None = None
        for iteration in range(self.max_iterations):
            if time.monotonic() - started >= self.timeout_s:
                return EdgePlanResult(
                    valid=False,
                    failure_code="RRT_CONNECT_TIMEOUT",
                    detail=f"timed out after {iteration} iterations",
                    min_clearance_m=minimum,
                )
            if rng.random() < self.goal_bias:
                sample = tree_b[0].q
            else:
                sample = tuple(float(value) for value in rng.uniform(lower, upper))
            status, new_index, clearance = self._extend(
                tree_a, sample, target_keyframe
            )
            if clearance is not None:
                minimum = clearance if minimum is None else min(minimum, clearance)
            if status != "TRAPPED" and new_index is not None:
                reached, other_index, connect_clearance = self._connect(
                    tree_b, tree_a[new_index].q, target_keyframe
                )
                if connect_clearance is not None:
                    minimum = (
                        connect_clearance
                        if minimum is None
                        else min(minimum, connect_clearance)
                    )
                if reached and other_index is not None:
                    a_path = self._root_path(tree_a, new_index)
                    b_path = self._root_path(tree_b, other_index)
                    if a_is_start:
                        path = a_path + list(reversed(b_path))[1:]
                    else:
                        path = b_path + list(reversed(a_path))[1:]
                    return EdgePlanResult(
                        valid=True,
                        joint_path=tuple(path),
                        min_clearance_m=minimum,
                    )
            tree_a, tree_b = tree_b, tree_a
            a_is_start = not a_is_start
        return EdgePlanResult(
            valid=False,
            failure_code="RRT_CONNECT_EXHAUSTED",
            detail=f"no connection after {self.max_iterations} iterations",
            min_clearance_m=minimum,
        )


@dataclass(frozen=True, slots=True)
class PlannerDispatchEdgePlanner:
    """Route each incoming keyframe edge to its declared planning algorithm."""

    joint: EdgePlanner
    cartesian: EdgePlanner
    sampling_based: EdgePlanner

    def plan(
        self,
        source: JointConfig,
        target: JointConfig,
        source_keyframe: RelativeKeyframeSpec | None,
        target_keyframe: RelativeKeyframeSpec,
    ) -> EdgePlanResult:
        planners = {
            KeyframePlannerType.JOINT: self.joint,
            KeyframePlannerType.CARTESIAN: self.cartesian,
            KeyframePlannerType.SAMPLING_BASED: self.sampling_based,
        }
        return planners[target_keyframe.planner].plan(
            source, target, source_keyframe, target_keyframe
        )


__all__ = [
    "CartesianEdgePlanner",
    "PlannerDispatchEdgePlanner",
    "RRTConnectEdgePlanner",
    "validate_joint_segment",
]
