"""Deterministic path shortcutting and constraint-respecting time scaling."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from tuj.m4_motion.schema import JointDynamicLimit, TrajectoryWaypoint

JointConfig = tuple[float, ...]
MotionValidator = Callable[[JointConfig, JointConfig], bool]


class TrajectoryProcessingError(ValueError):
    pass


def unwrap_joint_path(
    path: Sequence[Sequence[float]],
    *,
    start_reference: Sequence[float] | None = None,
    joint_limits_rad: Sequence[tuple[float, float]] | None = None,
) -> tuple[JointConfig, ...]:
    """Choose the nearest 2π-equivalent representation of a joint path.

    When physical limits are supplied, every selected equivalent remains
    inside them.  A finite ±2π wrist must not be unwrapped as though it were
    an unlimited continuous joint.
    """

    states = [np.asarray(state, dtype=float) for state in path]
    if len(states) < 2:
        raise TrajectoryProcessingError("a path requires at least two states")
    shape = states[0].shape
    if len(shape) != 1 or shape[0] == 0 or any(state.shape != shape for state in states):
        raise TrajectoryProcessingError("all path states must use one non-zero DOF")
    if any(not np.all(np.isfinite(state)) for state in states):
        raise TrajectoryProcessingError("path contains non-finite joint values")
    limits: tuple[tuple[float, float], ...] | None = None
    if joint_limits_rad is not None:
        limits = tuple(
            (float(lower), float(upper))
            for lower, upper in joint_limits_rad
        )
        if len(limits) != shape[0] or any(
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower > upper
            for lower, upper in limits
        ):
            raise TrajectoryProcessingError(
                "joint position limits must match the finite path DOF"
            )

    if start_reference is None:
        reference = states[0]
    else:
        reference = np.asarray(start_reference, dtype=float)
        if reference.shape != shape or not np.all(np.isfinite(reference)):
            raise TrajectoryProcessingError(
                "start_reference must match the finite path DOF"
            )

    def nearest_equivalent(
        state: np.ndarray, previous: np.ndarray
    ) -> np.ndarray:
        if limits is None:
            delta = (state - previous + np.pi) % (2.0 * np.pi) - np.pi
            return previous + delta
        selected = np.empty_like(state)
        period = 2.0 * np.pi
        for index, (value, prior, (lower, upper)) in enumerate(
            zip(state, previous, limits)
        ):
            minimum_turn = math.ceil((lower - float(value)) / period - 1e-12)
            maximum_turn = math.floor((upper - float(value)) / period + 1e-12)
            if minimum_turn > maximum_turn:
                raise TrajectoryProcessingError(
                    f"joint {index} has no 2pi-equivalent value inside its limit"
                )
            candidates = tuple(
                float(value) + period * turn
                for turn in range(minimum_turn, maximum_turn + 1)
            )
            selected[index] = min(
                candidates,
                key=lambda candidate: abs(candidate - float(prior)),
            )
        return selected

    previous = nearest_equivalent(states[0], reference)
    result = [previous.copy()]
    for state in states[1:]:
        previous = nearest_equivalent(state, previous)
        result.append(previous.copy())
    return tuple(
        tuple(float(value) for value in state) for state in result
    )


def clamp_joint_limit_roundoff(
    path: Sequence[Sequence[float]],
    joint_limits_rad: Sequence[tuple[float, float]],
    *,
    tolerance_rad: float = 1e-3,
) -> tuple[JointConfig, ...]:
    """Clamp only sub-milliradian joint-limit numerical overshoot.

    Dense IK, 2π unwrapping, and floating-point interpolation can place a
    state microscopically outside a physical joint bound.  Clamping that
    roundoff is safe; a larger violation still fails closed instead of being
    hidden as a valid path.
    """

    if not math.isfinite(tolerance_rad) or tolerance_rad < 0.0:
        raise TrajectoryProcessingError(
            "joint-limit roundoff tolerance must be finite and non-negative"
        )
    limits = tuple((float(lower), float(upper)) for lower, upper in joint_limits_rad)
    if not limits or any(
        not math.isfinite(lower) or not math.isfinite(upper) or lower > upper
        for lower, upper in limits
    ):
        raise TrajectoryProcessingError("joint position limits are invalid")
    result: list[JointConfig] = []
    for state_index, state in enumerate(path):
        values = tuple(float(value) for value in state)
        if len(values) != len(limits) or not all(math.isfinite(value) for value in values):
            raise TrajectoryProcessingError(
                "joint path and position limits must have matching finite DOF"
            )
        clamped: list[float] = []
        for joint_index, (value, (lower, upper)) in enumerate(zip(values, limits)):
            if value < lower:
                if lower - value > tolerance_rad:
                    raise TrajectoryProcessingError(
                        f"path state {state_index} joint {joint_index} is below "
                        "its physical limit"
                    )
                value = lower
            elif value > upper:
                if value - upper > tolerance_rad:
                    raise TrajectoryProcessingError(
                        f"path state {state_index} joint {joint_index} is above "
                        "its physical limit"
                    )
                value = upper
            clamped.append(value)
        result.append(tuple(clamped))
    return tuple(result)


def deterministic_shortcut(
    path: Sequence[Sequence[float]],
    motion_is_valid: MotionValidator,
) -> tuple[JointConfig, ...]:
    """Greedily take the farthest valid shortcut while preserving endpoints."""
    states = [tuple(float(value) for value in state) for state in path]
    if len(states) < 2:
        raise TrajectoryProcessingError("a path requires at least two states")
    dof = len(states[0])
    if dof == 0 or any(len(state) != dof for state in states):
        raise TrajectoryProcessingError("all path states must use one non-zero DOF")
    result = [states[0]]
    index = 0
    while index < len(states) - 1:
        chosen: int | None = None
        for candidate in range(len(states) - 1, index, -1):
            if motion_is_valid(states[index], states[candidate]):
                chosen = candidate
                break
        if chosen is None:
            raise TrajectoryProcessingError(
                f"original path edge {index}->{index + 1} is invalid"
            )
        result.append(states[chosen])
        index = chosen
    return tuple(result)


def deviation_bounded_shortcut(
    path: Sequence[Sequence[float]],
    *,
    max_deviation_rad: float,
) -> tuple[JointConfig, ...]:
    """Remove redundant samples without materially changing a joint path.

    This is a deterministic Ramer-Douglas-Peucker reduction in joint space.
    Every removed configuration is at most ``max_deviation_rad`` (Euclidean
    joint distance) from the retained chord.  Unlike an unconstrained
    collision shortcut, this preserves the shape of Cartesian approach and
    insertion paths while avoiding a rest-to-rest stop at every dense IK
    validation sample.
    """

    states = [np.asarray(state, dtype=float) for state in path]
    if len(states) < 2:
        raise TrajectoryProcessingError("a path requires at least two states")
    if not math.isfinite(max_deviation_rad) or max_deviation_rad < 0:
        raise TrajectoryProcessingError(
            "max_deviation_rad must be finite and non-negative"
        )
    shape = states[0].shape
    if len(shape) != 1 or shape[0] == 0 or any(state.shape != shape for state in states):
        raise TrajectoryProcessingError("all path states must use one non-zero DOF")
    if any(not np.all(np.isfinite(state)) for state in states):
        raise TrajectoryProcessingError("path contains non-finite joint values")

    retained = {0, len(states) - 1}
    pending = [(0, len(states) - 1)]
    while pending:
        left_index, right_index = pending.pop()
        if right_index <= left_index + 1:
            continue
        left = states[left_index]
        chord = states[right_index] - left
        chord_squared = float(np.dot(chord, chord))
        farthest_index: int | None = None
        farthest_distance = -1.0
        for index in range(left_index + 1, right_index):
            if chord_squared <= 1e-24:
                projected = left
            else:
                fraction = float(
                    np.clip(
                        np.dot(states[index] - left, chord) / chord_squared,
                        0.0,
                        1.0,
                    )
                )
                projected = left + fraction * chord
            distance = float(np.linalg.norm(states[index] - projected))
            if distance > farthest_distance:
                farthest_distance = distance
                farthest_index = index
        if (
            farthest_index is not None
            and farthest_distance > max_deviation_rad
        ):
            retained.add(farthest_index)
            pending.append((left_index, farthest_index))
            pending.append((farthest_index, right_index))

    return tuple(
        tuple(float(value) for value in states[index])
        for index in sorted(retained)
    )


@dataclass(frozen=True, slots=True)
class TimeParameterizedPath:
    waypoints: tuple[TrajectoryWaypoint, ...]
    duration_s: float
    algorithm: str = "QUINTIC_STOP"


class QuinticTimeParameterizer:
    """Rest-to-rest quintic interpolation with analytic v/a/j bounds.

    This is deliberately conservative: every geometric path waypoint is a stop.
    It fills position, velocity, and acceleration now while keeping a narrow
    adapter boundary for replacing it with TOTG/TOPP-RA plus Ruckig later.
    """

    _MAX_H1 = 1.875
    _MAX_H2 = 5.773502691896258
    _MAX_H3 = 60.0

    def __init__(self, *, sample_dt_s: float = 0.02, min_segment_duration_s: float = 0.05) -> None:
        if sample_dt_s <= 0 or min_segment_duration_s <= 0:
            raise ValueError("trajectory timing intervals must be positive")
        self._sample_dt_s = sample_dt_s
        self._min_segment_duration_s = min_segment_duration_s

    def parameterize(
        self,
        joint_names: Sequence[str],
        path: Sequence[Sequence[float]],
        limits: Mapping[str, JointDynamicLimit],
        *,
        velocity_scaling: float = 1.0,
        acceleration_scaling: float = 1.0,
        jerk_scaling: float = 1.0,
        start_time_s: float = 0.0,
    ) -> TimeParameterizedPath:
        names = tuple(joint_names)
        states = [np.asarray(state, dtype=float) for state in path]
        if len(states) < 2:
            raise TrajectoryProcessingError("a path requires at least two states")
        if not names or any(state.shape != (len(names),) for state in states):
            raise TrajectoryProcessingError("path DOF must match joint_names")
        if not all(0 < value <= 1 for value in (velocity_scaling, acceleration_scaling, jerk_scaling)):
            raise TrajectoryProcessingError("dynamic scaling factors must be in (0, 1]")
        missing = [name for name in names if name not in limits]
        if missing:
            raise TrajectoryProcessingError(f"missing dynamic limits for joints {missing}")
        if not math.isfinite(start_time_s) or start_time_s < 0:
            raise TrajectoryProcessingError("start_time_s must be finite and non-negative")

        velocity_limits = np.asarray(
            [limits[name].max_velocity_rad_s * velocity_scaling for name in names]
        )
        acceleration_limits = np.asarray(
            [limits[name].max_acceleration_rad_s2 * acceleration_scaling for name in names]
        )
        jerk_limits = np.asarray(
            [
                (
                    limits[name].max_jerk_rad_s3 * jerk_scaling
                    if limits[name].max_jerk_rad_s3 is not None
                    else math.inf
                )
                for name in names
            ]
        )

        waypoints: list[TrajectoryWaypoint] = []
        clock = float(start_time_s)
        for segment_index, (source, target) in enumerate(zip(states, states[1:])):
            if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
                raise TrajectoryProcessingError("path contains non-finite joint values")
            delta = target - source
            displacement = np.abs(delta)
            duration_v = np.max(self._MAX_H1 * displacement / velocity_limits)
            duration_a = np.max(
                np.sqrt(self._MAX_H2 * displacement / acceleration_limits)
            )
            finite_jerk = np.isfinite(jerk_limits)
            duration_j = (
                np.max(
                    np.cbrt(
                        self._MAX_H3
                        * displacement[finite_jerk]
                        / jerk_limits[finite_jerk]
                    )
                )
                if np.any(finite_jerk)
                else 0.0
            )
            duration = max(
                self._min_segment_duration_s,
                float(duration_v),
                float(duration_a),
                float(duration_j),
            )
            sample_count = max(1, int(math.ceil(duration / self._sample_dt_s)))
            for sample_index in range(sample_count + 1):
                if segment_index > 0 and sample_index == 0:
                    continue
                s = sample_index / sample_count
                h = 10 * s**3 - 15 * s**4 + 6 * s**5
                h1 = 30 * s**2 - 60 * s**3 + 30 * s**4
                h2 = 60 * s - 180 * s**2 + 120 * s**3
                position = source + delta * h
                velocity = delta * h1 / duration
                acceleration = delta * h2 / (duration * duration)
                time_from_start = clock + duration * s
                waypoints.append(
                    TrajectoryWaypoint(
                        time_from_start_s=float(time_from_start),
                        joint_positions_rad=[float(value) for value in position],
                        joint_velocities_rad_s=[float(value) for value in velocity],
                        joint_accelerations_rad_s2=[
                            float(value) for value in acceleration
                        ],
                    )
                )
            clock += duration

        return TimeParameterizedPath(
            waypoints=tuple(waypoints),
            duration_s=clock - start_time_s,
        )
