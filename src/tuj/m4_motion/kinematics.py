"""UR5e kinematics backed by the MuJoCo model robosuite actually loads.

Only what a feasibility oracle needs: "is there an IK solution inside the joint
limits for this end-effector pose?" No trajectories, no collision — those are
separate concerns handled elsewhere in the package.

Determinism is a hard requirement of the Task Planner contract, so IK restarts
come from a fixed seed ladder rather than random sampling.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np

DEFAULT_EEF_BODY = "right_hand"

# Deterministic restart ladder. Six joints; each row is a seed configuration in
# radians. Ordered cheapest-first: the home pose solves most in-workspace goals.
_SEED_LADDER: tuple[tuple[float, ...], ...] = (
    (0.0, -1.57, 1.57, -1.57, -1.57, 0.0),
    (0.0, -1.0, 1.0, -1.57, -1.57, 0.0),
    (1.57, -1.57, 1.57, -1.57, -1.57, 0.0),
    (-1.57, -1.57, 1.57, -1.57, -1.57, 0.0),
    (0.0, -2.2, 2.2, -1.57, -1.57, 0.0),
)


def default_model_path() -> str:
    """Path to the UR5e model inside the installed robosuite."""
    import robosuite

    return os.path.join(
        os.path.dirname(robosuite.__file__),
        "models",
        "assets",
        "robots",
        "ur5e",
        "robot.xml",
    )


@dataclass(frozen=True, slots=True)
class IKResult:
    solved: bool
    qpos: tuple[float, ...] = ()
    position_error_m: float = float("inf")
    orientation_error_rad: float = float("inf")
    iterations: int = 0
    detail: str = ""
    branch_id: str = ""
    seed_index: int = -1


@dataclass(frozen=True, slots=True)
class IKSolutionSet:
    """All distinct deterministic IK branches found for one Cartesian pose."""

    solutions: tuple[IKResult, ...] = ()
    best_position_error_m: float = float("inf")
    best_orientation_error_rad: float = float("inf")
    attempted_seeds: int = 0
    solver_id: str = "MUJOCO_DLS_MULTI_START"
    enumeration_complete: bool = False
    detail: str = ""

    @property
    def solved(self) -> bool:
        return bool(self.solutions)


def _quaternion_xyzw_to_matrix(values: Sequence[float]) -> np.ndarray:
    if len(values) != 4:
        raise ValueError("orientation quaternion must contain four values")
    x, y, z, w = (float(value) for value in values)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("orientation quaternion must be finite and non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper rotation matrix to a normalized xyzw quaternion."""
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m[2, 1] - m[1, 2]) / scale
        y = (m[0, 2] - m[2, 0]) / scale
        z = (m[1, 0] - m[0, 1]) / scale
    else:
        diagonal = int(np.argmax(np.diag(m)))
        if diagonal == 0:
            scale = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0)) * 2.0
            x = 0.25 * scale
            y = (m[0, 1] + m[1, 0]) / scale
            z = (m[0, 2] + m[2, 0]) / scale
            w = (m[2, 1] - m[1, 2]) / scale
        elif diagonal == 1:
            scale = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 0.0)) * 2.0
            x = (m[0, 1] + m[1, 0]) / scale
            y = 0.25 * scale
            z = (m[1, 2] + m[2, 1]) / scale
            w = (m[0, 2] - m[2, 0]) / scale
        else:
            scale = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 0.0)) * 2.0
            x = (m[0, 2] + m[2, 0]) / scale
            y = (m[1, 2] + m[2, 1]) / scale
            z = 0.25 * scale
            w = (m[1, 0] - m[0, 1]) / scale
    quaternion = np.asarray((x, y, z, w), dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    # q and -q are the same rotation. Canonicalize for stable artifacts.
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def _rotation_error_world(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """SO(3) logarithm of ``target * current.T`` in the world/base frame."""
    relative = target @ current.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-5:
        # The ordinary skew formula is singular at pi. Recover a stable axis
        # from the diagonal and use off-diagonal signs for its direction.
        axis = np.sqrt(np.maximum((np.diag(relative) + 1.0) * 0.5, 0.0))
        if axis[0] >= axis[1] and axis[0] >= axis[2] and axis[0] > 1e-9:
            axis[1] = math.copysign(axis[1], relative[0, 1] + relative[1, 0])
            axis[2] = math.copysign(axis[2], relative[0, 2] + relative[2, 0])
        elif axis[1] >= axis[2] and axis[1] > 1e-9:
            axis[0] = math.copysign(axis[0], relative[0, 1] + relative[1, 0])
            axis[2] = math.copysign(axis[2], relative[1, 2] + relative[2, 1])
        elif axis[2] > 1e-9:
            axis[0] = math.copysign(axis[0], relative[0, 2] + relative[2, 0])
            axis[1] = math.copysign(axis[1], relative[1, 2] + relative[2, 1])
        norm = float(np.linalg.norm(axis))
        return angle * axis / norm if norm > 1e-9 else np.array((angle, 0.0, 0.0))
    scale = angle / (2.0 * math.sin(angle))
    return scale * np.array(
        (
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        )
    )


def _wrapped_delta(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _branch_id(qpos: Sequence[float]) -> str:
    """Stable UR branch signature without claiming a cell-specific left/right convention."""

    def sign(value: float) -> str:
        sine = math.sin(value)
        if abs(sine) < 1e-5:
            return "0"
        return "+" if sine > 0.0 else "-"

    q = tuple(qpos)
    return f"S{sign(q[0])}_E{sign(q[2])}_W{sign(q[4])}"


class UR5eKinematics:
    """Damped least-squares IK against the MuJoCo UR5e.

    ``base_pos`` mirrors ``set_base_xpos`` in the experiment scene, so callers
    can pass world-frame targets and let this class handle the offset.
    """

    def __init__(
        self,
        model_path: str | None = None,
        base_pos: tuple[float, float, float] = (-0.45, 0.0, 0.0),
        eef_body: str = DEFAULT_EEF_BODY,
        target_position_in_eef_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        target_orientation_in_eef_xyzw: tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    ) -> None:
        self._model = mujoco.MjModel.from_xml_path(model_path or default_model_path())
        self._data = mujoco.MjData(self._model)
        self._base_pos = np.asarray(base_pos, dtype=float)
        self._eef_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, eef_body
        )
        if self._eef_id < 0:
            raise ValueError(f"end-effector body {eef_body!r} not found in the model")
        target_position = np.asarray(target_position_in_eef_m, dtype=float)
        if target_position.shape != (3,) or not np.all(np.isfinite(target_position)):
            raise ValueError("target_position_in_eef_m must contain three finite values")
        self._target_position_in_eef = target_position
        self._target_rotation_in_eef = _quaternion_xyzw_to_matrix(
            target_orientation_in_eef_xyzw
        )
        self._nq = self._model.nq
        limits = self._model.jnt_range[: self._nq].copy()
        unlimited = self._model.jnt_limited[: self._nq] == 0
        limits[unlimited] = (-np.pi * 2, np.pi * 2)
        self._lower, self._upper = limits[:, 0], limits[:, 1]

    @classmethod
    def from_robosuite_env(
        cls,
        env: object,
        *,
        model_path: str | None = None,
        eef_body: str = DEFAULT_EEF_BODY,
        target_position_in_eef_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        target_orientation_in_eef_xyzw: tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    ) -> "UR5eKinematics":
        """Use the world base transform of the UR5e loaded by robosuite.

        The lightweight IK model currently supports a translated, unrotated
        base.  Failing on a rotated base is safer than silently solving poses
        in the wrong frame.
        """

        try:
            scene_model = env.sim.model._model  # type: ignore[attr-defined]
            scene_data = env.sim.data._data  # type: ignore[attr-defined]
            robot = env.robots[0]  # type: ignore[attr-defined]
            root_body_name = str(robot.robot_model.root_body)
        except (AttributeError, IndexError, TypeError) as error:
            raise ValueError(
                "env must be a reset robosuite environment with one robot"
            ) from error
        root_id = mujoco.mj_name2id(
            scene_model, mujoco.mjtObj.mjOBJ_BODY, root_body_name
        )
        if root_id < 0:
            raise ValueError(
                f"robot root body {root_body_name!r} is absent from scene"
            )
        rotation = np.asarray(scene_data.xmat[root_id], dtype=float).reshape(3, 3)
        if not np.allclose(rotation, np.eye(3), atol=1e-9):
            raise ValueError(
                "UR5eKinematics does not yet support a rotated robot base"
            )
        base_position = tuple(
            float(value) for value in scene_data.xpos[root_id]
        )
        return cls(
            model_path=model_path,
            base_pos=base_position,
            eef_body=eef_body,
            target_position_in_eef_m=target_position_in_eef_m,
            target_orientation_in_eef_xyzw=target_orientation_in_eef_xyzw,
        )

    @property
    def base_position_m(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._base_pos)

    @property
    def joint_limits_rad(self) -> tuple[tuple[float, float], ...]:
        """Finite bounds used by sampling-based joint planners."""

        return tuple(
            (float(lower), float(upper))
            for lower, upper in zip(self._lower, self._upper)
        )

    # -- geometry helpers -------------------------------------------------- #

    @property
    def max_reach_m(self) -> float:
        """Farthest the EEF origin can get from the base, by construction."""
        return self._envelope()[1]

    def _envelope(self) -> tuple[float, float]:
        """(min, max) EEF distance from the base over the seed ladder + extremes.

        A cheap pre-IK filter: a target outside this shell can never be reached,
        so IK is not even attempted.
        """
        if getattr(self, "_cached_envelope", None) is None:
            distances = []
            for q in (*_SEED_LADDER, tuple(self._lower), tuple(self._upper)):
                pos = self._forward(np.asarray(q, dtype=float))
                distances.append(float(np.linalg.norm(pos)))
            # Fully extended arm: sum of the planar link contributions.
            stretched = self._forward(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            distances.append(float(np.linalg.norm(stretched)))
            self._cached_envelope = (0.0, max(distances))
        return self._cached_envelope

    def _forward(self, qpos: np.ndarray) -> np.ndarray:
        """EEF position in *base* frame for a joint configuration.

        ``mj_comPos`` is required as well as ``mj_kinematics``: ``mj_jacBody``
        reads the CoM-based dof axes it fills in, and without it the Jacobian
        stays zero and the IK never moves.
        """
        self._data.qpos[: self._nq] = qpos
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)
        return np.array(self._data.xpos[self._eef_id], dtype=float)

    def _forward_pose(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position = self._forward(qpos)
        rotation = np.array(self._data.xmat[self._eef_id], dtype=float).reshape(3, 3)
        return position, rotation

    def to_base_frame(self, world_pos) -> np.ndarray:
        return np.asarray(world_pos, dtype=float) - self._base_pos

    def forward_pose_world(
        self, qpos: Sequence[float]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Public FK helper used by tests and trajectory validation."""
        values = np.asarray(qpos, dtype=float)
        if values.shape != (self._nq,):
            raise ValueError(f"expected {self._nq} joint values, got {values.size}")
        eef_position, eef_rotation = self._forward_pose(values)
        position = (
            eef_position + eef_rotation @ self._target_position_in_eef
        )
        rotation = eef_rotation @ self._target_rotation_in_eef
        world_position = position + self._base_pos
        return (
            tuple(float(value) for value in world_position),
            _matrix_to_quaternion_xyzw(rotation),
        )

    def jacobian_singular_values(
        self, qpos: Sequence[float]
    ) -> tuple[float, ...]:
        """Return geometric-Jacobian singular values for singularity checks."""

        values = np.asarray(qpos, dtype=float)
        if values.shape != (self._nq,) or not np.all(np.isfinite(values)):
            raise ValueError(f"expected {self._nq} finite joint values")
        self._forward(values)
        jac_pos = np.zeros((3, self._model.nv))
        jac_rot = np.zeros((3, self._model.nv))
        mujoco.mj_jacBody(
            self._model, self._data, jac_pos, jac_rot, self._eef_id
        )
        geometric = np.vstack(
            (jac_pos[:, : self._nq], jac_rot[:, : self._nq])
        )
        return tuple(
            float(value)
            for value in np.linalg.svd(geometric, compute_uv=False)
        )

    # -- IK ---------------------------------------------------------------- #

    def solve_ik(
        self,
        world_pos,
        *,
        tolerance_m: float = 5e-3,
        max_iterations: int = 120,
        damping: float = 1e-2,
    ) -> IKResult:
        """Compatibility helper returning the first deterministic position IK hit.

        Task Planner's coarse feasibility oracle only needs an existential answer.
        Trajectory generation must call :meth:`solve_all_ik` and preserve every
        branch instead of using this method.
        """
        target = self.to_base_frame(world_pos)
        _, reach = self._envelope()
        distance = float(np.linalg.norm(target))
        if distance > reach + tolerance_m:
            return IKResult(
                solved=False,
                position_error_m=distance - reach,
                detail=(
                    f"target is {distance:.3f} m from the base, beyond the "
                    f"{reach:.3f} m envelope"
                ),
            )

        jac_pos = np.zeros((3, self._model.nv))
        jac_rot = np.zeros((3, self._model.nv))
        best = IKResult(False)
        for seed in _SEED_LADDER:
            qpos = np.clip(np.asarray(seed, dtype=float), self._lower, self._upper)
            for iteration in range(max_iterations):
                current = self._forward(qpos)
                error = target - current
                norm = float(np.linalg.norm(error))
                if norm < tolerance_m:
                    return IKResult(
                        solved=True,
                        qpos=tuple(float(v) for v in qpos),
                        position_error_m=norm,
                        orientation_error_rad=0.0,
                        iterations=iteration,
                        detail="converged",
                        branch_id=_branch_id(qpos),
                    )
                mujoco.mj_jacBody(
                    self._model, self._data, jac_pos, jac_rot, self._eef_id
                )
                j = jac_pos[:, : self._nq]
                # Damped least squares: (J^T J + kI)^-1 J^T e
                jjt = j @ j.T + damping * np.eye(3)
                delta = j.T @ np.linalg.solve(jjt, error)
                qpos = np.clip(qpos + delta, self._lower, self._upper)
                if norm < best.position_error_m:
                    best = IKResult(
                        solved=False,
                        qpos=tuple(float(v) for v in qpos),
                        position_error_m=norm,
                        iterations=iteration,
                        detail="stalled",
                    )
        return IKResult(
            solved=False,
            qpos=best.qpos,
            position_error_m=best.position_error_m,
            iterations=best.iterations,
            detail=f"no IK solution within {tolerance_m} m from any seed",
        )

    def solve_all_ik(
        self,
        world_pos: Sequence[float],
        orientation_xyzw: Sequence[float],
        *,
        seed_qpos: Sequence[float] | None = None,
        position_tolerance_m: float = 5e-3,
        orientation_tolerance_rad: float = 5e-2,
        max_iterations: int = 180,
        damping: float = 1e-3,
        duplicate_tolerance_rad: float = 2e-2,
    ) -> IKSolutionSet:
        """Return every distinct full-pose IK branch found from a fixed seed set.

        This is a deterministic multi-start numerical solver against the exact
        MuJoCo model loaded by the simulator.  It deliberately returns a set,
        because choosing one solution here would couple a keyframe's fate to a
        seed ordering and would make branch-continuous path planning impossible.
        """
        target_reference_position = self.to_base_frame(world_pos)
        target_reference_rotation = _quaternion_xyzw_to_matrix(
            orientation_xyzw
        )
        # Keyframe poses refer to the active EE's TCP / grasp reference.  The
        # lightweight UR5e model solves its ``right_hand`` body, so convert the
        # desired reference pose through the fixed MJCF-backed hand->TCP
        # transform before running multi-start IK.
        target_rotation = (
            target_reference_rotation @ self._target_rotation_in_eef.T
        )
        target_position = target_reference_position - (
            target_rotation @ self._target_position_in_eef
        )
        _, reach = self._envelope()
        distance = float(np.linalg.norm(target_position))
        if distance > reach + position_tolerance_m:
            return IKSolutionSet(
                best_position_error_m=distance - reach,
                attempted_seeds=0,
                detail=(
                    f"target is {distance:.3f} m from the base, beyond the "
                    f"{reach:.3f} m envelope"
                ),
            )

        # A Cartesian edge is a continuation problem, not a fresh global IK
        # query.  Put its previous joint state first so a locally continuous
        # solution is retained before enumerating the global UR branches.  A
        # fixed global seed ladder alone can miss the nearby branch at an
        # intermediate pose; joint interpolation to the remaining far-away
        # solution then makes the EEF leave the requested Cartesian line and
        # come back in a large, physically unsafe loop.
        seeds: list[tuple[float, ...]] = []
        if seed_qpos is not None:
            seed_values = np.asarray(seed_qpos, dtype=float)
            if seed_values.shape != (self._nq,) or not np.all(
                np.isfinite(seed_values)
            ):
                raise ValueError(
                    f"seed_qpos must contain {self._nq} finite joint values"
                )
            if np.any(seed_values < self._lower - 1e-9) or np.any(
                seed_values > self._upper + 1e-9
            ):
                raise ValueError("seed_qpos must remain inside physical joint limits")
            seeds.append(tuple(float(value) for value in seed_values))

        # Cover the three binary UR branch choices deterministically after the
        # local continuation seed.  Global pose queries keep the original
        # deterministic ordering when no seed is supplied.
        seeds.extend(_SEED_LADDER)
        for shoulder in (-1.57, 1.57):
            for elbow in (-1.57, 1.57):
                for wrist in (-1.57, 1.57):
                    seeds.append(
                        (
                            shoulder,
                            -1.57,
                            elbow,
                            -1.57,
                            wrist,
                            0.0,
                        )
                    )

        jac_pos = np.zeros((3, self._model.nv))
        jac_rot = np.zeros((3, self._model.nv))
        solutions: list[IKResult] = []
        best_position = float("inf")
        best_orientation = float("inf")

        for seed_index, seed in enumerate(seeds):
            qpos = np.clip(np.asarray(seed, dtype=float), self._lower, self._upper)
            for iteration in range(max_iterations):
                current_position, current_rotation = self._forward_pose(qpos)
                position_error = target_position - current_position
                orientation_error = _rotation_error_world(
                    current_rotation, target_rotation
                )
                position_norm = float(np.linalg.norm(position_error))
                orientation_norm = float(np.linalg.norm(orientation_error))
                best_position = min(best_position, position_norm)
                best_orientation = min(best_orientation, orientation_norm)

                if (
                    position_norm <= position_tolerance_m
                    and orientation_norm <= orientation_tolerance_rad
                ):
                    candidate = IKResult(
                        solved=True,
                        qpos=tuple(float(value) for value in qpos),
                        position_error_m=position_norm,
                        orientation_error_rad=orientation_norm,
                        iterations=iteration,
                        detail="converged",
                        branch_id=_branch_id(qpos),
                        seed_index=seed_index,
                    )
                    if not any(
                        np.max(np.abs(_wrapped_delta(candidate.qpos, known.qpos)))
                        <= duplicate_tolerance_rad
                        for known in solutions
                    ):
                        solutions.append(candidate)
                    break

                mujoco.mj_jacBody(
                    self._model, self._data, jac_pos, jac_rot, self._eef_id
                )
                jacobian = np.vstack(
                    (jac_pos[:, : self._nq], jac_rot[:, : self._nq])
                )
                error = np.concatenate((position_error, orientation_error))
                regularized = (
                    jacobian @ jacobian.T
                    + damping * np.eye(jacobian.shape[0])
                )
                try:
                    delta = jacobian.T @ np.linalg.solve(regularized, error)
                except np.linalg.LinAlgError:
                    break
                # Avoid a numerical restart jumping across multiple branches in
                # one iteration; branch discovery comes from the seed set.
                max_step = float(np.max(np.abs(delta)))
                if max_step > 0.25:
                    delta *= 0.25 / max_step
                qpos = np.clip(qpos + delta, self._lower, self._upper)

        solutions.sort(
            key=lambda item: (
                item.branch_id,
                tuple(round(value, 10) for value in item.qpos),
            )
        )
        return IKSolutionSet(
            solutions=tuple(solutions),
            best_position_error_m=best_position,
            best_orientation_error_rad=best_orientation,
            attempted_seeds=len(seeds),
            detail=(
                f"found {len(solutions)} distinct IK branches"
                if solutions
                else "no full-pose IK solution converged from deterministic seeds"
            ),
        )

    def is_reachable(self, world_pos, *, tolerance_m: float = 5e-3) -> bool:
        return self.solve_ik(world_pos, tolerance_m=tolerance_m).solved
