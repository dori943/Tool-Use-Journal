"""Runtime grasp / EE events and deterministic MotionPlan playback.

MuJoCo model topology is immutable after compilation.  A real gripper change
therefore cannot be represented by changing an ACM flag or a body pose inside
one ``MjModel``.  This runtime creates the matching Tool-Use-Journal
environment variant (bare / 2F / 3F / vac), transfers every common named joint
state, and atomically replaces the live environment at TOOL_UNLOCK/TOOL_LOCK.

Two players share the same event runtime.  The kinematic player applies timed
joint samples directly to MuJoCo qpos.  The controller player feeds absolute
joint targets to robosuite's ``JointPositionController`` and advances the real
torque / actuator physics loop.  Object attachment either projects a rigid
grasp transform or applies a mass-scaled, breakable 6-DoF penalty weld whose
contact force and wrench limits can produce ``GRASP_LOST``.
"""

from __future__ import annotations

import math
import time
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np

from motion_planner.mujoco_collision import MuJoCoCollisionModelRegistry
from motion_planner.schema import (
    ArtifactProvenance,
    EventExecutionStatus,
    EventType,
    ExecutedEvent,
    ExecutionReport,
    ExecutionStatus,
    FailureObservation,
    GripperMode,
    GripperState,
    ModuleName,
    MotionPlan,
    RobotState,
    SimulationMetrics,
    SimulationRun,
    TrajectoryEvent,
    TrajectorySegment,
    TrajectoryWaypoint,
)
from motion_planner.tool_use_journal import (
    TOOL_USE_JOURNAL_TESTED_REVISION,
    ToolUseJournalCompatibilityError,
    ToolUseJournalEnvironmentAdapter,
    _descendant_body_ids,
    _name,
    _physical_ee,
    _raw_model_data,
    make_tool_use_journal_env,
)


class ToolUseJournalRuntimeError(RuntimeError):
    """An EE event or state-preserving model transition failed."""


class ToolUseJournalAttachmentBroken(ToolUseJournalRuntimeError):
    """A breakable grasp exceeded its force / contact contract."""

    def __init__(self, observation: "AttachmentBreakObservation") -> None:
        self.observation = observation
        super().__init__(
            f"grasp of {observation.object_id!r} broke: "
            + ", ".join(observation.reasons)
        )


class CollisionProbe(Protocol):
    def check(
        self,
        joint_config: Sequence[float],
        *,
        context: Any,
    ) -> Any: ...


def tool_use_journal_joint_position_controller_config(
    *,
    kp: float = 50.0,
    damping_ratio: float = 1.0,
) -> dict[str, Any]:
    """Return the robosuite config required by the controller-backed player.

    The target environments normally use an OSC pose controller.  MotionPlan
    contains joint-space trajectories, so replay uses absolute joint targets
    and leaves the environment-specific GRIP controller in place.
    """

    if (
        isinstance(kp, bool)
        or not isinstance(kp, (int, float))
        or not math.isfinite(float(kp))
        or not 0.0 < float(kp) <= 300.0
    ):
        raise ValueError("kp must be finite and within (0, 300]")
    if (
        isinstance(damping_ratio, bool)
        or not isinstance(damping_ratio, (int, float))
        or not math.isfinite(float(damping_ratio))
        or not 0.0 < float(damping_ratio) <= 10.0
    ):
        raise ValueError("damping_ratio must be finite and within (0, 10]")

    from robosuite.controllers import load_composite_controller_config

    config = load_composite_controller_config(robot="UR5e")
    gripper = dict(
        config.get("body_parts", {})
        .get("right", {})
        .get("gripper", {"type": "GRIP"})
    )
    config["body_parts"]["right"] = {
        "type": "JOINT_POSITION",
        "input_type": "absolute",
        # Absolute UR5e targets are radians, not normalized policy actions.
        "input_max": 2.0 * math.pi,
        "input_min": -2.0 * math.pi,
        "output_max": 0.05,
        "output_min": -0.05,
        "kp": float(kp),
        "damping_ratio": float(damping_ratio),
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "qpos_limits": None,
        "interpolation": None,
        "ramp_ratio": 0.2,
        "gripper": gripper,
    }
    return config


def _joint_qpos_width(joint_type: int) -> int:
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def _joint_dof_width(joint_type: int) -> int:
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 6
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 3
    return 1


@dataclass(frozen=True, slots=True)
class _RuntimeState:
    qpos_by_joint: Mapping[str, tuple[float, ...]]
    qvel_by_joint: Mapping[str, tuple[float, ...]]
    ctrl_by_actuator: Mapping[str, float]
    simulation_time_s: float


def _capture_runtime_state(env: object) -> _RuntimeState:
    model, data = _raw_model_data(env)
    qpos: dict[str, tuple[float, ...]] = {}
    qvel: dict[str, tuple[float, ...]] = {}
    for joint_id in range(model.njnt):
        name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            continue
        joint_type = int(model.jnt_type[joint_id])
        qpos_start = int(model.jnt_qposadr[joint_id])
        qvel_start = int(model.jnt_dofadr[joint_id])
        qpos_width = _joint_qpos_width(joint_type)
        qvel_width = _joint_dof_width(joint_type)
        qpos[name] = tuple(
            float(value)
            for value in data.qpos[qpos_start : qpos_start + qpos_width]
        )
        qvel[name] = tuple(
            float(value)
            for value in data.qvel[qvel_start : qvel_start + qvel_width]
        )
    controls = {
        _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id): float(
            data.ctrl[actuator_id]
        )
        for actuator_id in range(model.nu)
        if _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    }
    return _RuntimeState(
        qpos_by_joint=qpos,
        qvel_by_joint=qvel,
        ctrl_by_actuator=controls,
        simulation_time_s=float(data.time),
    )


def _restore_runtime_state(env: object, state: _RuntimeState) -> None:
    model, data = _raw_model_data(env)
    for joint_id in range(model.njnt):
        name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            continue
        joint_type = int(model.jnt_type[joint_id])
        qpos_values = state.qpos_by_joint.get(name)
        qvel_values = state.qvel_by_joint.get(name)
        if qpos_values is not None:
            width = _joint_qpos_width(joint_type)
            if len(qpos_values) != width:
                raise ToolUseJournalRuntimeError(
                    f"joint {name!r} qpos width changed during EE transition"
                )
            start = int(model.jnt_qposadr[joint_id])
            data.qpos[start : start + width] = qpos_values
        if qvel_values is not None:
            width = _joint_dof_width(joint_type)
            if len(qvel_values) != width:
                raise ToolUseJournalRuntimeError(
                    f"joint {name!r} qvel width changed during EE transition"
                )
            start = int(model.jnt_dofadr[joint_id])
            data.qvel[start : start + width] = qvel_values
    for actuator_id in range(model.nu):
        name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if name in state.ctrl_by_actuator:
            data.ctrl[actuator_id] = state.ctrl_by_actuator[name]
    data.time = state.simulation_time_s
    mujoco.mj_forward(model, data)


@dataclass(frozen=True, slots=True)
class EERuntimeTransition:
    from_ee: str | None
    to_ee: str | None
    simulation_time_s: float
    preserved_joint_names: tuple[str, ...]
    hidden_rack_ee: str | None


class AttachmentMode(str, Enum):
    KINEMATIC = "KINEMATIC"
    BREAKABLE_WELD = "BREAKABLE_WELD"


@dataclass(frozen=True, slots=True)
class BreakableWeldConfig:
    """6-DoF penalty-weld gains and grasp failure thresholds."""

    linear_stiffness_n_m: float = 600.0
    linear_damping_ns_m: float = 35.0
    angular_stiffness_nm_rad: float = 12.0
    angular_damping_nms_rad: float = 0.15
    natural_frequency_hz: float = 8.0
    damping_ratio: float = 1.0
    max_weld_force_n: float = 40.0
    max_weld_torque_nm: float = 12.0
    max_position_error_m: float = 0.05
    max_orientation_error_rad: float = 0.6
    max_contact_force_n: float = 250.0
    require_contact: bool = True
    require_retention_contact: bool = True
    min_contact_count: int = 1
    min_normal_force_n: float = 0.0
    required_contact_groups: tuple[str, ...] = ()
    startup_grace_steps: int = 10
    contact_loss_grace_steps: int = 5
    break_debounce_steps: int = 3

    def __post_init__(self) -> None:
        positive = {
            "linear_stiffness_n_m": self.linear_stiffness_n_m,
            "linear_damping_ns_m": self.linear_damping_ns_m,
            "angular_stiffness_nm_rad": self.angular_stiffness_nm_rad,
            "angular_damping_nms_rad": self.angular_damping_nms_rad,
            "natural_frequency_hz": self.natural_frequency_hz,
            "damping_ratio": self.damping_ratio,
            "max_weld_force_n": self.max_weld_force_n,
            "max_weld_torque_nm": self.max_weld_torque_nm,
            "max_position_error_m": self.max_position_error_m,
            "max_orientation_error_rad": self.max_orientation_error_rad,
            "max_contact_force_n": self.max_contact_force_n,
        }
        invalid: list[str] = []
        for name, value in positive.items():
            try:
                valid = math.isfinite(value) and value > 0.0
            except TypeError:
                valid = False
            if not valid:
                invalid.append(name)
        if invalid:
            raise ValueError(
                f"breakable weld parameters must be finite and positive: {invalid}"
            )
        if not isinstance(self.require_contact, bool):
            raise ValueError("require_contact must be boolean")
        if not isinstance(self.require_retention_contact, bool):
            raise ValueError("require_retention_contact must be boolean")
        if (
            not isinstance(self.min_contact_count, int)
            or isinstance(self.min_contact_count, bool)
            or self.min_contact_count <= 0
        ):
            raise ValueError("min_contact_count must be a positive integer")
        if (
            isinstance(self.min_normal_force_n, bool)
            or not isinstance(self.min_normal_force_n, (int, float))
            or not math.isfinite(float(self.min_normal_force_n))
            or self.min_normal_force_n < 0.0
        ):
            raise ValueError("min_normal_force_n must be finite and non-negative")
        if (
            not isinstance(self.required_contact_groups, tuple)
            or any(
                not isinstance(group, str) or not group.strip()
                for group in self.required_contact_groups
            )
            or len(set(self.required_contact_groups))
            != len(self.required_contact_groups)
        ):
            raise ValueError(
                "required_contact_groups must be a tuple of unique names"
            )
        if (
            not isinstance(self.startup_grace_steps, int)
            or isinstance(self.startup_grace_steps, bool)
            or self.startup_grace_steps < 0
        ):
            raise ValueError("startup_grace_steps must be non-negative")
        if (
            not isinstance(self.contact_loss_grace_steps, int)
            or isinstance(self.contact_loss_grace_steps, bool)
            or self.contact_loss_grace_steps < 0
        ):
            raise ValueError("contact_loss_grace_steps must be non-negative")
        if (
            not isinstance(self.break_debounce_steps, int)
            or isinstance(self.break_debounce_steps, bool)
            or self.break_debounce_steps <= 0
        ):
            raise ValueError("break_debounce_steps must be positive")

    @classmethod
    def from_parameters(
        cls, parameters: Mapping[str, Any]
    ) -> "BreakableWeldConfig":
        fields = {
            "linear_stiffness_n_m",
            "linear_damping_ns_m",
            "angular_stiffness_nm_rad",
            "angular_damping_nms_rad",
            "natural_frequency_hz",
            "damping_ratio",
            "max_weld_force_n",
            "max_weld_torque_nm",
            "max_position_error_m",
            "max_orientation_error_rad",
            "max_contact_force_n",
            "require_contact",
            "require_retention_contact",
            "min_contact_count",
            "min_normal_force_n",
            "required_contact_groups",
            "startup_grace_steps",
            "contact_loss_grace_steps",
            "break_debounce_steps",
        }
        values = {name: parameters[name] for name in fields if name in parameters}
        for boolean_name in ("require_contact", "require_retention_contact"):
            if boolean_name in values and not isinstance(
                values[boolean_name], bool
            ):
                raise ValueError(f"{boolean_name} must be boolean")
        for integer_name in (
            "min_contact_count",
            "startup_grace_steps",
            "contact_loss_grace_steps",
            "break_debounce_steps",
        ):
            if integer_name in values and (
                not isinstance(values[integer_name], int)
                or isinstance(values[integer_name], bool)
            ):
                raise ValueError(f"{integer_name} must be an integer")
        if "required_contact_groups" in values:
            raw_groups = values["required_contact_groups"]
            if (
                not isinstance(raw_groups, (list, tuple))
                or isinstance(raw_groups, (str, bytes))
            ):
                raise ValueError("required_contact_groups must be a list of names")
            values["required_contact_groups"] = tuple(raw_groups)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class AttachmentContactMetrics:
    contact_count: int = 0
    normal_force_n: float = 0.0
    tangential_force_n: float = 0.0
    total_force_n: float = 0.0
    contact_groups: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AttachmentBreakObservation:
    object_id: str
    reasons: tuple[str, ...]
    simulation_time_s: float
    required_force_n: float
    required_torque_nm: float
    position_error_m: float
    orientation_error_rad: float
    contact_count: int
    contact_force_n: float
    contact_groups: tuple[str, ...] = ()

    def as_mapping(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "reasons": list(self.reasons),
            "simulation_time_s": self.simulation_time_s,
            "required_force_n": self.required_force_n,
            "required_torque_nm": self.required_torque_nm,
            "position_error_m": self.position_error_m,
            "orientation_error_rad": self.orientation_error_rad,
            "contact_count": self.contact_count,
            "contact_force_n": self.contact_force_n,
            "contact_groups": list(self.contact_groups),
        }


@dataclass(frozen=True, slots=True)
class AttachedObjectState:
    """Rigid object-to-grasp-frame relation maintained by the runtime."""

    object_id: str
    free_joint_name: str
    reference_kind: str
    reference_name: str
    position_in_reference_m: tuple[float, float, float]
    rotation_in_reference: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    attach_distance_m: float
    mode: AttachmentMode = AttachmentMode.KINEMATIC
    breakable_weld: BreakableWeldConfig | None = None


@dataclass(slots=True)
class _BreakableAttachmentRuntime:
    step_count: int = 0
    violation_steps: int = 0
    contact_loss_steps: int = 0
    applied_object_wrench: np.ndarray | None = None
    applied_reference_wrench: np.ndarray | None = None
    object_body_id: int | None = None
    reference_body_id: int | None = None
    latest_contact: AttachmentContactMetrics = AttachmentContactMetrics()


class ToolUseJournalEERuntime:
    """Own the live target environment and replace its physical EE model."""

    def __init__(
        self,
        env: object,
        environment_factory: Callable[[str | None], object],
        *,
        source_revision: str = TOOL_USE_JOURNAL_TESTED_REVISION,
        close_replaced_environments: bool = True,
    ) -> None:
        adapter = ToolUseJournalEnvironmentAdapter(
            env, source_revision=source_revision
        )
        self._env = env
        self._factory = environment_factory
        self._source_revision = source_revision
        self._environment_name = adapter.environment_name
        self._active_ee = adapter.physical_active_ee
        self._close_replaced = close_replaced_environments
        self._closed = False
        self._transitions: list[EERuntimeTransition] = []
        self._gripper_command = -1.0
        self._grasp_engaged = False
        self._attachment: AttachedObjectState | None = None
        self._breakable_runtime: _BreakableAttachmentRuntime | None = None
        self._last_attachment_break: AttachmentBreakObservation | None = None
        self._set_declared_active_ee(env, self._active_ee)
        self._hidden_rack_ee = self._apply_rack_visibility(
            env, self._active_ee
        )

    @classmethod
    def from_repository(
        cls,
        repository_root: str | Path,
        env_name: str,
        *,
        active_ee: str | None,
        seed: int = 0,
        source_revision: str = TOOL_USE_JOURNAL_TESTED_REVISION,
        **suite_make_kwargs: Any,
    ) -> "ToolUseJournalEERuntime":
        options = dict(suite_make_kwargs)

        def factory(ee: str | None) -> object:
            return make_tool_use_journal_env(
                repository_root,
                env_name,
                active_ee=ee,
                seed=seed,
                **options,
            )

        env = factory(active_ee)
        try:
            env.reset()  # type: ignore[attr-defined]
            return cls(
                env,
                factory,
                source_revision=source_revision,
            )
        except Exception:
            close = getattr(env, "close", None)
            if callable(close):
                close()
            raise

    @classmethod
    def from_repository_for_controller(
        cls,
        repository_root: str | Path,
        env_name: str,
        *,
        active_ee: str | None,
        seed: int = 0,
        control_timestep_s: float = 0.02,
        joint_position_kp: float = 50.0,
        joint_position_damping_ratio: float = 1.0,
        source_revision: str = TOOL_USE_JOURNAL_TESTED_REVISION,
        **suite_make_kwargs: Any,
    ) -> "ToolUseJournalEERuntime":
        """Build all EE variants with an absolute joint-space controller."""

        if control_timestep_s <= 0.0:
            raise ValueError("control_timestep_s must be positive")
        control_freq = round(1.0 / control_timestep_s)
        if not math.isclose(
            control_freq * control_timestep_s,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "control_timestep_s must be the reciprocal of an integer frequency"
            )
        options = dict(suite_make_kwargs)
        options["controller_configs"] = (
            tool_use_journal_joint_position_controller_config(
                kp=joint_position_kp,
                damping_ratio=joint_position_damping_ratio,
            )
        )
        options["control_freq"] = control_freq
        return cls.from_repository(
            repository_root,
            env_name,
            active_ee=active_ee,
            seed=seed,
            source_revision=source_revision,
            **options,
        )

    @property
    def env(self) -> object:
        if self._closed:
            raise ToolUseJournalRuntimeError("EE runtime is closed")
        return self._env

    @property
    def active_ee(self) -> str | None:
        return self._active_ee

    @property
    def transitions(self) -> tuple[EERuntimeTransition, ...]:
        return tuple(self._transitions)

    @property
    def gripper_command(self) -> float:
        """Current normalized GRIP command (-1 open/off, +1 close/on)."""

        return self._gripper_command

    @property
    def grasp_engaged(self) -> bool:
        return self._grasp_engaged

    @property
    def attachment(self) -> AttachedObjectState | None:
        return self._attachment

    @property
    def attached_object_id(self) -> str | None:
        return self._attachment.object_id if self._attachment is not None else None

    @property
    def last_attachment_break(self) -> AttachmentBreakObservation | None:
        return self._last_attachment_break

    @property
    def attachment_contact_metrics(self) -> AttachmentContactMetrics | None:
        runtime = self._breakable_runtime
        return runtime.latest_contact if runtime is not None else None

    @staticmethod
    def _set_declared_active_ee(env: object, active_ee: str | None) -> None:
        setattr(env, "current_ee_id", active_ee)
        robot_spec = getattr(env, "robot_spec", None)
        if isinstance(robot_spec, dict):
            robot_spec["current_ee"] = active_ee

    @staticmethod
    def _rack_geom_ids(env: object, ee: str) -> tuple[int, ...]:
        model, _ = _raw_model_data(env)
        try:
            root_name = str(env.ee_rack_info[ee]["rack_body"])  # type: ignore[attr-defined]
        except (AttributeError, KeyError, TypeError) as error:
            raise ToolUseJournalRuntimeError(
                f"rack has no EE record for {ee!r}"
            ) from error
        root_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, root_name
        )
        if root_id < 0:
            raise ToolUseJournalRuntimeError(
                f"rack EE body {root_name!r} is absent from runtime model"
            )
        bodies = _descendant_body_ids(model, root_id)
        return tuple(
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in bodies
        )

    @classmethod
    def _apply_rack_visibility(
        cls, env: object, active_ee: str | None
    ) -> str | None:
        model, _ = _raw_model_data(env)
        rack_info = getattr(env, "ee_rack_info", {})
        for ee in rack_info:
            alpha = 0.0 if ee == active_ee else 1.0
            for geom_id in cls._rack_geom_ids(env, str(ee)):
                model.geom_rgba[geom_id, 3] = alpha
        return active_ee

    def rack_ee_visible(self, ee: str) -> bool:
        model, _ = _raw_model_data(self.env)
        geom_ids = self._rack_geom_ids(self.env, ee)
        return bool(geom_ids) and any(
            float(model.geom_rgba[geom_id, 3]) > 0.0 for geom_id in geom_ids
        )

    @staticmethod
    def _grasp_reference(
        env: object,
    ) -> tuple[str, str, np.ndarray, np.ndarray]:
        """Return the MJCF-backed grasp site, or the hand body as fallback."""

        adapter = ToolUseJournalEnvironmentAdapter(env)
        model, data = adapter.model, adapter.data
        robot = adapter.robot
        grippers = getattr(robot, "gripper", {})
        gripper = grippers.get("right") if isinstance(grippers, Mapping) else None
        sites = getattr(gripper, "important_sites", {})
        site_name = sites.get("grip_site") if isinstance(sites, Mapping) else None
        if site_name:
            site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, str(site_name)
            )
            if site_id >= 0:
                return (
                    "site",
                    str(site_name),
                    np.asarray(data.site_xpos[site_id], dtype=float).copy(),
                    np.asarray(data.site_xmat[site_id], dtype=float)
                    .reshape(3, 3)
                    .copy(),
                )
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, adapter.hand_body
        )
        if body_id < 0:
            raise ToolUseJournalRuntimeError("runtime grasp frame is absent")
        return (
            "body",
            adapter.hand_body,
            np.asarray(data.xpos[body_id], dtype=float).copy(),
            np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3).copy(),
        )

    @staticmethod
    def _reference_pose(
        env: object, kind: str, name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        model, data = _raw_model_data(env)
        if kind == "site":
            object_type = mujoco.mjtObj.mjOBJ_SITE
            positions, rotations = data.site_xpos, data.site_xmat
        elif kind == "body":
            object_type = mujoco.mjtObj.mjOBJ_BODY
            positions, rotations = data.xpos, data.xmat
        else:
            raise ToolUseJournalRuntimeError(
                f"unsupported attachment reference kind {kind!r}"
            )
        reference_id = mujoco.mj_name2id(model, object_type, name)
        if reference_id < 0:
            raise ToolUseJournalRuntimeError(
                f"attachment reference {name!r} is absent"
            )
        return (
            np.asarray(positions[reference_id], dtype=float).copy(),
            np.asarray(rotations[reference_id], dtype=float)
            .reshape(3, 3)
            .copy(),
        )

    @staticmethod
    def _object_free_joint(
        env: object, object_id: str
    ) -> tuple[int, int, str]:
        adapter = ToolUseJournalEnvironmentAdapter(env)
        try:
            body_id = int(adapter.object_body_ids[object_id])
        except KeyError as error:
            raise ToolUseJournalRuntimeError(
                f"unknown attach target {object_id!r}"
            ) from error
        model = adapter.model
        free_joint_ids = [
            joint_id
            for joint_id in range(model.njnt)
            if int(model.jnt_bodyid[joint_id]) == body_id
            and int(model.jnt_type[joint_id])
            == int(mujoco.mjtJoint.mjJNT_FREE)
        ]
        if len(free_joint_ids) != 1:
            raise ToolUseJournalRuntimeError(
                f"object {object_id!r} must have exactly one root free joint"
            )
        if int(model.body_parentid[body_id]) != 0:
            raise ToolUseJournalRuntimeError(
                f"object {object_id!r} free joint is not world-relative"
            )
        joint_id = free_joint_ids[0]
        return (
            body_id,
            joint_id,
            _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
        )

    @staticmethod
    def _subtree_geom_ids(model: mujoco.MjModel, root_id: int) -> tuple[int, ...]:
        bodies = _descendant_body_ids(model, root_id)
        return tuple(
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in bodies
        )

    def _minimum_ee_object_distance(self, object_body_id: int) -> float:
        adapter = ToolUseJournalEnvironmentAdapter(self.env)
        model, data = adapter.model, adapter.data
        mounted_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, adapter.mounted_root_body
        )
        if mounted_id < 0:
            raise ToolUseJournalRuntimeError("mounted EE body is absent")
        ee_geoms = tuple(
            geom_id
            for geom_id in self._subtree_geom_ids(model, mounted_id)
            if int(model.geom_contype[geom_id])
            or int(model.geom_conaffinity[geom_id])
        )
        object_geoms = tuple(
            geom_id
            for geom_id in self._subtree_geom_ids(model, object_body_id)
            if int(model.geom_contype[geom_id])
            or int(model.geom_conaffinity[geom_id])
        )
        distances: list[float] = []
        for ee_geom in ee_geoms:
            for object_geom in object_geoms:
                from_to = np.empty(6, dtype=float)
                distances.append(
                    float(
                        mujoco.mj_geomDistance(
                            model,
                            data,
                            ee_geom,
                            object_geom,
                            10.0,
                            from_to,
                        )
                    )
                )
        if distances:
            return min(distances)
        _, _, reference_position, _ = self._grasp_reference(self.env)
        return float(
            np.linalg.norm(data.xpos[object_body_id] - reference_position)
        )

    @staticmethod
    def _normalized_command(
        command: float | bool | str | None, *, engaged: bool
    ) -> float:
        if command is None:
            return 1.0 if engaged else -1.0
        if isinstance(command, bool):
            return 1.0 if command else -1.0
        try:
            value = float(command)
        except (TypeError, ValueError) as error:
            raise ToolUseJournalRuntimeError(
                f"gripper command {command!r} is not numeric"
            ) from error
        if not math.isfinite(value) or value < -1.0 or value > 1.0:
            raise ToolUseJournalRuntimeError(
                "gripper command must be finite and within [-1, 1]"
            )
        return value

    def command_gripper(
        self,
        *,
        engaged: bool,
        suction: bool,
        command: float | bool | str | None = None,
    ) -> float:
        """Update the persistent normalized command used by both players."""

        if not engaged and self._attachment is not None:
            raise ToolUseJournalRuntimeError(
                f"detach object {self._attachment.object_id!r} before opening "
                "the gripper or disabling suction"
            )
        if suction:
            if self._active_ee != "vac":
                raise ToolUseJournalRuntimeError(
                    f"suction requires 'vac'; active EE is {self._active_ee!r}"
                )
        elif self._active_ee not in {"2F", "3F"}:
            raise ToolUseJournalRuntimeError(
                f"finger command requires '2F' or '3F'; active EE is "
                f"{self._active_ee!r}"
            )
        normalized = self._normalized_command(command, engaged=engaged)
        if engaged and normalized <= 0.0:
            raise ToolUseJournalRuntimeError(
                "close / suction-on command must be greater than zero"
            )
        if not engaged and normalized >= 0.0:
            raise ToolUseJournalRuntimeError(
                "open / suction-off command must be less than zero"
            )
        self._gripper_command = normalized
        self._grasp_engaged = engaged
        return self._gripper_command

    @staticmethod
    def _attachment_reference_body_id(
        model: mujoco.MjModel, attachment: AttachedObjectState
    ) -> int:
        if attachment.reference_kind == "body":
            body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                attachment.reference_name,
            )
        elif attachment.reference_kind == "site":
            site_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                attachment.reference_name,
            )
            body_id = int(model.site_bodyid[site_id]) if site_id >= 0 else -1
        else:
            body_id = -1
        if body_id < 0:
            raise ToolUseJournalRuntimeError(
                f"attachment reference {attachment.reference_name!r} is absent"
            )
        return body_id

    def _attachment_contact_metrics(
        self, attachment: AttachedObjectState
    ) -> AttachmentContactMetrics:
        model, data = _raw_model_data(self.env)
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            attachment.free_joint_name,
        )
        if joint_id < 0:
            raise ToolUseJournalRuntimeError(
                f"object free joint {attachment.free_joint_name!r} is absent"
            )
        object_body_id = int(model.jnt_bodyid[joint_id])
        adapter = ToolUseJournalEnvironmentAdapter(self.env)
        mounted_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            adapter.mounted_root_body,
        )
        if mounted_body_id < 0:
            raise ToolUseJournalRuntimeError("mounted EE body is absent")
        object_geoms = set(self._subtree_geom_ids(model, object_body_id))
        ee_geoms = set(self._subtree_geom_ids(model, mounted_body_id))
        normal_force = 0.0
        tangential_force = 0.0
        total_force = 0.0
        contact_count = 0
        contact_groups: set[str] = set()
        for contact_id in range(int(data.ncon)):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if not (
                (geom1 in object_geoms and geom2 in ee_geoms)
                or (geom2 in object_geoms and geom1 in ee_geoms)
            ):
                continue
            wrench = np.empty(6, dtype=float)
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            normal_force += abs(float(wrench[0]))
            tangential_force += float(np.linalg.norm(wrench[1:3]))
            total_force += float(np.linalg.norm(wrench[:3]))
            contact_count += 1
            ee_geom_id = geom2 if geom2 in ee_geoms else geom1
            ee_geom_name = (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, ee_geom_id
                )
                or ""
            ).lower()
            # Robosuite prefixes mounted right-arm geoms with
            # ``gripper0_right_``.  Match the finger token after that prefix so
            # the arm-side "right" does not accidentally classify a left pad
            # as right-finger contact.
            mounted_suffix = ee_geom_name.rsplit("gripper0_right_", 1)[-1]
            if mounted_suffix.startswith("left_"):
                contact_groups.add("left_finger")
            elif mounted_suffix.startswith("right_"):
                contact_groups.add("right_finger")
            if "suction" in mounted_suffix or "vacuum" in mounted_suffix:
                contact_groups.add("suction")
        return AttachmentContactMetrics(
            contact_count=contact_count,
            normal_force_n=normal_force,
            tangential_force_n=tangential_force,
            total_force_n=total_force,
            contact_groups=tuple(sorted(contact_groups)),
        )

    @staticmethod
    def _contact_contract_failures(
        contact: AttachmentContactMetrics,
        config: BreakableWeldConfig,
    ) -> tuple[str, ...]:
        failures: list[str] = []
        if contact.contact_count < config.min_contact_count:
            failures.append("CONTACT_COUNT")
        if contact.normal_force_n < config.min_normal_force_n:
            failures.append("CONTACT_NORMAL_FORCE")
        missing_groups = sorted(
            set(config.required_contact_groups) - set(contact.contact_groups)
        )
        if missing_groups:
            failures.append("CONTACT_GROUPS:" + ",".join(missing_groups))
        return tuple(failures)

    @staticmethod
    def _rotation_error_vector(
        desired_rotation: np.ndarray, actual_rotation: np.ndarray
    ) -> np.ndarray:
        desired_quaternion = np.empty(4, dtype=float)
        actual_quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(
            desired_quaternion,
            np.ascontiguousarray(desired_rotation.reshape(9)),
        )
        mujoco.mju_mat2Quat(
            actual_quaternion,
            np.ascontiguousarray(actual_rotation.reshape(9)),
        )
        inverse_actual = np.empty(4, dtype=float)
        mujoco.mju_negQuat(inverse_actual, actual_quaternion)
        difference = np.empty(4, dtype=float)
        mujoco.mju_mulQuat(
            difference, desired_quaternion, inverse_actual
        )
        if difference[0] < 0.0:
            difference *= -1.0
        result = np.empty(3, dtype=float)
        mujoco.mju_quat2Vel(result, difference, 1.0)
        return result

    def _clear_attachment_wrench(self) -> None:
        runtime = self._breakable_runtime
        if runtime is None:
            return
        _, data = _raw_model_data(self.env)
        if (
            runtime.object_body_id is not None
            and runtime.applied_object_wrench is not None
        ):
            data.xfrc_applied[runtime.object_body_id] -= (
                runtime.applied_object_wrench
            )
        if (
            runtime.reference_body_id is not None
            and runtime.applied_reference_wrench is not None
        ):
            data.xfrc_applied[runtime.reference_body_id] -= (
                runtime.applied_reference_wrench
            )
        runtime.applied_object_wrench = None
        runtime.applied_reference_wrench = None
        runtime.object_body_id = None
        runtime.reference_body_id = None

    def _break_attachment(
        self,
        attachment: AttachedObjectState,
        *,
        reasons: Sequence[str],
        required_force_n: float,
        required_torque_nm: float,
        position_error_m: float,
        orientation_error_rad: float,
        contact: AttachmentContactMetrics,
    ) -> None:
        _, data = _raw_model_data(self.env)
        observation = AttachmentBreakObservation(
            object_id=attachment.object_id,
            reasons=tuple(reasons),
            simulation_time_s=float(data.time),
            required_force_n=required_force_n,
            required_torque_nm=required_torque_nm,
            position_error_m=position_error_m,
            orientation_error_rad=orientation_error_rad,
            contact_count=contact.contact_count,
            contact_force_n=contact.total_force_n,
            contact_groups=contact.contact_groups,
        )
        self._clear_attachment_wrench()
        self._attachment = None
        self._breakable_runtime = None
        self._last_attachment_break = observation
        raise ToolUseJournalAttachmentBroken(observation)

    def prepare_attachment_step(self) -> None:
        """Apply a 6-DoF penalty weld and fail if its contract breaks."""

        attachment = self._attachment
        runtime = self._breakable_runtime
        if (
            attachment is None
            or attachment.mode is not AttachmentMode.BREAKABLE_WELD
            or runtime is None
        ):
            return
        config = attachment.breakable_weld
        if config is None:  # pragma: no cover - construction invariant
            raise ToolUseJournalRuntimeError(
                "breakable attachment configuration is absent"
            )
        self._clear_attachment_wrench()
        model, data = _raw_model_data(self.env)
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            attachment.free_joint_name,
        )
        if joint_id < 0:
            raise ToolUseJournalRuntimeError(
                f"object free joint {attachment.free_joint_name!r} is absent"
            )
        object_body_id = int(model.jnt_bodyid[joint_id])
        reference_body_id = self._attachment_reference_body_id(
            model, attachment
        )
        reference_position, reference_rotation = self._reference_pose(
            self.env,
            attachment.reference_kind,
            attachment.reference_name,
        )
        relative_position = np.asarray(
            attachment.position_in_reference_m, dtype=float
        )
        relative_rotation = np.asarray(
            attachment.rotation_in_reference, dtype=float
        )
        desired_position = (
            reference_position + reference_rotation @ relative_position
        )
        desired_rotation = reference_rotation @ relative_rotation
        actual_position = np.asarray(data.xpos[object_body_id], dtype=float)
        actual_rotation = np.asarray(
            data.xmat[object_body_id], dtype=float
        ).reshape(3, 3)
        position_error = desired_position - actual_position
        orientation_error = self._rotation_error_vector(
            desired_rotation, actual_rotation
        )

        object_velocity = np.empty(6, dtype=float)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            object_body_id,
            object_velocity,
            0,
        )
        if attachment.reference_kind == "site":
            reference_type = mujoco.mjtObj.mjOBJ_SITE
        else:
            reference_type = mujoco.mjtObj.mjOBJ_BODY
        reference_id = mujoco.mj_name2id(
            model, reference_type, attachment.reference_name
        )
        reference_velocity = np.empty(6, dtype=float)
        mujoco.mj_objectVelocity(
            model,
            data,
            reference_type,
            reference_id,
            reference_velocity,
            0,
        )
        # Cap user gains by a mass / inertia scaled natural frequency.  The
        # target scenes range from 2.4 g blocks to an 0.8 kg plate; fixed gains
        # that are stable for the latter explosively accelerate the former.
        omega = 2.0 * math.pi * config.natural_frequency_hz
        # Tool-Use-Journal adjusts plate body_mass / body_inertia after model
        # compilation (for example heavy_plate becomes 0.8 kg).  MuJoCo's
        # derived body_subtreemass array is not recomputed by those direct
        # assignments, so using it here can under-estimate the payload by more
        # than two orders of magnitude and make the penalty weld too weak to
        # lift the object.  Sum the live subtree arrays instead.
        object_body_ids = _descendant_body_ids(model, object_body_id)
        object_mass = max(
            sum(float(model.body_mass[body_id]) for body_id in object_body_ids),
            1e-6,
        )
        subtree_inertia = np.sum(
            np.asarray(model.body_inertia[list(object_body_ids)], dtype=float),
            axis=0,
        )
        object_inertia = max(float(np.max(subtree_inertia)), 1e-8)
        linear_stiffness = min(
            config.linear_stiffness_n_m, object_mass * omega * omega
        )
        linear_damping = min(
            config.linear_damping_ns_m,
            2.0 * config.damping_ratio * object_mass * omega,
        )
        angular_stiffness = min(
            config.angular_stiffness_nm_rad,
            object_inertia * omega * omega,
        )
        angular_damping = min(
            config.angular_damping_nms_rad,
            2.0 * config.damping_ratio * object_inertia * omega,
        )
        required_force = (
            linear_stiffness * position_error
            + linear_damping
            * (reference_velocity[3:] - object_velocity[3:])
        )
        required_torque = (
            angular_stiffness * orientation_error
            + angular_damping
            * (reference_velocity[:3] - object_velocity[:3])
        )
        required_force_n = float(np.linalg.norm(required_force))
        required_torque_nm = float(np.linalg.norm(required_torque))
        position_error_m = float(np.linalg.norm(position_error))
        orientation_error_rad = float(np.linalg.norm(orientation_error))
        contact = self._attachment_contact_metrics(attachment)
        runtime.latest_contact = contact
        runtime.step_count += 1
        # Opposed contact groups and minimum formation force are checked when
        # ATTACH_OBJECT creates the grasp.  Once the 6-DoF grasp constraint is
        # active, MuJoCo may reduce a redundant bilateral contact manifold to
        # one active side.  Retention contact is therefore a separate policy:
        # when disabled, weld force / pose-error limits remain responsible for
        # detecting overload or slip after a valid grasp has been formed.
        if (
            config.require_contact
            and config.require_retention_contact
            and contact.contact_count == 0
        ):
            runtime.contact_loss_steps += 1
        else:
            runtime.contact_loss_steps = 0

        reasons: list[str] = []
        if required_force_n > config.max_weld_force_n:
            reasons.append("WELD_FORCE_LIMIT")
        if required_torque_nm > config.max_weld_torque_nm:
            reasons.append("WELD_TORQUE_LIMIT")
        if position_error_m > config.max_position_error_m:
            reasons.append("POSITION_ERROR_LIMIT")
        if orientation_error_rad > config.max_orientation_error_rad:
            reasons.append("ORIENTATION_ERROR_LIMIT")
        if contact.total_force_n > config.max_contact_force_n:
            reasons.append("CONTACT_FORCE_LIMIT")
        if runtime.contact_loss_steps > config.contact_loss_grace_steps:
            reasons.append("CONTACT_LOST")
        if runtime.step_count <= config.startup_grace_steps:
            reasons = []
        runtime.violation_steps = (
            runtime.violation_steps + 1 if reasons else 0
        )
        if runtime.violation_steps >= config.break_debounce_steps:
            self._break_attachment(
                attachment,
                reasons=reasons,
                required_force_n=required_force_n,
                required_torque_nm=required_torque_nm,
                position_error_m=position_error_m,
                orientation_error_rad=orientation_error_rad,
                contact=contact,
            )

        force_scale = min(
            1.0,
            config.max_weld_force_n / max(required_force_n, 1e-12),
        )
        torque_scale = min(
            1.0,
            config.max_weld_torque_nm / max(required_torque_nm, 1e-12),
        )
        applied_force = required_force * force_scale
        applied_torque = required_torque * torque_scale
        object_wrench = np.concatenate((applied_force, applied_torque))
        lever = actual_position - np.asarray(
            data.xpos[reference_body_id], dtype=float
        )
        reference_wrench = np.concatenate(
            (
                -applied_force,
                -applied_torque - np.cross(lever, applied_force),
            )
        )
        data.xfrc_applied[object_body_id] += object_wrench
        data.xfrc_applied[reference_body_id] += reference_wrench
        runtime.object_body_id = object_body_id
        runtime.reference_body_id = reference_body_id
        runtime.applied_object_wrench = object_wrench
        runtime.applied_reference_wrench = reference_wrench

    def finish_attachment_step(self) -> None:
        """Refresh contact telemetry after a physics substep."""

        attachment = self._attachment
        runtime = self._breakable_runtime
        if (
            attachment is not None
            and attachment.mode is AttachmentMode.BREAKABLE_WELD
            and runtime is not None
        ):
            runtime.latest_contact = self._attachment_contact_metrics(
                attachment
            )

    def attach_object(
        self,
        object_id: str,
        *,
        max_attach_distance_m: float = 0.02,
        max_attach_penetration_m: float = 0.01,
        require_grasp_command: bool = True,
        attachment_mode: AttachmentMode | str = AttachmentMode.KINEMATIC,
        breakable_weld: BreakableWeldConfig | None = None,
    ) -> AttachedObjectState:
        """Attach a free-joint object at its current contact pose."""

        if self._active_ee is None:
            raise ToolUseJournalRuntimeError(
                "cannot attach an object to a bare flange"
            )
        if self._attachment is not None:
            raise ToolUseJournalRuntimeError(
                f"object {self._attachment.object_id!r} is already attached"
            )
        if require_grasp_command and not self._grasp_engaged:
            raise ToolUseJournalRuntimeError(
                "ATTACH_OBJECT requires GRIPPER_CLOSE or SUCTION_ON first"
            )
        if (
            not math.isfinite(max_attach_distance_m)
            or max_attach_distance_m < 0.0
        ):
            raise ToolUseJournalRuntimeError(
                "max_attach_distance_m must be finite and non-negative"
            )
        if (
            not math.isfinite(max_attach_penetration_m)
            or max_attach_penetration_m < 0.0
        ):
            raise ToolUseJournalRuntimeError(
                "max_attach_penetration_m must be finite and non-negative"
            )
        try:
            mode = AttachmentMode(attachment_mode)
        except ValueError as error:
            raise ToolUseJournalRuntimeError(
                f"unsupported attachment_mode {attachment_mode!r}"
            ) from error
        if mode is AttachmentMode.BREAKABLE_WELD:
            breakable_weld = breakable_weld or BreakableWeldConfig()
        elif breakable_weld is not None:
            raise ToolUseJournalRuntimeError(
                "breakable_weld config requires BREAKABLE_WELD mode"
            )
        model, data = _raw_model_data(self.env)
        mujoco.mj_forward(model, data)
        body_id, _, free_joint_name = self._object_free_joint(
            self.env, object_id
        )
        distance = self._minimum_ee_object_distance(body_id)
        if distance > max_attach_distance_m:
            raise ToolUseJournalRuntimeError(
                f"object {object_id!r} is {distance:.4f} m from the EE; "
                f"attach limit is {max_attach_distance_m:.4f} m"
            )
        if distance < -max_attach_penetration_m:
            raise ToolUseJournalRuntimeError(
                f"object {object_id!r} penetrates EE geometry by "
                f"{-distance:.4f} m; limit is "
                f"{max_attach_penetration_m:.4f} m"
            )
        kind, name, reference_position, reference_rotation = (
            self._grasp_reference(self.env)
        )
        object_position = np.asarray(data.xpos[body_id], dtype=float)
        object_rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
        relative_position = reference_rotation.T @ (
            object_position - reference_position
        )
        relative_rotation = reference_rotation.T @ object_rotation
        attachment = AttachedObjectState(
            object_id=object_id,
            free_joint_name=free_joint_name,
            reference_kind=kind,
            reference_name=name,
            position_in_reference_m=tuple(
                float(value) for value in relative_position
            ),
            rotation_in_reference=tuple(
                tuple(float(value) for value in row)
                for row in relative_rotation
            ),
            attach_distance_m=distance,
            mode=mode,
            breakable_weld=breakable_weld,
        )
        self._attachment = attachment
        self._last_attachment_break = None
        if mode is AttachmentMode.BREAKABLE_WELD:
            contact = self._attachment_contact_metrics(attachment)
            if breakable_weld is None:  # pragma: no cover - guarded above
                raise ToolUseJournalRuntimeError(
                    "breakable weld configuration is absent"
                )
            contract_failures = self._contact_contract_failures(
                contact, breakable_weld
            )
            if breakable_weld.require_contact and contract_failures:
                self._attachment = None
                raise ToolUseJournalRuntimeError(
                    f"BREAKABLE_WELD grasp-contact contract failed for "
                    f"{object_id!r}: {', '.join(contract_failures)}; observed "
                    f"count={contact.contact_count}, "
                    f"normal_force={contact.normal_force_n:.3f} N, "
                    f"groups={list(contact.contact_groups)}"
                )
            self._breakable_runtime = _BreakableAttachmentRuntime(
                latest_contact=contact
            )
        else:
            self._breakable_runtime = None
            self.synchronize_attached_object()
        return attachment

    def synchronize_attached_object(self) -> None:
        """Project the attached object's free joint onto the grasp transform."""

        attachment = self._attachment
        if attachment is None:
            return
        if attachment.mode is AttachmentMode.BREAKABLE_WELD:
            return
        model, data = _raw_model_data(self.env)
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            attachment.free_joint_name,
        )
        if joint_id < 0 or int(model.jnt_type[joint_id]) != int(
            mujoco.mjtJoint.mjJNT_FREE
        ):
            raise ToolUseJournalRuntimeError(
                f"attached free joint {attachment.free_joint_name!r} is absent"
            )
        reference_position, reference_rotation = self._reference_pose(
            self.env,
            attachment.reference_kind,
            attachment.reference_name,
        )
        relative_position = np.asarray(
            attachment.position_in_reference_m, dtype=float
        )
        relative_rotation = np.asarray(
            attachment.rotation_in_reference, dtype=float
        )
        object_position = (
            reference_position + reference_rotation @ relative_position
        )
        object_rotation = reference_rotation @ relative_rotation
        quaternion_wxyz = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(
            quaternion_wxyz,
            np.ascontiguousarray(object_rotation.reshape(9)),
        )
        qpos_start = int(model.jnt_qposadr[joint_id])
        qvel_start = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_start : qpos_start + 3] = object_position
        data.qpos[qpos_start + 3 : qpos_start + 7] = quaternion_wxyz
        data.qvel[qvel_start : qvel_start + 6] = 0.0
        mujoco.mj_forward(model, data)

    def detach_object(self, object_id: str | None = None) -> AttachedObjectState:
        """Release the attached object while preserving its current world pose."""

        attachment = self._attachment
        if attachment is None:
            raise ToolUseJournalRuntimeError("no object is attached")
        if object_id is not None and object_id != attachment.object_id:
            raise ToolUseJournalRuntimeError(
                f"cannot detach {object_id!r}; attached object is "
                f"{attachment.object_id!r}"
            )
        if attachment.mode is AttachmentMode.KINEMATIC:
            self.synchronize_attached_object()
        else:
            self._clear_attachment_wrench()
        model, data = _raw_model_data(self.env)
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, attachment.free_joint_name
        )
        if joint_id >= 0 and attachment.mode is AttachmentMode.KINEMATIC:
            qvel_start = int(model.jnt_dofadr[joint_id])
            data.qvel[qvel_start : qvel_start + 6] = 0.0
        self._attachment = None
        self._breakable_runtime = None
        mujoco.mj_forward(model, data)
        return attachment

    def switch_active_ee(
        self,
        to_ee: str | None,
        *,
        expected_from_ee: str | None,
    ) -> EERuntimeTransition:
        """Atomically replace the env variant while preserving common state."""

        if self._closed:
            raise ToolUseJournalRuntimeError("EE runtime is closed")
        if self._active_ee != expected_from_ee:
            raise ToolUseJournalRuntimeError(
                f"expected mounted EE {expected_from_ee!r}, observed "
                f"{self._active_ee!r}"
            )
        if to_ee == self._active_ee:
            raise ToolUseJournalRuntimeError(
                f"EE {to_ee!r} is already the active model"
            )
        if self._attachment is not None:
            raise ToolUseJournalRuntimeError(
                f"cannot exchange EE while object "
                f"{self._attachment.object_id!r} is attached"
            )

        old_env = self._env
        state = _capture_runtime_state(old_env)
        new_env: object | None = None
        try:
            new_env = self._factory(to_ee)
            new_env.reset()  # type: ignore[attr-defined]
            adapter = ToolUseJournalEnvironmentAdapter(
                new_env, source_revision=self._source_revision
            )
            if adapter.environment_name != self._environment_name:
                raise ToolUseJournalRuntimeError(
                    "environment factory changed task during EE transition"
                )
            if adapter.physical_active_ee != to_ee:
                raise ToolUseJournalRuntimeError(
                    f"factory requested {to_ee!r} but built "
                    f"{adapter.physical_active_ee!r}"
                )
            _restore_runtime_state(new_env, state)
            self._set_declared_active_ee(new_env, to_ee)
            hidden = self._apply_rack_visibility(new_env, to_ee)

            # Verify every common named joint survived bit-for-bit within
            # floating-point transfer tolerance before committing the swap.
            restored = _capture_runtime_state(new_env)
            common = sorted(
                set(state.qpos_by_joint) & set(restored.qpos_by_joint)
            )
            mismatched = [
                name
                for name in common
                if not np.allclose(
                    state.qpos_by_joint[name],
                    restored.qpos_by_joint[name],
                    atol=1e-12,
                    rtol=0.0,
                )
            ]
            if mismatched:
                raise ToolUseJournalRuntimeError(
                    f"state transfer changed joints {mismatched}"
                )
        except Exception as error:
            if new_env is not None:
                close = getattr(new_env, "close", None)
                if callable(close):
                    close()
            if isinstance(error, ToolUseJournalRuntimeError):
                raise
            if isinstance(error, ToolUseJournalCompatibilityError):
                raise ToolUseJournalRuntimeError(str(error)) from error
            raise ToolUseJournalRuntimeError(
                f"failed to build EE runtime state {to_ee!r}: {error}"
            ) from error

        self._env = new_env
        previous = self._active_ee
        self._active_ee = to_ee
        self._gripper_command = -1.0
        self._grasp_engaged = False
        self._hidden_rack_ee = hidden
        transition = EERuntimeTransition(
            from_ee=previous,
            to_ee=to_ee,
            simulation_time_s=state.simulation_time_s,
            preserved_joint_names=tuple(common),
            hidden_rack_ee=hidden,
        )
        self._transitions.append(transition)
        if self._close_replaced:
            close = getattr(old_env, "close", None)
            if callable(close):
                close()
        return transition

    def unlock(self, ee: str) -> EERuntimeTransition:
        if self._active_ee != ee:
            raise ToolUseJournalRuntimeError(
                f"cannot unlock {ee!r}; active EE is {self._active_ee!r}"
            )
        return self.switch_active_ee(None, expected_from_ee=ee)

    def lock(self, ee: str) -> EERuntimeTransition:
        if self._active_ee is not None:
            raise ToolUseJournalRuntimeError(
                f"cannot lock {ee!r}; EE {self._active_ee!r} is still mounted"
            )
        return self.switch_active_ee(ee, expected_from_ee=None)

    def verify_tool_release(self, ee: str) -> None:
        if self._active_ee is not None:
            raise ToolUseJournalRuntimeError(
                f"release verification failed; active EE is {self._active_ee!r}"
            )
        if not self.rack_ee_visible(ee):
            raise ToolUseJournalRuntimeError(
                f"release verification failed; rack EE {ee!r} is hidden"
            )

    def verify_tool_lock(self, ee: str) -> None:
        if self._active_ee != ee or _physical_ee(self.env) != ee:
            raise ToolUseJournalRuntimeError(
                f"lock verification failed for EE {ee!r}"
            )
        if self.rack_ee_visible(ee):
            raise ToolUseJournalRuntimeError(
                f"lock verification failed; rack duplicate {ee!r} is visible"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._clear_attachment_wrench()
        close = getattr(self._env, "close", None)
        if callable(close):
            close()
        self._closed = True

    def __enter__(self) -> "ToolUseJournalEERuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class _PlaybackFailure:
    code: str
    message: str
    segment_id: str | None = None
    waypoint_index: int | None = None
    event_id: str | None = None
    observed: Mapping[str, Any] | None = None


class ToolUseJournalKinematicTrajectoryPlayer:
    """Replay timed joint samples and execute runtime EE exchange events."""

    _TIME_TOLERANCE_S = 1e-9
    _PLAYER_ID = "TOOL_USE_JOURNAL_KINEMATIC_V2"
    _PLAYBACK_MODE = "DIRECT_QPOS_MJ_FORWARD"
    _CONTROLLER_TRACKING = False

    def __init__(
        self,
        runtime: ToolUseJournalEERuntime,
        *,
        collision_probe: MuJoCoCollisionModelRegistry | CollisionProbe | None = None,
    ) -> None:
        self.runtime = runtime
        self._collision_probe = collision_probe

    @staticmethod
    def _arm_joint_addresses(
        env: object, joint_names: Sequence[str]
    ) -> tuple[tuple[int, int], ...]:
        model, _ = _raw_model_data(env)
        result: list[tuple[int, int]] = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise ToolUseJournalRuntimeError(
                    f"plan joint {name!r} is absent from runtime model"
                )
            if int(model.jnt_type[joint_id]) not in {
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            }:
                raise ToolUseJournalRuntimeError(
                    f"plan joint {name!r} is not scalar"
                )
            result.append(
                (
                    int(model.jnt_qposadr[joint_id]),
                    int(model.jnt_dofadr[joint_id]),
                )
            )
        return tuple(result)

    def _apply_waypoint(
        self,
        plan: MotionPlan,
        waypoint: TrajectoryWaypoint,
    ) -> float:
        model, data = _raw_model_data(self.runtime.env)
        addresses = self._arm_joint_addresses(
            self.runtime.env, plan.joint_names
        )
        velocities = waypoint.joint_velocities_rad_s or [0.0] * len(addresses)
        for (qpos_address, qvel_address), position, velocity in zip(
            addresses,
            waypoint.joint_positions_rad,
            velocities,
        ):
            data.qpos[qpos_address] = float(position)
            data.qvel[qvel_address] = float(velocity)
        data.time = float(waypoint.time_from_start_s)
        mujoco.mj_forward(model, data)
        self.runtime.synchronize_attached_object()
        actual = np.asarray(
            [data.qpos[qpos_address] for qpos_address, _ in addresses],
            dtype=float,
        )
        desired = np.asarray(waypoint.joint_positions_rad, dtype=float)
        return float(np.max(np.abs(actual - desired)))

    @staticmethod
    def _eef_position(env: object) -> np.ndarray:
        # MotionPlan EEF poses are expressed at the active EE's grasp / TCP
        # site.  Comparing them with the wrist-hand body adds the fixed tool
        # offset (145 mm for the C1_1 2F gripper) to every tracking sample.
        _, _, position, _ = ToolUseJournalEERuntime._grasp_reference(env)
        return position

    def _execute_event(self, event: TrajectoryEvent) -> str:
        target = event.target_id
        if event.event_type is EventType.GRIPPER_OPEN:
            value = self.runtime.command_gripper(
                engaged=False,
                suction=False,
                command=event.command,
            )
            return f"finger gripper open command set to {value:.3f}"
        if event.event_type is EventType.GRIPPER_CLOSE:
            value = self.runtime.command_gripper(
                engaged=True,
                suction=False,
                command=event.command,
            )
            return f"finger gripper close command set to {value:.3f}"
        if event.event_type is EventType.SUCTION_ON:
            value = self.runtime.command_gripper(
                engaged=True,
                suction=True,
                command=event.command,
            )
            return f"vacuum suction enabled with command {value:.3f}"
        if event.event_type is EventType.SUCTION_OFF:
            value = self.runtime.command_gripper(
                engaged=False,
                suction=True,
                command=event.command,
            )
            return f"vacuum suction disabled with command {value:.3f}"
        if event.event_type is EventType.ATTACH_OBJECT:
            if not target:
                raise ToolUseJournalRuntimeError(
                    "ATTACH_OBJECT requires target_id"
                )
            require_grasp_command = event.parameters.get(
                "require_grasp_command", True
            )
            if not isinstance(require_grasp_command, bool):
                raise ToolUseJournalRuntimeError(
                    "require_grasp_command must be boolean"
                )
            attachment_mode = event.parameters.get(
                "attachment_mode", AttachmentMode.KINEMATIC.value
            )
            try:
                mode = (
                    attachment_mode
                    if isinstance(attachment_mode, AttachmentMode)
                    else AttachmentMode(str(attachment_mode))
                )
            except ValueError as error:
                raise ToolUseJournalRuntimeError(
                    f"unsupported attachment_mode {attachment_mode!r}"
                ) from error
            if (
                mode is AttachmentMode.BREAKABLE_WELD
                and not self._CONTROLLER_TRACKING
            ):
                raise ToolUseJournalRuntimeError(
                    "BREAKABLE_WELD requires the controller-backed player"
                )
            try:
                breakable_weld = (
                    BreakableWeldConfig.from_parameters(event.parameters)
                    if mode is AttachmentMode.BREAKABLE_WELD
                    else None
                )
            except (TypeError, ValueError) as error:
                raise ToolUseJournalRuntimeError(str(error)) from error
            attachment = self.runtime.attach_object(
                target,
                max_attach_distance_m=float(
                    event.parameters.get("max_attach_distance_m", 0.02)
                ),
                max_attach_penetration_m=float(
                    event.parameters.get("max_attach_penetration_m", 0.01)
                ),
                require_grasp_command=require_grasp_command,
                attachment_mode=mode,
                breakable_weld=breakable_weld,
            )
            message = (
                f"attached {target} in {attachment.mode.value} mode to "
                f"{attachment.reference_kind} "
                f"{attachment.reference_name} at "
                f"{attachment.attach_distance_m:.4f} m"
            )
            contact = self.runtime.attachment_contact_metrics
            if contact is not None:
                message += (
                    f"; formation contacts={contact.contact_count}, "
                    f"groups={list(contact.contact_groups)}, "
                    f"normal_force={contact.normal_force_n:.3f} N"
                )
            return message
        if event.event_type is EventType.DETACH_OBJECT:
            attachment = self.runtime.detach_object(target)
            return f"detached {attachment.object_id} and preserved world pose"
        if event.event_type is EventType.TOOL_UNLOCK:
            if not target:
                raise ToolUseJournalRuntimeError(
                    "TOOL_UNLOCK requires target_id"
                )
            self.runtime.unlock(target)
            return f"unlocked {target}; active model is bare flange"
        if event.event_type is EventType.VERIFY_TOOL_RELEASE:
            if not target:
                raise ToolUseJournalRuntimeError(
                    "VERIFY_TOOL_RELEASE requires target_id"
                )
            self.runtime.verify_tool_release(target)
            return f"verified {target} released to rack"
        if event.event_type is EventType.TOOL_LOCK:
            if not target:
                raise ToolUseJournalRuntimeError("TOOL_LOCK requires target_id")
            self.runtime.lock(target)
            return f"locked {target}; runtime model replaced"
        if event.event_type is EventType.VERIFY_TOOL_LOCK:
            if not target:
                raise ToolUseJournalRuntimeError(
                    "VERIFY_TOOL_LOCK requires target_id"
                )
            self.runtime.verify_tool_lock(target)
            return f"verified {target} mounted and rack duplicate hidden"
        if event.event_type is EventType.WAIT:
            return "wait boundary acknowledged"
        raise ToolUseJournalRuntimeError(
            f"event {event.event_type.value} is not implemented by the "
            "EE-exchange kinematic player"
        )

    @staticmethod
    def _events_by_time(
        plan: MotionPlan,
    ) -> dict[float, list[TrajectoryEvent]]:
        waypoint_times = [
            waypoint.time_from_start_s
            for segment in plan.segments
            for waypoint in segment.waypoints
        ]
        result: dict[float, list[TrajectoryEvent]] = {}
        for _, event in sorted(
            enumerate(plan.events),
            key=lambda item: (item[1].time_from_start_s, item[0]),
        ):
            matching = next(
                (
                    time_value
                    for time_value in waypoint_times
                    if math.isclose(
                        time_value,
                        event.time_from_start_s,
                        abs_tol=ToolUseJournalKinematicTrajectoryPlayer._TIME_TOLERANCE_S,
                    )
                ),
                None,
            )
            if matching is None:
                raise ToolUseJournalRuntimeError(
                    f"event {event.event_id!r} is not synchronized to a waypoint"
                )
            result.setdefault(float(matching), []).append(event)
        return result

    @staticmethod
    def _final_state(runtime: ToolUseJournalEERuntime) -> RobotState:
        state = ToolUseJournalEnvironmentAdapter(
            runtime.env
        ).world_snapshot(
            attached_object_id=runtime.attached_object_id
        ).robot_state
        if runtime.active_ee is None:
            gripper_mode = GripperMode.UNKNOWN
        elif runtime.attached_object_id is not None:
            gripper_mode = GripperMode.HOLDING
        elif runtime.grasp_engaged:
            gripper_mode = GripperMode.CLOSED
        else:
            gripper_mode = GripperMode.OPEN
        return state.model_copy(
            update={
                "gripper": GripperState(
                    mode=gripper_mode,
                    command=runtime.gripper_command,
                )
            }
        )

    def _report_provenance(
        self, run: SimulationRun, report_id: str
    ) -> ArtifactProvenance:
        return ArtifactProvenance(
            artifact_id=report_id,
            artifact_type="ExecutionReport",
            produced_by=ModuleName.SIMULATOR,
            invocation_id=run.run_id,
            input_artifact_ids=[run.provenance.artifact_id, run.plan.provenance.artifact_id],
            metadata={"player": self._PLAYER_ID},
        )

    def _build_report(
        self,
        run: SimulationRun,
        *,
        report_id: str,
        status: ExecutionStatus,
        executed_duration_s: float,
        max_tracking_error: float,
        max_eef_error: float | None,
        collision_count: int,
        events: Sequence[ExecutedEvent],
        failure: _PlaybackFailure | None,
    ) -> ExecutionReport:
        final_state: RobotState | None
        try:
            final_state = self._final_state(self.runtime)
        except Exception:  # noqa: BLE001
            final_state = None
        observation = None
        if failure is not None:
            observation = FailureObservation(
                code=failure.code,
                category="SIMULATION_RUNTIME",
                message=failure.message,
                segment_id=failure.segment_id,
                waypoint_index=failure.waypoint_index,
                event_id=failure.event_id,
                observed=dict(failure.observed or {}),
            )
        return ExecutionReport(
            report_id=report_id,
            run_id=run.run_id,
            plan_id=run.plan.plan_id,
            provenance=self._report_provenance(run, report_id),
            status=status,
            final_robot_state=final_state,
            metrics=SimulationMetrics(
                executed_duration_s=max(0.0, executed_duration_s),
                max_joint_tracking_error_rad=max_tracking_error,
                max_eef_tracking_error_m=max_eef_error,
                collision_count=collision_count,
            ),
            executed_events=list(events),
            failure=observation,
            metadata={
                "player": self._PLAYER_ID,
                "playback_mode": self._PLAYBACK_MODE,
                "controller_tracking_simulated": self._CONTROLLER_TRACKING,
                "collision_probe_enabled": self._collision_probe is not None,
                "final_active_ee": self.runtime.active_ee,
                "final_attached_object_id": self.runtime.attached_object_id,
                "final_gripper_command": self.runtime.gripper_command,
                "ee_transition_count": len(self.runtime.transitions),
                "attachment_mode": (
                    self.runtime.attachment.mode.value
                    if self.runtime.attachment is not None
                    else None
                ),
                "attachment_contact": (
                    {
                        "contact_count": (
                            self.runtime.attachment_contact_metrics.contact_count
                        ),
                        "normal_force_n": (
                            self.runtime.attachment_contact_metrics.normal_force_n
                        ),
                        "tangential_force_n": (
                            self.runtime.attachment_contact_metrics.tangential_force_n
                        ),
                        "total_force_n": (
                            self.runtime.attachment_contact_metrics.total_force_n
                        ),
                        "contact_groups": list(
                            self.runtime.attachment_contact_metrics.contact_groups
                        ),
                    }
                    if self.runtime.attachment_contact_metrics is not None
                    else None
                ),
                "last_attachment_break": (
                    self.runtime.last_attachment_break.as_mapping()
                    if self.runtime.last_attachment_break is not None
                    else None
                ),
            },
        )

    def _verify_final_runtime_state(self, plan: MotionPlan) -> None:
        expected_attachment = plan.expected_final_state.attached_object_id
        if expected_attachment != self.runtime.attached_object_id:
            raise ToolUseJournalRuntimeError(
                f"plan expects attached object {expected_attachment!r}, runtime has "
                f"{self.runtime.attached_object_id!r}"
            )

    def _verify_runtime_context(self, context: Any, *, label: str) -> None:
        if context is None:
            return
        if context.active_ee != self.runtime.active_ee:
            raise ToolUseJournalRuntimeError(
                f"{label} expects EE {context.active_ee!r}, runtime has "
                f"{self.runtime.active_ee!r}"
            )
        expected_objects = set(context.attached_object_ids)
        actual_objects = (
            {self.runtime.attached_object_id}
            if self.runtime.attached_object_id is not None
            else set()
        )
        if expected_objects != actual_objects:
            raise ToolUseJournalRuntimeError(
                f"{label} expects attached objects {sorted(expected_objects)}, "
                f"runtime has {sorted(actual_objects)}"
            )

    def execute(
        self,
        run: SimulationRun,
        *,
        report_id: str | None = None,
    ) -> ExecutionReport:
        """Replay a MotionPlan and return a schema-valid execution artifact."""

        report_name = report_id or f"{run.run_id}:execution-report"
        plan = run.plan
        executed_events: list[ExecutedEvent] = []
        executed_event_ids: set[str] = set()
        max_tracking_error = 0.0
        max_eef_error: float | None = None
        collision_count = 0
        executed_time = 0.0
        previous_wall_time = 0.0

        try:
            events_by_time = self._events_by_time(plan)
            first_context = plan.segments[0].collision_context_before
            self._verify_runtime_context(first_context, label="plan start")

            for segment in plan.segments:
                self._verify_runtime_context(
                    segment.collision_context_before,
                    label=f"segment {segment.segment_id!r} start",
                )
                for waypoint_index, waypoint in enumerate(segment.waypoints):
                    if (
                        run.config.max_duration_s is not None
                        and waypoint.time_from_start_s
                        > run.config.max_duration_s + self._TIME_TOLERANCE_S
                    ):
                        failure = _PlaybackFailure(
                            code="SIMULATION_TIMEOUT",
                            message="MotionPlan exceeded max_duration_s",
                            segment_id=segment.segment_id,
                            waypoint_index=waypoint_index,
                        )
                        return self._build_report(
                            run,
                            report_id=report_name,
                            status=ExecutionStatus.TIMEOUT,
                            executed_duration_s=executed_time,
                            max_tracking_error=max_tracking_error,
                            max_eef_error=max_eef_error,
                            collision_count=collision_count,
                            events=executed_events,
                            failure=failure,
                        )

                    tracking_error = self._apply_waypoint(plan, waypoint)
                    max_tracking_error = max(
                        max_tracking_error, tracking_error
                    )
                    executed_time = float(waypoint.time_from_start_s)
                    if waypoint.eef_pose is not None:
                        eef_error = float(
                            np.linalg.norm(
                                self._eef_position(self.runtime.env)
                                - np.asarray(
                                    waypoint.eef_pose.position_m, dtype=float
                                )
                            )
                        )
                        max_eef_error = (
                            eef_error
                            if max_eef_error is None
                            else max(max_eef_error, eef_error)
                        )

                    if (
                        self._collision_probe is not None
                        and segment.collision_context_before is not None
                    ):
                        collision = self._collision_probe.check(
                            waypoint.joint_positions_rad,
                            context=segment.collision_context_before,
                        )
                        if not collision.valid:
                            collision_count += 1
                            if run.config.terminate_on_collision:
                                failure = _PlaybackFailure(
                                    code="EXECUTION_COLLISION",
                                    message=collision.detail,
                                    segment_id=segment.segment_id,
                                    waypoint_index=waypoint_index,
                                    observed={
                                        "failure_code": collision.failure_code,
                                        "min_clearance_m": collision.min_clearance_m,
                                    },
                                )
                                return self._build_report(
                                    run,
                                    report_id=report_name,
                                    status=ExecutionStatus.FAILED,
                                    executed_duration_s=executed_time,
                                    max_tracking_error=max_tracking_error,
                                    max_eef_error=max_eef_error,
                                    collision_count=collision_count,
                                    events=executed_events,
                                    failure=failure,
                                )

                    for event in events_by_time.get(executed_time, []):
                        if event.event_id in executed_event_ids:
                            continue
                        try:
                            message = self._execute_event(event)
                        except Exception as error:  # noqa: BLE001
                            executed_events.append(
                                ExecutedEvent(
                                    event_id=event.event_id,
                                    scheduled_time_s=event.time_from_start_s,
                                    actual_time_s=executed_time,
                                    status=EventExecutionStatus.FAILED,
                                    message=str(error),
                                )
                            )
                            failure = _PlaybackFailure(
                                code="EVENT_EXECUTION_FAILED",
                                message=str(error),
                                segment_id=segment.segment_id,
                                waypoint_index=waypoint_index,
                                event_id=event.event_id,
                                observed={"active_ee": self.runtime.active_ee},
                            )
                            return self._build_report(
                                run,
                                report_id=report_name,
                                status=ExecutionStatus.FAILED,
                                executed_duration_s=executed_time,
                                max_tracking_error=max_tracking_error,
                                max_eef_error=max_eef_error,
                                collision_count=collision_count,
                                events=executed_events,
                                failure=failure,
                            )
                        executed_event_ids.add(event.event_id)
                        executed_events.append(
                            ExecutedEvent(
                                event_id=event.event_id,
                                scheduled_time_s=event.time_from_start_s,
                                actual_time_s=executed_time,
                                status=EventExecutionStatus.SUCCESS,
                                message=message,
                            )
                        )

                    if run.config.render:
                        render = getattr(self.runtime.env, "render", None)
                        if callable(render):
                            render()
                    if run.config.realtime_factor > 0.0:
                        delta = max(0.0, executed_time - previous_wall_time)
                        time.sleep(delta / run.config.realtime_factor)
                    previous_wall_time = executed_time

                self._verify_runtime_context(
                    segment.collision_context_after,
                    label=f"segment {segment.segment_id!r} end",
                )

            missing_events = [
                event.event_id
                for event in plan.events
                if event.event_id not in executed_event_ids
            ]
            if missing_events:
                raise ToolUseJournalRuntimeError(
                    f"events were not executed: {missing_events}"
                )
            final_context = plan.segments[-1].collision_context_after
            self._verify_runtime_context(final_context, label="plan end")
            self._verify_final_runtime_state(plan)
        except Exception as error:  # noqa: BLE001
            failure = _PlaybackFailure(
                code="PLAYBACK_RUNTIME_FAILED",
                message=str(error),
                observed={"active_ee": self.runtime.active_ee},
            )
            return self._build_report(
                run,
                report_id=report_name,
                status=ExecutionStatus.FAILED,
                executed_duration_s=executed_time,
                max_tracking_error=max_tracking_error,
                max_eef_error=max_eef_error,
                collision_count=collision_count,
                events=executed_events,
                failure=failure,
            )

        return self._build_report(
            run,
            report_id=report_name,
            status=ExecutionStatus.SUCCESS,
            executed_duration_s=executed_time,
            max_tracking_error=max_tracking_error,
            max_eef_error=max_eef_error,
            collision_count=collision_count,
            events=executed_events,
            failure=None,
        )


class ToolUseJournalControllerTrajectoryPlayer(
    ToolUseJournalKinematicTrajectoryPlayer
):
    """Track MotionPlan samples through robosuite's torque controller loop.

    C1/C2 deliberately do not implement ``reward()``, so their public
    ``env.step()`` raises after physics has already advanced.  This player uses
    RobotEnv's controller / physics portion of that loop and intentionally
    omits only reward, termination and observation packaging.
    """

    _PLAYER_ID = "TOOL_USE_JOURNAL_CONTROLLER_V1"
    _PLAYBACK_MODE = "ROBOSUITE_ABSOLUTE_JOINT_POSITION_CONTROLLER"
    _CONTROLLER_TRACKING = True

    @staticmethod
    def _timeline(plan: MotionPlan) -> tuple[TrajectoryWaypoint, ...]:
        timeline: list[TrajectoryWaypoint] = []
        for segment in plan.segments:
            for waypoint in segment.waypoints:
                if timeline and math.isclose(
                    timeline[-1].time_from_start_s,
                    waypoint.time_from_start_s,
                    abs_tol=ToolUseJournalControllerTrajectoryPlayer._TIME_TOLERANCE_S,
                ):
                    if not np.allclose(
                        timeline[-1].joint_positions_rad,
                        waypoint.joint_positions_rad,
                        atol=1e-9,
                        rtol=0.0,
                    ):
                        raise ToolUseJournalRuntimeError(
                            "duplicate boundary waypoints have different joint positions"
                        )
                    timeline[-1] = waypoint
                else:
                    timeline.append(waypoint)
        if not timeline:
            raise ToolUseJournalRuntimeError("MotionPlan has no waypoints")
        return tuple(timeline)

    @staticmethod
    def _desired_joint_position(
        timeline: Sequence[TrajectoryWaypoint], time_s: float
    ) -> np.ndarray:
        times = [waypoint.time_from_start_s for waypoint in timeline]
        if time_s <= times[0]:
            return np.asarray(timeline[0].joint_positions_rad, dtype=float)
        if time_s >= times[-1]:
            return np.asarray(timeline[-1].joint_positions_rad, dtype=float)
        right_index = bisect_right(times, time_s)
        left = timeline[right_index - 1]
        right = timeline[right_index]
        duration = right.time_from_start_s - left.time_from_start_s
        ratio = (time_s - left.time_from_start_s) / duration
        q0 = np.asarray(left.joint_positions_rad, dtype=float)
        q1 = np.asarray(right.joint_positions_rad, dtype=float)
        if (
            left.joint_velocities_rad_s is None
            or right.joint_velocities_rad_s is None
        ):
            return q0 + ratio * (q1 - q0)
        v0 = np.asarray(left.joint_velocities_rad_s, dtype=float)
        v1 = np.asarray(right.joint_velocities_rad_s, dtype=float)
        u2, u3 = ratio * ratio, ratio * ratio * ratio
        return (
            (2 * u3 - 3 * u2 + 1) * q0
            + (u3 - 2 * u2 + ratio) * duration * v0
            + (-2 * u3 + 3 * u2) * q1
            + (u3 - u2) * duration * v1
        )

    @staticmethod
    def _segment_at_time(
        plan: MotionPlan, time_s: float
    ) -> TrajectorySegment:
        for segment in plan.segments:
            if (
                segment.start_time_s
                <= time_s
                <= segment.end_time_s
                + ToolUseJournalControllerTrajectoryPlayer._TIME_TOLERANCE_S
            ):
                return segment
        return plan.segments[-1]

    def _controller_action(
        self, plan: MotionPlan, desired_plan_order: Sequence[float]
    ) -> np.ndarray:
        env = self.runtime.env
        try:
            robot = env.robots[0]  # type: ignore[attr-defined]
            controller = robot.part_controllers["right"]
            split_indexes = robot.composite_controller._action_split_indexes
            arm_start, arm_end = split_indexes["right"]
            robot_joint_names = tuple(
                str(name) for name in robot.robot_model.joints
            )
            action = np.zeros(int(robot.action_dim), dtype=float)
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise ToolUseJournalRuntimeError(
                "runtime is missing robosuite composite controller metadata"
            ) from error
        if getattr(controller, "name", None) != "JOINT_POSITION" or getattr(
            controller, "input_type", None
        ) != "absolute":
            raise ToolUseJournalRuntimeError(
                "controller player requires absolute JOINT_POSITION; build the "
                "runtime with from_repository_for_controller()"
            )
        if set(robot_joint_names) != set(plan.joint_names):
            raise ToolUseJournalRuntimeError(
                "MotionPlan joint names do not match runtime arm joints"
            )
        if arm_end - arm_start != len(robot_joint_names):
            raise ToolUseJournalRuntimeError(
                "runtime arm action width does not match robot joint count"
            )
        desired_by_name = dict(zip(plan.joint_names, desired_plan_order))
        action[arm_start:arm_end] = [
            float(desired_by_name[name]) for name in robot_joint_names
        ]
        gripper_slice = split_indexes.get("right_gripper")
        if gripper_slice is not None:
            gripper_start, gripper_end = gripper_slice
            action[gripper_start:gripper_end] = self.runtime.gripper_command
        return action

    def _advance_controller(self, action: np.ndarray) -> float:
        """Advance one robosuite policy period without calling reward()."""

        env = self.runtime.env
        model, data = _raw_model_data(env)
        try:
            control_timestep = float(env.control_timestep)  # type: ignore[attr-defined]
            model_timestep = float(env.model_timestep)  # type: ignore[attr-defined]
            pre_action = env._pre_action  # type: ignore[attr-defined]
            sim = env.sim  # type: ignore[attr-defined]
        except (AttributeError, TypeError) as error:
            raise ToolUseJournalRuntimeError(
                "runtime environment has no robosuite physics loop"
            ) from error
        substeps = round(control_timestep / model_timestep)
        if substeps <= 0 or not math.isclose(
            substeps * model_timestep,
            control_timestep,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ToolUseJournalRuntimeError(
                "control timestep must be an integer multiple of model timestep"
            )
        if hasattr(env, "timestep"):
            env.timestep += 1  # type: ignore[attr-defined]
        policy_step = True
        for _ in range(substeps):
            self.runtime.synchronize_attached_object()
            if bool(getattr(env, "lite_physics", False)):
                sim.step1()
            else:
                sim.forward()
            self.runtime.prepare_attachment_step()
            pre_action(action, policy_step)
            if bool(getattr(env, "lite_physics", False)):
                sim.step2()
            else:
                sim.step()
            self.runtime.synchronize_attached_object()
            self.runtime.finish_attachment_step()
            update_observables = getattr(env, "_update_observables", None)
            if callable(update_observables):
                update_observables()
            policy_step = False
        if hasattr(env, "cur_time"):
            env.cur_time += control_timestep  # type: ignore[attr-defined]
        # Use MuJoCo time as the source of truth across EE topology swaps.
        return float(data.time)

    def _prime_gripper_command(
        self, plan: MotionPlan, desired: Sequence[float]
    ) -> None:
        """Write a synchronized gripper event to actuators without time advance."""

        env = self.runtime.env
        action = self._controller_action(plan, desired)
        pre_action = getattr(env, "_pre_action", None)
        if not callable(pre_action):
            raise ToolUseJournalRuntimeError(
                "runtime environment has no robosuite controller hook"
            )
        pre_action(action, True)

    @staticmethod
    def _actual_joint_positions(
        env: object, joint_names: Sequence[str]
    ) -> np.ndarray:
        model, data = _raw_model_data(env)
        values: list[float] = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise ToolUseJournalRuntimeError(
                    f"runtime arm joint {joint_name!r} is absent"
                )
            values.append(float(data.qpos[model.jnt_qposadr[joint_id]]))
        return np.asarray(values, dtype=float)

    def execute(
        self,
        run: SimulationRun,
        *,
        report_id: str | None = None,
    ) -> ExecutionReport:
        """Execute a MotionPlan through controller torques and physics."""

        report_name = report_id or f"{run.run_id}:execution-report"
        plan = run.plan
        executed_events: list[ExecutedEvent] = []
        executed_event_ids: set[str] = set()
        max_tracking_error = 0.0
        max_eef_error: float | None = None
        collision_count = 0
        executed_time = 0.0
        failure: _PlaybackFailure | None = None

        def failed_report(
            status: ExecutionStatus = ExecutionStatus.FAILED,
        ) -> ExecutionReport:
            return self._build_report(
                run,
                report_id=report_name,
                status=status,
                executed_duration_s=executed_time,
                max_tracking_error=max_tracking_error,
                max_eef_error=max_eef_error,
                collision_count=collision_count,
                events=executed_events,
                failure=failure,
            )

        try:
            self._events_by_time(plan)
            timeline = self._timeline(plan)
            first_context = plan.segments[0].collision_context_before
            self._verify_runtime_context(first_context, label="plan start")
            env = self.runtime.env
            model, data = _raw_model_data(env)
            control_timestep = float(env.control_timestep)  # type: ignore[attr-defined]
            model_timestep = float(env.model_timestep)  # type: ignore[attr-defined]
            if not math.isclose(
                control_timestep,
                run.config.control_timestep_s,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ToolUseJournalRuntimeError(
                    "SimulationConfig control_timestep_s differs from runtime"
                )
            if not math.isclose(
                model_timestep,
                run.config.physics_timestep_s,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ToolUseJournalRuntimeError(
                    "SimulationConfig physics_timestep_s differs from runtime"
                )
            # Validate the controller before any event mutates runtime state.
            initial_desired = self._desired_joint_position(timeline, 0.0)
            self._controller_action(plan, initial_desired)
            start_simulation_time = float(data.time)
            sorted_events = [
                event
                for _, event in sorted(
                    enumerate(plan.events),
                    key=lambda item: (item[1].time_from_start_s, item[0]),
                )
            ]
            next_event_index = 0
            next_eef_waypoint_index = 0
            verified_segment_ends: set[str] = set()

            while True:
                desired_now = self._desired_joint_position(
                    timeline, min(executed_time, plan.duration_s)
                )
                while (
                    next_event_index < len(sorted_events)
                    and sorted_events[next_event_index].time_from_start_s
                    <= executed_time + self._TIME_TOLERANCE_S
                ):
                    event = sorted_events[next_event_index]
                    try:
                        message = self._execute_event(event)
                        if event.event_type in {
                            EventType.GRIPPER_OPEN,
                            EventType.GRIPPER_CLOSE,
                            EventType.SUCTION_ON,
                            EventType.SUCTION_OFF,
                        }:
                            self._prime_gripper_command(plan, desired_now)
                    except Exception as error:  # noqa: BLE001
                        executed_events.append(
                            ExecutedEvent(
                                event_id=event.event_id,
                                scheduled_time_s=event.time_from_start_s,
                                actual_time_s=executed_time,
                                status=EventExecutionStatus.FAILED,
                                message=str(error),
                            )
                        )
                        failure = _PlaybackFailure(
                            code="EVENT_EXECUTION_FAILED",
                            message=str(error),
                            event_id=event.event_id,
                            observed={
                                "active_ee": self.runtime.active_ee,
                                "attached_object_id": self.runtime.attached_object_id,
                            },
                        )
                        return failed_report()
                    executed_event_ids.add(event.event_id)
                    executed_events.append(
                        ExecutedEvent(
                            event_id=event.event_id,
                            scheduled_time_s=event.time_from_start_s,
                            actual_time_s=executed_time,
                            status=EventExecutionStatus.SUCCESS,
                            message=message,
                        )
                    )
                    next_event_index += 1

                for completed_segment in plan.segments:
                    if (
                        completed_segment.segment_id
                        not in verified_segment_ends
                        and completed_segment.end_time_s
                        <= executed_time + self._TIME_TOLERANCE_S
                    ):
                        self._verify_runtime_context(
                            completed_segment.collision_context_after,
                            label=(
                                f"segment {completed_segment.segment_id!r} end"
                            ),
                        )
                        verified_segment_ends.add(
                            completed_segment.segment_id
                        )

                if executed_time >= plan.duration_s - self._TIME_TOLERANCE_S:
                    break
                target_time = min(
                    executed_time + control_timestep, plan.duration_s
                )
                desired = self._desired_joint_position(timeline, target_time)
                action = self._controller_action(plan, desired)
                try:
                    self._advance_controller(action)
                except ToolUseJournalAttachmentBroken as error:
                    executed_time = max(
                        0.0,
                        error.observation.simulation_time_s
                        - start_simulation_time,
                    )
                    failure = _PlaybackFailure(
                        code="GRASP_LOST",
                        message=str(error),
                        observed=error.observation.as_mapping(),
                    )
                    return failed_report()
                _, current_data = _raw_model_data(self.runtime.env)
                executed_time = float(current_data.time) - start_simulation_time
                actual = self._actual_joint_positions(
                    self.runtime.env, plan.joint_names
                )
                desired_at_actual = self._desired_joint_position(
                    timeline, min(executed_time, plan.duration_s)
                )
                max_tracking_error = max(
                    max_tracking_error,
                    float(np.max(np.abs(actual - desired_at_actual))),
                )
                while (
                    next_eef_waypoint_index < len(timeline)
                    and timeline[next_eef_waypoint_index].time_from_start_s
                    <= executed_time + self._TIME_TOLERANCE_S
                ):
                    eef_waypoint = timeline[next_eef_waypoint_index]
                    if eef_waypoint.eef_pose is not None:
                        eef_error = float(
                            np.linalg.norm(
                                self._eef_position(self.runtime.env)
                                - np.asarray(
                                    eef_waypoint.eef_pose.position_m,
                                    dtype=float,
                                )
                            )
                        )
                        max_eef_error = (
                            eef_error
                            if max_eef_error is None
                            else max(max_eef_error, eef_error)
                        )
                    next_eef_waypoint_index += 1
                if (
                    run.config.max_duration_s is not None
                    and executed_time
                    > run.config.max_duration_s + self._TIME_TOLERANCE_S
                ):
                    failure = _PlaybackFailure(
                        code="SIMULATION_TIMEOUT",
                        message="controller execution exceeded max_duration_s",
                    )
                    return failed_report(ExecutionStatus.TIMEOUT)

                segment = self._segment_at_time(
                    plan, min(executed_time, plan.duration_s)
                )
                if (
                    self._collision_probe is not None
                    and segment.collision_context_before is not None
                ):
                    collision = self._collision_probe.check(
                        actual,
                        context=segment.collision_context_before,
                    )
                    if not collision.valid:
                        collision_count += 1
                        if run.config.terminate_on_collision:
                            failure = _PlaybackFailure(
                                code="EXECUTION_COLLISION",
                                message=collision.detail,
                                segment_id=segment.segment_id,
                                observed={
                                    "failure_code": collision.failure_code,
                                    "min_clearance_m": collision.min_clearance_m,
                                },
                            )
                            return failed_report()
                if run.config.render:
                    render = getattr(self.runtime.env, "render", None)
                    if callable(render):
                        render()
                if run.config.realtime_factor > 0.0:
                    time.sleep(control_timestep / run.config.realtime_factor)

            missing_events = [
                event.event_id
                for event in plan.events
                if event.event_id not in executed_event_ids
            ]
            if missing_events:
                raise ToolUseJournalRuntimeError(
                    f"events were not executed: {missing_events}"
                )
            final_context = plan.segments[-1].collision_context_after
            self._verify_runtime_context(final_context, label="plan end")
            self._verify_final_runtime_state(plan)
        except Exception as error:  # noqa: BLE001
            failure = _PlaybackFailure(
                code="PLAYBACK_RUNTIME_FAILED",
                message=str(error),
                observed={
                    "active_ee": self.runtime.active_ee,
                    "attached_object_id": self.runtime.attached_object_id,
                },
            )
            return failed_report()

        return self._build_report(
            run,
            report_id=report_name,
            status=ExecutionStatus.SUCCESS,
            executed_duration_s=executed_time,
            max_tracking_error=max_tracking_error,
            max_eef_error=max_eef_error,
            collision_count=collision_count,
            events=executed_events,
            failure=None,
        )
