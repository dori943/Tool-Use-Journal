"""Versioned, model-checked checkpoints for an M5 execution runtime.

Unlike a raw MuJoCo ``qpos`` dump, a checkpoint preserves the named physical
state together with the logical gripper and attachment state needed to resume
planning without pretending that an already-held object is a new grasp.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import mujoco

from tuj.m5_motion.tool_use_journal import _name, _raw_model_data
from tuj.m5_motion.tool_use_journal_runtime import (
    AttachedObjectState,
    AttachmentMode,
    BreakableWeldConfig,
    ToolUseJournalEERuntime,
    ToolUseJournalRuntimeError,
    _RuntimeState,
    _capture_runtime_state,
    _joint_dof_width,
    _joint_qpos_width,
    _restore_runtime_state,
)


CHECKPOINT_FORMAT = "tuj-m5-runtime-checkpoint"
CHECKPOINT_VERSION = 1
MODEL_SIGNATURE_KIND = "tuj-m5-model-topology-v1"


class RuntimeCheckpointError(ToolUseJournalRuntimeError):
    """A checkpoint is malformed or does not describe the current runtime."""


@dataclass(frozen=True, slots=True)
class RuntimeCheckpointRestore:
    path: Path
    progress: Mapping[str, Any]
    attached_object_id: str | None


def _model_signature(runtime: ToolUseJournalEERuntime) -> str:
    model, _data = _raw_model_data(runtime.env)
    joints = []
    for joint_id in range(model.njnt):
        name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = int(model.jnt_type[joint_id])
        joints.append(
            (
                name,
                joint_type,
                _joint_qpos_width(joint_type),
                _joint_dof_width(joint_type),
            )
        )
    actuators = [
        _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        for actuator_id in range(model.nu)
    ]
    bodies = [
        (
            _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
            int(model.body_parentid[body_id]),
        )
        for body_id in range(model.nbody)
    ]
    geoms = [
        (
            _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
            int(model.geom_type[geom_id]),
            int(model.geom_bodyid[geom_id]),
            int(model.geom_contype[geom_id]),
            int(model.geom_conaffinity[geom_id]),
        )
        for geom_id in range(model.ngeom)
    ]
    payload = {
        "environment_name": runtime.environment_name,
        "active_ee": runtime.active_ee,
        "model_dimensions": [
            model.nq,
            model.nv,
            model.nu,
            model.njnt,
            model.nbody,
            model.ngeom,
        ],
        "joints": joints,
        "actuators": actuators,
        "bodies": bodies,
        "geoms": geoms,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _attachment_payload(attachment: AttachedObjectState | None) -> dict[str, Any] | None:
    if attachment is None:
        return None
    payload = asdict(attachment)
    payload["mode"] = attachment.mode.value
    return payload


def _attachment_from_payload(raw: object) -> AttachedObjectState | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RuntimeCheckpointError("checkpoint attachment must be an object")
    try:
        weld_raw = raw.get("breakable_weld")
        if isinstance(weld_raw, Mapping) and set(weld_raw) - {
            field.name for field in fields(BreakableWeldConfig)
        }:
            raise ValueError("unknown checkpoint weld configuration field")
        weld = (
            BreakableWeldConfig.from_parameters(weld_raw)
            if isinstance(weld_raw, Mapping)
            else None
        )
        rotation = tuple(
            tuple(float(value) for value in row)
            for row in raw["rotation_in_reference"]
        )
        if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
            raise ValueError("rotation must be 3x3")
        return AttachedObjectState(
            object_id=str(raw["object_id"]),
            free_joint_name=str(raw["free_joint_name"]),
            reference_kind=str(raw["reference_kind"]),
            reference_name=str(raw["reference_name"]),
            position_in_reference_m=tuple(
                float(value) for value in raw["position_in_reference_m"]
            ),
            rotation_in_reference=rotation,
            attach_distance_m=float(raw["attach_distance_m"]),
            mode=AttachmentMode(str(raw["mode"])),
            breakable_weld=weld,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeCheckpointError(
            f"invalid checkpoint attachment: {error}"
        ) from error


def capture_runtime_checkpoint(
    runtime: ToolUseJournalEERuntime,
    *,
    progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture physical and logical state using stable joint/actuator names."""

    if runtime.attachment is not None and runtime.attachment.mode is AttachmentMode.KINEMATIC:
        runtime.synchronize_attached_object()
    state = _capture_runtime_state(runtime.env)
    captured = runtime.captured_gripper_action
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "environment_name": runtime.environment_name,
        "active_ee": runtime.active_ee,
        "model_signature_kind": MODEL_SIGNATURE_KIND,
        "model_signature": _model_signature(runtime),
        "physical_state": {
            "qpos_by_joint": state.qpos_by_joint,
            "qvel_by_joint": state.qvel_by_joint,
            "ctrl_by_actuator": state.ctrl_by_actuator,
            "simulation_time_s": state.simulation_time_s,
        },
        "logical_state": {
            "gripper_command": runtime.gripper_command,
            "grasp_engaged": runtime.grasp_engaged,
            "captured_gripper_action": (
                list(captured) if captured is not None else None
            ),
            "attachment": _attachment_payload(runtime.attachment),
            "held_tool_id": runtime.held_tool_id,
        },
        "progress": dict(progress or {}),
    }


def save_runtime_checkpoint(
    path: str | Path,
    runtime: ToolUseJournalEERuntime,
    *,
    progress: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = capture_runtime_checkpoint(runtime, progress=progress)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def restore_runtime_checkpoint(
    path: str | Path,
    runtime: ToolUseJournalEERuntime,
    *,
    attachment_position_tolerance_m: float = 5e-3,
    attachment_orientation_tolerance_rad: float = 5e-2,
) -> RuntimeCheckpointRestore:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeCheckpointError(f"cannot read checkpoint {source}: {error}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeCheckpointError("checkpoint root must be an object")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeCheckpointError("unsupported checkpoint format")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise RuntimeCheckpointError(
            f"unsupported checkpoint version {payload.get('version')!r}"
        )
    if payload.get("environment_name") != runtime.environment_name:
        raise RuntimeCheckpointError(
            "checkpoint environment does not match the runtime"
        )
    if payload.get("active_ee") != runtime.active_ee:
        raise RuntimeCheckpointError(
            f"checkpoint requires active EE {payload.get('active_ee')!r}; "
            f"runtime has {runtime.active_ee!r}"
        )
    physical = payload.get("physical_state")
    logical = payload.get("logical_state")
    progress = payload.get("progress", {})
    if not isinstance(physical, Mapping) or not isinstance(logical, Mapping):
        raise RuntimeCheckpointError("checkpoint physical/logical state is missing")
    if not isinstance(progress, Mapping):
        raise RuntimeCheckpointError("checkpoint progress must be an object")
    try:
        state = _RuntimeState(
            qpos_by_joint={
                str(name): tuple(float(value) for value in values)
                for name, values in dict(physical["qpos_by_joint"]).items()
            },
            qvel_by_joint={
                str(name): tuple(float(value) for value in values)
                for name, values in dict(physical["qvel_by_joint"]).items()
            },
            ctrl_by_actuator={
                str(name): float(value)
                for name, value in dict(physical["ctrl_by_actuator"]).items()
            },
            simulation_time_s=float(physical["simulation_time_s"]),
        )
        attachment = _attachment_from_payload(logical.get("attachment"))
        captured = logical.get("captured_gripper_action")
        captured_values = (
            tuple(float(value) for value in captured)
            if captured is not None
            else None
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeCheckpointError(f"invalid checkpoint state: {error}") from error

    signature_kind = payload.get("model_signature_kind")
    if signature_kind == MODEL_SIGNATURE_KIND:
        if payload.get("model_signature") != _model_signature(runtime):
            raise RuntimeCheckpointError("checkpoint model topology does not match")
    elif signature_kind is None:
        # Early v1 checkpoints hashed the full generated MJCF. That string can
        # change with reset seed even when the compiled topology is compatible.
        # Accept those files only after an exact named-state layout check.
        current = _capture_runtime_state(runtime.env)
        qpos_layout = {name: len(values) for name, values in state.qpos_by_joint.items()}
        qvel_layout = {name: len(values) for name, values in state.qvel_by_joint.items()}
        current_qpos_layout = {
            name: len(values) for name, values in current.qpos_by_joint.items()
        }
        current_qvel_layout = {
            name: len(values) for name, values in current.qvel_by_joint.items()
        }
        if (
            qpos_layout != current_qpos_layout
            or qvel_layout != current_qvel_layout
            or set(state.ctrl_by_actuator) != set(current.ctrl_by_actuator)
        ):
            raise RuntimeCheckpointError(
                "legacy checkpoint named-state topology does not match"
            )
    else:
        raise RuntimeCheckpointError(
            f"unsupported model signature kind {signature_kind!r}"
        )

    current_state = _capture_runtime_state(runtime.env)
    qpos_layout = {name: len(values) for name, values in state.qpos_by_joint.items()}
    qvel_layout = {name: len(values) for name, values in state.qvel_by_joint.items()}
    current_qpos_layout = {
        name: len(values) for name, values in current_state.qpos_by_joint.items()
    }
    current_qvel_layout = {
        name: len(values) for name, values in current_state.qvel_by_joint.items()
    }
    if (
        qpos_layout != current_qpos_layout
        or qvel_layout != current_qvel_layout
        or set(state.ctrl_by_actuator) != set(current_state.ctrl_by_actuator)
    ):
        raise RuntimeCheckpointError(
            "checkpoint named-state topology does not match"
        )
    numeric_values = [
        state.simulation_time_s,
        *state.ctrl_by_actuator.values(),
        *(value for values in state.qpos_by_joint.values() for value in values),
        *(value for values in state.qvel_by_joint.values() for value in values),
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise RuntimeCheckpointError("checkpoint physical state must be finite")

    previous_logical = {
        "gripper_command": runtime._gripper_command,
        "grasp_engaged": runtime._grasp_engaged,
        "captured_gripper_action": (
            runtime._captured_gripper_action.copy()
            if runtime._captured_gripper_action is not None
            else None
        ),
        "attachment": runtime._attachment,
        "held_tool_id": runtime._held_tool_id,
        "contact_friction_retention": runtime._contact_friction_retention,
        "last_attachment_break": runtime._last_attachment_break,
        "breakable_runtime": copy.deepcopy(runtime._breakable_runtime),
    }
    try:
        _restore_runtime_state(runtime.env, state)
        runtime.restore_logical_state(
            gripper_command=float(logical.get("gripper_command", -1.0)),
            grasp_engaged=bool(logical.get("grasp_engaged", False)),
            captured_gripper_action=captured_values,
            attachment=attachment,
            held_tool_id=(
                str(logical["held_tool_id"])
                if logical.get("held_tool_id") is not None
                else None
            ),
            attachment_position_tolerance_m=attachment_position_tolerance_m,
            attachment_orientation_tolerance_rad=attachment_orientation_tolerance_rad,
        )
    except Exception as error:
        try:
            _restore_runtime_state(runtime.env, current_state)
        except Exception as rollback_error:  # pragma: no cover - catastrophic MuJoCo failure
            raise RuntimeCheckpointError(
                "checkpoint restore failed and physical rollback also failed: "
                f"{rollback_error}"
            ) from error
        finally:
            runtime._gripper_command = previous_logical["gripper_command"]
            runtime._grasp_engaged = previous_logical["grasp_engaged"]
            runtime._captured_gripper_action = previous_logical[
                "captured_gripper_action"
            ]
            runtime._attachment = previous_logical["attachment"]
            runtime._held_tool_id = previous_logical["held_tool_id"]
            runtime._contact_friction_retention = previous_logical[
                "contact_friction_retention"
            ]
            runtime._last_attachment_break = previous_logical[
                "last_attachment_break"
            ]
            runtime._breakable_runtime = previous_logical["breakable_runtime"]
        raise RuntimeCheckpointError(
            f"checkpoint restore rejected and was rolled back: {error}"
        ) from error
    return RuntimeCheckpointRestore(
        path=source,
        progress=dict(progress),
        attached_object_id=runtime.attached_object_id,
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "MODEL_SIGNATURE_KIND",
    "RuntimeCheckpointError",
    "RuntimeCheckpointRestore",
    "capture_runtime_checkpoint",
    "restore_runtime_checkpoint",
    "save_runtime_checkpoint",
]
