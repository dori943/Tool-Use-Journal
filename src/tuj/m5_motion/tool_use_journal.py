"""Compatibility layer for Tool-Use-Journal task environments.

The target environments expose useful task and EE-rack metadata, but their
rack geometry is display-only and an omitted ``gripper_types`` argument builds
a physical ``NullGripper`` even though ``robot_spec.json`` declares ``2F`` as
the current EE.  This module keeps those facts explicit:

* live MuJoCo state becomes an immutable :class:`WorldSnapshot`;
* rack dock poses come from MJCF-backed ``ee_rack_info``;
* planner-owned model copies promote rack collision geometry; and
* bare / 2F / 3F / vacuum models are compiled separately and synchronized to
  the same live object and arm state.

The source robosuite environment is never mutated.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from tuj.m5_motion.ee_exchange import EEExchangeTemplateGenerator
from tuj.m5_motion.kinematics import UR5eKinematics
from tuj.m5_motion.mujoco_collision import (
    MuJoCoCollisionConfigurationError,
    MuJoCoCollisionModelRegistry,
    MuJoCoCollisionValidator,
)
from tuj.m5_motion.schema import (
    AttachedObjectTransform,
    CollisionContext,
    GripperMode,
    GripperState,
    Pose,
    RobotState,
    SceneRef,
    WorldSnapshot,
)


TOOL_USE_JOURNAL_EE_GRIPPER_TYPES: dict[str, str] = {
    "2F": "Robotiq85Gripper",
    "3F": "JacoThreeFingerDexterousGripper",
    "vac": "VacuumGripper",
}
TOOL_USE_JOURNAL_TESTED_REVISION = (
    "113f84686d94203dbd90f1836187e351aa0b246d"
)
_EXPECTED_EES = frozenset(TOOL_USE_JOURNAL_EE_GRIPPER_TYPES)
_REFERENCE_ACTIVE_EE = object()
_PHYSICAL_EE_BY_CLASS = {
    "NullGripper": None,
    "Robotiq85Gripper": "2F",
    "JacoThreeFingerDexterousGripper": "3F",
    "VacuumGripper": "vac",
}


class ToolUseJournalCompatibilityError(ValueError):
    """The target environment cannot satisfy the planner contract."""


def _raw_model_data(env: object) -> tuple[mujoco.MjModel, mujoco.MjData]:
    try:
        return env.sim.model._model, env.sim.data._data  # type: ignore[attr-defined]
    except AttributeError as error:
        raise ToolUseJournalCompatibilityError(
            "env must be a reset robosuite environment"
        ) from error


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, object_id: int) -> str:
    return mujoco.mj_id2name(model, kind, object_id) or ""


def _quaternion_wxyz_to_xyzw(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) != 4:
        raise ToolUseJournalCompatibilityError("quaternion must have four values")
    w, x, y, z = (float(value) for value in values)
    quaternion = np.asarray((x, y, z, w), dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ToolUseJournalCompatibilityError(
            "quaternion must be finite and non-zero"
        )
    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def _physical_ee(env: object) -> str | None:
    try:
        grippers = env.robots[0].gripper  # type: ignore[attr-defined]
    except (AttributeError, IndexError) as error:
        raise ToolUseJournalCompatibilityError(
            "env must expose exactly one robosuite robot"
        ) from error
    values = list(grippers.values()) if isinstance(grippers, Mapping) else [grippers]
    if len(values) != 1:
        raise ToolUseJournalCompatibilityError(
            f"expected one mounted gripper, found {len(values)}"
        )
    class_name = type(values[0]).__name__
    if class_name not in _PHYSICAL_EE_BY_CLASS:
        raise ToolUseJournalCompatibilityError(
            f"unsupported mounted gripper {class_name!r}"
        )
    return _PHYSICAL_EE_BY_CLASS[class_name]


def _mounted_root_and_hand(env: object) -> tuple[str, str]:
    model, _ = _raw_model_data(env)
    grippers = env.robots[0].gripper  # type: ignore[attr-defined]
    gripper = next(iter(grippers.values())) if isinstance(grippers, Mapping) else grippers
    root = str(gripper.root_body)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
    if root_id < 0:
        raise ToolUseJournalCompatibilityError(
            f"mounted gripper root {root!r} is absent from MuJoCo model"
        )
    parent_id = int(model.body_parentid[root_id])
    hand = _name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
    if not hand:
        raise ToolUseJournalCompatibilityError(
            f"mounted gripper root {root!r} has no named parent"
        )
    return root, hand


def _nearest_collision_ancestor(
    model: mujoco.MjModel, body_name: str
) -> str:
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )
    while body_id > 0:
        own_collision = any(
            int(model.geom_bodyid[geom_id]) == body_id
            and (
                int(model.geom_contype[geom_id])
                or int(model.geom_conaffinity[geom_id])
            )
            for geom_id in range(model.ngeom)
        )
        if own_collision:
            return _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        body_id = int(model.body_parentid[body_id])
    raise ToolUseJournalCompatibilityError(
        f"body {body_name!r} has no collision-enabled ancestor"
    )


def _descendant_body_ids(model: mujoco.MjModel, root_body_id: int) -> set[int]:
    result: set[int] = set()
    for candidate in range(model.nbody):
        current = candidate
        for _ in range(model.nbody + 1):
            if current == root_body_id:
                result.add(candidate)
                break
            if current == 0:
                break
            current = int(model.body_parentid[current])
    return result


def _subtree_geom_ids(model: mujoco.MjModel, root_body_id: int) -> list[int]:
    bodies = _descendant_body_ids(model, root_body_id)
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in bodies
    ]


def _box_corners(half_extents: Sequence[float]) -> np.ndarray:
    x, y, z = (float(value) for value in half_extents)
    return np.asarray(
        [
            (sx * x, sy * y, sz * z)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=float,
    )


def _geom_local_points(model: mujoco.MjModel, geom_id: int) -> np.ndarray | None:
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            return None
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        return np.asarray(model.mesh_vert[start : start + count], dtype=float)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return _box_corners((size[0], size[0], size[0]))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return _box_corners(size[:3])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return _box_corners(size[:3])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return _box_corners((size[0], size[0], size[1]))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return _box_corners((size[0], size[0], size[1] + size[0]))
    return None


def _body_local_bounds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_body_id: int,
    *,
    collision_only: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    root_position = np.asarray(data.xpos[root_body_id], dtype=float)
    root_rotation = np.asarray(data.xmat[root_body_id], dtype=float).reshape(3, 3)
    points: list[np.ndarray] = []
    for geom_id in _subtree_geom_ids(model, root_body_id):
        collision_enabled = bool(
            int(model.geom_contype[geom_id])
            or int(model.geom_conaffinity[geom_id])
        )
        if collision_only and not collision_enabled:
            continue
        local = _geom_local_points(model, geom_id)
        if local is None or local.size == 0:
            continue
        geom_position = np.asarray(data.geom_xpos[geom_id], dtype=float)
        geom_rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        world = local @ geom_rotation.T + geom_position
        points.append((world - root_position) @ root_rotation)
    if not points:
        return None
    combined = np.concatenate(points, axis=0)
    return np.min(combined, axis=0), np.max(combined, axis=0)


def _pose_record(
    position: Sequence[float], quaternion_wxyz: Sequence[float]
) -> dict[str, Any]:
    return {
        "frame_id": "world",
        "position_m": [float(value) for value in position],
        "orientation_xyzw": list(_quaternion_wxyz_to_xyzw(quaternion_wxyz)),
    }


def _object_record(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_id: str,
    body_id: int,
) -> dict[str, Any]:
    geom_ids = _subtree_geom_ids(model, body_id)
    collision_ids = [
        geom_id
        for geom_id in geom_ids
        if int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])
    ]
    bounds = _body_local_bounds(
        model, data, body_id, collision_only=bool(collision_ids)
    )
    record: dict[str, Any] = {
        "body_name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
        "pose": _pose_record(data.xpos[body_id], data.xquat[body_id]),
        "collision_enabled": bool(collision_ids),
        "geom_names": [
            _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for geom_id in geom_ids
        ],
    }
    if bounds is not None:
        lower, upper = bounds
        center = (lower + upper) * 0.5
        record["dimensions_m"] = [float(value) for value in upper - lower]
        record["anchors"] = {
            "center": [float(value) for value in center],
            "top": [float(center[0]), float(center[1]), float(upper[2])],
            "top_center": [
                float(center[0]),
                float(center[1]),
                float(upper[2]),
            ],
            "bottom": [float(center[0]), float(center[1]), float(lower[2])],
            "bottom_center": [
                float(center[0]),
                float(center[1]),
                float(lower[2]),
            ],
        }
    free_joints = [
        _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if int(model.jnt_bodyid[joint_id]) == body_id
        and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if free_joints:
        record["free_joint_name"] = free_joints[0]
    record["object_id"] = object_id
    return record


def _declared_current_ee(env: object) -> str | None:
    direct = getattr(env, "current_ee_id", None)
    if direct is not None:
        return str(direct)
    spec = getattr(env, "robot_spec", None)
    if isinstance(spec, Mapping) and spec.get("current_ee") is not None:
        return str(spec["current_ee"])
    return None


def _load_repository_environments(repository_root: str | Path) -> object:
    """Import the checkout-local ``environments`` registration package."""

    root = Path(repository_root).resolve()
    if not (root / "environments" / "__init__.py").is_file():
        raise ToolUseJournalCompatibilityError(
            f"{root} is not a Tool-Use-Journal checkout"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    loaded = sys.modules.get("environments")
    if loaded is not None:
        loaded_file = Path(str(getattr(loaded, "__file__", ""))).resolve()
        if root not in loaded_file.parents:
            raise ToolUseJournalCompatibilityError(
                "another top-level 'environments' package is already imported"
            )
    importlib.invalidate_caches()
    return importlib.import_module("environments")


def registered_tool_use_journal_environments(
    repository_root: str | Path,
) -> frozenset[str]:
    """Discover checkout-local robosuite environments after package import.

    Importing ``environments/__init__.py`` is the single registration step.
    Any robosuite-registered class implemented by that package is accepted;
    the runtime adapter separately verifies the robot, rack, and object API.
    """

    _load_repository_environments(repository_root)
    from robosuite.environments.base import REGISTERED_ENVS

    names = {
        str(name)
        for name, environment_type in REGISTERED_ENVS.items()
        if str(getattr(environment_type, "__module__", "")) == "environments"
        or str(getattr(environment_type, "__module__", "")).startswith(
            "environments."
        )
    }
    if not names:
        raise ToolUseJournalCompatibilityError(
            "environments/__init__.py did not register any checkout-local "
            "robosuite environments"
        )
    return frozenset(names)


def _environment_name(env: object) -> str:
    name = type(env).__name__
    if not name:
        raise ToolUseJournalCompatibilityError(
            "environment class has no registered name"
        )
    return name


def make_tool_use_journal_env(
    repository_root: str | Path,
    env_name: str,
    *,
    active_ee: str | None,
    **suite_make_kwargs: Any,
) -> object:
    """Create one target environment with a physically matching mounted EE.

    ``active_ee=None`` deliberately requests robosuite's ``NullGripper`` bare
    model.  The caller owns the returned environment and should close it.
    """

    registered = registered_tool_use_journal_environments(repository_root)
    if env_name not in registered:
        raise ToolUseJournalCompatibilityError(
            f"environment {env_name!r} is not registered by "
            "environments/__init__.py; available checkout environments are "
            f"{sorted(registered)}"
        )
    if active_ee is not None and active_ee not in _EXPECTED_EES:
        raise ToolUseJournalCompatibilityError(
            f"unsupported EE {active_ee!r}; expected one of {sorted(_EXPECTED_EES)}"
        )
    import robosuite as suite

    options = {
        "robots": "UR5e",
        "gripper_types": (
            TOOL_USE_JOURNAL_EE_GRIPPER_TYPES[active_ee]
            if active_ee is not None
            else None
        ),
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        # C1's branch default names a camera that is not in the compiled model.
        "render_camera": "frontview",
        "initialization_noise": None,
        "hard_reset": False,
    }
    options.update(suite_make_kwargs)
    options["gripper_types"] = (
        TOOL_USE_JOURNAL_EE_GRIPPER_TYPES[active_ee]
        if active_ee is not None
        else None
    )
    return suite.make(env_name=env_name, **options)


class ToolUseJournalEnvironmentAdapter:
    """Read a reset registered workcell into the Motion Planner contract."""

    def __init__(
        self,
        env: object,
        *,
        source_revision: str = TOOL_USE_JOURNAL_TESTED_REVISION,
    ) -> None:
        self.env = env
        self.environment_name = _environment_name(env)
        self.source_revision = source_revision
        self.model, self.data = _raw_model_data(env)
        try:
            self.robot = env.robots[0]  # type: ignore[attr-defined]
            self.rack_info = dict(env.ee_rack_info)  # type: ignore[attr-defined]
            self.object_body_ids = {
                str(name): int(body_id)
                for name, body_id in dict(env.obj_body_id).items()  # type: ignore[attr-defined]
            }
            self.source_mjcf = str(env.model.get_xml())  # type: ignore[attr-defined]
        except (AttributeError, IndexError, TypeError) as error:
            raise ToolUseJournalCompatibilityError(
                "env is missing Tool-Use-Journal runtime metadata; call reset() first"
            ) from error
        if set(self.rack_info) != _EXPECTED_EES:
            raise ToolUseJournalCompatibilityError(
                f"rack EE ids are {sorted(self.rack_info)}; expected "
                f"{sorted(_EXPECTED_EES)}"
            )
        self.physical_active_ee = _physical_ee(env)
        self.declared_active_ee = _declared_current_ee(env)
        self.mounted_root_body, self.hand_body = _mounted_root_and_hand(env)

    @property
    def ee_metadata_matches_physics(self) -> bool:
        return self.declared_active_ee == self.physical_active_ee

    def require_physical_ee(self, expected: str | None = None) -> None:
        """Fail closed if metadata / requested EE differs from mounted geometry."""

        requested = self.declared_active_ee if expected is None else expected
        if requested != self.physical_active_ee:
            raise ToolUseJournalCompatibilityError(
                f"declared/requested EE {requested!r} does not match mounted "
                f"MuJoCo EE {self.physical_active_ee!r}; create the environment "
                "with make_tool_use_journal_env(..., active_ee=requested)"
            )

    def make_kinematics(self) -> UR5eKinematics:
        """Create IK for the active EE's MJCF-backed grasp reference pose."""

        grippers = getattr(self.robot, "gripper", {})
        gripper = (
            grippers.get("right") if isinstance(grippers, Mapping) else None
        )
        sites = getattr(gripper, "important_sites", {})
        site_name = sites.get("grip_site") if isinstance(sites, Mapping) else None
        if not site_name:
            return UR5eKinematics.from_robosuite_env(self.env)
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, str(site_name)
        )
        hand_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.hand_body
        )
        if site_id < 0 or hand_id < 0:
            raise ToolUseJournalCompatibilityError(
                "active EE grasp site or hand body is absent from MuJoCo"
            )
        hand_position = np.asarray(self.data.xpos[hand_id], dtype=float)
        hand_rotation = np.asarray(
            self.data.xmat[hand_id], dtype=float
        ).reshape(3, 3)
        site_position = np.asarray(self.data.site_xpos[site_id], dtype=float)
        site_rotation = np.asarray(
            self.data.site_xmat[site_id], dtype=float
        ).reshape(3, 3)
        position_in_hand = hand_rotation.T @ (
            site_position - hand_position
        )
        rotation_in_hand = hand_rotation.T @ site_rotation
        quaternion_wxyz = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(
            quaternion_wxyz,
            np.ascontiguousarray(rotation_in_hand.reshape(9)),
        )
        return UR5eKinematics.from_robosuite_env(
            self.env,
            target_position_in_eef_m=tuple(
                float(value) for value in position_in_hand
            ),
            target_orientation_in_eef_xyzw=_quaternion_wxyz_to_xyzw(
                quaternion_wxyz
            ),
        )

    def _arm_state(self, attached_object_id: str | None) -> RobotState:
        joint_names = tuple(str(name) for name in self.robot.robot_model.joints)
        positions: list[float] = []
        velocities: list[float] = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise ToolUseJournalCompatibilityError(
                    f"arm joint {joint_name!r} is absent from MuJoCo model"
                )
            if int(self.model.jnt_type[joint_id]) not in {
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            }:
                raise ToolUseJournalCompatibilityError(
                    f"arm joint {joint_name!r} is not scalar"
                )
            positions.append(float(self.data.qpos[self.model.jnt_qposadr[joint_id]]))
            velocities.append(float(self.data.qvel[self.model.jnt_dofadr[joint_id]]))
        hand_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.hand_body
        )
        robot_spec = getattr(self.env, "robot_spec", {})
        robot_id = (
            str(robot_spec.get("robot_id", "ur5e_0"))
            if isinstance(robot_spec, Mapping)
            else "ur5e_0"
        )
        return RobotState(
            robot_id=robot_id,
            joint_names=list(joint_names),
            joint_positions_rad=positions,
            joint_velocities_rad_s=velocities,
            eef_pose=Pose(
                frame_id="world",
                position_m=tuple(float(value) for value in self.data.xpos[hand_id]),
                orientation_xyzw=_quaternion_wxyz_to_xyzw(
                    self.data.xquat[hand_id]
                ),
            ),
            gripper=GripperState(mode=GripperMode.UNKNOWN),
            attached_object_id=attached_object_id,
        )

    def _rack_records(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for ee_id, raw in self.rack_info.items():
            if not isinstance(raw, Mapping):
                raise ToolUseJournalCompatibilityError(
                    f"rack record {ee_id!r} is not a mapping"
                )
            position = raw.get("rack_position")
            orientation = raw.get("rack_orientation")
            try:
                position_values = np.asarray(position, dtype=float)
                orientation_values = np.asarray(orientation, dtype=float)
            except (TypeError, ValueError) as error:
                raise ToolUseJournalCompatibilityError(
                    f"rack record {ee_id!r} has no MJCF-backed pose"
                ) from error
            if position_values.shape != (3,) or orientation_values.shape != (4,):
                raise ToolUseJournalCompatibilityError(
                    f"rack record {ee_id!r} has no MJCF-backed pose"
                )
            result[ee_id] = {
                "dock_pose": _pose_record(position_values, orientation_values),
                # Rack display quaternion flips local +Z to world -Z.  A
                # negative template offset therefore stages above the rack.
                "approach_axis_xyz": [0.0, 0.0, 1.0],
                "staging_distance_m": 0.15,
                "pre_dock_distance_m": 0.04,
                "rack_body": str(raw.get("rack_body", "")),
                "rack_slot": str(raw.get("rack_slot", "")),
                "support_geom": f"ee_rack_support_{ee_id}",
                "rack_support_top_z": float(raw["rack_support_top_z"]),
                "rack_support_height": float(raw["rack_support_height"]),
                "gripper_class": str(raw.get("gripper_class", "")),
                "source_collision_enabled": False,
            }
        return result

    def _obstacles(self) -> list[Any]:
        result: list[Any] = []
        for geom_id in range(self.model.ngeom):
            geom_name = _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not (
                geom_name.startswith("table_collision")
                or geom_name == "ee_rack_base"
                or geom_name.startswith("ee_rack_support_")
            ):
                continue
            local = _geom_local_points(self.model, geom_id)
            if local is None:
                continue
            world = (
                local
                @ np.asarray(self.data.geom_xmat[geom_id], dtype=float).reshape(3, 3).T
                + np.asarray(self.data.geom_xpos[geom_id], dtype=float)
            )
            lower, upper = np.min(world, axis=0), np.max(world, axis=0)
            result.append(
                {
                    "obstacle_id": geom_name,
                    "aabb_min_m": [float(value) for value in lower],
                    "aabb_max_m": [float(value) for value in upper],
                    "collision_enabled_in_source": bool(
                        int(self.model.geom_contype[geom_id])
                        or int(self.model.geom_conaffinity[geom_id])
                    ),
                }
            )
        return result

    def world_snapshot(
        self,
        *,
        completed_subgoals: Iterable[str] = (),
        facts: Iterable[str] = (),
        attached_object_id: str | None = None,
        attached_object_transform: AttachedObjectTransform | None = None,
    ) -> WorldSnapshot:
        """Capture one deterministic planning start state without mutating env."""

        if attached_object_transform is not None:
            if (
                attached_object_id is not None
                and attached_object_id != attached_object_transform.object_id
            ):
                raise ToolUseJournalCompatibilityError(
                    "attached_object_id does not match attached_object_transform"
                )
            attached_object_id = attached_object_transform.object_id
        mujoco.mj_forward(self.model, self.data)
        objects = {
            object_id: _object_record(
                self.model, self.data, object_id, body_id
            )
            for object_id, body_id in sorted(self.object_body_ids.items())
        }
        xml_hash = hashlib.sha256(self.source_mjcf.encode("utf-8")).hexdigest()
        signature_payload = {
            "adapter": "tool-use-journal-v1",
            "environment": self.environment_name,
            "source_revision": self.source_revision,
            "source_mjcf_sha256": xml_hash,
            "qpos": [float(value) for value in self.data.qpos],
            "physical_active_ee": self.physical_active_ee,
            "attached_object_id": attached_object_id,
            "attached_object_transform": (
                attached_object_transform.model_dump(mode="json")
                if attached_object_transform is not None
                else None
            ),
        }
        encoded = json.dumps(
            signature_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        signature = "tool-use-journal:" + hashlib.sha256(encoded).hexdigest()
        return WorldSnapshot(
            scene=SceneRef(
                signature=signature,
                completed_subgoals=list(completed_subgoals),
                facts=list(facts),
            ),
            robot_state=self._arm_state(attached_object_id),
            objects=objects,
            obstacles=self._obstacles(),
            rack=self._rack_records(),
            metadata={
                "adapter": "tool-use-journal-v1",
                "environment_name": self.environment_name,
                "source_revision": self.source_revision,
                "source_mjcf_sha256": xml_hash,
                "physical_active_ee": self.physical_active_ee,
                "declared_active_ee": self.declared_active_ee,
                "ee_metadata_matches_physics": self.ee_metadata_matches_physics,
                "rack_collision_policy": "PROMOTE_IN_PLANNER_COPY",
                "attached_object_transforms": (
                    {
                        attached_object_transform.object_id: (
                            attached_object_transform.model_dump(mode="json")
                        )
                    }
                    if attached_object_transform is not None
                    else {}
                ),
            },
        )


def _joint_qpos_width(joint_type: int) -> int:
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def _joint_state_map(
    model: mujoco.MjModel, data: mujoco.MjData
) -> dict[str, tuple[float, ...]]:
    result: dict[str, tuple[float, ...]] = {}
    for joint_id in range(model.njnt):
        name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            continue
        start = int(model.jnt_qposadr[joint_id])
        width = _joint_qpos_width(int(model.jnt_type[joint_id]))
        result[name] = tuple(float(value) for value in data.qpos[start : start + width])
    return result


@dataclass(frozen=True, slots=True)
class _CapturedEnvironment:
    environment_name: str
    active_ee: str | None
    source_mjcf: str
    source_joint_states: Mapping[str, tuple[float, ...]]
    joint_names: tuple[str, ...]
    robot_root_body_name: str
    hand_body_name: str
    qc_collision_body_name: str
    mounted_root_body_name: str
    rack_root_body_names: Mapping[str, str]
    object_body_names: Mapping[str, str]


def _capture_environment(env: object, expected_active_ee: str | None) -> _CapturedEnvironment:
    adapter = ToolUseJournalEnvironmentAdapter(env)
    if adapter.physical_active_ee != expected_active_ee:
        raise ToolUseJournalCompatibilityError(
            f"variant {expected_active_ee!r} physically contains "
            f"{adapter.physical_active_ee!r}"
        )
    rack_roots = {
        ee_id: str(record["rack_body"])
        for ee_id, record in adapter.rack_info.items()
    }
    object_bodies = {
        object_id: _name(
            adapter.model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        for object_id, body_id in adapter.object_body_ids.items()
    }
    return _CapturedEnvironment(
        environment_name=adapter.environment_name,
        active_ee=expected_active_ee,
        source_mjcf=adapter.source_mjcf,
        source_joint_states=_joint_state_map(adapter.model, adapter.data),
        joint_names=tuple(str(name) for name in adapter.robot.robot_model.joints),
        robot_root_body_name=str(adapter.robot.robot_model.root_body),
        hand_body_name=adapter.hand_body,
        qc_collision_body_name=_nearest_collision_ancestor(
            adapter.model, adapter.hand_body
        ),
        mounted_root_body_name=adapter.mounted_root_body,
        rack_root_body_names=rack_roots,
        object_body_names=object_bodies,
    )


def _body_elements(root: ET.Element) -> dict[str, ET.Element]:
    return {
        str(body.get("name")): body
        for body in root.findall(".//body")
        if body.get("name")
    }


def _promote_rack_collision_geometry(root: ET.Element) -> tuple[str, ...]:
    promoted: list[str] = []
    for geom in root.findall(".//geom"):
        name = str(geom.get("name", ""))
        group = int(geom.get("group", "0"))
        is_rack_structure = name == "ee_rack_base" or name.startswith(
            "ee_rack_support_"
        )
        is_rack_ee_collision = name.startswith("gripperrack_") and group == 0
        if not (is_rack_structure or is_rack_ee_collision):
            continue
        geom.set("contype", "1")
        geom.set("conaffinity", "1")
        promoted.append(name)
    if not promoted:
        raise ToolUseJournalCompatibilityError(
            "source MJCF has no Tool-Use-Journal rack geometry to promote"
        )
    return tuple(sorted(promoted))


def _remove_body(root: ET.Element, body_name: str) -> None:
    bodies = _body_elements(root)
    body = bodies.get(body_name)
    if body is None:
        raise ToolUseJournalCompatibilityError(
            f"source MJCF has no rack body {body_name!r}"
        )
    parent_map = {child: parent for parent in root.iter() for child in parent}
    parent = parent_map.get(body)
    if parent is None:
        raise ToolUseJournalCompatibilityError(
            f"rack body {body_name!r} has no XML parent"
        )
    parent.remove(body)


def _apply_joint_states(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    states: Mapping[str, Sequence[float]],
) -> None:
    for name, values in states.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        width = _joint_qpos_width(int(model.jnt_type[joint_id]))
        if len(values) != width:
            raise ToolUseJournalCompatibilityError(
                f"joint {name!r} state width changed between scene variants"
            )
        start = int(model.jnt_qposadr[joint_id])
        qpos[start : start + width] = values


def _selector_has_collision(model: mujoco.MjModel, selector: str) -> bool:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, selector)
    if geom_id >= 0:
        return bool(
            int(model.geom_contype[geom_id])
            or int(model.geom_conaffinity[geom_id])
        )
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, selector)
    if body_id < 0:
        return False
    return any(
        int(model.geom_contype[candidate])
        or int(model.geom_conaffinity[candidate])
        for candidate in _subtree_geom_ids(model, body_id)
    )


@dataclass(frozen=True, slots=True)
class CompiledToolUseJournalCollisionModel:
    """One planner-owned collision scene for a mounted EE state."""

    model: mujoco.MjModel
    collision_model_version: str
    active_ee: str | None
    joint_names: tuple[str, ...]
    robot_root_body_name: str
    baseline_qpos: tuple[float, ...]
    entity_selectors: tuple[tuple[str, tuple[str, ...]], ...]
    promoted_rack_geom_names: tuple[str, ...]
    mjcf_sha256: str

    def make_validator(
        self,
        *,
        collision_margin_m: float,
        collision_contexts: Mapping[str, CollisionContext] | None = None,
        allowed_collision_pairs: Iterable[tuple[str, str]] = (),
    ) -> MuJoCoCollisionValidator:
        base_allowed_pairs = list(allowed_collision_pairs)
        if self.active_ee is not None:
            # Robotiq/Jaco collision hulls overlap at their nominal coupled
            # finger state.  Those joints are fixed in an arm-path model, so
            # EE-internal contacts are invariant and are not arm self-collision.
            base_allowed_pairs.append((self.active_ee, self.active_ee))
        return MuJoCoCollisionValidator(
            self.model,
            joint_names=self.joint_names,
            robot_root_body_name=self.robot_root_body_name,
            baseline_qpos=self.baseline_qpos,
            collision_margin_m=collision_margin_m,
            collision_model_version=self.collision_model_version,
            collision_contexts=collision_contexts,
            entity_geoms=dict(self.entity_selectors),
            allowed_collision_pairs=base_allowed_pairs,
        )


class ToolUseJournalCollisionModelCompiler:
    """Compile synchronized bare and mounted-EE workcell model variants."""

    def __init__(
        self,
        captures: Mapping[str | None, _CapturedEnvironment],
        *,
        reference_joint_states: Mapping[str, tuple[float, ...]],
        source_revision: str = TOOL_USE_JOURNAL_TESTED_REVISION,
        reference_active_ee: str | None,
    ) -> None:
        expected_keys: set[str | None] = {None, *_EXPECTED_EES}
        if set(captures) != expected_keys:
            raise ToolUseJournalCompatibilityError(
                "collision compiler requires bare, 2F, 3F, and vac variants"
            )
        environments = {capture.environment_name for capture in captures.values()}
        joint_orders = {capture.joint_names for capture in captures.values()}
        robot_roots = {
            capture.robot_root_body_name for capture in captures.values()
        }
        if len(environments) != 1 or len(joint_orders) != 1 or len(robot_roots) != 1:
            raise ToolUseJournalCompatibilityError(
                "all collision variants must use the same task and UR5e arm"
            )
        if reference_active_ee not in expected_keys:
            raise ToolUseJournalCompatibilityError(
                f"unsupported reference EE {reference_active_ee!r}"
            )
        self._captures = dict(captures)
        self._reference_joint_states = dict(reference_joint_states)
        self.source_revision = source_revision
        self.environment_name = next(iter(environments))
        self.joint_names = next(iter(joint_orders))
        self.robot_root_body_name = next(iter(robot_roots))
        self.reference_active_ee = reference_active_ee
        revision_tag = (source_revision or "unknown")[:12]
        environment_tag = self.environment_name.lower()
        self.bare_model_version = (
            f"tool-use-journal-{environment_tag}-bare-{revision_tag}-v1"
        )
        self.attached_model_versions = {
            ee: f"tool-use-journal-{environment_tag}-{ee}-attached-{revision_tag}-v1"
            for ee in sorted(_EXPECTED_EES)
        }

    @classmethod
    def from_environments(
        cls,
        reference_env: object,
        variants: Mapping[str | None, object],
        *,
        source_revision: str = TOOL_USE_JOURNAL_TESTED_REVISION,
    ) -> "ToolUseJournalCollisionModelCompiler":
        reference_adapter = ToolUseJournalEnvironmentAdapter(
            reference_env, source_revision=source_revision
        )
        captures = {
            active_ee: _capture_environment(env, active_ee)
            for active_ee, env in variants.items()
        }
        return cls(
            captures,
            reference_joint_states=_joint_state_map(
                reference_adapter.model, reference_adapter.data
            ),
            source_revision=source_revision,
            reference_active_ee=reference_adapter.physical_active_ee,
        )

    @classmethod
    def from_environment_factory(
        cls,
        reference_env: object,
        factory: Callable[[str | None], object],
        *,
        source_revision: str = TOOL_USE_JOURNAL_TESTED_REVISION,
    ) -> "ToolUseJournalCollisionModelCompiler":
        variants: dict[str | None, object] = {}
        try:
            for active_ee in (None, "2F", "3F", "vac"):
                env = factory(active_ee)
                env.reset()  # type: ignore[attr-defined]
                variants[active_ee] = env
            return cls.from_environments(
                reference_env,
                variants,
                source_revision=source_revision,
            )
        finally:
            for env in variants.values():
                close = getattr(env, "close", None)
                if callable(close):
                    close()

    @classmethod
    def from_repository(
        cls,
        reference_env: object,
        repository_root: str | Path,
        *,
        seed: int = 0,
        source_revision: str = TOOL_USE_JOURNAL_TESTED_REVISION,
        **suite_make_kwargs: Any,
    ) -> "ToolUseJournalCollisionModelCompiler":
        env_name = _environment_name(reference_env)

        def factory(active_ee: str | None) -> object:
            return make_tool_use_journal_env(
                repository_root,
                env_name,
                active_ee=active_ee,
                seed=seed,
                **suite_make_kwargs,
            )

        return cls.from_environment_factory(
            reference_env,
            factory,
            source_revision=source_revision,
        )

    def model_version_for(self, active_ee: str | None) -> str:
        if active_ee is None:
            return self.bare_model_version
        try:
            return self.attached_model_versions[active_ee]
        except KeyError as error:
            raise ToolUseJournalCompatibilityError(
                f"unknown EE {active_ee!r}"
            ) from error

    def with_reference_environment(
        self, reference_env: object
    ) -> "ToolUseJournalCollisionModelCompiler":
        """Reuse captured model variants with the current live scene state.

        Constructing the four EE variants is intentionally expensive.  A
        receding-horizon planner only needs to refresh the common arm and
        free-object joint states between replans, while the underlying MJCF
        variants remain unchanged.
        """
        reference_adapter = ToolUseJournalEnvironmentAdapter(
            reference_env, source_revision=self.source_revision
        )
        if reference_adapter.environment_name != self.environment_name:
            raise ToolUseJournalCompatibilityError(
                "reference environment does not match captured collision "
                f"models: {reference_adapter.environment_name!r} != "
                f"{self.environment_name!r}"
            )
        reference_joint_names = tuple(
            str(name) for name in reference_adapter.robot.robot_model.joints
        )
        if reference_joint_names != self.joint_names:
            raise ToolUseJournalCompatibilityError(
                "reference environment robot joint order does not match "
                "captured collision models"
            )
        return type(self)(
            self._captures,
            reference_joint_states=_joint_state_map(
                reference_adapter.model, reference_adapter.data
            ),
            source_revision=self.source_revision,
            reference_active_ee=reference_adapter.physical_active_ee,
        )

    def build_ee_exchange_contexts(
        self, *, from_ee: str | None, to_ee: str
    ) -> dict[str, CollisionContext]:
        return EEExchangeTemplateGenerator().build_collision_contexts(
            from_ee=from_ee,
            to_ee=to_ee,
            bare_flange_model_version=self.bare_model_version,
            attached_model_versions=self.attached_model_versions,
        )

    def compile(
        self, active_ee: str | None
    ) -> CompiledToolUseJournalCollisionModel:
        version = self.model_version_for(active_ee)
        capture = self._captures[active_ee]
        root = ET.fromstring(capture.source_mjcf)
        promoted = _promote_rack_collision_geometry(root)
        if active_ee is not None:
            # The target scene always displays all rack EEs.  Once one is
            # mounted, its display duplicate must disappear from the physical
            # planner model so the slot is actually empty.
            _remove_body(root, capture.rack_root_body_names[active_ee])
        compiled_xml = ET.tostring(root, encoding="unicode")
        try:
            model = mujoco.MjModel.from_xml_string(compiled_xml)
        except Exception as error:  # noqa: BLE001
            raise ToolUseJournalCompatibilityError(
                f"MuJoCo failed to compile {version!r}: {error}"
            ) from error
        baseline = np.asarray(model.qpos0, dtype=float).copy()
        _apply_joint_states(model, baseline, capture.source_joint_states)
        # Reference state wins for common arm and free-object joints.  Gripper
        # joints that do not exist in the live variant keep their reset state.
        _apply_joint_states(model, baseline, self._reference_joint_states)

        entity_candidates: dict[str, tuple[str, ...]] = {
            "qc_master": (capture.qc_collision_body_name,),
            "rack": ("ee_rack_base",),
        }
        for object_id, body_name in capture.object_body_names.items():
            entity_candidates[object_id] = (body_name,)
        for ee in sorted(_EXPECTED_EES):
            entity_candidates[ee] = (
                (capture.mounted_root_body_name,)
                if ee == active_ee
                else (capture.rack_root_body_names[ee],)
            )
            entity_candidates[f"rack_support:{ee}"] = (
                f"ee_rack_support_{ee}",
            )
        entities = tuple(
            sorted(
                (entity, selectors)
                for entity, selectors in entity_candidates.items()
                if all(_selector_has_collision(model, selector) for selector in selectors)
            )
        )
        missing_required = [
            entity
            for entity in (
                "qc_master",
                "rack",
                *_EXPECTED_EES,
                *(f"rack_support:{ee}" for ee in _EXPECTED_EES),
            )
            if entity not in dict(entities)
        ]
        if missing_required:
            raise ToolUseJournalCompatibilityError(
                f"compiled model {version!r} lost collision entities "
                f"{sorted(missing_required)}"
            )
        return CompiledToolUseJournalCollisionModel(
            model=model,
            collision_model_version=version,
            active_ee=active_ee,
            joint_names=self.joint_names,
            robot_root_body_name=self.robot_root_body_name,
            baseline_qpos=tuple(float(value) for value in baseline),
            entity_selectors=entities,
            promoted_rack_geom_names=promoted,
            mjcf_sha256=hashlib.sha256(compiled_xml.encode("utf-8")).hexdigest(),
        )

    def build_collision_registry(
        self,
        collision_contexts: Mapping[str, CollisionContext],
        *,
        collision_margin_m: float,
        allowed_collision_pairs: Iterable[tuple[str, str]] = (),
        default_active_ee: str | None | object = _REFERENCE_ACTIVE_EE,
        include_all_models: bool = False,
    ) -> MuJoCoCollisionModelRegistry:
        default_ee = (
            self.reference_active_ee
            if default_active_ee is _REFERENCE_ACTIVE_EE
            else default_active_ee
        )
        if default_ee is not None and not isinstance(default_ee, str):
            raise ToolUseJournalCompatibilityError(
                "default_active_ee must be an EE id or None"
            )
        required: set[str | None] = {default_ee}
        for context in collision_contexts.values():
            if context.collision_model_version != self.model_version_for(
                context.active_ee
            ):
                raise ToolUseJournalCompatibilityError(
                    f"context {context.context_id!r} uses collision model "
                    f"{context.collision_model_version!r}, expected "
                    f"{self.model_version_for(context.active_ee)!r}"
                )
            required.add(context.active_ee)
        if include_all_models:
            required = {None, *_EXPECTED_EES}
        compiled = [self.compile(active_ee) for active_ee in required]
        validators = {
            item.collision_model_version: item.make_validator(
                collision_margin_m=collision_margin_m,
                collision_contexts=collision_contexts,
                allowed_collision_pairs=allowed_collision_pairs,
            )
            for item in compiled
        }
        try:
            return MuJoCoCollisionModelRegistry(
                validators,
                collision_contexts=collision_contexts,
                default_model_version=self.model_version_for(default_ee),
            )
        except MuJoCoCollisionConfigurationError as error:
            raise ToolUseJournalCompatibilityError(str(error)) from error
