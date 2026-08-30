"""Scene-backed MuJoCo collision validation for IK branches and joint paths.

The validator owns an ``MjData`` instance but may reuse an environment's
compiled ``MjModel``.  It therefore evaluates candidate states without
changing the live simulator state.  MuJoCo's own contact filtering remains the
authority for same-body, welded-body, parent-child, contact-bit, and explicit
exclude rules.

Safety clearance is implemented by temporarily expanding the contact margin of
moving collision geoms.  This makes MuJoCo report positive-distance proximity
contacts as well as penetrations, while retaining the model's collision rules.
"""

from __future__ import annotations

import fnmatch
import math
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np

from tuj.m4_motion.schema import CollisionContext, RelativeKeyframeSpec, TrajectoryWaypoint
from tuj.m4_motion.strategy import EdgePlanResult, JointConfig, wrapped_joint_delta


class MuJoCoCollisionConfigurationError(ValueError):
    """Raised when the compiled scene cannot satisfy the validator contract."""


@dataclass(frozen=True, slots=True)
class CollisionContact:
    """One relevant moving-vs-scene or self-collision proximity pair."""

    geom_a: str
    geom_b: str
    body_a: str
    body_b: str
    distance_m: float
    allowed: bool


@dataclass(frozen=True, slots=True)
class CollisionCheckResult:
    valid: bool
    failure_code: str | None = None
    detail: str = ""
    min_clearance_m: float | None = None
    clearance_is_lower_bound: bool = False
    contacts: tuple[CollisionContact, ...] = ()


@dataclass(frozen=True, slots=True)
class PathCollisionCheckResult:
    valid: bool
    checked_states: int
    failure_code: str | None = None
    detail: str = ""
    min_clearance_m: float | None = None
    failed_state_index: int | None = None


GeomSelector = str | int


class SceneCollisionValidator(Protocol):
    def check(
        self,
        joint_config: Sequence[float],
        keyframe: RelativeKeyframeSpec | None = None,
        *,
        context: CollisionContext | None = None,
        context_id: str | None = None,
    ) -> CollisionCheckResult: ...


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, object_id: int) -> str:
    return mujoco.mj_id2name(model, kind, object_id) or f"{kind.name.lower()}:{object_id}"


def _descendant_body_ids(model: mujoco.MjModel, root_body_id: int) -> frozenset[int]:
    descendants = {root_body_id}
    for body_id in range(root_body_id + 1, model.nbody):
        parent = int(model.body_parentid[body_id])
        while parent and parent not in descendants:
            parent = int(model.body_parentid[parent])
        if parent in descendants:
            descendants.add(body_id)
    return frozenset(descendants)


def _collision_enabled(model: mujoco.MjModel, geom_id: int) -> bool:
    return bool(int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id]))


class MuJoCoCollisionValidator:
    """Evaluate joint states against one compiled MuJoCo scene.

    ``entity_geoms`` binds logical schema names such as ``obj1`` or ``vacuum``
    to geom ids, geom names, or body names.  A body selector expands to its
    complete subtree.  These bindings drive attached-object / active-EE moving
    geometry and ACM matching.

    The compiled model is briefly mutated while contact margins are expanded,
    then restored before the call returns.  Calls through this instance are
    serialized.  A model shared with independently running simulator threads
    should instead be compiled separately for planning.
    """

    _DISTANCE_TOLERANCE_M = 1e-9

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        joint_names: Sequence[str],
        robot_root_body_name: str,
        baseline_qpos: Sequence[float] | None = None,
        collision_margin_m: float = 0.005,
        collision_model_version: str = "default",
        collision_contexts: Mapping[str, CollisionContext] | None = None,
        entity_geoms: Mapping[str, Iterable[GeomSelector]] | None = None,
        allowed_collision_pairs: Iterable[tuple[str, str]] = (),
    ) -> None:
        if collision_margin_m < 0 or not math.isfinite(collision_margin_m):
            raise MuJoCoCollisionConfigurationError(
                "collision_margin_m must be finite and non-negative"
            )
        if not joint_names:
            raise MuJoCoCollisionConfigurationError("joint_names must not be empty")
        if not collision_model_version:
            raise MuJoCoCollisionConfigurationError(
                "collision_model_version must not be empty"
            )

        self.model = model
        self.data = mujoco.MjData(model)
        self.collision_margin_m = float(collision_margin_m)
        self.collision_model_version = collision_model_version
        self._lock = threading.RLock()
        self._joint_names = tuple(joint_names)
        self._joint_ids = tuple(self._resolve_joint_id(name) for name in self._joint_names)
        self._qpos_addresses = tuple(
            self._scalar_joint_qpos_address(joint_id) for joint_id in self._joint_ids
        )

        root_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, robot_root_body_name
        )
        if root_id < 0:
            raise MuJoCoCollisionConfigurationError(
                f"unknown robot root body {robot_root_body_name!r}"
            )
        robot_bodies = _descendant_body_ids(model, root_id)
        robot_weld_ids = {int(model.body_weldid[body_id]) for body_id in robot_bodies}
        self._adjacent_robot_weld_pairs: frozenset[tuple[int, int]] = frozenset(
            tuple(
                sorted(
                    (
                        weld_id,
                        int(
                            model.body_weldid[
                                int(model.body_parentid[weld_id])
                            ]
                        ),
                    )
                )
            )
            for weld_id in robot_weld_ids
            if weld_id != 0
            and int(model.body_parentid[weld_id]) >= 0
            and int(model.body_weldid[int(model.body_parentid[weld_id])]) != weld_id
        )
        self._robot_geom_ids = frozenset(
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in robot_bodies
            and _collision_enabled(model, geom_id)
        )
        if not self._robot_geom_ids:
            raise MuJoCoCollisionConfigurationError(
                f"robot subtree {robot_root_body_name!r} has no collision-enabled geoms"
            )

        if baseline_qpos is None:
            self._baseline_qpos = np.zeros(model.nq, dtype=float)
        else:
            values = np.asarray(tuple(baseline_qpos), dtype=float)
            if values.shape != (model.nq,) or not np.all(np.isfinite(values)):
                raise MuJoCoCollisionConfigurationError(
                    f"baseline_qpos must contain exactly {model.nq} finite values"
                )
            self._baseline_qpos = values.copy()

        self._collision_contexts = dict(collision_contexts or {})
        self._base_allowed_pairs = tuple(
            self._canonical_pair(pair) for pair in allowed_collision_pairs
        )
        self._entity_geom_ids: dict[str, frozenset[int]] = {
            "robot": self._robot_geom_ids,
            robot_root_body_name: self._robot_geom_ids,
        }
        for entity, selectors in (entity_geoms or {}).items():
            if not entity:
                raise MuJoCoCollisionConfigurationError("entity names must not be empty")
            geom_ids: set[int] = set()
            for selector in selectors:
                geom_ids.update(self._resolve_geom_selector(selector))
            if not geom_ids:
                raise MuJoCoCollisionConfigurationError(
                    f"entity {entity!r} resolves to no collision-enabled geoms"
                )
            self._entity_geom_ids[entity] = frozenset(geom_ids)

        reverse: dict[int, set[str]] = {}
        for entity, geom_ids in self._entity_geom_ids.items():
            for geom_id in geom_ids:
                reverse.setdefault(geom_id, set()).add(entity)
        self._geom_entities = {
            geom_id: frozenset(entities) for geom_id, entities in reverse.items()
        }

    @classmethod
    def from_robosuite_env(
        cls,
        env: object,
        *,
        collision_margin_m: float = 0.005,
        collision_model_version: str = "default",
        collision_contexts: Mapping[str, CollisionContext] | None = None,
        entity_body_names: Mapping[str, str] | None = None,
        allowed_collision_pairs: Iterable[tuple[str, str]] = (),
    ) -> "MuJoCoCollisionValidator":
        """Create a planner-side validator from a reset robosuite environment."""

        try:
            model = env.sim.model._model  # type: ignore[attr-defined]
            baseline_qpos = np.asarray(env.sim.data._data.qpos, dtype=float)  # type: ignore[attr-defined]
            robot = env.robots[0]  # type: ignore[attr-defined]
            joint_names = tuple(robot.robot_model.joints)
            root_body = str(robot.robot_model.root_body)
        except (AttributeError, IndexError, TypeError) as error:
            raise MuJoCoCollisionConfigurationError(
                "env must be a reset robosuite environment with one robot"
            ) from error
        entity_geoms = {
            entity: (body_name,)
            for entity, body_name in (entity_body_names or {}).items()
        }
        return cls(
            model,
            joint_names=joint_names,
            robot_root_body_name=root_body,
            baseline_qpos=baseline_qpos,
            collision_margin_m=collision_margin_m,
            collision_model_version=collision_model_version,
            collision_contexts=collision_contexts,
            entity_geoms=entity_geoms,
            allowed_collision_pairs=allowed_collision_pairs,
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    @property
    def robot_geom_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                for geom_id in self._robot_geom_ids
            )
        )

    def _resolve_joint_id(self, requested_name: str) -> int:
        exact = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, requested_name
        )
        if exact >= 0:
            return exact
        suffix_matches = [
            joint_id
            for joint_id in range(self.model.njnt)
            if _name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id).endswith(
                requested_name
            )
        ]
        if len(suffix_matches) != 1:
            raise MuJoCoCollisionConfigurationError(
                f"joint {requested_name!r} has {len(suffix_matches)} scene matches"
            )
        return suffix_matches[0]

    def _scalar_joint_qpos_address(self, joint_id: int) -> int:
        joint_type = int(self.model.jnt_type[joint_id])
        if joint_type not in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }:
            joint_name = _name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            raise MuJoCoCollisionConfigurationError(
                f"planning joint {joint_name!r} must be hinge or slide"
            )
        return int(self.model.jnt_qposadr[joint_id])

    def _resolve_geom_selector(self, selector: GeomSelector) -> frozenset[int]:
        if isinstance(selector, int):
            if selector < 0 or selector >= self.model.ngeom:
                raise MuJoCoCollisionConfigurationError(
                    f"geom id {selector} is outside the compiled model"
                )
            return (
                frozenset({selector})
                if _collision_enabled(self.model, selector)
                else frozenset()
            )
        geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, selector
        )
        if geom_id >= 0:
            return (
                frozenset({geom_id})
                if _collision_enabled(self.model, geom_id)
                else frozenset()
            )
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, selector
        )
        if body_id < 0:
            raise MuJoCoCollisionConfigurationError(
                f"unknown geom or body selector {selector!r}"
            )
        bodies = _descendant_body_ids(self.model, body_id)
        return frozenset(
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in bodies
            and _collision_enabled(self.model, geom_id)
        )

    @staticmethod
    def _canonical_pair(pair: tuple[str, str]) -> tuple[str, str]:
        if len(pair) != 2 or not pair[0] or not pair[1]:
            raise MuJoCoCollisionConfigurationError(
                "allowed collision pairs require two non-empty selectors"
            )
        return tuple(sorted((str(pair[0]), str(pair[1]))))

    def _resolve_context(
        self,
        keyframe: RelativeKeyframeSpec | None,
        context: CollisionContext | None,
        context_id: str | None,
    ) -> tuple[CollisionContext | None, CollisionCheckResult | None]:
        if context is not None and context_id is not None:
            raise ValueError("provide context or context_id, not both")
        selected_id = context_id
        if selected_id is None and keyframe is not None:
            selected_id = keyframe.collision_context_id
        if context is not None:
            selected = context
        elif selected_id is None:
            return None, None
        else:
            selected = self._collision_contexts.get(selected_id)
            if selected is None:
                return None, CollisionCheckResult(
                    valid=False,
                    failure_code="COLLISION_CONTEXT_MISSING",
                    detail=f"collision context {selected_id!r} is not registered",
                )
        if selected.collision_model_version != self.collision_model_version:
            return None, CollisionCheckResult(
                valid=False,
                failure_code="COLLISION_MODEL_MISMATCH",
                detail=(
                    f"context {selected.context_id!r} requires collision model "
                    f"{selected.collision_model_version!r}, validator has "
                    f"{self.collision_model_version!r}"
                ),
            )
        return selected, None

    def _moving_geoms(
        self, context: CollisionContext | None
    ) -> tuple[frozenset[int], CollisionCheckResult | None]:
        moving = set(self._robot_geom_ids)
        if context is None:
            return frozenset(moving), None
        referenced_entities = list(context.attached_object_ids)
        if context.active_ee is not None:
            referenced_entities.append(context.active_ee)
        missing = [
            entity for entity in referenced_entities if entity not in self._entity_geom_ids
        ]
        if missing:
            return frozenset(), CollisionCheckResult(
                valid=False,
                failure_code="SCENE_ENTITY_UNMAPPED",
                detail=f"collision entities have no geom binding: {sorted(set(missing))}",
            )
        kinematic_attachment_ids = {
            item.object_id for item in context.attached_object_transforms
        }
        for entity in referenced_entities:
            entity_ids = self._entity_geom_ids[entity]
            if (
                not entity_ids.issubset(self._robot_geom_ids)
                and entity not in kinematic_attachment_ids
            ):
                return frozenset(), CollisionCheckResult(
                    valid=False,
                    failure_code="ATTACHED_GEOMETRY_NOT_KINEMATIC",
                    detail=(
                        f"entity {entity!r} is marked attached/active but its geoms "
                        "are not in the compiled robot subtree"
                    ),
                )
            moving.update(entity_ids)
        return frozenset(moving), None

    def _apply_context_state(
        self, context: CollisionContext | None
    ) -> CollisionCheckResult | None:
        if context is None:
            return None
        for free_object in context.free_object_poses:
            if free_object.object_id not in self._entity_geom_ids:
                return CollisionCheckResult(
                    valid=False,
                    failure_code="SCENE_ENTITY_UNMAPPED",
                    detail=(
                        f"free object {free_object.object_id!r} has no geom binding"
                    ),
                )
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                free_object.free_joint_name,
            )
            if joint_id < 0 or int(self.model.jnt_type[joint_id]) != int(
                mujoco.mjtJoint.mjJNT_FREE
            ):
                return CollisionCheckResult(
                    valid=False,
                    failure_code="FREE_OBJECT_JOINT_UNAVAILABLE",
                    detail=(
                        f"free object {free_object.object_id!r} requires free "
                        f"joint {free_object.free_joint_name!r}"
                    ),
                )
            address = int(self.model.jnt_qposadr[joint_id])
            pose = free_object.pose
            self.data.qpos[address : address + 3] = pose.position_m
            x, y, z, w = pose.orientation_xyzw
            self.data.qpos[address + 3 : address + 7] = (w, x, y, z)
        for joint_name, value in context.kinematic_joint_positions.items():
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                return CollisionCheckResult(
                    valid=False,
                    failure_code="KINEMATIC_JOINT_UNAVAILABLE",
                    detail=f"collision snapshot joint {joint_name!r} is absent",
                )
            if joint_id in self._joint_ids:
                return CollisionCheckResult(
                    valid=False,
                    failure_code="KINEMATIC_JOINT_CONFLICT",
                    detail=(
                        f"collision snapshot cannot override arm joint "
                        f"{joint_name!r}"
                    ),
                )
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type not in {
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            }:
                return CollisionCheckResult(
                    valid=False,
                    failure_code="KINEMATIC_JOINT_UNSUPPORTED",
                    detail=f"collision snapshot joint {joint_name!r} is not scalar",
                )
            self.data.qpos[int(self.model.jnt_qposadr[joint_id])] = float(value)

        if not context.attached_object_transforms:
            return None
        mujoco.mj_forward(self.model, self.data)
        for attachment in context.attached_object_transforms:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                attachment.free_joint_name,
            )
            if joint_id < 0 or int(self.model.jnt_type[joint_id]) != int(
                mujoco.mjtJoint.mjJNT_FREE
            ):
                return CollisionCheckResult(
                    valid=False,
                    failure_code="ATTACHED_OBJECT_FREE_JOINT_UNAVAILABLE",
                    detail=(
                        f"attached object {attachment.object_id!r} requires free "
                        f"joint {attachment.free_joint_name!r}"
                    ),
                )
            reference_kind = (
                mujoco.mjtObj.mjOBJ_BODY
                if attachment.reference_kind == "body"
                else mujoco.mjtObj.mjOBJ_SITE
            )
            reference_id = mujoco.mj_name2id(
                self.model, reference_kind, attachment.reference_name
            )
            if reference_id < 0:
                return CollisionCheckResult(
                    valid=False,
                    failure_code="ATTACHMENT_REFERENCE_UNAVAILABLE",
                    detail=(
                        f"{attachment.reference_kind} "
                        f"{attachment.reference_name!r} is absent"
                    ),
                )
            if attachment.reference_kind == "body":
                reference_position = np.asarray(
                    self.data.xpos[reference_id], dtype=float
                )
                reference_rotation = np.asarray(
                    self.data.xmat[reference_id], dtype=float
                ).reshape(3, 3)
            else:
                reference_position = np.asarray(
                    self.data.site_xpos[reference_id], dtype=float
                )
                reference_rotation = np.asarray(
                    self.data.site_xmat[reference_id], dtype=float
                ).reshape(3, 3)
            x, y, z, w = attachment.orientation_in_reference_xyzw
            relative_rotation_flat = np.empty(9, dtype=float)
            mujoco.mju_quat2Mat(
                relative_rotation_flat,
                np.asarray((w, x, y, z), dtype=float),
            )
            relative_rotation = relative_rotation_flat.reshape(3, 3)
            object_position = reference_position + reference_rotation @ np.asarray(
                attachment.position_in_reference_m, dtype=float
            )
            object_rotation = reference_rotation @ relative_rotation
            object_quaternion_wxyz = np.empty(4, dtype=float)
            mujoco.mju_mat2Quat(
                object_quaternion_wxyz,
                np.ascontiguousarray(object_rotation.reshape(9)),
            )
            address = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[address : address + 3] = object_position
            self.data.qpos[address + 3 : address + 7] = object_quaternion_wxyz
        return None

    def _labels(self, geom_id: int) -> frozenset[str]:
        body_id = int(self.model.geom_bodyid[geom_id])
        return frozenset(
            {
                _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                _name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                *self._geom_entities.get(geom_id, ()),
            }
        )

    def _is_adjacent_robot_pair(self, geom_a: int, geom_b: int) -> bool:
        if geom_a not in self._robot_geom_ids or geom_b not in self._robot_geom_ids:
            return False
        weld_a = int(self.model.body_weldid[int(self.model.geom_bodyid[geom_a])])
        weld_b = int(self.model.body_weldid[int(self.model.geom_bodyid[geom_b])])
        return tuple(sorted((weld_a, weld_b))) in self._adjacent_robot_weld_pairs

    @staticmethod
    def _selector_matches(selector: str, labels: frozenset[str]) -> bool:
        return any(fnmatch.fnmatchcase(label, selector) for label in labels)

    def _is_allowed(
        self,
        geom_a: int,
        geom_b: int,
        context: CollisionContext | None,
    ) -> bool:
        pairs = list(self._base_allowed_pairs)
        if context is not None:
            pairs.extend(self._canonical_pair(pair) for pair in context.allowed_collision_pairs)
            pairs.extend(
                self._canonical_pair((touch_link, attached))
                for touch_link in context.touch_links
                for attached in context.attached_object_ids
            )
        labels_a = self._labels(geom_a)
        labels_b = self._labels(geom_b)
        return any(
            (
                self._selector_matches(left, labels_a)
                and self._selector_matches(right, labels_b)
            )
            or (
                self._selector_matches(left, labels_b)
                and self._selector_matches(right, labels_a)
            )
            for left, right in pairs
        )

    def _joint_limit_failure(
        self, joint_config: Sequence[float]
    ) -> CollisionCheckResult | None:
        if len(joint_config) != len(self._joint_ids):
            return CollisionCheckResult(
                valid=False,
                failure_code="JOINT_DIMENSION_MISMATCH",
                detail=(
                    f"expected {len(self._joint_ids)} joints, got {len(joint_config)}"
                ),
            )
        if not all(math.isfinite(float(value)) for value in joint_config):
            return CollisionCheckResult(
                valid=False,
                failure_code="NONFINITE_JOINT_STATE",
                detail="joint configuration contains NaN or infinity",
            )
        for name, joint_id, value in zip(
            self._joint_names, self._joint_ids, joint_config
        ):
            if not bool(self.model.jnt_limited[joint_id]):
                continue
            lower, upper = (float(v) for v in self.model.jnt_range[joint_id])
            if float(value) < lower - 1e-9 or float(value) > upper + 1e-9:
                return CollisionCheckResult(
                    valid=False,
                    failure_code="JOINT_LIMIT_VIOLATION",
                    detail=(
                        f"joint {name!r}={float(value):.6f} outside "
                        f"[{lower:.6f}, {upper:.6f}]"
                    ),
                )
        return None

    def check(
        self,
        joint_config: Sequence[float],
        keyframe: RelativeKeyframeSpec | None = None,
        *,
        context: CollisionContext | None = None,
        context_id: str | None = None,
    ) -> CollisionCheckResult:
        """Check one arm state, failing closed on incomplete scene bindings."""

        limit_failure = self._joint_limit_failure(joint_config)
        if limit_failure is not None:
            return limit_failure
        selected_context, context_failure = self._resolve_context(
            keyframe, context, context_id
        )
        if context_failure is not None:
            return context_failure
        moving_geoms, entity_failure = self._moving_geoms(selected_context)
        if entity_failure is not None:
            return entity_failure

        with self._lock:
            self.data.qpos[:] = self._baseline_qpos
            self.data.qvel[:] = 0.0
            for address, value in zip(self._qpos_addresses, joint_config):
                self.data.qpos[address] = float(value)
            context_state_failure = self._apply_context_state(selected_context)
            if context_state_failure is not None:
                return context_state_failure

            moving_geom_indices = sorted(moving_geoms)
            original_margins = self.model.geom_margin[moving_geom_indices].copy()
            try:
                for geom_id in moving_geoms:
                    self.model.geom_margin[geom_id] = max(
                        float(self.model.geom_margin[geom_id]),
                        self.collision_margin_m,
                    )
                mujoco.mj_forward(self.model, self.data)
            finally:
                self.model.geom_margin[moving_geom_indices] = original_margins

            contacts: list[CollisionContact] = []
            disallowed_distances: list[float] = []
            violations: list[CollisionContact] = []
            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]
                geom_a = int(contact.geom1)
                geom_b = int(contact.geom2)
                if geom_a not in moving_geoms and geom_b not in moving_geoms:
                    continue
                # Adjacent robot links intentionally meet at their joint and
                # are conventionally disabled in a robot collision matrix.
                # Expanding geom margins can surface their nominal zero-gap
                # interface, so exclude that pair from safety clearance too.
                if self._is_adjacent_robot_pair(geom_a, geom_b):
                    continue
                allowed = self._is_allowed(
                    geom_a, geom_b, selected_context
                )
                body_a_id = int(self.model.geom_bodyid[geom_a])
                body_b_id = int(self.model.geom_bodyid[geom_b])
                record = CollisionContact(
                    geom_a=_name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_a
                    ),
                    geom_b=_name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_b
                    ),
                    body_a=_name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, body_a_id
                    ),
                    body_b=_name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, body_b_id
                    ),
                    distance_m=float(contact.dist),
                    allowed=allowed,
                )
                contacts.append(record)
                if allowed:
                    continue
                disallowed_distances.append(record.distance_m)
                if (
                    record.distance_m
                    < self.collision_margin_m - self._DISTANCE_TOLERANCE_M
                ):
                    violations.append(record)

            if disallowed_distances:
                min_clearance = min(disallowed_distances)
                lower_bound = False
            else:
                min_clearance = self.collision_margin_m
                lower_bound = True
            if violations:
                first = min(violations, key=lambda item: item.distance_m)
                return CollisionCheckResult(
                    valid=False,
                    failure_code="COLLISION_MARGIN_VIOLATION",
                    detail=(
                        f"{first.geom_a} <-> {first.geom_b} clearance "
                        f"{first.distance_m:.6f} m is below required "
                        f"{self.collision_margin_m:.6f} m"
                    ),
                    min_clearance_m=min_clearance,
                    clearance_is_lower_bound=False,
                    contacts=tuple(contacts),
                )
            return CollisionCheckResult(
                valid=True,
                min_clearance_m=min_clearance,
                clearance_is_lower_bound=lower_bound,
                contacts=tuple(contacts),
            )

    def __call__(
        self, joint_config: JointConfig, keyframe: RelativeKeyframeSpec
    ) -> bool:
        """Implement the compiler's ``StateValidator`` callable contract."""

        return self.check(joint_config, keyframe).valid

    def check_waypoints(
        self,
        waypoints: Sequence[TrajectoryWaypoint],
        context: CollisionContext,
    ) -> PathCollisionCheckResult:
        """Final collision check for time-parameterized segment waypoints."""

        minimum: float | None = None
        for index, waypoint in enumerate(waypoints):
            result = self.check(waypoint.joint_positions_rad, context=context)
            if result.min_clearance_m is not None:
                minimum = (
                    result.min_clearance_m
                    if minimum is None
                    else min(minimum, result.min_clearance_m)
                )
            if not result.valid:
                return PathCollisionCheckResult(
                    valid=False,
                    checked_states=index + 1,
                    failure_code=result.failure_code,
                    detail=f"waypoint {index}: {result.detail}",
                    min_clearance_m=minimum,
                    failed_state_index=index,
                )
        return PathCollisionCheckResult(
            valid=True,
            checked_states=len(waypoints),
            min_clearance_m=minimum,
        )

    def final_segment_validator(
        self,
        waypoints: tuple[TrajectoryWaypoint, ...],
        context: CollisionContext,
    ) -> bool:
        """Implement ``MotionPlanBuilder``'s final-validator seam."""

        return self.check_waypoints(waypoints, context).valid


class MuJoCoCollisionModelRegistry:
    """Route each collision context to its matching compiled scene model.

    EE lock/unlock and object attach/detach change kinematic geometry.  Those
    states must not be emulated by merely changing an allowed-collision list;
    callers register a separately compiled validator for every
    ``collision_model_version`` used by the strategy.
    """

    def __init__(
        self,
        validators: Mapping[str, MuJoCoCollisionValidator],
        *,
        collision_contexts: Mapping[str, CollisionContext],
        default_model_version: str,
    ) -> None:
        if default_model_version not in validators:
            raise MuJoCoCollisionConfigurationError(
                f"default collision model {default_model_version!r} is not registered"
            )
        if not validators:
            raise MuJoCoCollisionConfigurationError(
                "at least one collision model validator is required"
            )
        joint_orders = {validator.joint_names for validator in validators.values()}
        if len(joint_orders) != 1:
            raise MuJoCoCollisionConfigurationError(
                "all collision model validators must use the same joint order"
            )
        mismatched = [
            version
            for version, validator in validators.items()
            if validator.collision_model_version != version
        ]
        if mismatched:
            raise MuJoCoCollisionConfigurationError(
                f"validator registry keys do not match model versions: {mismatched}"
            )
        self._validators = dict(validators)
        self._contexts = dict(collision_contexts)
        self._default_model_version = default_model_version

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._validators[self._default_model_version].joint_names

    def _select_context(
        self,
        keyframe: RelativeKeyframeSpec | None,
        context: CollisionContext | None,
        context_id: str | None,
    ) -> tuple[CollisionContext | None, CollisionCheckResult | None]:
        if context is not None and context_id is not None:
            raise ValueError("provide context or context_id, not both")
        selected_id = context_id
        if selected_id is None and keyframe is not None:
            selected_id = keyframe.collision_context_id
        if context is not None:
            return context, None
        if selected_id is None:
            return None, None
        selected = self._contexts.get(selected_id)
        if selected is None:
            return None, CollisionCheckResult(
                valid=False,
                failure_code="COLLISION_CONTEXT_MISSING",
                detail=f"collision context {selected_id!r} is not registered",
            )
        return selected, None

    def check(
        self,
        joint_config: Sequence[float],
        keyframe: RelativeKeyframeSpec | None = None,
        *,
        context: CollisionContext | None = None,
        context_id: str | None = None,
    ) -> CollisionCheckResult:
        selected, failure = self._select_context(keyframe, context, context_id)
        if failure is not None:
            return failure
        version = (
            selected.collision_model_version
            if selected is not None
            else self._default_model_version
        )
        validator = self._validators.get(version)
        if validator is None:
            return CollisionCheckResult(
                valid=False,
                failure_code="COLLISION_MODEL_UNAVAILABLE",
                detail=f"compiled collision model {version!r} is not registered",
            )
        return validator.check(joint_config, context=selected)

    def __call__(
        self, joint_config: JointConfig, keyframe: RelativeKeyframeSpec
    ) -> bool:
        return self.check(joint_config, keyframe).valid

    def check_waypoints(
        self,
        waypoints: Sequence[TrajectoryWaypoint],
        context: CollisionContext,
    ) -> PathCollisionCheckResult:
        minimum: float | None = None
        for index, waypoint in enumerate(waypoints):
            result = self.check(waypoint.joint_positions_rad, context=context)
            if result.min_clearance_m is not None:
                minimum = (
                    result.min_clearance_m
                    if minimum is None
                    else min(minimum, result.min_clearance_m)
                )
            if not result.valid:
                return PathCollisionCheckResult(
                    valid=False,
                    checked_states=index + 1,
                    failure_code=result.failure_code,
                    detail=f"waypoint {index}: {result.detail}",
                    min_clearance_m=minimum,
                    failed_state_index=index,
                )
        return PathCollisionCheckResult(
            valid=True,
            checked_states=len(waypoints),
            min_clearance_m=minimum,
        )

    def final_segment_validator(
        self,
        waypoints: tuple[TrajectoryWaypoint, ...],
        context: CollisionContext,
    ) -> bool:
        return self.check_waypoints(waypoints, context).valid


@dataclass(slots=True)
class MuJoCoInterpolatingEdgePlanner:
    """Joint interpolation whose every sample is checked in the full scene."""

    validator: SceneCollisionValidator
    max_joint_step_rad: float = 0.05

    def __post_init__(self) -> None:
        if self.max_joint_step_rad <= 0:
            raise ValueError("max_joint_step_rad must be positive")

    def plan(
        self,
        source: JointConfig,
        target: JointConfig,
        source_keyframe: RelativeKeyframeSpec | None,
        target_keyframe: RelativeKeyframeSpec,
    ) -> EdgePlanResult:
        del source_keyframe  # Target keyframe owns the incoming segment context.
        delta = wrapped_joint_delta(source, target)
        steps = max(
            1,
            int(
                math.ceil(
                    float(np.max(np.abs(delta))) / self.max_joint_step_rad
                )
            ),
        )
        source_values = np.asarray(source, dtype=float)
        path: list[JointConfig] = []
        minimum: float | None = None
        for index in range(steps + 1):
            state = source_values + delta * (index / steps)
            config = tuple(float(value) for value in state)
            report = self.validator.check(config, target_keyframe)
            if report.min_clearance_m is not None:
                minimum = (
                    report.min_clearance_m
                    if minimum is None
                    else min(minimum, report.min_clearance_m)
                )
            if not report.valid:
                return EdgePlanResult(
                    valid=False,
                    failure_code=report.failure_code or "COLLISION_STATE_INVALID",
                    detail=f"invalid sample {index}/{steps}: {report.detail}",
                    min_clearance_m=minimum,
                )
            path.append(config)
        return EdgePlanResult(
            valid=True,
            joint_path=tuple(path),
            min_clearance_m=minimum,
        )
