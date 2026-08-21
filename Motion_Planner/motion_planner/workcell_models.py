"""Compile bare-flange and hard-attached EE workcell collision models.

Rendered images are deliberately not consumed here.  They are visual evidence
for tool direction and contact posture; body names, coupling offsets, and
collision dimensions come from the compiled robosuite MJCF scene.
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from motion_planner.mujoco_collision import (
    MuJoCoCollisionConfigurationError,
    MuJoCoCollisionModelRegistry,
    MuJoCoCollisionValidator,
)
from motion_planner.schema import CollisionContext


class WorkcellModelCompilationError(ValueError):
    """Raised when MJCF structure cannot produce the requested scene state."""


def _mj_name(model: mujoco.MjModel, kind: mujoco.mjtObj, object_id: int) -> str:
    return mujoco.mj_id2name(model, kind, object_id) or ""


def _parse_vector(raw: str | None, *, length: int, label: str) -> np.ndarray:
    if raw is None:
        raise WorkcellModelCompilationError(f"{label} is missing")
    values = np.fromstring(raw, sep=" ", dtype=float)
    if values.shape != (length,) or not np.all(np.isfinite(values)):
        raise WorkcellModelCompilationError(
            f"{label} must contain {length} finite values"
        )
    return values


def _format_vector(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _quat_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _body_elements(root: ET.Element) -> dict[str, ET.Element]:
    result: dict[str, ET.Element] = {}
    for body in root.findall(".//body"):
        name = body.get("name")
        if not name:
            continue
        if name in result:
            raise WorkcellModelCompilationError(f"duplicate body name {name!r}")
        result[name] = body
    return result


def _is_descendant(model: mujoco.MjModel, body_id: int, root_id: int) -> bool:
    current = body_id
    for _ in range(model.nbody + 1):
        if current == root_id:
            return True
        if current == 0:
            return False
        current = int(model.body_parentid[current])
    return False


@dataclass(frozen=True, slots=True)
class CompiledWorkcellCollisionModel:
    """One immutable physical-scene identity backed by a compiled MjModel."""

    model: mujoco.MjModel
    collision_model_version: str
    active_ee: str | None
    joint_names: tuple[str, ...]
    robot_root_body_name: str
    baseline_qpos: tuple[float, ...]
    entity_body_names: tuple[tuple[str, str], ...]
    mjcf_sha256: str

    def make_validator(
        self,
        *,
        collision_margin_m: float,
        collision_contexts: Mapping[str, CollisionContext] | None = None,
        allowed_collision_pairs: Iterable[tuple[str, str]] = (),
    ) -> MuJoCoCollisionValidator:
        return MuJoCoCollisionValidator(
            self.model,
            joint_names=self.joint_names,
            robot_root_body_name=self.robot_root_body_name,
            baseline_qpos=self.baseline_qpos,
            collision_margin_m=collision_margin_m,
            collision_model_version=self.collision_model_version,
            collision_contexts=collision_contexts,
            entity_geoms={
                entity: (body_name,)
                for entity, body_name in self.entity_body_names
            },
            allowed_collision_pairs=allowed_collision_pairs,
        )


class EEWorkcellCollisionModelCompiler:
    """Reparent a rack EE under the QC master to create hard-attached models.

    The rack demo rigidizes detached EE models before producing the source
    MJCF.  Reparenting therefore produces a six-DOF arm collision model without
    adding uncontrolled gripper joints.  When articulated joints are present,
    ``CollisionContext.kinematic_joint_positions`` fixes OPEN/CLOSED/HOLDING
    collision snapshots without changing the arm planning DOF.
    """

    DEFAULT_BARE_VERSION = "ur5e-qc-bare-flange-v1"

    def __init__(
        self,
        source_mjcf: str,
        *,
        joint_names: Sequence[str],
        robot_root_body_name: str,
        qc_master_body_name: str,
        qc_coupling_body_name: str,
        ee_root_body_names: Mapping[str, str],
        rack_support_body_names: Mapping[str, str],
        baseline_joint_positions: Mapping[str, float],
        ee_coupling_offsets_m: Mapping[str, float] | None = None,
        ee_attach_quaternions_wxyz: Mapping[
            str, Sequence[float]
        ] | None = None,
        bare_model_version: str = DEFAULT_BARE_VERSION,
        attached_model_versions: Mapping[str, str] | None = None,
    ) -> None:
        if not source_mjcf.strip():
            raise WorkcellModelCompilationError("source_mjcf must not be empty")
        if not joint_names:
            raise WorkcellModelCompilationError("joint_names must not be empty")
        if set(joint_names) != set(baseline_joint_positions):
            raise WorkcellModelCompilationError(
                "baseline_joint_positions must exactly match joint_names"
            )
        if not ee_root_body_names:
            raise WorkcellModelCompilationError("at least one EE is required")
        if set(ee_root_body_names) != set(rack_support_body_names):
            raise WorkcellModelCompilationError(
                "every EE requires one rack support body"
            )
        if not bare_model_version:
            raise WorkcellModelCompilationError(
                "bare_model_version must not be empty"
            )

        self._source_mjcf = source_mjcf
        self._joint_names = tuple(joint_names)
        self._robot_root = robot_root_body_name
        self._qc_master = qc_master_body_name
        self._qc_coupling = qc_coupling_body_name
        self._ee_roots = dict(ee_root_body_names)
        self._rack_supports = dict(rack_support_body_names)
        self._baseline_joint_positions = {
            name: float(value) for name, value in baseline_joint_positions.items()
        }
        if not all(
            math.isfinite(value) for value in self._baseline_joint_positions.values()
        ):
            raise WorkcellModelCompilationError(
                "baseline joint positions must be finite"
            )
        self._coupling_offsets = {
            ee: float((ee_coupling_offsets_m or {}).get(ee, 0.0))
            for ee in self._ee_roots
        }
        self._attach_quaternions = {
            ee: tuple(
                float(value)
                for value in (ee_attach_quaternions_wxyz or {}).get(
                    ee, (1.0, 0.0, 0.0, 0.0)
                )
            )
            for ee in self._ee_roots
        }
        for ee, quaternion in self._attach_quaternions.items():
            if len(quaternion) != 4 or not all(
                math.isfinite(value) for value in quaternion
            ):
                raise WorkcellModelCompilationError(
                    f"EE {ee!r} attach quaternion must contain four finite values"
                )
            norm = math.sqrt(sum(value * value for value in quaternion))
            if not math.isclose(norm, 1.0, abs_tol=1e-6):
                raise WorkcellModelCompilationError(
                    f"EE {ee!r} attach quaternion must be normalized"
                )
        if not all(math.isfinite(value) for value in self._coupling_offsets.values()):
            raise WorkcellModelCompilationError(
                "EE coupling offsets must be finite"
            )
        self.bare_model_version = bare_model_version
        configured_versions = dict(attached_model_versions or {})
        self._attached_versions = {
            ee: configured_versions.get(ee, f"ur5e-qc-{ee}-attached-v1")
            for ee in self._ee_roots
        }
        if any(not version for version in self._attached_versions.values()):
            raise WorkcellModelCompilationError(
                "attached collision model versions must not be empty"
            )

        # Validate all structural names before accepting the source artifact.
        root = ET.fromstring(source_mjcf)
        bodies = _body_elements(root)
        required = {
            self._robot_root,
            self._qc_master,
            self._qc_coupling,
            *self._ee_roots.values(),
            *self._rack_supports.values(),
        }
        missing = sorted(name for name in required if name not in bodies)
        if missing:
            raise WorkcellModelCompilationError(
                f"source MJCF is missing bodies: {missing}"
            )
        parent_map = {child: parent for parent in root.iter() for child in parent}
        if parent_map.get(bodies[self._qc_coupling]) is not bodies[self._qc_master]:
            raise WorkcellModelCompilationError(
                "QC coupling body must be a direct child of QC master body"
            )
        self._qc_coupling_position = _parse_vector(
            bodies[self._qc_coupling].get("pos"),
            length=3,
            label="QC coupling body position",
        )

    @classmethod
    def from_ee_rack_env(
        cls,
        env: object,
        *,
        bare_model_version: str = DEFAULT_BARE_VERSION,
        attached_model_versions: Mapping[str, str] | None = None,
    ) -> "EEWorkcellCollisionModelCompiler":
        """Capture the MJCF source of a reset ``EERackLayoutEnv`` instance."""

        try:
            source_mjcf = str(env.model.get_xml())  # type: ignore[attr-defined]
            raw_model = env.sim.model._model  # type: ignore[attr-defined]
            raw_data = env.sim.data._data  # type: ignore[attr-defined]
            robot = env.robots[0]  # type: ignore[attr-defined]
            joint_names = tuple(robot.robot_model.joints)
            robot_root = str(robot.robot_model.root_body)
            ee_body_ids = dict(env.ee_body_ids)  # type: ignore[attr-defined]
        except (AttributeError, IndexError, TypeError) as error:
            raise WorkcellModelCompilationError(
                "env must be a reset EERackLayoutEnv"
            ) from error

        body_names = {
            body_id: _mj_name(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in range(raw_model.nbody)
        }
        qc_candidates = [
            name for name in body_names.values() if name.endswith("qc_master_base")
        ]
        if len(qc_candidates) != 1:
            raise WorkcellModelCompilationError(
                f"expected one QC master body, found {qc_candidates}"
            )
        qc_master = qc_candidates[0]

        xml_root = ET.fromstring(source_mjcf)
        bodies = _body_elements(xml_root)
        qc_element = bodies[qc_master]
        coupling_candidates = [
            child.get("name")
            for child in qc_element.findall("body")
            if child.get("name", "").endswith("_eef")
        ]
        if len(coupling_candidates) != 1 or coupling_candidates[0] is None:
            raise WorkcellModelCompilationError(
                f"expected one QC coupling body, found {coupling_candidates}"
            )

        ee_roots = {
            ee: body_names[int(body_id)] for ee, body_id in ee_body_ids.items()
        }
        supports = {ee: f"rack_{ee}_support" for ee in ee_roots}
        baseline: dict[str, float] = {}
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(
                raw_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise WorkcellModelCompilationError(
                    f"robot joint {joint_name!r} is absent from source scene"
                )
            baseline[joint_name] = float(
                raw_data.qpos[int(raw_model.jnt_qposadr[joint_id])]
            )

        coupling_offsets: dict[str, float] = {}
        plan_entries = getattr(env, "plan", {}).get("entries", {})
        for ee in ee_roots:
            entry = plan_entries.get(ee, {})
            if all(key in entry for key in ("coupling_pos", "root_pos", "R_root")):
                difference = np.asarray(entry["coupling_pos"], dtype=float) - np.asarray(
                    entry["root_pos"], dtype=float
                )
                local = np.asarray(entry["R_root"], dtype=float).T @ difference
                coupling_offsets[ee] = float(local[2])
            else:
                coupling_offsets[ee] = 0.0

        return cls(
            source_mjcf,
            joint_names=joint_names,
            robot_root_body_name=robot_root,
            qc_master_body_name=qc_master,
            qc_coupling_body_name=coupling_candidates[0],
            ee_root_body_names=ee_roots,
            rack_support_body_names=supports,
            baseline_joint_positions=baseline,
            ee_coupling_offsets_m=coupling_offsets,
            bare_model_version=bare_model_version,
            attached_model_versions=attached_model_versions,
        )

    @property
    def ee_names(self) -> tuple[str, ...]:
        return tuple(self._ee_roots)

    @property
    def attached_model_versions(self) -> dict[str, str]:
        return dict(self._attached_versions)

    def model_version_for(self, active_ee: str | None) -> str:
        if active_ee is None:
            return self.bare_model_version
        try:
            return self._attached_versions[active_ee]
        except KeyError as error:
            raise WorkcellModelCompilationError(
                f"unknown EE {active_ee!r}; expected one of {sorted(self._ee_roots)}"
            ) from error

    def _entity_body_names(self) -> tuple[tuple[str, str], ...]:
        entities = {"qc_master": self._qc_master}
        entities.update(self._ee_roots)
        entities.update(
            {
                f"rack_support:{ee}": body_name
                for ee, body_name in self._rack_supports.items()
            }
        )
        return tuple(sorted(entities.items()))

    def compile(
        self, active_ee: str | None = None
    ) -> CompiledWorkcellCollisionModel:
        """Compile one physical state; ``None`` preserves the bare rack scene."""

        version = self.model_version_for(active_ee)
        root = ET.fromstring(self._source_mjcf)
        bodies = _body_elements(root)
        if active_ee is not None:
            active_root = bodies[self._ee_roots[active_ee]]
            qc_master = bodies[self._qc_master]
            parent_map = {child: parent for parent in root.iter() for child in parent}
            old_parent = parent_map.get(active_root)
            if old_parent is None:
                raise WorkcellModelCompilationError(
                    f"EE root {self._ee_roots[active_ee]!r} has no parent"
                )
            old_parent.remove(active_root)

            attach_rotation = _quat_wxyz_to_matrix(
                self._attach_quaternions[active_ee]
            )
            root_position = self._qc_coupling_position - attach_rotation @ np.array(
                [0.0, 0.0, self._coupling_offsets[active_ee]], dtype=float
            )
            active_root.set("pos", _format_vector(root_position))
            active_root.set(
                "quat", _format_vector(self._attach_quaternions[active_ee])
            )
            qc_master.append(active_root)

        compiled_xml = ET.tostring(root, encoding="unicode")
        try:
            model = mujoco.MjModel.from_xml_string(compiled_xml)
        except Exception as error:  # noqa: BLE001
            raise WorkcellModelCompilationError(
                f"MuJoCo failed to compile collision model {version!r}: {error}"
            ) from error

        baseline_qpos = np.asarray(model.qpos0, dtype=float).copy()
        for joint_name in self._joint_names:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise WorkcellModelCompilationError(
                    f"compiled model {version!r} lost joint {joint_name!r}"
                )
            joint_type = int(model.jnt_type[joint_id])
            if joint_type not in {
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            }:
                raise WorkcellModelCompilationError(
                    f"joint {joint_name!r} is not scalar"
                )
            baseline_qpos[int(model.jnt_qposadr[joint_id])] = (
                self._baseline_joint_positions[joint_name]
            )

        self._validate_compiled_model(model, baseline_qpos, active_ee)
        return CompiledWorkcellCollisionModel(
            model=model,
            collision_model_version=version,
            active_ee=active_ee,
            joint_names=self._joint_names,
            robot_root_body_name=self._robot_root,
            baseline_qpos=tuple(float(value) for value in baseline_qpos),
            entity_body_names=self._entity_body_names(),
            mjcf_sha256=hashlib.sha256(compiled_xml.encode("utf-8")).hexdigest(),
        )

    def _validate_compiled_model(
        self,
        model: mujoco.MjModel,
        baseline_qpos: np.ndarray,
        active_ee: str | None,
    ) -> None:
        robot_root_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self._robot_root
        )
        qc_master_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self._qc_master
        )
        if robot_root_id < 0 or qc_master_id < 0:
            raise WorkcellModelCompilationError(
                "compiled model lost robot root or QC master"
            )
        for ee, root_name in self._ee_roots.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
            if body_id < 0:
                raise WorkcellModelCompilationError(
                    f"compiled model lost EE root {root_name!r}"
                )
            attached = _is_descendant(model, body_id, robot_root_id)
            if attached != (ee == active_ee):
                raise WorkcellModelCompilationError(
                    f"EE {ee!r} attachment state does not match {active_ee!r}"
                )
            if ee != active_ee and int(model.body_parentid[body_id]) != 0:
                raise WorkcellModelCompilationError(
                    f"inactive EE {ee!r} must remain world-fixed on its rack"
                )

        if active_ee is None:
            return
        data = mujoco.MjData(model)
        data.qpos[:] = baseline_qpos
        mujoco.mj_forward(model, data)
        active_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self._ee_roots[active_ee]
        )
        if int(model.body_parentid[active_id]) != qc_master_id:
            raise WorkcellModelCompilationError(
                f"active EE {active_ee!r} is not a direct child of QC master"
            )
        qc_rotation = np.asarray(data.xmat[qc_master_id]).reshape(3, 3)
        attach_rotation = _quat_wxyz_to_matrix(
            self._attach_quaternions[active_ee]
        )
        expected_position = np.asarray(data.xpos[qc_master_id]) + qc_rotation @ (
            self._qc_coupling_position
            - attach_rotation
            @ np.array([0.0, 0.0, self._coupling_offsets[active_ee]])
        )
        if not np.allclose(data.xpos[active_id], expected_position, atol=1e-9):
            raise WorkcellModelCompilationError(
                f"active EE {active_ee!r} root is not on the QC coupling face"
            )
        expected_rotation = qc_rotation @ attach_rotation
        actual_rotation = np.asarray(data.xmat[active_id]).reshape(3, 3)
        if not np.allclose(actual_rotation, expected_rotation, atol=1e-9):
            raise WorkcellModelCompilationError(
                f"active EE {active_ee!r} root orientation is not aligned to QC"
            )

    def build_collision_registry(
        self,
        collision_contexts: Mapping[str, CollisionContext],
        *,
        collision_margin_m: float,
        allowed_collision_pairs: Iterable[tuple[str, str]] = (),
        include_all_attached_models: bool = True,
    ) -> MuJoCoCollisionModelRegistry:
        """Compile required scene states and route contexts by model version."""

        required_ees = set(self._ee_roots) if include_all_attached_models else {
            context.active_ee
            for context in collision_contexts.values()
            if context.active_ee is not None
        }
        for context in collision_contexts.values():
            expected = self.model_version_for(context.active_ee)
            if context.collision_model_version != expected:
                raise WorkcellModelCompilationError(
                    f"context {context.context_id!r} requires "
                    f"{context.collision_model_version!r}; expected {expected!r}"
                )

        compiled = [self.compile(None)]
        compiled.extend(self.compile(ee) for ee in sorted(required_ees))
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
                default_model_version=self.bare_model_version,
            )
        except MuJoCoCollisionConfigurationError as error:
            raise WorkcellModelCompilationError(str(error)) from error
