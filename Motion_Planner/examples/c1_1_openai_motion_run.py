"""Plan and replay the C1_1 plate pick and sweep with OpenAI keyframes.

The OpenAI model proposes only scene-relative Cartesian keyframe strategies.
UR5e IK branches, collision contexts, path connections, timing, attachment,
controller tracking, and MuJoCo execution remain deterministic local code.

``OPENAI_API_KEY`` must be supplied in the process environment.  It is never
written to the generated artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
TASK_PLANNER = WORKSPACE / "Task_Planner"
for package_root in (PROJECT, TASK_PLANNER):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from task_planner.models import GraspSpec  # noqa: E402
from task_planner.serialization import PlanningResult  # noqa: E402

from motion_planner.geometry import RelativePoseResolver  # noqa: E402
from motion_planner.pipeline import MotionPlanningPipeline  # noqa: E402
from motion_planner.schema import (  # noqa: E402
    ArtifactProvenance,
    AttachedObjectTransform,
    CollisionContext,
    GoalType,
    JointDynamicLimit,
    KeyframeEventType,
    KeyframePlanArtifact,
    KeyframePlanCandidate,
    KeyframePlannerType,
    KeyframeType,
    ModuleName,
    MotionConstraints,
    MotionGoal,
    MotionPlanRequest,
    MotionTask,
    PlannerOptions,
    Pose,
    RelativeKeyframeSpec,
    SimulationConfig,
    SimulationRun,
    StrategyGenerationProvenance,
    StrategyGeneratorKind,
)
from motion_planner.tool_use_journal import (  # noqa: E402
    ToolUseJournalCollisionModelCompiler,
    ToolUseJournalEnvironmentAdapter,
)
from motion_planner.tool_use_journal_runtime import (  # noqa: E402
    ToolUseJournalControllerTrajectoryPlayer,
    ToolUseJournalEERuntime,
)
from motion_planner.tool_use_journal_planning import (  # noqa: E402
    ToolUseJournalCollisionContextFactory,
    attached_object_transform_from_state,
)
from motion_planner.vlm_provider import (  # noqa: E402
    OpenAIKeyframeProvider,
    OpenAIKeyframeProviderConfig,
)


@dataclass(frozen=True)
class _FrozenProvider:
    artifact: KeyframePlanArtifact

    def generate(self, request: MotionPlanRequest) -> KeyframePlanArtifact:
        if self.artifact.scene_signature != request.world.scene.signature:
            raise ValueError("frozen keyframes belong to another scene")
        return self.artifact


def _provenance(
    artifact_id: str,
    artifact_type: str,
    module: ModuleName,
    *inputs: str,
) -> ArtifactProvenance:
    return ArtifactProvenance(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        produced_by=module,
        invocation_id=f"{artifact_id}:invocation",
        input_artifact_ids=list(inputs),
    )


def _write_model(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = value.model_dump_json(indent=2)  # type: ignore[attr-defined]
    path.write_text(rendered + "\n", encoding="utf-8")


def _rotation_from_xyzw(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = quaternion
    result = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(result, np.asarray((w, x, y, z), dtype=float))
    return result.reshape(3, 3)


def _xyzw_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    result = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(result, np.ascontiguousarray(rotation.reshape(9)))
    return (float(result[1]), float(result[2]), float(result[3]), float(result[0]))


def _world_pose(record: dict) -> tuple[np.ndarray, np.ndarray]:
    raw = record["pose"]
    return (
        np.asarray(raw["position_m"], dtype=float),
        _rotation_from_xyzw(tuple(raw["orientation_xyzw"])),
    )


def _target_site_pose(
    env: object,
    runtime: ToolUseJournalEERuntime,
    target_hand_pose: Pose,
) -> tuple[str, str, np.ndarray, np.ndarray]:
    model = env.sim.model._model  # type: ignore[attr-defined]
    data = env.sim.data._data  # type: ignore[attr-defined]
    adapter = ToolUseJournalEnvironmentAdapter(env)
    hand_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, adapter.hand_body
    )
    reference_kind, reference_name, site_position, site_rotation = (
        runtime._grasp_reference(env)
    )
    if reference_kind != "site":
        raise RuntimeError("C1_1 runtime did not expose a grasp site")
    hand_position = np.asarray(data.xpos[hand_id], dtype=float)
    hand_rotation = np.asarray(data.xmat[hand_id], dtype=float).reshape(3, 3)
    site_in_hand = hand_rotation.T @ (site_position - hand_position)
    site_rotation_in_hand = hand_rotation.T @ site_rotation
    target_hand_position = np.asarray(target_hand_pose.position_m, dtype=float)
    target_hand_rotation = _rotation_from_xyzw(
        tuple(target_hand_pose.orientation_xyzw)
    )
    target_site_position = (
        target_hand_position + target_hand_rotation @ site_in_hand
    )
    target_site_rotation = target_hand_rotation @ site_rotation_in_hand
    return (
        reference_kind,
        reference_name,
        target_site_position,
        target_site_rotation,
    )


def _attachment_at_keyframe(
    runtime: ToolUseJournalEERuntime,
    request: MotionPlanRequest,
    keyframe: object,
    object_id: str,
) -> AttachedObjectTransform:
    target_hand_pose = RelativePoseResolver(request.world).resolve(keyframe)
    kind, name, site_position, site_rotation = _target_site_pose(
        runtime.env, runtime, target_hand_pose
    )
    record = request.world.objects[object_id]
    object_position, object_rotation = _world_pose(record)
    relative_position = site_rotation.T @ (object_position - site_position)
    relative_rotation = site_rotation.T @ object_rotation
    return AttachedObjectTransform(
        object_id=object_id,
        free_joint_name=str(record["free_joint_name"]),
        reference_kind=kind,
        reference_name=name,
        position_in_reference_m=tuple(float(value) for value in relative_position),
        orientation_in_reference_xyzw=_xyzw_from_rotation(relative_rotation),
    )


def _runtime_attachment_transform(
    runtime: ToolUseJournalEERuntime,
) -> AttachedObjectTransform:
    attachment = runtime.attachment
    if attachment is None:
        raise RuntimeError("runtime has no attached plate")
    return AttachedObjectTransform(
        object_id=attachment.object_id,
        free_joint_name=attachment.free_joint_name,
        reference_kind=attachment.reference_kind,  # type: ignore[arg-type]
        reference_name=attachment.reference_name,
        position_in_reference_m=attachment.position_in_reference_m,
        orientation_in_reference_xyzw=_xyzw_from_rotation(
            np.asarray(attachment.rotation_in_reference, dtype=float)
        ),
    )


def _limits(joint_names: list[str]) -> dict[str, JointDynamicLimit]:
    return {
        name: JointDynamicLimit(
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=2.0,
            max_jerk_rad_s3=20.0,
        )
        for name in joint_names
    }


def _constraints(joint_names: list[str]) -> MotionConstraints:
    return MotionConstraints(
        collision_margin_m=0.002,
        position_tolerance_m=0.008,
        orientation_tolerance_rad=0.08,
        velocity_scaling=0.25,
        acceleration_scaling=0.25,
        jerk_scaling=0.25,
        max_cartesian_speed_m_s=0.18,
        max_joint_path_step_rad=0.035,
        joint_limits=_limits(joint_names),
    )


def _options(seed: int) -> PlannerOptions:
    return PlannerOptions(
        allowed_planning_time_s=12.0,
        max_attempts=5,
        interpolation_dt_s=0.02,
        cartesian_translation_step_m=0.01,
        cartesian_rotation_step_rad=0.08,
        rrt_extension_step_rad=0.2,
        rrt_max_iterations=5000,
        rrt_goal_bias=0.2,
        random_seed=seed,
    )


def _request(
    *,
    request_id: str,
    world: object,
    task: MotionTask,
    seed: int,
    task_planner_artifact: Path,
) -> MotionPlanRequest:
    return MotionPlanRequest(
        request_id=request_id,
        provenance=_provenance(
            f"{request_id}:artifact",
            "MotionPlanRequest",
            ModuleName.TASK_PLANNER,
            str(task_planner_artifact.resolve()),
        ),
        world=world,
        task=task,
        constraints=_constraints(list(world.robot_state.joint_names)),
        options=_options(seed),
    )


def _openai_artifact(
    request: MotionPlanRequest,
    *,
    model: str,
    candidates: int,
    cache_dir: Path,
) -> KeyframePlanArtifact:
    config = OpenAIKeyframeProviderConfig.from_environment(
        model=model,
        candidate_count=candidates,
        reasoning_effort="medium",
        timeout_s=120.0,
        cache_dir=cache_dir,
    )
    return OpenAIKeyframeProvider(config).generate(request)


def _install_sweep_reference_frames(world: object) -> tuple[str, ...]:
    """Add non-physical frames computed from C1_1 MJCF task geometry."""

    lanes = (-0.21, 0.0, 0.21)
    # UR5e reach shrinks toward the lateral edges of this workcell.  The plate
    # is 181.8 mm wide, so x=0.28 still begins behind the outermost blocks for
    # the side lanes; the center lane can start at x=0.32.
    lane_start_x = (0.28, 0.32, 0.28)
    names: list[str] = []
    for lane_index, lane_y in enumerate(lanes):
        for phase, x_position in (
            ("start", lane_start_x[lane_index]),
            ("end", 0.04),
        ):
            name = f"sweep_lane_{lane_index}_{phase}"
            world.objects[name] = {
                "pose": {
                    "frame_id": "world",
                    "position_m": [x_position, lane_y, 0.91],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "anchors": {"center": [0.0, 0.0, 0.0]},
                "collision_enabled": False,
                "reference_frame_kind": "TASK_GEOMETRY",
            }
            names.append(name)
    return tuple(names)


def _sweep_template_artifact(
    request: MotionPlanRequest,
) -> KeyframePlanArtifact:
    """Build two deterministic three-lane sweep orders from scene geometry."""

    input_payload = {
        "scene_signature": request.world.scene.signature,
        "subgoal_id": request.task.subgoal_id,
        "frames": {
            name: request.world.objects[name]
            for name in sorted(request.world.objects)
            if name.startswith("sweep_lane_")
        },
    }
    digest = hashlib.sha256(
        json.dumps(input_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    candidates: list[KeyframePlanCandidate] = []
    for candidate_index, lane_order in enumerate(((0, 1, 2), (2, 1, 0)), start=1):
        keyframes: list[RelativeKeyframeSpec] = []
        for sequence_index, lane_index in enumerate(lane_order, start=1):
            common = {
                "anchor": "center",
                "approach_axis_xyz": (1.0, 0.0, 0.0),
                "tool_axis_to_align": "+z",
                "offset_along_approach_m": 0.0,
                "roll_rad": 0.0,
            }
            keyframes.append(
                RelativeKeyframeSpec(
                    keyframe_id=(
                        f"{request.task.subgoal_id}:lane-{lane_index}:stage"
                    ),
                    keyframe_type=KeyframeType.TRANSFER,
                    frame_ref=f"object:sweep_lane_{lane_index}_start",
                    planner=KeyframePlannerType.SAMPLING_BASED,
                    **common,
                )
            )
            keyframes.append(
                RelativeKeyframeSpec(
                    keyframe_id=(
                        f"{request.task.subgoal_id}:lane-{lane_index}:sweep"
                    ),
                    keyframe_type=KeyframeType.CUSTOM,
                    frame_ref=f"object:sweep_lane_{lane_index}_end",
                    planner=KeyframePlannerType.CARTESIAN,
                    **common,
                )
            )
        candidates.append(
            KeyframePlanCandidate(
                strategy_id=(
                    f"{request.task.subgoal_id}:mjcf-three-lane-{candidate_index}"
                ),
                keyframes=keyframes,
                rationale=(
                    "Orient the 181.8 mm plate vertically and sweep three "
                    "overlapping y lanes from the block field into the zone."
                ),
                provenance=StrategyGenerationProvenance(
                    generator_kind=StrategyGeneratorKind.TASK_GEOMETRY,
                    generator_id="C1_1_MJCF_SWEEP_LANES_V1",
                    input_hash=digest,
                    attempt_index=1,
                ),
            )
        )
    return KeyframePlanArtifact(
        artifact_id=f"keyframe-plan:c1-1-sweep:{digest[:24]}",
        provenance=_provenance(
            f"keyframe-plan-artifact:c1-1-sweep:{digest[:24]}",
            "KeyframePlanArtifact",
            ModuleName.MOTION_PLANNER,
            request.provenance.artifact_id,
        ),
        scene_signature=request.world.scene.signature,
        subgoal_id=request.task.subgoal_id,
        candidates=candidates,
    )


def _bind_2f_plate_rim_grasps(
    raw: KeyframePlanArtifact,
    request: MotionPlanRequest,
    *,
    finger_centerline_inset_m: float = 0.021,
) -> KeyframePlanArtifact:
    """Bind symbolic top-down proposals to MJCF-derived plate rim grasps.

    OpenAI still supplies the high-level approach strategy.  Exact metric
    anchors come from the plate collision bounds and the Robotiq 85 finger-pad
    geometry.  Center grasps are invalid for this thin plate because the two
    fingers never oppose one another around an edge.
    """

    plate = request.world.objects.get("heavy_plate")
    if not isinstance(plate, dict):
        raise RuntimeError("heavy_plate MJCF record is absent")
    dimensions = plate.get("dimensions_m")
    anchors = plate.get("anchors")
    if (
        not isinstance(dimensions, list)
        or len(dimensions) < 3
        or not isinstance(anchors, dict)
        or not isinstance(anchors.get("center"), list)
        or not isinstance(anchors.get("top"), list)
    ):
        raise RuntimeError("heavy_plate MJCF bounds and anchors are absent")
    center = np.asarray(anchors["center"], dtype=float)
    top = np.asarray(anchors["top"], dtype=float)
    half_x = float(dimensions[0]) * 0.5
    half_y = float(dimensions[1]) * 0.5
    if min(half_x, half_y) <= finger_centerline_inset_m:
        raise RuntimeError("heavy_plate is too small for the 2F rim template")
    radial_x = half_x - finger_centerline_inset_m
    radial_y = half_y - finger_centerline_inset_m
    rim_anchors = {
        "2f_rim_x_pos": [center[0] + radial_x, center[1], top[2]],
        "2f_rim_x_neg": [center[0] - radial_x, center[1], top[2]],
        "2f_rim_y_pos": [center[0], center[1] + radial_y, top[2]],
        "2f_rim_y_neg": [center[0], center[1] - radial_y, top[2]],
    }
    anchors.update(
        {
            name: [float(value) for value in position]
            for name, position in rim_anchors.items()
        }
    )
    rim_variants = (
        ("2f_rim_x_pos", 0.0),
        ("2f_rim_x_neg", 0.0),
        ("2f_rim_y_pos", np.pi / 2.0),
        ("2f_rim_y_neg", np.pi / 2.0),
    )
    candidates: list[KeyframePlanCandidate] = []
    for candidate_index, candidate in enumerate(raw.candidates):
        anchor, roll = rim_variants[candidate_index % len(rim_variants)]
        keyframes: list[RelativeKeyframeSpec] = []
        for keyframe in candidate.keyframes:
            updates: dict[str, object] = {}
            if keyframe.frame_ref == "object:heavy_plate":
                updates.update(
                    anchor=anchor,
                    approach_axis_xyz=(0.0, 0.0, 1.0),
                    tool_axis_to_align="-z",
                    roll_rad=float(roll),
                )
                if keyframe.keyframe_type is KeyframeType.PRE_GRASP:
                    updates["offset_along_approach_m"] = max(
                        0.08, keyframe.offset_along_approach_m
                    )
                elif keyframe.keyframe_type is KeyframeType.GRASP:
                    updates["offset_along_approach_m"] = 0.0
                elif keyframe.keyframe_type in {
                    KeyframeType.LIFT,
                    KeyframeType.RETREAT,
                }:
                    updates["offset_along_approach_m"] = max(
                        0.10, keyframe.offset_along_approach_m
                    )
            keyframes.append(keyframe.model_copy(update=updates))
        candidates.append(
            candidate.model_copy(
                update={
                    "strategy_id": f"{candidate.strategy_id}:mjcf-2f-rim",
                    "keyframes": keyframes,
                    "rationale": (
                        f"{candidate.rationale} Exact grasp pose is bound to "
                        f"MJCF rim anchor {anchor!r} for opposed 2F contact."
                    ),
                    "metadata": {
                        **candidate.metadata,
                        "geometry_binder": "C1_1_MJCF_2F_PLATE_RIM_V1",
                        "rim_anchor": anchor,
                        "finger_centerline_inset_m": (
                            finger_centerline_inset_m
                        ),
                    },
                }
            )
        )
    digest = hashlib.sha256(
        json.dumps(
            {
                "source_artifact_id": raw.artifact_id,
                "plate_dimensions_m": dimensions,
                "rim_anchors": rim_anchors,
                "finger_centerline_inset_m": finger_centerline_inset_m,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return raw.model_copy(
        update={
            "artifact_id": f"{raw.artifact_id}:mjcf-2f-rim:{digest}",
            "provenance": raw.provenance.model_copy(
                update={
                    "artifact_id": (
                        f"{raw.provenance.artifact_id}:mjcf-2f-rim:{digest}"
                    ),
                    "metadata": {
                        **raw.provenance.metadata,
                        "geometry_binder": "C1_1_MJCF_2F_PLATE_RIM_V1",
                        "source_keyframe_artifact_id": raw.artifact_id,
                    },
                }
            ),
            "candidates": candidates,
        }
    )


def _with_breakable_weld(
    raw: KeyframePlanArtifact,
    *,
    grasp_clearance_reserve_m: float = 0.0,
) -> KeyframePlanArtifact:
    """Require a settled, opposed-finger grasp before enabling the weld."""

    if grasp_clearance_reserve_m < 0.0:
        raise ValueError("grasp_clearance_reserve_m must be non-negative")

    bound = raw.model_copy(deep=True)
    for candidate in bound.candidates:
        grasp = next(
            (
                keyframe
                for keyframe in candidate.keyframes
                if keyframe.keyframe_type is KeyframeType.GRASP
            ),
            None,
        )
        if grasp is None:
            raise RuntimeError(
                f"strategy {candidate.strategy_id!r} has no GRASP keyframe"
            )
        grasp.offset_along_approach_m += grasp_clearance_reserve_m
        grasp.metadata = {
            **grasp.metadata,
            "hold_duration_after_s": 0.25,
            "event_time_offsets_s": {
                "GRIPPER_CLOSE": 0.0,
                "ATTACH_OBJECT": 0.25,
            },
            "event_parameters": {
                "ATTACH_OBJECT": {
                    "attachment_mode": "BREAKABLE_WELD",
                    "max_attach_distance_m": 0.015,
                    "max_attach_penetration_m": 0.02,
                    "require_grasp_command": True,
                    "require_contact": True,
                    "require_retention_contact": False,
                    "min_contact_count": 2,
                    "min_normal_force_n": 1.0,
                    "required_contact_groups": [
                        "left_finger",
                        "right_finger",
                    ],
                    "natural_frequency_hz": 6.0,
                    "damping_ratio": 1.0,
                    "max_weld_force_n": 100.0,
                    "max_weld_torque_nm": 25.0,
                    "max_position_error_m": 0.08,
                    "max_orientation_error_rad": 0.8,
                    "max_contact_force_n": 450.0,
                    "startup_grace_steps": 30,
                    "contact_loss_grace_steps": 60,
                    "break_debounce_steps": 4,
                }
            },
        }
    return bound


def _contextualize_pick(
    raw: KeyframePlanArtifact,
    request: MotionPlanRequest,
    runtime: ToolUseJournalEERuntime,
    compiler: ToolUseJournalCollisionModelCompiler,
) -> tuple[KeyframePlanArtifact, dict[str, CollisionContext], str]:
    model_version = compiler.model_version_for("2F")
    contact_id = "c1_1:pick:contact"
    contexts: dict[str, CollisionContext] = {
        contact_id: CollisionContext(
            context_id=contact_id,
            scene_state_id="c1_1:plate-free",
            active_ee="2F",
            allowed_collision_pairs=[
                ("2F", "heavy_plate"),
                ("heavy_plate", "table*"),
            ],
            collision_model_version=model_version,
        )
    }
    candidates = []
    for candidate_index, candidate in enumerate(raw.candidates):
        grasp_indices = [
            index
            for index, keyframe in enumerate(candidate.keyframes)
            if keyframe.keyframe_type is KeyframeType.GRASP
        ]
        if len(grasp_indices) != 1:
            raise RuntimeError(
                f"strategy {candidate.strategy_id!r} needs exactly one GRASP"
            )
        grasp_index = grasp_indices[0]
        transform = _attachment_at_keyframe(
            runtime,
            request,
            candidate.keyframes[grasp_index],
            "heavy_plate",
        )
        attached_id = f"c1_1:pick:attached:{candidate_index}"
        contexts[attached_id] = CollisionContext(
            context_id=attached_id,
            scene_state_id=f"c1_1:plate-attached:{candidate_index}",
            active_ee="2F",
            attached_object_ids=["heavy_plate"],
            attached_object_transforms=[transform],
            touch_links=["2F"],
            allowed_collision_pairs=[
                ("2F", "heavy_plate"),
                ("heavy_plate", "table*"),
            ],
            collision_model_version=model_version,
        )
        keyframes = []
        for index, keyframe in enumerate(candidate.keyframes):
            metadata = dict(keyframe.metadata)
            updates: dict[str, object] = {
                "collision_context_id": (
                    contact_id if index <= grasp_index else attached_id
                )
            }
            if index == grasp_index:
                metadata["event_target_id"] = "heavy_plate"
                metadata["hold_duration_after_s"] = 0.25
                metadata["event_time_offsets_s"] = {
                    "GRIPPER_CLOSE": 0.0,
                    "ATTACH_OBJECT": 0.25,
                }
                metadata["event_parameters"] = {
                    "ATTACH_OBJECT": {
                        "attachment_mode": "BREAKABLE_WELD",
                        "max_attach_distance_m": 0.015,
                        "max_attach_penetration_m": 0.02,
                        "require_grasp_command": True,
                        "require_contact": True,
                        "require_retention_contact": False,
                        "min_contact_count": 2,
                        "min_normal_force_n": 1.0,
                        "required_contact_groups": [
                            "left_finger",
                            "right_finger",
                        ],
                        "natural_frequency_hz": 6.0,
                        "damping_ratio": 1.0,
                        "max_weld_force_n": 100.0,
                        "max_weld_torque_nm": 25.0,
                        "max_position_error_m": 0.08,
                        "max_orientation_error_rad": 0.8,
                        "max_contact_force_n": 450.0,
                        "startup_grace_steps": 30,
                        "contact_loss_grace_steps": 60,
                        "break_debounce_steps": 4,
                    }
                }
                updates.update(
                    events_after=[
                        KeyframeEventType.GRIPPER_CLOSE,
                        KeyframeEventType.ATTACH_OBJECT,
                    ],
                    collision_context_after_events_id=attached_id,
                    metadata=metadata,
                )
            keyframes.append(keyframe.model_copy(update=updates))
        candidates.append(candidate.model_copy(update={"keyframes": keyframes}))
    suffix = hashlib.sha256(
        "|".join(sorted(contexts)).encode("utf-8")
    ).hexdigest()[:12]
    artifact = raw.model_copy(
        update={
            "artifact_id": f"{raw.artifact_id}:context:{suffix}",
            "provenance": raw.provenance.model_copy(
                update={
                    "artifact_id": f"{raw.provenance.artifact_id}:context:{suffix}",
                    "metadata": {
                        **raw.provenance.metadata,
                        "contextualizer": "C1_1_PICK_BREAKABLE_WELD_V1",
                    },
                }
            ),
            "candidates": candidates,
        }
    )
    return artifact, contexts, contact_id


def _contextualize_sweep(
    raw: KeyframePlanArtifact,
    compiler: ToolUseJournalCollisionModelCompiler,
    transform: AttachedObjectTransform,
    block_ids: list[str],
) -> tuple[KeyframePlanArtifact, dict[str, CollisionContext], str]:
    context_id = "c1_1:sweep:plate-attached"
    allowed = [
        ("2F", "heavy_plate"),
        ("heavy_plate", "table*"),
        *(("heavy_plate", block_id) for block_id in block_ids),
    ]
    context = CollisionContext(
        context_id=context_id,
        scene_state_id="c1_1:sweep:plate-attached",
        active_ee="2F",
        attached_object_ids=["heavy_plate"],
        attached_object_transforms=[transform],
        touch_links=["2F"],
        allowed_collision_pairs=allowed,
        collision_model_version=compiler.model_version_for("2F"),
    )
    candidates = [
        candidate.model_copy(
            update={
                "keyframes": [
                    keyframe.model_copy(
                        update={"collision_context_id": context_id}
                    )
                    for keyframe in candidate.keyframes
                ]
            }
        )
        for candidate in raw.candidates
    ]
    artifact = raw.model_copy(
        update={
            "artifact_id": f"{raw.artifact_id}:sweep-context",
            "provenance": raw.provenance.model_copy(
                update={
                    "artifact_id": (
                        f"{raw.provenance.artifact_id}:sweep-context"
                    ),
                    "metadata": {
                        **raw.provenance.metadata,
                        "contextualizer": "C1_1_SWEEP_ATTACHED_TOOL_V1",
                    },
                }
            ),
            "candidates": candidates,
        }
    )
    return artifact, {context_id: context}, context_id


def _plan(
    request: MotionPlanRequest,
    artifact: KeyframePlanArtifact,
    adapter: ToolUseJournalEnvironmentAdapter,
    collision_factory: ToolUseJournalCollisionContextFactory,
):
    setup = collision_factory.prepare(request, artifact)
    pipeline = MotionPlanningPipeline(
        _FrozenProvider(setup.keyframe_artifact), adapter.make_kinematics()
    )
    result = pipeline.plan(
        request,
        state_validator=setup.state_validator,
        collision_contexts=setup.collision_contexts,
        initial_collision_context_id=setup.initial_collision_context_id,
        final_segment_validator=setup.final_segment_validator,
    )
    return result, setup.state_validator


def _run(
    runtime: ToolUseJournalEERuntime,
    plan: object,
    collision_registry: object,
    run_id: str,
):
    env = runtime.env
    simulation_run = SimulationRun(
        run_id=run_id,
        provenance=_provenance(
            f"{run_id}:artifact",
            "SimulationRun",
            ModuleName.SIMULATOR,
            plan.provenance.artifact_id,
        ),
        plan=plan,
        config=SimulationConfig(
            physics_timestep_s=float(env.model_timestep),
            control_timestep_s=float(env.control_timestep),
            realtime_factor=0.0,
            max_duration_s=max(30.0, float(plan.duration_s) + 5.0),
            terminate_on_collision=True,
            render=False,
            random_seed=0,
        ),
    )
    report = ToolUseJournalControllerTrajectoryPlayer(
        runtime, collision_probe=collision_registry
    ).execute(simulation_run)
    return simulation_run, report


def _inside_zone(env: object) -> list[str]:
    model = env.sim.model._model  # type: ignore[attr-defined]
    data = env.sim.data._data  # type: ignore[attr-defined]
    zone = np.asarray(env.collection_zone_center, dtype=float)  # type: ignore[attr-defined]
    size = np.asarray(env.collection_zone_size, dtype=float)  # type: ignore[attr-defined]
    inside = []
    for block in env.blocks:  # type: ignore[attr-defined]
        body_id = env.obj_body_id[block.name]  # type: ignore[attr-defined]
        position = np.asarray(data.xpos[body_id, :2], dtype=float)
        if np.all(np.abs(position - zone) <= size / 2.0):
            inside.append(block.name)
    return inside


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("--task-planner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--stop-after-pick",
        action="store_true",
        help="Replay and report only the OpenAI-generated heavy-plate pick plan.",
    )
    parser.add_argument(
        "--sweep-provider",
        choices=("task-geometry", "openai"),
        default="task-geometry",
        help="Use MJCF-derived sweep lanes or request general sweep keyframes.",
    )
    args = parser.parse_args()

    repository = args.repository.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    planning_result = PlanningResult.model_validate_json(
        args.task_planner.read_text(encoding="utf-8")
    )
    if planning_result.selected_plan is None:
        raise RuntimeError("Task Planner did not select a plan")
    selected = planning_result.selected_plan
    if set(selected.group_ee_assignments.values()) != {"2F"}:
        raise RuntimeError("C1_1 run expects Task Planner to select 2F")
    if set(selected.group_tool_assignments.values()) != {"heavy_plate"}:
        raise RuntimeError("C1_1 run expects Task Planner to select heavy_plate")

    runtime = ToolUseJournalEERuntime.from_repository_for_controller(
        repository,
        "C1_1_LegoSweep",
        active_ee="2F",
        seed=args.seed,
        ignore_done=True,
        use_camera_obs=True,
        has_offscreen_renderer=True,
        camera_names="agentview",
        camera_heights=args.height,
        camera_widths=args.width,
        render_camera="agentview",
    )
    reports = []
    plans = []
    try:
        adapter = ToolUseJournalEnvironmentAdapter(runtime.env)
        adapter.require_physical_ee("2F")
        world = adapter.world_snapshot()
        pick_task = MotionTask(
            task_id="c1_1:pick-heavy-plate",
            subgoal_id="SG1_d1",
            action_type="PICK",
            ee="2F",
            tool="heavy_plate",
            target_ids=["heavy_plate"],
            grasp=GraspSpec(
                grasp_id="openai:c1_1:heavy_plate",
                owner_kind="tool",
                owner_id="heavy_plate",
                source="openai_keyframe_provider",
            ),
            goal=MotionGoal(
                goal_type=GoalType.PICK,
                target_object_id="heavy_plate",
                approach_distance_m=0.08,
                retreat_distance_m=0.10,
            ),
            allowed_touch_objects=["heavy_plate"],
            metadata={"task_planner_subgoal": "SG1_d1", "operation": "PICK_TOOL"},
        )
        pick_request = _request(
            request_id="c1_1:motion-request:pick-heavy-plate",
            world=world,
            task=pick_task,
            seed=args.seed,
            task_planner_artifact=args.task_planner,
        )
        pick_request.constraints.allowed_collision_pairs = [
            ("heavy_plate", "table*"),
        ]
        _write_model(output / "pick_request.json", pick_request)
        raw_pick = _openai_artifact(
            pick_request,
            model=args.model,
            candidates=args.candidates,
            cache_dir=output / "keyframe_cache",
        )
        _write_model(output / "pick_keyframes_raw.json", raw_pick)
        rim_pick = _bind_2f_plate_rim_grasps(raw_pick, pick_request)
        _write_model(output / "pick_keyframes_rim_bound.json", rim_pick)
        pick_compiler = ToolUseJournalCollisionModelCompiler.from_repository(
            runtime.env,
            repository,
            seed=args.seed,
            ignore_done=True,
            use_camera_obs=False,
            has_offscreen_renderer=False,
        )
        pick_artifact = _with_breakable_weld(rim_pick)
        reference_kind, reference_name, _, _ = runtime._grasp_reference(
            runtime.env
        )
        pick_factory = ToolUseJournalCollisionContextFactory(
            pick_compiler,
            attachment_reference_name=reference_name,
            attachment_reference_kind=reference_kind,
        )
        pick_result, pick_registry = _plan(
            pick_request,
            pick_artifact,
            adapter,
            pick_factory,
        )
        _write_model(
            output / "pick_keyframes_contextualized.json",
            pick_result.keyframe_artifact,
        )
        pick_plan = pick_result.plan
        plans.append(pick_plan)
        _write_model(output / "pick_motion_plan.json", pick_plan)
        pick_run, pick_report = _run(
            runtime,
            pick_plan,
            pick_registry,
            "c1_1:controller:pick-heavy-plate",
        )
        _write_model(output / "pick_simulation_run.json", pick_run)
        _write_model(output / "pick_execution_report.json", pick_report)
        reports.append(pick_report)
        if pick_report.status.value != "SUCCESS":
            raise RuntimeError(
                f"pick controller replay failed: {pick_report.failure}"
            )

        adapter = ToolUseJournalEnvironmentAdapter(runtime.env)
        if runtime.attachment is None:
            raise RuntimeError("pick replay completed without an attachment")
        if args.stop_after_pick:
            frame = runtime.env.sim.render(
                camera_name="agentview", height=args.height, width=args.width
            )[::-1]
            final_frame = output / "pick_controller_final.png"
            cv2.imwrite(str(final_frame), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            summary = {
                "status": "SUCCESS",
                "phase": "PICK",
                "selected_ee": "2F",
                "selected_tool": "heavy_plate",
                "model": args.model,
                "candidate_count": args.candidates,
                "seed": args.seed,
                "plan_id": pick_plan.plan_id,
                "plan_duration_s": pick_plan.duration_s,
                "trajectory_waypoint_count": sum(
                    len(segment.waypoints) for segment in pick_plan.segments
                ),
                "execution_status": pick_report.status.value,
                "attachment_mode": "BREAKABLE_WELD",
                "final_attached_object_id": runtime.attached_object_id,
                "last_attachment_break": (
                    runtime.last_attachment_break.as_mapping()
                    if runtime.last_attachment_break is not None
                    else None
                ),
                "artifacts": {
                    "pick_plan": str((output / "pick_motion_plan.json").resolve()),
                    "pick_report": str(
                        (output / "pick_execution_report.json").resolve()
                    ),
                    "final_frame": str(final_frame.resolve()),
                },
            }
            (output / "pick_run_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        world = adapter.world_snapshot(
            attached_object_transform=attached_object_transform_from_state(
                runtime.attachment
            )
        )
        world.metadata["held_tool"] = "heavy_plate"
        world.metadata["attachment_mode"] = "BREAKABLE_WELD"
        _install_sweep_reference_frames(world)
        block_ids = [f"block_{index}" for index in range(12)]
        zone_pose = Pose.model_validate(world.objects["collection_zone_visual"]["pose"])
        sweep_task = MotionTask(
            task_id="c1_1:sweep-collect",
            subgoal_id="SG1_d2",
            action_type="tool_act:sweep",
            ee="2F",
            tool="heavy_plate",
            target_ids=block_ids,
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_pose=zone_pose,
                target_region_id="collection_zone_visual",
            ),
            allowed_touch_objects=block_ids,
            metadata={"task_planner_subgoal": "SG1_d2", "held_tool": "heavy_plate"},
        )
        sweep_request = _request(
            request_id="c1_1:motion-request:sweep",
            world=world,
            task=sweep_task,
            seed=args.seed,
            task_planner_artifact=args.task_planner,
        )
        sweep_request.constraints.allowed_collision_pairs = [
            ("heavy_plate", "table*"),
            *(("heavy_plate", block_id) for block_id in block_ids),
        ]
        _write_model(output / "sweep_request.json", sweep_request)
        raw_sweep = (
            _openai_artifact(
                sweep_request,
                model=args.model,
                candidates=args.candidates,
                cache_dir=output / "keyframe_cache",
            )
            if args.sweep_provider == "openai"
            else _sweep_template_artifact(sweep_request)
        )
        _write_model(output / "sweep_keyframes_raw.json", raw_sweep)
        sweep_compiler = ToolUseJournalCollisionModelCompiler.from_repository(
            runtime.env,
            repository,
            seed=args.seed,
            ignore_done=True,
            use_camera_obs=False,
            has_offscreen_renderer=False,
        )
        reference_kind, reference_name, _, _ = runtime._grasp_reference(
            runtime.env
        )
        sweep_factory = ToolUseJournalCollisionContextFactory(
            sweep_compiler,
            attachment_reference_name=reference_name,
            attachment_reference_kind=reference_kind,
        )
        sweep_result, sweep_registry = _plan(
            sweep_request,
            raw_sweep,
            adapter,
            sweep_factory,
        )
        _write_model(
            output / "sweep_keyframes_contextualized.json",
            sweep_result.keyframe_artifact,
        )
        sweep_plan = sweep_result.plan
        plans.append(sweep_plan)
        _write_model(output / "sweep_motion_plan.json", sweep_plan)
        sweep_run, sweep_report = _run(
            runtime,
            sweep_plan,
            sweep_registry,
            "c1_1:controller:sweep",
        )
        _write_model(output / "sweep_simulation_run.json", sweep_run)
        _write_model(output / "sweep_execution_report.json", sweep_report)
        reports.append(sweep_report)

        inside = _inside_zone(runtime.env)
        frame = runtime.env.sim.render(
            camera_name="agentview", height=args.height, width=args.width
        )[::-1]
        final_frame = output / "controller_final.png"
        cv2.imwrite(str(final_frame), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        summary = {
            "status": (
                "SUCCESS"
                if all(report.status.value == "SUCCESS" for report in reports)
                else "FAILED"
            ),
            "task_goal_satisfied": len(inside) == 12,
            "selected_ee": "2F",
            "selected_tool": "heavy_plate",
            "model": args.model,
            "candidate_count": args.candidates,
            "sweep_provider": args.sweep_provider,
            "seed": args.seed,
            "plan_ids": [plan.plan_id for plan in plans],
            "execution_statuses": [report.status.value for report in reports],
            "blocks_inside": inside,
            "blocks_inside_count": len(inside),
            "blocks_total": 12,
            "attachment_mode": "BREAKABLE_WELD",
            "final_attached_object_id": runtime.attached_object_id,
            "last_attachment_break": (
                runtime.last_attachment_break.as_mapping()
                if runtime.last_attachment_break is not None
                else None
            ),
            "artifacts": {
                "pick_plan": str((output / "pick_motion_plan.json").resolve()),
                "sweep_plan": str((output / "sweep_motion_plan.json").resolve()),
                "pick_report": str((output / "pick_execution_report.json").resolve()),
                "sweep_report": str((output / "sweep_execution_report.json").resolve()),
                "final_frame": str(final_frame.resolve()),
            },
        }
        (output / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["status"] == "SUCCESS" else 2
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
