"""Vacuum GRASP binds TCP to the approach-facing surface without double-offset."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tuj.m5_motion.geometry import RelativePoseResolver
from tuj.m5_motion.grasp_geometry import (
    SUCTION_SURFACE_CONTACT,
    SuctionSurfaceContactBinder,
    bind_suction_surface_contact,
)
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    GoalType,
    KeyframePlanArtifact,
    KeyframePlanCandidate,
    KeyframePlannerType,
    KeyframeType,
    ModuleName,
    MotionGoal,
    MotionPlanRequest,
    MotionTask,
    RelativeKeyframeSpec,
    RobotState,
    SceneRef,
    StrategyGenerationProvenance,
    StrategyGeneratorKind,
    WorldSnapshot,
)


def _pose(
    *,
    z: float = 1.0,
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> dict:
    return {
        "frame_id": "world",
        "position_m": [0.0, 0.0, z],
        "orientation_xyzw": list(orientation_xyzw),
    }


def _request(
    *,
    ee: str = "vac",
    action_type: str = "acquire",
    capabilities: list[str] | None = None,
    dimensions: tuple[float, float, float] = (0.16, 0.16, 0.01),
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    object_z: float = 0.92,
) -> MotionPlanRequest:
    half_z = dimensions[2] * 0.5
    return MotionPlanRequest(
        request_id="request:suction",
        provenance=ArtifactProvenance(
            artifact_id="request-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="test",
        ),
        world=WorldSnapshot(
            scene=SceneRef(signature="scene"),
            robot_state=RobotState(
                robot_id="robot",
                joint_names=["j1"],
                joint_positions_rad=[0.0],
            ),
            objects={
                "part": {
                    "pose": _pose(z=object_z, orientation_xyzw=orientation_xyzw),
                    "dimensions_m": list(dimensions),
                    "anchors": {
                        "center": [0.0, 0.0, 0.0],
                        "top": [0.0, 0.0, half_z],
                        "top_center": [0.0, 0.0, half_z],
                        "bottom": [0.0, 0.0, -half_z],
                        "bottom_center": [0.0, 0.0, -half_z],
                    },
                }
            },
            metadata={"physical_active_ee": ee},
        ),
        task=MotionTask(
            task_id="task",
            subgoal_id="SG",
            action_type=action_type,
            ee=ee,
            target_ids=["part"],
            goal=MotionGoal(goal_type=GoalType.POSE, target_object_id="part"),
            metadata={
                "operation": action_type.upper(),
                "ee_capabilities": list(
                    capabilities
                    if capabilities is not None
                    else (["suction"] if ee in {"vac", "vacuum"} else [])
                ),
            },
        ),
    )


def _keyframe(
    *,
    kind: KeyframeType,
    anchor: str,
    offset: float,
    approach: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> RelativeKeyframeSpec:
    return RelativeKeyframeSpec(
        keyframe_id=f"{kind.value}:{anchor}:{offset}",
        keyframe_type=kind,
        frame_ref="object:part",
        anchor=anchor,
        approach_axis_xyz=approach,
        tool_axis_to_align="-z",
        offset_along_approach_m=offset,
        planner=KeyframePlannerType.CARTESIAN,
    )


def _artifact(*keyframes: RelativeKeyframeSpec) -> KeyframePlanArtifact:
    frames = list(keyframes)
    if len(frames) < 2:
        frames.append(
            _keyframe(
                kind=KeyframeType.LIFT,
                anchor=frames[0].anchor,
                offset=max(0.1, float(frames[0].offset_along_approach_m) + 0.05),
                approach=tuple(float(v) for v in frames[0].approach_axis_xyz),
            )
        )
    return KeyframePlanArtifact(
        artifact_id="keyframes",
        provenance=ArtifactProvenance(
            artifact_id="keyframes-artifact",
            artifact_type="KeyframePlanArtifact",
            produced_by=ModuleName.MOTION_PLANNER,
            invocation_id="test",
        ),
        scene_signature="scene",
        subgoal_id="SG",
        candidates=[
            KeyframePlanCandidate(
                strategy_id="direct",
                keyframes=frames,
                rationale="fixture",
                provenance=StrategyGenerationProvenance(
                    generator_kind=StrategyGeneratorKind.TEMPLATE,
                    generator_id="fixture",
                    input_hash="input",
                ),
            )
        ],
    )


def test_vacuum_center_grasp_binds_to_approach_facing_surface() -> None:
    request = _request(dimensions=(0.16, 0.16, 0.009592404), object_z=0.924790364)
    artifact = _artifact(
        _keyframe(kind=KeyframeType.PRE_GRASP, anchor="center", offset=0.1),
        _keyframe(kind=KeyframeType.GRASP, anchor="center", offset=0.0),
        _keyframe(kind=KeyframeType.LIFT, anchor="center", offset=0.15),
    )
    bound = bind_suction_surface_contact(artifact, request)
    grasp = bound.candidates[0].keyframes[1]
    pre = bound.candidates[0].keyframes[0]
    lift = bound.candidates[0].keyframes[2]
    assert grasp.anchor == "center"
    assert grasp.offset_along_approach_m == pytest.approx(0.004796202)
    assert grasp.metadata["contact_geometry_source"] == SUCTION_SURFACE_CONTACT
    assert pre.offset_along_approach_m == pytest.approx(0.1)
    assert lift.offset_along_approach_m == pytest.approx(0.15)
    pose = RelativePoseResolver(request.world).resolve(grasp)
    top = RelativePoseResolver(request.world).resolve(
        _keyframe(kind=KeyframeType.GRASP, anchor="top_center", offset=0.0)
    )
    assert pose.position_m[2] == pytest.approx(top.position_m[2])


def test_vacuum_top_center_grasp_is_not_double_offset() -> None:
    request = _request()
    artifact = _artifact(
        _keyframe(kind=KeyframeType.GRASP, anchor="top_center", offset=0.0),
    )
    bound = bind_suction_surface_contact(artifact, request)
    grasp = bound.candidates[0].keyframes[0]
    assert grasp.anchor == "top_center"
    assert grasp.offset_along_approach_m == pytest.approx(0.0)
    assert "contact_geometry_source" not in grasp.metadata


def test_vacuum_top_grasp_with_positive_offset_unchanged() -> None:
    request = _request()
    artifact = _artifact(
        _keyframe(kind=KeyframeType.GRASP, anchor="top", offset=0.01),
    )
    bound = bind_suction_surface_contact(artifact, request)
    grasp = bound.candidates[0].keyframes[0]
    assert grasp.offset_along_approach_m == pytest.approx(0.01)
    assert bound.artifact_id == artifact.artifact_id


def test_thickness_scales_surface_offset() -> None:
    thin = _request(dimensions=(0.1, 0.1, 0.01))
    thick = _request(dimensions=(0.1, 0.1, 0.04))
    artifact = _artifact(
        _keyframe(kind=KeyframeType.GRASP, anchor="center", offset=0.0),
    )
    thin_grasp = bind_suction_surface_contact(artifact, thin).candidates[0].keyframes[0]
    thick_grasp = bind_suction_surface_contact(artifact, thick).candidates[0].keyframes[0]
    assert thin_grasp.offset_along_approach_m == pytest.approx(0.005)
    assert thick_grasp.offset_along_approach_m == pytest.approx(0.02)


def test_oriented_approach_uses_support_function_not_world_z() -> None:
    # 90 deg about Y: object +Z maps to world -X. Approach in object frame +Z
    # still selects the local top face via the AABB support function.
    orientation = (0.0, math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0))
    request = _request(
        dimensions=(0.1, 0.1, 0.02),
        orientation_xyzw=orientation,
        object_z=0.0,
    )
    artifact = _artifact(
        _keyframe(kind=KeyframeType.GRASP, anchor="center", offset=0.0),
    )
    grasp = bind_suction_surface_contact(artifact, request).candidates[0].keyframes[0]
    assert grasp.offset_along_approach_m == pytest.approx(0.01)
    pose = RelativePoseResolver(request.world).resolve(grasp)
    top = RelativePoseResolver(request.world).resolve(
        _keyframe(kind=KeyframeType.GRASP, anchor="top_center", offset=0.0)
    )
    assert np.allclose(pose.position_m, top.position_m, atol=1e-9)


def test_transport_and_place_keyframes_are_untouched() -> None:
    request = _request(action_type="transport")
    artifact = _artifact(
        _keyframe(kind=KeyframeType.TRANSFER, anchor="center", offset=0.0),
    )
    bound = bind_suction_surface_contact(artifact, request)
    assert bound.artifact_id == artifact.artifact_id
    assert bound.candidates[0].keyframes[0].offset_along_approach_m == 0.0

    place_request = _request(action_type="place")
    place_artifact = _artifact(
        _keyframe(kind=KeyframeType.PLACE, anchor="center", offset=0.0),
        _keyframe(kind=KeyframeType.RETREAT, anchor="center", offset=0.1),
    )
    place_bound = bind_suction_surface_contact(place_artifact, place_request)
    assert place_bound.artifact_id == place_artifact.artifact_id


def test_two_finger_and_three_finger_grasps_are_untouched() -> None:
    for ee, capabilities in (("2F", ["opposed_finger_contact"]), ("3F", [])):
        request = _request(ee=ee, capabilities=capabilities)
        artifact = _artifact(
            _keyframe(kind=KeyframeType.GRASP, anchor="center", offset=0.0),
        )
        bound = bind_suction_surface_contact(artifact, request)
        assert bound.candidates[0].keyframes[0].offset_along_approach_m == 0.0
        assert bound.artifact_id == artifact.artifact_id


def test_pick_tool_is_out_of_scope() -> None:
    request = _request(action_type="PICK_TOOL")
    request.task.metadata["operation"] = "PICK_TOOL"
    artifact = _artifact(
        _keyframe(kind=KeyframeType.GRASP, anchor="center", offset=0.0),
    )
    bound = SuctionSurfaceContactBinder().bind(artifact, request)
    assert bound.artifact_id == artifact.artifact_id


def test_center_with_large_standoff_not_pulled_down() -> None:
    request = _request(dimensions=(0.1, 0.1, 0.01))
    artifact = _artifact(
        _keyframe(kind=KeyframeType.GRASP, anchor="center", offset=0.08),
    )
    bound = bind_suction_surface_contact(artifact, request)
    assert bound.candidates[0].keyframes[0].offset_along_approach_m == pytest.approx(
        0.08
    )
