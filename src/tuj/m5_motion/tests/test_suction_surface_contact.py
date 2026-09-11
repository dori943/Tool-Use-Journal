"""Vacuum GRASP binds TCP to the approach-facing TARGET collision surface."""

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
    collision_points_m: list[list[float]] | None = None,
) -> MotionPlanRequest:
    half_z = dimensions[2] * 0.5
    record: dict = {
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
    if collision_points_m is not None:
        record["collision_points_m"] = collision_points_m
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
            objects={"part": record},
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


def _flat_top_collision_points(*, half_xy: float, half_z: float) -> list[list[float]]:
    """Sparse points whose approach-facing max matches the AABB top face."""

    return [
        [0.0, 0.0, half_z],
        [half_xy, 0.0, half_z],
        [-half_xy, 0.0, half_z],
        [0.0, half_xy, half_z],
        [0.0, -half_xy, half_z],
        [0.0, 0.0, -half_z],
    ]


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
    assert grasp.metadata["suction_surface_source"] == "aabb_fallback"
    assert pre.offset_along_approach_m == pytest.approx(0.1)
    assert lift.offset_along_approach_m == pytest.approx(0.15)
    pose = RelativePoseResolver(request.world).resolve(grasp)
    top = RelativePoseResolver(request.world).resolve(
        _keyframe(kind=KeyframeType.GRASP, anchor="top_center", offset=0.0)
    )
    assert pose.position_m[2] == pytest.approx(top.position_m[2])


def test_recessed_collision_surface_moves_grasp_off_aabb_face() -> None:
    """GRASP on AABB top is lowered to the reachable TARGET collision surface."""

    half_z = 0.02
    recessed_z = 0.012
    request = _request(
        dimensions=(0.12, 0.12, 0.04),
        collision_points_m=[
            [0.0, 0.0, recessed_z],
            [0.01, 0.0, recessed_z],
            [-0.01, 0.0, recessed_z],
            [0.0, 0.01, recessed_z],
            [0.0, -0.01, recessed_z],
            # Rim vertices that define a taller AABB than the cup footprint.
            [0.05, 0.05, half_z],
            [-0.05, 0.05, half_z],
            [0.05, -0.05, half_z],
            [-0.05, -0.05, half_z],
            [0.0, 0.0, -half_z],
        ],
    )
    artifact = _artifact(
        _keyframe(kind=KeyframeType.PRE_GRASP, anchor="top_center", offset=0.08),
        _keyframe(kind=KeyframeType.GRASP, anchor="top_center", offset=0.0),
        _keyframe(kind=KeyframeType.LIFT, anchor="top_center", offset=0.12),
    )
    bound = bind_suction_surface_contact(artifact, request)
    grasp = bound.candidates[0].keyframes[1]
    pre = bound.candidates[0].keyframes[0]
    lift = bound.candidates[0].keyframes[2]
    assert grasp.offset_along_approach_m == pytest.approx(recessed_z - half_z)
    assert grasp.metadata["suction_surface_source"] == "collision_points"
    assert grasp.metadata["suction_surface_offset_from_center_m"] == pytest.approx(
        recessed_z
    )
    assert grasp.metadata["suction_aabb_offset_from_center_m"] == pytest.approx(half_z)
    assert pre.offset_along_approach_m == pytest.approx(0.08)
    assert lift.offset_along_approach_m == pytest.approx(0.12)


def test_hollow_cup_footprint_relocates_to_nearby_surface_patch() -> None:
    """Empty cup disk shifts laterally onto a covered TARGET patch."""

    half_z = 0.02
    recessed_z = 0.0115  # dish floor under hollow center
    rim_z = half_z
    request = _request(
        dimensions=(0.16, 0.16, 0.04),
        collision_points_m=[
            # Empty under cup r<=0.03; recessed floor just outside.
            [0.035, 0.0, recessed_z],
            [-0.035, 0.0, recessed_z],
            [0.0, 0.035, recessed_z],
            [0.0, -0.035, recessed_z],
            # Covered high rim patch within search radius (preferred).
            [0.055, 0.0, rim_z],
            [0.055, 0.01, rim_z],
            [0.055, -0.01, rim_z],
            [0.065, 0.0, rim_z],
            [0.045, 0.0, rim_z],
            [0.055, 0.02, rim_z],
            # Underside corners outside the preferred patch.
            [0.07, 0.07, -half_z],
            [-0.07, -0.07, -half_z],
        ],
    )
    artifact = _artifact(
        _keyframe(kind=KeyframeType.PRE_GRASP, anchor="top_center", offset=0.1),
        _keyframe(kind=KeyframeType.GRASP, anchor="top_center", offset=0.0),
        _keyframe(kind=KeyframeType.LIFT, anchor="top_center", offset=0.15),
    )
    bound = bind_suction_surface_contact(artifact, request)
    grasp = bound.candidates[0].keyframes[1]
    pre = bound.candidates[0].keyframes[0]
    lift = bound.candidates[0].keyframes[2]
    assert grasp.metadata["suction_surface_source"] == "nearby_surface_patch"
    assert grasp.metadata["suction_surface_offset_from_center_m"] == pytest.approx(rim_z)
    assert grasp.metadata["suction_lateral_relocation_m"] > 0.02
    assert grasp.anchor.startswith("suction_surface_patch_")
    assert grasp.offset_along_approach_m == pytest.approx(0.0)
    # PRE/LIFT follow the relocated patch; standoff offsets unchanged.
    assert pre.anchor == grasp.anchor
    assert lift.anchor == grasp.anchor
    assert pre.offset_along_approach_m == pytest.approx(0.1)
    assert lift.offset_along_approach_m == pytest.approx(0.15)
    pose = RelativePoseResolver(request.world).resolve(grasp)
    assert pose.position_m[0] == pytest.approx(0.055, abs=0.02)
    assert pose.position_m[2] == pytest.approx(
        request.world.objects["part"]["pose"]["position_m"][2] + rim_z,
        abs=1e-6,
    )


def test_flat_collision_surface_matching_aabb_needs_no_extra_offset() -> None:
    half_z = 0.02
    request = _request(
        dimensions=(0.1, 0.1, 0.04),
        collision_points_m=_flat_top_collision_points(half_xy=0.04, half_z=half_z),
    )
    artifact = _artifact(
        _keyframe(kind=KeyframeType.GRASP, anchor="top_center", offset=0.0),
    )
    bound = bind_suction_surface_contact(artifact, request)
    grasp = bound.candidates[0].keyframes[0]
    assert grasp.offset_along_approach_m == pytest.approx(0.0)
    assert "contact_geometry_source" not in grasp.metadata


def test_missing_collision_geometry_falls_back_to_aabb_surface() -> None:
    request = _request(dimensions=(0.1, 0.1, 0.02))
    artifact = _artifact(
        _keyframe(kind=KeyframeType.GRASP, anchor="center", offset=0.0),
    )
    grasp = bind_suction_surface_contact(artifact, request).candidates[0].keyframes[0]
    assert grasp.offset_along_approach_m == pytest.approx(0.01)
    assert grasp.metadata["suction_surface_source"] == "aabb_fallback"


def test_distant_geometry_only_falls_back_to_aabb_without_immersion() -> None:
    """No cup-disk or nearby patch TARGET points: keep AABB, no immersion."""

    half_z = 0.004796202
    # Collision only beyond the 80 mm patch search radius.
    request = _request(
        dimensions=(0.16, 0.16, 2.0 * half_z),
        collision_points_m=[
            [0.10, 0.0, half_z],
            [-0.10, 0.0, half_z],
            [0.0, 0.10, half_z],
            [0.0, -0.10, half_z],
            [0.09, 0.09, -0.003],
            [0.09, -0.09, -half_z],
        ],
    )
    artifact = _artifact(
        _keyframe(kind=KeyframeType.GRASP, anchor="top_center", offset=0.0),
    )
    grasp = bind_suction_surface_contact(artifact, request).candidates[0].keyframes[0]
    # AABB fallback: stay on the named top face. Fixed immersion would use -half_z.
    assert grasp.offset_along_approach_m == pytest.approx(0.0)
    assert grasp.anchor == "top_center"
    assert "contact_geometry_source" not in grasp.metadata


def test_nearby_patch_side_approach_relocates_along_lateral_plane() -> None:
    """Empty cup disk on a lateral approach relocates onto a covered TARGET patch."""

    half_x = 0.05
    patch_along = 0.04
    request = _request(
        dimensions=(0.1, 0.1, 0.1),
        collision_points_m=[
            # Empty under cup along +X at center; covered patch offset in Y
            # (all seeds have |YZ| > cup radius so the original disk stays empty).
            [patch_along, 0.045, 0.0],
            [patch_along, 0.05, 0.01],
            [patch_along, 0.05, -0.01],
            [patch_along, 0.055, 0.0],
            [patch_along, 0.05, 0.0],
            [patch_along, 0.06, 0.0],
            # Far off-axis points (outside any nearby cup patch).
            [half_x, 0.12, 0.0],
            [half_x, -0.12, 0.0],
            [-half_x, 0.12, 0.0],
            [-half_x, -0.12, 0.0],
        ],
    )
    artifact = _artifact(
        _keyframe(
            kind=KeyframeType.PRE_GRASP,
            anchor="center",
            offset=0.12,
            approach=(1.0, 0.0, 0.0),
        ),
        _keyframe(
            kind=KeyframeType.GRASP,
            anchor="center",
            offset=0.0,
            approach=(1.0, 0.0, 0.0),
        ),
        _keyframe(
            kind=KeyframeType.LIFT,
            anchor="center",
            offset=0.18,
            approach=(1.0, 0.0, 0.0),
        ),
    )
    bound = bind_suction_surface_contact(artifact, request)
    grasp = bound.candidates[0].keyframes[1]
    pre = bound.candidates[0].keyframes[0]
    lift = bound.candidates[0].keyframes[2]
    assert grasp.metadata["suction_surface_source"] == "nearby_surface_patch"
    assert grasp.metadata["suction_surface_offset_from_center_m"] == pytest.approx(
        patch_along
    )
    assert grasp.metadata["suction_lateral_relocation_m"] > 0.02
    assert grasp.anchor.startswith("suction_surface_patch_")
    assert grasp.offset_along_approach_m == pytest.approx(0.0)
    assert pre.anchor == grasp.anchor
    assert lift.anchor == grasp.anchor
    assert pre.offset_along_approach_m == pytest.approx(0.12)
    assert lift.offset_along_approach_m == pytest.approx(0.18)
    pose = RelativePoseResolver(request.world).resolve(grasp)
    assert pose.position_m[0] == pytest.approx(patch_along, abs=1e-6)
    assert abs(pose.position_m[1]) > 0.02


def test_thin_supported_object_does_not_immerse_past_aabb() -> None:
    """Regression: no min(10 mm, half-extent) immersion toward the support."""

    half_z = 0.005
    request = _request(
        dimensions=(0.1, 0.1, 0.01),
        collision_points_m=_flat_top_collision_points(half_xy=0.02, half_z=half_z),
    )
    artifact = _artifact(
        _keyframe(kind=KeyframeType.GRASP, anchor="top_center", offset=0.0),
    )
    grasp = bind_suction_surface_contact(artifact, request).candidates[0].keyframes[0]
    assert grasp.offset_along_approach_m == pytest.approx(0.0)
    assert grasp.offset_along_approach_m > -half_z + 1e-9


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
