from __future__ import annotations

import numpy as np
import pytest

from tuj.m5_motion.attachment_retarget import retarget_resolved_pose
from tuj.m5_motion.c4_2_packing import C4_2PackingKeyframeProvider
from tuj.m5_motion.geometry import RelativePoseResolver, quaternion_matrix_xyzw
from tuj.m5_motion.packing import PackingKeyframeProvider
from tuj.m5_motion.phase_contract import validate_keyframe_phase_contract
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    GoalType,
    JointDynamicLimit,
    KeyframeEventType,
    KeyframePlannerType,
    ModuleName,
    MotionConstraints,
    MotionGoal,
    MotionPlanRequest,
    MotionTask,
    RobotState,
    SceneRef,
    WorldSnapshot,
)


class _UnexpectedFallback:
    def generate(self, request):  # pragma: no cover - failure sentinel
        raise AssertionError(f"unexpected fallback for {request.task.subgoal_id}")


def _request(action: str) -> MotionPlanRequest:
    return MotionPlanRequest(
        request_id=f"milk-{action}",
        provenance=ArtifactProvenance(
            artifact_id="request-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="task-planner",
        ),
        world=WorldSnapshot(
            scene=SceneRef(signature="scene"),
            robot_state=RobotState(
                robot_id="ur5e",
                joint_names=["j1"],
                joint_positions_rad=[0.0],
                attached_object_id="milk",
            ),
            objects={
                "packing_box": {
                    "dimensions_m": [0.20, 0.26, 0.21],
                    "pose": {
                        "position_m": [2.55, -3.02, 0.92],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "anchors": {"top_center": [0.0, 0.0, 0.21]},
                },
                "milk": {
                    "dimensions_m": [0.0456612, 0.0456612, 0.144],
                    "pose": {
                        "position_m": [2.27, -3.44, 1.17],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                },
            },
            metadata={
                "environment_name": "C4_2_DiagonalFitPacking",
                "attached_object_transforms": {
                    "milk": {
                        "object_id": "milk",
                        "free_joint_name": "milk_joint0",
                        "reference_kind": "site",
                        "reference_name": "grip_site",
                        "position_in_reference_m": [0.0, 0.0, 0.032],
                        "orientation_in_reference_xyzw": [1.0, 0.0, 0.0, 0.0],
                    }
                },
            },
        ),
        task=MotionTask(
            task_id="c4_2",
            subgoal_id=f"milk-{action}",
            action_type=action,
            ee="3F",
            target_ids=["milk"],
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_object_id="milk",
                target_region_id="packing_box",
            ),
        ),
        constraints=MotionConstraints(
            joint_limits={
                "j1": JointDynamicLimit(
                    max_velocity_rad_s=1.0,
                    max_acceleration_rad_s2=2.0,
                )
            }
        ),
    )


def test_milk_transport_uses_box_frame_and_attached_object_pose():
    request = _request("TRANSPORT")
    artifact = C4_2PackingKeyframeProvider(_UnexpectedFallback()).generate(request)

    validate_keyframe_phase_contract(request, artifact)
    assert len(artifact.candidates) == 6
    assert all(len(candidate.keyframes) == 2 for candidate in artifact.candidates)
    assert all(
        keyframe.frame_ref == "object:packing_box"
        and keyframe.metadata["pose_subject"] == "ATTACHED_OBJECT"
        for candidate in artifact.candidates
        for keyframe in candidate.keyframes
    )
    assert all(
        keyframe.events_after == []
        for candidate in artifact.candidates
        for keyframe in candidate.keyframes
    )
    first = request.world.objects["packing_box"]["anchors"]["packing_transport_1"]
    assert first == pytest.approx([-0.0721694, -0.04086776, 0.37992])


def test_milk_place_retargets_until_release_then_retreats_as_eef():
    request = _request("PLACE")
    artifact = C4_2PackingKeyframeProvider(_UnexpectedFallback()).generate(request)

    validate_keyframe_phase_contract(request, artifact)
    for candidate in artifact.candidates:
        pre_place, place, retreat = candidate.keyframes
        assert pre_place.metadata["pose_subject_object_id"] == "milk"
        assert place.events_after == [
            KeyframeEventType.DETACH_OBJECT,
            KeyframeEventType.GRIPPER_OPEN,
        ]
        assert pre_place.planner is KeyframePlannerType.SAMPLING_BASED
        assert place.planner is KeyframePlannerType.CARTESIAN
        assert pre_place.metadata["packing_motion_role"] == "FREE_SPACE_ALIGNMENT"
        assert place.metadata["packing_motion_role"] == "CONSTRAINED_INSERTION"
        assert pre_place.metadata["packing_inset_margin_m"] == pytest.approx(0.005)
        assert "pose_subject" not in retreat.metadata


def test_non_pose_retreat_preserves_release_eef_pose_and_moves_up_region_axis():
    request = _request("PLACE")
    artifact = PackingKeyframeProvider(_UnexpectedFallback()).generate(request)
    _, place, retreat = artifact.candidates[0].keyframes
    resolver = RelativePoseResolver(request.world)

    place_eef = retarget_resolved_pose(
        request.world,
        place,
        resolver.resolve(place),
    )
    retreat_eef = retarget_resolved_pose(
        request.world,
        retreat,
        resolver.resolve(retreat),
    )
    region_rotation = quaternion_matrix_xyzw(
        request.world.objects["packing_box"]["pose"]["orientation_xyzw"]
    )
    displacement = np.asarray(retreat_eef.position_m) - np.asarray(
        place_eef.position_m
    )

    assert displacement == pytest.approx(region_rotation[:, 2] * 0.045)
    assert quaternion_matrix_xyzw(retreat_eef.orientation_xyzw) == pytest.approx(
        quaternion_matrix_xyzw(place_eef.orientation_xyzw)
    )


def test_packing_orientation_is_world_relative_for_rotated_region():
    request = _request("TRANSPORT")
    half_sqrt = 2.0**-0.5
    request.world.objects["packing_box"]["pose"]["orientation_xyzw"] = [
        0.0,
        0.0,
        half_sqrt,
        half_sqrt,
    ]
    request.world.objects["milk"]["packing_metadata"] = {
        "orientation_candidates": [
            {
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "dimensions_m": [0.0456612, 0.0456612, 0.144],
                "center_offset_m": [0.0, 0.0, 0.0],
            }
        ]
    }

    artifact = PackingKeyframeProvider(_UnexpectedFallback()).generate(request)
    object_pose = RelativePoseResolver(request.world).resolve(
        artifact.candidates[0].keyframes[0]
    )

    assert quaternion_matrix_xyzw(object_pose.orientation_xyzw) == pytest.approx(
        np.eye(3)
    )


def test_generic_provider_uses_bound_ids_without_environment_name_dependency():
    request = _request("TRANSPORT")
    request.world.metadata["environment_name"] = "UnseenPackingTask"
    request.world.objects["crate"] = request.world.objects.pop("packing_box")
    request.world.objects["carton"] = request.world.objects.pop("milk")
    request.world.robot_state.attached_object_id = "carton"
    transforms = request.world.metadata["attached_object_transforms"]
    transforms["carton"] = transforms.pop("milk")
    transforms["carton"]["object_id"] = "carton"
    request.task.target_ids = ["carton"]
    request.task.goal.target_object_id = "carton"
    request.task.goal.target_region_id = "crate"

    artifact = PackingKeyframeProvider(_UnexpectedFallback()).generate(request)

    assert all(
        keyframe.frame_ref == "object:crate"
        and keyframe.metadata["pose_subject_object_id"] == "carton"
        for candidate in artifact.candidates
        for keyframe in candidate.keyframes
    )


def test_generic_provider_delegates_a_planar_target_region():
    request = _request("TRANSPORT")
    request.world.objects["packing_box"]["dimensions_m"] = [0.5, 0.5, 0.01]

    class _Fallback:
        def generate(self, supplied):
            assert supplied is request
            return "fallback"

    assert PackingKeyframeProvider(_Fallback()).generate(request) == "fallback"


def test_task_packing_profile_overrides_release_timing_and_candidate_count():
    request = _request("PLACE")
    request.task.metadata["packing_profile"] = {
        "candidate_count": 2,
        "release_hold_duration_s": 2.25,
        "release_event_offset_s": 0.2,
    }

    artifact = PackingKeyframeProvider(_UnexpectedFallback()).generate(request)

    assert len(artifact.candidates) == 2
    release = artifact.candidates[0].keyframes[1]
    assert release.metadata["hold_duration_after_s"] == pytest.approx(2.25)
    assert release.metadata["event_time_offsets_s"] == {
        "DETACH_OBJECT": pytest.approx(0.2),
        "GRIPPER_OPEN": pytest.approx(0.2),
    }


def test_object_metadata_can_prioritize_normalized_xy_candidates():
    request = _request("TRANSPORT")
    request.world.objects["milk"]["packing_metadata"] = {
        "normalized_xy_candidates": [[-0.73, 0.80]],
    }

    PackingKeyframeProvider(_UnexpectedFallback()).generate(request)

    safe_x = 0.20 / 2.0 - 0.0456612 / 2.0 - 0.005
    safe_y = 0.26 / 2.0 - 0.0456612 / 2.0 - 0.005
    first = request.world.objects["packing_box"]["anchors"][
        "packing_transport_1"
    ]
    assert first[:2] == pytest.approx([-0.73 * safe_x, 0.80 * safe_y])


def test_generic_provider_avoids_current_container_occupants():
    request = _request("TRANSPORT")
    request.world.objects["packing_box"]["packing_metadata"] = {
        "interior_dimensions_m": [0.20, 0.26, 0.21],
        "interior_center_m": [0.0, 0.0, 0.105],
        "opening_top_z_m": 0.21,
    }
    request.world.objects["cereal"] = {
        "dimensions_m": [0.04, 0.12, 0.15],
        "pose": {
            "position_m": [2.475, -3.062, 1.005],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }

    PackingKeyframeProvider(_UnexpectedFallback()).generate(request)

    first = request.world.objects["packing_box"]["anchors"][
        "packing_transport_1"
    ]
    # The old first slot (-safe_x, -0.4*safe_y) overlaps cereal. The generic
    # occupancy filter must promote a free slot without using either name.
    safe_x = 0.20 / 2.0 - 0.0456612 / 2.0 - 0.005
    safe_y = 0.26 / 2.0 - 0.0456612 / 2.0 - 0.005
    assert first[:2] != pytest.approx([-safe_x, -0.4 * safe_y])


def test_packing_profile_rejects_non_integer_candidate_count():
    request = _request("PLACE")
    request.task.metadata["packing_profile"] = {"candidate_count": 2.0}

    with pytest.raises(ValueError, match="candidate_count must be an integer"):
        PackingKeyframeProvider(_UnexpectedFallback()).generate(request)


def test_diagonal_object_place_offers_geometry_scaled_elevated_release():
    request = _request("PLACE")
    request.world.objects["milk"]["packing_metadata"] = {
        "orientation_candidates": [
            {
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "dimensions_m": [0.15, 0.20, 0.17],
                "center_offset_m": [0.0, 0.0, 0.0],
            }
        ]
    }

    artifact = PackingKeyframeProvider(_UnexpectedFallback()).generate(request)

    first_release = artifact.candidates[0].keyframes[1]
    elevated_release = artifact.candidates[1].keyframes[1]
    first_anchor = request.world.objects["packing_box"]["anchors"][
        "packing_release_1"
    ]
    elevated_anchor = request.world.objects["packing_box"]["anchors"][
        "packing_release_2"
    ]
    assert first_release.metadata["packing_release_clearance_m"] == pytest.approx(
        0.006
    )
    assert elevated_release.metadata[
        "packing_release_clearance_m"
    ] == pytest.approx(0.02754)
    assert elevated_anchor[2] > first_anchor[2]
