from __future__ import annotations

import math

import numpy as np
import pytest

from tuj.m5_motion.attachment_retarget import (
    ATTACHED_OBJECT_POSE_SUBJECT,
    POSE_SUBJECT_KEY,
    POSE_SUBJECT_OBJECT_ID_KEY,
    end_effector_pose_for_object_pose,
    retarget_resolved_pose,
)
from tuj.m5_motion.geometry import quaternion_matrix_xyzw
from tuj.m5_motion.schema import (
    AttachedObjectTransform,
    KeyframePlannerType,
    KeyframeType,
    Pose,
    RelativeKeyframeSpec,
    RobotState,
    SceneRef,
    WorldSnapshot,
)


def _transform() -> AttachedObjectTransform:
    half_sqrt = math.sqrt(0.5)
    return AttachedObjectTransform(
        object_id="whisk",
        free_joint_name="whisk_free",
        reference_kind="body",
        reference_name="robot0_right_hand",
        position_in_reference_m=(0.1, 0.0, 0.0),
        orientation_in_reference_xyzw=(0.0, 0.0, half_sqrt, half_sqrt),
    )


def test_object_pose_is_retargeted_through_inverse_grasp_transform() -> None:
    desired_object_pose = Pose(
        frame_id="world",
        position_m=(1.0, 2.0, 3.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    transform = _transform()

    reference_pose = end_effector_pose_for_object_pose(
        desired_object_pose,
        transform,
    )

    reference_rotation = quaternion_matrix_xyzw(reference_pose.orientation_xyzw)
    relative_rotation = quaternion_matrix_xyzw(
        transform.orientation_in_reference_xyzw
    )
    reconstructed_rotation = reference_rotation @ relative_rotation
    reconstructed_position = (
        np.asarray(reference_pose.position_m)
        + reference_rotation @ np.asarray(transform.position_in_reference_m)
    )
    assert reconstructed_position == pytest.approx(desired_object_pose.position_m)
    assert reconstructed_rotation == pytest.approx(np.eye(3))


def test_tagged_pose_uses_runtime_attachment_transform() -> None:
    transform = _transform()
    world = WorldSnapshot(
        scene=SceneRef(signature="scene"),
        robot_state=RobotState(
            robot_id="robot",
            joint_names=["j1"],
            joint_positions_rad=[0.0],
            attached_object_id="whisk",
        ),
        metadata={
            "attached_object_transforms": {
                "whisk": transform.model_dump(mode="json")
            }
        },
    )
    keyframe = RelativeKeyframeSpec(
        keyframe_id="transfer",
        keyframe_type=KeyframeType.TRANSFER,
        frame_ref="world",
        anchor="center",
        approach_axis_xyz=(0.0, 0.0, 1.0),
        planner=KeyframePlannerType.CARTESIAN,
        metadata={
            POSE_SUBJECT_KEY: ATTACHED_OBJECT_POSE_SUBJECT,
            POSE_SUBJECT_OBJECT_ID_KEY: "whisk",
        },
    )
    object_pose = Pose(
        frame_id="world",
        position_m=(1.0, 2.0, 3.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    retargeted = retarget_resolved_pose(world, keyframe, object_pose)

    assert retargeted.position_m == pytest.approx((1.0, 2.1, 3.0))
